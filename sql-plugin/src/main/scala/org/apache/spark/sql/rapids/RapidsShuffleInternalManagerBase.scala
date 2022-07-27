/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.rapids

import java.io.{EOFException, File, FileInputStream, InputStream}
import java.util.{Optional, Random}
import java.util.concurrent.{Callable, Executors, LinkedBlockingQueue}
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleRequestHandler, RapidsShuffleServer, RapidsShuffleTransport}
import org.apache.spark.{InterruptibleIterator, MapOutputTracker, ShuffleDependency, SparkConf, SparkEnv, TaskContext}
import org.apache.spark.executor.ShuffleWriteMetrics
import org.apache.spark.internal.{Logging, config}
import org.apache.spark.io.CompressionCodec
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.serializer.{DeserializationStream, Serializer, SerializerManager}
import org.apache.spark.shuffle.{ShuffleWriter, _}
import org.apache.spark.shuffle.api._
import org.apache.spark.shuffle.sort.{BypassMergeSortShuffleHandle, SortShuffleManager}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.rapids.shims.GpuShuffleBlockResolver
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{RapidsShuffleBlockFetcherIterator, _}
import org.apache.spark.util.collection.ExternalSorter
import org.apache.spark.util.{CompletionIterator, Utils}

class GpuShuffleHandle[K, V](
    val wrapped: ShuffleHandle,
    override val dependency: GpuShuffleDependency[K, V, V])
  extends BaseShuffleHandle(wrapped.shuffleId, dependency) {

  override def toString: String = s"GPU SHUFFLE HANDLE $shuffleId"
}

abstract class GpuShuffleBlockResolverBase(
    protected val wrapped: ShuffleBlockResolver,
    catalog: ShuffleBufferCatalog)
  extends ShuffleBlockResolver with Logging {
  override def getBlockData(blockId: BlockId, dirs: Option[Array[String]]): ManagedBuffer = {
    val hasActiveShuffle: Boolean = blockId match {
      case sbbid: ShuffleBlockBatchId =>
        catalog.hasActiveShuffle(sbbid.shuffleId)
      case sbid: ShuffleBlockId =>
        catalog.hasActiveShuffle(sbid.shuffleId)
      case _ => throw new IllegalArgumentException(s"${blockId.getClass} $blockId "
          + "is not currently supported")
    }
    if (hasActiveShuffle) {
      throw new IllegalStateException(s"The block $blockId is being managed by the catalog")
    }
    wrapped.getBlockData(blockId, dirs)
  }

  override def stop(): Unit = wrapped.stop()
}

/**
 * The `ShuffleWriteMetricsReporter` is based on accumulators, which are not thread safe.
 * This class is a thin wrapper that adds synchronization, since these metrics will be written
 * by multiple threads.
 * @param wrapped
 */
class ThreadSafeShuffleWriteMetricsReporter(wrapped: ShuffleWriteMetricsReporter)
  extends ShuffleWriteMetrics {
  override private[spark] def incBytesWritten(v: Long): Unit = synchronized {
    wrapped.incBytesWritten(v)
  }
  override private[spark] def incRecordsWritten(v: Long): Unit = synchronized {
    wrapped.incRecordsWritten(v)
  }
  override private[spark] def incWriteTime(v: Long): Unit = synchronized {
    wrapped.incWriteTime(v)
  }
  override private[spark] def decBytesWritten(v: Long): Unit = synchronized {
    wrapped.decBytesWritten(v)
  }
  override private[spark] def decRecordsWritten(v: Long): Unit = synchronized {
    wrapped.decRecordsWritten(v)
  }
}

object RapidsShuffleInternalManagerBase extends Logging {
  def unwrapHandle(handle: ShuffleHandle): ShuffleHandle = handle match {
    case gh: GpuShuffleHandle[_, _] => gh.wrapped
    case other => other
  }

  /**
   * "slots" are a thread + queue thin wrapper that is used
   * to execute tasks that need to be done in sequentially.
   * This is done such that the threaded shuffle writer posts
   * tasks that are for writer_i, and that writer is guaranteed
   * to be written to sequentially, but writer_j may end up
   * in a different slot, and could perform its work in parallel.
   * @param slotNum this slot's unique number only used to name its executor
   */
  private class Slot(slotNum: Int) {
    private val p = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder()
        .setNameFormat(s"rapids-shuffle-writer-$slotNum")
        .setDaemon(true)
        .build())

    def offer[T](task: Callable[T]): Unit = {
      p.submit(task)
    }

    def shutdownNow(): Unit = p.shutdownNow()
  }

  // this is set by the executor on startup, when the MULTITHREADED
  // shuffle mode is utilized, as per this config:
  //   spark.rapids.shuffle.multiThreaded.writer.threads
  private var numSlots: Int = 0
  private lazy val slots = new mutable.HashMap[Int, Slot]()

  // used by callers to obtain a unique slot
  private val slotNumber = new AtomicInteger(0)

  /**
   * Send a task to a specific slot.
   * @param slotNum the slot to submit to
   * @param task a task to execute
   * @note there must not be an uncaught exception while calling
   *      `task`.
   */
  def queueTask[T](slotNum: Int, task: Callable[T]): Unit = {
    slots(slotNum % numSlots).offer(task)
  }

  def startThreadPoolIfNeeded(numConfiguredThreads: Int): Unit = synchronized {
    if (slots.isEmpty) {
      numSlots = numConfiguredThreads
      (0 until numSlots).foreach { slotNum =>
        slots.put(slotNum, new Slot(slotNum))
      }
    }
  }

  def stopThreadPool(): Unit = synchronized {
    slots.values.foreach(_.shutdownNow())
    slots.clear()
  }

  def getNextSlot: Int = Math.abs(slotNumber.incrementAndGet())
}

trait RapidsShuffleWriterShimHelper {
  def setChecksumIfNeeded(writer: DiskBlockObjectWriter, partition: Int): Unit = {
    // noop until Spark 3.2.0+
  }

  // Partition lengths, used for MapStatus, but also exposed in Spark 3.2.0+
  private var myPartitionLengths: Array[Long] = null

  // This is a Spark 3.2.0+ function, adding a default here for testing purposes
  def getPartitionLengths: Array[Long] = myPartitionLengths

  def commitAllPartitions(writer: ShuffleMapOutputWriter, emptyChecksums: Boolean): Array[Long] = {
    myPartitionLengths = doCommitAllPartitions(writer, emptyChecksums)
    myPartitionLengths
  }

  def doCommitAllPartitions(writer: ShuffleMapOutputWriter, emptyChecksums: Boolean): Array[Long]
}

abstract class RapidsShuffleThreadedWriterBase[K, V](
    blockManager: BlockManager,
    handle: BypassMergeSortShuffleHandle[K, V],
    mapId: Long,
    sparkConf: SparkConf,
    writeMetrics: ShuffleWriteMetricsReporter,
    shuffleExecutorComponents: ShuffleExecutorComponents)
      extends ShuffleWriter[K, V]
        with RapidsShuffleWriterShimHelper
        with Arm
        with Logging {

  private var myMapStatus: Option[MapStatus] = None

  private val dep: ShuffleDependency[K, V, V] = handle.dependency
  private val shuffleId = dep.shuffleId
  private val partitioner = dep.partitioner
  private val numPartitions = partitioner.numPartitions
  private val serializer = dep.serializer.newInstance()
  private val transferToEnabled = sparkConf.getBoolean("spark.file.transferTo", true)
  private val fileBufferSize = sparkConf.get(config.SHUFFLE_FILE_BUFFER_SIZE).toInt * 1024
  /**
   * Are we in the process of stopping? Because map tasks can call stop() with success = true
   * and then call stop() with success = false if they get an exception, we want to make sure
   * we don't try deleting files, etc twice.
   */
  private var stopping = false

  val diskBlockObjectWriters = new mutable.HashMap[Int, (Int, DiskBlockObjectWriter)]()

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    withResource(new NvtxRange("ThreadedWriter.write", NvtxColor.RED)) { _ =>
      withResource(new NvtxRange("compute", NvtxColor.GREEN)) { _ =>
        var errorOcurred: Throwable = null
        val mapOutputWriter = shuffleExecutorComponents.createMapOutputWriter(
          shuffleId,
          mapId,
          numPartitions)
        try {
          val partLengths = if (!records.hasNext) {
            commitAllPartitions(mapOutputWriter, true /*empty checksum*/)
          } else {
            // per reduce partition id
            // open all the writers ahead of time (Spark does this already)
            val openStartTime = System.nanoTime()
            (0 until numPartitions).map { i =>
              val (blockId, file) = blockManager.diskBlockManager.createTempShuffleBlock()
              val writer: DiskBlockObjectWriter = blockManager.getDiskWriter(
                blockId, file, serializer, fileBufferSize, writeMetrics)
              setChecksumIfNeeded(writer, i) // spark3.2.0+

              // Places writer objects at round robin slot numbers apriori
              // this choice is for simplicity but likely needs to change so that
              // we can handle skew better
              val slotNum = RapidsShuffleInternalManagerBase.getNextSlot
              diskBlockObjectWriters.put(i, (slotNum, writer))
            }
            writeMetrics.incWriteTime(System.nanoTime() - openStartTime);

            // we call write on every writer for every record in parallel
            val scheduledWrites = new AtomicLong(0L)
            records.foreach { record =>
              val key = record._1
              val value = record._2
              val reducePartitionId: Int = partitioner.getPartition(key)
              scheduledWrites.incrementAndGet()
              val (slotNum, myWriter) = diskBlockObjectWriters(reducePartitionId)

              // we close batches actively in the `records` iterator as we get the next batch
              // this makes sure it is kept alive while a task is able to handle it.
              val cb = value match {
                case columnarBatch: ColumnarBatch =>
                  SlicedGpuColumnVector.incRefCount(columnarBatch)
                case _ =>
                  null
              }

              RapidsShuffleInternalManagerBase.queueTask(slotNum, () => {
                try {
                  withResource(cb) { _ =>
                    myWriter.write(key, value)
                  }
                  synchronized {
                    scheduledWrites.decrementAndGet()
                    notifyAll()
                  }
                } catch {
                  case e: Throwable =>
                    synchronized {
                      errorOcurred = e
                      notifyAll()
                    }
                }
              })
            }

            withResource(new NvtxRange("WaitingForWrites", NvtxColor.PURPLE)) { _ =>
              synchronized {
                while (errorOcurred == null && scheduledWrites.get() > 0) {
                  wait()
                }
                if (errorOcurred != null) {
                  throw errorOcurred
                }
              }
            }
            writePartitionedData(mapOutputWriter)
          }
          myMapStatus = Some(MapStatus(blockManager.shuffleServerId, partLengths, mapId))
        } catch {
          // taken directly from BypassMergeSortShuffleWriter
          case e: Exception =>
            try {
              mapOutputWriter.abort(e)
            } catch {
              case e2: Exception =>
                logError("Failed to abort the writer after failing to write map output.", e2);
                e.addSuppressed(e2);
            }
            throw e
        }
      }
    }
  }

  def writePartitionedData(mapOutputWriter: ShuffleMapOutputWriter): Array[Long] = {
    // after all temporary shuffle writes are done, we need to produce a single
    // file (shuffle_[map_id]_0) which is done during this commit phase
    withResource(new NvtxRange("CommitShuffle", NvtxColor.RED)) { _ =>
      // per reduce partition
      val segments = (0 until numPartitions).map {
        reducePartitionId =>
          withResource(diskBlockObjectWriters(reducePartitionId)._2) { writer =>
            val segment = writer.commitAndGet()
            (reducePartitionId, segment.file)
          }
      }

      val writeStartTime = System.nanoTime()
      segments.foreach { case (reducePartitionId, file) =>
        val partWriter = mapOutputWriter.getPartitionWriter(reducePartitionId)
        if (file.exists()) {
          if (transferToEnabled) {
            val maybeOutputChannel: Optional[WritableByteChannelWrapper] =
              partWriter.openChannelWrapper()
            if (maybeOutputChannel.isPresent) {
              writePartitionedDataWithChannel(file, maybeOutputChannel.get())
            } else {
              writePartitionedDataWithStream(file, partWriter)
            }
          } else {
            writePartitionedDataWithStream(file, partWriter)
          }
          file.delete()
        }
      }
      writeMetrics.incWriteTime(System.nanoTime() - writeStartTime)
    }
    commitAllPartitions(mapOutputWriter, false /*non-empty checksums*/)
  }

  // taken from BypassMergeSortShuffleWriter
  // this code originally called into guava.Closeables.close
  // and had logic to silence exceptions thrown while copying
  // I am ignoring this for now.
  def writePartitionedDataWithStream(file: java.io.File, writer: ShufflePartitionWriter): Unit = {
    withResource(new FileInputStream(file)) { in =>
      withResource(writer.openStream()) { os =>
        Utils.copyStream(in, os, false, false)
      }
    }
  }

  // taken from BypassMergeSortShuffleWriter
  // this code originally called into guava.Closeables.close
  // and had logic to silence exceptions thrown while copying
  // I am ignoring this for now.
  def writePartitionedDataWithChannel(
    file: File,
    outputChannel: WritableByteChannelWrapper): Unit = {
    // note outputChannel.close() doesn't actually close it.
    // The call is there to record keep the partition lengths
    // after the serialization completes.
    withResource(outputChannel) { _ =>
      withResource(new FileInputStream(file)) { in =>
        withResource(in.getChannel) { inputChannel =>
          Utils.copyFileStreamNIO(
            inputChannel, outputChannel.channel, 0L, inputChannel.size)
        }
      }
    }
  }

  override def stop(success: Boolean): Option[MapStatus] = {
    if (stopping) {
      None
    } else {
      stopping = true
      if (success) {
        if (myMapStatus.isEmpty) {
          // should not happen, but adding it just in case (this differs from Spark)
          cleanupTempData()
          throw new IllegalStateException("Cannot call stop(true) without having called write()");
        }
        myMapStatus
      } else {
        cleanupTempData()
        None
      }
    }
  }

  private def cleanupTempData(): Unit = {
    // The map task failed, so delete our output data.
    try {
      diskBlockObjectWriters.values.foreach { case (_, writer) =>
        val file = writer.revertPartialWritesAndClose()
        if (!file.delete()) logError(s"Error while deleting file ${file.getAbsolutePath()}")
      }
    } finally {
      diskBlockObjectWriters.clear()
    }
  }
}

class RapidsShuffleThreadedReader[K, C](
    handle: BaseShuffleHandle[K, _, C],
    blocksByAddress: Iterator[(BlockManagerId, Seq[(BlockId, Long, Int)])],
    context: TaskContext,
    readMetrics: ShuffleReadMetricsReporter,
    shuffleReaderThreads: Int,
    serializerManager: SerializerManager = SparkEnv.get.serializerManager,
    blockManager: BlockManager = SparkEnv.get.blockManager,
    mapOutputTracker: MapOutputTracker = SparkEnv.get.mapOutputTracker,
    shouldBatchFetch: Boolean = false)
  extends ShuffleReader[K, C] with Logging with Arm {

  private val dep = handle.dependency

  private def fetchContinuousBlocksInBatch: Boolean = {
    val conf = SparkEnv.get.conf
    val serializerRelocatable = dep.serializer.supportsRelocationOfSerializedObjects
    val compressed = conf.get(config.SHUFFLE_COMPRESS)
    val codecConcatenation = if (compressed) {
      CompressionCodec.supportsConcatenationOfSerializedStreams(CompressionCodec.createCodec(conf))
    } else {
      true
    }
    val useOldFetchProtocol = conf.get(config.SHUFFLE_USE_OLD_FETCH_PROTOCOL)
    // SPARK-34790: Fetching continuous blocks in batch is incompatible with io encryption.
    val ioEncryption = conf.get(config.IO_ENCRYPTION_ENABLED)

    val doBatchFetch = shouldBatchFetch && serializerRelocatable &&
      (!compressed || codecConcatenation) && !useOldFetchProtocol && !ioEncryption
    if (shouldBatchFetch && !doBatchFetch) {
      logDebug("The feature tag of continuous shuffle block fetching is set to true, but " +
        "we can not enable the feature because other conditions are not satisfied. " +
        s"Shuffle compress: $compressed, serializer relocatable: $serializerRelocatable, " +
        s"codec concatenation: $codecConcatenation, use old shuffle fetch protocol: " +
        s"$useOldFetchProtocol, io encryption: $ioEncryption.")
    }
    doBatchFetch
  }

  class ParallelDeserializerIterator(
                                      streams: Iterator[(BlockId, ManagedBuffer, InputStream)],
                                      serializer: Serializer,
                                      streamWrapper: (BlockId, InputStream) => InputStream)
    extends Iterator[(Any, Any)] with Arm {
    val inFlight = new AtomicLong(0L)
    val queued = new LinkedBlockingQueue[(Any, Any)]

    override def hasNext: Boolean = queued.synchronized {
      streams.hasNext || !queued.isEmpty || inFlight.get() > 0
    }

    val serializerInstance = serializer.newInstance()

    def deserialize(deserStream: DeserializationStream): Unit = {
      // could be really bad if consumer is slow
      var count = 0
      try {
        // queued and inflight need to be updated in the lock?
        var finished = false

        // highest level
        withResource(new NvtxRange("reader.deser", NvtxColor.DARK_GREEN)) { _ =>
          val results = new ArrayBuffer[(Any, Any)]()
          while (!finished) {
            try {
              val r = (deserStream.readKey[Any](), deserStream.readValue[Any]())
              results.append(r)
              count = count + 1
            } catch {
              case eof: EOFException =>
                finished = true
            }
          }
          queued.synchronized {
            results.foreach(queued.add)
            inFlight.decrementAndGet()
          }
        }
      } catch {
        case t: Throwable =>
          logError(s"error: got ${count}", t)
          throw t
      }
    }

    var toClose = new ArrayBuffer[AutoCloseable]()
    val rnd = new Random()

    TaskContext.get().addTaskCompletionListener[Unit](_ => {
      if (toClose.nonEmpty) {
        logWarning(s"CLOSING AT TASK COMPLETION ${toClose.size}")
        toClose.foreach(_.close())
      }
    })

    override def next(): (Any, Any) = {
      if (streams.hasNext) {
        queued.synchronized {
          require(hasNext, "called next on an empty iterator")

          // queue all of it?
          while (streams.hasNext) {
            val (blockId, managedBuffer, inputStream) = streams.next()
            // n is  (BlockId, BufferReleasingInputStream)
            inFlight.incrementAndGet()

            val slot = RapidsShuffleInternalManagerBase.getNextSlot

            // TODO: bring back toClose
            toClose.append(() => {
              managedBuffer.release()
            })
            toClose.append(inputStream)

            def runIt(): Unit = {
              withResource(new NvtxRange("reader.deser", NvtxColor.DARK_GREEN)) { _ =>
                managedBuffer.retain() // TODO: weird
                // JCudfSerialization
                val deserStream = serializerInstance.deserializeStream(inputStream)
                val it = deserStream.asKeyValueIterator
                val res = new ArrayBuffer[(Any, Any)]()
                while (it.hasNext) {
                  res.append(it.next())
                }
                queued.synchronized {
                  res.foreach(queued.offer)
                  inFlight.decrementAndGet()
                  managedBuffer.release()
                }
              }
            }

            if (shuffleReaderThreads > 0) {
              RapidsShuffleInternalManagerBase.queueTask(slot, () => {
                runIt()
              })
            } else {
              runIt()
            }
          }
        }
      }
      val res = queued.take()
      if (!hasNext) {
        logInfo(s"closing ${toClose.size} streams")
        toClose.foreach(_.close)
        toClose.clear()
      }
      res
    }
  }

  /** Read the combined key-values for this reduce task */
  override def read(): Iterator[Product2[K, C]] = {
    val myStreamWrapper =
      (block: BlockId, inputStream: InputStream) => {
        val wrapped = serializerManager.wrapStream(block, inputStream)
        new InputStream {
          override def read() = {
            // lowest level
            withResource(new NvtxRange("wrap.read", NvtxColor.ORANGE)) { _ =>
              wrapped.read()
            }
          }

          override def close() = {
            withResource(new NvtxRange("wrap.close", NvtxColor.RED)) { _ =>
              wrapped.close()
              inputStream.close()
            }
          }
        }
      }

    val wrappedStreams = new RapidsShuffleBlockFetcherIterator(
      context,
      blockManager.blockStoreClient,
      blockManager,
      blocksByAddress,
      myStreamWrapper,
      //serializerManager.wrapStream,
      // Note: we use getSizeAsMb when no suffix is provided for backwards compatibility
      SparkEnv.get.conf.get(config.REDUCER_MAX_SIZE_IN_FLIGHT) * 1024 * 1024,
      SparkEnv.get.conf.get(config.REDUCER_MAX_REQS_IN_FLIGHT),
      SparkEnv.get.conf.get(config.REDUCER_MAX_BLOCKS_IN_FLIGHT_PER_ADDRESS),
      SparkEnv.get.conf.get(config.MAX_REMOTE_BLOCK_SIZE_FETCH_TO_MEM),
      SparkEnv.get.conf.get(config.SHUFFLE_DETECT_CORRUPT),
      SparkEnv.get.conf.get(config.SHUFFLE_DETECT_CORRUPT_MEMORY),
      readMetrics,
      fetchContinuousBlocksInBatch)

    //val serializerInstance = dep.serializer.newInstance()

    // need a "ParallelizedDesrializerIterator"
    // it will submit deserialization tasks in parallel and queue results
    // until the caller is ready to call next


    val recordIter = new ParallelDeserializerIterator(
      wrappedStreams, dep.serializer, myStreamWrapper)

    // Create a key/value iterator for each stream
    //val recordIter: Iterator[(Any, Any)] =
    // wrappedStreams.flatMap { case (blockId, wrappedStream) =>
    //  // Note: the asKeyValueIterator below wraps a key/value iterator inside of a
    //  // NextIterator. The NextIterator makes sure that close() is called on the
    //  // underlying InputStream when all records have been read.
    //  serializerInstance.deserializeStream(wrappedStream).asKeyValueIterator
    //}

    // Update the context task metrics for each record read.
    val metricIter = CompletionIterator[(Any, Any), Iterator[(Any, Any)]](
      recordIter.map { record =>
        readMetrics.incRecordsRead(1)
        record
      },
      context.taskMetrics().mergeShuffleReadMetrics())

    // An interruptible iterator must be used here in order to support task cancellation
    val interruptibleIter = new InterruptibleIterator[(Any, Any)](context, metricIter)

    val aggregatedIter: Iterator[Product2[K, C]] = if (dep.aggregator.isDefined) {
      if (dep.mapSideCombine) {
        // We are reading values that are already combined
        val combinedKeyValuesIterator = interruptibleIter.asInstanceOf[Iterator[(K, C)]]
        dep.aggregator.get.combineCombinersByKey(combinedKeyValuesIterator, context)
      } else {
        // We don't know the value type, but also don't care -- the dependency *should*
        // have made sure its compatible w/ this aggregator, which will convert the value
        // type to the combined type C
        val keyValuesIterator = interruptibleIter.asInstanceOf[Iterator[(K, Nothing)]]
        dep.aggregator.get.combineValuesByKey(keyValuesIterator, context)
      }
    } else {
      interruptibleIter.asInstanceOf[Iterator[Product2[K, C]]]
    }

    // Sort the output if there is a sort ordering defined.
    val resultIter = dep.keyOrdering match {
      case Some(keyOrd: Ordering[K]) =>
        // Create an ExternalSorter to sort the data.
        val sorter =
          new ExternalSorter[K, C, C](context, ordering = Some(keyOrd), serializer = dep.serializer)
        sorter.insertAll(aggregatedIter)
        context.taskMetrics().incMemoryBytesSpilled(sorter.memoryBytesSpilled)
        context.taskMetrics().incDiskBytesSpilled(sorter.diskBytesSpilled)
        context.taskMetrics().incPeakExecutionMemory(sorter.peakMemoryUsedBytes)
        // Use completion callback to stop sorter if task was finished/cancelled.
        context.addTaskCompletionListener[Unit](_ => {
          sorter.stop()
        })
        CompletionIterator[Product2[K, C], Iterator[Product2[K, C]]](sorter.iterator, sorter.stop())
      case None =>
        aggregatedIter
    }

    resultIter match {
      case _: InterruptibleIterator[Product2[K, C]] => resultIter
      case _ =>
        // Use another interruptible iterator here to support task cancellation as aggregator
        // or(and) sorter may have consumed previous interruptible iterator.
        new InterruptibleIterator[Product2[K, C]](context, resultIter)
    }
  }
}

class RapidsCachingWriter[K, V](
                                 blockManager: BlockManager,
    // Never keep a reference to the ShuffleHandle in the cache as it being GCed triggers
    // the data being released
    handle: GpuShuffleHandle[K, V],
    mapId: Long,
    metricsReporter: ShuffleWriteMetricsReporter,
    catalog: ShuffleBufferCatalog,
    shuffleStorage: RapidsDeviceMemoryStore,
    rapidsShuffleServer: Option[RapidsShuffleServer],
    metrics: Map[String, SQLMetric]) extends ShuffleWriter[K, V] with Logging {

  private val numParts = handle.dependency.partitioner.numPartitions
  private val sizes = new Array[Long](numParts)
  private val writtenBufferIds = new ArrayBuffer[ShuffleBufferId](numParts)
  private val uncompressedMetric: SQLMetric = metrics("dataSize")

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    // NOTE: This MUST NOT CLOSE the incoming batches because they are
    //       closed by the input iterator generated by GpuShuffleExchangeExec
    val nvtxRange = new NvtxRange("RapidsCachingWriter.write", NvtxColor.CYAN)
    try {
      var bytesWritten: Long = 0L
      var recordsWritten: Long = 0L
      records.foreach { p =>
        val partId = p._1.asInstanceOf[Int]
        val batch = p._2.asInstanceOf[ColumnarBatch]
        logDebug(s"Caching shuffle_id=${handle.shuffleId} map_id=$mapId, partId=$partId, "
            + s"batch=[num_cols=${batch.numCols()}, num_rows=${batch.numRows()}]")
        recordsWritten = recordsWritten + batch.numRows()
        var partSize: Long = 0
        val blockId = ShuffleBlockId(handle.shuffleId, mapId, partId)
        val bufferId = catalog.nextShuffleBufferId(blockId)
        if (batch.numRows > 0 && batch.numCols > 0) {
          // Add the table to the shuffle store
          batch.column(0) match {
            case c: GpuPackedTableColumn =>
              val contigTable = c.getContiguousTable
              partSize = c.getTableBuffer.getLength
              uncompressedMetric += partSize
              shuffleStorage.addContiguousTable(
                bufferId,
                contigTable,
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_PRIORITY,
                // we don't need to sync here, because we sync on the cuda
                // stream after sliceInternalOnGpu (contiguous_split)
                needsSync = false)
            case c: GpuCompressedColumnVector =>
              val buffer = c.getTableBuffer
              buffer.incRefCount()
              partSize = buffer.getLength
              val tableMeta = c.getTableMeta
              // update the table metadata for the buffer ID generated above
              tableMeta.bufferMeta.mutateId(bufferId.tableId)
              uncompressedMetric += tableMeta.bufferMeta().uncompressedSize()
              shuffleStorage.addBuffer(
                bufferId,
                buffer,
                tableMeta,
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_PRIORITY,
                // we don't need to sync here, because we sync on the cuda
                // stream after compression.
                needsSync = false)
            case c => throw new IllegalStateException(s"Unexpected column type: ${c.getClass}")
          }
          bytesWritten += partSize
          sizes(partId) += partSize
        } else {
          // no device data, tracking only metadata
          val tableMeta = MetaUtils.buildDegenerateTableMeta(batch)
          catalog.registerNewBuffer(new DegenerateRapidsBuffer(bufferId, tableMeta))

          // The size of the data is really only used to tell if the data should be shuffled or not
          // a 0 indicates that we should not shuffle anything.  This is here for the special case
          // where we have no columns, because of predicate push down, but we have a row count as
          // metadata.  We still want to shuffle it. The 100 is an arbitrary number and can be
          // any non-zero number that is not too large.
          if (batch.numRows > 0) {
            sizes(partId) += 100
          }
        }
        writtenBufferIds.append(bufferId)
      }
      metricsReporter.incBytesWritten(bytesWritten)
      metricsReporter.incRecordsWritten(recordsWritten)
    } finally {
      nvtxRange.close()
    }
  }

  /**
   * Used to remove shuffle buffers when the writing task detects an error, calling `stop(false)`
   */
  private def cleanStorage(): Unit = {
    writtenBufferIds.foreach(catalog.removeBuffer)
  }

  override def stop(success: Boolean): Option[MapStatus] = {
    val nvtxRange = new NvtxRange("RapidsCachingWriter.close", NvtxColor.CYAN)
    try {
      if (!success) {
        cleanStorage()
        None
      } else {
        // upon seeing this port, the other side will try to connect to the port
        // in order to establish an UCX endpoint (on demand), if the topology has "rapids" in it.
        val shuffleServerId = if (rapidsShuffleServer.isDefined) {
          val originalShuffleServerId = rapidsShuffleServer.get.originalShuffleServerId
          val server = rapidsShuffleServer.get
          BlockManagerId(
            originalShuffleServerId.executorId,
            originalShuffleServerId.host,
            originalShuffleServerId.port,
            Some(s"${RapidsShuffleTransport.BLOCK_MANAGER_ID_TOPO_PREFIX}=${server.getPort}"))
        } else {
          blockManager.shuffleServerId
        }
        logInfo(s"Done caching shuffle success=$success, server_id=$shuffleServerId, "
            + s"map_id=$mapId, sizes=${sizes.mkString(",")}")
        Some(MapStatus(shuffleServerId, sizes, mapId))
      }
    } finally {
      nvtxRange.close()
    }
  }

  def getPartitionLengths(): Array[Long] = {
    throw new UnsupportedOperationException("TODO")
  }
}

/**
 * A shuffle manager optimized for the RAPIDS Plugin For Apache Spark.
 * @note This is an internal class to obtain access to the private
 *       `ShuffleManager` and `SortShuffleManager` classes. When configuring
 *       Apache Spark to use the RAPIDS shuffle manager,
 */
abstract class RapidsShuffleInternalManagerBase(conf: SparkConf, val isDriver: Boolean)
    extends ShuffleManager with RapidsShuffleHeartbeatHandler with Logging {

  def getServerId: BlockManagerId = server.fold(blockManager.blockManagerId)(_.getId)

  override def addPeer(peer: BlockManagerId): Unit = {
    transport.foreach { t =>
      try {
        t.connect(peer)
      } catch {
        case ex: Exception =>
          // We ignore the exception after logging in this instance because
          // we may have a peer that doesn't exist anymore by the time `addPeer` is invoked
          // due to a heartbeat response from the driver, or the peer may have a temporary network
          // issue.
          //
          // This is safe because `addPeer` is only invoked due to a heartbeat that is used to
          // opportunistically hide cost of initializing transport connections. The transport
          // will re-try if it must fetch from this executor at a later time, in that case
          // a connection failure causes the tasks to fail.
          logWarning(s"Unable to connect to peer $peer, ignoring!", ex)
      }
    }
  }

  private val rapidsConf = new RapidsConf(conf)

  if (!isDriver && rapidsConf.isMultiThreadedShuffleManagerMode) {
    RapidsShuffleInternalManagerBase.startThreadPoolIfNeeded(
      rapidsConf.shuffleMultiThreadedWriterThreads)
  }

  protected val wrapped = new SortShuffleManager(conf)

  private[this] val transportEnabledMessage =
    if (!rapidsConf.isUCXShuffleManagerMode) {
      if (rapidsConf.isCacheOnlyShuffleManagerMode) {
        "Transport disabled (local cached blocks only)"
      } else {
        "Experimental threaded shuffle writer mode"
      }
    } else {
      s"Transport enabled (remote fetches will use ${rapidsConf.shuffleTransportClassName}"
    }

  logWarning(s"Rapids Shuffle Plugin enabled. ${transportEnabledMessage}. To disable the " +
      s"RAPIDS Shuffle Manager set `${RapidsConf.SHUFFLE_MANAGER_ENABLED}` to false")

  //Many of these values like blockManager are not initialized when the constructor is called,
  // so they all need to be lazy values that are executed when things are first called

  // NOTE: this can be null in the driver side.
  protected lazy val env = SparkEnv.get
  protected lazy val blockManager = env.blockManager
  protected lazy val shouldFallThroughOnEverything = {
    val fallThroughReasons = new ListBuffer[String]()
    if (GpuShuffleEnv.isExternalShuffleEnabled) {
      fallThroughReasons += "External Shuffle Service is enabled"
    }
    if (GpuShuffleEnv.isSparkAuthenticateEnabled) {
      fallThroughReasons += "Spark authentication is enabled"
    }
    if (rapidsConf.isSqlExplainOnlyEnabled) {
      fallThroughReasons += "Plugin is in explain only mode"
    }
    if (fallThroughReasons.nonEmpty) {
      logWarning(s"Rapids Shuffle Plugin is falling back to SortShuffleManager " +
          s"because: ${fallThroughReasons.mkString(", ")}")
    }
    fallThroughReasons.nonEmpty
  }

  private lazy val localBlockManagerId = blockManager.blockManagerId

  // Used to prevent stopping multiple times RAPIDS Shuffle Manager internals.
  // see the `stop` method
  private var stopped: Boolean = false

  // Code that expects the shuffle catalog to be initialized gets it this way,
  // with error checking in case we are in a bad state.
  protected def getCatalogOrThrow: ShuffleBufferCatalog =
    Option(GpuShuffleEnv.getCatalog).getOrElse(
      throw new IllegalStateException("The ShuffleBufferCatalog is not initialized but the " +
          "RapidsShuffleManager is configured"))

  protected lazy val resolver =
    if (shouldFallThroughOnEverything || rapidsConf.isMultiThreadedShuffleManagerMode) {
      wrapped.shuffleBlockResolver
    } else {
      new GpuShuffleBlockResolver(wrapped.shuffleBlockResolver, getCatalogOrThrow)
    }

  private[this] lazy val transport: Option[RapidsShuffleTransport] = {
    if (rapidsConf.isUCXShuffleManagerMode && !isDriver) {
      Some(RapidsShuffleTransport.makeTransport(blockManager.shuffleServerId, rapidsConf))
    } else {
      None
    }
  }

  private[this] lazy val server: Option[RapidsShuffleServer] = {
    if (rapidsConf.isGPUShuffle && !isDriver) {
      val catalog = getCatalogOrThrow
      val requestHandler = new RapidsShuffleRequestHandler() {
        override def acquireShuffleBuffer(tableId: Int): RapidsBuffer = {
          val shuffleBufferId = catalog.getShuffleBufferId(tableId)
          catalog.acquireBuffer(shuffleBufferId)
        }

        override def getShuffleBufferMetas(sbbId: ShuffleBlockBatchId): Seq[TableMeta] = {
          (sbbId.startReduceId to sbbId.endReduceId).flatMap(rid => {
            catalog.blockIdToMetas(ShuffleBlockId(sbbId.shuffleId, sbbId.mapId, rid))
          })
        }
      }
      val server = transport.get.makeServer(requestHandler)
      server.start()
      Some(server)
    } else {
      None
    }
  }

  override def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]): ShuffleHandle = {
    // Always register with the wrapped handler so we can write to it ourselves if needed
    val orig = wrapped.registerShuffle(shuffleId, dependency)

    dependency match {
      case _ if shouldFallThroughOnEverything ||
        rapidsConf.isMultiThreadedShuffleManagerMode => orig
      case gpuDependency: GpuShuffleDependency[K, V, C] if gpuDependency.useGPUShuffle =>
        new GpuShuffleHandle(orig,
          dependency.asInstanceOf[GpuShuffleDependency[K, V, V]])
      case _ => orig
    }
  }

  lazy val execComponents: Option[ShuffleExecutorComponents] = {
    import scala.collection.JavaConverters._
    val executorComponents = ShuffleDataIOUtils.loadShuffleDataIO(conf).executor()
    val extraConfigs = conf.getAllWithPrefix(ShuffleDataIOUtils.SHUFFLE_SPARK_CONF_PREFIX).toMap
    executorComponents.initializeExecutor(
      conf.getAppId,
      SparkEnv.get.executorId,
      extraConfigs.asJava)
    Some(executorComponents)
  }

  /**
   * Creates a BypassMergeSortShuffleWriter specific to a spark version.
   * Implemented in subclasses of `RapidsShuffleInternalManagerBase` per shim.
   */
  def makeBypassMergeSortShuffleWriter[K, V](
     handle: BypassMergeSortShuffleHandle[K, V],
     mapId: Long,
     context: TaskContext,
     metricsReporter: ShuffleWriteMetricsReporter): ShuffleWriter[K, V]

  override def getWriter[K, V](
    handle: ShuffleHandle, mapId: Long, context: TaskContext,
    metricsReporter: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    handle match {
      case gpu: GpuShuffleHandle[_, _] =>
        registerGpuShuffle(handle.shuffleId)
        new RapidsCachingWriter(
          env.blockManager,
          gpu.asInstanceOf[GpuShuffleHandle[K, V]],
          mapId,
          metricsReporter,
          getCatalogOrThrow,
          RapidsBufferCatalog.getDeviceStorage,
          server,
          gpu.dependency.metrics)
      case bmssh: BypassMergeSortShuffleHandle[_, _] =>
        bmssh.dependency match {
          case g: GpuShuffleDependency[_, _, _] if g.useMultiThreadedShuffle =>
            makeBypassMergeSortShuffleWriter(
              handle.asInstanceOf[BypassMergeSortShuffleHandle[K, V]],
              mapId, context,
              new ThreadSafeShuffleWriteMetricsReporter(metricsReporter))
          case _ =>
            wrapped.getWriter(handle, mapId, context, metricsReporter)
        }
      case _ =>
        wrapped.getWriter(handle, mapId, context, metricsReporter)
    }
  }

  override def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    handle match {
      case gpu: GpuShuffleHandle[_, _] =>
        logInfo(s"Asking map output tracker for dependency ${gpu.dependency}, " +
            s"map output sizes for: ${gpu.shuffleId}, parts=$startPartition-$endPartition")
        if (gpu.dependency.keyOrdering.isDefined) {
          // very unlikely, but just in case
          throw new IllegalStateException("A key ordering was requested for a gpu shuffle "
              + s"dependency ${gpu.dependency.keyOrdering.get}, this is not supported.")
        }

        val nvtxRange = new NvtxRange("getMapSizesByExecId", NvtxColor.CYAN)
        val blocksByAddress = try {
          SparkEnv.get.mapOutputTracker.getMapSizesByExecutorId(gpu.shuffleId,
            startMapIndex, endMapIndex, startPartition, endPartition)
        } finally {
          nvtxRange.close()
        }

        new RapidsCachingReader(rapidsConf, localBlockManagerId,
          blocksByAddress,
          context,
          metrics,
          transport,
          getCatalogOrThrow,
          gpu.dependency.sparkTypes)
      case other if rapidsConf.isMultiThreadedShuffleManagerMode =>
        val nvtxRange = new NvtxRange("getMapSizesByExecId", NvtxColor.CYAN)
        val blocksByAddress = try {
          SparkEnv.get.mapOutputTracker.getMapSizesByExecutorId(other.shuffleId,
            startMapIndex, endMapIndex, startPartition, endPartition)
        } finally {
          nvtxRange.close()
        }
        new RapidsShuffleThreadedReader(
          handle = other.asInstanceOf[BaseShuffleHandle[K,C,C]],
          blocksByAddress,
          context,
          metrics,
          shuffleReaderThreads = rapidsConf.shuffleMultiThreadedWriterThreads,
          shouldBatchFetch = true)
      case other =>
        val shuffleHandle = RapidsShuffleInternalManagerBase.unwrapHandle(other)
        wrapped.getReader(shuffleHandle, startMapIndex, endMapIndex, startPartition,
          endPartition, context, metrics)
    }
  }

  def registerGpuShuffle(shuffleId: Int): Unit = {
    val catalog = GpuShuffleEnv.getCatalog
    if (catalog != null) {
      // Note that in local mode this can be called multiple times.
      logInfo(s"Registering shuffle $shuffleId")
      catalog.registerShuffle(shuffleId)
    }
  }

  def unregisterGpuShuffle(shuffleId: Int): Unit = {
    val catalog = GpuShuffleEnv.getCatalog
    if (catalog != null) {
      logInfo(s"Unregistering shuffle $shuffleId")
      catalog.unregisterShuffle(shuffleId)
    }
  }

  override def unregisterShuffle(shuffleId: Int): Boolean = {
    unregisterGpuShuffle(shuffleId)
    wrapped.unregisterShuffle(shuffleId)
  }

  override def shuffleBlockResolver: ShuffleBlockResolver = resolver

  override def stop(): Unit = synchronized {
    wrapped.stop()
    if (!stopped) {
      stopped = true
      server.foreach(_.close())
      transport.foreach(_.close())
      if (!isDriver && rapidsConf.isMultiThreadedShuffleManagerMode) {
        RapidsShuffleInternalManagerBase.stopThreadPool()
      }
    }
  }
}

/**
 * Trait that makes it easy to check whether we are dealing with the
 * a RAPIDS Shuffle Manager
 */
trait RapidsShuffleManagerLike {
  def isDriver: Boolean
  def initialize: Unit
}

/**
 * A simple proxy wrapper allowing to delay loading of the
 * real implementation to a later point when ShimLoader
 * has already updated Spark classloaders.
 *
 * @param conf
 * @param isDriver
 */
class ProxyRapidsShuffleInternalManagerBase(
    conf: SparkConf,
    override val isDriver: Boolean
) extends RapidsShuffleManagerLike with Proxy {

  // touched in the plugin code after the shim initialization
  // is complete
  lazy val self: ShuffleManager = ShimLoader.newInternalShuffleManager(conf, isDriver)
      .asInstanceOf[ShuffleManager]

  // This function touches the lazy val `self` so we actually instantiate
  // the manager. This is called from both the driver and executor.
  // In the driver, it's mostly to display information on how to enable/disable the manager,
  // in the executor, the UCXShuffleTransport starts and allocates memory at this time.
  override def initialize: Unit = self

  def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter
  ): ShuffleWriter[K, V] = {
    self.getWriter(handle, mapId, context, metrics)
  }

  def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    self.getReader(handle,
      startMapIndex, endMapIndex, startPartition, endPartition,
      context, metrics)
  }

  def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]
  ): ShuffleHandle = {
    self.registerShuffle(shuffleId, dependency)
  }

  def unregisterShuffle(shuffleId: Int): Boolean = self.unregisterShuffle(shuffleId)

  def shuffleBlockResolver: ShuffleBlockResolver = self.shuffleBlockResolver

  def stop(): Unit = self.stop()
}
