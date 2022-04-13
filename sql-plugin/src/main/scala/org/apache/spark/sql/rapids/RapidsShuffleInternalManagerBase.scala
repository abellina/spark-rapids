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

import java.io.{ByteArrayOutputStream, File, InputStream, OutputStream}
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.{Callable, ConcurrentHashMap, ExecutorService, Executors, Future, LinkedBlockingQueue}
import java.util.Random
import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleRequestHandler, RapidsShuffleServer, RapidsShuffleTransport}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import org.apache.spark.shuffle.api.metadata.MapOutputCommitMessage
import org.apache.spark.{ShuffleDependency, SparkConf, SparkEnv, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.internal.config.SHUFFLE_UNSAFE_FAST_MERGE_ENABLE
import org.apache.spark.io.{CompressionCodec, NioBufferedFileInputStream}
import org.apache.spark.memory.TaskMemoryManager
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.network.util.LimitedInputStream
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.shuffle.{ShuffleWriter, _}
import org.apache.spark.shuffle.api._
import org.apache.spark.shuffle.sort.{BypassMergeSortShuffleHandle, RapidsShuffleExternalSorter, RapidsSpillInfo, SerializedShuffleHandle, SortShuffleManager, UnsafeShuffleWriter}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.rapids.shims.GpuShuffleBlockResolver
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage._
import org.apache.spark.unsafe.Platform
import org.sparkproject.guava.io.{ByteStreams, Closeables}

import java.util
import scala.collection.mutable
import scala.reflect.{ClassTag, classTag}

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


object RapidsShuffleInternalManagerBase extends Logging {
  def unwrapHandle(handle: ShuffleHandle): ShuffleHandle = handle match {
    case gh: GpuShuffleHandle[_, _] => gh.wrapped
    case other => other
  }

  class Pool {
    val tasks = new LinkedBlockingQueue[() => Unit]()
    val p = Executors.newSingleThreadExecutor()
    def offer(task: () => Unit): Unit = {
      tasks.offer(task)
    }
    val exec = p.submit[Unit](() => {
      while (true) {
        val t = tasks.take()
        t()
      }
    })
  }

  val numPools = 128

  lazy val pools = new mutable.HashMap[Int, Pool]()

  def queueTask(part: Int, task: () => Unit): Unit = {
    logDebug(s"$part: queueing task at ${part % numPools}")
    pools.get(part % numPools).get.offer(task)
  }

  def startPools(): Unit = synchronized  {
    if (pools.isEmpty) {
      (0 until numPools).foreach { i  =>
        pools.put(i, new Pool())
      }
    }
  }
}

class ThreadedUnsafeThreadedWriter[K, V](
    blockManager: BlockManager,
    taskMemoryManager: TaskMemoryManager,
    handle: SerializedShuffleHandle[K, V],
    mapId: Long,
    sparkConf: SparkConf,
    writeMetrics: ShuffleWriteMetricsReporter,
    shuffleExecutorComponents: ShuffleExecutorComponents)
   extends ShuffleWriter [K,V] with Logging {

  val numPartitions = handle.dependency.partitioner.numPartitions
  // TODO: check that numPartitions <= MAX_SHUFFLE_OUTPUT_PARTITIONS_FOR_SERIALIZED_MODE
  val dep = handle.dependency
  val shuffleId = dep.shuffleId
  val serializer = dep.serializer.newInstance()
  val partitioner = dep.partitioner
  val taskContext = TaskContext.get

  val sorter = new RapidsShuffleExternalSorter(
    taskMemoryManager, blockManager, taskContext, 4096,
    numPartitions, sparkConf, writeMetrics)

  /** Subclass of ByteArrayOutputStream that exposes `buf` directly. */
  class MyByteArrayOutputStream(override val size: Int)
    extends ByteArrayOutputStream(size) {
    def getBuf: Array[Byte] = buf
  }

  val serBuffer = new MyByteArrayOutputStream(1024*1024)
  val serOutputStream = serializer.serializeStream(serBuffer)
  //private val OBJECT_CLASS_TAG: ClassTag[K] = ClassTag[Object]()

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    try {
      records.foreach { record =>
        insertRecordIntoSorter(record)
      }
      closeAndWriteOutput()
    } finally {
      if (sorter != null) {
        sorter.cleanupResources()
      }
    }
  }

  private def insertRecordIntoSorter(record: Product2[K, V]): Unit = {
    val nv = new NvtxRange("writing  to stream", NvtxColor.RED)
    // per thread ? => so N SerOutputStreams
    val key = record._1
    val partitionId = partitioner.getPartition(key)
    serBuffer.reset()
    serOutputStream.writeKey[Object](key.asInstanceOf[Object])
    serOutputStream.writeValue[Object](record._2.asInstanceOf[Object])
    serOutputStream.flush
    nv.close()

    val nv2 = new NvtxRange("insert into sorter", NvtxColor.ORANGE)
    val serializedRecordSize = serBuffer.size
    sorter.insertRecord(
      serBuffer.getBuf,
      Platform.BYTE_ARRAY_OFFSET,
      serializedRecordSize,
      partitionId)
    nv2.close()
  }

  private var mapStatus: Option[MapStatus] = None

  private def closeAndWriteOutput(): Unit = {
    val spills = sorter.closeAndGetSpills()
    val partitionLengths = mergeSpills(spills)
    //mapStatus = Some(MapStatus(blockManager.shuffleServerId, partitionLengths, mapId))
  }

  /**
   * Merge zero or more spill files together, choosing the fastest merging strategy based on the
   * number of spills and the IO compression codec.
   *
   * @return the partition lengths in the merged file.
   */
  private def mergeSpills(spills: Array[RapidsSpillInfo]): Array[Long] = {
    var partitionLengths: Array[Long] = null
    if (spills.length == 0) {
      val mapWriter =
        shuffleExecutorComponents.createMapOutputWriter(
          shuffleId, mapId, partitioner.numPartitions)
      return mapWriter.commitAllPartitions().getPartitionLengths
    }
    else if (spills.length == 1) {
      //val maybeSingleFileWriter =
      //  shuffleExecutorComponents.createSingleFileMapOutputWriter(shuffleId, mapId)
      //if (maybeSingleFileWriter.isPresent) {
      //  // Here, we don't need to perform any metrics updates because the bytes written to this
      //  // output file would have already been counted as shuffle bytes written.
      //  partitionLengths = spills(0).partitionLengths
      //  // FIXME: logInfo("Merge shuffle spills for mapId {} with length {}", mapId, partitionLengths.length)
      //  //  maybeSingleFileWriter.get.transferMapSpillFile(
      //  //    spills(0).file, partitionLengths, sorter.getChecksums)
      //}
      partitionLengths = mergeSpillsUsingStandardWriter(spills)
    }
    else partitionLengths = mergeSpillsUsingStandardWriter(spills)
    partitionLengths
  }

  val transferToEnabled = true

  private def mergeSpillsUsingStandardWriter(spills: Array[RapidsSpillInfo]): Array[Long] = {
    var partitionLengths: Array[Long] = null
    val compressionEnabled: Boolean = true
    val compressionCodec = CompressionCodec.createCodec(sparkConf)
    val fastMergeEnabled: Boolean = sparkConf.get(SHUFFLE_UNSAFE_FAST_MERGE_ENABLE)
    val fastMergeIsSupported =
      !(compressionEnabled) ||
        CompressionCodec.supportsConcatenationOfSerializedStreams(compressionCodec)
    logInfo("supports compressed concatenated merge")

    val encryptionEnabled: Boolean = blockManager.serializerManager.encryptionEnabled
    val mapWriter: ShuffleMapOutputWriter =
      shuffleExecutorComponents.createMapOutputWriter(
        shuffleId, mapId, partitioner.numPartitions)

    try { // There are multiple spills to merge, so none of these spill files' lengths were counted
      // towards our shuffle write count or shuffle write time. If we use the slow merge path,
      // then the final output file's size won't necessarily be equal to the sum of the spill
      // files' sizes. To guard against this case, we look at the output file's actual size when
      // computing shuffle bytes written.
      //
      // We allow the individual merge methods to report their own IO times since different merge
      // strategies use different IO techniques.  We count IO during merge towards the shuffle
      // write time, which appears to be consistent with the "not bypassing merge-sort" branch in
      // ExternalSorter.
      if (fastMergeEnabled && fastMergeIsSupported) {
        // Compression is disabled or we are using an IO compression codec that supports
        // decompression of concatenated compressed streams, so we can perform a fast spill merge
        // that doesn't need to interpret the spilled bytes.
        //if (transferToEnabled && !(encryptionEnabled)) {
        //  logInfo("Using transferTo-based fast merge")
        //  mergeSpillsWithTransferTo(spills, mapWriter)
        //} else {
          logInfo("Using fileStream-based fast merge")
          mergeSpillsWithFileStream(spills, mapWriter, null)
       // }
      }
      else {
        logInfo("Using slow merge")
        mergeSpillsWithFileStream(spills, mapWriter, compressionCodec)
      }
      // When closing an UnsafeShuffleExternalSorter that has already spilled once but also has
      // in-memory records, we write out the in-memory records to a file but do not count that
      // final write as bytes spilled (instead, it's accounted as shuffle write). The merge needs
      // to be counted as shuffle write, but this will lead to double-counting of the final
      // SpillInfo's bytes.
      writeMetrics.decBytesWritten(spills(spills.length - 1).file.length)
      partitionLengths = mapWriter.commitAllPartitions().getPartitionLengths
    } catch {
      case e: Exception =>
        try mapWriter.abort(e)
        catch {
          case e2: Exception =>
            logWarning("Failed to abort writing the map output.", e2)
            e.addSuppressed(e2)
        }
        throw e
    }
    partitionLengths
  }

  /**
   * Merges spill files using Java FileStreams. This code path is typically slower than
   * the NIO-based merge, {@link UnsafeShuffleWriter# mergeSpillsWithTransferTo ( SpillInfo [ ],
   * ShuffleMapOutputWriter)}, and it's mostly used in cases where the IO compression codec
   * does not support concatenation of compressed data, when encryption is enabled, or when
   * users have explicitly disabled use of {@code transferTo} in order to work around kernel bugs.
   * This code path might also be faster in cases where individual partition size in a spill
   * is small and UnsafeShuffleWriter#mergeSpillsWithTransferTo method performs many small
   * disk ios which is inefficient. In those case, Using large buffers for input and output
   * files helps reducing the number of disk ios, making the file merging faster.
   *
   * @param spills           the spills to merge.
   * @param mapWriter        the map output writer to use for output.
   * @param compressionCodec the IO compression codec, or null if shuffle compression is disabled.
   * @return the partition lengths in the merged file.
   */

  private def mergeSpillsWithFileStream(
      spills: Array[RapidsSpillInfo],
      mapWriter: ShuffleMapOutputWriter,
      compressionCodec: CompressionCodec): Unit = {
    logDebug(s"Merge shuffle spills with FileStream for mapId $mapId")
    val numPartitions: Int = partitioner.numPartitions
    val spillInputStreams: Array[InputStream] = new Array[InputStream](spills.length)
    var threwException: Boolean = true
    try {
      for (i <- 0 until spills.length) {
        spillInputStreams(i) =
          new NioBufferedFileInputStream(spills(i).file, 1024*1024)
        // Only convert the partitionLengths when debug level is enabled.
        //logDebug("Partition lengths for mapId {} in Spill {}: {}",
        //  mapId, i, util.Arrays.toString(spills(i).partitionLengths))
      }
      for (partition <- 0 until numPartitions) {
        var copyThrewException: Boolean = true
        val writer: ShufflePartitionWriter = mapWriter.getPartitionWriter(partition)
        var partitionOutput: OutputStream = writer.openStream
        try {
          partitionOutput = new TimeTrackingOutputStream(writeMetrics, partitionOutput)
          partitionOutput = blockManager.serializerManager.wrapForEncryption(partitionOutput)
          if (compressionCodec != null) {
            partitionOutput = compressionCodec.compressedOutputStream(partitionOutput)
          }
          for (i <- 0 until spills.length) {
            val partitionLengthInSpill: Long = spills(i).partitionLengths(partition)
            if (partitionLengthInSpill > 0) {
              var partitionInputStream: InputStream = null
              var copySpillThrewException: Boolean = true
              try {
                partitionInputStream =
                  new LimitedInputStream(spillInputStreams(i), partitionLengthInSpill, false)
                partitionInputStream =
                  blockManager.serializerManager.wrapForEncryption(partitionInputStream)
                if (compressionCodec != null) {
                  partitionInputStream =
                    compressionCodec.compressedInputStream(partitionInputStream)
                }
                ByteStreams.copy(partitionInputStream, partitionOutput)
                copySpillThrewException = false
              } finally {
                Closeables.close(partitionInputStream, copySpillThrewException)
              }
            }
          }
          copyThrewException = false
        } finally {
          Closeables.close(partitionOutput, copyThrewException)
        }
        val numBytesWritten: Long = writer.getNumBytesWritten
        writeMetrics.incBytesWritten(numBytesWritten)
      }
      threwException = false
    } finally {
      // To avoid masking exceptions that caused us to prematurely enter the finally block, only
      // throw exceptions during cleanup if threwException == false.
      for (stream <- spillInputStreams) {
        Closeables.close(stream, threwException)
      }
    }
  }

  ///**
  // * Merges spill files by using NIO's transferTo to concatenate spill partitions' bytes.
  // * This is only safe when the IO compression codec and serializer support concatenation of
  // * serialized streams.
  // *
  // * @param spills    the spills to merge.
  // * @param mapWriter the map output writer to use for output.
  // * @return the partition lengths in the merged file.
  // */
  //@throws[IOException]
  //private def mergeSpillsWithTransferTo(spills: Array[SpillInfo], mapWriter: ShuffleMapOutputWriter): Unit = {
  //  logger.debug("Merge shuffle spills with TransferTo for mapId {}", mapId)
  //  val numPartitions: Int = partitioner.numPartitions
  //  val spillInputChannels: Array[FileChannel] = new Array[FileChannel](spills.length)
  //  val spillInputChannelPositions: Array[Long] = new Array[Long](spills.length)
  //  var threwException: Boolean = true
  //  try {
  //    for (i <- 0 until spills.length) {
  //      spillInputChannels(i) = new FileInputStream(spills(i).file).getChannel
  //      if (logger.isDebugEnabled) {
  //        logger.debug("Partition lengths for mapId {} in Spill {}: {}", mapId, i, Arrays.toString(spills(i).partitionLengths))
  //      }
  //    }
  //    for (partition <- 0 until numPartitions) {
  //      var copyThrewException: Boolean = true
  //      val writer: ShufflePartitionWriter = mapWriter.getPartitionWriter(partition)
  //      val resolvedChannel: WritableByteChannelWrapper = writer.openChannelWrapper.orElseGet(() => new UnsafeShuffleWriter.StreamFallbackChannelWrapper(openStreamUnchecked(writer)))
  //      try for (i <- 0 until spills.length) {
  //        val partitionLengthInSpill: Long = spills(i).partitionLengths(partition)
  //        val spillInputChannel: FileChannel = spillInputChannels(i)
  //        val writeStartTime: Long = System.nanoTime
  //        Utils.copyFileStreamNIO(spillInputChannel, resolvedChannel.channel, spillInputChannelPositions(i), partitionLengthInSpill)
  //        copyThrewException = false
  //        spillInputChannelPositions(i) += partitionLengthInSpill
  //        writeMetrics.incWriteTime(System.nanoTime - writeStartTime)
  //      }
  //      finally {
  //        Closeables.close(resolvedChannel, copyThrewException)
  //      }
  //      val numBytes: Long = writer.getNumBytesWritten
  //      writeMetrics.incBytesWritten(numBytes)
  //    }
  //    threwException = false
  //  } finally {
  //    for (i <- 0 until spills.length) {
  //      assert((spillInputChannelPositions(i) == spills(i).file.length))
  //      Closeables.close(spillInputChannels(i), threwException)
  //    }
  //  }
  //}

  override def stop(success: Boolean): Option[MapStatus] = {
    mapStatus
  }
}

class ThreadedWriter[K, V](
    blockManager: BlockManager,
    handle: BypassMergeSortShuffleHandle[K, V],
    mapId: Long,
    sparkConf: SparkConf,
    writeMetrics: ShuffleWriteMetricsReporter,
    shuffleExecutorComponents: ShuffleExecutorComponents) extends ShuffleWriter [K,V] with Logging {
  //extends org.apache.spark.shuffle.sort.BypassMergeSortShuffleWriter[K, V](
  //  blockManager, handle, mapId,
  //  sparkConf, writeMetrics, shuffleExecutorComponents) {

  var myMapStatus: Option[MapStatus] = None
  var fs: Array[FileSegment] = null


  var stillWriting: Boolean = true
  val rng = new Random(0)

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    val nvtxRange = new NvtxRange("ThreadedWriter.write", NvtxColor.RED)
    RapidsShuffleInternalManagerBase.startPools()
    val serializer = handle.dependency.serializer.newInstance()

    val writer = shuffleExecutorComponents.createMapOutputWriter(
      handle.shuffleId,
      mapId,
      handle.dependency.partitioner.numPartitions)

    val writers = new mutable.HashMap[Int, (Int, DiskBlockObjectWriter)]()
    // per reduce partition id
    (0 until handle.dependency.partitioner.numPartitions).map { i =>
      logDebug(s"Creating writer for partition $i")
      val r1 = new NvtxRange(s"creating writer", NvtxColor.GREEN)
      val (blockId, file) = blockManager.diskBlockManager.createTempShuffleBlock()
      writers.put(i, (Math.abs(rng.nextInt()), blockManager.getDiskWriter(
        blockId, file, serializer, 4 * 1024, writeMetrics)))
      r1.close()
    }

    val scheduledWrites = new AtomicLong(0L)
    val doneQueue = new ArrayBuffer[FileSegment]()
    val r = new NvtxRange("foreach", NvtxColor.DARK_GREEN)
    records.foreach { case (key, value) =>
      val reducePartitionId = handle.dependency.partitioner.getPartition(key)
      logDebug(s"Writing $reducePartitionId from ${TaskContext.get().taskAttemptId()}")
      scheduledWrites.incrementAndGet()
      val (slot, myWriter) = writers(reducePartitionId)
      val cb = if (value.isInstanceOf[ColumnarBatch]) {
        val cb = value.asInstanceOf[ColumnarBatch]
        (0 until cb.numCols()).foreach {
          c => cb.column(c).asInstanceOf[SlicedGpuColumnVector].getBase.incRefCount()
        }
        cb
      } else {
        null
      }

      // per reducer partition
      RapidsShuffleInternalManagerBase.queueTask(slot, () => {
        val r2 = new NvtxRange("Draining", NvtxColor.ORANGE)
        if (key == null) {
          logWarning("NULL KEY??")
        } else {
          logDebug(s"writing ${key} and ${value}")
          try {
            myWriter.write(key, value)
          } catch {
            case e: Throwable => logError("Error writing", e)
          }
          if (cb != null) {
            (0 until cb.numCols()).foreach {
              c => cb.column(c).asInstanceOf[SlicedGpuColumnVector].close()
            }
          }
        }
        logDebug(s"done writing ${key} and ${value}")
        r2.close()
        doneQueue.synchronized {
          scheduledWrites.decrementAndGet()
          doneQueue.notifyAll()
        }
      })
    }
    r.close()

    val r2a = new NvtxRange("waiting..", NvtxColor.PURPLE)
    doneQueue.synchronized {
      while(scheduledWrites.get() > 0) {
        doneQueue.wait()
      }
    }
    r2a.close()

    val r3 = new NvtxRange("committing...", NvtxColor.RED)
    val r4 = new NvtxRange("commit_and_get", NvtxColor.ORANGE)
    // per reduce partition
    val partWriters = (0 until handle.dependency.partitioner.numPartitions).map { reducePartitionId =>
      val segment = writers(reducePartitionId)._2.commitAndGet()
      val file = segment.file
      (reducePartitionId, file, writer.getPartitionWriter(reducePartitionId))
    }
    r4.close()


    val r5 = new NvtxRange("write partitioned by map", NvtxColor.GREEN)
    partWriters.foreach { case (_, file, partWriter) =>
      if (file.exists()) {
        writePartitionedDataWithStream(file, partWriter)
      }
    }
    r5.close()
    val r6 = new NvtxRange("commit all", NvtxColor.DARK_GREEN)
    val lengths = writer.commitAllPartitions().getPartitionLengths
    myMapStatus = Some(MapStatus(blockManager.shuffleServerId, lengths, mapId))
    r6.close()
    r3.close()
    nvtxRange.close()
    stillWriting.synchronized {
      stillWriting = false
    }
  }

  def writePartitionedDataWithStream(file: java.io.File, writer: ShufflePartitionWriter): Unit = {
    val in = new java.io.FileInputStream(file)
    var os: OutputStream = writer.openStream()
    logDebug(s"Writing segment from ${file} to ${writer}")
    org.apache.spark.util.Utils.copyStream(in, os, false, false)
    os.close()
    in.close()
  }

  override def stop(success: Boolean): Option[MapStatus] = {
    stillWriting.synchronized { 
      if (stillWriting) {
        throw new IllegalStateException("still writing")
      }
    }
    myMapStatus
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
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_PRIORITY)
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
                SpillPriorities.OUTPUT_FOR_SHUFFLE_INITIAL_PRIORITY)
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
        logDebug(s"Done caching shuffle success=$success, server_id=$shuffleServerId, "
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

  protected val wrapped = new SortShuffleManager(conf)

  private[this] val transportEnabledMessage = if (!rapidsConf.shuffleTransportEnabled) {
    "Transport disabled (local cached blocks only)"
  } else {
    s"Transport enabled (remote fetches will use ${rapidsConf.shuffleTransportClassName}"
  }

  logWarning(s"Rapids Shuffle Plugin enabled. ${transportEnabledMessage}. To disable the " +
      s"RAPIDS Shuffle Manager set `${RapidsConf.SHUFFLE_MANAGER_ENABLED}` to false")

  //Many of these values like blockManager are not initialized when the constructor is called,
  // so they all need to be lazy values that are executed when things are first called

  // NOTE: this can be null in the driver side.
  private lazy val env = SparkEnv.get
  private lazy val blockManager = env.blockManager
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

  protected lazy val resolver = if (shouldFallThroughOnEverything) {
    wrapped.shuffleBlockResolver
  } else {
    new GpuShuffleBlockResolver(wrapped.shuffleBlockResolver, getCatalogOrThrow)
  }

  private[this] lazy val transport: Option[RapidsShuffleTransport] = {
    if (rapidsConf.shuffleTransportEnabled && !isDriver) {
      Some(RapidsShuffleTransport.makeTransport(blockManager.shuffleServerId, rapidsConf))
    } else {
      None
    }
  }

  private[this] lazy val server: Option[RapidsShuffleServer] = {
    if (rapidsConf.shuffleTransportEnabled && !isDriver) {
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
      case _ if shouldFallThroughOnEverything => orig
      case gpuDependency: GpuShuffleDependency[K, V, C] if gpuDependency.useRapidsShuffle =>
        new GpuShuffleHandle(orig,
          dependency.asInstanceOf[GpuShuffleDependency[K, V, V]])
      case _ => orig
    }
  }

  lazy val execComponents: Option[ShuffleExecutorComponents] = {
    import scala.collection.JavaConverters._
    val executorComponents = ShuffleDataIOUtils.loadShuffleDataIO(conf).executor()
    // TODO: extra configs
    executorComponents.initializeExecutor(
      conf.getAppId, SparkEnv.get.executorId, Map.empty[String, String].asJava)
    Some(executorComponents)
  }

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
      case other => other match {
        case _: BypassMergeSortShuffleHandle[_, _] =>
          new ThreadedWriter[K, V](
            blockManager, 
            other.asInstanceOf[BypassMergeSortShuffleHandle[K, V]], mapId, 
            conf, 
            metricsReporter, 
            execComponents.get)
        case _: SerializedShuffleHandle[_, _] =>
          new ThreadedUnsafeThreadedWriter[K, V](
            blockManager, TaskContext.get().taskMemoryManager(),
            other.asInstanceOf[SerializedShuffleHandle[K, V]], 
            mapId, conf, metricsReporter, execComponents.get)
      }
    }
  }

  def getReaderInternal[K, C](
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
      case other => {
        val shuffleHandle = RapidsShuffleInternalManagerBase.unwrapHandle(other)
        wrapped.getReader(shuffleHandle, startPartition, endPartition, context, metrics)
      }
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
    }
  }
}

/**
 * Trait that makes it easy to check whether we are dealing with the
 * a RAPIDS Shuffle Manager
 *
 * TODO name does not match its function anymore
 */
trait VisibleShuffleManager {
  def isDriver: Boolean
  def initialize: Unit
}


class RapidsShuffleExecutorComponents extends ShuffleExecutorComponents with Logging with Arm {
  override def initializeExecutor(
    appId: String,
    execId: String,
    extraConfigs: java.util.Map[String, String]) = {
    logWarning(s"initializing executor ${execId}")
  }

  override def createMapOutputWriter(
    shuffleId: Int, mapTaskId: Long, numPartitions: Int): ShuffleMapOutputWriter = {
    withResource(new NvtxRange("createMapOutputWriter", NvtxColor.YELLOW)) { _ =>
      logWarning(s"get map output writer for ${shuffleId} ${mapTaskId} ${numPartitions}")
      val bbos = new ByteArrayOutputStream()
      val nvbbos = new OutputStream {
        override def write(i: Int): Unit = {
          withResource(new NvtxRange("write", NvtxColor.CYAN)) { _ =>
          }
        }
      }
      new ShuffleMapOutputWriter {
        val partLengths = new ArrayBuffer[Long]()
        override def getPartitionWriter(reducePartitionId: Int): ShufflePartitionWriter = {
          withResource(new NvtxRange("getPartitionWriter", NvtxColor.ORANGE)) { _ =>
            logInfo(s"getPartitionWriter ${reducePartitionId}")
            partLengths.append(0L)
            new ShufflePartitionWriter {
              override def openStream(): OutputStream = {
                withResource(new NvtxRange("open_stream", NvtxColor.GREEN)) { _ =>
                  logWarning(s"opening stream for ${reducePartitionId}")
                  nvbbos
                }
              }

              override def getNumBytesWritten: Long = {
                withResource(new NvtxRange("getNumBytesWritten", NvtxColor.DARK_GREEN)) { _ =>
                  bbos.size()
                }
              }
            }
          }
        }

        def commitAllPartitions(checksums: Array[Long]): MapOutputCommitMessage = {
          withResource(new NvtxRange("commit", NvtxColor.BLUE)) { _ =>
            logWarning(s"commit all partitions for ${shuffleId}_${mapTaskId}_${numPartitions}")
            MapOutputCommitMessage.of(partLengths.toArray)
          }
        }

        def commitAllPartitions(): MapOutputCommitMessage = {
          withResource(new NvtxRange("commit", NvtxColor.BLUE)) { _ =>
            logWarning(s"commit all partitions for ${shuffleId}_${mapTaskId}_${numPartitions}")
            MapOutputCommitMessage.of(partLengths.toArray)
          }
        }

        override def abort(error: Throwable): Unit = {
        }
      }
    }
  }
}

/**
 * A simple proxy wrapper allowing to delay loading of the
 * real implementation to a later point when ShimLoader
 * has already updated Spark classloaders.
 *
 * @param conf
 * @param isDriver
 */
class RapidsShuffleDataIO(conf: SparkConf) extends ShuffleDataIO with Logging with Arm {

  override def executor: ShuffleExecutorComponents = {
    logWarning("getting ShuffleExecutorComponents")
    new RapidsShuffleExecutorComponents()
  }

  override def driver: ShuffleDriverComponents = {
    logInfo("getting ShuffleDriverComponents")
    new ShuffleDriverComponents {
      override def initializeApplication: java.util.Map[String, String] = {
        logWarning("Initialize APP")
        new java.util.HashMap[String, String]()
      }
      override def cleanupApplication(): Unit = {
        logWarning("clean app")
      }
      override def registerShuffle(shuffleId: Int): Unit = {
        logWarning(s"register shuffle ${shuffleId}")
      }
      override def removeShuffle(shuffleId: Int, blocking: Boolean): Unit = {
        logWarning(s"un-register shuffle ${shuffleId}. Blocking? ${blocking}")
      }
    }
  }
}

abstract class ProxyRapidsShuffleInternalManagerBase(
    conf: SparkConf,
    override val isDriver: Boolean
) extends VisibleShuffleManager with Proxy with Logging {

  // touched in the plugin code after the shim initialization
  // is complete
  lazy val self: ShuffleManager = ShimLoader.newInternalShuffleManager(conf, isDriver)
      .asInstanceOf[ShuffleManager]

  // This function touches the lazy val `self` so we actually instantiate
  // the manager. This is called from both the driver and executor.
  // In the driver, it's mostly to display information on how to enable/disable the manager,
  // in the executor, the UCXShuffleTransport starts and allocates memory at this time.
  def initialize: Unit = self

  //
  // Signatures unchanged since 3.0.1 follow
  //

  def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter
  ): ShuffleWriter[K, V] = {
    self.getWriter(handle, mapId, context, metrics)
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
