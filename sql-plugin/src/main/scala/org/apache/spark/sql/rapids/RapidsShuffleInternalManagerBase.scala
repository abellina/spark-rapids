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

import java.io.{ByteArrayOutputStream, File, FileInputStream, OutputStream}
import java.util.{Optional, Random}
import java.util.concurrent.{Executors, LinkedBlockingQueue}
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.shims.SparkShimImpl
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleRequestHandler, RapidsShuffleServer, RapidsShuffleTransport}
import org.sparkproject.guava.io.Closeables

import org.apache.spark.{ShuffleDependency, SparkConf, SparkEnv, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.shuffle.{ShuffleWriter, _}
import org.apache.spark.shuffle.api._
import org.apache.spark.shuffle.api.metadata.MapOutputCommitMessage
import org.apache.spark.shuffle.sort.{BypassMergeSortShuffleHandle, SerializedShuffleHandle, SortShuffleManager, UnsafeShuffleWriter}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.rapids.shims.GpuShuffleBlockResolver
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage._
import org.apache.spark.util.Utils

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

  class Pool(tid: Int) {
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

  var numPools: Int = 128

  lazy val pools = new mutable.HashMap[Int, Pool]()

  def queueTask(part: Int, task: () => Unit): Unit = {
    pools(part % numPools).offer(task)
  }

  def startPoolsIfNeeded(): Unit = synchronized  {
    if (pools.isEmpty) {
      (0 until numPools).foreach { i  =>
        pools.put(i, new Pool(i)) }
    }
  }

  // used to place partition writers in the task pool in a pseudo-random thread
  val slotNumber = new AtomicInteger(0)
}

trait RapidsShuffleWriterShimHelper {
  def setChecksumIfNeeded(writer: DiskBlockObjectWriter, partition: Int): Unit = {
    // noop
  }

  def getPartitionLengths(): Array[Long] =
    throw new UnsupportedOperationException(
      "getPartitionLengths was added in Spark 3.2.0 yet it is being called for older versions.")

  def commitAllPartitions(writer: ShuffleMapOutputWriter): Array[Long] =
    throw new UnsupportedOperationException(
      "commitAllPartitions must be implemented by subclasses of this trait")
}

abstract class RapidsShuffleThreadedWriter[K, V](
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

  logInfo(s"Starting threaded writer for ${mapId} and ${handle}")

  private var myMapStatus: Option[MapStatus] = None

  private val dep: ShuffleDependency[K, V, V] = handle.dependency
  private val shuffleId = dep.shuffleId
  private val partitioner = dep.partitioner
  private val numPartitions = partitioner.numPartitions
  private val serializer = dep.serializer.newInstance()

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    withResource(new NvtxRange("ThreadedWriter.write", NvtxColor.RED)) { _ =>
      withResource(new NvtxRange("compute", NvtxColor.GREEN)) { _ =>
        val mapOutputWriter = shuffleExecutorComponents.createMapOutputWriter(
          shuffleId,
          mapId,
          numPartitions)

        val diskBlockObjectWriters = new mutable.HashMap[Int, (Int, DiskBlockObjectWriter)]()
        // per reduce partition id
        // open all the writers ahead of time (Spark does this already)
        (0 until numPartitions).map { i =>
          val (blockId, file) = blockManager.diskBlockManager.createTempShuffleBlock()
          val writer: DiskBlockObjectWriter = blockManager.getDiskWriter(
            blockId, file, serializer, 4 * 1024, writeMetrics)
          setChecksumIfNeeded(writer, i) // spark3.2.0+

          // Places writer objects at round robin slot numbers apriori
          // this choice is for simplicity but likely needs to change so that
          // we can handle skew better
          val slotNumber = Math.abs(RapidsShuffleInternalManagerBase.slotNumber.incrementAndGet())
          diskBlockObjectWriters.put(i, (slotNumber, writer))
        }

        // we call write on every writer for every record in parallel
        val scheduledWrites = new AtomicLong(0L)
        val doneQueue = new ArrayBuffer[FileSegment]()
        records.foreach { record =>
          val key = record._1
          val value = record._2
          val reducePartitionId: Int = partitioner.getPartition(key)
          scheduledWrites.incrementAndGet()
          val (slot, myWriter) = diskBlockObjectWriters(reducePartitionId)

          // we close batches actively in the `records` iterator as we get the next batch
          // this makes sure it is kept alive while a task is able to handle it.
          val cb = if (value.isInstanceOf[ColumnarBatch]) {
            val cb = value.asInstanceOf[ColumnarBatch]
            (0 until cb.numCols()).foreach {
              c => cb.column(c).asInstanceOf[SlicedGpuColumnVector].getBase.incRefCount()
            }
            cb
          } else {
            null
          }

          RapidsShuffleInternalManagerBase.queueTask(slot, () => {
            if (key == null) {
              logWarning("NULL KEY??")
            } else {
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
            doneQueue.synchronized {
              scheduledWrites.decrementAndGet()
              doneQueue.notifyAll()
            }
          })
        }

        withResource(new NvtxRange("WaitingForWrites", NvtxColor.PURPLE)) { _ =>
          doneQueue.synchronized {
            while (scheduledWrites.get() > 0) {
              doneQueue.wait()
            }
          }
        }
        // this is similar to Spark
        withResource(new NvtxRange("CommitShuffle", NvtxColor.RED)) { _ =>
          // per reduce partition
          var offset = 0L
          val segments = (0 until numPartitions).map {
            reducePartitionId =>
              val segment = diskBlockObjectWriters(reducePartitionId)._2.commitAndGet()
              val file = segment.file
              val res = (reducePartitionId, file, offset)
              offset += segment.length
              res
          }

          segments.foreach { case (reducePartitionId, file, offset) =>
            val partWriter = mapOutputWriter.getPartitionWriter(reducePartitionId)
            if (file.exists()) {
              val maybeOutputChannel: Optional[WritableByteChannelWrapper] =
                partWriter.openChannelWrapper()
              if (maybeOutputChannel.isPresent) {
                writePartitionedDataWithChannel(file, maybeOutputChannel.get(), offset)
              } else {
                writePartitionedDataWithStream(file, partWriter)
              }
              file.delete()
            }
            diskBlockObjectWriters(reducePartitionId)._2.close()
          }

          val partLengths = commitAllPartitions(mapOutputWriter)
          myMapStatus = Some(MapStatus(blockManager.shuffleServerId, partLengths, mapId))
        }
      }
    }
  }

  def writePartitionedDataWithStream(file: java.io.File, writer: ShufflePartitionWriter): Unit = {
    val in = new java.io.FileInputStream(file)
    val os: OutputStream = writer.openStream()
    logDebug(s"Writing segment from ${file} to ${writer}")
    org.apache.spark.util.Utils.copyStream(in, os, false, false)
    os.close()
    in.close()
  }

  def writePartitionedDataWithChannel(
    file: File, outputChannel: WritableByteChannelWrapper, offset: Long): Unit = {
    val in = new FileInputStream(file)
    val inputChannel = in.getChannel
    Utils.copyFileStreamNIO(
      inputChannel, outputChannel.channel, 0L, inputChannel.size)
    Closeables.close(in, false)
    inputChannel.close()
    Closeables.close(outputChannel, false)
  }

  override def stop(success: Boolean): Option[MapStatus] = myMapStatus
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

  if (!isDriver && rapidsConf.shuffleThreads > 0) {
    RapidsShuffleInternalManagerBase.numPools = rapidsConf.shuffleThreads
    RapidsShuffleInternalManagerBase.startPoolsIfNeeded()
  }

  protected val wrapped = new SortShuffleManager(conf)

  private[this] val transportEnabledMessage =
    if (!rapidsConf.shuffleTransportEnabled || rapidsConf.shuffleThreads > 0) {
      if (rapidsConf.shuffleThreads == 0) {
        "Transport disabled (local cached blocks only)"
      } else {
        "Transport disabled (threaded shuffle writer)"
      }
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
    if (rapidsConf.shuffleThreads > 0) {
      fallThroughReasons += "Using the threaded shuffle writer"
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
    if (rapidsConf.shuffleTransportEnabled && !isDriver && rapidsConf.shuffleThreads == 0) {
      Some(RapidsShuffleTransport.makeTransport(blockManager.shuffleServerId, rapidsConf))
    } else {
      None
    }
  }

  private[this] lazy val server: Option[RapidsShuffleServer] = {
    if (rapidsConf.shuffleTransportEnabled && !isDriver && rapidsConf.shuffleThreads == 0) {
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
    val extraConfigs = conf.getAllWithPrefix(ShuffleDataIOUtils.SHUFFLE_SPARK_CONF_PREFIX).toMap
    executorComponents.initializeExecutor(
      conf.getAppId,
      SparkEnv.get.executorId,
      extraConfigs.asJava)
    Some(executorComponents)
  }

  def getWriterInternal[K, V](
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
      case _ =>
        wrapped.getWriter(handle, mapId, context, metricsReporter)
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

/**
 * A simple proxy wrapper allowing to delay loading of the
 * real implementation to a later point when ShimLoader
 * has already updated Spark classloaders.
 *
 * @param conf
 * @param isDriver
 */
// TODO: make this the internal manager base, so remove proxy
//   can we remove VisibleShuffleManager or call it diferently
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
