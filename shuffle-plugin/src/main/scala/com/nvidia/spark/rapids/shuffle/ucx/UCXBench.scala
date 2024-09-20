package com.nvidia.spark.rapids.shuffle.ucx

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.{GpuDeviceManager, MetaUtils, RapidsConf, RapidsBuffer, RapidsBufferCatalog, RapidsBufferHandle, RapidsBufferId, ShuffleReceivedBufferCatalog, StorageTier}
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleFetchHandler, RapidsShuffleRequestHandler}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.{GpuShuffleEnv, RapidsDiskBlockManager}
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.ShuffleBlockBatchId

import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import com.nvidia.spark.rapids.ShimLoader

// make the trait open
// make the impl part of shims
// use shim loader to get impl
class UCXBench(
  localHost: String,
  localPort: String,
  peerHost: String,
  peerPort: String,
  maxInFlight: Integer) 
    extends Logging {

  def start(): Unit = {
    val server = peerHost == null
    val myId = if (server) { "0" } else { "1" }
    val rowCount = 1000000
    val batchSize = rowCount * 8

    val rapidsConf = new RapidsConf(
      Map(
          "spark.rapids.shuffle.ucx.activeMessages.forceRnd" -> "false",
          "spark.rapids.shuffle.ucx.bounceBuffers.device.count" -> "128",
          "spark.rapids.shuffle.ucx.bounceBuffers.device.count" -> "128",
          "spark.rapids.shuffle.ucx.bounceBuffers.size" -> "8MB",
          "spark.rapids.shuffle.transport.maxReceiveInflightBytes"-> "4GB",
          "spark.rapids.shuffle.ucx.useWakeup" -> "false",
          "spark.rapids.shuffle.ucx.listenerStartPort" -> localPort,
          "spark.rapids.memory.gpu.allocFraction" -> "0.3",
          "spark.rapids.memory.gpu.minAllocFraction"->"0.1"))

    GpuDeviceManager.initializeMemory(None, Some(rapidsConf))

    val receiveCatalog = new ShuffleReceivedBufferCatalog(RapidsBufferCatalog.singleton)
    GpuShuffleEnv.setReceivedBufferCatalog(receiveCatalog)

    val ucx = new UCXShuffleTransport(
      TrampolineUtil.newBlockManagerId(
        myId, localHost, localPort.toInt, Some(s"rapids=${localPort}")),
      rapidsConf
    )

    val longs = new Array[Long](rowCount)
    val ct =
      withResource(ai.rapids.cudf.ColumnVector.fromLongs(longs:_*)) { cv =>
        withResource(new ai.rapids.cudf.Table(cv)) { tbl =>
          tbl.contiguousSplit()
        }
    }.head

    val tableMeta = MetaUtils.buildTableMeta(1, ct)

    val ucxServer = ucx.makeServer(new RapidsShuffleRequestHandler {
      override def getShuffleBufferMetas(
          shuffleBlockBatchId: ShuffleBlockBatchId): Seq[TableMeta] = {
        logInfo("received getShuffleBufferMetas request")
        Seq(tableMeta)
      }

      override def acquireShuffleBuffer(tableId: Int): RapidsBuffer = {
        new RapidsBuffer {
          override val id: RapidsBufferId = new RapidsBufferId {
            override val tableId: Int = 1
            override def getDiskPath(diskBlockManager: RapidsDiskBlockManager): File = null
          }
          override val memoryUsedBytes: Long = tableMeta.bufferMeta().uncompressedSize()
          override def meta: TableMeta = tableMeta
          override val storageTier: StorageTier = StorageTier.DEVICE
          override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = null
          override def getMemoryBuffer: MemoryBuffer = {
            val b = ct.getBuffer
            b.incRefCount()
            b
          }
          override def copyToMemoryBuffer(
            srcOffset: Long,
            dst: MemoryBuffer,
            dstOffset: Long,
            length: Long,
            stream: Cuda.Stream): Unit = {
            dst.copyFromMemoryBufferAsync(dstOffset, getMemoryBuffer, srcOffset, length, stream)
          }
          override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
            val b = ct.getBuffer
            b.incRefCount()
            b
          }
          override def getHostMemoryBuffer: HostMemoryBuffer = null
          override def addReference(): Boolean = true
          override def free(): Unit = {}
          override def getSpillPriority: Long = 0
          override def setSpillPriority(priority: Long): Unit = {}
          override def withMemoryBufferReadLock[K](body: MemoryBuffer => K): K = {
            throw new NotImplementedError("error")
          }
          override def withMemoryBufferWriteLock[K](body: MemoryBuffer => K): K = {
            throw new NotImplementedError("error")
          }
          override def close(): Unit = {}
        }
      }
    })
    ucxServer.start()

    logInfo("press ctrl-c to exit")

    val received = new AtomicLong(0L)
    if (!server) {
      val clientProducer = Executors.newSingleThreadExecutor()
      val reqsInFlight = new LinkedBlockingQueue[Int](maxInFlight)
      val fetchHandler = new RapidsShuffleFetchHandler {
        override def start(expectedBatches: Int): Unit = {
        }

        override def batchReceived(handle: RapidsBufferHandle): Boolean = {
          received.addAndGet(batchSize)
          handle.close()
          reqsInFlight.poll()
          true
        }

        override def transferError(errorMessage: String, throwable: Throwable): Unit = {}

        override def getTaskIds: Array[Long] = {
          Array.empty
        }
      }

      val client  =
        ucx.makeClient(TrampolineUtil.newBlockManagerId(
          "0", peerHost, peerPort.toInt, Some(s"rapids=${peerPort}")))

      // client.mockTableMeta = Some(tableMeta)

      Thread.sleep(1000L)

      clientProducer.execute(() => {
        while (true) {
          val doFetch = reqsInFlight.offer(1, 1, TimeUnit.SECONDS)
          if (doFetch) {
            client.doFetch(ShuffleBlockBatchId(1, 1L, 1, 1) :: Nil, fetchHandler)
          }
        }
      })
      
      var ix = 0
      while (true) {
        Thread.sleep(1000L)
        val sofar = received.getAndSet(0L)
        logInfo(s"$ix: received ${sofar/1024/1024} MB/s, inflight: ${reqsInFlight.size()}")
        ix += 1
      }
    }

    while (true) {
      Thread.sleep(1000L)
    }
  }
}

object UCXBench extends Logging {
  def main(args: Array[String]): Unit = {
    val isServer = args(0) == "-s"
    val localHost = args(1)
    val localPort = args(2)
    val peerHost = if (isServer) null else args(3)
    val peerPort = if (isServer) null else args(4)
    val maxInFlight: Integer = if (isServer) null else args(5).toInt
    val b = 
      ShimLoader.newUCXShuffleBench(
        localHost, localPort, peerHost, peerPort, maxInFlight)
        .asInstanceOf[UCXBench]
    b.start()
  }
}