package com.nvidia.spark.rapids.shuffle.ucx

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.{GpuDeviceManager, MetaUtils, RapidsBuffer, RapidsBufferCatalog, RapidsBufferHandle, RapidsBufferId, ShuffleReceivedBufferCatalog, StorageTier}
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleFetchHandler, RapidsShuffleRequestHandler}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.{GpuShuffleEnv, RapidsDiskBlockManager}
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.ShuffleBlockBatchId

import java.io.File

object UCXBench extends Logging {
  def main(args: Array[String]): Unit = {
    val server = args(0).equalsIgnoreCase("-s")
    val localHost = args(1)
    val localPort = args(2)
    val myId = if (server) { "0" } else { "1" }

    val rapidsConf = new com.nvidia.spark.rapids.RapidsConf(
      Map("spark.rapids.shuffle.ucx.bounceBuffers.device.count" -> "256",
          "spark.rapids.shuffle.ucx.bounceBuffers.host.count" -> "256",
          "spark.rapids.shuffle.transport.maxReceiveInflightBytes"-> "4GB",
          "spark.rapids.shuffle.ucx.useWakeup" -> "true",
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


    val longs = new Array[Long](1000000)
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
          override def getMemoryBuffer: MemoryBuffer = ct.getBuffer
          override def copyToMemoryBuffer(
            srcOffset: Long,
            dst: MemoryBuffer,
            dstOffset: Long,
            length: Long,
            stream: Cuda.Stream): Unit = {
            dst.copyFromMemoryBufferAsync(dstOffset, getMemoryBuffer, srcOffset, length, stream)
          }
          override def getDeviceMemoryBuffer: DeviceMemoryBuffer = ct.getBuffer
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

    if (!server) {
      val peer = args(3)
      val peerPort = args(4)
      val client  =
        ucx.makeClient(TrampolineUtil.newBlockManagerId(
          "0", peer, peerPort.toInt, Some(s"rapids=${peerPort}")))

      Thread.sleep(1000L)

      (0 until 1000).foreach { _ =>
        client.doFetch(ShuffleBlockBatchId(1, 1L, 1, 1) :: Nil, new RapidsShuffleFetchHandler {
          override def start(expectedBatches: Int): Unit = {
          }

          override def batchReceived(handle: RapidsBufferHandle): Boolean = {
            handle.close()
            true
          }

          override def transferError(errorMessage: String, throwable: Throwable): Unit = {}

          override def getTaskIds: Array[Long] = {
            Array.empty
          }
        })
      }
      logInfo("done issuing")
    }

    logInfo("press ctrl-c to exit")
    while (true) {
      Thread.sleep(100000L)
    }
  }
}
