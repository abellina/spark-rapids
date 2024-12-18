package com.nvidia.spark.rapids.shuffle.ucx

import java.util.Properties
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import com.nvidia.spark.rapids.{GpuDeviceManager, MetaUtils, RapidsConf, RapidsShuffleHandle, ShimLoader, ShuffleReceivedBufferCatalog, ThreadFactoryBuilder}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleFetchHandler, RapidsShuffleRequestHandler}
import com.nvidia.spark.rapids.spill.{SpillFramework, SpillableDeviceBufferHandle}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.storage.ShuffleBlockBatchId

// make the trait open
// make the impl part of shims
// use shim loader to get impl
class UCXBench(
  configPath: String,
  localHost: String,
  localPort: String,
  peerHost: String,
  peerPort: String,
  maxInFlight: Integer,
  numIter: Integer,
  msgSize: java.lang.Long)
    extends Logging {

  def start(): Unit = {
    val server = peerHost == null
    val myId = if (server) { "0" } else { "1" }
    val rowCount = 1000000
    val batchSize = rowCount * 8

    val properties = new Properties()
    val source = scala.io.Source.fromURL(s"file://$configPath")
    properties.load(source.bufferedReader())
    val configMap = scala.collection.mutable.HashMap[String, String]()
    properties.keySet().forEach { kobj => 
      val key = kobj.asInstanceOf[String]
      configMap.put(key, properties.getProperty(key))
    }
    configMap.put("spark.rapids.shuffle.ucx.listenerStartPort", localPort)

    val sb = new StringBuilder()
    sb.append("\n*********************************************\n")
    sb.append("****** NVIDIA spark-rapids UCXBench p2p \n")
    sb.append(s"*** mode=${if (server) "SERVER" else "CLIENT"} " +
      s"maxInFlight=$maxInFlight numIter: $numIter\n")
    sb.append(s"*** localHost=$localHost localPort=$localPort\n")
    if (!server) {
      sb.append(s"*** peerHost=$peerHost peerPort=$peerPort\n")
    }
    
    sb.append(s"*** Configuration: \n")
    configMap.foreach { case (k,v) => 
      sb.append(s"*** $k = $v\n")
    }
    sb.append("*********************************************\n")
    logInfo(sb.toString)

    val rapidsConf = new RapidsConf(configMap.toMap)

    GpuDeviceManager.initializeMemory(None, Some(rapidsConf))

    val receiveCatalog = new ShuffleReceivedBufferCatalog()
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
    var receivedFirst = false
    val handle = SpillableDeviceBufferHandle(ct.getBuffer)

    val ucxServer = ucx.makeServer(new RapidsShuffleRequestHandler {
      override def getShuffleBufferMetas(
          shuffleBlockBatchId: ShuffleBlockBatchId): Seq[TableMeta] = {
        receivedFirst = true
        Seq(tableMeta)
      }

      override def getShuffleHandle(tableId: Int): RapidsShuffleHandle = {
        RapidsShuffleHandle(handle, tableMeta)
      }
    })
    ucxServer.start()

    logInfo("press ctrl-c to exit")

    val received = new AtomicLong(0L)
    if (!server) {
      val clientProducer = Executors.newSingleThreadExecutor(
        new ThreadFactoryBuilder()
          .setNameFormat("ucx-client-producer")
          .setDaemon(true)
          .build)

      var reqsInFlight: LinkedBlockingQueue[Int] = null
      val fetchHandler = new RapidsShuffleFetchHandler {
        override def start(expectedBatches: Int): Unit = {
        }

        override def batchReceived(buffer: RapidsShuffleHandle): Boolean = {
          received.addAndGet(batchSize)
          buffer.close()
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

      if (numIter > 0 ) {
        reqsInFlight = new LinkedBlockingQueue[Int](maxInFlight)
        clientProducer.execute(() => {
          while (true) {
            val doFetch = reqsInFlight.offer(1, 1, TimeUnit.SECONDS)
            if (doFetch) {
              client.doFetch(ShuffleBlockBatchId(1, 1L, 1, 1) :: Nil, fetchHandler)
            }
          }
        })
      }
      
      var ix = 0
      var continue = true
      if (numIter > 0) {
        while (continue) {
          Thread.sleep(1000L)
          val sofar = received.getAndSet(0L)
          logInfo(s"$ix: received ${sofar / 1024 / 1024} MB/s, inflight: ${reqsInFlight.size()}")
          ix += 1
          if (numIter != null && numIter > 0 && ix > numIter) {
            logInfo("done!")
            continue = false
          }
        }
      }
    } else {
      Thread.sleep(5000L)
      var ix = 0
      var continue = numIter > 0
      while (continue) {
        Thread.sleep(1000L)
        if (receivedFirst) {
          ix += 1
          if (numIter != null && numIter > 0 && ix > numIter) {
            logInfo("done!")
            continue = false
          }
        }
      }
    }
  }
}