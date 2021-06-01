/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.shuffle.ucx

import java.nio.ByteBuffer
import java.util.concurrent._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.nvidia.spark.rapids.{GpuDeviceManager, HashedPriorityQueue, RapidsConf}
import com.nvidia.spark.rapids.shuffle._
import com.nvidia.spark.rapids.shuffle.{BounceBufferManager, BufferReceiveState, ClientConnection, PendingTransferRequest, RapidsShuffleClient, RapidsShuffleRequestHandler, RapidsShuffleServer, RapidsShuffleTransport, RefCountedDirectByteBuffer}

import org.apache.spark.internal.Logging
import org.apache.spark.storage.BlockManagerId

/**
 * UCXShuffleTransport is the UCX implementation for the `RapidsShuffleTransport`. It provides
 * a way to create a `RapidsShuffleServer` and one `RapidsShuffleClient` per peer, that are
 * able to send/receive via UCX.
 *
 * Additionally, this class maintains pools of memory used to limit the cost of memory
 * pinning and registration (bounce buffers), a metadata message pool for small flatbuffers used
 * to describe shuffled data, and implements a simple throttle mechanism to keep GPU memory
 * usage at bay by way of configuration settings.
 *
 * @param shuffleServerId `BlockManagerId` for this executor
 * @param rapidsConf      plugin configuration
 */
class UCXShuffleTransport(shuffleServerId: BlockManagerId, rapidsConf: RapidsConf)
  extends RapidsShuffleTransport
    with Logging {

  private[this] var inflightSize = 0L
  private[this] val inflightLimit = rapidsConf.shuffleTransportMaxReceiveInflightBytes
  private[this] val inflightMonitor = new Object

  private[this] val shuffleMetadataPool = new DirectByteBufferPool(
    rapidsConf.shuffleMaxMetadataSize)

  private[this] val bounceBufferSize = rapidsConf.shuffleUcxBounceBuffersSize
  private[this] val deviceNumBuffers = rapidsConf.shuffleUcxDeviceBounceBuffersCount
  private[this] val hostNumBuffers = rapidsConf.shuffleUcxHostBounceBuffersCount

  private[this] var deviceSendBuffMgr: BounceBufferManager[DeviceMemoryBuffer] = null
  private[this] var hostSendBuffMgr: BounceBufferManager[HostMemoryBuffer] = null
  private[this] var deviceReceiveBuffMgr: BounceBufferManager[DeviceMemoryBuffer] = null

  private[this] val clients = new ConcurrentHashMap[Long, RapidsShuffleClient]()

  val requestTracker = new RequestTracker()

  private[this] lazy val ucx = {
    logWarning("UCX Shuffle Transport Enabled")
    val ucxImpl = new UCX(this, shuffleServerId, rapidsConf)
    ucxImpl.init()

    initBounceBufferPools(bounceBufferSize,
      deviceNumBuffers, hostNumBuffers)

    // Perform transport (potentially IB) registration early
    // NOTE: on error we log and close things, which should fail other parts of the job in a bad
    // way in reality we should take a stab at lowering the requirement, and registering a smaller
    // buffer.
    val mgrs = Seq(deviceSendBuffMgr, deviceReceiveBuffMgr, hostSendBuffMgr)
    ucxImpl.register(mgrs.map(_.getRootBuffer()), ex => {
      if (ex.isDefined) {
        logError(s"Error registering bounce buffers", ex.get)
        ucxImpl.close()
      }
    })

    exec.execute(() => {
      while (requestTracker.inflightStarted) {
        try {
          var numBuffersAvailable = deviceReceiveBuffMgr.numFree()
          while (numBuffersAvailable == 0) {
            deviceReceiveBuffMgr.synchronized {
              numBuffersAvailable = deviceReceiveBuffMgr.numFree()
              if (numBuffersAvailable == 0) {
                deviceReceiveBuffMgr.wait(100)
              }
            }
          } // have at least 1 bounce buffer

          // get requests from clients that would utilize `numBuffersAvailable`
          // this method will block until a new request is available
          val perClientReq  = requestTracker.getRequestsPerClient
          perClientReq.foreach { case (client, brs) =>
            client.issueBufferReceives(brs)
          }
        } catch {
          case t: Throwable =>
            logError("Error in the UCX throttle loop", t)
        }
      }
    })

    ucxImpl
  }

  private val altList = new HashedPriorityQueue[PendingTransferRequest](
    1000,
    (t: PendingTransferRequest, t1: PendingTransferRequest) => {
      if (t.getLength < t1.getLength) {
        -1;
      } else if (t.getLength > t1.getLength) {
        1;
      } else {
        0
      }
    })

  // access to this set must hold the `altList` lock
  val validHandlers =
    new mutable.HashSet[RapidsShuffleFetchHandler]()

  override def getDirectByteBuffer(size: Long): RefCountedDirectByteBuffer = {
    if (size > rapidsConf.shuffleMaxMetadataSize) {
      logWarning(s"Large metadata message size $size B, larger " +
        s"than ${rapidsConf.shuffleMaxMetadataSize} B. " +
        s"Consider setting ${RapidsConf.SHUFFLE_MAX_METADATA_SIZE.key} higher.")
      new RefCountedDirectByteBuffer(ByteBuffer.allocateDirect(size.toInt), None)
    } else {
      shuffleMetadataPool.getBuffer(size)
    }
  }

  /**
   * Initialize the bounce buffer pools that are to be used to send and receive data against UCX
   *
   * We have 2 pools for the send side, since buffers may come from spilled memory (host),
   * or device memory.
   *
   * We have 1 pool for the receive side, since all receives are targeted for the GPU.
   *
   * The size of buffers is the same for all pools, since send/receive sizes need to match. The
   * count can be set independently.
   *
   * @param bounceBufferSize the size for a single bounce buffer
   * @param deviceNumBuffers number of buffers to allocate for the device
   * @param hostNumBuffers   number of buffers to allocate for the host
   */
  def initBounceBufferPools(
      bounceBufferSize: Long,
      deviceNumBuffers: Int,
      hostNumBuffers: Int): Unit = {

    deviceSendBuffMgr =
      new BounceBufferManager[DeviceMemoryBuffer](
        "device-send",
        bounceBufferSize,
        deviceNumBuffers,
        (size: Long) => DeviceMemoryBuffer.allocate(size))

    deviceReceiveBuffMgr =
      new BounceBufferManager[DeviceMemoryBuffer](
        "device-receive",
        bounceBufferSize,
        deviceNumBuffers,
        (size: Long) => DeviceMemoryBuffer.allocate(size))

    hostSendBuffMgr =
      new BounceBufferManager[HostMemoryBuffer](
        "host-send",
        bounceBufferSize,
        hostNumBuffers,
        (size: Long) => HostMemoryBuffer.allocate(size))
  }

  def freeBounceBufferPools(): Unit = {
    Seq(deviceSendBuffMgr, deviceReceiveBuffMgr, hostSendBuffMgr).foreach(_.close())
  }

  private def getNumBounceBuffers(remaining: Long, totalRequired: Int): Int = {
    val numBuffers = (remaining + bounceBufferSize - 1) / bounceBufferSize
    Math.min(numBuffers, totalRequired).toInt
  }

  override def tryGetSendBounceBuffers(
      remaining: Long,
      totalRequired: Int): Seq[SendBounceBuffers] = {
    val numBuffs = getNumBounceBuffers(remaining, totalRequired)
    val deviceBuffer = tryAcquireBounceBuffers(deviceSendBuffMgr, numBuffs)
    if (deviceBuffer.nonEmpty) {
      val hostBuffer = tryAcquireBounceBuffers(hostSendBuffMgr, numBuffs)
      if (hostBuffer.nonEmpty) {
        deviceBuffer.zip(hostBuffer).map { case (d, h) =>
          SendBounceBuffers(d, Some(h))
        }
      } else {
        deviceBuffer.map(d => SendBounceBuffers(d, None))
      }
    } else {
      Seq.empty
    }
  }

  override def tryGetReceiveBounceBuffers(
      remaining: Long, totalRequired: Int): Seq[BounceBuffer] = {
    val numBuffs = getNumBounceBuffers(remaining, totalRequired)
    tryAcquireBounceBuffers(deviceReceiveBuffMgr, numBuffs)
  }

  private def tryAcquireBounceBuffers[T <: MemoryBuffer](
      bounceBuffMgr: BounceBufferManager[T],
      numBuffs: Integer): Seq[BounceBuffer] = {
    // if the # of buffers requested is more than what the pool has, we would deadlock
    // this ensures we only get as many buffers as the pool could possibly give us.
    val possibleNumBuffers = Math.min(bounceBuffMgr.numBuffers, numBuffs)
    val bounceBuffers =
      bounceBuffMgr.acquireBuffersNonBlocking(possibleNumBuffers)
    logTrace(s"Got ${bounceBuffers.size} bounce buffers from pool " +
      s"out of ${numBuffs} requested.")
    bounceBuffers
  }

  override def connect(peerBlockManagerId: BlockManagerId, okToFail: Boolean): ClientConnection = {
    val topo = peerBlockManagerId.topologyInfo
    val connection: ClientConnection = if (topo.isDefined) {
      val topoParts = topo.get.split("=")
      if (topoParts.size == 2 &&
          topoParts(0).equalsIgnoreCase(RapidsShuffleTransport.BLOCK_MANAGER_ID_TOPO_PREFIX)) {
        val peerExecutorId = peerBlockManagerId.executorId.toInt
        ucx.getConnection(peerExecutorId, peerBlockManagerId.host, topoParts(1).toInt, okToFail)
      } else {
        // in the future this may create connections in other transports
        throw new IllegalStateException(s"Invalid block manager id for the rapids " +
          s"shuffle $peerBlockManagerId")
      }
    } else {
      throw new IllegalStateException(s"Invalid block manager id for the rapids " +
        s"shuffle $peerBlockManagerId")
    }

    connection
  }

  class CallerRunsAndLogs extends ThreadPoolExecutor.CallerRunsPolicy {
    override def rejectedExecution(
        runnable: Runnable,
        threadPoolExecutor: ThreadPoolExecutor): Unit = {

      logWarning(s"Rejected execution for ${threadPoolExecutor}, running in caller's thread.")
      super.rejectedExecution(runnable, threadPoolExecutor)
    }
  }

  // NOTE: this pool, as is, will add a new thread per task. This will likely change.
  private[this] val clientExecutor = new ThreadPoolExecutor(1,
    rapidsConf.shuffleMaxClientThreads,
    rapidsConf.shuffleClientThreadKeepAliveTime,
    TimeUnit.SECONDS,
    new ArrayBlockingQueue[Runnable](1),
    GpuDeviceManager.wrapThreadFactory(
      new ThreadFactoryBuilder()
        .setNameFormat("shuffle-transport-client-exec-%d")
        .setDaemon(true)
        .build),
    // if we can't hand off because we are too busy, block the caller (in UCX's case,
    // the progress thread)
    new CallerRunsAndLogs())

  // This executor handles any task that would block (e.g. wait for spill synchronously due to OOM)
  private[this] val clientCopyExecutor = Executors.newSingleThreadExecutor(
    GpuDeviceManager.wrapThreadFactory(new ThreadFactoryBuilder()
      .setNameFormat("shuffle-client-copy-thread-%d")
      .setDaemon(true)
      .build))

  override def makeClient(localExecutorId: Long,
                 blockManagerId: BlockManagerId): RapidsShuffleClient = {
    val peerExecutorId = blockManagerId.executorId.toLong
    val clientConnection = connect(blockManagerId, false)
    clients.computeIfAbsent(peerExecutorId, _ => {
      new RapidsShuffleClient(
        localExecutorId,
        clientConnection,
        this,
        clientExecutor,
        clientCopyExecutor)
    })
  }

  // NOTE: this is a single thread for the executor, nothing prevents us from having a pool here.
  // This will likely change.
  private[this] val serverExecutor = Executors.newSingleThreadExecutor(
    GpuDeviceManager.wrapThreadFactory(new ThreadFactoryBuilder()
      .setNameFormat(s"shuffle-server-conn-thread-${shuffleServerId.executorId}-%d")
      .setDaemon(true)
      .build))

  // This executor handles any task that would block (e.g. wait for spill synchronously due to OOM)
  private[this] val serverCopyExecutor = Executors.newSingleThreadExecutor(
    GpuDeviceManager.wrapThreadFactory(new ThreadFactoryBuilder()
      .setNameFormat(s"shuffle-server-copy-thread-%d")
      .setDaemon(true)
      .build))

  // This is used to queue up on the server all the [[BufferSendState]] as the server waits for
  // bounce buffers to become available (it is the equivalent of the transport's throttle, minus
  // the inflight limit)
  private[this] val bssExecutor = Executors.newSingleThreadExecutor(
    GpuDeviceManager.wrapThreadFactory(new ThreadFactoryBuilder()
      .setNameFormat(s"shuffle-server-bss-thread-%d")
      .setDaemon(true)
      .build))

  /**
   * Construct a server instance
   *
   * @param requestHandler used to get metadata info, and acquire tables used in the shuffle.
   * @return the server instance
   */
  override def makeServer(requestHandler: RapidsShuffleRequestHandler): RapidsShuffleServer = {
    new RapidsShuffleServer(
      this,
      ucx.getServerConnection,
      shuffleServerId,
      requestHandler,
      serverExecutor,
      serverCopyExecutor,
      bssExecutor,
      rapidsConf)
  }

  /**
   * Updates the inflightSize by adding the `neededAmount`
   * @param neededAmount amount of bytes needed.
   * @note This function is called only after a successful call to `wouldFitInFlightLimit`. It also
   *       calls `wouldFitInFlightLimit` as a sanity check.
   */
  private def markBytesInFlight(neededAmount: Long): Unit = inflightMonitor.synchronized {
    require(wouldFitInFlightLimit(neededAmount),
      s"Inflight limit can't allow this size $neededAmount of request")
    inflightSize = inflightSize + neededAmount
  }

  /**
   * Returns true if the neededAmount fits within the throttle, or if the throttle is at 0.
   * The "at 0" case helps in the case where we have buffer sizes that are greater than
   * inflightLimit
   * @param neededAmount amount of bytes needed
   * @return true if `neededAmount` would be allowed in the throttle
   */
  private def wouldFitInFlightLimit(neededAmount: Long): Boolean = inflightMonitor.synchronized {
    inflightSize + neededAmount <= inflightLimit || inflightSize == 0
  }

  /**
   * Decreases the inflight size.
   * @note We are holding onto the inflightMonitor lock here, which we use to update the limit
   *       in all cases.
   * @param bytesCompleted amount of bytes handled
   */
  override def doneBytesInFlight(bytesCompleted: Long): Unit = inflightMonitor.synchronized {
    inflightSize = inflightSize - bytesCompleted
    logDebug(s"Done with ${bytesCompleted} bytes inflight, " +
      s"new inflightSize is ${inflightSize}")
    inflightMonitor.notifyAll()
  }

  class RequestTracker extends AutoCloseable {
    var inflightStarted = true
    override def close(): Unit = {
      synchronized {
        inflightStarted = false
        notifyAll()
      }
    }

    class ClientRequests(val client: RapidsShuffleClient) {

      def remove(handler: RapidsShuffleFetchHandler): Int = {
        val pre = altList.size
        altList.removeIf(_.handler == handler)
        pre - altList.size
      }

      var lastTouched = 0L
      def getLastTouched: Long = lastTouched
      private val altList = new HashedPriorityQueue[PendingTransferRequest](
        1000,
        (t: PendingTransferRequest, t1: PendingTransferRequest) => {
          if (t.getLength < t1.getLength) {
            -1;
          } else if (t.getLength > t1.getLength) {
            1;
          } else {
            0
          }
        })

      def addAll(pending: Seq[PendingTransferRequest]): Unit = {
        import collection.JavaConverters._
        altList.addAll(pending.asJava)
      }

      def popRequest(): PendingTransferRequest = {
        require(altList.size > 0)
        lastTouched = System.currentTimeMillis()
        altList.poll()
      }

      def pendingCount = altList.size()
    }

    private val perClientRequests = new HashedPriorityQueue[ClientRequests](
      (r: ClientRequests, r1: ClientRequests) => {
        if (r.getLastTouched < r1.getLastTouched) {
          -1;
        } else if (r.getLastTouched > r1.getLastTouched) {
          1;
        } else {
          0
        }
      })

    private val handlerToRequests =
      new mutable.HashMap[RapidsShuffleFetchHandler, mutable.Set[RapidsShuffleClient]]()
    private val clientToRequests = new mutable.HashMap[RapidsShuffleClient, ClientRequests]()
    var pendingCount = 0

    def add(handler: RapidsShuffleFetchHandler, pending: Seq[PendingTransferRequest]): Unit =
      synchronized {
        val cr = clientToRequests.get(pending.head.client)
        var clientRequests: ClientRequests = null
        if (cr.isEmpty) {
          // peer -> ClientRequests
          clientRequests = new ClientRequests(pending.head.client)
          clientToRequests.put(pending.head.client, clientRequests)
          perClientRequests.add(clientRequests)
        } else {
          clientRequests = cr.get
        }
        // handler -> Clients
        var handlerToClientSet: mutable.Set[RapidsShuffleClient] = null
        if (!handlerToRequests.contains(handler)) {
          handlerToClientSet = new mutable.HashSet[RapidsShuffleClient]()
          handlerToRequests.put(handler, handlerToClientSet)
        }
        handlerToClientSet.add(pending.head.client)
        clientRequests.addAll(pending)
        pendingCount += pending.size
        notifyAll()
      }

    def remove(handler: RapidsShuffleFetchHandler): Unit = synchronized {
      // remove any requests that match the handler, as the handler
      // is done
      val c = handlerToRequests.remove(handler)
      if (c.isDefined) {
        val clients = c.get
        clients.foreach(c => {
          val cr = clientToRequests.get(c)
          pendingCount -= cr.get.remove(handler)

          // if no more requests for this client
          if (cr.get.pendingCount == 0) {
            perClientRequests.remove(cr.get)
            clientToRequests.remove(cr.get.client)
          }
        })
      }
    }

    def getRequestsPerClient: Seq[(RapidsShuffleClient, BufferReceiveState)] =
      synchronized {
        // wait until we have pending requests (for any client)
        while (inflightStarted && pendingCount == 0) {
          wait(100)
        }

        val startNumFree = deviceReceiveBuffMgr.numFree()
        var totalBounceBufferSize = bounceBufferSize * startNumFree
        if (inflightStarted) {
          // NOTE: this must be more than 1, since it is checked prior
          // to calling `getRequestsPerClient`

          val brss = new ArrayBuffer[(RapidsShuffleClient, BufferReceiveState)]()
          val startPending = pendingCount
          val startClients = perClientRequests.size()

          // a buffer can be used for a single client at a time
          var haveBounceBuffers = true
          while (haveBounceBuffers && pendingCount > 0) {
            val reqsAndBbs = new ArrayBuffer[(BounceBuffer, ArrayBuffer[PendingTransferRequest])]()
            var index = 0
            // get `ClientRequests` in priority order (last touched)
            // this means a client that hasn't been serviced will likely
            var req = perClientRequests.peek()

            // fill 1 bounce buffer length
            while (req.pendingCount > 0) {
              var runningSize = 0L
              val bb = tryGetReceiveBounceBuffers(1, 1)
              if (bb.nonEmpty) {
                reqsAndBbs.append((bb.head, new ArrayBuffer[PendingTransferRequest]()))
              }
              val requestsToHandle = new ArrayBuffer[PendingTransferRequest]()
              while (req.pendingCount > 0 && runningSize < bounceBufferSize) {
                val popped = req.popRequest()
                perClientRequests.priorityUpdated(req)
                requestsToHandle.append(popped)
                pendingCount -= 1
                runningSize += popped.getLength
              }
              reqsAndBbs(index % reqsAndBbs.size)._2.appendAll(requestsToHandle)
              index = index + 1
              haveBounceBuffers = deviceReceiveBuffMgr.numFree() > 0
            }

            reqsAndBbs.foreach { case (bounceBuffer, reqs) =>
              if (reqs.nonEmpty) {
                brss.append((reqs.head.client,
                  new BufferReceiveState(bounceBuffer, reqs)))
              }
            }

            if (req.pendingCount == 0) { // no need to track it anymore
              perClientRequests.remove(req)
              clientToRequests.remove(req.client)
            }
          }
          logInfo(s"Created ${brss.size} BufferReceiveStates for a total of " +
            s"${brss.map(_._2.getRequests.map(_.getLength).sum).mkString("[",", ","]")} " +
            s"bytes each. Started with ${startNumFree} bounce buffers and ${startClients} " +
            s"waiting with ${startPending} pending. Current pending is ${pendingCount}")
          brss
        } else {
          Seq.empty
        }
      }
    }

  private[this] val exec = Executors.newSingleThreadExecutor(
    GpuDeviceManager.wrapThreadFactory(
      new ThreadFactoryBuilder()
        .setNameFormat(s"shuffle-transport-throttle-monitor")
        .setDaemon(true)
        .build))

  // helper class to hold transfer requests that have a bounce buffer
  // and should be ready to be handled by a `BufferReceiveState`
  class PerClientReadyRequests(val bounceBuffer: BounceBuffer) {
    val transferRequests = new ArrayBuffer[PendingTransferRequest]()
    var runningSize = 0L
    def addRequest(req: PendingTransferRequest): Unit = {
      transferRequests.append(req)
      runningSize += req.getLength
    }
  }

  override def queuePending(handler: RapidsShuffleFetchHandler,
                            reqs: Seq[PendingTransferRequest]): Unit =
    requestTracker.add(handler, reqs)

  override def cancelPending(handler: RapidsShuffleFetchHandler): Unit = {
    requestTracker.remove(handler)
  }

  override def close(): Unit = {
    logInfo("UCX transport closing")
    exec.shutdown()
    bssExecutor.shutdown()
    clientExecutor.shutdown()
    serverExecutor.shutdown()
    requestTracker.close()

    if (!exec.awaitTermination(500, TimeUnit.MILLISECONDS)) {
      logError("UCX Shuffle Transport throttle failed to terminate correctly")
    }
    if (!clientExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
      logError("UCX Shuffle Client failed to terminate correctly")
    }
    if (!serverExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
      logError("UCX Shuffle Server main executor failed to terminate correctly")
    }
    if (!bssExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
      logError("UCX Shuffle Server BufferSendState executor failed to terminate correctly")
    }

    ucx.close()
    freeBounceBufferPools()
  }
}
