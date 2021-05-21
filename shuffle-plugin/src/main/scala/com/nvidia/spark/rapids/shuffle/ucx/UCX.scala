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

import java.net._
import java.nio.ByteBuffer
import java.util.concurrent.{ConcurrentHashMap, ConcurrentLinkedQueue, Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import ai.rapids.cudf.{DeviceMemoryBuffer, MemoryBuffer, NvtxColor, NvtxRange}
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.nvidia.spark.rapids.{Arm, GpuDeviceManager, RapidsConf}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.shuffle.{ClientConnection, MemoryRegistrationCallback, MessageType, MetadataTransportBuffer, TransportBuffer, TransportUtils}
import org.openucx.jucx._
import org.openucx.jucx.ucp._
import org.openucx.jucx.ucs.UcsConstants

import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.storage.RapidsStorageUtils
import org.apache.spark.storage.BlockManagerId

case class Rkeys(rkeys: Seq[ByteBuffer])

/**
 * A simple wrapper for an Active Message Id and a header. This pair
 * is used together when dealing with Active Messages, with `activeMessageId`
 * being a fire-and-forget registration with UCX, and `header` being a dynamic long
 * we continue to update (it contains the local executor id, and the transaction id).
 *
 * This allows us to send a request (with a header that the response handler knows about),
 * and for the request handler to echo back that header when it's done.
 */
case class UCXActiveMessage(activeMessageId: Int, header: Long, forceRndv: Boolean) {
  override def toString: String =
    UCX.formatAmIdAndHeader(activeMessageId, header)
}

/**
 * The UCX class wraps JUCX classes and handles all communication with UCX from other
 * parts of the shuffle code. It manages a `UcpContext` and `UcpWorker`, for the
 * local executor, and maintain a set of `UcpEndpoint` for peers.
 *
 * This class uses an extra TCP management connection to perform a handshake with remote peers,
 * this port should be distributed to peers by other means (e.g. via the `BlockManagerId`)
 *
 * @param transport transport instance for UCX
 * @param executor blockManagerId of the local executorId
 * @param rapidsConf rapids configuration
 */
class UCX(transport: UCXShuffleTransport, executor: BlockManagerId, rapidsConf: RapidsConf)
    extends AutoCloseable with Logging with Arm {
  private[this] val context = {
    val contextParams = new UcpParams()
      .requestTagFeature()
      .requestAmFeature()
    if (rapidsConf.shuffleUcxUseWakeup) {
      contextParams.requestWakeupFeature()
    }
    new UcpContext(contextParams)
  }

  logInfo(s"UCX context created")

  def getExecutorId: Int = executor.executorId.toInt

  // this object implements the transport-friendly interface for UCX
  private[this] val serverConnection = new UCXServerConnection(this, transport)

  // monotonically increasing counter that holds the txId (for debug purposes, at this stage)
  private[this] val txId = new AtomicLong(0L)

  private var worker: UcpWorker = _
  private var listener: Option[UcpListener] = None

  @volatile private var initialized = false

  // this is a monotonically increasing id, used for buffers
  private val uniqueIds = new AtomicLong(0)

  // event loop, used to call [[UcpWorker.progress]], and perform all UCX work
  private val progressThread = Executors.newFixedThreadPool(1,
    GpuDeviceManager.wrapThreadFactory(
      new ThreadFactoryBuilder()
        .setNameFormat("progress-thread-%d")
        .setDaemon(true)
        .build))

  // The pending queues are used to enqueue [[PendingReceive]] or [[PendingSend]], from executor
  // task threads and [[progressThread]] will hand them to the UcpWorker thread.
  private val workerTasks = new ConcurrentLinkedQueue[() => Unit]()

  // holds memory registered against UCX that should be de-register on exit (used for bounce
  // buffers)
  // NOTE: callers should hold the `registeredMemory` lock before modifying this array
  val registeredMemory = new ArrayBuffer[UcpMemory]

  // when this flag is set to true, an async call to `register` hasn't completed in
  // the worker thread. We need this to complete prior to getting the `rkeys`.
  private var pendingRegistration = false

  // There will be 1 entry in this map per UCX-registered Active Message. Presently
  // that means: 2 request active messages (Metadata and Transfer Request), and 2
  // response active messages (Metadata and Transfer response).
  private val amRegistrations = new ConcurrentHashMap[Int, ActiveMessageRegistration]()

  // Keeps endpoint-related state, and handles UCP endpoint failure.
  private val endpointManager = new UcpEndpointManager

  /**
   * Initializes the UCX context and local worker and starts up the worker progress thread.
   * UCX worker/endpoint relationship.
   */
  def init(): Unit = {
    synchronized {
      if (initialized) {
        throw new IllegalStateException("UCX already initialized")
      }

      var workerParams = new UcpWorkerParams()

      if (rapidsConf.shuffleUcxUseWakeup) {
        workerParams = workerParams
          .requestWakeupTagSend()
          .requestWakeupTagRecv()
      }

      worker = context.newWorker(workerParams)
      logInfo(s"UCX Worker created")
      initialized = true
    }

    progressThread.execute(() => {
      // utility function to make all the progress possible in each iteration
      // this could change in the future to 1 progress call per loop, or be used
      // entirely differently once polling is figured out
      def drainWorker(): Unit = {
        withResource(new NvtxRange("UCX Draining Worker", NvtxColor.RED)) { _ =>
          while (worker.progress() > 0) {}
        }
      }

      while(initialized) {
        try {
          worker.progress()
          // else worker.progress returned 0
          if (rapidsConf.shuffleUcxUseWakeup) {
            drainWorker()
            if (workerTasks.isEmpty) {
              withResource(new NvtxRange("UCX Sleeping", NvtxColor.PURPLE)) { _ =>
                // Note that `waitForEvents` checks any events that have occurred, and will
                // return early in those cases. Therefore, `waitForEvents` is safe to be
                // called after `worker.signal()`, as it will wake up right away.
                worker.waitForEvents()
              }
            }
          }

          withResource(new NvtxRange("UCX Handling Tasks", NvtxColor.CYAN)) { _ =>
            while (!workerTasks.isEmpty) {
              val wt = workerTasks.poll()
              if (wt != null) {
                wt()
              }
            }
            worker.progress()
          }
        } catch {
          case t: Throwable =>
            logError("Exception caught in UCX progress thread. Continuing.", t)
        }
      }

      logWarning("Exiting UCX progress thread.")
      Seq(endpointManager, worker, context).safeClose()
    })
  }

  private def startControlRequestHandler(): Unit = {
    val controlAmId = UCXConnection.composeRequestAmId(MessageType.Control)
    registerRequestHandler(controlAmId, () => new UCXAmCallback {
      override def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit = {
        logError(s"Error with ${am}")
      }

      override def onMessageStarted(receiveAm: UcpRequest): Unit = {
      }

      override def onSuccess(am: UCXActiveMessage, buff: TransportBuffer): Unit = {
        withResource(buff) { _ =>
          val (peerExecId, peerRkeys) =
            UCXConnection.unpackHandshake(buff.getBuffer())

          val existingEp = endpointManager.getEndpointByExecutorId(peerExecId.toLong)

          logDebug(s"Success RECEIVING active message, am ${am} " +
            s"existing? ${existingEp} " +
            s"peer exec $peerExecId, peer keys ${peerRkeys}")

          onWorkerThreadAsync(() => {
            peerRkeys.foreach(existingEp.unpackRemoteKey)
          })

          val hs = UCXConnection.packHandshake(getExecutorId, localRkeys)

          // reply
          sendActiveMessage(
            peerExecId.toLong,
            UCXActiveMessage(
              UCXConnection.composeResponseAmId(MessageType.Control), am.header, false),
            hs,
            new UcxCallback {
              override def onError(ucsStatus: Int, errorMsg: String): Unit = {
                logError(s"ERROR REPLYING TO CONTROL $am ${ucsStatus} ${errorMsg}")
              }

              override def onSuccess(request: UcpRequest): Unit = {
                logInfo(s"SUCCESS REPLYING TO CONTROL $am with $hs")
                // hs here to keep the reference
              }
            })
        }
      }
      override def onCancel(am: UCXActiveMessage): Unit = {
      }

      override def onMessageReceived(size: Long, header: Long,
          finalizeCb: TransportBuffer => Unit): Unit = {
        finalizeCb(new MetadataTransportBuffer(transport.getDirectByteBuffer(size.toInt)))
      }
    })
  }

  /**
   * Starts a UCP Listener. UCX will start either a TCP port or use RDMACM to
   * allow peers to connect.
   * @param host
   * @return
   */
  def startListener(host: String): Int = {
    // after a connection happens via the listener, the control message is used
    // to hanshake with the peer
    startControlRequestHandler()

    // For now backward endpoints are not used, but need to create
    // an endpoint from connectionHandler in order to use ucpListener connections.
    // With AM this endpoints would be used as replyEp.
    val ucpListenerParams = new UcpListenerParams().setConnectionHandler(endpointManager)

    val maxRetries = SparkEnv.get.conf.getInt("spark.port.maxRetries", 16)
    val startPort = if (rapidsConf.shuffleUcxListenerStartPort != 0) {
      rapidsConf.shuffleUcxListenerStartPort
    } else {
      // TODO: remove this once ucx1.11 with random port selection would be released
      1024 + Random.nextInt(65535 - 1024)
    }
    var attempt = 0
    while (listener.isEmpty && attempt < maxRetries) {
      val sockAddress = new InetSocketAddress(executor.host, startPort + attempt)
      attempt += 1
      try {
        ucpListenerParams.setSockAddr(sockAddress)
        listener = Option(worker.newListener(ucpListenerParams))
      } catch {
        case _: UcxException =>
          logDebug(s"Failed to bind UcpListener on $sockAddress. " +
            s"Attempt ${attempt} out of $maxRetries.")
          listener = None
      }
    }
    if (listener.isEmpty) {
      throw new BindException(s"Couldn't start UcpListener " +
        s"on port range $startPort-${startPort + maxRetries}")
    }
    logInfo(s"Started UcpListener on ${listener.get.getAddress}")
    listener.get.getAddress.getPort
  }

  /**
   * This trait and next two implementations represent the mapping between an Active Message Id
   * and the callback that should be triggered when a message is received.
   *
   * There are three types of Active Messages we care about: requests, responses, and buffers.
   *
   * For requests:
   *   - `activeMessageId` for requests is the value of the `MessageType` enum, and it is
   *   set once when the `RapidsShuffleServer` is initialized, and no new request handlers
   *   are established.
   *
   *   - The Active Message header is handed to the request handler, via the transaction.
   *   The request handler needs to echo the header back for the response handler on the
   *   other side of the request.
   *
   *   - On a request, a callback is instantiated using `requestCallbackGen`, which creates
   *   a transaction each time. This is one way to handle several requests inbound to a server.
   *
   *   - useRndv: is set to false. We expect UCX to be able to use eager protocols or rndv
   *   at its leasure. `rapidsConf.shuffleUcxActiveMessagesForceRndv` can be used to force rndv.
   *
   * For responses:
   *   - `activeMessageId` for responses is the value of the `MessageType` enum with an extra
   *   bit flipped (see `UCXConnection.composeResponseAmId`). These are also set once, as requests
   *   are sent out.
   *
   *   - The Active Message header is used to pick the correct callback to call. In this case
   *   there could be several expected responses, for a single response `activeMessageId`, so the
   *   server echoes back our header so we can invoke the correct response callback.
   *
   *   - Each response received at the response activeMessageId, will be demuxed using the header:
   *   responseActiveMessageId1 -> [callbackForHeader1, callbackForHeader2, ..., callbackForHeaderN]
   *
   *   - useRndv: is set to false. We expect UCX to be able to use eager protocols or rndv
   *   at its leasure. `rapidsConf.shuffleUcxActiveMessagesForceRndv` can be used to force rndv.
   *
   * For buffers:
   *   - `activeMessageId` for buffers is the value of the `MessageType` enum (`MessageType.Buffer`)
   *
   *   - The Active Message header is used to pick the correct callback to call. Each header
   *   encodes the peer we expect to receive from and a monotonically inscreasing id per request.
   *
   *   - useRndv: is set to true. We prefer the zero-copy method since the buffer pointer
   *   provided to UCX could be on the GPU. An eager message would require us to issue an H2D in
   *   the handler.
   */
  trait ActiveMessageRegistration {
    val activeMessageId: Int
    def getCallback(header: Long): UCXAmCallback
    def useRndv: Boolean
  }

  class ReceiveActiveMessageRegistration(override val activeMessageId: Int, mask: Long)
      extends ActiveMessageRegistration {

    private[this] val handlers = new ConcurrentHashMap[Long, () => UCXAmCallback]()

    def addWildcardHeaderHandler(wildcardHdr: Long, cbGen: () => UCXAmCallback): Unit = {
      handlers.put(wildcardHdr, cbGen)
    }

    override def getCallback(header: Long): UCXAmCallback = {
      // callback is picked using a wildcard
      val cb = handlers.get(header & mask)
      if (cb == null) {
        throw new IllegalStateException(s"UCX Receive Active Message callback not found for " +
          s"${TransportUtils.toHex(header)} and mask ${TransportUtils.toHex(mask)}")
      }
      cb()
    }
    override def useRndv: Boolean = true
  }

  class RequestActiveMessageRegistration(override val activeMessageId: Int,
                                         requestCbGen: () => UCXAmCallback)
      extends ActiveMessageRegistration {

    def getCallback(header: Long): UCXAmCallback = requestCbGen()

    override def useRndv: Boolean = rapidsConf.shuffleUcxActiveMessagesForceRndv
  }

  class ResponseActiveMessageRegistration(override val activeMessageId: Int)
      extends ActiveMessageRegistration {
    private[this] val responseCallbacks = new ConcurrentHashMap[Long, UCXAmCallback]()

    def getCallback(header: Long): UCXAmCallback = {
      val cb = responseCallbacks.remove(header) // 1 callback per header
      require (cb != null,
        s"Failed to get a response Active Message callback for " +
          s"${UCX.formatAmIdAndHeader(activeMessageId, header)}")
      cb
    }

    def addResponseActiveMessageHandler(
        header: Long, responseCallback: UCXAmCallback): Unit = {
      val prior = responseCallbacks.putIfAbsent(header, responseCallback)
      require(prior == null,
        s"Invalid Active Message re-registration of response handler for " +
          s"${UCX.formatAmIdAndHeader(activeMessageId, header)}")
    }

    override def useRndv: Boolean = rapidsConf.shuffleUcxActiveMessagesForceRndv
  }

  /**
   * Register a response handler (clients will use this)
   *
   * @note This function will be called for each client, with the same `am.activeMessageId`
   * @param activeMessageId (up to 5 bits) used to register with UCX an Active Message
   * @param header a long used to demux responses arriving at `activeMessageId`
   * @param responseCallback callback to handle a particular response
   */
  def registerResponseHandler(
      activeMessageId: Int, header: Long, responseCallback: UCXAmCallback): Unit = {
    logDebug(s"Register Active Message " +
      s"${UCX.formatAmIdAndHeader(activeMessageId, header)} response handler")

    amRegistrations.computeIfAbsent(activeMessageId,
      _ => {
        val reg = new ResponseActiveMessageRegistration(activeMessageId)
        registerActiveMessage(reg)
        reg
      }) match {
      case reg: ResponseActiveMessageRegistration =>
        reg.addResponseActiveMessageHandler(header, responseCallback)
      case other =>
        throw new IllegalStateException(
          s"Attempted to add a response Active Message handler to existing registration $other " +
            s"for ${UCX.formatAmIdAndHeader(activeMessageId, header)}")
    }
  }

  /**
   * Register a request handler (the server will use this)
   * @note This function will be called once for the server for an `activeMessageId`
   * @param activeMessageId (up to 5 bits) used to register with UCX an Active Message
   * @param requestCallbackGen a function that instantiates a callback to handle
   *                           a particular request
   */
  def registerRequestHandler(activeMessageId: Int,
      requestCallbackGen: () => UCXAmCallback): Unit = {
    logDebug(s"Register Active Message $TransportUtils.request handler")
    val reg = new RequestActiveMessageRegistration(activeMessageId, requestCallbackGen)
    val oldReg = amRegistrations.putIfAbsent(activeMessageId, reg)
    require(oldReg == null,
      s"Tried to re-register a request handler for $activeMessageId")
    registerActiveMessage(reg)
  }

  /**
   * Register a receive (one way) message, this is used by the clients.
   * @param activeMessageId (up to 5 bits) used to register with UCX an Active Message
   * @param hdrMask - a long with bits set for those bits that should be masked to find
   *                  the proper callback.
   * @param hdrWildcard - a long with an id that should be associated with a specific callback.
   * @param receiveCallbackGen - a function that instantiates a callback to handle
   *                             a particular receive.
   */
  def registerReceiveHandler(activeMessageId: Int, hdrMask: Long, hdrWildcard: Long,
      receiveCallbackGen: () => UCXAmCallback): Unit = {
    logDebug(s"Register Active Message ${TransportUtils.toHex(activeMessageId)} " +
      s"mask ${TransportUtils.toHex(hdrMask)} " +
      s"wild ${TransportUtils.toHex(hdrWildcard)} request handler")
    amRegistrations.computeIfAbsent(activeMessageId,
      _ => {
        val reg = new ReceiveActiveMessageRegistration(activeMessageId, hdrMask)
        registerActiveMessage(reg)
        reg
      }) match {
      case reg: ReceiveActiveMessageRegistration =>
        reg.addWildcardHeaderHandler(hdrWildcard, receiveCallbackGen)
      case other =>
        throw new IllegalStateException(
          s"Attempted to add a receive Active Message handler to existing registration $other " +
            s"for ${UCX.formatAmIdAndHeader(activeMessageId, hdrWildcard)}")
    }
  }

  private def registerActiveMessage(reg: ActiveMessageRegistration): Unit = {
    onWorkerThreadAsync(() => {
      worker.setAmRecvHandler(reg.activeMessageId,
        (headerAddr, headerSize, amData: UcpAmData, ep: UcpEndpoint) => {
          if (headerSize != 8) {
            // this is a coding error, so I am just blowing up. It should never happen.
            throw new IllegalStateException(
              s"Received message with wrong header size $headerSize")
          } else {
            val header = UcxUtils.getByteBufferView(headerAddr, headerSize).getLong()
            val am = UCXActiveMessage(reg.activeMessageId, header, reg.useRndv)

            withResource(new NvtxRange("AM Receive", NvtxColor.YELLOW)) { _ =>
              logDebug(s"Active Message received: $am")
              val cb = reg.getCallback(header)

              if (amData.isDataValid) {
                require(!reg.useRndv,
                  s"Handling an eager Active Message, but expected rndv for: " +
                    s"amId ${TransportUtils.toHex(reg.activeMessageId)}")
                val resp = UcxUtils.getByteBufferView(amData.getDataAddress, amData.getLength)

                // copy the data onto a buffer we own because it is going to be reused
                // in UCX
                cb.onMessageReceived(amData.getLength, header, {
                  case mtb: MetadataTransportBuffer =>
                    mtb.copy(resp)
                    cb.onSuccess(am, mtb)
                  case _ =>
                    cb.onError(am, 0,
                      "Received an eager message for non-metadata message")
                })

                // we return OK telling UCX `amData` is ok to be closed, along with the eagerly
                // received data
                UcsConstants.STATUS.UCS_OK
              } else {
                // RNDV case: we get a direct buffer and UCX will fill it with data at `receive`
                // callback
                cb.onMessageReceived(amData.getLength, header, (resp: TransportBuffer) => {
                  logDebug(s"Receiving Active Message ${am} using data address " +
                    s"${TransportUtils.toHex(resp.getAddress())}")

                  // we must call `receive` on the `amData` object within the progress thread
                  onWorkerThreadAsync(() => {
                    val receiveAm = amData.receive(resp.getAddress(),
                      new UcxCallback {
                        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
                          withResource(resp) { _ =>
                            withResource(amData) { _ =>
                              if (ucsStatus == UCX.UCS_ERR_CANCELED) {
                                logWarning(
                                  s"Cancelled Active Message " +
                                    s"${TransportUtils.toHex(reg.activeMessageId)}" +
                                    s" status=$ucsStatus, msg=$errorMsg")
                                cb.onCancel(am)
                              } else {
                                cb.onError(am, ucsStatus, errorMsg)
                              }
                            }
                          }
                        }

                        override def onSuccess(request: UcpRequest): Unit = {
                          withResource(new NvtxRange("AM Success", NvtxColor.ORANGE)) { _ =>
                            withResource(amData) { _ =>
                              logDebug(s"Success with Active Message ${am} using data address " +
                                s"${TransportUtils.toHex(resp.getAddress())}")
                              cb.onSuccess(am, resp)
                            }
                          }
                        }
                      })
                    cb.onMessageStarted(receiveAm)
                  })
                })
                UcsConstants.STATUS.UCS_INPROGRESS
              }
            }
          }
        })
    })
  }

  def sendActiveMessage(executorId: Long, am: UCXActiveMessage,
      data: MemoryBuffer, cb: UcxCallback): Unit = {
    onWorkerThreadAsync(() => {
      val ep = endpointManager.getEndpointByExecutorId(executorId)
      if (ep == null) {
        throw new IllegalStateException(
          s"Trying to send a message to an endpoint that doesn't exist ${executorId}")
      }
      sendActiveMessage(
        ep,
        am,
        data.getAddress,
        data.getLength,
        cb,
        isGpu = data.isInstanceOf[DeviceMemoryBuffer])
    })
  }

  def sendActiveMessage(executorId: Long, am: UCXActiveMessage,
      data: ByteBuffer, cb: UcxCallback): Unit = {
    onWorkerThreadAsync(() => {
      val ep = endpointManager.getEndpointByExecutorId(executorId)
      if (ep == null) {
        throw new IllegalStateException(
          s"Trying to send a message to an endpoint that doesn't exist ${executorId}")
      }
      sendActiveMessage(
        ep,
        am,
        TransportUtils.getAddress(data),
        data.remaining(),
        cb,
        isGpu = false)
    })
  }

  def sendActiveMessage(endpoint: UcpEndpoint, am: UCXActiveMessage,
                        data: ByteBuffer, cb: UcxCallback): Unit = {
    onWorkerThreadAsync(() => {
      sendActiveMessage(
        endpoint,
        am,
        TransportUtils.getAddress(data),
        data.remaining(),
        cb,
        isGpu = false)
    })
  }

  private def sendActiveMessage(ep: UcpEndpoint, am: UCXActiveMessage,
      dataAddress: Long, dataSize: Long,
      cb: UcxCallback, isGpu: Boolean): Unit = {
    val useRndv = am.forceRndv || rapidsConf.shuffleUcxActiveMessagesForceRndv
    logDebug(s"Sending $am msg of size $dataSize to peer ${ep} data addr: " +
      s"${TransportUtils.toHex(dataAddress)}. Is gpu? ${isGpu}")

    // This isn't coming from the pool right now because it would be a bit of a
    // waste to get a larger hard-partitioned buffer just for 8 bytes.
    // TODO: since we no longer have metadata limits, the pool can be managed using the
    //   address-space allocator, so we should obtain this direct buffer from that pool
    val header = ByteBuffer.allocateDirect(8)
    header.putLong(am.header)
    header.rewind()

    val flags = if (useRndv) {
      UcpConstants.UCP_AM_SEND_FLAG_RNDV
    } else {
      0L /* AUTO */
    }

    val memType = if (isGpu) {
      UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA
    } else {
      UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_HOST
    }

    withResource(new NvtxRange("AM Send", NvtxColor.GREEN)) { _ =>
      ep.sendAmNonBlocking(
        am.activeMessageId,
        TransportUtils.getAddress(header),
        8L,
        dataAddress,
        dataSize,
        flags,
        new UcxCallback {
          override def onSuccess(request: UcpRequest): Unit = {
            cb.onSuccess(request)
            RapidsStorageUtils.dispose(header)
          }

          override def onError(ucsStatus: Int, errorMsg: String): Unit = {
            cb.onError(ucsStatus, errorMsg)
            RapidsStorageUtils.dispose(header)
          }
        }, memType)
    }
  }

  def getServerConnection: UCXServerConnection = serverConnection

  def cancel(request: UcpRequest): Unit = {
    onWorkerThreadAsync(() => {
      try {
        worker.cancelRequest(request)
      } catch {
        case e: Throwable =>
          logError("Error while cancelling UCX request: ", e)
      }
    })
  }

  def assignUniqueId(): Long = uniqueIds.incrementAndGet()

  /**
   * Connect to a remote UCX management port.
   *
   * @param peerMgmtHost management TCP host
   * @param peerMgmtPort management TCP port
   * @return Connection object representing this connection
   */
  def getConnection(peerExecutorId: Long,
      peerMgmtHost: String,
      peerMgmtPort: Int): ClientConnection = {
    val getConnectionStartTime = System.currentTimeMillis()
    val result = endpointManager.getConnection(peerExecutorId, peerMgmtHost, peerMgmtPort)
    logDebug(s"Got connection for executor ${peerExecutorId} in " +
      s"${System.currentTimeMillis() - getConnectionStartTime} ms")
    result
  }

  def onWorkerThreadAsync(task: () => Unit): Unit = {
    workerTasks.add(task)
    if (rapidsConf.shuffleUcxUseWakeup) {
      withResource(new NvtxRange("UCX Signal", NvtxColor.RED)) { _ =>
        try {
          worker.signal()
        } catch {
          case npe: NullPointerException => logError(s"NPE while shutting down", npe)
        }
      }
    }
  }

  /**
   * Return rkeys (if we have registered memory)
   */
  private def localRkeys: Seq[ByteBuffer] = registeredMemory.synchronized {
    while (pendingRegistration) {
      registeredMemory.wait(100)
    }
    registeredMemory.map(_.getRemoteKeyBuffer)
  }

  /**
   * Register a set of `MemoryBuffers` against UCX.
   *
   * @param buffers to register
   * @param mmapCallback callback invoked when the memory map operation completes or fails
   */
  def register(buffers: Seq[MemoryBuffer], mmapCallback: MemoryRegistrationCallback): Unit =
    registeredMemory.synchronized {
      pendingRegistration = true

      onWorkerThreadAsync(() => {
        var error: Throwable = null
        registeredMemory.synchronized {
          try {
            buffers.foreach { buffer =>
              val mmapParam = new UcpMemMapParams()
                  .setAddress(buffer.getAddress)
                  .setLength(buffer.getLength)

              //note that this can throw, lets call back and let caller figure out how to handle
              try {
                val registered = context.memoryMap(mmapParam)
                registeredMemory += registered
              } catch {
                case t: Throwable =>
                  if (error == null) {
                    error = t
                  } else {
                    error.addSuppressed(t)
                  }
              }
            }
          } finally {
            mmapCallback(Option(error))
            pendingRegistration = false
            registeredMemory.notify()
          }
        }
      })
    }

  def getNextTransactionId: Long = txId.incrementAndGet()

  override def close(): Unit = {
    onWorkerThreadAsync(() => {
      amRegistrations.forEach { (activeMessageId, _) =>
        logDebug(s"Removing Active Message registration for " +
          s"${TransportUtils.toHex(activeMessageId)}")
        worker.removeAmRecvHandler(activeMessageId)
      }

      logInfo(s"De-registering UCX ${registeredMemory.size} memory buffers.")
      registeredMemory.synchronized {
        registeredMemory.foreach(_.deregister())
        registeredMemory.clear()
      }
      synchronized {
        initialized = false
        notifyAll()
        // exit the loop
      }
    })

    synchronized {
      while (initialized) {
        wait(100)
      }
    }

    progressThread.shutdown()
    if (!progressThread.awaitTermination(500, TimeUnit.MICROSECONDS)) {
      logError("UCX progress thread failed to terminate correctly")
    }
  }

  private val ucx = this

  class UcpEndpointManager
    extends UcpEndpointErrorHandler with UcpListenerConnectionHandler with AutoCloseable {

    // Multiple executor threads are going to try to call connect (from the
    // [[RapidsShuffleIterator]]) because they don't know what executors to connect to up until
    // shuffle fetch time.
    //
    // This makes sure that all executor threads get the same [[Connection]] object for a specific
    // management (host, port) key.
    private val connectionCache = new ConcurrentHashMap[Long, ClientConnection]()
    private val endpoints = new ConcurrentHashMap[Long, UcpEndpoint]()
    private val reverseLookupEndpoints = new ConcurrentHashMap[UcpEndpoint, Long]()

    def getConnection(peerExecutorId: Long, peerHost: String, peerPort: Int): ClientConnection = {
      connectionCache.computeIfAbsent(peerExecutorId, _ => {
        val connection = new UCXClientConnection(peerExecutorId, ucx, transport)
        startConnection(
          peerExecutorId, connection, peerHost, peerPort)
        connection
      })
    }

    // Common endpoint parameters.
    private def makeEndpointParameters: UcpEndpointParams = {
      val result = new UcpEndpointParams()
      if (rapidsConf.shuffleUcxUsePeerErrorHandler) {
        logDebug("Using peer error handling")
        result.setErrorHandler(endpointManager).setPeerErrorHandlingMode()
      }
      result
    }

    // UcpListenerConnectionHandler interface - called from progress thread
    override def onConnectionRequest(connectionRequest: UcpConnectionRequest): Unit = synchronized {
      val ep = worker.newEndpoint(makeEndpointParameters.setConnectionRequest(connectionRequest))
      logInfo(s"Got UcpListener request from ${connectionRequest.getClientAddress}" +
        s" and ep ${ep}")

      val handshakeMessage = UCXConnection.packHandshake(getExecutorId, localRkeys)

      val responseAm = UCXActiveMessage(
        UCXConnection.composeResponseAmId(MessageType.Control), ep.getNativeId, false)

      registerResponseHandler(
        responseAm.activeMessageId, responseAm.header, new UCXAmCallback {
          override def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit = {
            logError(s"ERROR WITH RESPONSE CONTROL ${ucsStatus} ${errorMsg}")
          }

          override def onMessageStarted(receiveAm: UcpRequest): Unit = {}

          override def onSuccess(am: UCXActiveMessage,
                                 buff: TransportBuffer): Unit = {

            withResource(buff) { _ =>
              val (peerExecId, peerRkeys) =
                UCXConnection.unpackHandshake(buff.getBuffer())

              logInfo(s"SUCCESS WITH RESPONSE CONTROL: " +
                s"from ${ep}: $peerExecId")

              if (!removePriorEndpointIfNeeded(peerExecId, ep)) {
                peerRkeys.foreach(ep.unpackRemoteKey)
              }
            }
          }

          override def onCancel(am: UCXActiveMessage): Unit = {}

          override def onMessageReceived(size: Long, header: Long,
              finalizeCb: TransportBuffer => Unit): Unit = {
            finalizeCb(new MetadataTransportBuffer(transport.getDirectByteBuffer(size)))
          }
        })

      val requestAm = UCXActiveMessage(
        UCXConnection.composeRequestAmId(MessageType.Control), ep.getNativeId, false)

      sendActiveMessage(ep, requestAm, handshakeMessage, new UcxCallback {
        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
          logError(s"Error sending handshake header ${ucsStatus} ${errorMsg}")
        }

        override def onSuccess(request: UcpRequest): Unit = {
          logInfo("Success sending handshake header!")
        }
      })
    }

    private def removePriorEndpointIfNeeded(executorId: Long, newEp: UcpEndpoint): Boolean =
      synchronized {
        val priorEp = getEndpointByExecutorId(executorId)
        if (priorEp != null && priorEp != newEp) {
          logError(s"ALREADY HAD AN ENDPOINT FOR ${executorId}, " +
            s"closing ${priorEp} in favor of ${newEp}")
          onWorkerThreadAsync(() => {
            try {
              logError(s"CLOSING IN PROGRESS THREAD ${priorEp}")
              priorEp.closeNonBlockingFlush()
            } catch {
              case e => logWarning(s"Exception closing endpoint $priorEp from UCX", e)
            }
            putEndpoint(executorId, newEp)
          })
          true
        } else {
          false
        }
      }

    private def putEndpoint(executorId: Long, endpoint: UcpEndpoint): Boolean = synchronized {
      endpoints.put(executorId, endpoint)
      reverseLookupEndpoints.put(endpoint, executorId)
      true
    }

    def getEndpointByExecutorId(executorId: Long): UcpEndpoint = synchronized {
      endpoints.get(executorId)
    }

    def getExecutorIdForEndpoint(endpoint: UcpEndpoint): Long = synchronized {
      reverseLookupEndpoints.get(endpoint)
    }

    // UcpEndpointErrorHandler interface - called from progress thread
    override def onError(ucpEndpoint: UcpEndpoint, errorCode: Int, errorString: String): Unit =
      synchronized {
        logError(s"UcpListener detected an error ${errorCode} ${errorString} " +
          s"${ucpEndpoint}")
        val executorId = reverseLookupEndpoints.get(ucpEndpoint)
        if (executorId != null) {
          logError(s"Error for executorId ${executorId}: ${errorString}")
          reverseLookupEndpoints.remove(ucpEndpoint)
          ucpEndpoint.close()
          val existingEp = endpoints.computeIfPresent(executorId, (_, ep) => {
            if (ep == ucpEndpoint) {
              logWarning(s"Removing endpoint $ep for $executorId")
              null
            } else {
              ep
            }
          })
          if (existingEp == null) {
            connectionCache.computeIfPresent(executorId, (_, conn) => {
              logWarning(s"Removed stale client connection for ${executorId}")
              conn.close()
              null
            })
          }
        } else {
          logWarning(s"Received error for unknown endpoint: ${errorCode} ${errorString}")
        }
      }

    override def close(): Unit = synchronized {
      endpoints.values().forEach(_.close())
      endpoints.clear()
      reverseLookupEndpoints.clear()
    }

    // client side
    private def startConnection(peerExecutorId: Long, connection: UCXClientConnection,
                        peerMgmtHost: String,
                        peerMgmtPort: Int) = synchronized {
      if (!endpoints.contains(peerExecutorId)) {
        logInfo(s"Connecting to $peerMgmtHost:$peerMgmtPort")
        onWorkerThreadAsync(() => {
          endpoints.computeIfAbsent(peerExecutorId, _ => {
            logWarning(s"CREATE AN ENDPOINT ON START CONNECTION for ${peerExecutorId}")
            val sockAddr = new InetSocketAddress(peerMgmtHost, peerMgmtPort)
            val epParams = makeEndpointParameters
              .setSocketAddress(sockAddr)
              .setNoLoopbackMode()
            val ep = worker.newEndpoint(epParams)
            reverseLookupEndpoints.put(ep, peerExecutorId)
            logInfo(s"Initiated an UCPListener connection to $peerMgmtHost, $peerMgmtPort, ${ep}")
            ep
          })
        })
      }
    }
  }
}

object UCX {
  // This is used to distinguish a cancelled request vs. other errors
  // as the callback is the same (onError)
  // from https://github.com/openucx/ucx/blob/master/src/ucs/type/status.h
  private val UCS_ERR_CANCELED = -16

  // We include a header with this size in our active messages
  private val ACTIVE_MESSAGE_HEADER_SIZE = 8L

  def formatAmIdAndHeader(activeMessageId: Int, header: Long) =
    s"[amId=${TransportUtils.toHex(activeMessageId)}, hdr=${TransportUtils.toHex(header)}]"
}
