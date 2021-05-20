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
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import ai.rapids.cudf.{MemoryBuffer, NvtxColor, NvtxRange}
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.nvidia.spark.rapids.{Arm, GpuDeviceManager, RapidsConf}
import com.nvidia.spark.rapids.shuffle.{AddressLengthTag, ClientConnection, MemoryRegistrationCallback, RefCountedDirectByteBuffer, RequestType, TransportUtils}
import org.openucx.jucx._
import org.openucx.jucx.ucp._
import org.openucx.jucx.ucs.UcsConstants

import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.storage.RapidsStorageUtils
import org.apache.spark.storage.BlockManagerId

case class WorkerAddress(address: ByteBuffer)

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
case class UCXActiveMessage(activeMessageId: Int, header: Long) {
  override def toString: String =
    UCX.formatAmIdAndHeader(activeMessageId, header)
}

/**
 * The UCX class wraps JUCX classes and handles all communication with UCX from other
 * parts of the shuffle code. It manages a `UcpContext` and `UcpWorker`, for the
 * local executor, and maintain a set of `UcpEndpoint` for peers.
 *
 * The current API supported from UCX is the tag based API. Tags are exposed in this class
 * via an `AddressLengthTag`.
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
  UCX.setInstance(this)

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
  private val endpoints = new ConcurrentHashMap[Long, UcpEndpoint]()
  private val backEndpoints = new ConcurrentHashMap[Long, UcpEndpoint]()
  @volatile private var initialized = false

  // a peer tag identifies an incoming connection uniquely
  private val peerTag = new AtomicLong(0) // peer tags

  // this is a monotonically increasing id for every response
  private val responseTag = new AtomicLong(0) // outgoing message tag

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

  // Multiple executor threads are going to try to call connect (from the
  // [[RapidsUCXShuffleIterator]]) because they don't know what executors to connect to up until
  // shuffle fetch time.
  //
  // This makes sure that all executor threads get the same [[Connection]] object for a specific
  // management (host, port) key.
  private val connectionCache = new ConcurrentHashMap[Long, ClientConnection]()

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

  // Error handler that would be invoked on endpoint failure.
  private val epErrorHandler = new UcpEndpointErrorHandler {
    override def onError(ucpEndpoint: UcpEndpoint, errorCode: Int, errorString: String): Unit = {
      endpoints.keySet().forEach((entry: Long) => {
        if (endpoints.get(entry) == ucpEndpoint) {
          logError(s"UcpListener detected an error ${errorCode} ${errorString} " +
            s"${ucpEndpoint}")
          logError(s"Error for executorId ${entry}: ${errorString}")
          val conn = connectionCache.remove(entry)
          logWarning(s"Removed stale client connection for ${entry}")
          conn.close()
        }
      })
      endpoints.values().removeIf(ep => ep == ucpEndpoint)
      ucpEndpoint.close()
    }
  }

  // Common endpoint parameters.
  private def getEpParams = {
    val result = new UcpEndpointParams()
    if (rapidsConf.shuffleUcxUsePeerErrorHandler) {
      logDebug("Using peer error handling")
      result.setErrorHandler(epErrorHandler).setPeerErrorHandlingMode()
    }
    result
  }

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
      doProgress()
    })
  }

  def doProgress(): Unit = {
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
            withResource(new NvtxRange("UCX Sleeping", NvtxColor.PURPLE)) { _ =>
              worker.waitForEvents()
            }
          }

          while (!workerTasks.isEmpty) {
            withResource(new NvtxRange("UCX Handling Tasks", NvtxColor.CYAN)) { _ =>
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

      logWarning("Exiting progress thread. Closing endpoints, worker and context.")
      endpoints.values().forEach(ep => ep.close())
      worker.close()
      context.close()
    })
  }

  def startListener(host: String): Int = {
    if (rapidsConf.shuffleUcxUseSockaddr) {

      registerRequestHandler(
          UCXConnection.composeRequestAmId(RequestType.Control), () => new UCXAmCallback {
        override def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit = {
          logError(s"Error with ${am}")
        }
        override def onMessageStarted(receiveAm: UcpRequest): Unit = {
        }
        override def onSuccess(
            am: UCXActiveMessage, buff: RefCountedDirectByteBuffer, ep: UcpEndpoint): Unit = {
          withResource(buff) { _ =>
            val (peerExecId, host, port, peerRkeys) =
              UCXConnection.unpackHandshake(buff.getBuffer())

            val existingEp = endpoints.get(peerExecId.toLong)

            logInfo(s"Success RECEIVING active message, am ${am} " +
            s"existing? ${endpoints.get(peerExecId.toLong)} " +
            s"peer exec $peerExecId, host $host, port $port, peer keys ${peerRkeys}")
            onWorkerThreadAsync(() => {
              peerRkeys.foreach(existingEp.unpackRemoteKey)
            })

            val hs = UCXConnection.packHandshake(
              getExecutorId, executor.host, listener.get.getAddress.getPort, localRkeys)

            // reply
            sendActiveMessage(
              existingEp,
              UCXActiveMessage(UCXConnection.composeResponseAmId(RequestType.Control), am.header),
              TransportUtils.getAddress(hs), hs.remaining(),
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
        override def onHostMessageReceived(size: Long): RefCountedDirectByteBuffer = {
          transport.getDirectByteBuffer(size)
        }
      })
      // For now backward endpoints are not used, but need to create
      // an endpoint from connectionHandler in order to use ucpListener connections.
      // With AM this endpoints would be used as replyEp.
      val ucpListenerParams = new UcpListenerParams().setConnectionHandler(
        (connectionRequest: UcpConnectionRequest) => {
          val ep = worker.newEndpoint(getEpParams.setConnectionRequest(connectionRequest))
          logInfo(s"Got UcpListener request from ${connectionRequest.getClientAddress}" +
            s" and ep ${ep}")

          val hdr = UCXConnection.packHandshake(
            getExecutorId, host, listener.get.getAddress.getPort, localRkeys)

          val responseAm = UCXActiveMessage(
            UCXConnection.composeResponseAmId(RequestType.Control), ep.getNativeId)

          registerResponseHandler(
            responseAm.activeMessageId, responseAm.header, new UCXAmCallback {
              override def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit = {
                logError(s"ERROR WITH RESPONSE CONTROL ${ucsStatus} ${errorMsg}")
              }
              override def onMessageStarted(receiveAm: UcpRequest): Unit = {}
              override def onSuccess(am: UCXActiveMessage,
                                     buff: RefCountedDirectByteBuffer,
                                     replyEp: UcpEndpoint): Unit = {

                withResource(buff) { _ =>
                  val (peerExecId, peerHost, peerPort, peerRkeys) =
                    UCXConnection.unpackHandshake(buff.getBuffer())

                  logInfo(s"SUCCESS WITH RESPONSE CONTROL: " +
                    s"from ${ep}: $peerExecId @ $peerHost:$peerPort")

                  //TODO: this seems fishy
                  endpoints.synchronized {
                    backEndpoints.put(peerExecId.toLong, ep)
                    val priorEp = endpoints.get(peerExecId.toLong)
                    if (priorEp != null && priorEp != ep) {
                      logError(s"ALREADY HAD AN ENDPOINT FOR ${peerExecId}, " +
                        s"closing ${priorEp} in favor of ${ep}")
                      onWorkerThreadAsync(() => {
                        logError(s"CLOSING IN PROGRESS THREAD ${priorEp}")
                        endpoints.synchronized {
                          priorEp.closeNonBlockingFlush()
                          endpoints.put(peerExecId.toLong, ep)
                        }
                      })
                    } else {
                      peerRkeys.foreach(ep.unpackRemoteKey)
                    }
                  }
                }
              }
              override def onCancel(am: UCXActiveMessage): Unit = {}
              override def onHostMessageReceived(size: Long): RefCountedDirectByteBuffer = {
                transport.getDirectByteBuffer(size)
              }
            })

          sendActiveMessage(ep,
            UCXActiveMessage(UCXConnection.composeRequestAmId(RequestType.Control), ep.getNativeId),
            TransportUtils.getAddress(hdr),
            hdr.remaining(), new UcxCallback {
              override def onError(ucsStatus: Int, errorMsg: String): Unit = {
                logError(s"Error sending handshake header ${ucsStatus} ${errorMsg}")
              }

              override def onSuccess(request: UcpRequest): Unit = {
                logInfo("Success sending handshake header!")
              }
            })
        })
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
    } else {
      throw new IllegalStateException("told to start listener, but not using ucx sockaddr")
    }
  }

  // LOW LEVEL API
  def send(endpointId: Long, alt: AddressLengthTag, cb: UCXTagCallback): Unit = {
    val ucxCb = new UcxCallback {
      override def onError(ucsStatus: Int, errorMsg: String): Unit = {
        if (ucsStatus == UCX.UCS_ERR_CANCELED) {
          logWarning(
            s"Cancelled: tag=${TransportUtils.toHex(alt.tag)}," +
              s" status=$ucsStatus, msg=$errorMsg")
          cb.onCancel(alt)
        } else {
          logError("error sending : " + ucsStatus + " " + errorMsg)
          cb.onError(alt, ucsStatus, errorMsg)
        }
      }

      override def onSuccess(request: UcpRequest): Unit = {
        cb.onSuccess(alt)
      }
    }

    onWorkerThreadAsync(() => {
      val ep = endpoints.get(endpointId)
      if (ep == null) {
        throw new IllegalStateException(s"I cant find endpoint $endpointId")
      }

      val request = ep.sendTaggedNonBlocking(alt.address, alt.length, alt.tag, ucxCb)
      cb.onMessageStarted(request)
    })
  }

  /**
   * This trait and next two implementations represent the mapping between an Active Message Id
   * and the callback that should be triggered when a message is received.
   *
   * There are two types of Active Messages we care about: requests and responses.
   *
   * For requests:
   *   - `activeMessageId` for requests is the value of the `RequestType` enum, and it is
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
   * For responses:
   *   - `activeMessageId` for responses is the value of the `RequestType` enum with an extra
   *   bit flipped (see `UCXConnection.composeResponseAmId`). These are also set once, as requests
   *   are sent out.
   *
   *   - The Active Message header is used to pick the correct callback to call. In this case
   *   there could be several expected responses, for a single response `activeMessageId`, so the
   *   server echoes back our header so we can invoke the correct response callback.
   *
   *   - Each response received at the response activeMessageId, will be demuxed using the header:
   *   responseActiveMessageId1 -> [callbackForHeader1, callbackForHeader2, ..., callbackForHeaderN]
   */
  trait ActiveMessageRegistration {
    val activeMessageId: Int
    def getCallback(header: Long): UCXAmCallback
  }

  class RequestActiveMessageRegistration(override val activeMessageId: Int,
                                         requestCbGen: () => UCXAmCallback)
      extends ActiveMessageRegistration {

    def getCallback(header: Long): UCXAmCallback = requestCbGen()
  }

  class ResponseActiveMessageRegistration(override val activeMessageId: Int)
      extends ActiveMessageRegistration {
    private val responseCallbacks = new ConcurrentHashMap[Long, UCXAmCallback]()

    def getCallback(header: Long): UCXAmCallback = {
      val cb = responseCallbacks.remove(header) // 1 callback per header
      require (cb != null,
        s"Failed to get a response Active Message callback for " +
          s"${UCX.formatAmIdAndHeader(activeMessageId, header)}")
      cb
    }

    def addResponseActiveMessageHandler(header: Long, responseCallback: UCXAmCallback): Unit = {
      val prior = responseCallbacks.putIfAbsent(header, responseCallback)
      require(prior == null,
        s"Invalid Active Message re-registration of response handler for " +
          s"${UCX.formatAmIdAndHeader(activeMessageId, header)}")
    }
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

  private def registerActiveMessage(reg: ActiveMessageRegistration): Unit = {
    onWorkerThreadAsync(() => {
      worker.setAmRecvHandler(reg.activeMessageId,
        (headerAddr, headerSize, amData: UcpAmData, ep: UcpEndpoint) => {
          if (headerSize != 8) {
            // this is a coding error, so I am just blowing up. It should never happen.
            throw new IllegalStateException(
              s"Received message with wrong header size $headerSize")
          } else {
            val header = UcxUtils.getByteBufferView(headerAddr, headerSize).getLong
            val am = UCXActiveMessage(reg.activeMessageId, header)

            logDebug(s"Active Message received: $am")

            val cb = reg.getCallback(header)

            if (amData.isDataValid) {
              require(notForcingAmRndv,
                s"Handling an eager Active Message, but we are using " +
                  s"'${rapidsConf.shuffleUcxActiveMessagesMode}' as our configured mode.")
              logDebug(s"Handling an EAGER active message receive ${amData}")
              val resp = UcxUtils.getByteBufferView(amData.getDataAddress, amData.getLength)

              // copy the data onto a buffer we own because it is going to be reused
              // in UCX
              val dbb = cb.onHostMessageReceived(amData.getLength)
              val bb = dbb.getBuffer()
              bb.put(resp)
              bb.rewind()
              cb.onSuccess(am, dbb, ep)

              // we return OK telling UCX `amData` is ok to be closed, along with the eagerly
              // received data
              UcsConstants.STATUS.UCS_OK
            } else {
              // RNDV case: we get a direct buffer and UCX will fill it with data at `receive`
              // callback
              val resp = cb.onHostMessageReceived(amData.getLength)

              val receiveAm = amData.receive(UcxUtils.getAddress(resp.getBuffer()),
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
                    withResource(amData) { _ =>
                      cb.onSuccess(am, resp, ep)
                    }
                  }
                })

              cb.onMessageStarted(receiveAm)
              UcsConstants.STATUS.UCS_INPROGRESS
            }
          }
        })
    })
  }

  // If we are not forcing RNDV (i.e. we are in auto or eager) other handling
  // can happen when we receive an Active Message message (it can contain
  // inline data that must be copied out in the callback).
  private lazy val notForcingAmRndv: Boolean = {
    !rapidsConf.shuffleUcxActiveMessagesMode
      .equalsIgnoreCase("rndv")
  }

  private lazy val activeMessageMode: Long = {
    rapidsConf.shuffleUcxActiveMessagesMode match {
      case "eager" =>
        UcpConstants.UCP_AM_SEND_FLAG_EAGER
      case "rndv" =>
        UcpConstants.UCP_AM_SEND_FLAG_RNDV
      case "auto" =>
        0L
      case _ =>
        throw new IllegalArgumentException(
          s"${rapidsConf.shuffleUcxActiveMessagesMode} is an invalid Active Message mode. " +
          s"Please ensure that ${RapidsConf.SHUFFLE_UCX_ACTIVE_MESSAGES_MODE.key} is set correctly")
    }
  }

  private def sendActiveMessage(endpoint: UcpEndpoint, am: UCXActiveMessage,
      dataAddress: Long, dataSize: Long, cb: UcxCallback): Unit = {
    logDebug(s"Sending $am msg of size $dataSize")

    // This isn't coming from the pool right now because it would be a bit of a
    // waste to get a larger hard-partitioned buffer just for 8 bytes.
    // TODO: since we no longer have metadata limits, the pool can be managed using the
    //   address-space allocator, so we should obtain this direct buffer from that pool
    val header = ByteBuffer.allocateDirect(8)
    header.putLong(am.header)
    header.rewind()

    endpoint.sendAmNonBlocking(
      am.activeMessageId,
      TransportUtils.getAddress(header),
      header.remaining(),
      dataAddress,
      dataSize,
      activeMessageMode,
      new UcxCallback {
        override def onSuccess(request: UcpRequest): Unit = {
          cb.onSuccess(request)
          RapidsStorageUtils.dispose(header)
        }

        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
          cb.onError(ucsStatus, errorMsg)
          RapidsStorageUtils.dispose(header)
        }
      })
  }

  def sendActiveMessage(endpointId: Long, am: UCXActiveMessage,
                        dataAddress: Long, dataSize: Long, cb: UcxCallback): Unit = {
    onWorkerThreadAsync(() => {
      endpoints.synchronized {
        val ep = endpoints.get(endpointId)
        if (ep == null) {
          throw new IllegalStateException(
            s"Trying to send a message to an endpoint that doesn't exist ${endpointId}")
        }
        sendActiveMessage(ep, am, dataAddress, dataSize, cb)
      }
    })
  }

  def getServerConnection: UCXServerConnection = serverConnection

  def receive(alt: AddressLengthTag, cb: UCXTagCallback): Unit = {
    val ucxCb = new UcxCallback {
      override def onError(ucsStatus: Int, errorMsg: String): Unit = {
        if (ucsStatus == UCX.UCS_ERR_CANCELED) {
          logWarning(
            s"Cancelled: tag=${TransportUtils.toHex(alt.tag)}," +
              s" status=$ucsStatus, msg=$errorMsg")
          cb.onCancel(alt)
        } else {
          logError(s"Error receiving: $ucsStatus $errorMsg => $alt")
          cb.onError(alt, ucsStatus, errorMsg)
        }
      }

      override def onSuccess(request: UcpRequest): Unit = {
        logTrace(s"Success receiving calling callback ${TransportUtils.toHex(alt.tag)}")
        cb.onSuccess(alt)
      }
    }

    onWorkerThreadAsync(() => {
      logTrace(s"Handling receive for tag ${TransportUtils.toHex(alt.tag)}")
      val request = worker.recvTaggedNonBlocking(
        alt.address,
        alt.length,
        alt.tag,
        UCX.MATCH_FULL_TAG,
        ucxCb)
      cb.onMessageStarted(request)
    })
  }

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

  def assignResponseTag(): Long = responseTag.incrementAndGet()

  private lazy val ucxAddress: ByteBuffer = if (rapidsConf.shuffleUcxUseSockaddr) {
    val listenerAddress = listener.get.getAddress
    val hostnameBytes = listenerAddress.getAddress.getAddress
    val result = ByteBuffer.allocateDirect(4 + hostnameBytes.length)
    result.putInt(listenerAddress.getPort)
    result.put(hostnameBytes)
    result.rewind()
    result
  } else {
    worker.getAddress
  }

  /**
   * Connect to a remote UCX management port.
   *
   * @param peerMgmtHost management TCP host
   * @param peerMgmtPort management TCP port
   * @return Connection object representing this connection
   */
  def getConnection(peerExecutorId: Int,
      peerMgmtHost: String,
      peerMgmtPort: Int): ClientConnection = {
    val getConnectionStartTime = System.currentTimeMillis()
    val result = connectionCache.computeIfAbsent(peerExecutorId, _ => {
      val connection = new UCXClientConnection(
        peerExecutorId, peerTag.incrementAndGet(), this, transport)
      startConnection(peerExecutorId, connection, peerMgmtHost, peerMgmtPort)
      connection
    })
    logDebug(s"Got connection for executor ${peerExecutorId} in " +
      s"${System.currentTimeMillis() - getConnectionStartTime} ms")
    result
  }

  def onWorkerThreadAsync(task: () => Unit): Unit = {
    workerTasks.add(task)
    if (rapidsConf.shuffleUcxUseWakeup) {
      try {
        worker.signal()
      } catch {
        case npe: NullPointerException => logError(s"NPE while shutting down", npe)
      }
    }
  }

  // client side
  private def startConnection(peerExecutorId: Int, connection: UCXClientConnection,
      peerMgmtHost: String,
      peerMgmtPort: Int) = endpoints.synchronized {
    if (!endpoints.contains(peerExecutorId.toLong)) {
      logInfo(s"Connecting to $peerMgmtHost:$peerMgmtPort")
      onWorkerThreadAsync(() => {
        val sockAddr = new InetSocketAddress(peerMgmtHost, peerMgmtPort)
        val epParams = getEpParams.setSocketAddress(sockAddr).setNoLoopbackMode()
        endpoints.synchronized {
          if (endpoints.contains(peerExecutorId.toLong)) {
            logWarning(s"ALREADY HAVE ENDPOINT ON START CONNECTION for ${peerExecutorId}")
          } else {
            logWarning(s"CREATE AN ENDPOINT ON START CONNECTION for ${peerExecutorId}")
            val ep = worker.newEndpoint(epParams)
            endpoints.put(peerExecutorId.toLong, ep)
            logInfo(s"Initiated an UCPListener connection to $peerMgmtHost, $peerMgmtPort, ${ep}")
          }
        }
      })
    } else {
      logInfo(s"Already have an endpoint for ${peerMgmtHost}:$peerMgmtPort")
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
}

object UCX {
  private var _instance: UCX = null

  def setInstance(u: UCX): Unit = { _instance = u }

  def doProgress(): Unit = { _instance.doProgress() }
  // This is used to distinguish a cancelled request vs. other errors
  // as the callback is the same (onError)
  // from https://github.com/openucx/ucx/blob/master/src/ucs/type/status.h
  private val UCS_ERR_CANCELED = -16

  // We may consider matching tags partially for different request types
  private val MATCH_FULL_TAG: Long = 0xFFFFFFFFFFFFFFFFL

  def formatAmIdAndHeader(activeMessageId: Int, header: Long) =
    s"[amId=${TransportUtils.toHex(activeMessageId)}, hdr=${TransportUtils.toHex(header)}]"
}
