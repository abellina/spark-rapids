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

import java.io._
import java.net.{InetSocketAddress, ServerSocket, Socket, SocketException}
import java.nio.ByteBuffer
import java.util.concurrent.{ConcurrentHashMap, ConcurrentLinkedQueue, Executors, TimeUnit}
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}

import scala.collection.mutable.ArrayBuffer
import ai.rapids.cudf.{MemoryBuffer, NvtxColor, NvtxRange}
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.nvidia.spark.rapids.GpuDeviceManager
import com.nvidia.spark.rapids.shuffle.{AddressLengthTag, ClientConnection, MemoryRegistrationCallback, RefCountedDirectByteBuffer, RequestType, Transaction, TransactionCallback, TransportUtils}
import org.apache.hadoop.yarn.client.api.AMRMClient
import org.openucx.jucx._
import org.openucx.jucx.ucp._
import org.apache.spark.internal.Logging
import org.openucx.jucx.ucs.UcsConstants
import org.openucx.jucx.ucs.UcsConstants.MEMORY_TYPE

case class WorkerAddress(address: ByteBuffer)

case class Rkeys(rkeys: Seq[ByteBuffer])

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
 * @param executorId unique id (int) that identifies the local executor
 * @param usingWakeupFeature (true by default) set to false to use a hot loop, as opposed to
 *                           UCP provided signal/wait
 */
class UCX(transport: UCXShuffleTransport,
          val executorId: Int,
          usingWakeupFeature: Boolean = true) extends AutoCloseable with Logging {

  private[this] val context = {
    val contextParams = new UcpParams().requestTagFeature()
    if (usingWakeupFeature) {
      contextParams.requestWakeupFeature().requestAmFeature()
    }
    new UcpContext(contextParams)
  }

  logInfo(s"UCX context created")

  // this object implements the transport-friendly interface for UCX
  private[this] val serverConnection = new UCXServerConnection(this)

  // monotonically increasing counter that holds the txId (for debug purposes, at this stage)
  private[this] val txId = new AtomicLong(0L)

  private var worker: UcpWorker = _
  private val endpoints = new ConcurrentHashMap[Long, UcpEndpoint]()
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

  // management port socket
  private var serverSocket: ServerSocket = _
  private val acceptService = Executors.newSingleThreadExecutor(
    new ThreadFactoryBuilder().setNameFormat("ucx-mgmt-thread-%d").build)

  private val serverService = Executors.newCachedThreadPool(
    new ThreadFactoryBuilder().setNameFormat("ucx-connection-server-%d").build)

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
  private val executorIdToPeerTag = new ConcurrentHashMap[Long, Long]()

  // holds memory registered against UCX that should be de-register on exit (used for bounce
  // buffers)
  // NOTE: callers should hold the `registeredMemory` lock before modifying this array
  val registeredMemory = new ArrayBuffer[UcpMemory]

  // when this flag is set to true, an async call to `register` hasn't completed in
  // the worker thread. We need this to complete prior to getting the `rkeys`.
  private var pendingRegistration = false

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

      if (usingWakeupFeature) {
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
        val nvtxRange = new NvtxRange("UCX Draining Worker", NvtxColor.RED)
        try {
          while (worker.progress() > 0) {}
        } finally {
          nvtxRange.close()
        }
      }

      while(initialized) {
        try {
          worker.progress()
          // else worker.progress returned 0
          if (usingWakeupFeature) {
            drainWorker()
            val sleepRange = new NvtxRange("UCX Sleeping", NvtxColor.PURPLE)
            try {
              worker.waitForEvents()
            } finally {
              sleepRange.close()
            }
          }

          while (!workerTasks.isEmpty) {
            val nvtxRange = new NvtxRange("UCX Handling Tasks", NvtxColor.CYAN)
            try {
              val wt = workerTasks.poll()
              if (wt != null) {
                wt()
              }
            } finally {
              nvtxRange.close()
            }
            worker.progress()
          }
        } catch {
          case t: Throwable =>
            logError("Exception caught in UCX progress thread. Continuing.", t)
        }
      }
    })
  }

  /**
   * Starts a TCP server to listen for external clients, returning with
   * what port it used.
   *
   * @param mgmtHost String the hostname to bind to
   * @return port bound
   */
  def startManagementPort(mgmtHost: String): Int = {
    var portBindAttempts = 100
    var portBound = false
    while (!portBound && portBindAttempts > 0) {
      try {
        logInfo(s"Starting ephemeral UCX management port at host $mgmtHost")
        // TODO: use ucx listener for this
        serverSocket = new ServerSocket()
        // open a TCP/IP socket to connect to a client
        // send the worker address to the client who wants to talk to us
        // associate with [[onNewConnection]]
        try {
          serverSocket.bind(new InetSocketAddress(mgmtHost, 0))
        } catch {
          case ioe: IOException =>
            logError(s"Unable to bind using host [$mgmtHost]", ioe)
            throw ioe
        }
        logInfo(s"Successfully bound to $mgmtHost:${serverSocket.getLocalPort}")
        portBound = true

        acceptService.execute(() => {
          while (initialized) {
            logInfo(s"Accepting UCX management connections.")
            try {
              val s = serverSocket.accept()
              // throw into a thread pool to actually handle the stream
              serverService.execute(() => {
                // disable Nagle's algorithm, in hopes of data not buffered by TCP
                s.setTcpNoDelay(true)
                handleSocket(s)
              })
            } catch {
              case e: Throwable if initialized =>
                // This will cause the `SparkUncaughtExceptionHandler` to get invoked
                // and it will shut down the executor (as it should).
                throw e
              case _: SocketException if !initialized =>
                // `initialized = false` means we are shutting down,
                // the socket will throw `SocketException` in this case
                // to unblock the accept, when `close()` is called.
                logWarning(s"UCX management socket closing")
              case ue: Throwable =>
                // a catch-all in case we get a non `SocketException` while closing (!initialized)
                logError(s"Unexpected exception while closing UCX management socket", ue)
            }
          }
        })
      } catch {
        case ioe: IOException =>
          logWarning(s"Retrying bind attempts $portBindAttempts", ioe)
          portBindAttempts = portBindAttempts - 1
      }
    }
    if (!portBound) {
      throw new IllegalStateException(s"Cannot bind UCX, tried $portBindAttempts times")
    }
    serverSocket.getLocalPort
  }

  // LOW LEVEL API
  def send(endpointId: Long, alt: AddressLengthTag, cb: UCXTagCallback): Unit = {
    val ucxCb = new UcxCallback {
      override def onError(ucsStatus: Int, errorMsg: String): Unit = {
        if (ucsStatus == UCX.UCS_ERR_CANCELED) {
          logWarning(
            s"Cancelled: tag=${TransportUtils.formatTag(alt.tag)}," +
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

  trait AmCallback {
    def apply(id: Option[Long], buff: RefCountedDirectByteBuffer, ucpEndpoint: UcpEndpoint): Unit
  }

  private val responseAmCallbacks = new ConcurrentHashMap[Int, Int]()

  private val callbacks = new ConcurrentHashMap[Long, AmCallback]()

  /**
   * Register a response handler (clients will use this)
   * @param amId - only 5 bits should be set!
   * @param peerExecutorId
   * @param cb
   */
  def registerResponseHandler(amId: Int, peerExecutorId: Int, hdr: Long, cb: AmCallback): Unit = {
    logDebug(s"setup responseHandler for ${peerExecutorId} and " +
      s"amId ${TransportUtils.formatTag(amId)}")
    callbacks.put(hdr, cb)
    responseAmCallbacks.computeIfAbsent(amId, _ => {
      setActiveMessageCallback(amId, (id, resp, responseEp) => {
        val peerExec = ((id.get & 0xFFFFFFFF00000000L) >> 32).toInt
        logDebug(s"Getting peer am callback for amId " +
          s"${TransportUtils.formatTag(amId)} " +
          s"header: ${TransportUtils.formatTag(id.get)} " +
          s"peerExec: $peerExec")
        logInfo(s"${callbacks.size()} active messages pending")
        callbacks.get(id.get)(id, resp, responseEp)
        callbacks.remove(id.get)
      })
    })
  }

  def setActiveMessageCallback(amId: Int,
      cb: (Option[Long], RefCountedDirectByteBuffer, UcpEndpoint) => Unit): Int = {
    onWorkerThreadAsync(() => {
      logInfo(s"Setting am recv handler for active message ${TransportUtils.formatTag(amId)}")
      worker.setAmRecvHandler(amId,
        (headerAddr, headerSize, amData: UcpAmData, replyEp: UcpEndpoint) => {
          logDebug(s"AT CALLBACK ${TransportUtils.formatTag(amId)} -- " +
            s"${headerAddr} -- ${headerSize} -- ${amData}")

          val hdr = if (headerSize == 8) {
            Option(UcxUtils.getByteBufferView(headerAddr, headerSize).getLong)
          } else {
            None
          }

          val resp = transport.getDirectByteBuffer(amData.getLength)

          if (amData.isDataValid) {
            amData.receive(UcxUtils.getAddress(resp.getBuffer()),new UcxCallback {
              override def onError(ucsStatus: Int, errorMsg: String): Unit = {
                logError(s"V. AM ERROR ${ucsStatus} ${errorMsg}")
                resp.close()
              }
              override def onSuccess(request: UcpRequest): Unit = {
                // TODO: request is null?
                cb(hdr, resp, replyEp)
              }
            })
            UcsConstants.STATUS.UCS_OK
          } else {
            amData.receive(UcxUtils.getAddress(resp.getBuffer()),new UcxCallback {
              override def onError(ucsStatus: Int, errorMsg: String): Unit = {
                logError(s"NV. AM ERROR ${ucsStatus} ${errorMsg}")
                resp.close()
                amData.close()
              }
              override def onSuccess(request: UcpRequest): Unit = {
                cb(hdr, resp, replyEp)
                amData.close()
              }
            })
            UcsConstants.STATUS.UCS_INPROGRESS
          }
        })
    })
    amId
  }

  def sendAm(epId: Long, hdr: Long, amId: Int, address: Long, size: Long, cb: UcxCallback): Unit = {
    onWorkerThreadAsync(() => {
      val ep = endpoints.get(epId)
      logDebug(s"sending to amId: ${TransportUtils.formatTag(amId)} msg of size ${size}")

      val header = new RefCountedDirectByteBuffer(ByteBuffer.allocateDirect(8))
      header.acquire()
      header.getBuffer().putLong(hdr)
      header.getBuffer().rewind()

      ep.sendAmNonBlocking(
        amId,
        TransportUtils.getAddress(header.getBuffer()),
        8L,
        address,
        size,
        UcpConstants.UCP_AM_SEND_FLAG_RNDV,
        new UcxCallback {
          override def onSuccess(request: UcpRequest): Unit = {
            if (cb != null) {
              cb.onSuccess(request)
            }
            header.close()
          }

          override def onError(ucsStatus: Int, errorMsg: String): Unit = {
            if (cb != null) {
              cb.onError(ucsStatus, errorMsg)
            }
            header.close()
          }
        })
    })
  }

  def getServerConnection: UCXServerConnection = serverConnection

  def receive(alt: AddressLengthTag, cb: UCXTagCallback): Unit = {
    val ucxCb = new UcxCallback {
      override def onError(ucsStatus: Int, errorMsg: String): Unit = {
        if (ucsStatus == UCX.UCS_ERR_CANCELED) {
          logWarning(
            s"Cancelled: tag=${TransportUtils.formatTag(alt.tag)}," +
              s" status=$ucsStatus, msg=$errorMsg")
          cb.onCancel(alt)
        } else {
          logError(s"Error receiving: $ucsStatus $errorMsg => $alt")
          cb.onError(alt, ucsStatus, errorMsg)
        }
      }

      override def onSuccess(request: UcpRequest): Unit = {
        logTrace(s"Success receiving calling callback ${TransportUtils.formatTag(alt.tag)}")
        cb.onSuccess(alt)
      }
    }

    onWorkerThreadAsync(() => {
      logTrace(s"Handling receive for tag ${TransportUtils.formatTag(alt.tag)}")
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
        //request.close()
      } catch {
        case e: Throwable =>
          logError("Error while cancelling UCX request: ", e)
      }
    })
  }

  private[ucx] def assignResponseTag(): Long = responseTag.incrementAndGet()

  private def ucxWorkerAddress: ByteBuffer = worker.getAddress

  /**
   * Establish a new [[UcpEndpoint]] given a [[WorkerAddress]]. It also
   * caches them s.t. at [[close]] time we can release resources.
   *
   * @param endpointId    presently an executorId, it is used to distinguish between endpoints
   *                      when routing messages outbound
   * @param workerAddress the worker address for the remote endpoint (ucx opaque object)
   * @param peerRkeys list of UCX rkeys that the peer has sent us for unpacking
   * @return returns a [[UcpEndpoint]] that can later be used to send on (from the
   *         progress thread)
   */
  private[ucx] def setupEndpoint(
      endpointId: Long, workerAddress: WorkerAddress, peerRkeys: Rkeys): UcpEndpoint = {
    logDebug(s"Starting/reusing an endpoint to $workerAddress with id $endpointId")
    // create an UCX endpoint using workerAddress
    endpoints.computeIfAbsent(endpointId,
      (_: Long) => {
        logInfo(s"No endpoint found for $endpointId. Adding it.")
        val ep = worker.newEndpoint(
          new UcpEndpointParams()
            .setUcpAddress(workerAddress.address))
        peerRkeys.rkeys.foreach(ep.unpackRemoteKey)
        ep
      })
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
      val connection = new UCXClientConnection(peerExecutorId, peerTag.incrementAndGet(), this)
      startConnection(connection, peerMgmtHost, peerMgmtPort)
      connection
    })
    logDebug(s"Got connection for executor ${peerExecutorId} in " +
      s"${System.currentTimeMillis() - getConnectionStartTime} ms")
    result
  }

  private[ucx] def onWorkerThreadAsync(task: () => Unit): Unit = {
    workerTasks.add(task)
    if (usingWakeupFeature) {
      worker.signal()
    }
  }

  // client side
  private def startConnection(connection: UCXClientConnection,
      peerMgmtHost: String,
      peerMgmtPort: Int) = {
    logInfo(s"Connecting to $peerMgmtHost to $peerMgmtPort")
    val nvtx = new NvtxRange(s"UCX Connect to $peerMgmtHost:$peerMgmtPort", NvtxColor.RED)
    try {
      val socket = new Socket(peerMgmtHost, peerMgmtPort)
      try {
        socket.setTcpNoDelay(true)
        val os = socket.getOutputStream
        val is = socket.getInputStream

        // "this executor id will receive on tmpLocalReceiveTag for this Connection"
        UCXConnection.writeHandshakeHeader(
          os, ucxWorkerAddress, executorId, localRkeys)

        // "the remote executor will receive on remoteReceiveTag, and expects this executor to
        // receive on localReceiveTag"
        val (peerWorkerAddress, remoteExecutorId, peerRkeys) = UCXConnection.readHandshakeHeader(is)

        val peerExecutorId = connection.getPeerExecutorId
        if (remoteExecutorId != peerExecutorId) {
          throw new IllegalStateException(s"Attempted to reach executor $peerExecutorId, but" +
            s" instead received reply from $remoteExecutorId")
        }

        onWorkerThreadAsync(() => {
          setupEndpoint(remoteExecutorId, peerWorkerAddress, peerRkeys)
        })

        logInfo(s"NEW OUTGOING UCX CONNECTION $connection")
      } finally {
        socket.close()
      }
      connection
    } finally {
      nvtx.close()
    }
  }

  def assignPeerTag(peerExecutorId: Long): Long =
    executorIdToPeerTag.computeIfAbsent(peerExecutorId, _ => peerTag.incrementAndGet())

  /**
   * Handle an incoming connection on the TCP management port
   * This will fetch the [[WorkerAddress]] from the peer, and establish a UcpEndpoint
   *
   * @param socket an accepted socket to a remote client
   */
  private[ucx] def handleSocket(socket: Socket): Unit = {
    val connectionRange =
      new NvtxRange(s"UCX Handle Connection from ${socket.getInetAddress}", NvtxColor.RED)
    try {
      logDebug(s"Reading worker address from: $socket")
      try {
        val is = socket.getInputStream
        val os = socket.getOutputStream

        // get the peer worker address, we need to store this so we can send to this tag
        val (peerWorkerAddress: WorkerAddress, peerExecutorId: Int, peerRkeys: Rkeys) =
          UCXConnection.readHandshakeHeader(is)

        logInfo(s"Got peer worker address from executor $peerExecutorId")

        // ack what we saw as the local and remote peer tags
        UCXConnection.writeHandshakeHeader(
          os, ucxWorkerAddress, executorId, localRkeys)

        onWorkerThreadAsync(() => {
          setupEndpoint(peerExecutorId, peerWorkerAddress, peerRkeys)
        })

        // peer would have established an endpoint peer -> local
        logInfo(s"Sent server UCX worker address to executor $peerExecutorId")
      } finally {
        // at this point we have handshaked, UCX is ready to go for this point-to-point connection.
        // assume that we get a list of block ids, tag tuples we want to transfer out
        socket.close()
      }
    } finally {
      connectionRange.close()
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

    if (serverSocket != null) {
      serverSocket.close()
      serverSocket = null
    }

    if (usingWakeupFeature && worker != null) {
      worker.signal()
    }

    serverService.shutdown()
    if (!serverService.awaitTermination(500, TimeUnit.MILLISECONDS)) {
      logError("UCX mgmt service failed to terminate correctly")
    }

    progressThread.shutdown()
    if (!progressThread.awaitTermination(500, TimeUnit.MICROSECONDS)) {
      logError("UCX progress thread failed to terminate correctly")
    }

    endpoints.values().forEach(ep => ep.close())

    if (worker != null) {
      worker.close()
    }

    context.close()
  }
}

object UCX {
  // This is used to distinguish a cancelled request vs. other errors
  // as the callback is the same (onError)
  // from https://github.com/openucx/ucx/blob/master/src/ucs/type/status.h
  private val UCS_ERR_CANCELED = -16

  // We may consider matching tags partially for different request types
  private val MATCH_FULL_TAG: Long = 0xFFFFFFFFFFFFFFFFL
}
