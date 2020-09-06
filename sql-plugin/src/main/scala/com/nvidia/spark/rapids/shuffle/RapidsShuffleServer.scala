/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.shuffle

import java.util.concurrent.{ConcurrentLinkedQueue, Executor}

import ai.rapids.cudf.{Cuda, CudaUtil, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.format.{BufferMeta, BufferTransferRequest, TableMeta, TransferRequest}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.GpuShuffleEnv
import org.apache.spark.storage.{BlockManagerId, ShuffleBlockBatchId}

/**
 * Trait used for the server to get buffer metadata (for metadata requests), and
 * also to acquire a buffer (for transfer requests)
 */
trait RapidsShuffleRequestHandler {
  /**
   * This is a query into the manager to get the `TableMeta` corresponding to a
   * shuffle block.
   * @param shuffleBlockBatchId `ShuffleBlockBatchId` with (shuffleId, mapId,
   *                            startReduceId, endReduceId)
   * @return a sequence of `TableMeta` describing batches corresponding to a block.
   */
  def getShuffleBufferMetas(shuffleBlockBatchId: ShuffleBlockBatchId): Seq[TableMeta]

  /**
   * Acquires (locks w.r.t. the memory tier) a [[RapidsBuffer]] corresponding to a table id.
   * @param tableId the unique id for a table in the catalog
   * @return a [[RapidsBuffer]] which is reference counted, and should be closed by the acquirer
   */
  def acquireShuffleBuffer(tableId: Int): RapidsBuffer
}

/**
 * A server that replies to shuffle metadata messages, and issues device/host memory sends.
 *
 * A single command thread is used to orchestrate sends/receives and to remove
 * from transport's progress thread.
 *
 * @param transport the transport we were configured with
 * @param serverConnection a connection object, which contains functions to send/receive
 * @param originalShuffleServerId spark's `BlockManagerId` for this executor
 * @param requestHandler instance of [[RapidsShuffleRequestHandler]]
 * @param exec Executor used to handle tasks that take time, and should not be in the
 *             transport's thread
 * @param copyExec Executor used to handle synchronous mem copies
 * @param bssExec Executor used to handle [[BufferSendState]]s that are waiting
 *                for bounce buffers to become available
 * @param rapidsConf plugin configuration instance
 */
class RapidsShuffleServer(transport: RapidsShuffleTransport,
                          serverConnection: ServerConnection,
                          val originalShuffleServerId: BlockManagerId,
                          requestHandler: RapidsShuffleRequestHandler,
                          exec: Executor,
                          copyExec: Executor,
                          bssExec: Executor,
                          rapidsConf: RapidsConf) extends AutoCloseable with Logging {
  /**
   * On close, this is set to false to indicate that the server is shutting down.
   */
  private[this] var started = true

  private object ShuffleServerOps {
    /**
     * When a transfer request is received during a callback, the handle code is offloaded via this
     * event to the server thread.
     * @param tx the live transaction that should be closed by the handler
     * @param metaRequestBuffer contains the metadata request that should be closed by the
     *                          handler
     */
    case class HandleMeta(tx: Transaction, metaRequestBuffer: RefCountedDirectByteBuffer)

    /**
     * When transfer request is received (to begin sending buffers), the handling is offloaded via
     * this event on the server thread. Note that, [[BufferSendState]] encapsulates one more more
     * requests to send buffers, and [[HandleTransferRequest]] may be posted multiple times
     * in order to handle the request fully.
     * @param sendState instance of [[BufferSendState]] used to complete a transfer request.
     */
    case class HandleTransferRequest(sendState: Seq[BufferSendState])
  }

  import ShuffleServerOps._

  private var port: Int = -1

  /**
   * Returns a TCP port that is expected to respond to rapids shuffle protocol.
   * Throws if this server is not started yet, which is an illegal state.
   * @return the port
   */
  def getPort: Int = {
    if (port == -1) {
      throw new IllegalStateException("RapidsShuffleServer port is not initialized")
    }
    port
  }

  /**
   * Kick off the underlying connection, and listen for initial requests.
   */
  def start(): Unit = {
    port = serverConnection.startManagementPort(originalShuffleServerId.host)
    // kick off our first receives
    doIssueReceive(RequestType.MetadataRequest)
    doIssueReceive(RequestType.MetadataRequest)
    doIssueReceive(RequestType.MetadataRequest)
    doIssueReceive(RequestType.MetadataRequest)
    doIssueReceive(RequestType.MetadataRequest)
    doIssueReceive(RequestType.MetadataRequest)
    doIssueReceive(RequestType.MetadataRequest)
    doIssueReceive(RequestType.MetadataRequest)

    doIssueReceive(RequestType.TransferRequest)
    doIssueReceive(RequestType.TransferRequest)
    doIssueReceive(RequestType.TransferRequest)
    doIssueReceive(RequestType.TransferRequest)
    doIssueReceive(RequestType.TransferRequest)
    doIssueReceive(RequestType.TransferRequest)
    doIssueReceive(RequestType.TransferRequest)
    doIssueReceive(RequestType.TransferRequest)

  }

  def handleOp(serverTask: Any): Unit = {
    try {
      serverTask match {
        case HandleMeta(tx, metaRequestBuffer) =>
          doHandleMeta(tx, metaRequestBuffer)
        case HandleTransferRequest(wt: Seq[BufferSendState]) =>
          doHandleTransferRequest(wt)
      }
    } catch {
      case t: Throwable => {
        logError("Exception occurred while handling shuffle server task.", t)
      }
    }
  }

  /**
   * Pushes a task onto the queue to be handled by the server executor.
   *
   * All callbacks handled in the server (from the transport) need to be offloaded into
   * this pool. Note, if this thread blocks we are blocking the progress thread of the transport.
   *
   * @param op One of the case classes in `ShuffleServerOps`
   */
  def asyncOrBlock(op: Any): Unit = {
    exec.execute(() => handleOp(op))
  }

  /**
   * Pushes a task onto the queue to be handled by the server's copy executor.
   *
   * @note - at this stage, tasks in this pool can block (it will grow as needed)
   *
   * @param op One of the case classes in [[ShuffleServerOps]]
   */
  private[this] def asyncOnCopyThread(op: Any): Unit = {
    copyExec.execute(() => handleOp(op))
  }

  /**
   * Keep a list of BufferSendState that are waiting for bounce buffers.
   */
  private[this] val bssQueue = new ConcurrentLinkedQueue[BufferSendState]()
  private[this] val bssContinueQueue = new ConcurrentLinkedQueue[BufferSendState]()

  /**
   * Executor that loops until it finds bounce buffers for [[BufferSendState]],
   * and when it does it hands them off to a thread pool for handling.
   */
  bssExec.execute(() => {
    // TODO: have this thread acquire the gpu
    var rangeMsg = "BSSQEmpty"
    while (started) {
      val bssExecRange = new NvtxRange(rangeMsg, NvtxColor.PURPLE)
      var bssContinue: BufferSendState = null
      var bssToIssue = Seq[BufferSendState]()
      try {
        bssContinue = bssContinueQueue.peek()
        while (bssContinue != null) {
          logInfo(s"Got something to continue ${bssContinue}")
          bssExec.synchronized {
            bssContinueQueue.remove(bssContinue)
            bssToIssue = bssToIssue :+ bssContinue
            bssContinue = bssContinueQueue.peek()
          }
        }
      } catch {
        case t: Throwable => {
          logError("Error while handling BufferSendState", t)
          if (bssContinue != null) {
            bssContinue.close()
            bssContinue = null
          }
        }
      }

      var bss: BufferSendState = null
      try {
        bss = bssQueue.peek()
        var continue = true
        while (bss != null && continue) {
          bssExec.synchronized {
            if (bss.acquireBounceBuffersNonBlocking) {
              bssQueue.remove(bss)
              bssToIssue = bssToIssue :+ bss
              bss = bssQueue.peek()
            } else {
              logInfo(s"Cant acquire server bounce buffers for ${bss}")
              continue = false
            }
          }
        }

        if (!continue) {
          rangeMsg = "BSSQWait"
        } else {
          rangeMsg = "BSSQEmpty"
        }

        if (bssToIssue.nonEmpty) {
          asyncOnCopyThread(HandleTransferRequest(bssToIssue))
        }

        bssExec.synchronized {
          bssExec.wait(100)
        }

        bssExecRange.close()
      } catch {
        case t: Throwable => {
          logError("Error while handling BufferSendState", t)
          if (bss != null) {
            bss.close()
            bss = null
          }
        }
      }
    }
  })

  val serverStream = new Cuda.Stream(true)

  /**
   * Handler for a metadata request. It queues request handlers for either
   * [[RequestType.MetadataRequest]] or [[RequestType.TransferRequest]], and re-issues
   * receives for either type of request.
   *
   * NOTE: This call must be non-blocking. It is called from the progress thread.
   *
   * @param requestType The request type received
   */
  private def doIssueReceive(requestType: RequestType.Value): Unit = {
    logDebug(s"Waiting for a new connection. Posting ${requestType} receive.")
    val metaRequest = transport.getMetaBuffer(rapidsConf.shuffleMaxMetadataSize)

    val alt = AddressLengthTag.from(
      metaRequest.acquire(),
      serverConnection.composeRequestTag(requestType))

    serverConnection.receive(alt,
      tx => {
        val handleMetaRange = new NvtxRange("Handle Meta Request", NvtxColor.PURPLE)
        try {
          if (requestType == RequestType.MetadataRequest) {
            doIssueReceive(RequestType.MetadataRequest)
            doHandleMeta(tx, metaRequest)
          } else {
            val bss = new BufferSendState(
              tx, metaRequest, transport, bssExec, requestHandler,
              rapidsConf.shuffleUcxBounceBuffersSize, serverStream)
            bssQueue.add(bss)

            // tell the bssExec to wake up to try to handle the new BufferSendState
            bssExec.synchronized {
              bssExec.notifyAll()
            }
            logDebug(s"Got a transfer request ${bss} from ${tx}. " +
              s"I now have ${bssQueue.size} and ${bssContinueQueue} BSSs")
            doIssueReceive(RequestType.TransferRequest)
          }
        } finally {
          handleMetaRange.close()
        }
      })
  }

  /**
   * Function to handle `MetadataRequest`s. It will populate and issue a
   * `MetadataResponse` response for the appropriate client.
   *
   * @param tx the inbound [[Transaction]]
   * @param metaRequest a [[RefCountedDirectByteBuffer]] holding a `MetadataRequest` message.
   */
  def doHandleMeta(tx: Transaction, metaRequest: RefCountedDirectByteBuffer): Unit = {
    val doHandleMetaRange = new NvtxRange("doHandleMeta", NvtxColor.PURPLE)
    val start = System.currentTimeMillis()
    try {
      if (tx.getStatus == TransactionStatus.Error) {
        logError("error getting metadata request: " + tx)
        metaRequest.close() // the buffer is not going to be handed anywhere else, so lets close it
        throw new IllegalStateException(s"Error occurred while while handling metadata $tx")
      } else {
        logDebug(s"Received metadata request: $tx => $metaRequest")
        try {
          handleMetadataRequest(metaRequest)
        } catch {
          case e: Throwable => {
            logError(s"Exception while handling metadata request from $tx: ", e)
            throw e
          }
        }
      }
    } finally {
      logDebug(s"Metadata request handled in ${TransportUtils.timeDiffMs(start)} ms")
      doHandleMetaRange.close()
      tx.close()
    }
  }

  /**
   * Handles the very first message that a client will send, in order to request Table/Buffer info.
   * @param metaRequest a [[RefCountedDirectByteBuffer]] holding a `MetadataRequest` message.
   */
  def handleMetadataRequest(metaRequest: RefCountedDirectByteBuffer): Unit = {
    try {
      val req = ShuffleMetadata.getMetadataRequest(metaRequest.getBuffer())

      // target executor to respond to
      val peerExecutorId = req.executorId()

      // tag to use for the response message
      val responseTag = req.responseTag()

      logDebug(s"Received request req:\n: ${ShuffleMetadata.printRequest(req)}")
      logDebug(s"HandleMetadataRequest for peerExecutorId $peerExecutorId and " +
        s"responseTag ${TransportUtils.formatTag(req.responseTag())}")

      // NOTE: MetaUtils will have a simpler/better way of handling creating a response.
      // That said, at this time, I see some issues with that approach from the flatbuffer
      // library, so the code to create the metadata response will likely change.
      val responseTables = (0 until req.blockIdsLength()).flatMap { i =>
        val blockId = req.blockIds(i)
        // this is getting shuffle buffer ids
        requestHandler.getShuffleBufferMetas(
          ShuffleBlockBatchId(blockId.shuffleId(), blockId.mapId(),
            blockId.startReduceId(), blockId.endReduceId()))
      }

      val metadataResponse = 
        ShuffleMetadata.buildMetaResponse(responseTables, req.maxResponseSize())
      // Wrap the buffer so we keep a reference to it, and we destroy it later on .close
      val respBuffer = new RefCountedDirectByteBuffer(metadataResponse)
      val materializedResponse = ShuffleMetadata.getMetadataResponse(metadataResponse)

      logDebug(s"Response will be at tag ${TransportUtils.formatTag(responseTag)}:\n"+
        s"${ShuffleMetadata.printResponse("responding", materializedResponse)}")

      val response = AddressLengthTag.from(respBuffer.acquire(), responseTag)

      // Issue the send against [[peerExecutorId]] as described by the metadata message
      val tx = serverConnection.send(peerExecutorId, response, tx => {
        try {
          if (tx.getStatus == TransactionStatus.Error) {
            logError(s"Error sending metadata response in tx $tx")
            throw new IllegalStateException(
              s"Error while handling a metadata response send for $tx")
          } else {
            val stats = tx.getStats
            logDebug(s"Sent metadata ${stats.sendSize} in ${stats.txTimeMs} ms")
          }
        } finally {
          respBuffer.close()
          tx.close()
        }
      })
      logDebug(s"Waiting for send metadata to complete: $tx")
    } finally {
      metaRequest.close()
    }
  }


  /**
   * This will kick off, or continue to work, a [[BufferSendState]] object
   * until all tables are fully transmitted.
   *
   * @param bufferSendState state object tracking sends needed to fulfill a TransferRequest
   */
  def doHandleTransferRequest(bufferSendStates: Seq[BufferSendState]): Unit = {
    val stuff = bufferSendStates.map { bufferSendState =>
      val doHandleTransferRequest =
        new NvtxRange(s"doHandleTransferRequest first=${bufferSendState.firstTime}", NvtxColor.CYAN)

      val start = System.currentTimeMillis()

      try {
        require(bufferSendState.hasNext, "Attempting to handle a complete transfer request.")

        // [[BufferSendState]] will continue to return buffers to send, as long as there
        // is work to be done.
        val buffersToSend = bufferSendState.next()

        val transferRequest = bufferSendState.getTransferRequest()

        logDebug(s"Handling transfer request for ${transferRequest.executorId()} " +
            s"with ${buffersToSend}")

        logDebug(s"[before]${bufferSendState}")
        (transferRequest, bufferSendState, buffersToSend)
      } finally {
        logDebug(s"Transfer request handled in ${TransportUtils.timeDiffMs(start)} ms")
        doHandleTransferRequest.close()
      }
    }

    // Get on the GPU
    serverStream.sync()

    stuff.foreach {
      case (transferRequest: TransferRequest,
      bufferSendState: BufferSendState,
      buffersToSend: AddressLengthTag) =>
      serverConnection.send(transferRequest.executorId(), buffersToSend, new TransactionCallback {
        override def apply(bufferTx: Transaction): Unit = {
          try {
            logDebug(s"Done with the send for ${bufferSendState} with ${buffersToSend}")

            if (bufferSendState.hasNext) {
              // continue issuing sends.
              logDebug(s"Buffer send state ${bufferSendState} is NOT done. " +
                  s"I now have ${bssQueue.size} BSSs")
              bssExec.synchronized {
                bssContinueQueue.add(bufferSendState)
                bssExec.notifyAll()
              }
            } else {
              val transferResponse = bufferSendState.getTransferResponse()

              // send the transfer response
              serverConnection.send(
                transferRequest.executorId,
                AddressLengthTag.from(transferResponse.acquire(), transferRequest.responseTag()),
                transferResponseTx => {
                  transferResponse.close()
                  transferResponseTx.close()
                })

              // close up the [[BufferSendState]] instance
              logInfo(s"Buffer send state ${buffersToSend.tag} is done. Closing. " +
                  s"I now have ${bssQueue.size} BSSs.")
              bufferSendState.close()
            }
          } finally {
            bufferTx.close()
          }
        }
      })
    }
  }

  override def close(): Unit = {
    started = false
    bssExec.synchronized {
      bssExec.notifyAll()
    }
  }
}
