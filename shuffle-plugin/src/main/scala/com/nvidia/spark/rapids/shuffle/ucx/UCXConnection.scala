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

import java.io.{InputStream, OutputStream}
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable.ArrayBuffer
import ai.rapids.cudf.{MemoryBuffer, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm
import com.nvidia.spark.rapids.shuffle.{RequestType, _}
import org.openucx.jucx.{UcxCallback, UcxUtils}
import org.openucx.jucx.ucp.{UcpAmData, UcpRequest}
import org.apache.spark.internal.Logging

/**
 * These are private apis used within the ucx package.
 */

/**
 * `UCXTagCallback` is used by [[Transaction]] to handle UCX tag-based operations.
 * It adds the `AddressLengthTag` instance as we use that to track the message and
 * for debugging.
 */
private[ucx] abstract class UCXTagCallback {
  def onError(alt: AddressLengthTag, ucsStatus: Int, errorMsg: String): Unit
  def onMessageStarted(ucxMessage: UcpRequest): Unit
  def onSuccess(alt: AddressLengthTag): Unit
  def onCancel(alt: AddressLengthTag): Unit
}


/**
 * `UCXAmCallback` is used by [[Transaction]] to handle UCX Active Messages operations.
 * The `UCXActiveMessage` object encapsulates an activeMessageId and a header.
 */
private[ucx] abstract class UCXAmCallback {
  def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit
  def onMessageStarted(receiveAm: UcpRequest): Unit
  def onSuccess(am: UCXActiveMessage, buff: TransportBuffer): Unit
  def onCancel(am: UCXActiveMessage): Unit

  // hook to allocate memory on the host
  // a similar hook will be needed for GPU memory
  def onMessageReceived(size: Long): TransportBuffer
}

class UCXServerConnection(ucx: UCX, transport: UCXShuffleTransport)
  extends UCXConnection(ucx) with ServerConnection with Logging {
  override def startManagementPort(host: String): Int = {
    ucx.startManagementPort(host)
  }

  override def registerRequestHandler(
      requestType: RequestType.Value, cb: TransactionCallback): Unit = {

    ucx.registerRequestHandler(
        UCXConnection.composeRequestAmId(requestType), () => new UCXAmCallback {
      private val tx = createTransaction

      tx.start(UCXTransactionType.Request, 1, cb)

      override def onSuccess(am: UCXActiveMessage, buff: TransportBuffer): Unit = {
        logDebug(s"At requestHandler for ${requestType} and am: " +
          s"$am")
        tx.completeWithSuccess(requestType, Option(am.header), Option(buff))
      }

      override def onMessageReceived(size: Long): TransportBuffer = {
        new HostTransportBuffer(transport.getDirectByteBuffer(size))
      }

      override def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit = {
        tx.completeWithError(errorMsg)
      }

      override def onCancel(am: UCXActiveMessage): Unit = {
        tx.completeCancelled(requestType, am.header)
      }

      override def onMessageStarted(receiveAm: UcpRequest): Unit = {
        tx.registerPendingMessage(receiveAm)
      }
    })
  }

  override def send(peerExecutorId: Long,
           messageType: RequestType.Value,
           buffer: AddressLengthTag,
           cb: TransactionCallback): Transaction = {
    val tx = createTransaction
    tx.start(UCXTransactionType.Send, 1, cb)

    val header = buffer.tag

    logDebug(s"Responding to ${peerExecutorId} at ${TransportUtils.toHex(header)} " +
      s"with ${buffer}")

    val responseAm = UCXActiveMessage(
      UCXConnection.composeResponseAmId(messageType), header)

    ucx.sendActiveMessage(peerExecutorId, responseAm,
      buffer.memoryBuffer.get.getAddress, buffer.length,
      new UcxCallback {
        override def onSuccess(request: UcpRequest): Unit = {
          logDebug(s"AM success respond $responseAm")
          tx.complete(TransactionStatus.Success)
        }

        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
          logError(s"AM Error responding ${ucsStatus} ${errorMsg} for $responseAm")
          tx.completeWithError(errorMsg)
        }
      })
    tx
  }

  override def respond(peerExecutorId: Long,
                       messageType: RequestType.Value,
                       header: Long,
                       response: ByteBuffer,
                       cb: TransactionCallback): Transaction = {

    val tx = createTransaction
    tx.start(UCXTransactionType.Request, 1, cb)

    logDebug(s"Responding to ${peerExecutorId} at ${TransportUtils.toHex(header)} " +
      s"with ${response}")

    val responseAm = UCXActiveMessage(
      UCXConnection.composeResponseAmId(messageType), header)

    ucx.sendActiveMessage(peerExecutorId, responseAm,
      TransportUtils.getAddress(response), response.remaining(),
      new UcxCallback {
        override def onSuccess(request: UcpRequest): Unit = {
          logDebug(s"AM success respond $responseAm")
          tx.complete(TransactionStatus.Success)
        }

        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
          logError(s"AM Error responding ${ucsStatus} ${errorMsg} for $responseAm")
          tx.completeWithError(errorMsg)
        }
      })
    tx
  }
}

class UCXClientConnection(peerExecutorId: Int, peerClientId: Long,
    ucx: UCX, transport: UCXShuffleTransport)
  extends UCXConnection(peerExecutorId, ucx) with Arm
  with ClientConnection {

  override def toString: String = {
    s"UCXClientConnection(ucx=$ucx, " +
      s"peerExecutorId=$peerExecutorId, " +
      s"peerClientId=$peerClientId) "
  }

  logInfo(s"UCX Client $this started")

  override def assignBufferTag(msgId: Int): Long =
    UCXConnection.composeBufferTag(peerClientId, msgId)

  override def getPeerExecutorId: Long = peerExecutorId

  override def request(requestType: RequestType.Value, request: ByteBuffer,
                       cb: TransactionCallback): Transaction = {
    val tx = createTransaction
    tx.start(UCXTransactionType.Request, 1, cb)

    // this header is unique, so we can send it with the request
    // expecting it to be echoed back in the response
    val requestHeader = UCXConnection.composeRequestHeader(ucx.getExecutorId.toLong, tx.txId)

    // This is the response active message handler, when the response shows up
    // we'll create a transaction, and set header/message and complete it.
    val amCallback = new UCXAmCallback {
      override def onMessageReceived(size: Long): TransportBuffer = {
        new HostTransportBuffer(transport.getDirectByteBuffer(size.toInt))
      }

      override def onSuccess(am: UCXActiveMessage, buff: TransportBuffer): Unit = {
        tx.completeWithSuccess(requestType, Option(am.header), Option(buff))
      }

      override def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit = {
        tx.completeWithError(errorMsg)
      }

      override def onMessageStarted(receiveAm: UcpRequest): Unit = {
        tx.registerPendingMessage(receiveAm)
      }

      override def onCancel(am: UCXActiveMessage): Unit = {
        tx.completeCancelled(requestType, am.header)
      }
    }

    // Register the active message response handler. Note that the `requestHeader`
    // is expected to come back with the response, and is used to find the
    // correct callback (this is an implementation detail in UCX.scala)
    ucx.registerResponseHandler(
      UCXConnection.composeResponseAmId(requestType), requestHeader, amCallback)

    // kick-off the request
    val requestAm = UCXActiveMessage(
      UCXConnection.composeRequestAmId(requestType), requestHeader)

    logDebug(s"Performing a ${requestType} request of size ${request.remaining()} " +
      s"with tx ${tx}. Active messages: request $requestAm")

    ucx.sendActiveMessage(peerExecutorId, requestAm,
      TransportUtils.getAddress(request), request.remaining(),
      new UcxCallback {
        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
          tx.completeWithError(errorMsg)
        }

        // we don't handle `onSuccess` here, because we want the response
        // to complete that
      })

    tx
  }

  def registerBufferReceiveState(bufferReceiveState: BufferReceiveState,
                                 cb: TransactionCallback): Unit = {

    val headers = bufferReceiveState.allTags

    // we are listening to requests for all clients in the same active message id
    val amId = UCXConnection.composeResponseAmId(RequestType.BufferReceive)

    val amCallback = new UCXAmCallback {
      var currentAlt: AddressLengthTag = null
      override def onError(am: UCXActiveMessage, ucsStatus: Int, errorMsg: String): Unit = {
        logError("Error receiving a window")
      }

      override def onMessageStarted(receiveAm: UcpRequest): Unit = {}

      override def onSuccess(am: UCXActiveMessage, buff: TransportBuffer): Unit = {
        val tx = createTransaction
        tx.start(UCXTransactionType.Receive, 1, cb)
        tx.completeWithSuccess(RequestType.BufferReceive, Some(currentAlt.tag), None)
      }

      override def onCancel(am: UCXActiveMessage): Unit = {}

      override def onMessageReceived(size: Long): TransportBuffer = {
        currentAlt = bufferReceiveState.next()
        require(currentAlt.length == size)
        new CudfTransportBuffer(currentAlt.memoryBuffer.get)
      }
    }

    // we register a specific handler callback for a tag
    headers.foreach(header => ucx.registerResponseHandler(amId, header, amCallback))
  }
}

class UCXConnection(peerExecutorId: Int, val ucx: UCX) extends Connection with Logging {
  // alternate constructor for a server connection (-1 because it is not to a peer)
  def this(ucx: UCX) = this(-1, ucx)

  private[this] val pendingTransactions = new ConcurrentHashMap[Long, UCXTransaction]()

  private[ucx] def send(executorId: Long, alt: AddressLengthTag,
    ucxCallback: UCXTagCallback): Unit = {
    val sendRange = new NvtxRange("Connection Send", NvtxColor.PURPLE)
    try {
      ucx.send(executorId, alt, ucxCallback)
    } finally {
      sendRange.close()
    }
  }

  def send(sendPeerExecutorId: Long,
           header: AddressLengthTag,
           buffers: Seq[AddressLengthTag],
           cb: TransactionCallback): Transaction = {
    val tx = createTransaction
    buffers.foreach (alt => tx.registerForSend(alt))

    val numMessages = if (header != null) buffers.size + 1 else buffers.size


    tx.start(UCXTransactionType.Send, numMessages, cb)

    val ucxCallback = new UCXTagCallback {
      override def onError(alt: AddressLengthTag, ucsStatus: Int, errorMsg: String): Unit = {
        tx.handleTagError(alt.tag)
        logError(s"Error sending: $errorMsg, tx: $tx")
        tx.completeWithError(errorMsg)
      }

      override def onSuccess(alt: AddressLengthTag): Unit = {
        logDebug(s"Successful send: ${TransportUtils.toHex(alt.tag)}, tx = $tx")
        tx.handleTagCompleted(alt.tag)
        if (tx.decrementPendingAndGet <= 0) {
          tx.complete(TransactionStatus.Success)
        }
      }

      override def onCancel(alt: AddressLengthTag): Unit = {
        tx.handleTagCancelled(alt.tag)
        tx.complete(TransactionStatus.Cancelled)
      }

      override def onMessageStarted(ucxMessage: UcpRequest): Unit = {
        tx.registerPendingMessage(ucxMessage)
      }
    }

    // send the header
    if (header != null) {
      logDebug(s"Sending meta [executor_id=$sendPeerExecutorId, " +
        s"tag=${TransportUtils.toHex(header.tag)}, size=${header.length}]")
      send(sendPeerExecutorId, header, ucxCallback)
      tx.incrementSendSize(header.length)
    }

    buffers.foreach { alt =>
      logDebug(s"Sending [executor_id=$sendPeerExecutorId, " +
        s"tag=${TransportUtils.toHex(alt.tag)}, size=${alt.length}]")
      send(sendPeerExecutorId, alt, ucxCallback)
      tx.incrementSendSize(alt.length)
    }

    tx
  }

  private[ucx] def cancel(msg: UcpRequest): Unit =
    ucx.cancel(msg)

  private[ucx] def createTransaction: UCXTransaction = {
    val thisTxId = ucx.getNextTransactionId

    val tx = new UCXTransaction(this, thisTxId)

    pendingTransactions.put(thisTxId, tx)
    logDebug(s"PENDING TRANSACTIONS AFTER ADD $pendingTransactions")
    tx
  }

  def removeTransaction(tx: UCXTransaction): Unit = {
    pendingTransactions.remove(tx.txId)
    logDebug(s"PENDING TRANSACTIONS AFTER REMOVE $pendingTransactions")
  }

  override def toString: String = {
    s"Connection(ucx=$ucx, peerExecutorId=$peerExecutorId) "
  }
}

object UCXConnection extends Logging {
  /**
   * 1) client gets upper 28 bits
   * 2) then comes the type, which gets 4 bits
   * 3) the remaining 32 bits are used for buffer specific tags
   */
  private final val bufferMsgType: Long = 0x0000000000000000BL

  // message type mask for UCX tags. The only message type using tags
  // is `bufferMsgType`
  private final val msgTypeMask: Long   = 0x000000000000000FL

  // UCX Active Message masks (we can use up to 5 bits for these ids)
  private final val amRequestMask: Int  = 0x0000000F
  private final val amResponseMask: Int = 0x0000001F

  // We pick the 5th bit set to 1 as a "response" active message
  private final val amResponseFlag: Int = 0x00000010

  // pick up the lower and upper parts of a long
  private final val lowerBitsMask: Long = 0x00000000FFFFFFFFL
  private final val upperBitsMask: Long = 0xFFFFFFFF00000000L

  def composeBufferTag(peerClientId: Long, bufferTag: Long): Long = {
    // buffer tags are [peerClientId, 0, bufferTag]
    composeTag(composeUpperBits(peerClientId, bufferMsgType), bufferTag)
  }

  def composeRequestAmId(requestType: RequestType.Value): Int = {
    val amId = requestType.id
    if ((amId & amRequestMask) != amId) {
      throw new IllegalArgumentException(
        s"Invalid request amId, it must be 4 bits: ${TransportUtils.toHex(amId)}")
    }
    amId
  }

  def composeResponseAmId(requestType: RequestType.Value): Int = {
    val amId = amResponseFlag | composeRequestAmId(requestType)
    if ((amId & amResponseMask) != amId) {
      throw new IllegalArgumentException(
        s"Invalid response amId, it must be 5 bits: ${TransportUtils.toHex(amId)}")
    }
    amId
  }

  def composeTag(upperBits: Long, lowerBits: Long): Long = {
    if ((upperBits & upperBitsMask) != upperBits) {
      throw new IllegalArgumentException(
        s"Invalid tag, upperBits would alias: ${TransportUtils.toHex(upperBits)}")
    }
    // the lower 32bits aliasing is not a big deal, we expect it with msg rollover
    // so we don't check for it
    upperBits | (lowerBits & lowerBitsMask)
  }

  private def composeUpperBits(peerClientId: Long, msgType: Long): Long = {
    if ((peerClientId & lowerBitsMask) != peerClientId) {
      throw new IllegalArgumentException(
        s"Invalid tag, peerClientId would alias: ${TransportUtils.toHex(peerClientId)}")
    }
    if ((msgType & msgTypeMask) != msgType) {
      throw new IllegalArgumentException(
        s"Invalid tag, msgType would alias: ${TransportUtils.toHex(msgType)}")
    }
    (peerClientId << 36) | (msgType << 32)
  }

  def composeRequestHeader(executorId: Long, txId: Long): Long = {
    require(executorId >= 0,
      s"Attempted to pack negative $executorId")
    require((executorId & lowerBitsMask) == executorId,
        s"ExecutorId would alias: ${TransportUtils.toHex(executorId)}")
    composeTag(executorId << 32, txId)
  }

  def extractExecutorId(header: Long): Long = {
    (header >> 32) & lowerBitsMask
  }
  //
  // Handshake message code. This, I expect, could be folded into the [[BlockManagerId]],
  // but I have not tried this. If we did, it would eliminate the extra TCP connection
  // in this class.
  //
  private def readBytesFromStream(direct: Boolean,
    is: InputStream, lengthToRead: Int): ByteBuffer = {
    var bytesRead = 0
    var read = 0
    val buff = new Array[Byte](lengthToRead)
    while (read >= 0 && bytesRead < lengthToRead) {
      logTrace(s"Reading ${lengthToRead}. Currently at ${bytesRead}")
      read = is.read(buff, bytesRead, lengthToRead - bytesRead)
      if (read > 0) {
        bytesRead = bytesRead + read
      }
    }

    if (bytesRead < lengthToRead) {
      throw new IllegalStateException("Read less bytes than expected!")
    }

    val byteBuffer = ByteBuffer.wrap(buff)

    if (direct) {
      // a direct buffer does not allow .array() (not implemented).
      // therefore we received in the JVM heap, and now we copy to the native heap
      // using a .put on that native buffer
      val directCopy = ByteBuffer.allocateDirect(lengthToRead)
      TransportUtils.copyBuffer(byteBuffer, directCopy, lengthToRead)
      directCopy.rewind() // reset position
      directCopy
    } else {
      byteBuffer
    }
  }

  /**
   * Given a java `InputStream`, obtain the peer's `WorkerAddress` and executor id,
   * returning them as a pair.
   *
   * @param is management port input stream
   * @return a tuple of worker address, the peer executor id, and rkeys
   */
  def readHandshakeHeader(is: InputStream): (WorkerAddress, Int, Rkeys)  = {
    val maxLen = 1024 * 1024

    // get the length from the stream, it's the first thing sent.
    val workerAddressLength = readBytesFromStream(false, is, 4).getInt()

    require(workerAddressLength <= maxLen,
      s"Received an abnormally large (>$maxLen Bytes) WorkerAddress " +
        s"(${workerAddressLength} Bytes), dropping.")

    val workerAddress = readBytesFromStream(true, is, workerAddressLength)

    // get the remote executor Id, that's the last part of the handshake
    val executorId = readBytesFromStream(false, is, 4).getInt()

    // get the number of rkeys expected next
    val numRkeys = readBytesFromStream(false, is, 4).getInt()

    val rkeys = new ArrayBuffer[ByteBuffer](numRkeys)
    (0 until numRkeys).foreach { _ =>
      val size = readBytesFromStream(false, is, 4).getInt()
      rkeys.append(readBytesFromStream(true, is, size))
    }

    (WorkerAddress(workerAddress), executorId, Rkeys(rkeys))
  }

  /**
   * Writes a header that is exchanged in the management port. The header contains:
   *  - UCP Worker address length (4 bytes)
   *  - UCP Worker address (variable length)
   *  - Local executor id (4 bytes)
   *  - Local rkeys count (4 bytes)
   *  - Per rkey: rkey length (4 bytes) + rkey payload (variable)
   *
   * @param os OutputStream to write to
   * @param workerAddress byte buffer that holds the local UCX worker address
   * @param localExecutorId The local executorId
   * @param rkeys The local rkeys to send to the peer
   */
  def writeHandshakeHeader(os: OutputStream,
                           workerAddress: ByteBuffer,
                           localExecutorId: Int,
                           rkeys: Seq[ByteBuffer]): Unit = {
    val headerSize = 4 + workerAddress.remaining() + 4 +
        4 + (4 * rkeys.size) + (rkeys.map(_.capacity).sum)
    val hsBuff = ByteBuffer.allocate(headerSize)

    // pack the worker address
    hsBuff.putInt(workerAddress.capacity)
    hsBuff.put(workerAddress)

    // send the executor id
    hsBuff.putInt(localExecutorId)

    // pack the rkeys
    hsBuff.putInt(rkeys.size)
    rkeys.foreach { rkey =>
      hsBuff.putInt(rkey.capacity)
      hsBuff.put(rkey)
    }
    hsBuff.flip()
    os.write(hsBuff.array)
    os.flush()
  }
}
