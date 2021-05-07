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
import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.shuffle
import com.nvidia.spark.rapids.shuffle.ucx.UCX.AmCallback
import com.nvidia.spark.rapids.shuffle.{RequestType, _}
import org.openucx.jucx.ucp.{UcpAmData, UcpEndpoint, UcpRequest}
import org.apache.spark.internal.Logging
import org.openucx.jucx.UcxCallback

/**
 * This is a private api used within the ucx package.
 * It is used by [[Transaction]] to call into the UCX functions. It adds the tag
 * as we use that to track the message and for debugging.
 */
private[ucx] abstract class UCXTagCallback {
  def onError(alt: AddressLengthTag, ucsStatus: Int, errorMsg: String): Unit
  def onMessageStarted(ucxMessage: UcpRequest): Unit
  def onSuccess(alt: AddressLengthTag): Unit
  def onCancel(alt: AddressLengthTag): Unit
}

trait UCXAmCallback {
  def onHostMessageReceived(size: Long): ByteBuffer
  def onSuccess(id: Option[Long], buff: RefCountedDirectByteBuffer): Unit
}

class UCXServerConnection(ucx: UCX) extends UCXConnection(ucx) with ServerConnection {
  override def startManagementPort(host: String): Int = {
    ucx.startManagementPort(host)
  }

  override def send(sendPeerExecutorId: Long, bounceBuffers: Seq[AddressLengthTag],
    cb: TransactionCallback): Transaction =
    send(sendPeerExecutorId, null, bounceBuffers, cb)

  override def send(sendPeerExecutorId: Long, header: AddressLengthTag,
    cb: TransactionCallback): Transaction =
    send(sendPeerExecutorId, header, Seq.empty, cb)

  override def registerRequestHandler(
      requestType: RequestType.Value, cb: TransactionCallback): Unit = {

    ucx.registerRequestHandler(composeRequestAmId(requestType), new UCXAmCallback {
      override def onSuccess(hdr: Option[Long], buff: RefCountedDirectByteBuffer): Unit = {
        logDebug(s"At requestHandler for ${requestType} and header " +
          s"${TransportUtils.formatTag(hdr.get)}")
        val tx = createTransaction
        tx.start(UCXTransactionType.Request, 1, cb)
        tx.setHeader(hdr)
        tx.setMessage(buff)
        tx.setMessageType(requestType)
        //TODO: do we care/want the responseEp? tx.setResponseEndpoint(responseEp)
        tx.txCallback(TransactionStatus.Success)
      }

      override def onHostMessageReceived(size: Long): ByteBuffer = {
        ByteBuffer.allocateDirect(size.toInt)
      }
    })
  }

}

class UCXClientConnection(peerExecutorId: Int, peerClientId: Long, ucx: UCX)
  extends UCXConnection(peerExecutorId, ucx)
  with ClientConnection {

  override def toString: String = {
    s"UCXClientConnection(ucx=$ucx, " +
      s"peerExecutorId=$peerExecutorId, " +
      s"peerClientId=$peerClientId) "
  }

  logInfo(s"UCX Client $this started")

  // tag used for a unique response to a request initiated by [[Transaction.request]]
  override def assignResponseTag: Long = composeResponseTag(peerClientId, ucx.assignResponseTag())

  override def assignBufferTag(msgId: Int): Long = composeBufferTag(peerClientId, msgId)

  override def getPeerExecutorId: Long = peerExecutorId

  private val responseHandlers = new ConcurrentHashMap[RequestType.Value, TransactionCallback]()

  def registerResponseHandler(requestType: RequestType.Value, cb: TransactionCallback)
                             (id: Option[Long], resp: RefCountedDirectByteBuffer): Unit = {
    val tx = createTransaction
    tx.start(UCXTransactionType.Request, 1, cb)
    tx.setHeader(id)
    tx.setMessage(resp)
    tx.txCallback(TransactionStatus.Success)
  }

  override def request(requestType: RequestType.Value,
                       request: ByteBuffer,
                       cb: TransactionCallback): Transaction = {
    val tx = createTransaction
    // register if we haven't already

    // my local executorId
    val hdr = composeTag((ucx.getExecutorId.toLong << 32), tx.txId)

    val responseAmId = composeResponseAmId(requestType)
    ucx.registerResponseHandler(responseAmId, peerExecutorId, hdr,
      registerResponseHandler(requestType, cb))

    tx.start(UCXTransactionType.Request, 1, cb)

    logDebug(s"Performing a ${requestType} request $request for tx ${tx} " +
      s"and header ${TransportUtils.formatTag(hdr)}")

    ucx.sendAm(
      peerExecutorId,
      hdr,
      requestType.id, // peer request amId
      TransportUtils.getAddress(request),
      request.remaining(),
      new UcxCallback {
        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
          //tx.handleRequestError()
        }

        override def onSuccess(request: UcpRequest): Unit = {
          //tx.handleRequestSent()
        }
      })

    tx
  }
}

class UCXConnection(peerExecutorId: Int, val ucx: UCX) extends Connection with Logging {
  // alternate constructor for a server connection (-1 because it is not to a peer)
  def this(ucx: UCX) = this(-1, ucx)

  private[this] val pendingTransactions = new ConcurrentHashMap[Long, UCXTransaction]()

  /**
   * 1) client gets upper 28 bits
   * 2) then comes the type, which gets 4 bits
   * 3) the remaining 32 bits are used for buffer specific tags
   */
  private val requestMsgType:  Long = 0x00000000000000000L
  private val responseMsgType: Long = 0x0000000000000000AL
  private val bufferMsgType:   Long = 0x0000000000000000BL

  override def composeRequestTag(requestType: RequestType.Value): Long = {
    val requestTypeId = requestType match {
      case RequestType.MetadataRequest => 0
      case RequestType.TransferRequest => 1
    }
    requestMsgType | requestTypeId
  }

  protected def composeResponseTag(peerClientId: Long, bufferTag: Long): Long = {
    // response tags are [peerClientId, 1, bufferTag]
    composeTag(composeUpperBits(peerClientId, responseMsgType), bufferTag)
  }

  protected def composeBufferTag(peerClientId: Long, bufferTag: Long): Long = {
    // buffer tags are [peerClientId, 0, bufferTag]
    composeTag(composeUpperBits(peerClientId, bufferMsgType), bufferTag)
  }

  protected def composeRequestAmId(requestType: RequestType.Value): Int = {
    val amId = requestType.id
    if ((amId & 0x0000000F) != amId) {
      throw new IllegalArgumentException(
        s"Invalid request amId, it must be 4 bits: ${TransportUtils.formatTag(amId)}")
    }
    amId
  }

  def composeResponseAmId(requestType: RequestType.Value): Int = {
    val amId = (0x00000010) | composeRequestAmId(requestType)
    if ((amId & 0x0000001F) != amId) {
      throw new IllegalArgumentException(
        s"Invalid response amId, it must be 5 bits: ${TransportUtils.formatTag(amId)}")
    }
    amId
  }

  def composeTag(upperBits: Long, lowerBits: Long): Long = {
    if ((upperBits & 0xFFFFFFFF00000000L) != upperBits) {
      throw new IllegalArgumentException(
        s"Invalid tag, upperBits would alias: ${TransportUtils.formatTag(upperBits)}")
    }
    // the lower 32bits aliasing is not a big deal, we expect it with msg rollover
    // so we don't check for it
    upperBits | (lowerBits & 0x00000000FFFFFFFFL)
  }

  def composeUpperBits(peerClientId: Long, msgType: Long): Long = {
    if ((peerClientId & 0x000000000FFFFFFFL) != peerClientId) {
      throw new IllegalArgumentException(
        s"Invalid tag, peerClientId would alias: ${TransportUtils.formatTag(peerClientId)}")
    }
    if ((msgType & 0x000000000000000FL) != msgType) {
      throw new IllegalArgumentException(
        s"Invalid tag, msgType would alias: ${TransportUtils.formatTag(msgType)}")
    }
    (peerClientId << 36) | (msgType << 32)
  }

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
        tx.handleTagError(alt.tag, errorMsg)
        logError(s"Error sending: $errorMsg, tx: $tx")
        tx.txCallback(TransactionStatus.Error)
      }

      override def onSuccess(alt: AddressLengthTag): Unit = {
        logDebug(s"Successful send: ${TransportUtils.formatTag(alt.tag)}, tx = $tx")
        tx.handleTagCompleted(alt.tag)
        if (tx.decrementPendingAndGet <= 0) {
          tx.txCallback(TransactionStatus.Success)
        }
      }

      override def onCancel(alt: AddressLengthTag): Unit = {
        tx.handleTagCancelled(alt.tag)
        tx.txCallback(TransactionStatus.Cancelled)
      }

      override def onMessageStarted(ucxMessage: UcpRequest): Unit = {
        tx.registerPendingMessage(ucxMessage)
      }
    }

    // send the header
    if (header != null) {
      logDebug(s"Sending meta [executor_id=$sendPeerExecutorId, " +
        s"tag=${TransportUtils.formatTag(header.tag)}, size=${header.length}]")
      send(sendPeerExecutorId, header, ucxCallback)
      tx.incrementSendSize(header.length)
    }

    buffers.foreach { alt =>
      logDebug(s"Sending [executor_id=$sendPeerExecutorId, " +
        s"tag=${TransportUtils.formatTag(alt.tag)}, size=${alt.length}]")
      send(sendPeerExecutorId, alt, ucxCallback)
      tx.incrementSendSize(alt.length)
    }

    tx
  }

  override def receive(alt: AddressLengthTag, cb: TransactionCallback): Transaction =  {
    val tx = createTransaction
    tx.registerForReceive(alt)
    tx.start(UCXTransactionType.Receive, 1, cb)
    val ucxCallback = new UCXTagCallback {
      override def onError(alt: AddressLengthTag, ucsStatus: Int, errorMsg: String): Unit = {
        logError(s"Got an error... for tag: ${TransportUtils.formatTag(alt.tag)}, tx $tx")
        tx.handleTagError(alt.tag, errorMsg)
        tx.txCallback(TransactionStatus.Error)
      }

      override def onSuccess(alt: AddressLengthTag): Unit = {
        logDebug(s"Successful receive: ${TransportUtils.formatTag(alt.tag)}, tx $tx")
        tx.handleTagCompleted(alt.tag)
        if (tx.decrementPendingAndGet <= 0) {
          logDebug(s"Receive done for tag: ${TransportUtils.formatTag(alt.tag)}, tx $tx")
          tx.txCallback(TransactionStatus.Success)
        }
      }

      override def onCancel(alt: AddressLengthTag): Unit = {
        tx.handleTagCancelled(alt.tag)
        tx.txCallback(TransactionStatus.Cancelled)
      }

      override def onMessageStarted(ucxMessage: UcpRequest): Unit = {
        tx.registerPendingMessage(ucxMessage)
      }
    }

    logDebug(s"Receiving [tag=${TransportUtils.formatTag(alt.tag)}, size=${alt.length}]")
    ucx.receive(alt, ucxCallback)
    tx.incrementReceiveSize(alt.length)
    tx
  }

  override def respond(peerExecutorId: Long, amId: Int, header: Long, response: ByteBuffer,
                       cb: TransactionCallback): Transaction = {
    val tx = createTransaction
    tx.start(UCXTransactionType.Request, 1, cb)

    logDebug(s"Responding to ${peerExecutorId} at ${TransportUtils.formatTag(header)} " +
      s"with ${response}")

    ucx.sendAm(peerExecutorId,
      header,
      amId,
      TransportUtils.getAddress(response),
      response.remaining(),
      new UcxCallback {
        override def onSuccess(request: UcpRequest): Unit = {
          logDebug(s"AM success respond")
          tx.txCallback(TransactionStatus.Success)
        }

        override def onError(ucsStatus: Int, errorMsg: String): Unit = {
          logError(s"AM Error responding ${ucsStatus} ${errorMsg}")
        }
      })
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
