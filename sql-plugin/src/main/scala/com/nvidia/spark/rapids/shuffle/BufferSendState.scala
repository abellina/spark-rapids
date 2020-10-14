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

import java.util.concurrent.Executor

import ai.rapids.cudf.{Cuda, CudaUtil, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.{Arm, RapidsBuffer, ShuffleMetadata}
import com.nvidia.spark.rapids.format.{BufferMeta, BufferTransferRequest, TransferRequest}

import org.apache.spark.internal.Logging

/**
 * A helper case class to maintain the state associated with a transfer request initiated by
 * a `TransferRequest` metadata message.
 *
 * The class implements the Iterator interface.
 *
 * On next(), a set of RapidsBuffer are copied onto a bounce buffer, and the
 * `AddressLengthTag` of the bounce buffer is returned. By convention, the tag used
 * is that of the first buffer contained in the payload. Buffers are copied to the bounce
 * buffer in TransferRequest order. The receiver has the same conventions.
 *
 * It also is AutoCloseable. close() should be called to free bounce buffers.
 *
 * In terms of the lifecycle of this object, it begins with the client asking for transfers to
 * start, it lasts through all buffers being transmitted, and ultimately finishes when a
 * TransferResponse is sent back to the client.
 *
 * @param request a transfer request
 * @note this class is not thread safe
 */
class BufferSendState(
    request: RefCountedDirectByteBuffer,
    sendBounceBuffers: SendBounceBuffers,
    requestHandler: RapidsShuffleRequestHandler,
    serverStream: Cuda.Stream = Cuda.DEFAULT_STREAM)
    extends Iterator[AddressLengthTag] with AutoCloseable with Logging with Arm {

  class SendBlock(val bufferTransferRequest: BufferTransferRequest,
      tableSize: Long) extends BlockWithSize {
    override def size: Long = tableSize
    def tag: Long = bufferTransferRequest.tag()
  }

  private[this] val transferRequest = ShuffleMetadata.getTransferRequest(request.getBuffer())
  private[this] var bufferMetas = Seq[BufferMeta]()
  private[this] var isClosed = false

  val blocksToSend: Seq[SendBlock] = (0 until transferRequest.requestsLength()).map { btr =>
    val bufferTransferRequest = transferRequest.requests(btr)
    withResource(requestHandler.acquireShuffleBuffer(bufferTransferRequest.bufferId())) { table =>
      bufferMetas = bufferMetas :+ table.meta.bufferMeta()
      new SendBlock(bufferTransferRequest, table.size)
    }
  }

  private[this] val windowedBlockIterator =
    new WindowedBlockIterator[SendBlock](blocksToSend, sendBounceBuffers.bounceBufferSize)

  var markedDone = false

  private[this] var bounceBuffer: BounceBuffer = null
  private[this] var hostBounceBuffer: BounceBuffer = null

  def getTransferRequest: TransferRequest = synchronized {
    transferRequest
  }

  def hasNext: Boolean = !markedDone

  /**
   * Used to pop a [[BufferSendState]] from its queue if and only if there are bounce
   * buffers available
   * @return true if bounce buffers are available to proceed
   */
  def acquireBounceBuffersNonBlocking: Boolean = synchronized {
    if (bounceBuffer == null) {
      // TODO: maybe take % of buffers in device vs those in the host
      val bounceBuffers = transport.tryGetSendBounceBuffers(
        true,
        bounceBufferSize,
        1)
      if (bounceBuffers.nonEmpty) {
        bounceBuffer = bounceBuffers.head
      }
    }

    if (bounceBuffer != null && hostBounceBuffer == null) {
      val hostBounceBuffers = transport.tryGetSendBounceBuffers(
        false,
        bounceBufferSize,
        1)
      if (hostBounceBuffers.nonEmpty) {
        hostBounceBuffer = hostBounceBuffers.head
      }
    }
    bounceBuffer != null
  }

  private[this] def freeBounceBuffers(): Unit = {
    if (bounceBuffer != null) {
      bounceBuffer.close()
      bounceBuffer = null
    }

    if (hostBounceBuffer != null) {
      hostBounceBuffer.close()
      hostBounceBuffer = null
    }
  }

  def getTransferResponse(): RefCountedDirectByteBuffer = synchronized {
    new RefCountedDirectByteBuffer(ShuffleMetadata.buildBufferTransferResponse(bufferMetas))
  }

  override def close(): Unit = synchronized {
    require(markedDone)
    if (isClosed){
      throw new IllegalStateException("ALREADY CLOSED!")
    }
    isClosed = true
    freeBounceBuffers()
    request.close()
  }

  /**
   * This function returns bounce buffers that are ready to be sent. To get there,
   * it will:
   *   1) acquire the bounce buffers in the first place (if it hasn't already)
   *   2) copy data from the source buffer to the bounce buffers, updating the offset accordingly
   *   3) return either the full set of bounce buffers, or a subset, depending on how much is
   *      left to send.
   * @return bounce buffers ready to be sent.
   */
  var firstTime = true
  var blockRanges: Seq[BlockRange[SendBlock]] = Seq.empty

  def next(): AddressLengthTag = synchronized {
    if (firstTime) {
      blockRanges = windowedBlockIterator.next()
    }

    logDebug(s"Using window $windowedBlockIterator")

    firstTime = false

    var bounceBuffToUse: MemoryBuffer = null
    var buffOffset = 0L

    val buffsToSend = try {
      if (!markedDone) {
        var deviceBuffs = 0L
        var hostBuffs = 0L
        val acquiredBuffs = blockRanges.map { blockRange =>
          val transferRequest =
            blockRange.block.bufferTransferRequest

          val rapidsBuffer = requestHandler.acquireShuffleBuffer(transferRequest.bufferId())
          val memBuff = rapidsBuffer.getMemoryBuffer
          var shouldClose = false
          if (memBuff.isInstanceOf[DeviceMemoryBuffer]) {
            deviceBuffs += memBuff.getLength
            shouldClose = true
          } else {
            hostBuffs += memBuff.getLength
          }
          (blockRange, rapidsBuffer, memBuff)
        }

        logDebug(s"DEVICE occupancy for bounce buffer is d=${deviceBuffs} vs h=${hostBuffs}")

        bounceBuffToUse = if (deviceBuffs >= hostBuffs || hostBounceBuffer == null) {
          bounceBuffer.buffer
        } else {
          hostBounceBuffer.buffer
        }

        acquiredBuffs.foreach { case (blockRange, rapidsBuffer, memBuff) =>
          require(blockRange.rangeSize() <= bounceBuffToUse.getLength - buffOffset)

          bounceBuffToUse match {
            case _: HostMemoryBuffer =>
              CudaUtil.copy(
                memBuff,
                blockRange.rangeStart,
                bounceBuffToUse,
                buffOffset,
                blockRange.rangeSize())
            case d: DeviceMemoryBuffer =>
              memBuff match {
                case mh: HostMemoryBuffer =>
                  d.copyFromHostBufferAsync(
                    buffOffset,
                    mh,
                    blockRange.rangeStart,
                    blockRange.rangeSize(),
                    serverStream)
                case md: DeviceMemoryBuffer =>
                  d.copyFromDeviceBufferAsync(
                    buffOffset,
                    md,
                    blockRange.rangeStart,
                    blockRange.rangeSize(),
                    serverStream)
                case _ => throw new IllegalStateException("What buffer is this")
              }
            case _ => throw new IllegalStateException("What buffer is this")
          }

          memBuff.close()
          rapidsBuffer.close()
          buffOffset += blockRange.rangeSize()
        }

        val alt = AddressLengthTag.from(bounceBuffToUse,
          blockRanges.head.block.tag)

        alt.resetLength(buffOffset)

        if (windowedBlockIterator.hasNext) {
          blockRanges = windowedBlockIterator.next()
        } else {
          blockRanges = Seq.empty
          markedDone = true
        }

        alt
      } else {
        null
      }
    } catch {
      case t: Throwable =>
        logError("Error while copying to bounce buffers on send.", t)
        throw t
    }

    logDebug(s"Sending ${buffsToSend} for transfer request, " +
        s" [peer_executor_id=${transferRequest.executorId()}]")

    buffsToSend
  }
}