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

import ai.rapids.cudf.{Cuda, CudaUtil, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.{Arm, RapidsBuffer, ShuffleMetadata, StorageTier}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
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

  private[this] val bounceBuffer: BounceBuffer =
    sendBounceBuffers.deviceBounceBuffer
  private[this] val hostBounceBuffer: BounceBuffer =
    sendBounceBuffers.hostBounceBuffer.orNull

  def getTransferRequest: TransferRequest = synchronized {
    transferRequest
  }

  def hasNext: Boolean = !markedDone

  private[this] def freeBounceBuffers(): Unit = {
    sendBounceBuffers.close()
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

  case class RangeBuffer(
      range: BlockRange[SendBlock], rapidsBuffer: RapidsBuffer)
      extends AutoCloseable {
    override def close(): Unit = {
      rapidsBuffer.close()
    }
  }

  var acquiredBuffs: Seq[RangeBuffer] = Seq.empty

  def next(): AddressLengthTag = synchronized {
    if (firstTime) {
      blockRanges = windowedBlockIterator.next()
    }

    require(acquiredBuffs.isEmpty,
      "Called next without calling `releaseAcquiredToCatalog` first")

    logDebug(s"Using window $windowedBlockIterator")

    firstTime = false

    var bounceBuffToUse: MemoryBuffer = null
    var buffOffset = 0L

    val buffsToSend = try {
      if (!markedDone) {
        var deviceBuffs = 0L
        var hostBuffs = 0L
        acquiredBuffs = blockRanges.safeMap { blockRange =>
          val transferRequest =
            blockRange.block.bufferTransferRequest

          // we acquire these buffers now, and keep them until the caller releases them
          // using `releaseAcquiredToCatalog`
          closeOnExcept(
            requestHandler.acquireShuffleBuffer(transferRequest.bufferId())) { rapidsBuffer =>
            //these are closed later, after we synchronize streams
            rapidsBuffer.storageTier match {
              case StorageTier.DEVICE =>
                deviceBuffs += blockRange.rangeSize()
              case _ => // host/disk
                hostBuffs += blockRange.rangeSize()
            }
            RangeBuffer(blockRange, rapidsBuffer)
          }
        }

        logDebug(s"Occupancy for bounce buffer is [device=${deviceBuffs}, host=${hostBuffs}] Bytes")

        bounceBuffToUse = if (deviceBuffs >= hostBuffs || hostBounceBuffer == null) {
          bounceBuffer.buffer
        } else {
          hostBounceBuffer.buffer
        }

        acquiredBuffs.foreach { case RangeBuffer(blockRange, rapidsBuffer) =>
          require(blockRange.rangeSize() <= bounceBuffToUse.getLength - buffOffset)
          withResource(rapidsBuffer.getMemoryBuffer) { memBuff =>
            bounceBuffToUse match {
              case _: HostMemoryBuffer =>
                //TODO: HostMemoryBuffer needs the same functionality that
                // DeviceMemoryBuffer has to copy from/to device/host buffers
                CudaUtil.copy(
                  memBuff,
                  blockRange.rangeStart,
                  bounceBuffToUse,
                  buffOffset,
                  blockRange.rangeSize())
              case d: DeviceMemoryBuffer =>
                memBuff match {
                  case mh: HostMemoryBuffer =>
                    // host original => device bounce
                    d.copyFromHostBufferAsync(buffOffset, mh, blockRange.rangeStart,
                      blockRange.rangeSize(), serverStream)
                  case md: DeviceMemoryBuffer =>
                    // device original => device bounce
                    d.copyFromDeviceBufferAsync(buffOffset, md, blockRange.rangeStart,
                      blockRange.rangeSize(), serverStream)
                  case _ => throw new IllegalStateException("What buffer is this")
                }
              case _ => throw new IllegalStateException("What buffer is this")
            }
          }
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

  /**
   * Called by the RapidsShuffleServer when it has synchronized with its stream,
   * allowing us to safely return buffers to the catalog to be potentially freed if spilling.
   */
  def releaseAcquiredToCatalog(): Unit = {
    require(acquiredBuffs.nonEmpty, "Told to close rapids buffers, but nothing was acquired")
    acquiredBuffs.foreach(_.close())
    acquiredBuffs= Seq.empty
  }
}