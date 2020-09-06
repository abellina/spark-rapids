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
import com.nvidia.spark.rapids.{RapidsBuffer, ShuffleMetadata}
import com.nvidia.spark.rapids.format.{BufferMeta, BufferTransferRequest, TransferRequest}

import org.apache.spark.internal.Logging

/**
 * A helper case class to maintain the state associated with a transfer request initiated by
 * a `TransferRequest` metadata message.
 *
 * This class is *not thread safe*. The way the code is currently designed, bounce buffers
 * being used to send, or copied to, are acted on a sequential basis, in time and in space.
 *
 * Callers use this class, like so:
 *
 * 1) [[getBuffersToSend]]: is used to get bounce buffers that the server should .send on.
 *    -- first time:
 *          a) the corresponding catalog table is acquired,
 *          b) bounce buffers are acquired,
 *          c) data is copied from the original catalog table into the bounce buffers available
 *          d) the length of the last bounce buffer is adjusted if it would satisfy the full
 *             length of the catalog-backed buffer.
 *          e) bounce buffers are returned
 *
 *    -- subsequent times:
 *      if we are not done sending the acquired table:
 *          a) data is copied from the original catalog table into the bounce buffers available
 *             at sequentially incrementing offsets.
 *          b) the length of the last bounce buffer is adjusted if it would satisfy the full
 *             length of the catalog-backed buffer.
 *          c) bounce buffers are returned
 *
 * 2) [[close]]: used to free state as the [[BufferSendState]] object is no longer needed
 *
 * In terms of the lifecycle of this object, it begins with the client asking for transfers to
 * start, it lasts through all buffers being transmitted, and ultimately finishes when a
 * `TransferResponse` is sent back to the client.
 *
 * @param tx the original `Transaction` from the `TransferRequest`.
 * @param request a transfer request
 */
class BufferSendState(
    tx: Transaction,
    request: RefCountedDirectByteBuffer,
    transport: RapidsShuffleTransport,
    bssExec: Executor,
    requestHandler: RapidsShuffleRequestHandler,
    bounceBufferSize: Long,
    serverStream: Cuda.Stream = Cuda.DEFAULT_STREAM)
    extends Iterator[AddressLengthTag] with AutoCloseable with Logging {

  class SendBlock(val bufferTransferRequest: BufferTransferRequest,
      tableSize: Long) extends BlockWithSize {
    override def size: Long = tableSize
    def tag: Long = bufferTransferRequest.tag()
  }

  private[this] var acquiredAlts: Seq[AddressLengthTag] = Seq.empty
  private[this] val transferRequest = ShuffleMetadata.getTransferRequest(request.getBuffer())
  private[this] var bufferMetas = Seq[BufferMeta]()
  private[this] var isClosed = false

  def getTransferRequest(): TransferRequest = synchronized {
    transferRequest
  }

  val blocksToSend: Seq[SendBlock] = (0 until transferRequest.requestsLength()).map { btr =>
    val bufferTransferRequest = transferRequest.requests(btr)
    val table = requestHandler.acquireShuffleBuffer(bufferTransferRequest.bufferId())
    val transferBlock = new SendBlock(
      bufferTransferRequest, table.size)
    table.close()
    transferBlock
  }

  val windowedBlockIterator = new WindowedBlockIterator[SendBlock](blocksToSend, bounceBufferSize)

  var markedDone = false

  def hasNext: Boolean = !markedDone

  // hold acquired tables and buffers, while a transfer occurs
  private[this] var acquiredTables: Seq[RapidsBuffer] = Seq.empty

  // the set of buffers we will acquire and use to work the entirety of this transfer.
  private[this] var bounceBuffer: MemoryBuffer = null
  private[this] var hostBounceBuffer: MemoryBuffer = null

  /**
   * Used to pop a [[BufferSendState]] from its queue if and only if there are bounce
   * buffers available
   * @return true if bounce buffers are available to proceed
   */
  def acquireBounceBuffersNonBlocking: Boolean = synchronized {
    if (bounceBuffer == null) {
      // TODO: maybe take % of buffers in device vs those in the host
      val useDevBuffer = true
      val bounceBuffers = transport.tryGetSendBounceBuffers(
        useDevBuffer,
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
    // else, we may want to make acquisition of the table and state separate so
    // the act of acquiring the table from the catalog and getting the bounce buffer
    // doesn't affect the state in [[BufferSendState]], this is in the case where we are at
    // the limit, and we want to spill everything in a tier (including the one buffer
    // we are trying to pop from the [[BufferSendState]] queue)
    bounceBuffer != null
  }

  private[this] def freeBounceBuffers(): Unit = {
    if (bounceBuffer != null) {
      transport.freeSendBounceBuffers(Seq(bounceBuffer))
      bounceBuffer = null
    }

    if (hostBounceBuffer != null) {
      transport.freeSendBounceBuffers(Seq(hostBounceBuffer))
      hostBounceBuffer = null
    }

    // wake up the bssExec since bounce buffers became available
    bssExec.synchronized {
      bssExec.notifyAll()
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
    tx.close()
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
          bounceBuffer
        } else {
          hostBounceBuffer
        }

        acquiredBuffs.zipWithIndex.foreach { case ((blockRange, rapidsBuffer, memBuff), ix) =>
          logDebug(s"copying from ${memBuff}, srcOffset ${blockRange.rangeStart} @ ${buffOffset} " +
              s"in dest ${bounceBuffToUse}, size = ${blockRange.rangeSize()}")
          val isLastOne = ix == acquiredBuffs.size - 1

          require(blockRange.rangeSize() <= bounceBuffToUse.getLength - buffOffset)

          if (true || !isLastOne) {
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
          } else {
            CudaUtil.copy(
              memBuff,
              blockRange.rangeStart,
              bounceBuffToUse,
              buffOffset,
              blockRange.rangeSize())
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
        close()
        throw t
    }

    logDebug(s"Sending ${buffsToSend} for transfer request, " +
        s" [peer_executor_id=${transferRequest.executorId()}, " +
        s"table_id=${acquiredTables.map(_.id).mkString(",")}, " +
        s"tag=${acquiredAlts.map(t => TransportUtils.formatTag(t.tag))}]")

    buffsToSend
  }
}