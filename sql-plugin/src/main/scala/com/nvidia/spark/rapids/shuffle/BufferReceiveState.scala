/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import java.util

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{BaseDeviceMemoryBuffer, Cuda, DeviceMemoryBuffer, NvtxColor, NvtxRange, Rmm}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.jni.RmmSpark

import org.apache.spark.internal.Logging

case class ConsumedBatchFromBounceBuffer(
    contigBuffer: DeviceMemoryBuffer,
    meta: TableMeta,
    handler: RapidsShuffleFetchHandler)

/**
 * A helper case class to maintain the state associated with a transfer request to a peer.
 *
 * On getBufferWhenReady(finalizeCb, size), a bounce buffer is either made
 * immediately available to `finalizeCb` or it will be made available later, via `toFinalize`.
 *
 * By convention, the `id` is used as the header for receives for this `BufferReceiveState`
 *
 * `consumeWindow` is called when data has arrived at the bounce buffer, copying
 * onto the ranges the bytes received.
 *
 * It also is AutoCloseable. close() should be called to free bounce buffers.
 *
 * @param id - a numeric id that is used in all headers for this `BufferReceiveState`
 * @param bounceBuffer - bounce buffer to use (device memory strictly)
 * @param requests - collection of `PendingTransferRequest` as issued by iterators
 *                 currently requesting
 * @param transportOnClose - a callback invoked when the `BufferReceiveState` closes
 * @param stream - CUDA stream to use for allocations and copies
 */
class BufferReceiveState(
    val id: Long,
    bounceBuffer: BounceBuffer,
    requests: Array[PendingTransferRequest],
    transportOnClose: () => Unit,
    stream: Cuda.Stream = Cuda.DEFAULT_STREAM)
    extends AutoCloseable with Logging {

  val transportBuffer = new CudfTransportBuffer(bounceBuffer.buffer)
  // we use this to keep a list (should be depth 1) of "requests for receives"
  //  => the transport is ready to receive again, but we are not done consuming the
  //     buffers from the previous receive, so we must delay the transport.
  var toFinalize = new util.ArrayDeque[TransportBuffer => Unit]()

  // if this is > 0, we are waiting to consume, so we need to queue up in `toFinalize`
  // any callbacks
  var toConsume = 0

  class ReceiveBlock(val request: PendingTransferRequest) extends BlockWithSize {
    override def size: Long = request.getLength
  }

  // flags whether we need to populate a window, and if the client should send a
  // transfer request
  private[this] var iterated = false

  // block ranges we'll work on next
  private[this] var nextBlocks: Array[BlockRange[ReceiveBlock]] = Array.empty

  // ranges we are working on now
  private[this] var currentBlocks: Array[BlockRange[ReceiveBlock]] = Array.empty

  // if a block overshoots a window, it will be in `workingOn`. This happens if the
  // block is larger than window also.
  private[this] var workingOn: DeviceMemoryBuffer = null

  // offset tracking
  private[this] var workingOnOffset: Long = 0L
  private[this] var bounceBufferByteOffset = 0L

  // get block ranges for us to work with
  private[this] val windowedBlockIterator = new WindowedBlockIterator[ReceiveBlock](
    requests.map(r => new ReceiveBlock(r)), bounceBuffer.buffer.getLength)

  private[this] var hasMoreBuffers_ = windowedBlockIterator.hasNext

  def getBufferWhenReady(finalizeCb: TransportBuffer => Unit, size: Long): Unit =
    synchronized {
      require(transportBuffer.getLength() >= size,
        "Asked to receive a buffer greater than the available bounce buffer.")

      if (toConsume == 0) {
        logDebug(s"Calling callback immediately for ${TransportUtils.toHex(id)}")
        finalizeCb(transportBuffer)
      } else {
        // have pending and haven't consumed it yet, consume will call the callback
        logDebug(s"Deferring callback for ${TransportUtils.toHex(id)}, " +
          s"have ${toFinalize.size} pending")
        toFinalize.add(finalizeCb)
      }
      toConsume += 1
    }

  def getRequests: Seq[PendingTransferRequest] = requests

  override def close(): Unit = synchronized {
    if (bounceBuffer != null) {
      bounceBuffer.close()
    }
    if (workingOn != null) {
      logWarning(s"BufferReceiveState closing, but there are unfinished batches")
      workingOn.close()
    }
    transportOnClose()
  }

  /**
   * Calls `transferError` on each `RapidsShuffleFetchHandler`
   * @param errMsg - the message to pass onto the handlers
   */
  def errorOccurred(errMsg: String, throwable: Throwable = null): Unit = synchronized {
    // for current and future blocks, tell handlers of error
    (currentBlocks ++ windowedBlockIterator.toSeq.flatten)
      .foreach(_.block.request.handler.transferError(errMsg, throwable))
  }

  def hasMoreBlocks: Boolean = synchronized { hasMoreBuffers_ }

  private def advance(): Unit = {
    withResource(new NvtxRange("advance", NvtxColor.YELLOW)) { _ =>
      if (!hasMoreBuffers_) {
        throw new NoSuchElementException(
          s"BufferReceiveState is done for ${TransportUtils.toHex(id)}, yet " +
              s"received a call to next")
      }

      if (!iterated) {
        nextBlocks = windowedBlockIterator.next()
        iterated = true
      }
      currentBlocks = nextBlocks

      if (windowedBlockIterator.hasNext) {
        nextBlocks = windowedBlockIterator.next()
      } else {
        nextBlocks = Array.empty
        hasMoreBuffers_ = false
      }
    }
  }

  case class MultiCopyAction(
      srcBase: BaseDeviceMemoryBuffer,
      var dstBase: DeviceMemoryBuffer,
      srcOffset: Long,
      dstOffset: Long,
      copySize: Long,
      complete: Boolean,
      pendingTransferRequest: PendingTransferRequest)

  /**
   * When a receive is complete, the client calls `consumeWindow` to copy out
   * of the bounce buffer in this `BufferReceiveState` any complete batches, or to
   * buffer up a remaining batch in `workingOn`
   *
   * @return - a sequence of batches that were successfully consumed, or an empty
   *         sequence if still working on a batch.
   */
  def consumeWindow(): Array[ConsumedBatchFromBounceBuffer] = synchronized {
    // once we reach 0 here the transport will be allowed to reuse the bounce buffer
    // e.g. after the synchronized block, or after we sync with GPU in this function.
    toConsume -= 1
    withResource(new NvtxRange("consumeWindow", NvtxColor.PURPLE)) { _ =>
      advance()
        val taskIds = currentBlocks.flatMap(_.block.request.handler.getTaskIds)
        RmmSpark.shuffleThreadWorkingOnTasks(taskIds)

        val multiCopyActions = new Array[MultiCopyAction](currentBlocks.length)

        // Receive buffers are always in the device, and so it is safe to assume
        // that they are `BaseDeviceMemoryBuffer`s here.
        val deviceBounceBuffer = bounceBuffer.buffer.asInstanceOf[BaseDeviceMemoryBuffer]
        currentBlocks.zipWithIndex.foreach { case (b, ix) =>
          val pendingTransferRequest = b.block.request
          val fullSize = pendingTransferRequest.getLength
          if (fullSize == b.rangeSize()) {
            // we have the full buffer!
            multiCopyActions(ix) = MultiCopyAction(
              srcBase = deviceBounceBuffer,
              dstBase = null,
              srcOffset = bounceBufferByteOffset,
              dstOffset = 0,
              copySize = b.rangeSize(),
              complete = true,
              pendingTransferRequest)
          } else {
            if (workingOnOffset != 0) {
              multiCopyActions(ix) = MultiCopyAction(
                srcBase = deviceBounceBuffer,
                dstBase = workingOn,
                srcOffset = bounceBufferByteOffset,
                dstOffset = workingOnOffset,
                copySize = b.rangeSize(),
                complete = workingOnOffset + b.rangeSize() == fullSize,
                pendingTransferRequest)
              workingOnOffset += b.rangeSize()
              if (workingOnOffset == fullSize) {
                workingOnOffset = 0
              }
            } else {
              // need to keep it around
              workingOn = withResource(new NvtxRange("call alloc", NvtxColor.BLUE)) { _ =>
                Rmm.alloc(fullSize, stream)
              }
              multiCopyActions(ix) = MultiCopyAction(
                srcBase = deviceBounceBuffer,
                dstBase = workingOn,
                srcOffset = bounceBufferByteOffset,
                dstOffset = workingOnOffset,
                copySize = b.rangeSize(),
                complete = workingOnOffset + b.rangeSize() == fullSize,
                pendingTransferRequest)
              workingOnOffset += b.rangeSize()
            }
          }
          bounceBufferByteOffset += b.rangeSize()
          if (bounceBufferByteOffset >= deviceBounceBuffer.getLength) {
            bounceBufferByteOffset = 0
          }
        }

        var results: Array[ConsumedBatchFromBounceBuffer] = Array.empty

        if (multiCopyActions.length > 0) {
          val toAlloc = multiCopyActions.filter { _.dstBase == null }
          val buffs =
            withResource(new NvtxRange("call batch alloc", NvtxColor.GREEN)) { _ =>
              Rmm.alloc(toAlloc.map {_.copySize }, stream)
            }
          toAlloc.zip(buffs).foreach { case (a,b) => a.dstBase = b }
          val srcAddresses = new Array[Long](multiCopyActions.length)
          val dstAddresses = new Array[Long](multiCopyActions.length)
          val bufferSizes = new Array[Long](multiCopyActions.length)

          multiCopyActions.zipWithIndex.foreach { case (mca, ix) =>
            srcAddresses(ix) = mca.srcBase.getAddress + mca.srcOffset
            dstAddresses(ix) = mca.dstBase.getAddress + mca.dstOffset
            bufferSizes(ix) = mca.copySize
          }

          logInfo(s"Issuing multiCopy for src=${srcAddresses.mkString(",")} " +
              s"dst=${dstAddresses.mkString(",")} " +
              s"buffSizes=${bufferSizes.mkString(",")} ")

          ai.rapids.cudf.Cuda.multiCopy(
            srcAddresses,
            dstAddresses,
            bufferSizes,
            bufferSizes.length,
            stream.getStream)

          val complete = multiCopyActions.filter(_.complete)
          results = complete.map { c =>
            ConsumedBatchFromBounceBuffer(
              c.dstBase,
              c.pendingTransferRequest.tableMeta,
              c.pendingTransferRequest.handler)
          }
          if (workingOnOffset == 0) {
            workingOn = null
          }
        }

        // Sync once, instead of for each copy.
        // We need to synchronize, because we can't ask ucx to overwrite our bounce buffer
        // unless all that data has truly moved to our final buffer in our stream
        stream.sync()

        RmmSpark.poolThreadFinishedForTasks(taskIds)

        // cpu is in sync, we can recycle the bounce buffer
        if (!toFinalize.isEmpty) {
          val firstCb = toFinalize.pop()
          firstCb(transportBuffer)
        }

        results
      }
    }
}
