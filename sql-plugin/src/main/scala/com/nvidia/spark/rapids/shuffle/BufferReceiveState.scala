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

  // we use this to keep a list (should be depth 1) of "requests for receives"
  //  => the transport is ready to receive again, but we are not done consuming the
  //     buffers from the previous receive, so we must delay the transport.
  var toFinalize = new util.ArrayDeque[TransportBuffer => Unit]()

  val dmb = bounceBuffer.buffer.asInstanceOf[BaseDeviceMemoryBuffer]
  val brs =
    Cuda.createBufferReceiveState(
      dmb.getAddress,
      dmb.getLength,
      Seq(10L, 512L, 512L).toArray,
      Seq(1,2,3).toArray,
      stream.getStream())

  def getRequests: Seq[PendingTransferRequest] = requests

  override def close(): Unit = synchronized {
    if (bounceBuffer != null) {
      bounceBuffer.close()
    }
    transportOnClose()
  }

  /**
   * Calls `transferError` on each `RapidsShuffleFetchHandler`
   * @param errMsg - the message to pass onto the handlers
   */
  def errorOccurred(errMsg: String, throwable: Throwable = null): Unit = synchronized {
    // for current and future blocks, tell handlers of error
    // TODO
    //(currentBlocks ++ windowedBlockIterator.toSeq.flatten)
    //  .foreach(_.block.request.handler.transferError(errMsg, throwable))
  }

  def hasMoreBlocks: Boolean = synchronized { Cuda.bufferReceiveStateHasNext(brs) }

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
    withResource(new NvtxRange("consumeWindow", NvtxColor.PURPLE)) { _ =>
      val taskIds = currentBlocks.flatMap(_.block.request.handler.getTaskIds)
      RmmSpark.shuffleThreadWorkingOnTasks(taskIds)
      val results = Cuda.bufferReceiveStateConsume(brs)
      val resultsWithMeta = results.map { r =>
        val (meta, handler)  = getMetaAndHandler(r.blockId)
        ConsumedBatchFromBounceBuffer(
          r.packedAddress,
          r.size,
          meta,
          handler)
      }
      RmmSpark.poolThreadFinishedForTasks(taskIds)
      // cpu is in sync, we can recycle the bounce buffer
      if (!toFinalize.isEmpty) {
        val firstCb = toFinalize.pop()
        firstCb(transportBuffer)
      }
      resultsWithMeta
    }
  }
}
