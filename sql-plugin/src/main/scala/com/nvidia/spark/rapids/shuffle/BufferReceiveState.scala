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

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, MemoryBuffer, NvtxColor, NvtxRange, Rmm}
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.internal.Logging

class BufferReceiveState(
    transport: RapidsShuffleTransport,
    bounceBuffer: DeviceMemoryBuffer,
    requests: Seq[PendingTransferRequest],
    stream: Cuda.Stream = Cuda.DEFAULT_STREAM)
    extends Iterator[AddressLengthTag]
        with AutoCloseable with Logging {

  class ReceiveBlock(val request: PendingTransferRequest) extends BlockWithSize {
    override def size: Long = request.getLength
    def tag: Long = request.tag
  }

  // TODO: Until cudf has a DeviceMemoryBuffer.allocate(size, stream), we can't really
  //   get off the default stream for allocations and copies.

  private[this] var bounceBufferByteOffset = 0L
  var firstTime = true
  private[this] var markedAsDone = false

  val windowedBlockIterator = new WindowedBlockIterator[ReceiveBlock](
    requests.map(r => new ReceiveBlock(r)), bounceBuffer.getLength)

  def getRequests(): Seq[PendingTransferRequest] = requests

  override def close(): Unit = synchronized {
    if (bounceBuffer != null) {
      transport.freeReceiveBounceBuffers(Seq(bounceBuffer))
    }
    //TODO: check that all other buffers are gone
  }

  override def hasNext: Boolean = synchronized { !markedAsDone }

  private[this] var nextBlocks: Seq[BlockRange[ReceiveBlock]] = Seq.empty
  private[this] var currentBlocks: Seq[BlockRange[ReceiveBlock]] = Seq.empty

  override def next(): AddressLengthTag = synchronized {
    if (firstTime) {
      nextBlocks = windowedBlockIterator.next()
      firstTime = false
    }
    currentBlocks = nextBlocks

    val firstTag = getFirstTag(currentBlocks)

    val alt = AddressLengthTag.from(bounceBuffer, firstTag)
    alt.resetLength(currentBlocks.map(_.rangeSize()).sum)

    if (windowedBlockIterator.hasNext) {
      nextBlocks = windowedBlockIterator.next()
    } else {
      nextBlocks = Seq.empty
      markedAsDone = true
    }
    alt
  }

  def getFirstTag(blocks: Seq[BlockRange[ReceiveBlock]]): Long = {
    blocks.head.block.tag
  }

  def getCurrentHandlers(): Seq[RapidsShuffleFetchHandler] = {
    currentBlocks.map { case b =>
      b.block.request.handler
    }
  }

  private[this] var workingOn: DeviceMemoryBuffer = null
  private[this] var workingOnSoFar: Long = 0L

  def consumeWindow(): Seq[(MemoryBuffer, TableMeta, RapidsShuffleFetchHandler)] = synchronized {
    val windowRange = new NvtxRange("consumeWindow", NvtxColor.PURPLE)
    try {
      val results = currentBlocks.flatMap { case b =>
        val pendingTransferRequest =
          b.block.request

        val fullSize = pendingTransferRequest.tableMeta.bufferMeta().size()

        val consumed = if (fullSize == b.rangeSize()) {
          logTrace(s"have full buffer ${b}")
          // we have the full buffer!
          val buff = Rmm.alloc(b.rangeSize(), stream)

          buff.copyFromDeviceBufferAsync(
            0,
            bounceBuffer,
            bounceBufferByteOffset,
            b.rangeSize(),
            stream)

          Some((buff,
              pendingTransferRequest.tableMeta,
              pendingTransferRequest.handler))
        } else {
          logTrace(s"do not have full buffer ${b}")
          if (workingOn != null) {
            workingOn.copyFromDeviceBufferAsync(
              workingOnSoFar,
              bounceBuffer,
              bounceBufferByteOffset,
              b.rangeSize(),
              stream)

            workingOnSoFar += b.rangeSize()
            if (workingOnSoFar == fullSize) {
              val res = Some((workingOn,
                  pendingTransferRequest.tableMeta,
                  pendingTransferRequest.handler))

              workingOn = null
              workingOnSoFar = 0
              res
            } else {
              None
            }
          } else {
            // need to keep it around
            workingOn = Rmm.alloc(fullSize, stream)

            workingOn.copyFromDeviceBufferAsync(
              0,
              bounceBuffer,
              bounceBufferByteOffset,
              b.rangeSize(),
              stream)

            workingOnSoFar += b.rangeSize()
            None
          }
        }
        bounceBufferByteOffset += b.rangeSize()
        if (bounceBufferByteOffset >= bounceBuffer.getLength) {
          logInfo(s"And we are starting over => at end of ${bounceBuffer}")
          bounceBufferByteOffset = 0
        }
        consumed
      }

      // sync once, rather for each copy
      // I need to synchronize, because we can't ask ucx to overwrite our bounce buffer
      // unless all that data has truly moved to our final buffer in our stream
      stream.sync()

      results
    } finally {
      windowRange.close()
    }
  }
}
