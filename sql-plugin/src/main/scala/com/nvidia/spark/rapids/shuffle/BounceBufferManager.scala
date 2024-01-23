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

import java.util
import scala.collection.mutable

import ai.rapids.cudf.{MemoryBuffer, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.internal.Arm.withResource

import org.apache.spark.internal.Logging

/**
 * Class to hold a bounce buffer reference in `buffer`.
 *
 * It is `AutoCloseable`, where a call to `close` puts the bounce buffer
 * back into the corresponding source `BounceBufferManager`
 *
 * @param buffer - cudf MemoryBuffer to be used as a bounce buffer
 */
abstract class BounceBuffer(val buffer: MemoryBuffer, val offset: Long) extends AutoCloseable {
  var isClosed = false

  def free(bb: BounceBuffer): Unit

  override def close(): Unit = {
    if (isClosed) {
      throw new IllegalStateException("Bounce buffer closed too many times")
    }
    free(this)
    isClosed = true
  }
}

/**
 * This class can hold 1 or 2 `BounceBuffer`s and is only used in the send case.
 *
 * Ideally, the device buffer is used if most of the data to be sent is on
 * the device. The host buffer is used in the opposite case.
 *
 * @param deviceBounceBuffer - device buffer to use for sends
 * @param hostBounceBuffer - optional host buffer to use for sends
 */
case class SendBounceBuffers(
    deviceBounceBuffer: BounceBuffer,
    hostBounceBuffer: Option[BounceBuffer]) extends AutoCloseable {

  def bounceBufferSize: Long = {
    deviceBounceBuffer.buffer.getLength
  }

  override def close(): Unit = {
    deviceBounceBuffer.close()
    hostBounceBuffer.foreach(_.close())
  }
}

/**
 * This classes manages a set of bounce buffers, that are instances of `MemoryBuffer`.
 * The size/quantity of buffers is configurable, and so is the allocator.
 * @param poolName a human-friendly name to use for debug logs
 * @param bufferSize the size of buffer to use
 * @param numBuffers the number of buffers to allocate on instantiation
 * @param allocator function that takes a size, and returns a `MemoryBuffer` instance.
 * @tparam T the specific type of MemoryBuffer i.e. `DeviceMemoryBuffer`,
 *           `HostMemoryBuffer`, etc.
 */
class BounceBufferManager[T <: MemoryBuffer](
    poolName: String,
    val poolSize: Long,
    allocator: Long => T)
  extends AutoCloseable
  with Logging {

  val pool = new AddressSpaceAllocator(poolSize)

  def remaining(): Long = pool.available

  class BounceBufferImpl(buff: MemoryBuffer, offset: Long)
      extends BounceBuffer(buff, offset) {
    override def free(bb: BounceBuffer): Unit = {
      freeBuffer(bb)
    }
  }

  private[this] val rootBuffer = allocator(poolSize)

  /**
   * Acquires a [[BounceBuffer]] from the pool. Blocks if the pool is empty.
   *
   * @note calls to this function should have a lock on this [[BounceBufferManager]]
   * @return the acquired `BounceBuffer`
   */
  private def allocate(sz: Long): BounceBuffer = {
    val allocation = pool.allocate(sz)
    if (allocation.isEmpty) {
      throw new IllegalStateException(s"Buffer pool $poolName has exhausted!")
    }

    logDebug(s"$poolName: Buffer offset: ${allocation.get}")
    val res = rootBuffer.slice(allocation.get, sz)
    new BounceBufferImpl(res, allocation.get)
  }

  /**
   * Acquire `possibleNumBuffers` buffers from the pool. This method will not block.
   * @param possibleNumBuffers number of buffers to acquire
   * @return a sequence of `BounceBuffer`s, or empty if the request can't be satisfied
   */
  def acquireBuffersNonBlocking(sz: Long): BounceBuffer = synchronized {
    withResource(new NvtxRange(s"${poolName} acquire", NvtxColor.GREEN)) { _ =>
      if (pool.available < sz) {
        // would block
        return null
      }
      try {
        allocate(sz)
      } catch {
        case _: Throwable =>
          withResource(new NvtxRange(s"${poolName} exhausted", NvtxColor.RED)) { _ =>
            logWarning(s"pool ${poolName} exhausted avail: ${pool.available} attept: ${sz}")
          }
          null
      }
    }
  }

  /**
   * Free a `BounceBuffer`, putting it back into the pool.
   * @param bounceBuffer the memory buffer to free
   */
  def freeBuffer(bounceBuffer: BounceBuffer): Unit = synchronized {
    pool.free(bounceBuffer.offset)
    val buffer = bounceBuffer.buffer
    logDebug(s"$poolName: Free buffer index ${bounceBuffer.offset}")
    buffer.close()
    notifyAll() // notify any waiters that are checking the state of this manager
  }

  /**
   * Returns the root (backing) `MemoryBuffer`. This is used for a transport
   * that wants to register the bounce buffers against hardware, for pinning purposes.
   * @return the root (backing) memory buffer
   */
  def getRootBuffer(): MemoryBuffer = rootBuffer

  override def close(): Unit = rootBuffer.close()
}

/** Allocates blocks from an address space using a best-fit algorithm. */
class AddressSpaceAllocator(addressSpaceSize: Long) {
  /** Free blocks mapped by size of block for efficient size matching.  */
  private[this] val freeBlocks = new mutable.TreeMap[Long, mutable.Set[Block]]

  /** Allocated blocks mapped by block address. */
  private[this] val allocatedBlocks = new mutable.HashMap[Long, Block]

  /** Amount of memory allocated */
  private[this] var allocatedBytes: Long = 0L

  addFreeBlock(new Block(0, addressSpaceSize, allocated = false))

  private def addFreeBlock(block: Block): Unit = {
    val set = freeBlocks.getOrElseUpdate(block.blockSize, new mutable.HashSet[Block])
    val added = set.add(block)
    assert(added, "block was already in free map")
  }

  private def removeFreeBlock(block: Block): Unit = {
    val set = freeBlocks.getOrElse(block.blockSize,
      throw new IllegalStateException("block not in free map"))
    val removed = set.remove(block)
    assert(removed, "block not in free map")
    if (set.isEmpty) {
      freeBlocks.remove(block.blockSize)
    }
  }

  def allocate(size: Long): Option[Long] = synchronized {
    if (size <= 0) {
      return None
    }
    val it = freeBlocks.valuesIteratorFrom(size)
    if (it.hasNext) {
      val blockSet = it.next()
      val block = blockSet.head
      removeFreeBlock(block)
      allocatedBytes += size
      Some(block.allocate(size))
    } else {
      None
    }
  }

  def free(address: Long): Unit = synchronized {
    val block = allocatedBlocks.remove(address).getOrElse(
      throw new IllegalArgumentException(s"$address not allocated"))
    allocatedBytes -= block.blockSize
    block.free()
  }

  def allocatedSize: Long = synchronized {
    allocatedBytes
  }

  def available: Long = synchronized {
    addressSpaceSize - allocatedBytes
  }

  def numAllocatedBlocks: Long = synchronized {
    allocatedBlocks.size
  }

  def numFreeBlocks: Long = synchronized {
    freeBlocks.valuesIterator.map(_.size).sum
  }

  private class Block(
      val address: Long,
      var blockSize: Long,
      var lowerBlock: Option[Block] = None,
      var upperBlock: Option[Block] = None,
      var allocated: Boolean = false) {
    def allocate(amount: Long): Long = {
      assert(!allocated, "block being allocated already allocated")
      assert(amount <= blockSize, "allocating beyond block")
      if (amount != blockSize) {
        // split into an allocated and unallocated block
        val unallocated = new Block(
          address + amount,
          blockSize - amount,
          Some(this),
          upperBlock,
          allocated = false)
        addFreeBlock(unallocated)
        upperBlock.foreach { b =>
          assert(b.lowerBlock.get == this, "block linkage broken")
          assert(address + blockSize == b.address, "block adjacency broken")
          b.lowerBlock = Some(unallocated)
        }
        upperBlock = Some(unallocated)
        blockSize = amount
      }
      allocated = true
      allocatedBlocks.put(address, this)
      this.address
    }

    def free(): Unit = {
      assert(allocated, "block being freed not allocated")
      allocated = false
      upperBlock.foreach { b =>
        if (!b.allocated) {
          removeFreeBlock(b)
          coalesceUpper()
        }
      }
      var freeBlock = this
      lowerBlock.foreach { b =>
        if (!b.allocated) {
          removeFreeBlock(b)
          b.coalesceUpper()
          freeBlock = b
        }
      }
      addFreeBlock(freeBlock)
    }

    /** Coalesce the upper block into this block. Does not update freeBlocks. */
    private def coalesceUpper(): Unit = {
      val upper = upperBlock.getOrElse(throw new IllegalStateException("no upper block"))
      assert(upper.lowerBlock.orNull == this, "block linkage broken")
      assert(address + blockSize == upper.address, "block adjacency broken")
      blockSize += upper.blockSize
      upperBlock = upper.upperBlock
      upperBlock.foreach(_.lowerBlock = Some(this))
    }
  }
}