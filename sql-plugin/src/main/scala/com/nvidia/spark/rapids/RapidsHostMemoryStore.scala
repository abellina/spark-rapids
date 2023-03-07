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

package com.nvidia.spark.rapids

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, NvtxRange, PinnedMemoryPool, Table}
import com.nvidia.spark.rapids.SpillPriorities.{applyPriorityOffset, HOST_MEMORY_BUFFER_DIRECT_OFFSET, HOST_MEMORY_BUFFER_PAGEABLE_OFFSET, HOST_MEMORY_BUFFER_PINNED_OFFSET}
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.format.TableMeta

/**
 * A buffer store using host memory.
 * @param maxSize maximum size in bytes for all buffers in this store
 * @param pageableMemoryPoolSize maximum size in bytes for the internal pageable memory pool
 */
class RapidsHostMemoryStore(
    maxSize: Long,
    pageableMemoryPoolSize: Long)
    extends RapidsBufferStore(StorageTier.HOST) {
  private[this] val pool = HostMemoryBuffer.allocate(pageableMemoryPoolSize, false)
  private[this] val addressAllocator = new AddressSpaceAllocator(pageableMemoryPoolSize)
  private[this] var haveLoggedMaxExceeded = false

  private sealed abstract class AllocationMode(val spillPriorityOffset: Long)
  private case object Pinned extends AllocationMode(HOST_MEMORY_BUFFER_PINNED_OFFSET)
  private case object Pooled extends AllocationMode(HOST_MEMORY_BUFFER_PAGEABLE_OFFSET)
  private case object Direct extends AllocationMode(HOST_MEMORY_BUFFER_DIRECT_OFFSET)

  override def getMaxSize: Option[Long] = Some(maxSize)

  private def allocateHostBuffer(size: Long): (HostMemoryBuffer, AllocationMode) = {
    var buffer: HostMemoryBuffer = PinnedMemoryPool.tryAllocate(size)
    if (buffer != null) {
      return (buffer, Pinned)
    }

    val allocation = addressAllocator.allocate(size)
    if (allocation.isDefined) {
      buffer = pool.slice(allocation.get, size)
      return (buffer, Pooled)
    }

    if (!haveLoggedMaxExceeded) {
      logWarning(s"Exceeding host spill max of $pageableMemoryPoolSize bytes to accommodate " +
          s"a buffer of $size bytes. Consider increasing pageable memory store size.")
      haveLoggedMaxExceeded = true
    }
    (HostMemoryBuffer.allocate(size, false), Direct)
  }

  override protected def createBuffer(
      other: RapidsBuffer,
      otherBufferIterator: Iterator[(MemoryBuffer, Long)],
      shouldClose: Boolean,
      stream: Cuda.Stream): RapidsBufferBase = {
    val (hostBuffer, allocationMode) = allocateHostBuffer(other.getSize)
    var hostOffset = 0L
    while (otherBufferIterator.hasNext) {
      val (otherBuffer, deviceSize) = otherBufferIterator.next()
      try {
        otherBuffer match {
          case devBuffer: DeviceMemoryBuffer =>
            hostBuffer.copyFromMemoryBuffer(
              hostOffset, devBuffer, 0, deviceSize, stream)
            hostOffset += deviceSize
          case _ =>
            throw new IllegalStateException("copying from buffer without device memory")
        }
      } catch {
        case e: Exception =>
          hostBuffer.close()
          throw e
      }
      if (shouldClose) {
        otherBuffer.close()
      }
    }

    var meta: TableMeta = null
    otherBufferIterator match {
      case p: ChunkedPacker =>
        // host memory buffer now has the full buffer contiguously
        // try to unpack it
        //GpuColumnVector.debug("before", p.tbl)
        //withResource(DeviceMemoryBuffer.allocate(other.size)) { throwAway =>
        //  throwAway.copyFromHostBuffer(hostBuffer)
        //  withResource(p.tbl.contiguousSplit()(0)) { csplit => 
        //    val metaBuff = csplit.getMetadataDirectBuffer()
        //    meta = MetaUtils.buildTableMeta(other.id.tableId, csplit)
        //    withResource(Table.fromPackedTable(metaBuff, throwAway)) { tbl =>
        //      GpuColumnVector.debug("unpacked", tbl)
        //      logInfo(s"num cols: ${tbl.getNumberOfColumns}, row count: ${tbl.getRowCount}")
        //      (0 until tbl.getNumberOfColumns).foreach { c =>
        //        logInfo(s"col $c type ${tbl.getColumn(c)}")
        //      }
        //    }
        //  }
        //}
        p.close()
      case _ => // noop
    }
    new RapidsHostMemoryBuffer(
      other.id,
      other.getSize,
      other.getMeta,
      applyPriorityOffset(other.getSpillPriority, allocationMode.spillPriorityOffset),
      hostBuffer,
      allocationMode)
  }

  def numBytesFree: Long = maxSize - currentSize

  override def close(): Unit = {
    super.close()
    pool.close()
  }

  class RapidsHostMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      spillPriority: Long,
      buffer: HostMemoryBuffer,
      allocationMode: AllocationMode)
      extends RapidsBufferBase(
        id, meta, spillPriority) {
    override val storageTier: StorageTier = StorageTier.HOST

    override def getMemoryBuffer: MemoryBuffer = {
      buffer.incRefCount()
      buffer
    }

    override protected def releaseResources(): Unit = {
      allocationMode match {
        case Pooled =>
          assert(buffer.getAddress >= pool.getAddress)
          assert(buffer.getAddress < pool.getAddress + pool.getLength)
          addressAllocator.free(buffer.getAddress - pool.getAddress)
        case _ =>
      }
      buffer.close()
    }

    /** The size of this buffer in bytes. */
    override def getSize: Long = size
  }
}
