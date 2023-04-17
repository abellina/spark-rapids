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

import java.util.concurrent.Executors
import java.util.concurrent.Future

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, NvtxRange, PinnedMemoryPool, Table}
import com.nvidia.spark.rapids.Arm.withResource
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

  var allocated = false
  var buffer1: HostMemoryBuffer = null
  var buffer2: HostMemoryBuffer = null

  class DoubleBufferedCopy(target: HostMemoryBuffer) {
    
    val bbs = new Array[HostMemoryBuffer](2)
    if (allocated == false) {
      allocated = true
      val (b1, _) = allocateHostBuffer(100L*1024*1024)
      val (b2, _) = allocateHostBuffer(100L*1024*1024)
      buffer1 = b1
      buffer2 = b2
    }

    bbs(0) = buffer1
    bbs(1) = buffer2
    
    val futs = new Array[Future[Unit]](2)
    futs(0) = null
    futs(1) = null
    val evts = new Array[Cuda.Event](2)
    evts(0) = new Cuda.Event()
    evts(1) = new Cuda.Event()

    var currentBufferIx = 0

    val exec = Executors.newSingleThreadExecutor()
    var hostOffset: Long = 0L
    
    def copy(devBuffer: DeviceMemoryBuffer, size: Long) = {
      val hostBounceBuffer = bbs(currentBufferIx)
      evts(currentBufferIx).sync()
      if (futs(currentBufferIx) != null) {
        futs(currentBufferIx).get()
      }
      withResource(new NvtxRange(s"copy to bb ${currentBufferIx}", NvtxColor.YELLOW)) { _ =>
        hostBounceBuffer.copyFromMemoryBufferAsync(0, devBuffer, 0, size, Cuda.DEFAULT_STREAM)
        evts(currentBufferIx).record(Cuda.DEFAULT_STREAM)
      }

      val myix = currentBufferIx
      futs(myix) = exec.submit(() => {
        withResource(new NvtxRange(s"h2h ${myix}", NvtxColor.RED)) { _ =>
          evts(myix).sync()
          target.copyFromMemoryBuffer(
            hostOffset, hostBounceBuffer, 0L, size, Cuda.DEFAULT_STREAM) // h2h (blocking)
          hostOffset = hostOffset + size
        }
      })
      currentBufferIx = (currentBufferIx + 1) % 2
    }
  }

  var hostBounceBuffer: HostMemoryBuffer = null

  override protected def createBuffer(
      other: RapidsBuffer,
      otherBufferIterator: Iterator[(MemoryBuffer, Long)],
      shouldClose: Boolean,
      stream: Cuda.Stream): RapidsBufferBase = {
    if(hostBounceBuffer == null) {
      val (hostbb, mode) = allocateHostBuffer(100L*1024*1024)
      hostBounceBuffer = hostbb
      logWarning(s"host bounce buffer allocation mode: ${mode}")
    }

    val hostBuffSize = otherBufferIterator match {
      case p: ChunkedPacker => p.getMeta().bufferMeta().size()
      case _ => other.getSize
    }
    val (hostBuffer, allocationMode) = allocateHostBuffer(hostBuffSize)
    val startTime = System.currentTimeMillis()
    var hostOffset = 0L
    //val dbc = new DoubleBufferedCopy(hostBuffer)
    while (otherBufferIterator.hasNext) {
      val (otherBuffer, deviceSize) = otherBufferIterator.next()
      try {
        otherBuffer match {
          case devBuffer: DeviceMemoryBuffer =>
            //dbc.copy(devBuffer, deviceSize)
            hostBounceBuffer.copyFromMemoryBuffer(
              0, devBuffer, 0, deviceSize, stream)
            stream.sync()
            withResource(new NvtxRange("h2h", NvtxColor.RED)) { _ =>
              hostBuffer.copyFromHostBuffer(
                hostOffset,
                hostBounceBuffer,
                0L,
                deviceSize)
              stream.sync()
            }
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
    stream.sync() // pinned host buffer has data

    val endTime = System.currentTimeMillis()
    val amt = hostBuffSize.toDouble/1024.0/1024.0
    val bandwidth = (hostBuffSize.toDouble /1024.0/1024.0) / ((endTime - startTime).toDouble / 1000.0)

    logWarning(s"Spill bandwidth: copied ${amt} MB @ ${bandwidth} MB/sec")

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
      hostBuffSize,
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
