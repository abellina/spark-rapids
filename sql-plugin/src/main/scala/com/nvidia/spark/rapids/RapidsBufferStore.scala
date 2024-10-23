/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

import java.util.Comparator
import java.util.concurrent.locks.ReentrantReadWriteLock

import scala.collection.mutable

import ai.rapids.cudf.{BaseDeviceMemoryBuffer, Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.StorageTier.{DEVICE, HOST, StorageTier}
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Helper case classes that contain the buffer we spilled or unspilled from our current tier
 * and likely a new buffer created in a target store tier, but it can be set to None.
 * If the buffer already exists in the target store, `newBuffer` will be None.
 * @param spillBuffer a `RapidsBuffer` we spilled or unspilled from this store
 * @param newBuffer an optional `RapidsBuffer` in the target store.
 */
trait SpillAction {
  val spillBuffer: RapidsBuffer
  val newBuffer: Option[RapidsBuffer]
}

case class BufferSpill(spillBuffer: RapidsBuffer, newBuffer: Option[RapidsBuffer])
    extends SpillAction

case class BufferUnspill(spillBuffer: RapidsBuffer, newBuffer: Option[RapidsBuffer])
    extends SpillAction

/**
 * Base class for all buffer store types.
 *
 * @param tier storage tier of this store
 * @param catalog catalog to register this store
 */
abstract class RapidsBufferStore(val tier: StorageTier)
    extends AutoCloseable with Logging {

  val name: String = tier.toString

  private class BufferTracker {
    private[this] val comparator: Comparator[RapidsBufferBase] =
      (o1: RapidsBufferBase, o2: RapidsBufferBase) =>
        java.lang.Long.compare(o1.getSpillPriority, o2.getSpillPriority)
    // buffers: contains all buffers in this store, whether spillable or not
    private[this] val buffers = new java.util.HashMap[RapidsBufferId, RapidsBufferBase]
    // spillable: contains only those buffers that are currently spillable
    private[this] val spillable = new HashedPriorityQueue[RapidsBufferBase](comparator)
    // spilling: contains only those buffers that are currently being spilled, but
    // have not been removed from the store
    private[this] val spilling = new mutable.HashSet[RapidsBufferId]()
    // total bytes stored, regardless of spillable status
    private[this] var totalBytesStored: Long = 0L
    // total bytes that are currently eligible to be spilled
    private[this] var totalBytesSpillable: Long = 0L

    def add(buffer: RapidsBufferBase): Unit = synchronized {
      val old = buffers.put(buffer.id, buffer)
      // it is unlikely that the buffer was in this collection, but removing
      // anyway. We assume the buffer is safe in this tier, and is not spilling
      spilling.remove(buffer.id)
      if (old != null) {
        throw new DuplicateBufferException(s"duplicate buffer registered: ${buffer.id}")
      }
      totalBytesStored += buffer.memoryUsedBytes

      // device buffers "spillability" is handled via DeviceMemoryBuffer ref counting
      // so spillableOnAdd should be false, all other buffer tiers are spillable at
      // all times.
      if (spillableOnAdd && buffer.memoryUsedBytes > 0) {
        if (spillable.offer(buffer)) {
          totalBytesSpillable += buffer.memoryUsedBytes
        }
      }
    }

    def remove(id: RapidsBufferId): Unit = synchronized {
      // when removing a buffer we no longer need to know if it was spilling
      spilling.remove(id)
      val obj = buffers.remove(id)
      if (obj != null) {
        totalBytesStored -= obj.memoryUsedBytes
        if (spillable.remove(obj)) {
          totalBytesSpillable -= obj.memoryUsedBytes
        }
      }
    }

    def freeAll(): Unit = {
      val values = synchronized {
        val buffs = buffers.values().toArray(new Array[RapidsBufferBase](0))
        buffers.clear()
        spillable.clear()
        spilling.clear()
        buffs
      }
      // We need to release the `RapidsBufferStore` lock to prevent a lock order inversion
      // deadlock: (1) `RapidsBufferBase.free`     calls  (2) `RapidsBufferStore.remove` and
      //           (1) `RapidsBufferStore.freeAll` calls  (2) `RapidsBufferBase.free`.
      values.safeFree()
    }

    /**
     * Sets a buffers state to spillable or non-spillable.
     *
     * If the buffer is currently being spilled or it is no longer in the `buffers` collection
     * (e.g. it is not in this store), the action is skipped.
     *
     * @param buffer      the buffer to mark as spillable or not
     * @param isSpillable whether the buffer should now be spillable
     */
    def setSpillable(buffer: RapidsBufferBase, isSpillable: Boolean): Unit = synchronized {
      if (isSpillable && buffer.memoryUsedBytes > 0) {
        // if this buffer is in the store and isn't currently spilling
        if (!spilling.contains(buffer.id) && buffers.containsKey(buffer.id)) {
          // try to add it to the spillable collection
          if (spillable.offer(buffer)) {
            totalBytesSpillable += buffer.memoryUsedBytes
            logDebug(s"Buffer ${buffer.id} is spillable. " +
              s"total=${totalBytesStored} spillable=${totalBytesSpillable}")
          } // else it was already there (unlikely)
        }
      } else {
        if (spillable.remove(buffer)) {
          totalBytesSpillable -= buffer.memoryUsedBytes
          logDebug(s"Buffer ${buffer.id} is not spillable. " +
            s"total=${totalBytesStored}, spillable=${totalBytesSpillable}")
        } // else it was already removed
      }
    }

    def nextSpillableBuffer(): RapidsBufferBase = synchronized {
      val buffer = spillable.poll()
      if (buffer != null) {
        // mark the id as "spilling" (this buffer is in the middle of a spill operation)
        spilling.add(buffer.id)
        totalBytesSpillable -= buffer.memoryUsedBytes
        logDebug(s"Spilling buffer ${buffer.id}. size=${buffer.memoryUsedBytes} " +
          s"total=${totalBytesStored}, new spillable=${totalBytesSpillable}")
      }
      buffer
    }

    def updateSpillPriority(buffer: RapidsBufferBase, priority:Long): Unit = synchronized {
      buffer.updateSpillPriorityValue(priority)
      spillable.priorityUpdated(buffer)
    }

    def getTotalBytes: Long = synchronized { totalBytesStored }

    def getTotalSpillableBytes: Long = synchronized { totalBytesSpillable }
  }

  // Utility function to obtain meta from a buffer if it supports it
  def getMeta(rapidsBuffer: RapidsBuffer, copyIter: RapidsBufferCopyIterator): Option[TableMeta] = {
    // we have a couple of ways to get meta from a buffer
    // either the buffer implements `RapidsBufferWithMeta`, so it has a `meta` function
    // or the `copyIter` is operating on a chunked-pack buffer, in which case it can be used to
    // provide a meta.
    // In other cases, we don't have a meta to describe this buffer, so it is "just a buffer"
    rapidsBuffer match {
      case bwm: RapidsBufferWithMeta =>
        Some(bwm.meta)
      case _ =>
        if (copyIter.isChunked) {
          Some(copyIter.getMeta)
        } else {
          None
        }
    }
  }

  /**
   * Stores that need to stay within a specific byte limit of buffers stored override
   * this function. Only the `HostMemoryBufferStore` requires such a limit.
   * @return maximum amount of bytes that can be stored in the store, None for no
   *         limit
   */
  def getMaxSize: Option[Long] = None

  private[this] val buffers = new BufferTracker

  /** A store that can be used for spilling. */
  var spillStore: RapidsBufferStore = _

  /** Return the current byte total of buffers in this store. */
  def currentSize: Long = buffers.getTotalBytes

  def currentSpillableSize: Long = buffers.getTotalSpillableBytes

  /**
   * A store that manages spillability of buffers should override this method
   * to false, otherwise `BufferTracker` treats buffers as always spillable.
   */
  protected def spillableOnAdd: Boolean = true

  /**
   * Specify another store that can be used when this store needs to spill.
   * @note Only one spill store can be registered. This will throw if a
   * spill store has already been registered.
   */
  def setSpillStore(store: RapidsBufferStore): Unit = {
    require(spillStore == null, "spill store already registered")
    spillStore = store
  } 

  def setSpillable(buffer: RapidsBufferBase, isSpillable: Boolean): Unit = {
    buffers.setSpillable(buffer, isSpillable)
  }

  /**
   * Create a new buffer from an existing buffer in another store.
   * If the data transfer will be performed asynchronously, this method is responsible for
   * adding a reference to the existing buffer and later closing it when the transfer completes.
   *
   * @note DO NOT close the buffer unless adding a reference!
   * @note `createBuffer` impls should synchronize against `stream` before returning, if needed.
   * @param buffer data from another store
   * @param catalog RapidsBufferCatalog we may need to modify during this create
   * @param stream CUDA stream to use or null
   * @return the new buffer that was created.
   */
  def createBuffer(
    rapidsBuffer: RapidsBuffer,
    catalog: RapidsBufferCatalog,
    stream: Cuda.Stream,
    base: RapidsMemoryBuffer): RapidsBuffer

  /** Update bookkeeping for a new buffer */
  protected def addBuffer(buffer: RapidsBufferBase): Unit = {
    buffers.add(buffer)
    buffer.updateSpillability()
  }

  /**
   * Adds a buffer to the spill framework, stream synchronizing with the producer
   * stream to ensure that the buffer is fully materialized, and can be safely copied
   * as part of the spill.
   *
   * @param needsSync true if we should stream synchronize before adding the buffer
   */
  protected def addBuffer(buffer: RapidsBufferBase, needsSync: Boolean): Unit = {
    if (needsSync) {
      Cuda.DEFAULT_STREAM.sync()
    }
    addBuffer(buffer)
  }

  override def close(): Unit = {
    buffers.freeAll()
  }

  def nextSpillable(): RapidsBuffer = {
    buffers.nextSpillableBuffer()
  }

  def synchronousSpill(
      targetTotalSize: Long,
      stream: Cuda.Stream = Cuda.DEFAULT_STREAM): Long = {
    if (currentSpillableSize > targetTotalSize) {
      logWarning(s"Targeting a ${name} size of $targetTotalSize. " +
          s"Current total ${currentSize}. " +
          s"Current spillable ${currentSpillableSize}")
      val bufferSpills = new mutable.ArrayBuffer[BufferSpill]()
      withResource(new NvtxRange(s"${name} sync spill", NvtxColor.ORANGE)) { _ =>
        logWarning(s"${name} store spilling to reduce usage from " +
          s"${currentSize} total (${currentSpillableSize} spillable) " +
          s"to $targetTotalSize bytes")

        // If the store has 0 spillable bytes left, it has exhausted.
        try {
          var exhausted = false
          var totalSpilled = 0L
          while (!exhausted &&
            currentSpillableSize > targetTotalSize) {
            val nextSpillableBuffer = nextSpillable()
            if (nextSpillableBuffer != null) {
              if (nextSpillableBuffer.addReference()) {
                withResource(nextSpillableBuffer) { _ =>
                  val bufferSpill = spillBuffer(nextSpillableBuffer, this, stream)
                  totalSpilled += bufferSpill.spillBuffer.memoryUsedBytes
                  bufferSpills.append(bufferSpill)
                }
              }
            }
          }
          if (totalSpilled <= 0) {
            // we didn't spill in this iteration, exit loop
            exhausted = true
            logWarning("Unable to spill enough to meet request. " +
              s"Total=${currentSize} " +
              s"Spillable=${currentSpillableSize} " +
              s"Target=$targetTotalSize")
          }
          totalSpilled
        } finally {
          if (bufferSpills.nonEmpty) {
            // This is a hack in order to completely synchronize with the GPU before we free
            // a buffer. It is necessary because of non-synchronous cuDF calls that could fall
            // behind where the CPU is. Freeing a rapids buffer in these cases needs to wait for
            // all launched GPU work, otherwise crashes or data corruption could occur.
            // A more performant implementation would be to synchronize on the thread that read
            // the buffer via events.
            // https://github.com/NVIDIA/spark-rapids/issues/8610
            Cuda.deviceSynchronize()
            bufferSpills.foreach(_.spillBuffer.safeFree())
          }
        }
      }
    } else {
      0L // nothing spilled
    }
  }

  /**
   * Given a specific `RapidsBuffer` spill it to `spillStore`
   *
   * @return a `BufferSpill` instance with the target buffer in this store, and an optional
   *         new `RapidsBuffer` in the target spill store if this rapids buffer hadn't already
   *         spilled.
   * @note called with catalog lock held
   */
  private def spillBuffer(
      buffer: RapidsBuffer,
      store: RapidsBufferStore,
      stream: Cuda.Stream): BufferSpill = {
    val newBuffer = buffer.base.spill(store, store.spillStore, stream)
    // return the buffer to free and the new buffer to register
    BufferSpill(buffer, newBuffer)
  }

  /**
   * Tries to make room for `buffer` in the host store by spilling.
   *
   * @param buffer buffer that will be copied to the host store if it fits
   * @param stream CUDA stream to synchronize for memory operations
   * @return true if the buffer fits after a potential spill
   */
  protected def trySpillToMaximumSize(
      buffer: RapidsBuffer,
      stream: Cuda.Stream): Boolean = {
    true // default to success, HostMemoryStore overrides this
  }

  def setSpillPriority(buffer: RapidsBufferBase, newPriority: Long): Unit =
    buffers.updateSpillPriority(buffer, newPriority)

  def remove(id: RapidsBufferId): Unit = buffers.remove(id)

  /** Base class for all buffers in this store. */
  abstract class RapidsBufferBase(override val id: RapidsBufferId,
                                  initialSpillPriority: Long)
    extends RapidsBuffer with Logging {

    /** The storage tier for buffers in this store */
    override val storageTier: StorageTier = tier

    // isValid and refcount must be used with the `RapidsBufferBase` lock held
    protected[this] var isValid = true

    protected[this] var refcount = 0

    private[this] var spillPriority: Long = initialSpillPriority

    /** Release the underlying resources for this buffer. */
    protected def releaseResources(): Unit

    override def addReference(): Boolean = synchronized {
      if (isValid) {
        refcount += 1
      }
      isValid
    }

    /**
     * close() is called by client code to decrease the ref count of this RapidsBufferBase.
     * In the off chance that by the time close is invoked, the buffer was freed (not valid)
     * then this close call winds up freeing the resources of the rapids buffer.
     */
    override def close(): Unit = synchronized {
      if (refcount == 0) {
        throw new IllegalStateException("Buffer already closed")
      }
      refcount -= 1
      if (refcount == 0 && !isValid) {
        freeBuffer()
      }
    }

    /**
     * Mark the buffer as freed and no longer valid. This is called by the store when removing a
     * buffer (it is no longer tracked).
     *
     * @note The resources may not be immediately released if the buffer has outstanding references.
     * In that case the resources will be released when the reference count reaches zero.
     */
    override def free(): Unit = synchronized {
      if (isValid) {
        isValid = false
        remove(id)
        if (refcount == 0) {
          freeBuffer()
        }
      } else {
        logWarning(s"Trying to free an invalid buffer => $id, size = ${memoryUsedBytes}, $this")
      }
    }

    /**
     * Get the spill priority value for this buffer. Lower values are higher
     * priority for spilling, meaning buffers with lower values will be
     * preferred for spilling over buffers with a higher value.
     */
    override def getSpillPriority: Long = {
      spillPriority
    }

    override def setSpillPriority(newPriority: Long): Unit = {
      buffers.updateSpillPriority(this, newPriority)
    }

    /**
     * Function invoked by the `RapidsBufferStore.addBuffer` method that prompts
     * the specific `RapidsBuffer` to check its reference counting to make itself
     * spillable or not. Only `RapidsTable` and `RapidsHostMemoryBuffer` implement
     * this method.
     */
    def updateSpillability(): Unit = {}

    def updateSpillPriorityValue(priority: Long): Unit = {
      spillPriority = priority
    }

    /** Must be called with a lock on the buffer */
    private def freeBuffer(): Unit = {
      releaseResources()
    }

    override def toString: String = s"${this.getClass.getName} size=$memoryUsedBytes"
  }

  abstract class RapidsBufferBaseWithMeta(
      override val id: RapidsBufferId,
      _meta: TableMeta,
      initialSpillPriority: Long)
    extends RapidsBufferBase(id, initialSpillPriority)
      with RapidsBufferWithMeta {
    override def meta: TableMeta = _meta

    override def getColumnarBatch(
        sparkTypes: Array[DataType], stream: Cuda.Stream): ColumnarBatch = {
      // NOTE: Cannot hold a lock on this buffer here because memory is being
      // allocated. Allocations can trigger synchronous spills which can
      // deadlock if another thread holds the device store lock and is trying
      // to spill to this store.
      withResource(getDeviceMemoryBuffer(stream)) { deviceBuffer =>
        RapidsBuffer.columnarBatchFromDeviceBuffer(deviceBuffer, sparkTypes, meta)
      }
    }
  }
}

trait RapidsBufferWithMeta {
  def meta: TableMeta
}

trait CopyableRapidsBuffer extends RapidsBuffer {

  /**
   * Copy the content of this buffer into the specified memory buffer, starting from the given
   * offset.
   *
   * @param srcOffset offset to start copying from.
   * @param dst the memory buffer to copy into.
   * @param dstOffset offset to copy into.
   * @param length number of bytes to copy.
   * @param stream CUDA stream to use
   */
  def copyToMemoryBuffer(srcOffset: Long, dst: MemoryBuffer, dstOffset: Long,
                         length: Long, stream: Cuda.Stream): Unit = {
    withResource(getMemoryBuffer(stream)) { memBuff =>
      dst match {
        case _: HostMemoryBuffer =>
          // TODO: consider moving to the async version.
          dst.copyFromMemoryBuffer(dstOffset, memBuff, srcOffset, length, stream)
        case _: BaseDeviceMemoryBuffer =>
          dst.copyFromMemoryBufferAsync(dstOffset, memBuff, srcOffset, length, stream)
        case _ =>
          throw new IllegalStateException(s"Infeasible destination buffer type ${dst.getClass}")
      }
    }
  }
}

/**
 * Buffers that inherit from this type do not support changing the spillable status
 * of a `RapidsBuffer`. This is only used right now for disk.
 * @param tier storage tier of this store
 */
abstract class RapidsBufferStoreWithoutSpill(override val tier: StorageTier)
    extends RapidsBufferStore(tier) {

  override def setSpillable(rapidsBuffer: RapidsBufferBase, isSpillable: Boolean): Unit = {
    throw new NotImplementedError(s"This store ${this} does not implement setSpillable")
  }
}