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

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.Arm._
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.sql.rapids.TempSpillBufferId
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Buffer storage using device memory.
 * @param catalog catalog to register this store
 */
class RapidsDeviceMemoryStore
  extends RapidsBufferStore(StorageTier.DEVICE) {

  // The RapidsDeviceMemoryStore handles spillability via ref counting
  override protected def spillableOnAdd: Boolean = false

  override protected def createBuffer(
      other: RapidsBuffer,
      memoryBufferIterator: Iterator[(MemoryBuffer, Long)],
      shouldClose: Boolean,
      stream: Cuda.Stream): RapidsBufferBase = {
    var memoryBuffer : MemoryBuffer = null
    while(memoryBufferIterator.hasNext) {
      val (mb, _) = memoryBufferIterator.next()
      memoryBuffer = mb
      require(!memoryBufferIterator.hasNext,
        "Expected a single item, but got multiple")
    }
    val deviceBuffer = {
      memoryBuffer match {
        case d: DeviceMemoryBuffer => d
        case h: HostMemoryBuffer =>
          withResource(h) { _ =>
            closeOnExcept(DeviceMemoryBuffer.allocate(other.getSize)) { deviceBuffer =>
              logDebug(s"copying from host $h to device $deviceBuffer")
              deviceBuffer.copyFromHostBuffer(h, stream)
              deviceBuffer
            }
          }
        case b => throw new IllegalStateException(s"Unrecognized buffer: $b")
      }
    }
    new RapidsDeviceMemoryBuffer(
      other.id,
      deviceBuffer.getLength,
      other.getMeta,
      deviceBuffer,
      other.getSpillPriority)
  }

  /**
   * Adds a buffer to the device storage. This does NOT take ownership of the
   * buffer, so it is the responsibility of the caller to close it.
   *
   * This function is called only from the RapidsBufferCatalog, under the
   * catalog lock.
   *
   * @param id the RapidsBufferId to use for this buffer
   * @param buffer buffer that will be owned by the store
   * @param tableMeta metadata describing the buffer layout
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param needsSync whether the spill framework should stream synchronize while adding
   *                  this device buffer (defaults to true)
   * @return the RapidsBuffer instance that was added.
   */
  def addBuffer(
      id: RapidsBufferId,
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      needsSync: Boolean): RapidsBuffer = {
    buffer.incRefCount()
    val rapidsBuffer = new RapidsDeviceMemoryBuffer(
      id,
      buffer.getLength,
      tableMeta,
      buffer,
      initialSpillPriority)
    freeOnExcept(rapidsBuffer) { _ =>
      logDebug(s"Adding receive side table for: [id=$id, size=${buffer.getLength}, " +
        s"uncompressed=${rapidsBuffer.getMeta().bufferMeta.uncompressedSize}, " +
        s"meta_id=${tableMeta.bufferMeta.id}, " +
        s"meta_size=${tableMeta.bufferMeta.size}]")
      addBuffer(rapidsBuffer, needsSync)
      rapidsBuffer
    }
  }

  def addBatch(
      id: TempSpillBufferId,
      batch: ColumnarBatch,
      initialSpillPriority: Long,
      needsSync: Boolean): RapidsBuffer = {
    val rapidsBuffer = new RapidsDeviceMemoryBatch(
      id,
      batch,
      initialSpillPriority)
    freeOnExcept(rapidsBuffer) { _ =>
      addBuffer(rapidsBuffer, needsSync)
      // TODO: we need to figure out a different signal for batch
      doSetSpillable(rapidsBuffer, true)
      rapidsBuffer
    }
  }

  /**
   * Adds a device buffer to the spill framework, stream synchronizing with the producer
   * stream to ensure that the buffer is fully materialized, and can be safely copied
   * as part of the spill.
   *
   * @param needsSync true if we should stream synchronize before adding the buffer
   */
  private def addBuffer(
      buffer: RapidsBufferBase,
      needsSync: Boolean): Unit = {
    if (needsSync) {
      Cuda.DEFAULT_STREAM.sync()
    }
    addBuffer(buffer)
  }

  /**
   * The RapidsDeviceMemoryStore is the only store that supports setting a buffer spillable
   * or not.
   */
  override protected def setSpillable(buffer: RapidsBufferBase, spillable: Boolean): Unit = {
    doSetSpillable(buffer, spillable)
  }

  class EventTrackerForColumn(
    val colIx: Int, 
    var refCount: Int, 
    dmb: RapidsDeviceMemoryBatch,
    val wrapped: EventTrackerForColumn = null) 
    extends ai.rapids.cudf.ColumnVector.EventHandler {
    /**
     * Override from the ColumnVector.EventHandler interface.
     *
     * If we are being invoked we have the `ColumnVector` lock, as this callback
     * is being invoked from `ColumnVector.close`
     *
     * @param refCount - the vector's current refCount
     */
    override def onClosed(refCount: Int): Unit = {
      logWarning(s"for column ${this} ${colIx} new refCount is ${refCount}")
      this.refCount = refCount
      // refCount == 1 means only 1 reference exists to `contigBuffer` in the
      // RapidsDeviceMemoryBuffer (we own it)
      if (wrapped != null) {
        logWarning(s"Calling wrapped tracker for ${this} col ${colIx} and refCount ${refCount}")
        wrapped.onClosed(refCount)
      }
      if (refCount == 1) {
        dmb.onColumnSpillable()
      }
    }
  }

  class RapidsDeviceMemoryBatch(
      id: TempSpillBufferId,
      batch: ColumnarBatch,
      spillPriority: Long)
      extends RapidsBufferBase(
        id,
        null,
        spillPriority) {

    logWarning(s"Added RapidsDeviceMemoryBatch ${id}")
    GpuColumnVector.incRefCounts(batch)
    var refCounts = new Array[EventTrackerForColumn](batch.numCols())
    //makeSpillableTrackingBatch(batch)
    private def makeSpillableTrackingBatch(batch: ColumnarBatch): Unit = {
      val cudfCVs = GpuColumnVector.extractBases(batch)
      cudfCVs.zipWithIndex.foreach { case(c, ix) => 
        val tracker = if (c.getEventHandler != null) {
          val tracker = c.getEventHandler.asInstanceOf[EventTrackerForColumn]
          val parentTracker = new EventTrackerForColumn(ix, c.getRefCount(), this, tracker)
          c.setEventHandler(parentTracker)
          tracker
        } else {
          val tracker = new EventTrackerForColumn(ix, c.getRefCount(), this)
          c.setEventHandler(tracker)
          tracker
        }
        refCounts(ix) = tracker
      }
    }

    private def removeTracker(): Unit = {
      val cudfCVs = GpuColumnVector.extractBases(batch)
      cudfCVs.zipWithIndex.foreach { case(c, ix) => 
        val tracker = c.getEventHandler.asInstanceOf[EventTrackerForColumn]
        logWarning(s"Removing tracker ${tracker} for ${this} col ${ix} ${c}")
        c.setEventHandler(tracker.wrapped) // which could be null
      }
    }

    def onColumnSpillable(): Unit = {
      val otherSpillable = refCounts.zipWithIndex.map { case (tracker, ix) => 
        if (tracker == null) {
          true
        } else {
          logWarning(s"column $ix has refcount ${tracker.refCount}")
          tracker.refCount > 1
        }
      }.reduce(_ || _)

      logWarning(s"Making batch spillable? ${!otherSpillable}")

      if (!otherSpillable) {
        setSpillable(this, true)
      }
    }

    /** Release the underlying resources for this buffer. */
    override protected def releaseResources(): Unit = {
      batch.close()
    }

    /** The storage tier for this buffer */
    override val storageTier: StorageTier = StorageTier.DEVICE

    override val supportsChunkedPacker: Boolean = true


    // TODO: need a way to construct the packed chunked split without a user buffer
    // so we can get the packed meta
    val chunkedPacker: ChunkedPacker = {
      val chunkedPacker = new ChunkedPacker(id, batch)
      logInfo(s"chunkedPacker initialized for ${id}. " +
          s"Size is: ${chunkedPacker.getMeta().bufferMeta().size()}")
      chunkedPacker
    }

    override def getMeta(): TableMeta = {
      chunkedPacker.getMeta()
    }
    
    /** The size of this buffer in bytes. */
    override def getSize: Long = {
      chunkedPacker.getMeta().bufferMeta().size()
    }

    /**
     * Get the underlying memory buffer. This may be either a HostMemoryBuffer or a DeviceMemoryBuffer
     * depending on where the buffer currently resides.
     * The caller must have successfully acquired the buffer beforehand.
     *
     * @see [[addReference]]
     * @note It is the responsibility of the caller to close the buffer.
     */
    override def getChunkedPacker: ChunkedPacker = {
      chunkedPacker
    }

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      // TODO: assert that the sparkTypes match the CB types
      val res = GpuColumnVector.incRefCounts(batch)
      //setSpillable(this, false)
      res
    }

    /**
     * Get the underlying memory buffer. This may be either a HostMemoryBuffer or a DeviceMemoryBuffer
     * depending on where the buffer currently resides.
     * The caller must have successfully acquired the buffer beforehand.
     *
     * @see [[addReference]]
     * @note It is the responsibility of the caller to close the buffer.
     */
    override def getMemoryBuffer: MemoryBuffer = {
      throw new UnsupportedOperationException(
        "RapidsDeviceMemoryBatch doesn't support getMemoryBuffer")
    }

    override def free(): Unit = {
      super.free()
      //removeTracker()
      chunkedPacker.close()
    }
  }

  class RapidsDeviceMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      contigBuffer: DeviceMemoryBuffer,
      spillPriority: Long)
      extends RapidsBufferBase(id, meta, spillPriority)
        with MemoryBuffer.EventHandler {

    logWarning(s"Added RapidsDviceMemoryBuffer ${id}")
    override def getSize(): Long = size

    override val storageTier: StorageTier = StorageTier.DEVICE

    // If this require triggers, we are re-adding a `DeviceMemoryBuffer` outside of
    // the catalog lock, which should not possible. The event handler is set to null
    // when we free the `RapidsDeviceMemoryBuffer` and if the buffer is not free, we
    // take out another handle (in the catalog).
    // TODO: This is not robust (to rely on outside locking and addReference/free)
    //  and should be revisited.
    require(contigBuffer.setEventHandler(this) == null,
      "DeviceMemoryBuffer with non-null event handler failed to add!!")

    /**
     * Override from the MemoryBuffer.EventHandler interface.
     *
     * If we are being invoked we have the `contigBuffer` lock, as this callback
     * is being invoked from `MemoryBuffer.close`
     *
     * @param refCount - contigBuffer's current refCount
     */
    override def onClosed(refCount: Int): Unit = {
      // refCount == 1 means only 1 reference exists to `contigBuffer` in the
      // RapidsDeviceMemoryBuffer (we own it)
      if (refCount == 1) {
        // setSpillable is being called here as an extension of `MemoryBuffer.close()`
        // we hold the MemoryBuffer lock and we could be called from a Spark task thread
        // Since we hold the MemoryBuffer lock, `incRefCount` waits for us. The only other
        // call to `setSpillable` is also under this same MemoryBuffer lock (see:
        // `getDeviceMemoryBuffer`)
        setSpillable(this, true)
      }
    }

    override protected def releaseResources(): Unit = synchronized {
      // we need to disassociate this RapidsBuffer from the underlying buffer
      contigBuffer.close()
    }

    /**
     * Get and increase the reference count of the device memory buffer
     * in this RapidsBuffer, while making the RapidsBuffer non-spillable.
     *
     * @note It is the responsibility of the caller to close the DeviceMemoryBuffer
     */
    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = synchronized {
      contigBuffer.synchronized {
        setSpillable(this, false)
        contigBuffer.incRefCount()
        contigBuffer
      }
    }

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      // calling `getDeviceMemoryBuffer` guarantees that we have marked this RapidsBuffer
      // as not spillable and increased its refCount atomically
      withResource(getDeviceMemoryBuffer) { buff =>
        columnarBatchFromDeviceBuffer(buff, sparkTypes)
      }
    }

    /**
     * We overwrite free to make sure we don't have a handler for the underlying
     * contigBuffer, since this `RapidsBuffer` is no longer tracked.
     */
    override def free(): Unit = synchronized {
      if (isValid) {
        // it is going to be invalid when calling super.free()
        contigBuffer.setEventHandler(null)
      }
      super.free()
    }
  }
}
