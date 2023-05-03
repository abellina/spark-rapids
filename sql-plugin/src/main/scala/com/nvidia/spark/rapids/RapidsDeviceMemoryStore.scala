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
import ai.rapids.cudf.ColumnVector

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
      stream: Cuda.Stream): RapidsBufferBase = {
    val (memoryBuffer, totalCopySize) = withResource(other.getCopyIterator) { copyIterator =>
      copyIterator.next()
    }
    val deviceBuffer = {
      memoryBuffer match {
        case d: DeviceMemoryBuffer => d
        case h: HostMemoryBuffer =>
          withResource(h) { _ =>
            closeOnExcept(DeviceMemoryBuffer.allocate(totalCopySize)) { deviceBuffer =>
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
    GpuColumnVector.incRefCounts(batch)
    val rapidsBuffer = new RapidsDeviceMemoryBatch(
      id,
      batch,
      initialSpillPriority)
    freeOnExcept(rapidsBuffer) { _ =>
      addBuffer(rapidsBuffer, needsSync)
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

  class RapidsDeviceColumnEventHandler(
    val rapidsBuffer: RapidsDeviceMemoryBatch,
    columnIx: Int,
    var wrapped: Option[RapidsDeviceColumnEventHandler] = None)
      extends ColumnVector.EventHandler {
    override def onClosed(refCount: Int): Unit = {
      if (refCount == 1) {
        rapidsBuffer.onColumnSpillable(columnIx)
      }
      wrapped.foreach(_.onClosed(refCount))
    }
  }

  def registerCallbacks(batch: ColumnarBatch, rapidsBuffer: RapidsDeviceMemoryBatch): Unit = {
    val cudfColumns = GpuColumnVector.extractBases(batch)
    val repeated = cudfColumns.distinct.length != cudfColumns.length
    require(!repeated, s"Batch has repeated cols ${cudfColumns.mkString(",")}")
    cudfColumns.zipWithIndex.foreach { case (cv, columnIx) =>
      cv.synchronized {
        val priorEventHandler = cv.getEventHandler.asInstanceOf[RapidsDeviceColumnEventHandler]
        val columnEventHandler =
          new RapidsDeviceColumnEventHandler(
            rapidsBuffer,
            columnIx,
            Option(priorEventHandler))
        cv.setEventHandler(columnEventHandler)
      }
    }
  }

  def removeCallbacks(batch: ColumnarBatch, rapidsBuffer: RapidsDeviceMemoryBatch): Unit = {
    val cudfColumns = GpuColumnVector.extractBases(batch)
    cudfColumns.zipWithIndex.foreach { case (cv, columnIx) =>
      cv.synchronized {
        var priorEventHandler = cv.getEventHandler.asInstanceOf[RapidsDeviceColumnEventHandler]
        // find the event handler that belongs to this rapidsBuffer
        var isHead = true
        var parent = priorEventHandler
        while (priorEventHandler.rapidsBuffer != rapidsBuffer) {
          isHead = false
          parent = priorEventHandler
          priorEventHandler = priorEventHandler.wrapped.get
        }

        if (isHead) {
          cv.setEventHandler(priorEventHandler.wrapped.orNull)
        } else {
          logInfo(s"Removing non-head event handler for ${batch}")
          parent.wrapped = priorEventHandler.wrapped
        }
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

    registerCallbacks(batch, this)

    val columnSpillability = Array.fill(batch.numCols())(false)

    def onColumnSpillable(columnIx: Int): Unit = {
      columnSpillability(columnIx) = true
      val batchSpillable = !columnSpillability.contains(false)
      doSetSpillable(this, batchSpillable)
    }

    /** Release the underlying resources for this buffer. */
    override protected def releaseResources(): Unit = {
      batch.close()
    }

    /** The storage tier for this buffer */
    override val storageTier: StorageTier = StorageTier.DEVICE

    override val supportsChunkedPacker: Boolean = true

    var initializedChunkedPacker: Boolean = false

    lazy val chunkedPacker: ChunkedPacker = {
      initializedChunkedPacker = true
      new ChunkedPacker(id, batch)
    }

    override def getMeta(): TableMeta = {
      chunkedPacker.getMeta()
    }
    
    /** The size of this buffer in bytes. */
    override def getSize: Long = {
      // NOTE: this size is an estimate due to alignment differences
      // the actual size for the contiguous buffer will be available once
      // `chunkedPacker` is instantiated.
      GpuColumnVector.getTotalDeviceMemoryUsed(batch)
    }

    override def getChunkedPacker: ChunkedPacker = {
      chunkedPacker
    }

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      // TODO: assert that the sparkTypes match the CB types
      GpuColumnVector.incRefCounts(batch)
      doSetSpillable(this, false)
      batch
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
      if (initializedChunkedPacker) {
        chunkedPacker.close()
      }
      removeCallbacks(batch, this)
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
