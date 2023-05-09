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

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, Table}
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

  var bounceBuffer: DeviceMemoryBuffer = DeviceMemoryBuffer.allocate(128L * 1024 * 1024)

  override protected def createBuffer(
      other: RapidsBuffer,
      stream: Cuda.Stream): RapidsBufferBase = {
    val memoryBuffer = withResource(other.getCopyIterator) { copyIterator =>
      copyIterator.next()
    }
    withResource(memoryBuffer) { _ =>
      val deviceBuffer = {
        memoryBuffer match {
          case d: DeviceMemoryBuffer => d
          case h: HostMemoryBuffer =>
            closeOnExcept(DeviceMemoryBuffer.allocate(memoryBuffer.getLength)) { deviceBuffer =>
              logDebug(s"copying from host $h to device $deviceBuffer")
              deviceBuffer.copyFromHostBuffer(h, stream)
              deviceBuffer
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

  /**
   * Adds a table to the device storage.
   *
   * This takes ownership of the table.
   *
   * This function is called only from the RapidsBufferCatalog, under the
   * catalog lock.
   *
   * @param id                   the RapidsBufferId to use for this table
   * @param table                table that will be owned by the store
   * @param initialSpillPriority starting spill priority value
   * @param needsSync            whether the spill framework should stream synchronize while adding
   *                             this table (defaults to true)
   * @return the RapidsBuffer instance that was added.
   */
  def addTable(
      id: TempSpillBufferId,
      table: Table,
      initialSpillPriority: Long,
      needsSync: Boolean): RapidsBuffer = {
    val rapidsBuffer = new RapidsTable(
      id,
      table,
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
                                        val rapidsBuffer: RapidsTable,
                                        columnIx: Int,
                                        var wrapped: Option[RapidsDeviceColumnEventHandler] = None)
      extends ColumnVector.EventHandler {
    override def onClosed(refCount: Int): Unit = {
      if (refCount == 1) {
        rapidsBuffer.onColumnSpillable(columnIx)
        wrapped.foreach(_.onClosed(refCount))
      }
    }
  }

  class RapidsTable(
      id: TempSpillBufferId,
      table: Table,
      spillPriority: Long)
      extends RapidsBufferBase(
        id,
        null,
        spillPriority) {

    registerOnCloseEventHandler()

    private val columnSpillability = Array.fill(table.getNumberOfColumns)(false)

    def onColumnSpillable(columnIx: Int): Unit = {
      columnSpillability(columnIx) = true
      doSetSpillable(this, !columnSpillability.contains(false))
    }

    /** Release the underlying resources for this buffer. */
    override protected def releaseResources(): Unit = {
      table.close()
    }

    /** The storage tier for this buffer */
    override val storageTier: StorageTier = StorageTier.DEVICE

    override val supportsChunkedPacker: Boolean = true

    private var initializedChunkedPacker: Boolean = false

    lazy val chunkedPacker: ChunkedPacker = {
      initializedChunkedPacker = true
      new ChunkedPacker(id, table, bounceBuffer)
    }

    override def getMeta(): TableMeta = {
      chunkedPacker.getMeta()
    }

    // This is the current size in batch form. It is to be used while this
    // table hasn't migrated to another store.
    private val unpackedSizeInBytes: Long = GpuColumnVector.getTotalDeviceMemoryUsed(table)

    override def getMemoryUsedBytes: Long = unpackedSizeInBytes

    override def getPackedSizeBytes: Long = getChunkedPacker.getTotalContiguousSize

    override def getChunkedPacker: ChunkedPacker = {
      chunkedPacker
    }

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      doSetSpillable(this, false)
      GpuColumnVector.from(table, sparkTypes)
    }

    /**
     * Get the underlying memory buffer. This may be either a HostMemoryBuffer or a
     * DeviceMemoryBuffer depending on where the buffer currently resides.
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
      // lets remove our handler from the chain of handlers for each column
      removeOnCloseEventHandler()
      super.free()
      if (initializedChunkedPacker) {
        chunkedPacker.close()
        initializedChunkedPacker = false
      }
    }

    private def registerOnCloseEventHandler(): Unit = {
      val cudfColumns = (0 until table.getNumberOfColumns).map(table.getColumn)

      /**
       * If a batch has two columns that are the same instance, we here associate two callbacks:
       * - The first one for column at index 1
       * - The second one for column at index 2 (which wraps the callback for index 1)
       */
      cudfColumns.zipWithIndex.foreach { case (cv, columnIx) =>
        cv.synchronized {
          val priorEventHandler = cv.getEventHandler.asInstanceOf[RapidsDeviceColumnEventHandler]
          val columnEventHandler =
            new RapidsDeviceColumnEventHandler(
              this,
              columnIx,
              Option(priorEventHandler))
          cv.setEventHandler(columnEventHandler)
        }
      }
    }

    private def removeOnCloseEventHandler(): Unit = {
      val cudfColumns = (0 until table.getNumberOfColumns).map(table.getColumn)

      cudfColumns.foreach { cv =>
        cv.synchronized {
          cv.getEventHandler match {
            case handler: RapidsDeviceColumnEventHandler =>
              // find the event handler that belongs to this rapidsBuffer
              var isHead = true
              var priorEventHandler = handler
              var parent = priorEventHandler
              while (handler.rapidsBuffer != this) {
                isHead = false
                parent = priorEventHandler
                priorEventHandler = priorEventHandler.wrapped.get
              }

              if (isHead) {
                cv.setEventHandler(priorEventHandler.wrapped.orNull)
              } else {
                parent.wrapped = priorEventHandler.wrapped
              }
            case t =>
              throw new IllegalStateException(s"Unknown column event handler $t")
          }
        }
      }
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

    override def getMemoryUsedBytes(): Long = size

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
  override def close(): Unit = {
    super.close()
    bounceBuffer.close()
  }
}
