/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import ai.rapids.cudf.{ContiguousTable, Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, Table}
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.sql.rapids.TempSpillBufferId
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Buffer storage using device memory.
 * @param catalog catalog to register this store
 */
class RapidsDeviceMemoryStore(catalog: RapidsBufferCatalog = RapidsBufferCatalog.singleton)
    extends RapidsBufferStore(StorageTier.DEVICE, catalog) with Arm {

  override protected def createBuffer(other: RapidsBuffer, memoryBuffer: MemoryBuffer,
      stream: Cuda.Stream): RapidsBufferBase = {
    val deviceBuffer = {
      memoryBuffer match {
        case d: DeviceMemoryBuffer => d
        case h: HostMemoryBuffer =>
          withResource(h) { _ =>
            closeOnExcept(DeviceMemoryBuffer.allocate(other.size)) { deviceBuffer =>
              logDebug(s"copying from host $h to device $deviceBuffer")
              deviceBuffer.copyFromHostBuffer(h, stream)
              deviceBuffer
            }
          }
        case b => throw new IllegalStateException(s"Unrecognized buffer: $b")
      }
    }
    val spillBuffer = getOrElseCreateBuffer(
      deviceBuffer,
      other.meta,
      None,
      other.getSpillPriority,
      other.spillCallback,
      needsSync = false)
    val facadeBuffer = new RapidsDeviceMemoryBuffer(
      other.id,
      other.size,
      other.meta,
      None,
      spillBuffer,
      other.getSpillPriority,
      other.spillCallback)
    addFacadeBuffer(facadeBuffer)
    facadeBuffer
  }

  def getOrElseCreateBuffer(
      contigBuffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      table: Option[Table],
      initialSpillPriority: Long,
      spillCallback: SpillCallback,
      needsSync: Boolean = false): RapidsDeviceSpillableMemoryBuffer = {
    val internalId = TempSpillBufferId()
    val spillBuffer = new RapidsDeviceSpillableMemoryBuffer(
      internalId,
      contigBuffer.getLength,
      tableMeta,
      contigBuffer,
      table,
      initialSpillPriority,
      spillCallback)
    addDeviceBuffer(spillBuffer, needsSync)
    spillBuffer
  }

  /**
   * Adds a contiguous table to the device storage, taking ownership of the table.
   * @param id buffer ID to associate with this buffer
   * @param table cudf table based from the contiguous buffer
   * @param contigBuffer device memory buffer backing the table
   * @param tableMeta metadata describing the buffer layout
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   */
  def addTable(
      id: RapidsBufferId,
      table: Table,
      contigBuffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback): Unit = {

    val spillBuffer = getOrElseCreateBuffer(
      contigBuffer,
      tableMeta,
      Some(table),
      initialSpillPriority,
      spillCallback,
      needsSync = true)
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        contigBuffer.getLength,
        tableMeta,
        Some(table),
        spillBuffer,
        initialSpillPriority,
        spillCallback)) { buffer =>
      logDebug(s"Adding table for: [id=$id, size=${buffer.size}, " +
          s"meta_id=${buffer.meta.bufferMeta.id}, meta_size=${buffer.meta.bufferMeta.size}]")
      addFacadeBuffer(buffer)
    }
  }

  /**
   * Adds a contiguous table to the device storage. This does NOT take ownership of the
   * contiguous table, so it is the responsibility of the caller to close it. The refcount of the
   * underlying device buffer will be incremented so the contiguous table can be closed before
   * this buffer is destroyed.
   * @param id buffer ID to associate with this buffer
   * @param contigTable contiguous table to track in storage
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @param needsSync whether the spill framework should stream synchronize while adding
   *                  this device buffer (defaults to true)
   */
  def addContiguousTable(
      id: RapidsBufferId,
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback,
      needsSync: Boolean = true): Unit = {
    val contigBuffer = contigTable.getBuffer
    val size = contigBuffer.getLength
    val meta = MetaUtils.buildTableMeta(id.tableId, contigTable)
    contigBuffer.incRefCount()
    val spillBuffer = getOrElseCreateBuffer(
      contigBuffer,
      meta,
      None,
      initialSpillPriority,
      spillCallback,
      needsSync)
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        size,
        meta,
        None,
        spillBuffer,
        initialSpillPriority,
        spillCallback)) { buffer =>
      logDebug(s"Adding table for: [id=$id, size=${buffer.size}, " +
          s"uncompressed=${buffer.meta.bufferMeta.uncompressedSize}, " +
          s"meta_id=${buffer.meta.bufferMeta.id}, meta_size=${buffer.meta.bufferMeta.size}]")
      addFacadeBuffer(buffer)
    }
  }

  /**
   * Adds a buffer to the device storage, taking ownership of the buffer.
   * @param id buffer ID to associate with this buffer
   * @param buffer buffer that will be owned by the store
   * @param tableMeta metadata describing the buffer layout
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @param needsSync whether the spill framework should stream synchronize while adding
   *                  this device buffer (defaults to true)
   */
  def addBuffer(
      id: RapidsBufferId,
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback,
      needsSync: Boolean = true): Unit = {

    val spillBuffer = getOrElseCreateBuffer(
      buffer,
      tableMeta,
      None,
      initialSpillPriority,
      spillCallback,
      needsSync)
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        buffer.getLength,
        tableMeta,
        None,
        spillBuffer,
        initialSpillPriority,
        spillCallback)) { buff =>
      logDebug(s"Adding receive side table for: [id=$id, size=${buffer.getLength}, " +
          s"uncompressed=${buff.meta.bufferMeta.uncompressedSize}, " +
          s"meta_id=${tableMeta.bufferMeta.id}, " +
          s"meta_size=${tableMeta.bufferMeta.size}]")
      addFacadeBuffer(buff)
    }
  }

  private def addFacadeBuffer(facadeBuff: RapidsDeviceMemoryBuffer): Unit = {
    addBuffer(facadeBuff);
  }

  /**
   * Adds a device buffer to the spill framework, stream synchronizing with the producer
   * stream to ensure that the buffer is fully materialized, and can be safely copied
   * as part of the spill.
   * @param needsSync true if we should stream synchronize before adding the buffer
   */
  private def addDeviceBuffer(
    buffer: RapidsDeviceSpillableMemoryBuffer, needsSync: Boolean): Unit = {
    if (needsSync) {
      Cuda.DEFAULT_STREAM.sync()
    }
    addBuffer(buffer);
  }

  class RapidsDeviceSpillableMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      contigBuffer: DeviceMemoryBuffer,
      table: Option[Table],
      spillPriority: Long,
      override val spillCallback: SpillCallback)
    extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback) {

    override def releaseResources(): Unit = {
      contigBuffer.close()
      table.foreach(_.close())
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      contigBuffer.incRefCount()
      contigBuffer
    }

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      if (table.isDefined) {
        //REFCOUNT ++ of all columns
        GpuColumnVectorFromBuffer.from(table.get, contigBuffer, meta, sparkTypes)
      } else {
        columnarBatchFromDeviceBuffer(contigBuffer, sparkTypes)
      }
    }

    /** The storage tier for this buffer */
    override val storageTier: StorageTier = StorageTier.DEVICE

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer
  }

  class RapidsDeviceMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      table: Option[Table],
      spillable: RapidsDeviceSpillableMemoryBuffer,
      spillPriority: Long,
      override val spillCallback: SpillCallback)
      extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback) {
    override val storageTier: StorageTier = StorageTier.DEVICE

    override def isSpillable(): Boolean = false

    override protected def releaseResources(): Unit = {
      logInfo(s"releaseResources called on facade... $id")
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      spillable.getDeviceMemoryBuffer
    }

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      spillable.getColumnarBatch(sparkTypes)
    }

    override def free(): Unit = synchronized {
      spillable.free()
      super.free()
    }
  }
}
