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

import scala.collection.mutable

/**
 * Buffer storage using device memory.
 * @param catalog catalog to register this store
 */
class RapidsDeviceMemoryStore(catalog: RapidsBufferCatalog = RapidsBufferCatalog.singleton)
    extends RapidsBufferStore(StorageTier.DEVICE, catalog) with Arm {

  // we need to keep a collection of rapids device memory buffers
  // customer id -> rapids device memory buffer
  // internal ones are de-duped always -> buffer tracker tracks those
  // front door (is one of the aliases of a de-duped one)

  // "front door" functions in device memory store deal with front door buffers
  // internal buffers need to map to the front door ones with a hash set or something to
  // figure out the ref count and max(spill priority)

  // cuDF id -> MemoryBuffer class (cuDF id needs to be opened)
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
    val spillable = getOrCreateSpillable(
      deviceBuffer,
      deviceBuffer.getLength,
      other.meta,
      None)
    new RapidsDeviceMemoryBuffer(
      other.id,
      spillable,
      other.size,
      other.meta,
      other.getSpillPriority,
      other.spillCallback)
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
    val spillable = getOrCreateSpillable(
      contigBuffer, contigBuffer.getLength, tableMeta, Some(table))
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        spillable,
        contigBuffer.getLength,
        tableMeta,
        initialSpillPriority,
        spillCallback)) { buffer =>
      logDebug(s"Adding table for: [id=$id, size=${buffer.size}, " +
          s"meta_id=${buffer.meta.bufferMeta.id}, meta_size=${buffer.meta.bufferMeta.size}]")
      addDeviceBuffer(buffer, needsSync = true)
    }
  }


  def getOrCreateSpillable(
    buffer: DeviceMemoryBuffer,
    size: Long,
    meta: TableMeta,
    table: Option[Table]): SpillableDeviceRapidsBuffer = {
    buffer.getEventHandler match {
      case existing: SpillableDeviceRapidsBuffer =>
        existing
      case null =>
        val newSpillable =
          new SpillableDeviceRapidsBuffer(
            TempSpillBufferId(),
            buffer,
            size,
            meta,
            table)
        buffer.setEventHandler(newSpillable)
        // TODO: needs sync?
        Cuda.DEFAULT_STREAM.sync() // just sync for now
        addBuffer(newSpillable)
        newSpillable
      case h: Any =>
        throw new IllegalStateException(s"Invalid handler $h detected")
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
    val spillable = getOrCreateSpillable(contigBuffer, size, meta, None)
    val buffer = new RapidsDeviceMemoryBuffer(
      id,
      spillable,
      size,
      meta,
      initialSpillPriority,
      spillCallback)
    freeOnExcept(buffer) { _ =>
      logDebug(s"Adding table for: [id=$id, size=${buffer.size}, " +
          s"uncompressed=${buffer.meta.bufferMeta.uncompressedSize}, " +
          s"meta_id=${buffer.meta.bufferMeta.id}, meta_size=${buffer.meta.bufferMeta.size}]")
      addDeviceBuffer(buffer, needsSync)
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
    val spillable = getOrCreateSpillable(buffer, buffer.getLength, tableMeta, None)
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        spillable,
        buffer.getLength,
        tableMeta,
        initialSpillPriority,
        spillCallback)) { buff =>
      logDebug(s"Adding receive side table for: [id=$id, size=${buffer.getLength}, " +
          s"uncompressed=${buff.meta.bufferMeta.uncompressedSize}, " +
          s"meta_id=${tableMeta.bufferMeta.id}, " +
          s"meta_size=${tableMeta.bufferMeta.size}]")
      addDeviceBuffer(buff, needsSync)
    }
  }

  /**
   * Adds a device buffer to the spill framework, stream synchronizing with the producer
   * stream to ensure that the buffer is fully materialized, and can be safely copied
   * as part of the spill.
   * @param needsSync true if we should stream synchronize before adding the buffer
   */
  private def addDeviceBuffer(buffer: RapidsDeviceMemoryBuffer, needsSync: Boolean): Unit = {
    if (needsSync) {
      Cuda.DEFAULT_STREAM.sync()
    }
    addBuffer(buffer);
  }
  // SpillableWrapper
  /// id
  /// has a set of RapidsBufferIds that are associated with me
  /// has reference to DeviceMemoryBuffer


  class SpillableDeviceRapidsBuffer(
      id: RapidsBufferId,
      memoryBuffer: DeviceMemoryBuffer,
      size: Long,
      meta: TableMeta,
      table: Option[Table])
    extends RapidsBufferBase(id, size, meta, -1, new SpillCallback() {
      override def apply(from: StorageTier, to: StorageTier, amount: Long): Unit = {
        logInfo(s"At spill callback for ${id}")
      }
      override def semaphoreWaitTime: GpuMetric = NoopMetric
    })
      with MemoryBuffer.EventHandler
      with Spillable {

    // ensure we own it
    memoryBuffer.incRefCount()

    override def toString: String = s"ID: $id spillable $name buffer size=$size"

    val aliases = new mutable.HashSet[RapidsBufferId]

    def registerAlias(buffer: RapidsBufferBase): Unit = {
      aliases.add(buffer.id)
    }

    override def addReference(): Boolean = {
      val added = super.addReference()
      if (added) {
        getMemoryBufferInternal.foreach(_.incRefCount())
        removeSpillable(this)
      }
      added
    }

    override def close(): Unit = {
      super.close()
      if (isValid) {
        getMemoryBufferInternal.foreach(_.close())
      }
    }

    override def onClosed(refCount: Int): Unit = {
      if (refCount == 1) {
        makeSpillable(this)
      } else {
        removeSpillable(this)
      }
    }

    override def releaseResources(): Unit = {
      memoryBuffer.close()
      table.foreach(_.close())
      memoryBuffer.setEventHandler(null)
    }

    override def getMemoryBufferInternal: Option[MemoryBuffer] = {
      Some(memoryBuffer)
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      memoryBuffer.incRefCount()
      memoryBuffer
    }

    override val storageTier: StorageTier = StorageTier.DEVICE

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      if (table.isDefined) {
        //REFCOUNT ++ of all columns
        GpuColumnVectorFromBuffer.from(table.get, memoryBuffer, meta, sparkTypes)
      } else {
        columnarBatchFromDeviceBuffer(memoryBuffer, sparkTypes)
      }
    }
  }

  // TODO: only store should create one of this, make it protected
  class RapidsDeviceMemoryBuffer(
    id: RapidsBufferId,
    spillable: SpillableDeviceRapidsBuffer,
    size: Long,
    meta: TableMeta,
    // the actual  priority becomes the MAX of all aliased buffers at the time
    spillPriority: Long,
    // spill callback is problematic, pick one arbitrarily for now
    override val spillCallback: SpillCallback)
    // in the future SpillCallback may be a bag  of metrics or something different PR
      extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback)
      with AliasRapidsBuffer {

    override def toString: String = s"ID: $id alias of (${spillable.toString}) size=$size"

    spillable.registerAlias(this)

    override def addReference(): Boolean = spillable.addReference()

    override def close(): Unit = spillable.close()

    override val storageTier: StorageTier = StorageTier.DEVICE

    override def getMemoryBufferInternal: Option[MemoryBuffer] =
      spillable.getMemoryBufferInternal

    override protected def releaseResources(): Unit = {
      spillable.releaseResources()
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
