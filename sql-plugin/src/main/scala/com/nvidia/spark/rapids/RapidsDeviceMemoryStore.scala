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

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch


trait MemoryStoreHandler {
  def onSpillStoreSizeChange(delta: Long): Unit
}
/**
 * Buffer storage using device memory.
 * @param catalog catalog to register this store
 */
class RapidsDeviceMemoryStore(catalog: RapidsBufferCatalog = RapidsBufferCatalog.singleton)
    extends RapidsBufferStore(StorageTier.DEVICE, catalog) with Arm {

  var memoryStoreHandler: MemoryStoreHandler = _
  def setEventHandler(handler: MemoryStoreHandler) = {
    memoryStoreHandler = handler
  }

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
    new RapidsDeviceMemoryBuffer(other.id, other.size, other.meta, None,
      deviceBuffer, other.getSpillPriority, other.spillCallback,
      memoryStoreHandler)
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
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        contigBuffer.getLength,
        tableMeta,
        Some(table),
        contigBuffer,
        initialSpillPriority,
        spillCallback,
        memoryStoreHandler)) { buffer =>
      logDebug(s"Adding table for: [id=$id, size=${buffer.size}, " +
          s"meta_id=${buffer.meta.bufferMeta.id}, meta_size=${buffer.meta.bufferMeta.size}]")
      addDeviceBuffer(buffer, needsSync = true)
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
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        size,
        meta,
        None,
        contigBuffer,
        initialSpillPriority,
        spillCallback,
        memoryStoreHandler)) { buffer =>
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
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        buffer.getLength,
        tableMeta,
        None,
        buffer,
        initialSpillPriority,
        spillCallback,
        memoryStoreHandler)) { buff =>
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
    //TODO: memoryStoreHandler.onSpillStoreSizeChange(buffer.size)
  }

  class RapidsDeviceMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      table: Option[Table],
      contigBuffer: DeviceMemoryBuffer,
      spillPriority: Long,
      override val spillCallback: SpillCallback,
      memoryStoreHandler: MemoryStoreHandler)
      extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback)
        with MemoryBuffer.EventHandler {
    logInfo(s"Adding buffer ${contigBuffer} with orig refCount = ${contigBuffer.getRefCount}")
    override val storageTier: StorageTier = StorageTier.DEVICE

    // we now own the buffer, the caller will close
    // refcount will be 2 momentarily, and will return to 1 shortly
    // maybe we should just ensure this is refcount 1 so we stop with
    // withResource shananigans
    val lease = contigBuffer.slice(0, contigBuffer.getLength)
    require(null == lease.setEventHandler(this)) // starts at refcount 1

    def isLeased() = lease.getRefCount > 1

    def getLease(): DeviceMemoryBuffer = {
      removeSpillable(this)
      lease.incRefCount
      logInfo(s"getLease refCount ${lease.getRefCount}, refcount=$refcount " +
        s"${id} ${lease} isLeased: ${isLeased} ${printStackTrace()}")
      lease
    }

    override def onClosed(refCount: Int): Unit = synchronized {
      logInfo(s"onClosed!! refCount ${refCount} ${id} ${lease} isLeased: ${isLeased}")
      if (refCount == 1) {
        makeSpillable(this)
      } else {
        // make sure we removed ourselves
        removeSpillable(this)
      }
    }

    override protected def releaseResources(): Unit = synchronized {
      logInfo(s"at releaseResources ${id}")
      // should be last lease
      table.foreach(_.close())
      // spill store has removed this, we need to discount
      // TODO: memoryStoreHandler.onSpillStoreSizeChange(-1L * size)
      //removeSpillable(this)
      lease.close()
      //require(!isLeased(),
      //  s"lease refcount > 0 (refcount=${lease.getRefCount}) for ${id} at releaseResources")
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = synchronized {
      getLease()
    }

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      withResource(getDeviceMemoryBuffer) { clone =>
        if (table.isDefined) {
          //REFCOUNT ++ of all columns
          GpuColumnVectorFromBuffer.from(table.get, clone, meta, sparkTypes)
        } else {
          columnarBatchFromDeviceBuffer(clone, sparkTypes)
        }
      }
    }

    override def addReference(): Boolean = synchronized {
      logInfo(s"Acquiring RapidsBuffer ${id}: ${refcount}")
      // refcount for contigBuffer cou
      val res = super.addReference()
      // TODO: fix memoryStoreHandler.onSpillStoreSizeChange(-1L * size)
      logInfo(s"Removing bc addReference ${id}")
      // this buffer is no longer a candidate for spilling
      removeSpillable(this)
      res
    }

    override def close(): Unit = synchronized {
      super.close()
      logInfo(s"At close() RapidsBuffer ${id}: refcount=${refcount}. ${printStackTrace}")
      if (refcount == 0) {
        if (!isLeased) {
          logInfo(s"Making ${id} spillable")
          makeSpillable(this)
        }
        // it now is not acquired, we want to mark this as spillable
        //TODO: fix memoryStoreHandler.onSpillStoreSizeChange(1L * size)
      }
    }
  }
}
