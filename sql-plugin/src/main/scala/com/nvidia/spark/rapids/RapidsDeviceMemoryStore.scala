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

import scala.collection.mutable

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
    new RapidsDeviceMemoryBuffer(other.id, other.size, other.meta, None,
      deviceBuffer, other.getSpillPriority, other.spillCallback)
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
      table: Table,
      contigBuffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback): RapidsBufferHandle  = {
    contigBuffer.getEventHandler match {
      case null =>
        contigBuffer.incRefCount()
        val id = TempSpillBufferId()
        freeOnExcept(
          new RapidsDeviceMemoryBuffer(
            id,
            contigBuffer.getLength,
            tableMeta,
            Some(table),
            contigBuffer,
            initialSpillPriority,
            spillCallback)) { buffer =>
          logDebug(s"Adding table for: [id=$id, size=${buffer.size}, " +
            s"meta_id=${buffer.meta.bufferMeta.id}, meta_size=${buffer.meta.bufferMeta.size}]")
          addDeviceBuffer(buffer, needsSync = true)
        }
      case existing: RapidsBuffer =>
        // existing case
        table.close() // we will not use this
        // TODO: do i need to acquire buffer here?
        withResource(catalog.acquireBuffer(existing.id)) { rapidsBuffer =>
          catalog.makeNewHandle(rapidsBuffer.id, initialSpillPriority)
        }
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
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback,
      needsSync: Boolean = true): RapidsBufferHandle = {
    val contigBuffer = contigTable.getBuffer
    contigBuffer.getEventHandler match {
      case null =>
        addContiguousTable(
          TempSpillBufferId(),
          contigTable,
          initialSpillPriority,
          spillCallback,
          needsSync)
      case existing: RapidsBuffer =>
        // existing case
        // TODO: do i need to acquire buffer here?
        withResource(catalog.acquireBuffer(existing.id)) { rapidsBuffer =>
          catalog.makeNewHandle(rapidsBuffer.id, initialSpillPriority)
        }
    }
  }

  def addContiguousTable(
      id: RapidsBufferId,
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback,
      needsSync: Boolean): RapidsBufferHandle = {
    val contigBuffer = contigTable.getBuffer
    require(contigBuffer.getEventHandler == null,
      "Tried to add a buffer with a pre-existing memory handler!")
    val size = contigBuffer.getLength
    val meta = MetaUtils.buildTableMeta(id.tableId, contigTable)
    contigBuffer.incRefCount()
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        size,
        meta,
        None,
        contigBuffer,
        initialSpillPriority,
        spillCallback)) { buffer =>
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
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback,
      needsSync: Boolean = true): RapidsBufferHandle = {
    buffer.getEventHandler match {
      case null =>
        addBuffer(
          TempSpillBufferId(),
          buffer,
          tableMeta,
          initialSpillPriority,
          spillCallback,
          needsSync)
      case existing: RapidsBuffer =>
        // TODO: do i need to acquire buffer here?
        withResource(catalog.acquireBuffer(existing.id)) { rapidsBuffer =>
          catalog.makeNewHandle(rapidsBuffer.id, initialSpillPriority)
        }
    }
  }

  def addBuffer(
      id: RapidsBufferId,
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback,
      needsSync: Boolean): RapidsBufferHandle = {
    require(buffer.getEventHandler == null,
      "Tried to add a buffer with a pre-existing memory handler!")
    buffer.incRefCount()
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        buffer.getLength,
        tableMeta,
        None,
        buffer,
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
   *
   * @param needsSync true if we should stream synchronize before adding the buffer
   */
  private def addDeviceBuffer(
      buffer: RapidsDeviceMemoryBuffer,
      needsSync: Boolean): RapidsBufferHandle = {
    if (needsSync) {
      Cuda.DEFAULT_STREAM.sync()
    }
    addBufferAndGetHandle(buffer);
  }

  class RapidsDeviceMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      table: Option[Table],
      contigBuffer: DeviceMemoryBuffer,
      spillPriority: Long,
      override val spillCallback: SpillCallback)
      extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback)
      with MemoryBuffer.EventHandler {
    override val storageTier: StorageTier = StorageTier.DEVICE

    val sb = new mutable.StringBuilder()
    Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
      sb.append("    " + stackTraceElement + "\n")
    }
    val myStack = sb.toString()
    logWarning(s"ADDED BUFFER ${id} $contigBuffer with refCount ${contigBuffer.getRefCount()} ${myStack}")

    val prior = contigBuffer.setEventHandler(this)
    if (prior != null) {
      val pdmb = prior.asInstanceOf[RapidsDeviceMemoryBuffer]
      throw new IllegalStateException(
        s"Doubly associating event handler. Previously added in ${pdmb.myStack}")
    }

    override protected def releaseResources(): Unit = {
      logWarning(s"At releaseResources for ${id} with refCount=${contigBuffer.getRefCount}")
      contigBuffer.close()
      table.foreach(_.close())
      logWarning(s"EXIT releaseResources for ${id} with refCount=${contigBuffer.getRefCount}")
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      contigBuffer.incRefCount()
      contigBuffer
    }

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      if (table.isDefined) {
        //REFCOUNT ++ of all columns
        GpuColumnVectorFromBuffer.from(table.get, contigBuffer, meta, sparkTypes)
      } else {
        columnarBatchFromDeviceBuffer(contigBuffer, sparkTypes)
      }
    }

    override def releaseBatch(
        sparkTypes: Array[DataType],
        handle: RapidsBufferHandle): ColumnarBatch = {
      val cb = super.releaseBatch(sparkTypes, handle)
      if (released) {
        logWarning(s"Resetting event handler for ${id} because of releaseBatch. " +
          s"refCount=${refcount} and refcount=${refcount}")
        contigBuffer.setEventHandler(null)
      } else {
        logWarning(s"NOT resetting event handler for ${id} as still aliased")
      }
      cb
    }

    override def onClosed(refCount: Int): Unit = synchronized {
      logWarning(
        s"At onClosed for ${id} with buffer " +
          s"refCount=$refCount and refcount=$refcount " +
          s"hasCache=${cache.isDefined} cacheCount=${cache.map(_.numCols)}")
      val numCachedRefs = cache.map(_.numCols()).getOrElse(0)
      if (refCount - numCachedRefs == 1) {
        logWarning(s"$id: make spillable: Refcount is ${refCount} numCachedRefs is ${numCachedRefs}")
        makeSpillable()
      } else if (refCount == 0) {
        logWarning(s"$id: cannot make spillable: refCount == 0")
        logWarning(s"$id: resetting event handler to null")
        contigBuffer.setEventHandler(null)
      } else {
        logWarning(s"$id: remove spillable: Refcount is ${refCount} numCachedRefs is ${numCachedRefs}")
        removeSpillable()
      }
    }
  }
}
