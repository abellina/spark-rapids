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

import java.util.concurrent.ConcurrentHashMap

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
      registeredBuffer(other.id, deviceBuffer), other.getSpillPriority, other.spillCallback)
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
        registeredBuffer(id, contigBuffer),
        initialSpillPriority,
        spillCallback)) { buffer =>
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
    contigBuffer.incRefCount()
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        size,
        meta,
        None,
        registeredBuffer(id, contigBuffer),
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
        registeredBuffer(id, buffer),
        initialSpillPriority,
        spillCallback)) { buff =>
      logDebug(s"Adding receive side table for: [id=$id, size=${buffer.getLength}, " +
          s"uncompressed=${buff.meta.bufferMeta.uncompressedSize}, " +
          s"meta_id=${tableMeta.bufferMeta.id}, " +
          s"meta_size=${tableMeta.bufferMeta.size}]")
      addDeviceBuffer(buff, needsSync)
    }
  }

  val dmbs = new ConcurrentHashMap[RapidsBufferId, RegisteredDeviceMemoryBuffer]()

  class RegisteredDeviceMemoryBuffer(val id: RapidsBufferId, buffer: DeviceMemoryBuffer)
    extends MemoryBuffer.EventHandler
    with AutoCloseable {

    dmbs.put(id, this)

    buffer.setEventHandler(this)

    override def onClosed(refCount: Int): Unit = {
      logInfo(s"RegisteredDeviceMemoryBuffer ${buffer} closed with ${refCount}")
      if (refCount == 0) {
        dmbs.remove(id)
        logInfo(s"Removed RegisteredDeviceMemoryBuffer ${buffer} from cached: ${dmbs.size()}")
        buffer.setEventHandler(null)
      }
    }

    def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      if (id.tableId == 1) {
        logInfo(s"first one ${buffer.getRefCount}")
      }
      buffer.incRefCount()
      buffer
    }

    override def close(): Unit = {
      buffer.close()
      logInfo(s"At close for ${id} with buffer ref count ${buffer.getRefCount}")
    }

    val aliases = new ConcurrentHashMap[RapidsBufferId, Boolean]()

    def alias(aliasingId: RapidsBufferId): RegisteredDeviceMemoryBuffer = synchronized {
      if (id.tableId == 1) {
        logInfo(s"first one. ${buffer.getRefCount}")
      }
      if (aliases.contains(aliasingId)) {
        throw new IllegalStateException(s"Alias already exists for $id to $aliasingId")
      }
      buffer.incRefCount()
      aliases.put(aliasingId, true)
      logInfo(s"$id is aliased by $aliasingId now. Buffer ref count: ${buffer.getRefCount}")
      this
    }

    def removeAlias(aliasingId: RapidsBufferId) = synchronized {
      aliases.remove(aliasingId)
      logInfo(s"$id no longer aliased by ${aliasingId}. Number of aliases ${aliases.size()}")
      if (aliases.size() == 0) {
        logInfo(s"$id has no aliases left, closing it!")
        close()
      }
    }
  }

  def registeredBuffer(
      aliasingId: RapidsBufferId,
      buffer: DeviceMemoryBuffer): RegisteredDeviceMemoryBuffer = {
    val handler = buffer.getEventHandler
    val registered = handler match {
      case null =>
        buffer.incRefCount()
        new RegisteredDeviceMemoryBuffer(TempSpillBufferId(), buffer)
      case hndr: RegisteredDeviceMemoryBuffer =>
        hndr
    }
    withResource(buffer) { _ =>
      registered.alias(aliasingId)
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

  class RapidsDeviceMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      table: Option[Table],
      contigBuffer: RegisteredDeviceMemoryBuffer,
      spillPriority: Long,
      override val spillCallback: SpillCallback)
      extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback) {
    override val storageTier: StorageTier = StorageTier.DEVICE

    override protected def releaseResources(): Unit = {
      logInfo(s"releaseResources ${id} -- with registered buff ${contigBuffer.id}")
      contigBuffer.close()
      table.foreach(_.close())
      contigBuffer.removeAlias(id)
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      contigBuffer.getDeviceMemoryBuffer
    }

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      withResource(contigBuffer.getDeviceMemoryBuffer) { buff =>
        val startedWith = buff.getRefCount
        var endedWith: Int = -1
        logInfo(s"At getColumnarBatch with ref count ${buff.getRefCount}")
        val res = if (table.isDefined) {
          //REFCOUNT ++ of all columns
          if (buff.getRefCount == 8) {
            logError("this is it")
          }
          logInfo("went .from route")
          buff.incRefCount()
          val r = GpuColumnVectorFromBuffer.from(table.get, buff, meta, sparkTypes)
          endedWith = buff.getRefCount
          if (endedWith == startedWith) {
            logError("WHAT!!")
          }
          r
        } else {
          logInfo("went columnarBatchFromDeviceBuffer route")
          columnarBatchFromDeviceBuffer(buff, sparkTypes)
        }

        endedWith = buff.getRefCount
        require(endedWith > startedWith,
          s"endedWith=$endedWith, startedWith=$startedWith")
        res
      }
    }
  }
}
