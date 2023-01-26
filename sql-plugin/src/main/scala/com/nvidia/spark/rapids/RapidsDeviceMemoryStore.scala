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

import ai.rapids.cudf.{ContiguousTable, Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import alluxio.collections.ConcurrentHashSet
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

  // The RapidsDeviceMemoryStore handles spillability via ref counting
  override protected def spillableOnAdd: Boolean = false

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
    new RapidsDeviceMemoryBuffer(
      other.id,
      other.size,
      other.meta,
      deviceBuffer,
      other.getSpillPriority,
      other.getSpillCallback)
  }

  /**
   * Given a `DeviceMemoryBuffer` find out if a `MemoryBuffer.EventHandler` is associated
   * with it. If so, make sure that it is an instance of `RapidsDeviceMemoryBuffer`.
   *
   * After getting the `RapidsDeviceMemoryBuffer` try and keep it alive via `addReference`.
   * If successful, we can point to this buffer with a new handle, otherwise the buffer is
   * about to be removed/freed (unlikely, because we are holding onto the reference as we
   * are adding it again).
   *
   * This method must be invoked with `buffer`'s lock already held! (except in tests)
   *
   * @note public for testing
   * @param buffer - the `DeviceMemoryBuffer` to inspect
   * @return - Some(RapidsDeviceMemoryBuffer): the handler is associated with a rapids buffer
   *         and the rapids buffer is currently valid, or
   *
   *         - None: if no `RapidsDeviceMemoryBuffer` is associated with this buffer (it is
   *         brand new to the store, or the `RapidsDeviceMemoryBuffer` is invalid and
   *         about to be removed).
   */
  def getExistingRapidsBufferAndAcquire(
      buffer: DeviceMemoryBuffer): Option[RapidsDeviceMemoryHandler] = {
    var done = false
    var eventHandler: RapidsDeviceMemoryHandler = null
    while(!done) {
      val eh = buffer.getEventHandler
      val rapidsBuffer = eh match {
        case null =>
          None
        case rapidsBuffer: RapidsDeviceMemoryHandler =>
          Some(rapidsBuffer)
        case _ =>
          throw new IllegalStateException("Unknown event handler")
      }
      // lock free so far
      if (rapidsBuffer.isEmpty) {
        if (buffer.getEventHandler != null) {
          // we need to loop back, we now have a handler
        } else {
          // we don't have an event handler set, and now we can create one
          val eventHandler = new RapidsDeviceMemoryHandler
          buffer.setEventHandler(eventHandler)
          done = true
        }
      } else {
        if (buffer.getEventHandler == null) {
          // handler was unset on us
          done = true
        } else if (buffer.getEventHandler == rapidsBuffer.get) {
          // the handler matches what we found
        } else {
          // the handler is different? should throw

          require(contigBuffer.setEventHandler(this) == null,
            "Attempted to associate a device buffer with a memory handler, but it was " +
              "already associated with one.")
        }
      }
    }

    val eh = buffer.getEventHandler
    eh match {
      case null =>
        None
      case rapidsBuffer: RapidsDeviceMemoryBuffer =>
        if (rapidsBuffer.addReference()) {
          Some(rapidsBuffer)
        } else {
          // if we raced with `RapidsBufferBase.free` and lost, we'd end up here
          // because it means the RapidsBuffer is no longer valid and the event handler
          // was reset.
          None
        }
      case _ =>
        throw new IllegalStateException("Unknown event handler")
    }
  }

  val deviceBufferToId = new ConcurrentHashSet[DeviceMemoryBuffer, RapidsBufferId]()

  /**
   * Adds a contiguous table to the device storage. This does NOT take ownership of the
   * contiguous table, so it is the responsibility of the caller to close it. The refcount of the
   * underlying device buffer will be incremented so the contiguous table can be closed before
   * this buffer is destroyed.
   *
   * This version of `addContiguousTable` creates a `TempSpillBufferId` to use
   * to refer to this table.
   *
   * @param contigTable contiguous table to track in storage
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @param needsSync whether the spill framework should stream synchronize while adding
   *                  this device buffer (defaults to true)
   * @return RapidsBufferHandle handle for this table
   */
  def addContiguousTable(
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback,
      needsSync: Boolean = true): RapidsBufferHandle = {
    val underlyingBuffer = contigTable.getBuffer
    var handle: RapidsBufferHandle = null
    deviceBufferToId.computeIfAbsent(underlyingBuffer, (_, existingId) => {
      var id: RapidsBufferId = existingId
      if (id == null) {
        id = TempSpillBufferId()
        addContiguousTable(
          id,
          contigTable,
          initialSpillPriority,
          spillCallback,
          needsSync)
      }
      handle = catalog.makeNewHandle(id, initialSpillPriority, spillCallback)
      id
    })
    handle
  }

  /**
   * Adds a contiguous table to the device storage. This does NOT take ownership of the
   * contiguous table, so it is the responsibility of the caller to close it. The refcount of the
   * underlying device buffer will be incremented so the contiguous table can be closed before
   * this buffer is destroyed.
   *
   * @param id the RapidsBufferId to use for this buffer
   * @param contigTable contiguous table to track in storage
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @param needsSync whether the spill framework should stream synchronize while adding
   *                             this device buffer (defaults to true)
   * @return RapidsBufferHandle handle for this table
   */
  def addContiguousTable(
      id: RapidsBufferId,
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback,
      needsSync: Boolean): Unit = {
    val contigBuffer = contigTable.getBuffer
    val size = contigBuffer.getLength
    val meta = MetaUtils.buildTableMeta(id.tableId, contigTable)
    contigBuffer.incRefCount()
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        size,
        meta,
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
   * Adds a buffer to the device storage. This does NOT take ownership of the
   * buffer, so it is the responsibility of the caller to close it.
   *
   * This version of `addBuffer` creates a `TempSpillBufferId` to use to refer to
   * this buffer.
   *
   * @param buffer buffer that will be owned by the store
   * @param tableMeta metadata describing the buffer layout
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @param needsSync whether the spill framework should stream synchronize while adding
   *                  this device buffer (defaults to true)
   * @return RapidsBufferHandle handle for this buffer
   */
  def addBuffer(
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback,
      needsSync: Boolean = true): RapidsBufferHandle = {
    var handle: RapidsBufferHandle = null
    deviceBufferToId.computeIfAbsent(buffer, (_, existingId) => {
      var id: RapidsBufferId = existingId
      if (id == null) {
        id = TempSpillBufferId()
        addBuffer(
          TempSpillBufferId(),
          buffer,
          tableMeta,
          initialSpillPriority,
          spillCallback,
          needsSync)
      }
      handle = catalog.makeNewHandle(id, initialSpillPriority, spillCallback)
      id
    })
    handle
  }

  /**
   * Adds a buffer to the device storage. This does NOT take ownership of the
   * buffer, so it is the responsibility of the caller to close it.
   *
   * @param id the RapidsBufferId to use for this buffer
   * @param buffer buffer that will be owned by the store
   * @param tableMeta metadata describing the buffer layout
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @param needsSync whether the spill framework should stream synchronize while adding
   *                  this device buffer (defaults to true)
   * @return RapidsBufferHandle handle for this RapidsBuffer
   */
  def addBuffer(
      id: RapidsBufferId,
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback,
      needsSync: Boolean): Unit= {
    buffer.incRefCount()
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        buffer.getLength,
        tableMeta,
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

  class RapidsDeviceMemoryBuffer(
      id: RapidsBufferId,
      size: Long,
      meta: TableMeta,
      contigBuffer: DeviceMemoryBuffer,
      spillPriority: Long,
      spillCallback: SpillCallback)
      extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback)
        with MemoryBuffer.EventHandler {

    contigBuffer.setEventHandler(this)

    override val storageTier: StorageTier = StorageTier.DEVICE

    /**
     * Override from the MemoryBuffer.EventHandler interface.
     *
     * If we are being invoked we have the `contigBuffer` lock, as this callback
     * is being invoked from `MemoryBuffer.close`
     *
     * @param refCount - contigBuffer's current refCount
     */
    override def onClosed(refCount: Int): Unit = {
      logDebug(s"onClosed for $id refCount=$refCount")

      // refCount == 1 means o  class RapidsDeviceMemoryHandler extends MemoryBuffer.EventHandler {
    var rapidsBuffer: RapidsDeviceMemoryBuffer = null

    def setBuffer(rb: RapidsDeviceMemoryBuffer) = rapidsBuffer = rb

  }

nly 1 reference exists to `contigBuffer` in the
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

    override protected def releaseResources(): Unit = {
      // we need to disassociate this RapidsBuffer from the underlying buffer
      contigBuffer.close()
    }

    /**
     *  We overwrite setInvalid to make sure we don't have a handler for the underlying
     *  contigBuffer, since this `RapidsBuffer` is no longer tracked.
     */
    override protected def setInvalid(): Unit = synchronized {
      super.setInvalid()
      deviceBufferToId.compute(contigBuffer, (_, _) => {
        contigBuffer.setEventHandler(null)
        null
      })
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
  }
}
