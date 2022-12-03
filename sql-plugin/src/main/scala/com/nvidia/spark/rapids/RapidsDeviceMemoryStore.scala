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
import scala.collection.mutable

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
    logInfo(s"CREATE BUFFER!! ${other.id}")
    val (added, registered) = registeredBuffer(other.id, deviceBuffer, "createBuffer")
    new RapidsDeviceMemoryBuffer(
      other.id,
      other.size,
      other.meta,
      None,
      registered,
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
    val (added, registered) = registeredBuffer(id, contigBuffer, "addTable")
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        contigBuffer.getLength,
        tableMeta,
        Some(table),
        registered,
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
    println(s"addContiguousTable refCount is ${contigBuffer.getRefCount}")
    val size = contigBuffer.getLength
    val meta = MetaUtils.buildTableMeta(id.tableId, contigTable)
    val (added, registered) = registeredBuffer(id, contigBuffer, "addContiguousTable")

    // add always? if we don't we segfault
    contigBuffer.incRefCount()
    println(s"addContiguousTable after incRefCount refCount is ${contigBuffer.getRefCount}")
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        size,
        meta,
        None,
        registered,
        initialSpillPriority,
        spillCallback)) { buffer =>
      logDebug(s"Adding table for: [id=$id, size=${buffer.size}, " +
          s"uncompressed=${buffer.meta.bufferMeta.uncompressedSize}, " +
          s"meta_id=${buffer.meta.bufferMeta.id}, meta_size=${buffer.meta.bufferMeta.size}]")
      addDeviceBuffer(buffer, needsSync)
    }
    println(s"addContiguousTable after adding refCount is ${contigBuffer.getRefCount}")
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
    val (added, registered) = registeredBuffer(id, buffer, "addBuffer")
    freeOnExcept(
      new RapidsDeviceMemoryBuffer(
        id,
        buffer.getLength,
        tableMeta,
        None,
        registered,
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
    def getRefCount = buffer.getRefCount

    val sb = new mutable.StringBuilder()
    Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
      sb.append("    " + stackTraceElement + "\n")
    }
    val myStackTrace = sb.toString
    logInfo(s"Adding RegisteredDeviceMemoryBuffer ${buffer} to cached as ${id} ")
    dmbs.put(id, this)
    if (dmbs.size > 10) {
      logInfo(s"$id: ${myStackTrace}")
    }
    require(null == buffer.setEventHandler(this), "Overwrote an event handler!!")

    override def onClosed(refCount: Int): Unit = {
      logInfo(s"RegisteredDeviceMemoryBuffer ${buffer} closed with ${refCount}")
      if (refCount == 0) {
        val sb = new mutable.StringBuilder()
        Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
          sb.append("    " + stackTraceElement + "\n")
        }
        println(s"I just got closed ${id} ${sb.toString()}")
        dmbs.remove(id)
        val idsLeft = new mutable.ArrayBuffer[(RapidsBufferId, RegisteredDeviceMemoryBuffer)]()
        dmbs.forEach((k, v) => {
          idsLeft.append((k, v))
        })
        logInfo(s"Removed RegisteredDeviceMemoryBuffer ${buffer} from cached: ${dmbs.size()} " +
          s"$idsLeft")
        require(this == buffer.setEventHandler(null), "Stumped on an event handler that wasn't mine!!")
      }
    }

    def getDeviceMemoryBuffer: DeviceMemoryBuffer = synchronized {
      if (id.tableId == 1) {
        logInfo(s"first one ${buffer.getRefCount}")
      }
      buffer.incRefCount()
      buffer
    }

    override def close(): Unit = synchronized {
      buffer.close()
      logInfo(s"At close for ${id} with buffer ref count ${buffer.getRefCount}")
    }

    val aliases = new ConcurrentHashMap[RapidsBufferId, String]()

    def alias(aliasingId: RapidsBufferId,
              how: String): RegisteredDeviceMemoryBuffer = synchronized {
      if (aliases.contains(aliasingId)) {
        throw new IllegalStateException(s"Alias already exists for $id to $aliasingId")
      }
      aliases.put(aliasingId, how)
      //buffer.incRefCount()
      if (aliases.size == 1) {
        println(s"first alias ${id} being aliased by ${aliasingId}. " +
          s"Num aliases ${aliases.size()}. Buff ref count: ${getRefCount}")
      }
      if (aliases.size > 1) {
        println(s"already existing ${id} being aliased by ${aliasingId}. " +
          s"Num aliases ${aliases.size()}. Buff ref count: ${getRefCount}")
      }
      logInfo(s"$id is aliased by $aliasingId, via $how. Buffer ref count: ${buffer.getRefCount}")
      this
    }

    def removeAlias(aliasingId: RapidsBufferId) = synchronized {
      val how: String = aliases.remove(aliasingId)
      logInfo(s"$id no longer aliased by ${aliasingId} ($how). Number of aliases ${aliases.size()}." +
        s"buffer refCount=${getRefCount}")
      //close()
      //println(s"Closed, as standard for all removeAlias. ${getRefCount} ")
      if (aliases.size() == 0) {
        logInfo(s"$id has no aliases left") //, closing it!")

        println(s"Closing more time, aliases.size() == 0. ${getRefCount} ")
        close() // final close
      }
    }
  }

  def registeredBuffer(
      aliasingId: RapidsBufferId,
      buffer: DeviceMemoryBuffer,
      how: String): (Boolean, RegisteredDeviceMemoryBuffer) = {
    buffer.incRefCount()
    withResource(buffer) { _ =>
      val handler = buffer.getEventHandler
      var added = false
      val registered = handler match {
        case null =>
          // TODO: do I need this? buffer.incRefCount()
          added = true
          val reg = new RegisteredDeviceMemoryBuffer(TempSpillBufferId(), buffer)
          println(s"it is new, created registered, refCount is ${buffer.getRefCount}")
          reg
        case hndr: RegisteredDeviceMemoryBuffer =>
          hndr
      }
      (added, registered.alias(aliasingId, how))
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

    var released = false
    override protected def releaseResources(): Unit = contigBuffer.synchronized {
      println(s"releaseResources ${this.id}")
      if (released) {
        throw new IllegalStateException(s"Already released ${id} which aliases ${contigBuffer.id}")
      }
      released = true
      logInfo(s"releaseResources ${id} -- with registered buff ${contigBuffer.id}")
      println(s"closing table ${contigBuffer.getRefCount}")
      table.foreach(_.close())
      println(s"removing alias ${contigBuffer.getRefCount}")
      contigBuffer.removeAlias(id)
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      contigBuffer.getDeviceMemoryBuffer
    }

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(
      sparkTypes: Array[DataType]): ColumnarBatch = contigBuffer.synchronized {
      withResource(contigBuffer.getDeviceMemoryBuffer) { buff =>
        val startedWith = buff.getRefCount
        var endedWith: Int = -1
        logInfo(s"At getColumnarBatch with ref count ${buff.getRefCount}")
        val res = if (table.isDefined) {
          //REFCOUNT ++ of all columns, not necessarily the underlying buffer
          logInfo("went .from route")
          //buff.incRefCount()
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
        //require(endedWith > startedWith,
        //  s"endedWith=$endedWith, startedWith=$startedWith")
        res
      }
    }
  }
}
