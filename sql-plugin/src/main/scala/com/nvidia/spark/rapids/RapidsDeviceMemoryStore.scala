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
    val (added, registered) = registeredBuffer(
      other.id,
      deviceBuffer,
      other.meta,
      other.getSpillPriority,
      other.spillCallback,
      "createBuffer")

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
    val (added, registered) = registeredBuffer(
      id,
      contigBuffer,
      tableMeta,
      initialSpillPriority,
      spillCallback,
      "addTable")
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
    logInfo(s"addContiguousTable ${id} refCount is ${contigBuffer.getRefCount}")
    val size = contigBuffer.getLength
    val meta = MetaUtils.buildTableMeta(id.tableId, contigTable)
    val (added, registered) = registeredBuffer(
      id,
      contigBuffer,
      meta,
      initialSpillPriority,
      spillCallback,
      "addContiguousTable")

    // add always? if we don't we segfault
    //contigBuffer.incRefCount()
    println(s"addContiguousTable ${id} after incRefCount refCount is ${contigBuffer.getRefCount}")
    logInfo(s"addContiguousTable ${id} after incRefCount refCount is ${contigBuffer.getRefCount}")
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
    val (added, registered) = registeredBuffer(
      id,
      buffer,
      tableMeta,
      initialSpillPriority,
      spillCallback,
      "addBuffer")
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

  //val dmbs = new ConcurrentHashMap[RapidsBufferId, RegisteredDeviceMemoryBuffer]()

  class RegisteredDeviceMemoryBuffer(
      override val id: RapidsBufferId,
      buffer: DeviceMemoryBuffer,
      meta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback)
    extends RapidsBufferBase(id, buffer.getLength, meta, initialSpillPriority, spillCallback)
      with MemoryBuffer.EventHandler
      with AutoCloseable {

    def getDeviceMemoryBufferInternal: DeviceMemoryBuffer  = buffer

    override def isSpillable: Boolean = buffer.getRefCount == 1

    def getRefCount = buffer.getRefCount

    val sb = new mutable.StringBuilder()
    Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
      sb.append("    " + stackTraceElement + "\n")
    }
    val myStackTrace = sb.toString
    logInfo(s"Adding RegisteredDeviceMemoryBuffer ${buffer} to cached as ${id} with " +
      s"initial refCount ${buffer.getRefCount}")
    //dmbs.put(id, this)
    require(null == buffer.setEventHandler(this), "Overwrote an event handler!!")

    // TODO: fix removed registered.synchronized from here
    override def onClosed(refCount: Int): Unit = {
      logInfo(s"RegisteredDeviceMemoryBuffer ${id} ${buffer} closed with refCount ${refCount}")
      if (refCount > 0 && aliases.size == 0) {
        val sb = new mutable.StringBuilder()
        Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
          sb.append("    " + stackTraceElement + "\n")
        }
        logInfo(s"refCount is $refCount but no aliases for ${id} ${sb.toString()}")
        //buffer.cleaner.logRefCountDebug("foo")
      }
      if (aliases.size() > 0 && refCount == 1) {
        makeSpillable(this)
      }

      if (refCount == 0) {
        val sb = new mutable.StringBuilder()
        Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
          sb.append("    " + stackTraceElement + "\n")
        }
        println(s"I just got closed ${id} ${sb.toString()}")
        //dmbs.remove(id)
        val idsLeft = new mutable.ArrayBuffer[(RapidsBufferId, RegisteredDeviceMemoryBuffer)]()
        //dmbs.forEach((k, v) => {
        //  idsLeft.append((k, v))
        //})
        logInfo(s"Removed RegisteredDeviceMemoryBuffer ${id} " +
          s"${buffer} " + //from cached: ${dmbs.size()} " +
          s"$idsLeft")
        //removeBuffer(id)
        //require(this == buffer.setEventHandler(null), "Stumped on an event handler that wasn't mine!!")
      }
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      removeSpillable(this)
      buffer.incRefCount()
      logInfo(s"At getDeviceMemoryBuffer ${this}, refCount=${buffer.getRefCount}")
      buffer
    }

    override def close(): Unit = {
      super.close()
      logInfo(s"At close for ${id} with buffer ref count ${buffer.getRefCount}")
    }

    val aliases = new ConcurrentHashMap[RapidsBufferId, String]()

    var closed = false

    def alias(aliasingId: RapidsBufferId,
              how: String): RegisteredDeviceMemoryBuffer = {
      if (closed) {
        logWarning(s"New alias $aliasingId for previously dangled ${id}.")
        // TODO: just added this.. it kind of makes sense
        buffer.incRefCount()
        closed = false
        //throw new IllegalStateException(s"$aliasingId cannot alias $id since it is already closed!")
      }
      if (aliases.contains(aliasingId)) {
        throw new IllegalStateException(s"Alias already exists for $id to $aliasingId")
      }
      aliasingId.setAlias(id)
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

    def removeAlias(aliasingId: RapidsBufferId) = {
      if (closed) {
        throw new IllegalStateException(s"$aliasingId cannot remove alias to $id " +
          s"since it is already closed!")
      }
      val how: String = aliases.remove(aliasingId)
      logInfo(s"$id no longer aliased by ${aliasingId} ($how). Number of aliases ${aliases.size()}." +
        s"buffer refCount=${getRefCount}")
      //close()
      //println(s"Closed, as standard for all removeAlias. ${getRefCount} ")
      if (aliases.size() == 0) {
        buffer.synchronized {
          logInfo(s"$id has no aliases left") //, closing it!")
          closed = true
          buffer.close() // TODO: need this?
          buffer.setEventHandler(null)
          removeBuffer(this.id)
          // TODO: close() // final close
        }
      }
    }

    /** Release the underlying resources for this buffer. */
    override protected def releaseResources(): Unit = {
      logInfo(s"releaseResources in ${id}.. " +
        s"refCount= ${buffer.getRefCount}, aliases=${aliases.size()}")
    }

    /** The storage tier for this buffer */
    override val storageTier: StorageTier = StorageTier.DEVICE

    /**
     * Get the underlying memory buffer. This may be either a HostMemoryBuffer or a DeviceMemoryBuffer
     * depending on where the buffer currently resides.
     * The caller must have successfully acquired the buffer beforehand.
     *
     * @see [[addReference]]
     * @note It is the responsibility of the caller to close the buffer.
     */
    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer
}

  def registeredBuffer(
      aliasingId: RapidsBufferId,
      buffer: DeviceMemoryBuffer,
      meta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback,
      how: String): (Boolean, RegisteredDeviceMemoryBuffer) = buffer.synchronized {
    //buffer.incRefCount()
    //withResource(buffer) { _ =>
      val handler = buffer.getEventHandler
      var added = false
      val registered = handler match {
        case null =>
          // TODO: do I need this? buffer.incRefCount()
          buffer.incRefCount()
          added = true
          val reg = new RegisteredDeviceMemoryBuffer(
            TempSpillBufferId(),
            buffer,
            meta,
            initialSpillPriority,
            spillCallback)
          println(s"it is new, created registered, refCount is ${buffer.getRefCount}")
          addBuffer(reg)
          reg
        case hndr: RegisteredDeviceMemoryBuffer =>
          hndr
      }
      (added, registered.alias(aliasingId, how))
    //}
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
      var contigBuffer: RegisteredDeviceMemoryBuffer,
      spillPriority: Long,
      override val spillCallback: SpillCallback)
      extends RapidsBufferBase(id, size, meta, spillPriority, spillCallback) {
    override val storageTier: StorageTier = StorageTier.DEVICE

    override def isSpillable: Boolean = false

    var released = false
    override protected def releaseResources(): Unit = {
      println(s"releaseResources ${this.id}")
      if (released) {
        throw new IllegalStateException(s"Already released ${id} which aliases ${contigBuffer.id}")
      }
      released = true
      logInfo(s"releaseResources ${id} -- with registered buff ${contigBuffer.id} refCount ${contigBuffer.getRefCount}")
      println(s"closing table ${id} reg ${contigBuffer.id} refCount ${contigBuffer.getRefCount}")
      logInfo(s"closing table ${id} reg ${contigBuffer.id} refCount ${contigBuffer.getRefCount}")
      table.foreach(_.close())
      println(s"removing alias ${id} reg ${contigBuffer.id} refCount ${contigBuffer.getRefCount}")
      logInfo(s"removing alias ${id} reg ${contigBuffer.id} refCount ${contigBuffer.getRefCount}")
      contigBuffer.removeAlias(id)
      contigBuffer = null
    }

    override def getDeviceMemoryBuffer: DeviceMemoryBuffer = {
      contigBuffer.getDeviceMemoryBuffer
    }

    override def getMemoryBuffer: MemoryBuffer = getDeviceMemoryBuffer

    override def getColumnarBatch(
      sparkTypes: Array[DataType]): ColumnarBatch = contigBuffer.synchronized {
      // TODO: remember to bring back acquisition here... really we may need
      //   to always incRefCount here. Because, `columnarBatchFromDeviceBuffer`
      //   is going to incRefCount. So it is by definition not spillable..
      val buff = contigBuffer.getDeviceMemoryBufferInternal
      val startedWith = buff.getRefCount
      var endedWith: Int = -1

      val sb = new mutable.StringBuilder()
      Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
        sb.append("    " + stackTraceElement + "\n")
      }
      val myStackTrace = sb.toString

      logInfo(s"At getColumnarBatch ${id} and registered ${contigBuffer.id} " +
        s"with ref count ${buff.getRefCount}. stack trace: ${myStackTrace}")
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
        val r = columnarBatchFromDeviceBuffer(buff, sparkTypes)
        logInfo(s"went columnarBatchFromDeviceBuffer route ${id} reg ${contigBuffer.id} " +
          s"refCount is now ${buff.getRefCount}")
        r
      }

      endedWith = buff.getRefCount
      //require(endedWith > startedWith,
      //  s"endedWith=$endedWith, startedWith=$startedWith")
      res
    }
  }
}
