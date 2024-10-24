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

import java.nio.channels.WritableByteChannel
import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable

import ai.rapids.cudf.{ColumnVector, Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, Table}
import com.nvidia.spark.rapids.Arm._
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.sql.rapids.GpuTaskMetrics
import org.apache.spark.sql.rapids.storage.RapidsStorageUtils
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Buffer storage using device memory.
 */
class RapidsDeviceMemoryStore
  extends RapidsBufferStore(StorageTier.DEVICE) {

  // The RapidsDeviceMemoryStore handles spillability via ref counting
  override protected def spillableOnAdd: Boolean = false

  override def createBuffer(
      rapidsBuffer: RapidsBuffer,
      catalog: RapidsBufferCatalog,
      stream: Cuda.Stream,
      base: RapidsMemoryBuffer): RapidsBuffer = {
    withResource(rapidsBuffer.getCopyIterator(stream)) { copyIter =>
      catalog.withChunkedPackBounceBuffer { bounceBuffer =>
        val meta = getMeta(rapidsBuffer, copyIter)
        withResource(copyIter.next(bounceBuffer)) { memoryBuffer =>
          val deviceBuffer = {
            memoryBuffer match {
              case d: DeviceMemoryBuffer => d
              case h: HostMemoryBuffer =>
                GpuTaskMetrics.get.readSpillFromHostTime {
                  closeOnExcept(DeviceMemoryBuffer.allocate(memoryBuffer.getLength, stream)) {
                    deviceBuffer =>
                      logDebug(s"copying from host $h to device $deviceBuffer")
                      deviceBuffer.copyFromHostBuffer(h, stream)
                      deviceBuffer
                  }
                }
              case b => throw new IllegalStateException(s"Unrecognized buffer: $b")
            }
          }

          val devBuffer = meta.map { tableMeta =>
            new RapidsDeviceMemoryBufferWithMeta(
              rapidsBuffer.id,
              deviceBuffer.getLength,
              tableMeta,
              deviceBuffer,
              rapidsBuffer.getSpillPriority,
              catalog,
              base)
          }.getOrElse {
            new RapidsDeviceMemoryBuffer(
              rapidsBuffer.id,
              deviceBuffer.getLength,
              deviceBuffer,
              rapidsBuffer.getSpillPriority,
              catalog,
              base)
          }
          addBuffer(devBuffer)
          devBuffer
        }
      }
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
      initialSpillPriority: Long,
      needsSync: Boolean,
      catalog: RapidsBufferCatalog,
      tableMeta: Option[TableMeta] = None): RapidsMemoryBuffer = {
    buffer.incRefCount()
    val rmb = new RapidsMemoryBuffer(id)
    val rapidsBuffer = tableMeta match {
      case Some(meta) =>
        new RapidsDeviceMemoryBufferWithMeta(
          id,
          buffer.getLength,
          meta,
          buffer,
          initialSpillPriority,
          catalog,
          rmb)
      case None =>
        new RapidsDeviceMemoryBuffer(
          id,
          buffer.getLength,
          buffer,
          initialSpillPriority,
          catalog,
          rmb)
    }

    freeOnExcept(rapidsBuffer) { _ =>
      addBuffer(rapidsBuffer, needsSync)
      rmb.initialize(rapidsBuffer, StorageTier.DEVICE)
      rmb
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
      id: RapidsBufferId,
      table: Table,
      initialSpillPriority: Long,
      needsSync: Boolean,
      catalog: RapidsBufferCatalog): RapidsMemoryBuffer = {
    val rmb = new RapidsMemoryBuffer(id)
    val rapidsTable = new RapidsTable(
      id,
      table,
      initialSpillPriority,
      catalog,
      rmb)
    freeOnExcept(rapidsTable) { _ =>
      addBuffer(rapidsTable, needsSync)
      rmb.initialize(rapidsTable, StorageTier.DEVICE)
      rmb
    }
  }

  /**
   * A `RapidsTable` is the spill store holder of a cuDF `Table`.
   *
   * The table is not contiguous in GPU memory. Instead, this `RapidsBuffer` instance
   * allows us to use the cuDF chunked_pack API to make the table contiguous as the spill
   * is happening.
   *
   * This class owns the cuDF table and will close it when `close` is called.
   *
   * @param id the `RapidsBufferId` this table is associated with
   * @param table the cuDF table that we are managing
   * @param spillPriority a starting spill priority
   */
  class RapidsTable(id: RapidsBufferId,
                    table: Table,
                    spillPriority: Long,
                    catalog: RapidsBufferCatalog,
                    override val base: RapidsMemoryBuffer)
    extends RapidsBufferBase(
      id,
      spillPriority)
      with RapidsBufferChannelWritable {

    override def getCopyIterator(stream: Cuda.Stream): RapidsBufferCopyIterator =
      new RapidsBufferCopyIterator(
        chunkedPacker = Some(new ChunkedPacker(id, table, catalog.chunkedPackBounceBufferSize)))

    override def copyTo(otherStore: RapidsBufferStore, stream: Cuda.Stream): RapidsBuffer = {
      otherStore.createBuffer(this, catalog, stream, base)
    }

    override def writeToChannel(
        outputChannel: WritableByteChannel,
        stream: Cuda.Stream): (Long, Option[TableMeta]) = {
      // TODO: it is really strange that this is called from copyTo by way of the peer store
      // we should just be calling store methods in copyTo
      var written: Long = 0L
      catalog.withChunkedPackBounceBuffer { chunkedPackBounceBuffer =>
        withResource(getCopyIterator(stream)){ copyIter =>
          while(copyIter.hasNext) {
            withResource(copyIter.next(chunkedPackBounceBuffer)) { slice =>
              catalog.withHostSpillBounceBuffer { hostSpillBounceBuffer =>
                val iter =
                  new MemoryBufferToHostByteBufferIterator(
                    slice,
                    hostSpillBounceBuffer,
                    stream)
                iter.foreach { bb =>
                  try {
                    while (bb.hasRemaining) {
                      written += outputChannel.write(bb)
                    }
                  } finally {
                    RapidsStorageUtils.dispose(bb)
                  }
                }
              }
            }
          }
          (written, Some(copyIter.getMeta))
        }
      }
    }

    // By default all columns are NOT spillable since we are not the only owners of
    // the columns (the caller is holding onto a ColumnarBatch that will be closed
    // after instantiation, triggering onClosed callbacks)
    // This hash set contains the columns that are currently spillable.
    private val columnSpillability = new ConcurrentHashMap[ColumnVector, Boolean]()

    private val numDistinctColumns =
      (0 until table.getNumberOfColumns).map(table.getColumn).distinct.size

    // we register our event callbacks as the very first action to deal with
    // spillability
    registerOnCloseEventHandler()

    private var isClosed = false

    /** Release the underlying resources for this buffer. */
    override protected def releaseResources(): Unit = synchronized {
      if (isClosed) {
        println(s"double close for table ${id}")
      } else {
        table.close()
        isClosed = true
      }
    }

    // This is the current size in batch form. It is to be used while this
    // table hasn't migrated to another store.
    override val memoryUsedBytes: Long = GpuColumnVector.getTotalDeviceMemoryUsed(table)

    /**
     * Mark a column as spillable
     *
     * @param column the ColumnVector to mark as spillable
     */
    def onColumnSpillable(column: ColumnVector): Unit = {
      columnSpillability.put(column, true)
      updateSpillability()
    }

    /**
     * Update the spillability state of this RapidsTable. This is invoked from
     * two places:
     *
     * - from the onColumnSpillable callback, which is invoked from a
     * ColumnVector.EventHandler.onClosed callback.
     *
     * - after adding a table to the store to mark the table as spillable if
     * all columns are spillable.
     */
    override def updateSpillability(): Unit = {
      setSpillable(this, columnSpillability.size == numDistinctColumns)
    }

    /**
     * Produce a `ColumnarBatch` from our table, and in the process make ourselves
     * not spillable.
     *
     * @param sparkTypes the spark data types the batch should have
     */
    override def getColumnarBatch(
        sparkTypes: Array[DataType], stream: Cuda.Stream): ColumnarBatch = {
      columnSpillability.clear()
      setSpillable(this, false)
      GpuColumnVector.from(table, sparkTypes)
    }

    override def free(): Unit = {
      // lets remove our handler from the chain of handlers for each column
      removeOnCloseEventHandler()
      super.free()
    }

    private def registerOnCloseEventHandler(): Unit = {
      val columns = (0 until table.getNumberOfColumns).map(table.getColumn)
      // cudfColumns could contain duplicates. We need to take this into account when we are
      // deciding the floor refCount for a duplicated column
      val repetitionPerColumn = new mutable.HashMap[ColumnVector, Int]()
      columns.foreach { col =>
        val repetitionCount = repetitionPerColumn.getOrElse(col, 0)
        repetitionPerColumn(col) = repetitionCount + 1
      }
      repetitionPerColumn.foreach { case (distinctCv, repetition) =>
        // lock the column because we are setting its event handler, and we are inspecting
        // its refCount.
        distinctCv.synchronized {
          val eventHandler = distinctCv.getEventHandler match {
            case null =>
              val eventHandler = new RapidsDeviceColumnEventHandler
              distinctCv.setEventHandler(eventHandler)
              eventHandler
            case existing: RapidsDeviceColumnEventHandler =>
              existing
            case other =>
              throw new IllegalStateException(
                s"Invalid column event handler $other")
          }
          eventHandler.register(this, repetition)
          if (repetition == distinctCv.getRefCount) {
            onColumnSpillable(distinctCv)
          }
        }
      }
    }

    // this method is called from free()
    private def removeOnCloseEventHandler(): Unit = {
      val distinctColumns =
        (0 until table.getNumberOfColumns).map(table.getColumn).distinct
      distinctColumns.foreach { distinctCv =>
        distinctCv.synchronized {
          distinctCv.getEventHandler match {
            case eventHandler: RapidsDeviceColumnEventHandler =>
              eventHandler.deregister(this)
            case t =>
              throw new IllegalStateException(
                s"Invalid column event handler $t")
          }
        }
      }
    }
  }

  /**
   * A per cuDF column event handler that handles calls to .close()
   * inside of the `ColumnVector` lock.
   */
  class RapidsDeviceColumnEventHandler
    extends ColumnVector.EventHandler {

    // Every RapidsTable that references this column has an entry in this map.
    // The value represents the number of times (normally 1) that a ColumnVector
    // appears in the RapidsTable. This is also the ColumnVector refCount at which
    // the column is considered spillable.
    // The map is protected via the ColumnVector lock.
    private val registration = new mutable.HashMap[RapidsTable, Int]()

    /**
     * Every RapidsTable iterates through its columns and either creates
     * a `ColumnTracking` object and associates it with the column's
     * `eventHandler` or calls into the existing one, and registers itself.
     *
     * The registration has two goals: it accounts for repetition of a column
     * in a `RapidsTable`. If a table has the same column repeated it must adjust
     * the refCount at which this column is considered spillable.
     *
     * The second goal is to account for aliasing. If two tables alias this column
     * we are going to mark it as non spillable.
     *
     * @param rapidsTable - the table that is registering itself with this tracker
     */
    def register(rapidsTable: RapidsTable, repetition: Int): Unit = {
      registration.put(rapidsTable, repetition)
    }

    /**
     * This is invoked during `RapidsTable.free` in order to remove the entry
     * in `registration`.
     * @param rapidsTable - the table that is de-registering itself
     */
    def deregister(rapidsTable: RapidsTable): Unit = {
      registration.remove(rapidsTable)
    }

    // called with the cudfCv lock held from cuDF's side
    override def onClosed(cudfCv: ColumnVector, refCount: Int): Unit = {
      // we only handle spillability if there is a single table registered
      // (no aliasing)
      if (registration.size == 1) {
        val (rapidsTable, spillableRefCount) = registration.head
        if (spillableRefCount == refCount) {
          rapidsTable.onColumnSpillable(cudfCv)
        }
      }
    }
  }

  class RapidsDeviceMemoryBuffer(id: RapidsBufferId,
                                 size: Long,
                                 contigBuffer: DeviceMemoryBuffer,
                                 spillPriority: Long,
                                 catalog: RapidsBufferCatalog,
                                 override val base: RapidsMemoryBuffer)
    extends RapidsBufferBase(id, spillPriority)
      with MemoryBuffer.EventHandler
      with RapidsBufferChannelWritable {

    override val memoryUsedBytes: Long = size

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
    override def getDeviceMemoryBuffer(stream: Cuda.Stream): DeviceMemoryBuffer = synchronized {
      contigBuffer.synchronized {
        setSpillable(this, false)
        contigBuffer.incRefCount()
        contigBuffer
      }
    }

    override def getMemoryBuffer(stream: Cuda.Stream): MemoryBuffer =
      getDeviceMemoryBuffer(stream)

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

    override def writeToChannel(
        outputChannel: WritableByteChannel, stream: Cuda.Stream): (Long, Option[TableMeta])= {
      var written: Long = 0L
      catalog.withHostSpillBounceBuffer { hostSpillBounceBuffer =>
        val iter = new MemoryBufferToHostByteBufferIterator(
          contigBuffer,
          hostSpillBounceBuffer,
          stream)
        iter.foreach { bb =>
          try {
            while (bb.hasRemaining) {
              written += outputChannel.write(bb)
            }
          } finally {
            RapidsStorageUtils.dispose(bb)
          }
        }
        (written, None)
      }
    }

    def getCopyIterator(stream: Cuda.Stream): RapidsBufferCopyIterator =
      new RapidsBufferCopyIterator(
        singleShotBuffer = Some(getMemoryBuffer(stream)))

    override def copyTo(store: RapidsBufferStore, stream: Cuda.Stream): RapidsBuffer = {
      store.createBuffer(this, catalog, stream, base)
    }

    override def close(): Unit = {
      super.close()
    }
  }

  class RapidsDeviceMemoryBufferWithMeta(id: RapidsBufferId,
                                         size: Long,
                                         _meta: TableMeta,
                                         contigBuffer: DeviceMemoryBuffer,
                                         spillPriority: Long,
                                         catalog: RapidsBufferCatalog,
                                         override val base: RapidsMemoryBuffer)
    extends RapidsDeviceMemoryBuffer(id, size, contigBuffer, spillPriority, catalog, base)
      with RapidsBufferWithMeta
      with MemoryBuffer.EventHandler
      with RapidsBufferChannelWritable
      with CopyableRapidsBuffer {

    override def meta: TableMeta = _meta
  }
}

