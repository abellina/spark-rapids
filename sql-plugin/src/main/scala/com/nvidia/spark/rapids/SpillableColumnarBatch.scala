/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import ai.rapids.cudf.MemoryBuffer.EventHandler
import ai.rapids.cudf.{ContiguousTable, DeviceMemoryBuffer}
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.TempSpillBufferId
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

import java.util.concurrent.ConcurrentHashMap

/**
 * Holds a ColumnarBatch that the backing buffers on it can be spilled.
 */
trait SpillableColumnarBatch extends AutoCloseable with RapidsBufferAlias {
  /**
   * The number of rows stored in this batch.
   */
  def numRows(): Int

  /**
   * Set a new spill priority.
   */
  def setSpillPriority(priority: Long): Unit

  /**
   * Get the columnar batch.
   * @note It is the responsibility of the caller to close the batch.
   * @note If the buffer is compressed data then the resulting batch will be built using
   *       `GpuCompressedColumnVector`, and it is the responsibility of the caller to deal
   *       with decompressing the data if necessary.
   */
  def withColumnarBatch[T](fn: ColumnarBatch => T): T

  def releaseBatch(): ColumnarBatch

  def sizeInBytes: Long
}

/**
 * Cudf does not support a table with columns and no rows. This takes care of making one of those
 * spillable, even though in reality there is no backing buffer.  It does this by just keeping the
 * row count in memory, and not dealing with the catalog at all.
 */
class JustRowsColumnarBatch(numRows: Int, semWait: GpuMetric)
    extends SpillableColumnarBatch with Arm {
  override def numRows(): Int = numRows
  override def setSpillPriority(priority: Long): Unit = () // NOOP nothing to spill

  private def makeJustRowsBatch(): ColumnarBatch = {
    GpuSemaphore.acquireIfNecessary(TaskContext.get(), semWait)
    new ColumnarBatch(Array.empty, numRows)
  }
  override def close(): Unit = () // NOOP nothing to close
  override val sizeInBytes: Long = 0L

  override def releaseBatch(): ColumnarBatch = {
    makeJustRowsBatch()
  }

  def withColumnarBatch[T](fn: ColumnarBatch => T): T = {
    withResource(makeJustRowsBatch()) { cb =>
      fn(cb)
    }
  }

  override def getSpillPriority: Long = -1
}

/**
 * The implementation of [[SpillableColumnarBatch]] that points to buffers that can be spilled.
 * @note the buffer should be in the cache by the time this is created and this is taking over
 *       ownership of the life cycle of the batch.  So don't call this constructor directly please
 *       use `SpillableColumnarBatch.apply` instead.
 */
class SpillableColumnarBatchImpl (
    id: RapidsBufferId,
    rowCount: Int,
    sparkTypes: Array[DataType],
    var spillPriority: Long,
    semWait: GpuMetric)
    extends SpillableColumnarBatch
      with RapidsBufferAlias
      with Arm
      with Logging {

  RapidsBufferAliasTracker.track(id, this)

  private var closed = false

  /**
   * The number of rows stored in this batch.
   */
  override def numRows(): Int = rowCount

  override lazy val sizeInBytes: Long = withRapidsBuffer(_.size)

  def withRapidsBuffer[T](fn: RapidsBuffer => T): T = {
    withResource(RapidsBufferCatalog.acquireBuffer(id)) { rapidsBuffer =>
      fn(rapidsBuffer)
    }
  }

  /**
   * Set a new spill priority.
   */
  override def setSpillPriority(priority: Long): Unit = {
    withRapidsBuffer { _ =>
      spillPriority = priority
      RapidsBufferAliasTracker.priorityUpdated(id, this)
    }
  }

  override def withColumnarBatch[T](fn: ColumnarBatch => T): T = {
    withRapidsBuffer { rapidsBuffer =>
      GpuSemaphore.acquireIfNecessary(TaskContext.get(), semWait)
      rapidsBuffer.withColumnarBatch(sparkTypes) { cb =>
        fn(cb)
      }
    }
  }

  override def releaseBatch(): ColumnarBatch = {
    logWarning(s"At releaseBatch for ${id}")
    val batch = withRapidsBuffer { rapidsBuffer =>
      GpuSemaphore.acquireIfNecessary(TaskContext.get(), semWait)
      rapidsBuffer.releaseBatch(sparkTypes)
    }
    RapidsBufferAliasTracker.stopTracking(id, this)
    closed = true
    batch
  }

  /**
   * Remove the `ColumnarBatch` from the cache.
   */
  override def close(): Unit = {
    logWarning(s"At close for ${id}")
    if (!closed) {
      // closing my reference
      RapidsBufferAliasTracker.stopTracking(id, this)
      closed = true
    }
  }

  override def getSpillPriority: Long = spillPriority
}

trait RapidsBufferAlias {
  def getSpillPriority: Long
}

object RapidsBufferAliasTracker extends Arm with Logging {

  private val bufferSpillableCount =
    new ConcurrentHashMap[RapidsBufferId, HashedPriorityQueue[RapidsBufferAlias]]()

  private def comparator(alias1: RapidsBufferAlias, alias2: RapidsBufferAlias): Int = {
    if (alias1.getSpillPriority == alias2.getSpillPriority) {
      0
    } else if (alias1.getSpillPriority > alias2.getSpillPriority) {
      1
    } else {
      -1
    }
  }

  def track(rapidsBufferId: RapidsBufferId, alias: RapidsBufferAlias): Unit = {
    bufferSpillableCount.compute(rapidsBufferId, (_, a) => {
      var aliases = a
      if (aliases == null) {
        aliases = new HashedPriorityQueue[RapidsBufferAlias](comparator)
      }
      aliases.offer(alias)
      aliases
    })
  }

  def stopTracking(rapidsBufferId: RapidsBufferId, alias: RapidsBufferAlias): Boolean = {
    val newAliases = bufferSpillableCount.compute(rapidsBufferId, (_, aliases) => {
      if (aliases == null) {
        throw new IllegalStateException(
          s"$rapidsBufferId not found and we attempted to remove spillable!")
      }

      aliases.remove(alias)

      if (aliases.size() == 0) {
        null // remove since no more aliases exist
      } else {
        aliases
      }
    })

    if (newAliases == null) {
      // we can now remove the underlying RapidsBufferId
      logWarning(s"Removing buffer ${rapidsBufferId} as spillable count is now 0")
      RapidsBufferCatalog.removeBuffer(rapidsBufferId)
      true
    } else {
      logWarning(s"NOT removing buffer ${rapidsBufferId} as " +
        s"Unit spillable count is now ${newAliases.size()}")
      false
    }
  }

  def priorityUpdated(id: RapidsBufferId, alias: RapidsBufferAlias): Unit = {
    withResource(RapidsBufferCatalog.acquireBuffer(id)) { buffer =>
      val newAliases = bufferSpillableCount.compute(id, (_, aliases) => {
        if (aliases == null) {
          throw new IllegalStateException(
            s"Attempted to update priority for unknown RapidsBuffer ${id}")
        }
        aliases.priorityUpdated(alias)
        aliases
      })

      // update the priority of the underlying RapidsBuffer to be the
      // maximum priority for all aliases associated with it
      buffer.setSpillPriority(newAliases.peek().getSpillPriority)
    }
  }
}

object SpillableColumnarBatch extends Arm with Logging {
  /**
   * Create a new SpillableColumnarBatch.
   *
   * @note This takes over ownership of batch, and batch should not be used after this.
   * @param batch         the batch to make spillable
   * @param priority      the initial spill priority of this batch
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   */
  def apply(batch: ColumnarBatch,
            priority: Long,
            spillCallback: SpillCallback): SpillableColumnarBatch = {
    val numRows = batch.numRows()
    if (batch.numCols() <= 0) {
      // We consumed it
      batch.close()
      new JustRowsColumnarBatch(numRows, spillCallback.semaphoreWaitTime)
    } else {
      val types = GpuColumnVector.extractTypes(batch)
      val id = addBatch(batch, priority, spillCallback)
      new SpillableColumnarBatchImpl(
        id, numRows, types, priority, spillCallback.semaphoreWaitTime)
    }
  }

  /**
   * Create a new SpillableColumnarBatch
   * @note The caller is responsible for closing the contiguous table parameter.
   * @param ct contiguous table containing the batch GPU data
   * @param sparkTypes array of Spark types describing the data schema
   * @param priority the initial spill priority of this batch
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   */
  def apply(
      ct: ContiguousTable,
      sparkTypes: Array[DataType],
      priority: Long,
      spillCallback: SpillCallback): SpillableColumnarBatch = {
    val buffer = ct.getBuffer
    buffer.synchronized {
      val id = buffer.getEventHandler match {
        case null =>
          val id = TempSpillBufferId()
          RapidsBufferCatalog.addContiguousTable(id, ct, priority, spillCallback)
          id
        case rapidsBuff: RapidsBuffer =>
          // existing case
          rapidsBuff.id
      }
      withResource(RapidsBufferCatalog.acquireBuffer(id)) { _ =>
        new SpillableColumnarBatchImpl(
          id,
          ct.getRowCount.toInt,
          sparkTypes,
          priority,
          spillCallback.semaphoreWaitTime)
      }
    }
  }

  private[this] def addBatch(
      batch: ColumnarBatch,
      initialSpillPriority: Long,
      spillCallback: SpillCallback): RapidsBufferId = {
    withResource(batch) { batch =>
      val numColumns = batch.numCols()
      if (GpuCompressedColumnVector.isBatchCompressed(batch)) {
        val cv = batch.column(0).asInstanceOf[GpuCompressedColumnVector]
        val buff = cv.getTableBuffer
        buff.getEventHandler match {
          case null =>
            // incRefCount as we now own it here
            buff.incRefCount()
            val id = TempSpillBufferId()
            RapidsBufferCatalog.addBuffer(id, buff, cv.getTableMeta, initialSpillPriority,
              spillCallback)
            id
          case rapidsBuff: RapidsBuffer =>
            // existing case
            rapidsBuff.id
        }
      } else if (GpuPackedTableColumn.isBatchPacked(batch)) {
        val cv = batch.column(0).asInstanceOf[GpuPackedTableColumn]
        val buffer = cv.getContiguousTable.getBuffer
        buffer.getEventHandler match {
          case null =>
            val id = TempSpillBufferId()
            RapidsBufferCatalog.addContiguousTable(
              id, cv.getContiguousTable, initialSpillPriority, spillCallback)
            id
          case rapidsBuff: RapidsBuffer =>
            // existing case
            rapidsBuff.id
        }
      } else if (numColumns > 0 &&
          (0 until numColumns)
              .forall(i => batch.column(i).isInstanceOf[GpuColumnVectorFromBuffer])) {
        val cv = batch.column(0).asInstanceOf[GpuColumnVectorFromBuffer]
        val buff = cv.getBuffer
        buff.getEventHandler match {
          case null =>
            val id = TempSpillBufferId()
            val table = GpuColumnVector.from(batch)
            buff.incRefCount()
            RapidsBufferCatalog.addTable(id, table, buff, cv.getTableMeta, initialSpillPriority,
              spillCallback)
            id
          case rapidsBuff: RapidsBuffer =>
            // existing case
            rapidsBuff.id
        }
      } else {
        withResource(GpuColumnVector.from(batch)) { tmpTable =>
          withResource(tmpTable.contiguousSplit()) { contigTables =>
            require(contigTables.length == 1, "Unexpected number of contiguous spit tables")
            val id = TempSpillBufferId()
            RapidsBufferCatalog.addContiguousTable(id, contigTables.head, initialSpillPriority,
              spillCallback)
            id
          }
        }
      }
    }
  }

}


/**
 * Just like a SpillableColumnarBatch but for buffers.
 */
class SpillableBuffer (
    id: TempSpillBufferId,
    var spillPriority: Long,
    semWait: GpuMetric)
  extends RapidsBufferAlias
    with AutoCloseable
    with Arm {

  RapidsBufferAliasTracker.track(id, this)

  private var closed = false

  lazy val sizeInBytes: Long =
    withResource(RapidsBufferCatalog.acquireBuffer(id)) { buff =>
      buff.size
    }

  /**
   * Set a new spill priority.
   */
  def setSpillPriority(priority: Long): Unit = {
    spillPriority = priority
    RapidsBufferAliasTracker.priorityUpdated(id, this)
  }

  /**
   * Use the device buffer.
   */
  def withRapidsBuffer[T](fn: RapidsBuffer => T): T = {
    withResource(RapidsBufferCatalog.acquireBuffer(id)) { rapidsBuffer =>
      fn(rapidsBuffer)
    }
  }

  /**
   * Remove the buffer from the cache.
   */
  override def close(): Unit = {
    if (!closed) {
      RapidsBufferAliasTracker.stopTracking(id, this)
      closed = true
    }
  }

  override def getSpillPriority: Long = spillPriority
}

object SpillableBuffer extends Arm {

  /**
   * Create a new SpillableBuffer.
   * @note This takes over ownership of buffer, and buffer should not be used after this.
   * @param buffer the buffer to make spillable
   * @param priority the initial spill priority of this buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   */
  def apply(buffer: DeviceMemoryBuffer,
      priority: Long,
      spillCallback: SpillCallback): SpillableBuffer = {
    val id = TempSpillBufferId()
    val meta = MetaUtils.getTableMetaNoTable(buffer)
    RapidsBufferCatalog.addBuffer(id, buffer, meta, priority, spillCallback)
    new SpillableBuffer(id, priority, spillCallback.semaphoreWaitTime)
  }
}
