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

import java.io.File
import java.nio.channels.WritableByteChannel
import scala.collection.mutable.ArrayBuffer
import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.StorageTier.{DEVICE, StorageTier}
import com.nvidia.spark.rapids.format.TableMeta
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.RapidsDiskBlockManager
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * An identifier for a RAPIDS buffer that can be automatically spilled between buffer stores.
 * NOTE: Derived classes MUST implement proper hashCode and equals methods, as these objects are
 *       used as keys in hash maps. Scala case classes are recommended.
 */
trait RapidsBufferId {
  val tableId: Int

  /**
   * Indicates whether the buffer may share a spill file with other buffers.
   * If false then the spill file will be automatically removed when the buffer is freed.
   * If true then the spill file will not be automatically removed, and another subsystem needs
   * to be responsible for cleaning up the spill files for those types of buffers.
   */
  val canShareDiskPaths: Boolean = false

  /**
   * Generate a path to a local file that can be used to spill the corresponding buffer to disk.
   * The path must be unique across all buffers unless canShareDiskPaths is true.
   */
  def getDiskPath(diskBlockManager: RapidsDiskBlockManager): File
}

/** Enumeration of the storage tiers */
object StorageTier extends Enumeration {
  type StorageTier = Value
  val DEVICE: StorageTier = Value(0, "device memory")
  val HOST: StorageTier = Value(1, "host memory")
  val DISK: StorageTier = Value(2, "local disk")
}

/**
 * ChunkedPacker is an Iterator that uses a cudf::chunked_pack to copy a cuDF `Table`
 * to a target buffer in chunks.
 *
 * Each chunk is sized at most `bounceBuffer.getLength`, and the caller should cudaMemcpy
 * bytes from `bounceBuffer` to a target buffer after each call to `next()`.
 *
 * @note `ChunkedPacker` must be closed by the caller as it has GPU and host resources
 *       associated with it.
 *
 * @param id The RapidsBufferId for this pack operation to be included in the metadata
 * @param table cuDF Table to chunk_pack
 * @param bounceBuffer GPU memory to be used for packing. The buffer should be at least 1MB
 *                     in length.
 */
class ChunkedPacker(
    id: RapidsBufferId,
    table: Table,
    bounceBufferSize: Long)
    extends Logging with AutoCloseable {

  private var closed: Boolean = false

  // When creating cudf::chunked_pack use a pool if available, otherwise default to the
  // per-device memory resource
  private val chunkedPack = {
    val pool = GpuDeviceManager.chunkedPackMemoryResource
    val cudfChunkedPack = try {
      pool.flatMap { chunkedPool =>
        Some(table.makeChunkedPack(bounceBufferSize, chunkedPool))
      }
    } catch {
      case _: OutOfMemoryError =>
        if (!ChunkedPacker.warnedAboutPoolFallback) {
          ChunkedPacker.warnedAboutPoolFallback = true
          logWarning(
            s"OOM while creating chunked_pack using pool sized ${pool.map(_.getMaxSize)}B. " +
                "Falling back to the per-device memory resource.")
        }
        None
    }

    // if the pool is not configured, or we got an OOM, try again with the per-device pool
    cudfChunkedPack.getOrElse {
      table.makeChunkedPack(bounceBufferSize)
    }
  }

  private val tableMeta = withResource(chunkedPack.buildMetadata()) { packedMeta =>
    MetaUtils.buildTableMeta(
      id.tableId,
      chunkedPack.getTotalContiguousSize,
      packedMeta.getMetadataDirectBuffer,
      table.getRowCount)
  }

  def getTotalContiguousSize: Long = chunkedPack.getTotalContiguousSize

  def getMeta: TableMeta = {
    tableMeta
  }

  def hasNext: Boolean = synchronized {
    if (closed) {
      throw new IllegalStateException(s"ChunkedPacker for $id is closed")
    }
    chunkedPack.hasNext
  }

  def next(bounceBuffer: DeviceMemoryBuffer): MemoryBuffer = synchronized {
    require(bounceBuffer.getLength() == bounceBufferSize, 
      s"Bounce buffer ${bounceBuffer} doesn't match size ${bounceBufferSize} B.")

    if (closed) {
      throw new IllegalStateException(s"ChunkedPacker for $id is closed")
    }
    val bytesWritten = chunkedPack.next(bounceBuffer)
    // we increment the refcount because the caller has no idea where
    // this memory came from, so it should close it.
    bounceBuffer.slice(0, bytesWritten)
  }

  override def close(): Unit = synchronized {
    if (!closed) {
      closed = true
      chunkedPack.close()
    }
  }
}

object ChunkedPacker {
  private var warnedAboutPoolFallback: Boolean = false
}

/**
 * This iterator encapsulates a buffer's internal `MemoryBuffer` access
 * for spill reasons. Internally, there are two known implementations:
 * - either this is a "single shot" copy, where the entirety of the `RapidsBuffer` is
 *   already represented as a single contiguous blob of memory, then the expectation
 *   is that this iterator is exhausted with a single call to `next`
 * - or, we have a `RapidsBuffer` that isn't contiguous. This iteration will then
 *   drive a `ChunkedPacker` to pack the `RapidsBuffer`'s table as needed. The
 *   iterator will likely need several calls to `next` to be exhausted.
 *
 * @param buffer `RapidsBuffer` to copy out of its tier.
 */
class RapidsBufferCopyIterator(
  chunkedPacker: Option[ChunkedPacker] = None, 
  singleShotBuffer: Option[MemoryBuffer] = None)
    extends AutoCloseable with Logging {

  def isChunked: Boolean = chunkedPacker.isDefined

  // this is used for the single shot case to flag when `next` is call
  // to satisfy the Iterator interface
  private var singleShotCopyHasNext: Boolean = singleShotBuffer.isDefined

  def hasNext: Boolean =
    chunkedPacker.map(_.hasNext).getOrElse(singleShotCopyHasNext)

  def next(memoryBuffer: DeviceMemoryBuffer): MemoryBuffer = {
    require(hasNext,
      "next called on exhausted iterator")
    chunkedPacker.map(_.next(memoryBuffer)).getOrElse {
      singleShotCopyHasNext = false
      singleShotBuffer.get.slice(0, singleShotBuffer.get.getLength)
    }
  }

  def getTotalCopySize: Long = {
    chunkedPacker
        .map(_.getTotalContiguousSize)
        .getOrElse(singleShotBuffer.get.getLength)
  }

  def getMeta: TableMeta = {
    chunkedPacker
        .map(_.getMeta)
        .getOrElse {
          throw new IllegalStateException(
            "asked to get TableMeta but not chunk packed!")
        }
  }

  override def close(): Unit = {
    val toClose = new ArrayBuffer[AutoCloseable]()
    toClose.appendAll(chunkedPacker)
    toClose.appendAll(singleShotBuffer)
    toClose.safeClose()
  }
}

/** Interface provided by all types of RAPIDS buffers */
trait RapidsBuffer extends AutoCloseable {
  /** The buffer identifier for this buffer. */
  val id: RapidsBufferId

  val base: RapidsMemoryBuffer

  /**
   * The size of this buffer in bytes in its _current_ store. As the buffer goes through
   * contiguous split (either added as a contiguous table already, or spilled to host),
   * its size changes because contiguous_split adds its own alignment padding.
   *
   * @note Do not use this size to allocate a target buffer to copy, always use `getPackedSize.`
   */
  val memoryUsedBytes: Long

  def copyTo(store: RapidsBufferStore, stream: Cuda.Stream): RapidsBuffer

  def getCopyIterator(stream: Cuda.Stream): RapidsBufferCopyIterator

  /** The storage tier for this buffer */
  val storageTier: StorageTier
  
  /**
   * Try to add a reference to this buffer to acquire it.
   * @note The close method must be called for every successfully obtained reference.
   * @return true if the reference was added or false if this buffer is no longer valid
   */
  def addReference(): Boolean

  /**
   * Schedule the release of the buffer's underlying resources.
   * Subsequent attempts to acquire the buffer will fail. As soon as the
   * buffer has no outstanding references, the resources will be released.
   * <p>
   * This is separate from the close method which does not normally release
   * resources. close will only release resources if called as the last
   * outstanding reference and the buffer was previously marked as freed.
   */
  def free(): Unit

  def setSpillPriority(newPriority: Long): Unit

  def getSpillPriority: Long

  // helper methods
  def getColumnarBatch(sparkTypes: Array[DataType], stream: Cuda.Stream): ColumnarBatch =
    throw new UnsupportedOperationException(
      s"buffer ${this} does not support getColumnarBatch")
  def getHostColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch =
    throw new UnsupportedOperationException(
      s"buffer ${this} does not support getHostColumnarBatch")
  def getDeviceMemoryBuffer(stream: Cuda.Stream): DeviceMemoryBuffer =
    throw new UnsupportedOperationException(
      s"buffer ${this} does not support getDeviceMemoryBuffer")
  def getMemoryBuffer(stream: Cuda.Stream): MemoryBuffer =
    throw new UnsupportedOperationException(
      s"buffer ${this} does not support getMemoryBuffer")
  def getHostMemoryBuffer(stream: Cuda.Stream): HostMemoryBuffer =
    throw new UnsupportedOperationException(
      s"buffer ${this} does not support getHostMemoryBuffer")
}

/**
 * A buffer with no corresponding device data (zero rows or columns).
 * These buffers are not tracked in buffer stores since they have no
 * device memory. They are only tracked in the catalog and provide
 * a representative `ColumnarBatch` but cannot provide a
 * `MemoryBuffer`.
 * @param id buffer ID to associate with the buffer
 */
sealed class DegenerateRapidsBuffer(
    override val id: RapidsBufferId,
    val meta: TableMeta,
    val base: RapidsMemoryBuffer)
  extends RapidsBuffer {

  override def getCopyIterator(stream: Cuda.Stream): RapidsBufferCopyIterator = {
    throw new UnsupportedOperationException("degenerate buffer can't be copied")
  }

  override def copyTo(store: RapidsBufferStore, stream: Cuda.Stream): RapidsBuffer = {
    throw new UnsupportedOperationException("degenerate buffer can't be copied")
  }

  override def getColumnarBatch(sparkTypes: Array[DataType], stream: Cuda.Stream): ColumnarBatch = {
    val rowCount = meta.rowCount
    val packedMeta = meta.packedMetaAsByteBuffer()
    if (packedMeta != null) {
      withResource(DeviceMemoryBuffer.allocate(0)) { deviceBuffer =>
        withResource(Table.fromPackedTable(
          meta.packedMetaAsByteBuffer(), deviceBuffer)) { table =>
          GpuColumnVectorFromBuffer.from(table, deviceBuffer, meta, sparkTypes)
        }
      }
    } else {
      // no packed metadata, must be a table with zero columns
      new ColumnarBatch(Array.empty, rowCount.toInt)
    }
  }

  override val memoryUsedBytes: Long = 0L

  override val storageTier: StorageTier = StorageTier.DEVICE

  override def free(): Unit = {}

  override def addReference(): Boolean = true

  override def getSpillPriority: Long = Long.MaxValue

  override def setSpillPriority(newPriority: Long): Unit = {}

  override def close(): Unit = {}
}

trait RapidsHostBatchBuffer extends AutoCloseable {
  /**
   * Get the host-backed columnar batch from this buffer. The caller must have
   * successfully acquired the buffer beforehand.
   *
   * If this `RapidsBuffer` was added originally to the device tier, or if this is
   * a just a buffer (not a batch), this function will throw.
   *
   * @param sparkTypes the spark data types the batch should have
   * @see [[addReference]]
   * @note It is the responsibility of the caller to close the batch.
   */
  def getHostColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch

  val memoryUsedBytes: Long
}

class RapidsMemoryBuffer(val id: RapidsBufferId) {
  private val MAX_BUFFER_LOOKUP_ATTEMPTS = 100

  // a rapids memory buffer supports three "tiers"
  private var device: RapidsBuffer = _
  private var host: RapidsBuffer = _
  private var disk: RapidsBuffer = _

  def acquireBuffer(): RapidsBuffer = synchronized {
    def lookupAndReturn: Option[RapidsBuffer] = {
      val buffer = get(promote = false)
      if (buffer.addReference()) {
        Some(buffer)
      } else {
        None
      }
    }
    // fast path
    (0 until MAX_BUFFER_LOOKUP_ATTEMPTS).foreach { _ =>
      val mayBuffer = lookupAndReturn
      if (mayBuffer.isDefined) {
        return mayBuffer.get
      }
    }
    throw new IllegalStateException(s"Unable to acquire buffer for ID: $id")
  }

  def initialize(buffer: RapidsBuffer, tier: StorageTier): Unit = synchronized {
    tier match {
      case StorageTier.DEVICE =>
        device = buffer
      case StorageTier.HOST =>
        host = buffer
      case StorageTier.DISK =>
        disk = buffer
    }
  }

  /**
   * Spill a buffer from a store to a target store. In turn, this function calls
   * `copyTo` to copy the buffer, then returns the buffer at the tier we were
   * spilling from, nulling out that tier internally. It is the responsibility
   * of the caller to `free` the returned `RapidsBuffer`.
   * @param fromStore
   * @param targetStore
   * @param stream
   * @return
   */
  //TODO: AB: this function is not clear on what happens if the buffer already spilled, or if the buffer
  // could not spill to the target store
  def spill(
      fromStore: RapidsBufferStore, 
      targetStore: RapidsBufferStore, 
      stream: Cuda.Stream): Option[RapidsBuffer] = synchronized {
    val copyToTarget = copyTo(targetStore, stream)
    fromStore.tier match {
      case StorageTier.DEVICE => device = null
      case StorageTier.HOST => host = null
      case StorageTier.DISK => disk = null
    }
    copyToTarget
  }

  def copyTo(targetStore: RapidsBufferStore,
             stream: Cuda.Stream): Option[RapidsBuffer] = synchronized {
    val bufferAtTier = targetStore.tier match {
      case StorageTier.DEVICE => Option(device)
      case StorageTier.HOST => Option(host)
      case StorageTier.DISK => Option(disk)
    }
    if (bufferAtTier.isDefined) {
      None
    } else {
      def copyTo(store: RapidsBufferStore): RapidsBuffer = {
        if (device != null) {
          device.copyTo(store, stream)
        } else if (host != null) {
          host.copyTo(store, stream)
        } else {
          disk.copyTo(store, stream)
        }
      }

      // we are copying from a tier to another tier
      val newBuffer = targetStore.tier match {
        case StorageTier.DEVICE => 
          copyTo(targetStore)
        case StorageTier.HOST =>
          copyTo(targetStore)
        case StorageTier.DISK =>
          copyTo(targetStore)
      }

      newBuffer.storageTier match {
        case StorageTier.DEVICE =>
          device = newBuffer
        case StorageTier.HOST =>
          host = newBuffer
        case StorageTier.DISK =>
          disk = newBuffer
      }
      Some(newBuffer)
    }
  }

  def free(tier: StorageTier): Unit = synchronized {
    tier match {
      case StorageTier.DEVICE =>
        device.safeFree()
        device = null
      case StorageTier.HOST =>
        host.safeFree()
        host = null
      case StorageTier.DISK =>
        disk.safeFree()
        disk = null
    }
  }

  def free(): Unit = synchronized {
    Seq(device, host, disk).safeFree()
    device = null
    host = null
    disk = null
  }

  // promote to `device` if the caller intends to reuse this (hint)
  def get(promote: Boolean): RapidsBuffer = synchronized {
    val res = if (device != null) {
      device
    } else {
      if (promote) {
        copyTo(RapidsBufferCatalog.getDeviceStorage, Cuda.DEFAULT_STREAM).getOrElse(device)
      } else {
        if (host != null) {
          host
        } else {
          disk
        }
      }
    }
    if (res == null) {
      throw new IllegalStateException(s"Unable to acquire buffer for ID: ${id}")
    }
    res
  }

  // useful in tests
  def get(tier: StorageTier): Option[RapidsBuffer] = synchronized {
    val res = tier match {
      case StorageTier.DEVICE =>
        device
      case StorageTier.HOST =>
        host
      case StorageTier.DISK =>
        disk
    }
    Option(res)
  }

  def setSpillPriority(newPriority: Long): Unit = synchronized {
    val buffs = Seq(device, host, disk)
    buffs.foreach { b =>
      if (b != null) {
        b.setSpillPriority(newPriority)
      }
    }
  }
}

trait RapidsBufferChannelWritable {
  /**
   * At spill time, write this buffer to an nio WritableByteChannel.
   * @param writableChannel that this buffer can just write itself to, either byte-for-byte
   *                        or via serialization if needed.
   * @param stream the Cuda.Stream for the spilling thread. If the `RapidsBuffer` that
   *               implements this method is on the device, synchronization may be needed
   *               for staged copies.
   * @return the amount of bytes written to the channel
   */
  def writeToChannel(
    writableChannel: WritableByteChannel,
    stream: Cuda.Stream): (Long, Option[TableMeta])
}

object RapidsBuffer {
  def columnarBatchFromDeviceBuffer(
      devBuffer: DeviceMemoryBuffer,
      sparkTypes: Array[DataType],
      meta: TableMeta): ColumnarBatch = {
    val bufferMeta = meta.bufferMeta()
    if (bufferMeta == null || bufferMeta.codecBufferDescrsLength == 0) {
      MetaUtils.getBatchFromMeta(devBuffer, meta, sparkTypes)
    } else {
      GpuCompressedColumnVector.from(devBuffer, meta)
    }
  }
}