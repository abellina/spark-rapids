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

package com.nvidia.spark.rapids.spill

import java.io.File

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.{Arm, GpuMetric, NoopMetric}
import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.spill.StorageTier.StorageTier

import org.apache.spark.sql.rapids.RapidsDiskBlockManager

/**
 * An identifier for a RAPIDS buffer that can be automatically spilled between buffer stores.
 * NOTE: Derived classes MUST implement proper hashCode and equals methods, as these objects are
 *       used as keys in hash maps. Scala case classes are recommended.
 */
private[spill] trait RapidsBufferId {
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
  val GDS: StorageTier = Value(3, "GPUDirect Storage")
}

abstract class SpillMetricsCallback extends Serializable {

  /**
   * Callback type for when a batch is spilled from one storage tier to another. This is
   * intended to only be used for metrics gathering in parts of the GPU plan that can spill.
   * No GPU memory should ever be allocated from this callback, blocking in this function
   * is strongly discouraged. It should be as light weight as possible. It takes three arguments
   *
   * @param from the storage tier the data is being spilled from.
   * @param to the storage tier the data is being spilled to.
   * @param amount the amount of data in bytes that is spilled.
   */
  def apply (from: StorageTier, to: StorageTier, amount: Long): Unit

  def semaphoreWaitTime: GpuMetric
}

object RapidsBuffer {

  /**
   * A default NOOP callback for when a buffer is spilled
   */
  val defaultSpillCallback: SpillMetricsCallback = new SpillMetricsCallback {
    override def apply(from: StorageTier, to: StorageTier, amount: Long): Unit = ()

    override def semaphoreWaitTime: GpuMetric = NoopMetric
  }
}

/** Interface provided by all types of RAPIDS buffers */
trait RapidsBuffer extends AutoCloseable {
  /** The buffer identifier for this buffer. */
  val id: RapidsBufferId

  /** The size of this buffer in bytes. */
  val size: Long

  /** Descriptor for how the memory buffer is formatted */
  val meta: TableMeta

  /** The storage tier for this buffer */
  val storageTier: StorageTier

  /**
   * Materialize the memory buffer from the underlying storage.
   *
   * If the buffer resides in device or host memory, only reference count is incremented.
   * If the buffer resides in secondary storage, a new host or device memory buffer is created,
   * with the data copied to the new buffer.
   * The caller must have successfully acquired the buffer beforehand.
   *
   * @see [[addReference]]
   * @note It is the responsibility of the caller to close the buffer.
   * @note This is an internal API only used by Rapids buffer stores.
   */
  def getMemoryBuffer: MemoryBuffer

  /**
   * Get a DeviceMemoryBuffer. Only the RapidsDeviceMemoryStore implements this
   * and all other tiers throw.
   * @return the DeviceMemoryBuffer
   */
  def getDeviceMemoryBuffer: DeviceMemoryBuffer

  /**
   * Copy the content of this buffer into the specified memory buffer, starting from the given
   * offset.
   *
   * @param srcOffset offset to start copying from.
   * @param dst the memory buffer to copy into.
   * @param dstOffset offset to copy into.
   * @param length number of bytes to copy.
   * @param stream CUDA stream to use
   */
  def copyToMemoryBuffer(
      srcOffset: Long, dst: MemoryBuffer, dstOffset: Long, length: Long, stream: Cuda.Stream)

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

  /**
   * Get the spill priority value for this buffer. Lower values are higher
   * priority for spilling, meaning buffers with lower values will be
   * preferred for spilling over buffers with a higher value.
   */
  def getSpillPriority: Long

  /**
   * Gets the spill metrics callback currently associated with this buffer.
   * @return the current callback
   */
  def getSpillCallback: SpillMetricsCallback

  /**
   * Set the spill priority for this buffer. Lower values are higher priority
   * for spilling, meaning buffers with lower values will be preferred for
   * spilling over buffers with a higher value.
   * @note should only be called from the buffer catalog
   * @param priority new priority value for this buffer
   */
  def setSpillPriority(priority: Long): Unit

  /**
   * Update the metrics callback that will be invoked next time a spill occurs.
   * @note should only be called from the buffer catalog
   * @param spillCallback the new callback
   */
  def setSpillCallback(spillCallback: SpillMetricsCallback): Unit
}

/**
 * A buffer with no corresponding device data (zero rows or columns).
 * These buffers are not tracked in buffer stores since they have no
 * device memory. They are only tracked in the catalog and provide
 * a representative `ColumnarBatch` but cannot provide a
 * `MemoryBuffer`.
 * @param id buffer ID to associate with the buffer
 * @param meta schema metadata
 */
sealed class DegenerateRapidsBuffer(
    override val id: RapidsBufferId,
    override val meta: TableMeta) extends RapidsBuffer with Arm {
  override val size: Long = 0L
  override val storageTier: StorageTier = StorageTier.DEVICE

  override def free(): Unit = {}

  override def getMemoryBuffer: MemoryBuffer =
    throw new UnsupportedOperationException("degenerate buffer has no memory buffer")

  override def copyToMemoryBuffer(srcOffset: Long, dst: MemoryBuffer, dstOffset: Long, length: Long,
      stream: Cuda.Stream): Unit =
    throw new UnsupportedOperationException("degenerate buffer cannot copy to memory buffer")

  override def getDeviceMemoryBuffer: DeviceMemoryBuffer =
    throw new UnsupportedOperationException("degenerate buffer has no device memory buffer")

  override def addReference(): Boolean = true

  override def getSpillPriority: Long = Long.MaxValue

  override val getSpillCallback: SpillMetricsCallback = RapidsBuffer.defaultSpillCallback

  override def setSpillPriority(priority: Long): Unit = {}

  override def setSpillCallback(callback: SpillMetricsCallback): Unit = {}

  override def close(): Unit = {}
}
