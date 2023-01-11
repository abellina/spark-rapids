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

import java.util.concurrent.ConcurrentHashMap
import java.util.function.BiFunction

import ai.rapids.cudf.{ContiguousTable, DeviceMemoryBuffer, Rmm, Table}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.{SparkConf, SparkEnv}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.RapidsDiskBlockManager

/**
 *  Exception thrown when inserting a buffer into the catalog with a duplicate buffer ID
 *  and storage tier combination.
 */
class DuplicateBufferException(s: String) extends RuntimeException(s) {}

trait RapidsBufferHandle {
  def getId: RapidsBufferId

  def getSpillPriority: Long

  def setSpillPriority(newPriority: Long): Unit
}


/**
 * Catalog for lookup of buffers by ID. The constructor is only visible for testing, generally
 * `RapidsBufferCatalog.singleton` should be used instead.
 */
class RapidsBufferCatalog extends Logging with Arm {
  def makeNewHandle(
    id: RapidsBufferId, spillPriority: Long): RapidsBufferHandle = {
    val handle = new RapidsBufferHandleImpl(id, spillPriority)
    track(handle)
    handle
  }

  /** Map of buffer IDs to buffers sorted by storage tier */
  private[this] val bufferMap = new ConcurrentHashMap[RapidsBufferId, Seq[RapidsBuffer]]

  private[this] val bufferHandles =
    new ConcurrentHashMap[RapidsBufferId, HashedPriorityQueue[RapidsBufferHandle]]()

  class RapidsBufferHandleImpl(id: RapidsBufferId, var priority: Long)
    extends RapidsBufferHandle with Arm {

    override def getId: RapidsBufferId = id

    override def setSpillPriority(newPriority: Long): Unit = {
      priority = newPriority
      priorityUpdated(this)
    }

    override def getSpillPriority: Long = priority
  }

  def priorityUpdated(handle: RapidsBufferHandle): Unit = {
    val id = handle.getId
    withResource(RapidsBufferCatalog.acquireBuffer(id)) { buffer =>
      val newHandlePriorities = bufferHandles.compute(id, (_, handleQ) => {
        if (handleQ == null) {
          throw new IllegalStateException(
            s"Attempted to update priority for RapidsBuffer ${id} without a handle")
        }
        handleQ.priorityUpdated(handle)
        handleQ
      })

      // update the priority of the underlying RapidsBuffer to be the
      // maximum priority for all aliases associated with it
      buffer.setSpillPriority(newHandlePriorities.peek().getSpillPriority)
    }
  }

  private def comparator(handle1: RapidsBufferHandle, handle2: RapidsBufferHandle): Int = {
    val prio1 = handle1.getSpillPriority
    val prio2 = handle2.getSpillPriority
    if (prio1 == prio2) {
      0
    } else if (prio1 > prio2) {
      1
    } else {
      -1
    }
  }

  def track(handle: RapidsBufferHandle): Unit = {
    bufferHandles.compute(handle.getId, (_, h) => {
      var handles = h
      if (handles == null) {
        handles = new HashedPriorityQueue[RapidsBufferHandle](comparator)
      }
      handles.offer(handle)
      handles
    })
  }

  def stopTracking(handle: RapidsBufferHandle): Boolean = {
    val newHandles = bufferHandles.compute(handle.getId, (_, handles) => {
      if (handles== null) {
        throw new IllegalStateException(
          s"${handle.getId} not found and we attempted to remove handles!")
      }

      handles.remove(handle)

      if (handles.size() == 0) {
        null // remove since no more aliases exist
      } else {
        handles
      }
    })

    if (newHandles == null) {
      // we can now remove the underlying RapidsBufferId
      true
    } else {
      false
    }
  }


  /**
   * Lookup the buffer that corresponds to the specified buffer ID at the highest storage tier,
   * and acquire it.
   * NOTE: It is the responsibility of the caller to close the buffer.
   * @param id buffer identifier
   * @return buffer that has been acquired
   */
  def acquireBuffer(id: RapidsBufferId): RapidsBuffer = {
    (0 until RapidsBufferCatalog.MAX_BUFFER_LOOKUP_ATTEMPTS).foreach { _ =>
      val buffers = bufferMap.get(id)
      if (buffers == null || buffers.isEmpty) {
        throw new NoSuchElementException(s"Cannot locate buffers associated with ID: $id")
      }
      val buffer = buffers.head
      if (buffer.addReference()) {
        return buffer
      }
    }
    throw new IllegalStateException(s"Unable to acquire buffer for ID: $id")
  }

  def acquireBuffer(handle: RapidsBufferHandle): RapidsBuffer = {
    acquireBuffer(handle.getId)
  }

  /**
   * Lookup the buffer that corresponds to the specified buffer ID at the specified storage tier,
   * and acquire it.
   * NOTE: It is the responsibility of the caller to close the buffer.
   * @param id buffer identifier
   * @return buffer that has been acquired, None if not found
   */
  def acquireBuffer(id: RapidsBufferId, tier: StorageTier): Option[RapidsBuffer] = {
    val buffers = bufferMap.get(id)
    if (buffers != null) {
      buffers.find(_.storageTier == tier).foreach(buffer =>
        if (buffer.addReference()) {
          return Some(buffer)
        }
      )
    }
    None
  }

  /**
   * Check if the buffer that corresponds to the specified buffer ID is stored in a slower storage
   * tier.
   *
   * @param id   buffer identifier
   * @param tier storage tier to check
   * @return true if the buffer is stored in multiple tiers
   */
  def isBufferSpilled(id: RapidsBufferId, tier: StorageTier): Boolean = {
    val buffers = bufferMap.get(id)
    buffers != null && buffers.exists(_.storageTier > tier)
  }

  /** Get the table metadata corresponding to a buffer ID. */
  def getBufferMeta(id: RapidsBufferId): TableMeta = {
    val buffers = bufferMap.get(id)
    if (buffers == null || buffers.isEmpty) {
      throw new NoSuchElementException(s"Cannot locate buffer associated with ID: $id")
    }
    buffers.head.meta
  }

  /**
   * Register a new buffer with the catalog. An exception will be thrown if an
   * existing buffer was registered with the same buffer ID and storage tier.
   */
  def registerNewBuffer(buffer: RapidsBuffer): Unit = {
    val updater = new BiFunction[RapidsBufferId, Seq[RapidsBuffer], Seq[RapidsBuffer]] {
      override def apply(key: RapidsBufferId, value: Seq[RapidsBuffer]): Seq[RapidsBuffer] = {
        if (value == null) {
          Seq(buffer)
        } else {
          val(first, second) = value.partition(_.storageTier < buffer.storageTier)
          if (second.nonEmpty && second.head.storageTier == buffer.storageTier) {
            throw new DuplicateBufferException(
              s"Buffer ID ${buffer.id} at tier ${buffer.storageTier} already registered " +
                  s"${second.head}")
          }
          first ++ Seq(buffer) ++ second
        }
      }
    }

    bufferMap.compute(buffer.id, updater)
  }

  def registerNewBufferAndGetHandle(buffer: RapidsBuffer): RapidsBufferHandle = {
    registerNewBuffer(buffer)
    makeNewHandle(buffer.id, buffer.getSpillPriority)
  }

  /** Remove a buffer ID from the catalog at the specified storage tier. */
  def removeBufferTier(id: RapidsBufferId, tier: StorageTier): Unit = {
    val updater = new BiFunction[RapidsBufferId, Seq[RapidsBuffer], Seq[RapidsBuffer]] {
      override def apply(key: RapidsBufferId, value: Seq[RapidsBuffer]): Seq[RapidsBuffer] = {
        val updated = value.filter(_.storageTier != tier)
        if (updated.isEmpty) {
          null
        } else {
          updated
        }
      }
    }
    bufferMap.computeIfPresent(id, updater)
  }

  /** Remove a buffer ID from the catalog and release the resources of the registered buffers. */
  def removeBuffer(handle: RapidsBufferHandle): Boolean = {
    // if this is the last handle, remove the buffer
    if (stopTracking(handle)) {
      val buffers = bufferMap.remove(handle.getId)
      buffers.safeFree()
      true
    } else {
      false
    }
  }

  /**
   * Used by the ShuffleBufferCatalog only
   */
  def removeBuffer(id: RapidsBufferId): Boolean = {
    // if this is the last handle, remove the buffer
    val handles = bufferHandles.get(id)
    require(handles.size() == 1,
      s"Cannot remove a buffer by id ${id} when there are multiple handles")
    val handle = handles.peek()
    if (stopTracking(handle)) {
      val buffers = bufferMap.remove(handle.getId)
      buffers.safeFree()
      true
    } else {
      false
    }
  }

  /** Return the number of buffers currently in the catalog. */
  def numBuffers: Int = bufferMap.size()
}

object RapidsBufferCatalog extends Logging with Arm {
  private val MAX_BUFFER_LOOKUP_ATTEMPTS = 100

  val singleton = new RapidsBufferCatalog
  private var deviceStorage: RapidsDeviceMemoryStore = _
  private var hostStorage: RapidsHostMemoryStore = _
  private var diskBlockManager: RapidsDiskBlockManager = _
  private var diskStorage: RapidsDiskStore = _
  private var gdsStorage: RapidsGdsStore = _
  private var memoryEventHandler: DeviceMemoryEventHandler = _
  private var _shouldUnspill: Boolean = _

  private lazy val conf: SparkConf = {
    val env = SparkEnv.get
    if (env != null) {
      env.conf
    } else {
      // For some unit tests
      new SparkConf()
    }
  }

  // For testing
  def setDeviceStorage(rdms: RapidsDeviceMemoryStore): Unit = {
    deviceStorage = rdms
  }

  def init(rapidsConf: RapidsConf): Unit = {
    // We are going to re-initialize so make sure all of the old things were closed...
    closeImpl()
    assert(memoryEventHandler == null)
    deviceStorage = new RapidsDeviceMemoryStore()
    diskBlockManager = new RapidsDiskBlockManager(conf)
    if (rapidsConf.isGdsSpillEnabled) {
      gdsStorage = new RapidsGdsStore(diskBlockManager, rapidsConf.gdsSpillBatchWriteBufferSize)
      deviceStorage.setSpillStore(gdsStorage)
    } else {
      val hostSpillStorageSize = if (rapidsConf.hostSpillStorageSize == -1) {
        rapidsConf.pinnedPoolSize + rapidsConf.pageablePoolSize
      } else {
        rapidsConf.hostSpillStorageSize
      }
      hostStorage = new RapidsHostMemoryStore(hostSpillStorageSize, rapidsConf.pageablePoolSize)
      diskStorage = new RapidsDiskStore(diskBlockManager)
      deviceStorage.setSpillStore(hostStorage)
      hostStorage.setSpillStore(diskStorage)
    }

    logInfo("Installing GPU memory handler for spill")
    memoryEventHandler = new DeviceMemoryEventHandler(
      deviceStorage,
      rapidsConf.gpuOomDumpDir,
      rapidsConf.isGdsSpillEnabled,
      rapidsConf.gpuOomMaxRetries)
    Rmm.setEventHandler(memoryEventHandler)

    _shouldUnspill = rapidsConf.isUnspillEnabled
  }

  def close(): Unit = {
    logInfo("Closing storage")
    closeImpl()
  }

  private def closeImpl(): Unit = {
    if (memoryEventHandler != null) {
      // Workaround for shutdown ordering problems where device buffers allocated with this handler
      // are being freed after the handler is destroyed
      //Rmm.clearEventHandler()
      memoryEventHandler = null
    }

    if (deviceStorage != null) {
      deviceStorage.close()
      deviceStorage = null
    }
    if (hostStorage != null) {
      hostStorage.close()
      hostStorage = null
    }
    if (diskStorage != null) {
      diskStorage.close()
      diskStorage = null
    }
    if (gdsStorage != null) {
      gdsStorage.close()
      gdsStorage = null
    }
  }

  def getDeviceStorage: RapidsDeviceMemoryStore = deviceStorage

  def shouldUnspill: Boolean = _shouldUnspill

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
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback): RapidsBufferHandle =
    deviceStorage.addTable(table, contigBuffer, tableMeta, initialSpillPriority, spillCallback)

  /**
   * Adds a contiguous table to the device storage, taking ownership of the table.
   * @param id buffer ID to associate with this buffer
   * @param contigTable contiguous table to track in device storage
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   */
  def addContiguousTable(
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback): RapidsBufferHandle =
    deviceStorage.addContiguousTable(contigTable, initialSpillPriority, spillCallback)

  /**
   * Adds a buffer to the device storage, taking ownership of the buffer.
   * @param id buffer ID to associate with this buffer
   * @param buffer buffer that will be owned by the store
   * @param tableMeta metadata describing the buffer layout
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   */
  def addBuffer(
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback): RapidsBufferHandle =
    deviceStorage.addBuffer(buffer, tableMeta, initialSpillPriority, spillCallback)

  /**
   * Lookup the buffer that corresponds to the specified buffer ID and acquire it.
   * NOTE: It is the responsibility of the caller to close the buffer.
   * @param id buffer identifier
   * @return buffer that has been acquired
   */
  def acquireBuffer(id: RapidsBufferId): RapidsBuffer = singleton.acquireBuffer(id)

  def acquireBuffer(handle: RapidsBufferHandle): RapidsBuffer =
    singleton.acquireBuffer(handle.getId)

  /** Remove a buffer ID from the catalog and release the resources of the registered buffer. */
  def removeBuffer(handle: RapidsBufferHandle): Unit =
    singleton.removeBuffer(handle)

  def getDiskBlockManager(): RapidsDiskBlockManager = diskBlockManager
}
