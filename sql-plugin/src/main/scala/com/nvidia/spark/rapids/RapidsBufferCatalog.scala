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
import ai.rapids.cudf.{ContiguousTable, Cuda, DeviceMemoryBuffer, MemoryBuffer, NvtxColor, NvtxRange, Rmm}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.format.TableMeta
import org.apache.spark.{SparkConf, SparkEnv}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.{RapidsDiskBlockManager, TempSpillBufferId}

import java.util.concurrent.atomic.AtomicInteger

/**
 *  Exception thrown when inserting a buffer into the catalog with a duplicate buffer ID
 *  and storage tier combination.
 */
class DuplicateBufferException(s: String) extends RuntimeException(s) {}

/**
 * An object that client code uses to interact with an underlying RapidsBufferId.
 *
 * A handle is obtained when a buffer, batch, or table is added to the spill framework
 * via the `RapidsBufferCatalog` api.
 */
trait RapidsBufferHandle extends AutoCloseable {
  val id: RapidsBufferId

  /**
   * Sets the spill priority for this handle and updates the maximum priority
   * for the underlying `RapidsBuffer` if this new priority is the maximum.
   * @param newPriority new priority for this handle
   */
  def setSpillPriority(newPriority: Long): Unit
}

/**
 * Catalog for lookup of buffers by ID. The constructor is only visible for testing, generally
 * `RapidsBufferCatalog.singleton` should be used instead.
 */
class RapidsBufferCatalog extends AutoCloseable with Arm with Logging {

  /** Map of buffer IDs to buffers sorted by storage tier */
  private[this] val bufferMap = new ConcurrentHashMap[RapidsBufferId, Seq[RapidsBuffer]]

  /** Map of buffer IDs to buffer handles in insertion order */
  private[this] val bufferIdToHandles =
    new ConcurrentHashMap[RapidsBufferId, Seq[RapidsBufferHandleImpl]]()

  class RapidsBufferHandleImpl(
      override val id: RapidsBufferId,
      var priority: Long,
      spillCallback: SpillCallback)
    extends RapidsBufferHandle {

    private var closed = false

    override def setSpillPriority(newPriority: Long): Unit = {
      priority = newPriority
      updateUnderlyingRapidsBuffer(this)
    }

    /**
     * Get the spill priority that was associated with this handle. Since there can
     * be multiple handles associated with one `RapidsBuffer`, the priority returned
     * here is only useful for code in the catalog that updates the maximum priority
     * for the underlying `RapidsBuffer` as handles are added and removed.
     *
     * @return this handle's spill priority
     */
    def getSpillPriority: Long = priority

    /**
     * Each handle was created in a different part of the code and as such could have
     * different spill metrics callbacks. This function is used by the catalog to find
     * out what the last spill callback added. This last callback gets reports of
     * spill bytes if a spill were to occur to the `RapidsBuffer` this handle points to.
     *
     * @return the spill callback associated with this handle
     */
    def getSpillCallback: SpillCallback = spillCallback

    override def close(): Unit = synchronized {
      // since the handle is stored in the catalog in addition to being
      // handed out to potentially a `SpillableColumnarBatch` or `SpillableBuffer`
      // there is a chance we may double close it. For example, a broadcast exec
      // that is closing its spillable (and therefore the handle) + the handle being
      // closed from the catalog's close method. 
      if (!closed) {
        removeBuffer(this)
      }
      closed = true
    }
  }

  /**
   * Makes a new `RapidsBufferHandle` associated with `id`, keeping track
   * of the spill priority and callback within this handle.
   *
   * This function also adds the handle for internal tracking in the catalog.
   *
   * @param id the `RapidsBufferId` that this handle refers to
   * @param spillPriority the spill priority specified on creation of the handle
   * @param spillCallback this handle's spill callback
   * @note public for testing
   * @return a new instance of `RapidsBufferHandle`
   */
  private def makeNewHandle(
      id: RapidsBufferId,
      spillPriority: Long,
      spillCallback: SpillCallback): RapidsBufferHandle = {
    val handle = new RapidsBufferHandleImpl(id, spillPriority, spillCallback)
    trackNewHandle(handle)
    handle
  }

  /**
   * Adds a handle to the internal `bufferIdToHandles` map.
   *
   * The priority and callback of the `RapidsBuffer` will also be updated.
   *
   * @param handle handle to start tracking
   */
  private def trackNewHandle(handle: RapidsBufferHandleImpl): Unit = {
    bufferIdToHandles.compute(handle.id, (_, h) => {
      var handles = h
      if (handles == null) {
        handles = Seq.empty[RapidsBufferHandleImpl]
      }
      handles :+ handle
    })
    updateUnderlyingRapidsBuffer(handle)
  }

  /**
   * Called when the `RapidsBufferHandle` is no longer needed by calling code
   *
   * If this is the last handle associated with a `RapidsBuffer`, `stopTrackingHandle`
   * returns true, otherwise it returns false.
   *
   * @param handle handle to stop tracking
   * @return true: if this was the last `RapidsBufferHandle` associated with the
   *         underlying buffer.
   *         false: if there are remaining live handles
   */
  private def stopTrackingHandle(handle: RapidsBufferHandle): Boolean = {
    withResource(acquireBuffer(handle)) { buffer =>
      val id = handle.id
      var maxPriority = Long.MinValue
      val newHandles = bufferIdToHandles.compute(id, (_, handles) => {
        if (handles == null) {
          throw new IllegalStateException(
            s"$id not found and we attempted to remove handles!")
        }
        if (handles.size == 1) {
          require(handles.head == handle,
            "Tried to remove a single handle, and we couldn't match on it")
          null
        } else {
          val newHandles = handles.filter(h => h != handle).map { h =>
            maxPriority = maxPriority.max(h.getSpillPriority)
            h
          }
          if (newHandles.isEmpty) {
            null // remove since no more handles exist, should not happen
          } else {
            // we pick the last spillCallback inserted as the winner every time
            // this callback is going to get the metrics associated with this buffer's
            // spill
            newHandles
          }
        }
      })

      if (newHandles == null) {
        // tell calling code that no more handles exist,
        // for this RapidsBuffer
        true
      } else {
        // more handles remain, our priority changed so we need to update things
        buffer.setSpillPriority(maxPriority)
        buffer.setSpillCallback(newHandles.last.getSpillCallback)
        false // we have handles left
      }
    }
  }

  def registerDegenerateBuffer(
      bufferId: RapidsBufferId,
      meta: TableMeta,
      spillCallback: SpillCallback): RapidsBufferHandle = {
    val buffer = new DegenerateRapidsBuffer(bufferId, meta)
    registerNewBuffer(buffer)
    // TODO: what should the degenerate spill priority be..
    makeNewHandle(buffer.id, buffer.getSpillPriority, spillCallback)
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
   *           brand new to the store, or the `RapidsDeviceMemoryBuffer` is invalid and
   *           about to be removed).
   */
  private def getExistingRapidsBufferAndAcquire(
      buffer: DeviceMemoryBuffer): Option[RapidsBuffer] = {
    val eh = buffer.getEventHandler
    eh match {
      case null =>
        None
      case rapidsBuffer: RapidsBuffer =>
        // this doesn't work. addReference is going to inc refcount in the RapidsBuffer
        // but not prevent it from becoming invalid and being removed from the device store
        // because we call `free` from a spill, and that spill wants to stop tracking the buffer
        // the RapidsDeviceMemoryBuffer is no longer the owner.
        // we cannot allow the case above
        if (rapidsBuffer.addReference()) {
          Some(rapidsBuffer)
        } else {
          // if we raced with `RapidsBufferBase.free` and lost, we'd end up here
          // because it means the RapidsBuffer is no longer valid and the event handler
          // was reset, but the caller is trying to add a device memory buffer,
          // so it should create a new event handler.
          None
        }
      case _ =>
        throw new IllegalStateException("Unknown event handler")
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
      needsSync: Boolean = true): RapidsBufferHandle = synchronized {
    // first time we see `buffer`
    val existing = getExistingRapidsBufferAndAcquire(buffer)
    existing match {
      case None =>
        addBuffer(
          TempSpillBufferId(),
          buffer,
          tableMeta,
          initialSpillPriority,
          spillCallback,
          needsSync)
      case Some(rapidsBuffer) =>
        withResource(rapidsBuffer) { _ =>
          makeNewHandle(rapidsBuffer.id, initialSpillPriority, spillCallback)
        }
    }
  }
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
      needsSync: Boolean = true): RapidsBufferHandle = synchronized {
    val existing = getExistingRapidsBufferAndAcquire(contigTable.getBuffer)
    existing match {
      case None =>
        addContiguousTable(
          TempSpillBufferId(),
          contigTable,
          initialSpillPriority,
          spillCallback,
          needsSync)
      case Some(rapidsBuffer) =>
        withResource(rapidsBuffer) { _ =>
          makeNewHandle(rapidsBuffer.id, initialSpillPriority, spillCallback)
        }
    }
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
   *                  this device buffer (defaults to true)
   * @return RapidsBufferHandle handle for this table
   */
  def addContiguousTable(
      id: RapidsBufferId,
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback,
      needsSync: Boolean): RapidsBufferHandle = synchronized {
    addBuffer(
      id,
      contigTable.getBuffer,
      MetaUtils.buildTableMeta(id.tableId, contigTable),
      initialSpillPriority,
      spillCallback,
      needsSync)
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
      needsSync: Boolean): RapidsBufferHandle = synchronized {
    RapidsBufferCatalog.deviceStorage.addBuffer(
      id,
      buffer,
      tableMeta,
      initialSpillPriority,
      spillCallback,
      needsSync)
    makeNewHandle(id, initialSpillPriority, spillCallback)
  }


  /**
   * Called by the catalog when a handle is first added to the catalog, or to refresh
   * the priority of the underlying buffer if a handle's priority changed.
   */
  private def updateUnderlyingRapidsBuffer(handle: RapidsBufferHandle): Unit = {
    withResource(acquireBuffer(handle)) { buffer =>
      val handles = bufferIdToHandles.get(buffer.id)
      val maxPriority = handles.map(_.getSpillPriority).max
      // update the priority of the underlying RapidsBuffer to be the
      // maximum priority for all handles associated with it
      buffer.setSpillPriority(maxPriority)
      buffer.setSpillCallback(handles.last.getSpillCallback)
    }
  }

  /**
   * Lookup the buffer that corresponds to the specified handle at the highest storage tier,
   * and acquire it.
   * NOTE: It is the responsibility of the caller to close the buffer.
   * @param handle handle associated with this `RapidsBuffer`
   * @return buffer that has been acquired
   */
  def acquireBuffer(handle: RapidsBufferHandle): RapidsBuffer = {
    acquireBuffer(handle.id)
  }

  private def acquireBuffer(id: RapidsBufferId): RapidsBuffer = {
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
  private def isBufferSpilled(id: RapidsBufferId, tier: StorageTier): Boolean = {
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
  private def registerNewBuffer(buffer: RapidsBuffer): Unit = {
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

  private[this] val spillCount = new AtomicInteger(0)

  /**
   * Free memory in this store by spilling buffers to the spill store synchronously.
   *
   * @param targetTotalSize maximum total size of this store after spilling completes
   * @return optionally number of bytes that were spilled, or None if this called
   *         made no attempt to spill due to a detected spill race
   */
  def synchronousSpill(store: RapidsBufferStore, targetTotalSize: Long): Option[Long] =
    synchronousSpill(store, targetTotalSize, Cuda.DEFAULT_STREAM)

  /**
   * Free memory in this store by spilling buffers to the spill store synchronously.
   * @param targetTotalSize maximum total size of this store after spilling completes
   * @param stream CUDA stream to use or null for default stream
   * @return optionally number of bytes that were spilled, or None if this called
   *         made no attempt to spill due to a detected spill race
   */
  def synchronousSpill(
      store: RapidsBufferStore,
      targetTotalSize: Long,
      stream: Cuda.Stream): Option[Long] = {
    require(targetTotalSize >= 0, s"Negative spill target size: $targetTotalSize")

    var shouldRetry = false
    var totalSpilled: Long = 0

    if (store.getTotalSpillableBytes > targetTotalSize) {
      withResource(new NvtxRange(s"synchronousSpill ${store.name}", NvtxColor.ORANGE)) { _ =>
        logDebug(s"${store.name} store spilling to reduce usage from " +
          s"${store.getTotalBytes} total (${store.getTotalSpillableBytes} spillable) " +
          s"to $targetTotalSize bytes")

        var waited = false
        var exhausted = false

        while (!exhausted && !shouldRetry && store.getTotalSpillableBytes > targetTotalSize) {
          trySpillAndFreeBuffer(store, stream) match {
            case None =>
              // another thread won and spilled before this thread did. Lets retry the original
              // allocation again
              shouldRetry = true
            case Some(amountSpilled) =>
              if (amountSpilled != 0) {
                totalSpilled += amountSpilled
                waited = false
              } else {
                // we didn't spill in this iteration, and we'll try to wait a bit to see if
                // other threads finish up their work and release pointers to the released
                // buffer
                if (!waited && store.hasPendingFreeBytes) {
                  waited = true
                  store.waitForPending(targetTotalSize)
                } else {
                  exhausted = true
                  logWarning("Unable to spill enough to meet request. " +
                    s"Total=${store.getTotalBytes} " +
                    s"Spillable=${store.getTotalSpillableBytes} " +
                    s"Target=$targetTotalSize")
                }
              }
          }
        }
        logDebug(s"$this spill complete")
      }
    }

    if (totalSpilled > 0) {
      Some(totalSpilled)
    } else if (shouldRetry) {
      // if we are going to retry, and didn't spill, returning None prevents extra
      // logs where we say we spilled 0 bytes from X store
      None
    } else {
      Some(0)
    }
  }

  def trySpillAndFreeBuffer(store: RapidsBufferStore, stream: Cuda.Stream): Option[Long] = {
    val mySpillCount = spillCount.incrementAndGet()
    synchronized {
      if (spillCount.get() == mySpillCount) {
        val nextSpillable = store.nextSpillable()
        if (nextSpillable == null) {
          Some(0)
        } else {
          val spillStore = store.spillStore
          if (spillStore == null) {
            throw new OutOfMemoryError("Requested to spill without a spill store")
          }
          spillBufferToStore(nextSpillable, spillStore, stream)
          Some(nextSpillable.size)
        }
      } else {
        None
      }
    }
  }

  private def spillBufferToStore(
      buffer: RapidsBuffer,
      spillStore: RapidsBufferStore,
      stream: Cuda.Stream): Unit = synchronized {
    if (buffer.addReference()) {
      withResource(buffer) { _ =>
        logDebug(s"Spilling $buffer ${buffer.id} to ${spillStore.name} ")
        val removed = isBufferSpilled(buffer.id, buffer.storageTier)
        if (!removed) {
          val spillCallback = buffer.getSpillCallback
          spillCallback(buffer.storageTier, spillStore.tier, buffer.size)
          val newBuffer =
            spillStore.copyBuffer(buffer, buffer.getMemoryBuffer, stream)
          registerNewBuffer(newBuffer)
        }
        removeBufferTier(buffer.id, buffer.storageTier)
      }
    } else {
      logDebug(s"Skipping spilling $buffer ${buffer.id} to ${spillStore.name} as it is " +
        s"already stored in multiple tiers")
    }
  }

  def unspillBufferToDeviceStore(
    buffer: RapidsBuffer,
    memoryBuffer: MemoryBuffer,
    stream: Cuda.Stream): RapidsBuffer = synchronized {
    val newBuffer = RapidsBufferCatalog.deviceStorage.copyBuffer(
      buffer,
      memoryBuffer,
      stream)
    registerNewBuffer(newBuffer)
    newBuffer
  }

  /** Remove a buffer ID from the catalog at the specified storage tier. */
  private def removeBufferTier(id: RapidsBufferId, tier: StorageTier): Unit = synchronized {
    val updater = new BiFunction[RapidsBufferId, Seq[RapidsBuffer], Seq[RapidsBuffer]] {
      override def apply(key: RapidsBufferId, value: Seq[RapidsBuffer]): Seq[RapidsBuffer] = {
        val bufferAtTier = value.filter(_.storageTier == tier)
        require(bufferAtTier.size == 1,
          s"Multiple buffers at the same tier found for $id!")
        // since we are removing it from this tier, we can call free on it.
        bufferAtTier.head.safeFree()
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

  /**
   * Remove a buffer handle from the catalog and, if it this was the final handle,
   * release the resources of the registered buffers.
   *
   * @return true: if the buffer for this handle was removed from the spill framework
   *               (`handle` was the last handle)
   *         false: if buffer was not removed due to other live handles.
   */
  private def removeBuffer(handle: RapidsBufferHandle): Boolean = synchronized {
    // if this is the last handle, remove the buffer
    if (stopTrackingHandle(handle)) {
      logDebug(s"Removing buffer ${handle.id}")
      bufferMap.remove(handle.id).safeFree()
      true
    } else {
      false
    }
  }

  /** Return the number of buffers currently in the catalog. */
  def numBuffers: Int = bufferMap.size()

  override def close(): Unit = {
    bufferIdToHandles.values.forEach { handles =>
      handles.foreach(_.close())
    }
    bufferIdToHandles.clear()
  }
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
    singleton.close()

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
   * Adds a contiguous table to the device storage. This does NOT take ownership of the
   * contiguous table, so it is the responsibility of the caller to close it. The refcount of the
   * underlying device buffer will be incremented so the contiguous table can be closed before
   * this buffer is destroyed.
   * @param contigTable contiguous table to trackNewHandle in device storage
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @return RapidsBufferHandle associated with this buffer
   */
  def addContiguousTable(
      contigTable: ContiguousTable,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback): RapidsBufferHandle = {
    singleton.addContiguousTable(contigTable, initialSpillPriority, spillCallback)
  }

  /**
   * Adds a buffer to the device storage. This does NOT take ownership of the
   * buffer, so it is the responsibility of the caller to close it.
   * @param buffer buffer that will be owned by the store
   * @param tableMeta metadata describing the buffer layout
   * @param initialSpillPriority starting spill priority value for the buffer
   * @param spillCallback a callback when the buffer is spilled. This should be very light weight.
   *                      It should never allocate GPU memory and really just be used for metrics.
   * @return RapidsBufferHandle associated with this buffer
   */
  def addBuffer(
      buffer: DeviceMemoryBuffer,
      tableMeta: TableMeta,
      initialSpillPriority: Long,
      spillCallback: SpillCallback = RapidsBuffer.defaultSpillCallback): RapidsBufferHandle = {
    singleton.addBuffer(buffer, tableMeta, initialSpillPriority, spillCallback)
  }

  /**
   * Lookup the buffer that corresponds to the specified buffer handle and acquire it.
   * NOTE: It is the responsibility of the caller to close the buffer.
   * @param handle buffer handle
   * @return buffer that has been acquired
   */
  def acquireBuffer(handle: RapidsBufferHandle): RapidsBuffer =
    singleton.acquireBuffer(handle)

  def synchronousSpill(store: RapidsDeviceMemoryStore, targetSize: Long): Option[Long] = {
    singleton.synchronousSpill(store, targetSize)
  }

  def getDiskBlockManager(): RapidsDiskBlockManager = diskBlockManager
}
