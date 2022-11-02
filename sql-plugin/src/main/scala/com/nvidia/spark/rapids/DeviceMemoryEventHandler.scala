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

import java.io.File
import java.lang.management.ManagementFactory
import java.util.concurrent.atomic.AtomicLong
import ai.rapids.cudf.{Cuda, NvtxColor, NvtxRange, Rmm, RmmEventHandler}
import com.sun.management.HotSpotDiagnosticMXBean
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.execution.TrampolineUtil

import scala.collection.mutable

/**
 * RMM event handler to trigger spilling from the device memory store.
 * @param store device memory store that will be triggered to spill
 * @param oomDumpDir local directory to create heap dumps on GPU OOM
 * @param isGdsSpillEnabled true if GDS is enabled for device->disk spill
 * @param maxFailedOOMRetries maximum number of retries for OOMs after
 *                            depleting the device store
 */
class DeviceMemoryEventHandler(
    store: RapidsDeviceMemoryStore,
    oomDumpDir: Option[String],
    isGdsSpillEnabled: Boolean,
    maxFailedOOMRetries: Int)
  extends RmmEventHandler
    with MemoryStoreHandler
    with Logging with Arm {

  // Flag that ensures we dump stack traces once and not for every allocation
  // failure. The assumption is that unhandled allocations will be fatal
  // to the process at this stage, so we only need this once before we exit.
  private var dumpStackTracesOnFailureToHandleOOM = true

  // Thread local used to the number of times a sync has been attempted, when
  // handling OOMs after depleting the device store.
  private val oomRetryState =
    ThreadLocal.withInitial[OOMRetryState](() => new OOMRetryState)

  /**
   * A small helper class that helps keep track of retry counts as we trigger
   * synchronizes on a depleted store.
   */
  class OOMRetryState {
    private var synchronizeAttempts = 0
    private var retryCountLastSynced = 0

    def getRetriesSoFar: Int = synchronizeAttempts

    private def reset(): Unit = {
      synchronizeAttempts = 0
      retryCountLastSynced = 0
    }

    /**
     * If we have synchronized less times than `maxFailedOOMRetries` we allow
     * this retry to proceed, and track the `retryCount` provided by cuDF. If we
     * are above our counter, we reset our state.
     */
    def shouldTrySynchronizing(retryCount: Int): Boolean = {
      if (synchronizeAttempts < maxFailedOOMRetries) {
        retryCountLastSynced = retryCount
        synchronizeAttempts += 1
        true
      } else {
        reset()
        false
      }
    }

    /**
     * We reset our counters if `storeSize` is non-zero (as we only track when the store is
     * depleted), or `retryCount` is less than or equal to what we had previously
     * recorded in `shouldTrySynchronizing`. We do this to detect that the new failures
     * are for a separate allocation (we need to give this new attempt a new set of
     * retries.)
     *
     * For example, if an allocation fails and we deplete the store, `retryCountLastSynced`
     * is set to the last `retryCount` sent to us by cuDF as we keep allowing retries
     * from cuDF. If we succeed, cuDF resets `retryCount`, and so the new count sent to us
     * must be <= than what we saw last, so we can reset our tracking.
     */
    def resetIfNeeded(retryCount: Int, storeSize: Long): Unit = {
      if (storeSize != 0 || retryCount <= retryCountLastSynced) {
        reset()
      }
    }
  }

  /**
   * Handles RMM allocation failures by spilling buffers from device memory.
   * @param allocSize the byte amount that RMM failed to allocate
   * @param retryCount the number of times this allocation has been retried after failure
   * @return true if allocation should be reattempted or false if it should fail
   */
  override def onAllocFailure(allocSize: Long, retryCount: Int): Boolean = {
    // check arguments for good measure
    require(allocSize >= 0, 
      s"onAllocFailure invoked with invalid allocSize $allocSize")

    require(retryCount >= 0, 
      s"onAllocFailure invoked with invalid retryCount $retryCount")

    try {
      withResource(new NvtxRange("onAllocFailure", NvtxColor.RED)) { _ =>
        val storeSize = store.currentSize
        val attemptMsg = if (retryCount > 0) {
          s"Attempt ${retryCount}. "
        } else {
          "First attempt. "
        }

        val retryState = oomRetryState.get()
        retryState.resetIfNeeded(retryCount, storeSize)

        logInfo(s"Device allocation of $allocSize bytes failed, device store has " +
          s"$storeSize bytes. $attemptMsg" +
          s"Total RMM allocated is ${Rmm.getTotalBytesAllocated} bytes. ")
        if (storeSize == 0) {
          if (retryState.shouldTrySynchronizing(retryCount)) {
            Cuda.deviceSynchronize()
            logWarning(s"[RETRY ${retryState.getRetriesSoFar}] " +
              s"Retrying allocation of $allocSize after a synchronize. " +
              s"Total RMM allocated is ${Rmm.getTotalBytesAllocated} bytes.")
            true
          } else {
            logWarning(s"Device store exhausted, unable to allocate $allocSize bytes. " +
              s"Total RMM allocated is ${Rmm.getTotalBytesAllocated} bytes.")
            synchronized {
              if (dumpStackTracesOnFailureToHandleOOM) {
                dumpStackTracesOnFailureToHandleOOM = false
                GpuSemaphore.dumpActiveStackTracesToLog()
              }
            }
            oomDumpDir.foreach(heapDump)
            false
          }
        } else {
          val targetSize = Math.max(storeSize - allocSize, 0)
          logDebug(s"Targeting device store size of $targetSize bytes")
          val amountSpilled = store.synchronousSpill(targetSize)
          logInfo(s"Spilled $amountSpilled bytes from the device store")
          if (isGdsSpillEnabled) {
            TrampolineUtil.incTaskMetricsDiskBytesSpilled(amountSpilled)
          } else {
            TrampolineUtil.incTaskMetricsMemoryBytesSpilled(amountSpilled)
          }
          true
        }
      }
    } catch {
      case t: Throwable =>
        logError(s"Error handling allocation failure", t)
        false
    }
  }

  override def getAllocThresholds: Array[Long] = null

  override def getDeallocThresholds: Array[Long] = null

  override def onAllocThreshold(totalAllocated: Long): Unit = {
  }

  override def onDeallocThreshold(totalAllocated: Long): Unit = {
  }

  var lastMax: Long = 0

  override def onMaxAllocated(maxAllocated: Long): Unit = {
   //if (maxAllocated >= lastMax) {
   //  setMaxStackTrace()
   //}
   //lastMax = maxAllocated
  }
  var totalAllocated: Long = 0L
  var totalAllocatedIncludingSpillable: Long = 0L

  override def onSpillStoreSizeChange(delta: Long): Unit = synchronized {
    DeviceMemoryEventHandler.onSpillStoreSizeChange(store)

   //val actualMax = lastMax - delta
   //logInfo(s"Spill changed delta $delta. Last max was ${lastMax} is now ${actualMax}")
   //// adjust cuDF's max
   //Rmm.resetScopedMaximumBytesAllocated(actualMax, true)
   //if (actualMax >= lastMax) {
   //  setMaxStackTrace()
   //}
   //// keep track of the last max as we have just reset it
   //lastMax = actualMax
  }

  override def onAllocated(size: Long): Unit = synchronized {
    DeviceMemoryEventHandler.onAllocated(size, store)
  }

  override def onFreed(size: Long): Unit = synchronized {
    try {
      DeviceMemoryEventHandler.onFreed(size, store)
    } catch {
      case t: Throwable =>
        logError("Exception while handling free", t)
    }
  }

private def heapDump(dumpDir: String): Unit = {
    val dumpPath = getDumpPath(dumpDir)
    logWarning(s"Dumping heap to $dumpPath")
    val server = ManagementFactory.getPlatformMBeanServer
    val mxBean = ManagementFactory.newPlatformMXBeanProxy(server,
      "com.sun.management:type=HotSpotDiagnostic", classOf[HotSpotDiagnosticMXBean])
    mxBean.dumpHeap(dumpPath, false)
  }

  private val dumps: AtomicLong = new AtomicLong(0)
  private def getDumpPath(dumpDir: String): String = {
    // pid is typically before the '@' character in the name
    val pid = ManagementFactory.getRuntimeMXBean.getName.split('@').head
    new File(dumpDir, s"gpu-oom-$pid-${dumps.incrementAndGet}.hprof").toString
  }
}

object DeviceMemoryEventHandler extends Logging {
  var maxStackTrace: Option[String] = None
  var totalAllocated: Long = 0L
  var totalAllocatedWithSpillable: Long = 0L
  var lastMax: Long = 0L
  var lastMaxTotal: Long = 0L
  var storeHack: RapidsDeviceMemoryStore = null
  def setStore(store: RapidsDeviceMemoryStore): Unit = {
    storeHack = store
  }
  def resetTotalAllocated(): Unit = synchronized {
    totalAllocated = 0
    totalAllocatedWithSpillable = 0
    lastMax = 0
    lastMaxTotal = 0
    if (storeHack != null && storeHack.currentSize > 0) {
      logWarning(s"STORE HAS NON-ZERO SIZE AT RESET ${storeHack.currentSize} B")
    }
  }
  def onAllocated(size: Long, store: RapidsDeviceMemoryStore) = synchronized {
    if (storeHack == null) {
      storeHack = store
    }
    totalAllocated += size
    totalAllocatedWithSpillable = totalAllocated - store.currentSize
    if (totalAllocated > lastMaxTotal) {
      lastMaxTotal = totalAllocated
    }
    if (totalAllocatedWithSpillable > lastMax) {
      lastMax = totalAllocatedWithSpillable
      setMaxStackTrace()
    }
  }

  def onFreed(size: Long, store: RapidsDeviceMemoryStore): Unit = synchronized {
    if (storeHack == null) {
      storeHack = store
    }
    totalAllocated -= size
    totalAllocatedWithSpillable = totalAllocated - store.currentSize
    if (totalAllocated > lastMaxTotal) {
      lastMaxTotal = totalAllocated
    }
    if (totalAllocatedWithSpillable > lastMax) {
      lastMax = totalAllocatedWithSpillable
      setMaxStackTrace()
    }
  }

  def onSpillStoreSizeChange(store: RapidsDeviceMemoryStore): Unit = synchronized {
    if (storeHack == null) {
      storeHack = store
    }
    totalAllocatedWithSpillable = totalAllocated - store.currentSize
    if (totalAllocatedWithSpillable > lastMax) {
      lastMax = totalAllocatedWithSpillable
      setMaxStackTrace()
    }
  }

  private def setMaxStackTrace(): Unit = {
    val sb = new mutable.StringBuilder()
    Thread.currentThread().getStackTrace.foreach { stackTraceElement =>
      sb.append("    " + stackTraceElement + "\n")
    }
    maxStackTrace = Some(sb.toString())
  }
}
