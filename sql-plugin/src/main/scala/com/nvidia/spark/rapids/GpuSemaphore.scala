/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import java.util.concurrent.{ConcurrentHashMap, Semaphore}
import ai.rapids.cudf.{NvtxColor, NvtxRange}
import org.apache.commons.lang3.mutable.MutableInt
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging


object GpuSemaphore {

  private val enabled = {
    val propstr = System.getProperty("com.nvidia.spark.rapids.semaphore.enabled")
    if (propstr != null) {
      java.lang.Boolean.parseBoolean(propstr)
    } else {
      true
    }
  }

  // DO NOT ACCESS DIRECTLY!  Use `getInstance` instead.
  @volatile private var instance: GpuSemaphore = _

  private def getInstance: GpuSemaphore = {
    if (instance == null) {
      GpuSemaphore.synchronized {
        // The instance is trying to be used before it is initialized.
        // Since we don't have access to a configuration object here,
        // default to only one task per GPU behavior.
        if (instance == null) {
          initialize(1)
        }
      }
    }
    instance
  }

  /**
   * Initializes the GPU task semaphore.
   * @param tasksPerGpu number of tasks that will be allowed to use the GPU concurrently.
   */
  def initialize(tasksPerGpu: Int): Unit = synchronized {
    if (enabled) {
      if (instance != null) {
        throw new IllegalStateException("already initialized")
      }
      instance = new GpuSemaphore(tasksPerGpu)
    }
  }
  def acquireIfNecessary(context: TaskContext, waitMetric: GpuMetric): Unit = {
    acquireIfNecessary(context, waitMetric, true, -1)
  }

  /**
   * Tasks must call this when they begin to use the GPU.
   * If the task has not already acquired the GPU semaphore then it is acquired,
   * blocking if necessary.
   * NOTE: A task completion listener will automatically be installed to ensure
   *       the semaphore is always released by the time the task completes.
   */
  def acquireIfNecessary(context: TaskContext, waitMetric: GpuMetric,
                         couldExplode: Boolean, memNeeded: Long): Unit = {
    if (enabled && context != null) {
      getInstance.acquireIfNecessary(context, waitMetric, couldExplode, memNeeded)
    }
  }

  /**
   * Tasks must call this when they are finished using the GPU.
   */
  def releaseIfNecessary(context: TaskContext): Unit = {
    if (enabled && context != null) {
      getInstance.releaseIfNecessary(context)
    }
  }

  /**
   * Uninitialize the GPU semaphore.
   * NOTE: This does not wait for active tasks to release!
   */
  def shutdown(): Unit = synchronized {
    if (instance != null) {
      instance.shutdown()
      instance = null
    }
  }
}

private final class GpuSemaphore(tasksPerGpu: Int) extends Logging with Arm {
  private val creditsTotalMB: Int = ((GpuDeviceManager.poolAllocationTotal)/1024.0D/1024.0D).toInt
  private val perTaskCreditsMB: Int = creditsTotalMB/tasksPerGpu
  private val semaphore = new Semaphore(creditsTotalMB)
  // Map to track which tasks have acquired the semaphore.
  class ActiveTask { 
    val count: MutableInt = new MutableInt(1)
    var deviceMemorySize: Int = 0
    var couldExplode: Boolean = false
  }

  private val activeTasks = new ConcurrentHashMap[Long, ActiveTask]

  def checkCouldExplode: Boolean = {
    val keys = activeTasks.keys()
    var couldExplode = false
    while (keys.hasMoreElements) {
      val active = keys.nextElement()
      val t = activeTasks.get(active)
      if (t != null) {
        couldExplode = couldExplode || t.couldExplode
      }
    }
    couldExplode
  }

  def acquireIfNecessary(context: TaskContext,
                         waitMetric: GpuMetric,
                         couldExplode: Boolean,
                         memNeeded: Long): Unit = {
    withResource(new NvtxWithMetrics("Acquire GPU", NvtxColor.RED, waitMetric)) { _ =>
      val taskAttemptId = context.taskAttemptId()
      val refs = activeTasks.get(taskAttemptId)
      if (refs == null || refs.count.getValue == 0) {
        val memNeededMB =
          Math.min((memNeeded.toDouble / 1024.0 / 1024.0).toInt, perTaskCreditsMB)
        logInfo(s"Task $taskAttemptId acquiring GPU " +
          s"Available: ${semaphore.availablePermits()}MB, needing ${memNeededMB}MB")
        val amountAcquired = if (!couldExplode && memNeededMB < perTaskCreditsMB) {
          // we need to make sure we can fit this
          semaphore.acquire(memNeededMB)
          memNeededMB
        } else {
          semaphore.acquire(perTaskCreditsMB)
          perTaskCreditsMB
        }
        if (refs != null) {
          refs.count.increment()
        } else {
          // first time this task has been seen
          val at = new ActiveTask
          at.couldExplode = couldExplode
          at.deviceMemorySize = amountAcquired
          activeTasks.put(taskAttemptId, at)
          context.addTaskCompletionListener[Unit](completeTask)
        }
        GpuDeviceManager.initializeFromTask()
      } else if (refs != null) {
        val memNeededMB = (memNeeded / 1024.0D).toInt
        if (refs.deviceMemorySize > memNeededMB){
          logInfo(s"Task $taskAttemptId asking for less memory now. " +
            s"It had ${refs.deviceMemorySize} before, and now it wants $memNeededMB")
          semaphore.release(refs.deviceMemorySize - memNeededMB)
        } else {
          logInfo(s"Task $taskAttemptId asking for MORE memory now. " +
            s"It had ${refs.deviceMemorySize} before, and now it wants $memNeededMB")
          semaphore.acquire(memNeededMB - refs.deviceMemorySize)
        }
        refs.deviceMemorySize = memNeededMB
      }
    }
  }

  def releaseIfNecessary(context: TaskContext): Unit = {
    val nvtxRange = new NvtxRange("Release GPU", NvtxColor.RED)
    try {
      val taskAttemptId = context.taskAttemptId()
      val refs = activeTasks.get(taskAttemptId)
      if (refs != null && refs.count.getValue > 0) {
        if (refs.count.decrementAndGet() == 0) {
          logInfo(s"Task $taskAttemptId releasing GPU ${refs.deviceMemorySize}")
          semaphore.release(refs.deviceMemorySize)
        }
      }
    } finally {
      nvtxRange.close()
    }
  }

  def completeTask(context: TaskContext): Unit = {
    val taskAttemptId = context.taskAttemptId()
    val refs = activeTasks.remove(taskAttemptId)
    if (refs == null) {
      throw new IllegalStateException(s"Completion of unknown task $taskAttemptId")
    }
    if (refs.count.getValue > 0) {
      logInfo(s"Task $taskAttemptId releasing GPU ${refs.deviceMemorySize}")
      semaphore.release(refs.deviceMemorySize)
    }
  }

  def shutdown(): Unit = {
    if (!activeTasks.isEmpty) {
      logDebug(s"shutting down with ${activeTasks.size} tasks still registered")
    }
  }
}
