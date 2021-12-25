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
    acquireIfNecessary(context, waitMetric, true)
  }

  /**
   * Tasks must call this when they begin to use the GPU.
   * If the task has not already acquired the GPU semaphore then it is acquired,
   * blocking if necessary.
   * NOTE: A task completion listener will automatically be installed to ensure
   *       the semaphore is always released by the time the task completes.
   */
  def acquireIfNecessary(context: TaskContext, waitMetric: GpuMetric,
                         couldExplode: Boolean): Unit = {
    if (enabled && context != null) {
      getInstance.acquireIfNecessary(context, waitMetric, couldExplode)
    }
  }

  def taskUsedDeviceMemory(
      context: TaskContext, deviceMemorySize: Long, couldExplode: Boolean) = {
    if (enabled && context != null) {
      getInstance.taskUsedDeviceMemory(context, deviceMemorySize, couldExplode)
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
  private val semaphore = new Semaphore(tasksPerGpu)
  private val memSemaphore = new Semaphore(16)
  var fudge = 16 - tasksPerGpu
  // Map to track which tasks have acquired the semaphore.
  class ActiveTask { 
    val count: MutableInt = new MutableInt(1)
    var deviceMemorySize: Long = 0L
    var couldExplode: Boolean = false
  }

  private val activeTasks = new ConcurrentHashMap[Long, ActiveTask]

  def taskUsedDeviceMemory(context: TaskContext, deviceMemorySize: Long, couldExplode: Boolean) = {
    val taskAttemptId = context.taskAttemptId()
    val refs = activeTasks.get(taskAttemptId)
    var othersUsed = 0L
    if (refs != null) {
      refs.deviceMemorySize += deviceMemorySize
      val activeTaskIds = activeTasks.keys()
      while (activeTaskIds.hasMoreElements) {
        val tId = activeTaskIds.nextElement()
        if (tId != taskAttemptId) {
          val activeTask = activeTasks.get(tId)
          if (activeTask != null) {
            othersUsed += activeTask.deviceMemorySize
          }
        }
      }
      // iUsedMB + othersUsedMB is overall GPU memory used right now (more or less)
      // it comes from the scan tasks only, and it's actual memory, but it doesn't include
      // any effects of downstream tasks (will they filter, or magnify the data on the gpu)
      // Rule:
      // if all stages running end in an exchange, we can play games. If the stage has
      // an expand that may be bad, if the stage is the stream side of a broadcast that
      // may also be bad.
      val maxAllocationMB = (GpuDeviceManager.poolAllocationTotal.toDouble/1024L/1024L).toInt
      val iUsedMB = (refs.deviceMemorySize.toDouble/1024L/1024L).toInt
      val othersUsedMB = (othersUsed.toDouble/1024L/1024L).toInt
      val allocatedMB = (ai.rapids.cudf.Rmm.getTotalBytesAllocated.toDouble/1024L/1024L).toInt
      val underUtilFactor = (othersUsedMB.toDouble + iUsedMB.toDouble)/maxAllocationMB

      logInfo(s"Task $taskAttemptId actually used $iUsedMB MB, and " +
        s"others used $othersUsedMB MB. Under util factor ($underUtilFactor)." +
        s"We have $allocatedMB MB overall allocated out of $maxAllocationMB " +
        s"We have ${semaphore.availablePermits()} available permits, and " +
        s"${semaphore.getQueueLength} queued. Could explode? $couldExplode")

      if (couldExplode || semaphore.getQueueLength > 1) {
        if (couldExplode) {
          logInfo("COULD EXPLODE!!")
          if (semaphore.availablePermits() > tasksPerGpu){
            // block until we acquire all this
            semaphore.acquire(16 - tasksPerGpu)
            synchronized { 
              fudge = 16 - tasksPerGpu
            }
          } else {
            logInfo("Already reclaimed credits..")
          }
        } else if (underUtilFactor < 0.25) {
          synchronized { 
            logInfo(s"Under utilized ${underUtilFactor}. Available ${fudge}")
            if (fudge > 0) {
              logInfo(s"Adding permit. Under utilized ${underUtilFactor}, Available ${fudge}")
              semaphore.release(1)
              fudge = fudge - 1
            }
          }
        } else {
          synchronized { 
            logInfo(s"Over utilized ${underUtilFactor}, removing permit. Available ${fudge}")
            if (fudge < 16 - tasksPerGpu) {
              logInfo(s"Removing permit. Over utilized ${underUtilFactor}, Available ${fudge}")
              semaphore.acquire(1)
              fudge = fudge + 1
            }
          }
        }
      }
    }
  }

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
                         couldExplode: Boolean): Unit = {
    withResource(new NvtxWithMetrics("Acquire GPU", NvtxColor.RED, waitMetric)) { _ =>
      val taskAttemptId = context.taskAttemptId()
      val refs = activeTasks.get(taskAttemptId)
      val othersCouldExplode = checkCouldExplode

      if (!couldExplode) {
        // nothing is explody
        // acquire the non-explody semaphore, else acquire regular
        if (refs == null || refs.count.getValue == 0) {
          logInfo(s"Task $taskAttemptId acquiring mem GPU ${memSemaphore.availablePermits()}")
          synchronized {
            while (othersCouldExplode && semaphore.availablePermits() < 4) {
              logInfo(s"Waiting non others: ${othersCouldExplode}, permits ${semaphore.availablePermits()}")
              wait()
            }
          }
          memSemaphore.acquire()
          if (refs != null) {
            refs.count.increment()
          } else {
            // first time this task has been seen
            val at = new ActiveTask
            at.couldExplode = couldExplode
            activeTasks.put(taskAttemptId, at)
            context.addTaskCompletionListener[Unit](completeTask)
          }
          GpuDeviceManager.initializeFromTask()
        }
      } else {
        // if something non-explody exists, we have to wait for it to finish
        if (refs == null || refs.count.getValue == 0) {
          logInfo(s"Task $taskAttemptId acquiring GPU ${semaphore.availablePermits()}")
          synchronized {
            while (!othersCouldExplode && memSemaphore.availablePermits() < 16) {
              logInfo(s"Waiting others: ${othersCouldExplode}, permits ${memSemaphore.availablePermits()}")
              wait()
            }
          }
          semaphore.acquire()
          if (refs != null) {
            refs.count.increment()
          } else {
            // first time this task has been seen
            val at = new ActiveTask
            at.couldExplode = couldExplode
            activeTasks.put(taskAttemptId, at)
            context.addTaskCompletionListener[Unit](completeTask)
          }
          GpuDeviceManager.initializeFromTask()
        }
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
          logInfo(s"Task $taskAttemptId releasing GPU")
          if (refs.couldExplode) {
            semaphore.release()
          } else {
            memSemaphore.release()
          }
          synchronized {
            notifyAll()
          }
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
      logInfo(s"Task $taskAttemptId releasing GPU")
      if (refs.couldExplode) {
        semaphore.release()
      } else {
        memSemaphore.release()
      }
    }
  }

  def shutdown(): Unit = {
    if (!activeTasks.isEmpty) {
      logDebug(s"shutting down with ${activeTasks.size} tasks still registered")
    }
  }
}
