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
import ai.rapids.cudf.{NvtxColor, NvtxUniqueRange, NvtxRange, Table}
import org.apache.commons.lang3.mutable.MutableInt
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging

object GpuSemaphore extends Logging {
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

  val activeTasks =  new java.util.concurrent.atomic.AtomicInteger(0)

  /**
   * Tasks must call this when they begin to use the GPU.
   * If the task has not already acquired the GPU semaphore then it is acquired,
   * blocking if necessary.
   * NOTE: A task completion listener will automatically be installed to ensure
   *       the semaphore is always released by the time the task completes.
   */
  def acquireIfNecessary(context: TaskContext, waitMetric: GpuMetric): Unit = {
    if (enabled && context != null) {
      if (getInstance.acquireIfNecessary(context, waitMetric)) {
        logInfo(s"Semaphore ACQUIRED was at: ${activeTasks.getAndIncrement()}")
      }
    }
  }

  /**
   * Given a task that has acquired the semaphore, this updates the amount of memory
   * it actually needs (say a scan, which now knows it is outputing X MB).
   *
   * If the amount of memory needed is higher than the previous acquisition,
   * the can wait for the semaphore, otherwise the task will not block.
   * @param context
   * @param memoryEstimate
   */
  def updateMemory(context: TaskContext, memoryMB: Int, waitMetric: GpuMetric): Unit = {
    if (enabled && context != null) {
      getInstance.acquireIfNecessaryMemory(context, 4*memoryMB, waitMetric)
    }
  }

  def updateMemory(context: TaskContext, table: Table, waitMetric: GpuMetric): Unit = {
    if (enabled && context != null) {
      val memoryMB = math.ceil(table.getDeviceMemorySize.toDouble/1024/1024).toInt
      getInstance.acquireIfNecessaryMemory(context, 4*memoryMB, waitMetric)
      // if we couldn't acquire (say if we need more memory than what we had before)...
      // we have the table... make it spillable, release the semaphore, block
    }
  }

  /**
   * Tasks must call this when they are finished using the GPU.
   */
  def releaseIfNecessary(context: TaskContext): Unit = {
    if (enabled && context != null) {
      if (getInstance.releaseIfNecessary(context)) {
        logInfo(s"Semaphore RELEASED was at: ${activeTasks.getAndDecrement()}")
      }
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
  private val maxMemoryMB = (GpuDeviceManager.poolAllocation / 1024 / 1024).toInt
  private val mbPerTask = (maxMemoryMB / tasksPerGpu)
  private val semaphore = new Semaphore(maxMemoryMB)
  // Map to track which tasks have acquired the semaphore.
  case class TaskInfo(refs: MutableInt, acquiredMB: MutableInt, var range: NvtxUniqueRange)
  private val activeTasks = new ConcurrentHashMap[Long, TaskInfo]

  logInfo(s"Semaphore initialized with max=${maxMemoryMB}MB, mbPerTask=${mbPerTask}MB")

  def acquireIfNecessary(context: TaskContext, waitMetric: GpuMetric): Boolean = {
    var acquired =  false
    withResource(new NvtxWithMetrics("Acquire GPU", NvtxColor.RED, waitMetric)) { _ =>
      val taskAttemptId = context.taskAttemptId()
      val taskInfo = activeTasks.get(taskAttemptId)
      if (taskInfo == null || taskInfo.refs.getValue == 0) {
        logInfo(s"Task $taskAttemptId acquiring GPU: mbPerTask: ${mbPerTask}")
        semaphore.acquire(mbPerTask)
        logInfo(s"Acquired. Semaphore at ${semaphore.availablePermits()} MB")
        if (taskInfo != null) {
          taskInfo.refs.increment()
          taskInfo.acquiredMB.add(mbPerTask)
          if (taskInfo.range == null) {
            taskInfo.range = new NvtxUniqueRange(s"Task $taskAttemptId", NvtxColor.RED)
          }
        } else {
          // first time this task has been seen
          activeTasks.put(taskAttemptId,
            TaskInfo(
              new MutableInt(1), 
              new MutableInt(mbPerTask), 
              new NvtxUniqueRange(s"Task $taskAttemptId", NvtxColor.BLUE)))
          context.addTaskCompletionListener[Unit](completeTask)
        }
        acquired = true
        GpuDeviceManager.initializeFromTask()
      }
    }
    acquired
  }

  def acquireIfNecessaryMemory(
      context: TaskContext, reqMemoryMB: Int, waitMetric: GpuMetric): Unit = {
    val memoryMB = math.max(math.min(reqMemoryMB, mbPerTask), 0)
    if (memoryMB < reqMemoryMB) {
      logWarning(s"Acquiring less memory ($memoryMB instead of ${reqMemoryMB} MB)" +
        s" due to to small a pool")
    }
    withResource(new NvtxWithMetrics("Acquire Mem GPU", NvtxColor.RED, waitMetric)) { _ =>
      val taskAttemptId = context.taskAttemptId()
      val taskInfo = activeTasks.get(taskAttemptId)
      if (taskInfo.acquiredMB.getValue > memoryMB) {
        // ok, we are going to release some
        logInfo(s"Task $taskAttemptId RELEASING from ${taskInfo.acquiredMB.getValue} to $memoryMB")
        semaphore.release(taskInfo.acquiredMB.getValue - memoryMB)
      } else {
        // now we need to acquire, we could lock
        logInfo(s"Task $taskAttemptId ACQUIRING from ${taskInfo.acquiredMB.getValue} to $memoryMB")
        taskInfo.range.close()
        semaphore.acquire(memoryMB - taskInfo.acquiredMB.getValue)
        taskInfo.range = new NvtxUniqueRange(s"Task $taskAttemptId", NvtxColor.ORANGE)
        logInfo(s"Task $taskAttemptId ACQUIRED! $memoryMB")
      }
      taskInfo.acquiredMB.setValue(memoryMB)
      GpuDeviceManager.initializeFromTask()
    }
  }

  def releaseIfNecessary(context: TaskContext): Boolean = {
    val nvtxRange = new NvtxRange("Release GPU", NvtxColor.RED)
    var released=  false
    try {
      val taskAttemptId = context.taskAttemptId()
      val refs = activeTasks.get(taskAttemptId)
      if (refs != null && refs.refs.getValue > 0) {
        if (refs.refs.decrementAndGet() == 0) {
          semaphore.release(refs.acquiredMB.getValue)
          logInfo(s"Taks $taskAttemptId released ${refs.acquiredMB.getValue}. Semaphore at ${semaphore.availablePermits()} MB")
          refs.acquiredMB.setValue(0)
          refs.range.close()
          refs.range = null
          released = true
        }
      }
    } finally {
      nvtxRange.close()
    }
    released
  }

  def completeTask(context: TaskContext): Unit = {
    val taskAttemptId = context.taskAttemptId()
    val refs = activeTasks.remove(taskAttemptId)
    if (refs == null) {
      throw new IllegalStateException(s"Completion of unknown task $taskAttemptId")
    }
    if (refs.refs.getValue > 0) {
      logDebug(s"Task $taskAttemptId releasing GPU")
      semaphore.release(refs.acquiredMB.getValue)
      refs.range.close()
      refs.range = null
    }
  }

  def shutdown(): Unit = {
    if (!activeTasks.isEmpty) {
      logDebug(s"shutting down with ${activeTasks.size} tasks still registered")
    }
  }
}
