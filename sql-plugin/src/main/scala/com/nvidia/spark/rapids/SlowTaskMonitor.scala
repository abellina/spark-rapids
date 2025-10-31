/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

import java.util.concurrent.{ConcurrentHashMap, Executors, ScheduledExecutorService, TimeUnit}

import scala.collection.JavaConverters._

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging

/**
 * Monitors running tasks and detects when they appear to be slow (running longer than
 * a configured timeout). When a slow task is detected, periodically collects and logs
 * stack traces to help diagnose what the task is doing.
 */
object SlowTaskMonitor extends Logging {
  
  private[rapids] case class TaskInfo(
    taskAttemptId: Long,
    stageId: Int,
    partitionId: Int,
    thread: Thread,
    startTime: Long,
    var lastCheckTime: Long = 0L,
    var slowReported: Boolean = false
  )

  private val activeTasks = new ConcurrentHashMap[Long, TaskInfo]()
  private var monitorThread: Option[ScheduledExecutorService] = None
  private var enabled = false
  private var timeoutMillis = 0L
  private var checkIntervalMillis = 0L
  private var stackDepth = 10
  
  // Allow tests to inject a custom time source
  private[rapids] var currentTimeMillis: () => Long = () => System.currentTimeMillis()

  /**
   * Initialize the slow task monitor with the given configuration.
   */
  def initialize(conf: RapidsConf): Unit = synchronized {
    val timeoutSeconds = conf.get(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS)
    
    if (timeoutSeconds > 0) {
      enabled = true
      timeoutMillis = timeoutSeconds * 1000L
      checkIntervalMillis = conf.get(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS) * 1000L
      stackDepth = conf.get(RapidsConf.SLOW_TASK_STACK_DEPTH)
      
      logInfo(s"Slow task detection enabled: timeout=${timeoutMillis}ms, " +
        s"checkInterval=${checkIntervalMillis}ms, stackDepth=${stackDepth}")
      
      // Start the monitoring thread
      val executor = Executors.newSingleThreadScheduledExecutor(new java.util.concurrent.ThreadFactory {
        override def newThread(r: Runnable): Thread = {
          val t = new Thread(r, "slow-task-monitor")
          t.setDaemon(true)
          t
        }
      })
      
      executor.scheduleAtFixedRate(
        new Runnable {
          override def run(): Unit = checkForSlowTasks()
        },
        checkIntervalMillis,
        checkIntervalMillis,
        TimeUnit.MILLISECONDS
      )
      
      monitorThread = Some(executor)
    } else {
      logDebug("Slow task detection is disabled (timeout set to 0)")
    }
  }

  /**
   * Register a task as active when it starts.
   */
  def onTaskStart(): Unit = {
    if (enabled) {
      val tc = TaskContext.get()
      if (tc != null) {
        val taskInfo = TaskInfo(
          taskAttemptId = tc.taskAttemptId(),
          stageId = tc.stageId(),
          partitionId = tc.partitionId(),
          thread = Thread.currentThread(),
          startTime = currentTimeMillis()
        )
        activeTasks.put(tc.taskAttemptId(), taskInfo)
        logDebug(s"Registered task ${tc.taskAttemptId()} for slow task monitoring")
      }
    }
  }
  
  /**
   * For testing: directly register a task with a specific start time.
   */
  private[rapids] def registerTask(taskAttemptId: Long, stageId: Int, partitionId: Int,
      thread: Thread, startTime: Long): Unit = {
    val taskInfo = TaskInfo(taskAttemptId, stageId, partitionId, thread, startTime)
    activeTasks.put(taskAttemptId, taskInfo)
  }
  
  /**
   * For testing: directly unregister a task.
   */
  private[rapids] def unregisterTask(taskAttemptId: Long): Unit = {
    activeTasks.remove(taskAttemptId)
  }
  
  /**
   * For testing: get the count of active tasks.
   */
  private[rapids] def getActiveTaskCount: Int = activeTasks.size()
  
  /**
   * For testing: check if a task has been marked as slow.
   */
  private[rapids] def isTaskMarkedAsSlow(taskAttemptId: Long): Boolean = {
    Option(activeTasks.get(taskAttemptId)).exists(_.slowReported)
  }

  /**
   * Unregister a task when it completes (successfully or with failure).
   */
  def onTaskEnd(): Unit = {
    if (enabled) {
      val tc = TaskContext.get()
      if (tc != null) {
        val removed = activeTasks.remove(tc.taskAttemptId())
        if (removed != null) {
          logDebug(s"Unregistered task ${tc.taskAttemptId()} from slow task monitoring")
        }
      }
    }
  }

  /**
   * Check all active tasks for slow tasks and log stack traces.
   * Made package-private for testing.
   */
  private[rapids] def checkForSlowTasks(): Unit = {
    try {
      val currentTime = currentTimeMillis()
      
      activeTasks.values().asScala.foreach { taskInfo =>
        val runningTime = currentTime - taskInfo.startTime
        
        if (runningTime >= timeoutMillis) {
          // Task has been running longer than the timeout
          val timeSinceLastCheck = currentTime - taskInfo.lastCheckTime
          
          if (!taskInfo.slowReported) {
            // First time detecting this task as slow
            logWarning(s"Detected slow task: stage=${taskInfo.stageId}, " +
              s"partition=${taskInfo.partitionId}, taskAttemptId=${taskInfo.taskAttemptId}, " +
              s"runningTime=${runningTime}ms (threshold=${timeoutMillis}ms)")
            taskInfo.slowReported = true
            taskInfo.lastCheckTime = currentTime
            logStackTrace(taskInfo)
          } else if (timeSinceLastCheck >= checkIntervalMillis) {
            // Periodic check for already-slow task
            logWarning(s"Task still slow: stage=${taskInfo.stageId}, " +
              s"partition=${taskInfo.partitionId}, taskAttemptId=${taskInfo.taskAttemptId}, " +
              s"runningTime=${runningTime}ms")
            taskInfo.lastCheckTime = currentTime
            logStackTrace(taskInfo)
          }
        }
      }
    } catch {
      case e: Exception =>
        logError("Error while checking for slow tasks", e)
    }
  }

  /**
   * Log a summarized stack trace for a slow task, focusing on relevant frames.
   */
  private def logStackTrace(taskInfo: TaskInfo): Unit = {
    try {
      val stackTrace = taskInfo.thread.getStackTrace
      val summary = summarizeStackTrace(stackTrace)
      
      logWarning(s"Stack trace summary for slow task ${taskInfo.taskAttemptId}:\n$summary")
    } catch {
      case e: Exception =>
        logError(s"Failed to collect stack trace for task ${taskInfo.taskAttemptId}", e)
    }
  }

  /**
   * Summarize a stack trace by showing the top frames.
   * Returns a concise summary showing where the task is slow.
   */
  private def summarizeStackTrace(stackTrace: Array[StackTraceElement]): String = {
    val sb = new StringBuilder
    
    sb.append("  Top of stack:\n")
    stackTrace.take(stackDepth).foreach { frame =>
      sb.append(s"    at ${frame.getClassName}.${frame.getMethodName}")
      if (frame.isNativeMethod) {
        sb.append(" (native)")
      } else if (frame.getFileName != null) {
        sb.append(s" (${frame.getFileName}:${frame.getLineNumber})")
      }
      sb.append("\n")
    }
    if (stackTrace.length > stackDepth) {
      sb.append(s"    ... ${stackTrace.length - stackDepth} more frames\n")
    }
    
    sb.append(s"  Total stack depth: ${stackTrace.length} frames")
    sb.toString()
  }

  /**
   * Shutdown the monitor and clean up resources.
   */
  def shutdown(): Unit = synchronized {
    monitorThread.foreach { executor =>
      logInfo("Shutting down slow task monitor")
      executor.shutdown()
      try {
        if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
          executor.shutdownNow()
        }
      } catch {
        case _: InterruptedException =>
          executor.shutdownNow()
      }
    }
    monitorThread = None
    activeTasks.clear()
    enabled = false
    // Reset time source to default
    currentTimeMillis = () => System.currentTimeMillis()
  }
}

