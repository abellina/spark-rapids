/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.status

import java.util.concurrent.ConcurrentHashMap

import com.nvidia.spark.rapids.{MetricDefinition, MetricUpdates, SparkRapidsBuildInfoEvent}
import org.apache.spark.SparkConf
import org.apache.spark.scheduler._
import org.apache.spark.ui.SparkUI

import scala.collection.mutable

/**
 * Companion object to store listener references keyed by application ID.
 *
 * This is necessary because Spark's ServiceLoader may create different plugin
 * instances for createListeners() and setupUI() calls. We use the application ID
 * as the key to associate listeners with their corresponding UI.
 *
 * The flow is:
 * 1. createListeners() creates a CustomEventsListener
 * 2. During event log replay, onApplicationStart() is called with the app ID
 * 3. The listener registers itself with the app ID
 * 4. setupUI() retrieves the listener using SparkUI.appId
 */
object CustomEventsPlugin {
  // Map from appId -> Listener for cross-instance communication
  private val listenersByAppId = new ConcurrentHashMap[String, CustomEventsListener]()

  private[status] def registerListener(appId: String, listener: CustomEventsListener): Unit = {
    listenersByAppId.put(appId, listener)
  }

  private[status] def getListener(appId: String): Option[CustomEventsListener] = {
    if (appId == null) {
      None
    } else {
      Option(listenersByAppId.get(appId))
    }
  }

  /**
   * Remove a listener when the app is no longer needed.
   * Called during cleanup to prevent memory leaks for very long-running history servers.
   */
  private[status] def removeListener(appId: String): Unit = {
    if (appId != null) {
      listenersByAppId.remove(appId)
    }
  }
}

/**
 * Custom tab plugin for Spark History Server.
 * This plugin analyzes custom event log events and displays them in a custom tab.
 *
 * Each application gets its own listener instance with isolated event storage.
 * The listener is associated with the app via the shared KVStore reference.
 */
class CustomEventsPlugin extends AppHistoryServerPlugin {

  override def createListeners(
      conf: SparkConf,
      store: ElementTrackingStore): Seq[SparkListener] = {
    // Create a new listener for this application.
    // The listener will register itself with the app ID when it receives
    // the ApplicationStart event during log replay.
    Seq(new CustomEventsListener())
  }

  override def setupUI(ui: SparkUI): Unit = {
    // Try to get the app ID from multiple sources
    val appId: Option[String] = {
      // First try ui.appId
      Option(ui.appId).filter(_.nonEmpty).orElse {
        // Fall back to getting it from the store's application info
        try {
          Option(ui.store.applicationInfo().id).filter(_.nonEmpty)
        } catch {
          case _: Exception => None
        }
      }
    }

    // Retrieve the listener using the app ID (don't remove - may be called multiple times)
    val eventsOpt = appId.flatMap(CustomEventsPlugin.getListener)

    val events = eventsOpt match {
      case Some(listener) => listener.getEvents
      case None =>
        // This can happen if no ApplicationStart event was in the log
        List.empty[CustomEventData]
    }

    // Add custom tab to the UI with this app's events
    val customTab = new CustomEventsTab(ui, events)
    ui.attachTab(customTab)
  }

  override def displayOrder: Int = 1000 // Display order in the UI
}

/**
 * Data class to hold custom event information.
 * Immutable case class for thread-safe sharing.
 */
case class CustomEventData(
  timestamp: Long,
  eventType: String,
  eventData: Map[String, String],
  applicationId: String,
  applicationAttemptId: Option[String]
)

/**
 * Custom listener to capture and process events for a single application.
 *
 * This listener maintains its own private event store, ensuring isolation
 * between different applications in the History Server.
 *
 * Thread-safety: Uses synchronized access to the events buffer since
 * event callbacks may come from different threads during event log replay.
 */
class CustomEventsListener extends SparkListener {

  // Private per-listener event store - not shared with other listeners
  private val events = mutable.ListBuffer[CustomEventData]()

  // Current application context, set on ApplicationStart
  @volatile private var currentAppId: String = "unknown"
  @volatile private var currentAppAttemptId: Option[String] = None

  /**
   * Returns an immutable snapshot of all collected events.
   * Called by the plugin after event log replay is complete.
   */
  def getEvents: List[CustomEventData] = synchronized {
    events.toList
  }

  private def addEvent(
      timestamp: Long,
      eventType: String,
      eventData: Map[String, String]): Unit = synchronized {
    events += CustomEventData(
      timestamp = timestamp,
      eventType = eventType,
      eventData = eventData,
      applicationId = currentAppId,
      applicationAttemptId = currentAppAttemptId
    )
  }

  override def onApplicationStart(applicationStart: SparkListenerApplicationStart): Unit = {
    // Set the application context for subsequent events
    currentAppId = applicationStart.appId.getOrElse("unknown")
    currentAppAttemptId = applicationStart.appAttemptId

    // Register this listener with the app ID so setupUI can find it
    CustomEventsPlugin.registerListener(currentAppId, this)

    val eventData = Map(
      "appName" -> applicationStart.appName,
      "time" -> applicationStart.time.toString,
      "user" -> applicationStart.sparkUser
    )

    addEvent(applicationStart.time, "ApplicationStart", eventData)
  }

  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
    val eventData = Map(
      "time" -> applicationEnd.time.toString
    )

    addEvent(applicationEnd.time, "ApplicationEnd", eventData)
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    val eventData = Map(
      "jobId" -> jobStart.jobId.toString,
      "time" -> jobStart.time.toString,
      "stageIds" -> jobStart.stageIds.mkString(","),
      "stageCount" -> jobStart.stageIds.size.toString
    )

    addEvent(jobStart.time, "JobStart", eventData)
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
    val eventData = Map(
      "jobId" -> jobEnd.jobId.toString,
      "time" -> jobEnd.time.toString,
      "result" -> jobEnd.jobResult.toString
    )

    addEvent(jobEnd.time, "JobEnd", eventData)
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    val stageInfo = stageCompleted.stageInfo
    val taskMetrics = Option(stageInfo.taskMetrics)
    val startTime = stageInfo.submissionTime.getOrElse(
      stageInfo.completionTime.getOrElse(System.currentTimeMillis()))
    val endTime = stageInfo.completionTime.getOrElse(System.currentTimeMillis())

    val eventData = Map(
      "stageId" -> stageInfo.stageId.toString,
      "stageAttemptId" -> stageInfo.attemptNumber.toString,
      "stageName" -> stageInfo.name,
      "numTasks" -> stageInfo.numTasks.toString,
      "stageStartTime" -> startTime.toString,
      "stageEndTime" -> endTime.toString,
      "executorRunTime" -> taskMetrics.map(_.executorRunTime.toString).getOrElse("0"),
      "executorCpuTime" -> taskMetrics.map(_.executorCpuTime.toString).getOrElse("0"),
      "inputBytes" -> taskMetrics.map(_.inputMetrics.bytesRead.toString).getOrElse("0"),
      "outputBytes" -> taskMetrics.map(_.outputMetrics.bytesWritten.toString).getOrElse("0"),
      "shuffleReadBytes" -> taskMetrics.map(_.shuffleReadMetrics.totalBytesRead.toString)
        .getOrElse("0"),
      "shuffleWriteBytes" -> taskMetrics.map(_.shuffleWriteMetrics.bytesWritten.toString)
        .getOrElse("0")
    )

    addEvent(endTime, "StageCompleted", eventData)
  }

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    val taskInfo = taskEnd.taskInfo
    val taskMetrics = taskEnd.taskMetrics

    if (taskMetrics != null) {
      val eventData = Map(
        "taskId" -> taskInfo.taskId.toString,
        "stageId" -> taskEnd.stageId.toString,
        "executorId" -> taskInfo.executorId,
        "host" -> taskInfo.host,
        "duration" -> taskInfo.duration.toString,
        "executorRunTime" -> taskMetrics.executorRunTime.toString,
        "executorCpuTime" -> taskMetrics.executorCpuTime.toString,
        "resultSize" -> taskMetrics.resultSize.toString,
        "jvmGCTime" -> taskMetrics.jvmGCTime.toString,
        "memoryBytesSpilled" -> taskMetrics.memoryBytesSpilled.toString,
        "diskBytesSpilled" -> taskMetrics.diskBytesSpilled.toString
      )

      addEvent(taskInfo.finishTime, "TaskEnd", eventData)
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case bi: SparkRapidsBuildInfoEvent =>
        // Flatten build info maps into simple key/value strings
        def flatten(prefix: String, m: Map[String, String]): Map[String, String] =
          m.map { case (k, v) => s"$prefix.$k" -> v }

        val baseEventData =
          flatten("sparkRapidsBuildInfo", bi.sparkRapidsBuildInfo) ++
            flatten("sparkRapidsJniBuildInfo", bi.sparkRapidsJniBuildInfo) ++
            flatten("cudfBuildInfo", bi.cudfBuildInfo) ++
            flatten("sparkRapidsPrivateBuildInfo", bi.sparkRapidsPrivateBuildInfo)

        val withDisk = bi.monitoredDiskDevice match {
          case Some(dev) =>
            baseEventData + ("sparkRapidsBuildInfo.monitoredDiskDevice" -> dev)
          case None =>
            baseEventData
        }

        val eventData = bi.executorId match {
          case Some(execId) =>
            withDisk + ("sparkRapidsBuildInfo.executorId" -> execId)
          case None =>
            withDisk
        }

        addEvent(System.currentTimeMillis(), "SparkRapidsBuildInfo", eventData)

      case md: MetricDefinition =>
        val eventData = Map(
          "executorId" -> md.executorId,
          "metricNames" -> md.metricNames.mkString(",")
        )

        addEvent(System.currentTimeMillis(), "MetricDefinition", eventData)

      case mu: MetricUpdates =>
        val eventData = Map(
          "executorId" -> mu.executorId,
          "encodedMetricsB64" -> mu.encodedMetricsBase64
        )

        addEvent(System.currentTimeMillis(), "MetricUpdates", eventData)

      case _ =>
        // Ignore unknown events - don't pollute the event store with noise
        // Previously we captured all events as "CustomEvent" but this creates
        // unnecessary overhead for the common case
    }
  }
}
