package org.apache.spark.status

import com.nvidia.spark.rapids.{MetricDefinition, MetricUpdates}
import org.apache.spark.SparkConf
import org.apache.spark.scheduler._
import org.apache.spark.ui.SparkUI

import scala.collection.mutable

object CustomEventsPlugin {
  val customEvents = mutable.ListBuffer[CustomEventData]()
}
/**
 * Custom tab plugin for Spark History Server.
 * This plugin analyzes custom event log events and displays them in a custom tab.
 */
class CustomEventsPlugin extends AppHistoryServerPlugin {
  override def setupUI(ui: SparkUI): Unit = {
    // Add custom tab to the UI
    val customTab = new CustomEventsTab(ui, CustomEventsPlugin.customEvents.toList)
    ui.attachTab(customTab)
  }

  override def createListeners(
      conf: SparkConf,
      store: ElementTrackingStore): Seq[SparkListener] = {
    
    // Create and return a custom listener
    val listener = new CustomEventsListener(CustomEventsPlugin.customEvents)
    Seq(listener)
  }

  override def displayOrder: Int = 1000 // Display order in the UI
}

/**
 * Data class to hold custom event information
 */
case class CustomEventData(
  timestamp: Long,
  eventType: String,
  eventData: Map[String, String],
  applicationId: String,
  applicationAttemptId: Option[String]
)

/**
 * Custom listener to capture and process events
 */
class CustomEventsListener(customEvents: mutable.ListBuffer[CustomEventData]) 
    extends SparkListener {

  override def onApplicationStart(applicationStart: SparkListenerApplicationStart): Unit = {
    // Capture application start as a custom event
    val eventData = Map(
      "appName" -> applicationStart.appName,
      "time" -> applicationStart.time.toString,
      "user" -> applicationStart.sparkUser
    )
    
    customEvents += CustomEventData(
      timestamp = applicationStart.time,
      eventType = "ApplicationStart",
      eventData = eventData,
      applicationId = applicationStart.appId.getOrElse("unknown"),
      applicationAttemptId = applicationStart.appAttemptId
    )
  }

  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
    // Capture application end as a custom event
    val eventData = Map(
      "time" -> applicationEnd.time.toString
    )
    
    customEvents += CustomEventData(
      timestamp = applicationEnd.time,
      eventType = "ApplicationEnd",
      eventData = eventData,
      applicationId = "current",
      applicationAttemptId = None
    )
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    // Capture job start events
    val eventData = Map(
      "jobId" -> jobStart.jobId.toString,
      "time" -> jobStart.time.toString,
      "stageIds" -> jobStart.stageIds.mkString(","),
      "stageCount" -> jobStart.stageIds.size.toString
    )
    
    customEvents += CustomEventData(
      timestamp = jobStart.time,
      eventType = "JobStart",
      eventData = eventData,
      applicationId = "current",
      applicationAttemptId = None
    )
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
    // Capture job end events
    val eventData = Map(
      "jobId" -> jobEnd.jobId.toString,
      "time" -> jobEnd.time.toString,
      "result" -> jobEnd.jobResult.toString
    )
    
    customEvents += CustomEventData(
      timestamp = jobEnd.time,
      eventType = "JobEnd",
      eventData = eventData,
      applicationId = "current",
      applicationAttemptId = None
    )
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    // Capture stage completion with custom metrics
    val stageInfo = stageCompleted.stageInfo
    val taskMetrics = stageInfo.taskMetrics
    
    val eventData = Map(
      "stageId" -> stageInfo.stageId.toString,
      "stageName" -> stageInfo.name,
      "numTasks" -> stageInfo.numTasks.toString,
      "executorRunTime" -> taskMetrics.executorRunTime.toString,
      "executorCpuTime" -> taskMetrics.executorCpuTime.toString,
      "inputBytes" -> taskMetrics.inputMetrics.bytesRead.toString,
      "outputBytes" -> taskMetrics.outputMetrics.bytesWritten.toString,
      "shuffleReadBytes" -> taskMetrics.shuffleReadMetrics.totalBytesRead.toString,
      "shuffleWriteBytes" -> taskMetrics.shuffleWriteMetrics.bytesWritten.toString
    )
    
    customEvents += CustomEventData(
      timestamp = stageInfo.completionTime.getOrElse(System.currentTimeMillis()),
      eventType = "StageCompleted",
      eventData = eventData,
      applicationId = "current",
      applicationAttemptId = None
    )
  }

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    // Capture task-level events for detailed analysis
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
      
      customEvents += CustomEventData(
        timestamp = taskInfo.finishTime,
        eventType = "TaskEnd",
        eventData = eventData,
        applicationId = "current",
        applicationAttemptId = None
      )
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    // Capture any custom events that might be logged
    event match {
      case md: MetricDefinition =>
        val eventData = Map(
          "executorId" -> md.executorId,
          "metricNames" -> md.metricNames.mkString(",")
        )

        customEvents += CustomEventData(
          timestamp = System.currentTimeMillis(),
          eventType = "MetricDefinition",
          eventData = eventData,
          applicationId = "current",
          applicationAttemptId = None
        )

      case mu: MetricUpdates =>
        val eventData = Map(
          "executorId" -> mu.executorId,
          "encodedMetricsHex" -> mu.encodedMetricsHex
        )

        customEvents += CustomEventData(
          timestamp = System.currentTimeMillis(),
          eventType = "MetricUpdates",
          eventData = eventData,
          applicationId = "current",
          applicationAttemptId = None
        )

      case _ =>
        val eventData = Map(
          "eventClass" -> event.getClass.getSimpleName,
          "timestamp" -> System.currentTimeMillis().toString
        )

        customEvents += CustomEventData(
          timestamp = System.currentTimeMillis(),
          eventType = "CustomEvent",
          eventData = eventData,
          applicationId = "current",
          applicationAttemptId = None
        )
    }
  }
}

