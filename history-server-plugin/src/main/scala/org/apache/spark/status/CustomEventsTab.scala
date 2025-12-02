package org.apache.spark.status

import java.nio.{ByteBuffer, ByteOrder}

import org.apache.spark.ui.{SparkUI, SparkUITab, UIUtils, WebUIPage}

import javax.servlet.http.HttpServletRequest
import scala.collection.mutable
import scala.xml.Node

/**
 * Custom tab in the Spark History Server UI
 */
class CustomEventsTab(parent: SparkUI, customEvents: List[CustomEventData]) 
    extends SparkUITab(parent, "customevents") {

  attachPage(new CustomEventsPage(this, customEvents))
  attachPage(new CustomEventsApiPage(this, customEvents))
}

/**
 * Main page for the custom events tab
 */
class CustomEventsPage(parent: CustomEventsTab, customEvents: List[CustomEventData]) 
    extends WebUIPage("") {

  override def render(request: HttpServletRequest): Seq[Node] = {
    val content =
      <div class="row-fluid">
        <div class="span12">
          <h4>Custom Events Analysis</h4>
          <p>This tab shows custom event log events captured during application execution.</p>

          <div id="custom-events-summary">
            <h5>Event Summary</h5>
            <ul>
              <li>Total Events: {customEvents.size}</li>
              <li>Event Types: {customEvents.map(_.eventType).distinct.size}</li>
              <li>Application Start Events: {customEvents.count(_.eventType == "ApplicationStart")}</li>
              <li>Job Events: {customEvents.count(e => e.eventType == "JobStart" || e.eventType == "JobEnd")}</li>
              <li>Stage Events: {customEvents.count(_.eventType == "StageCompleted")}</li>
              <li>Task Events: {customEvents.count(_.eventType == "TaskEnd")}</li>
            </ul>
          </div>

          <div id="custom-events-table">
            <h5>Event Details</h5>
            {renderEventsTable()}
          </div>

          <div id="custom-events-charts">
            <h5>Event Timeline</h5>
            <div id="timeline-chart" style="width: 100%; height: 300px;">
              <p>Timeline visualization would be rendered here with JavaScript</p>
            </div>
          </div>

          <div id="metric-charts">
            <h5>RAPIDS Metrics</h5>
            <div id="jvm-offheap-chart" style="width: 100%; height: 300px; margin-top: 20px;"></div>
            <div id="gpu-chart" style="width: 100%; height: 300px; margin-top: 20px;"></div>
          </div>
        </div>
      </div>

    val highchartsScript =
      <script src="https://code.highcharts.com/highcharts.js"></script>

    val scriptContent =
      <script type="text/javascript">
        {scala.xml.Unparsed("""
          $(document).ready(function() {
            // Initialize DataTable for better table interaction
            if ($.fn.DataTable) {
              $('#events-table').DataTable({
                "order": [[ 0, "desc" ]],
                "pageLength": 25,
                "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]]
              });
            }

            // Fetch and render timeline data
            fetchTimelineData();

            // Fetch and render RAPIDS metric data
            fetchMetricData();
          });

          function fetchTimelineData() {
            $.getJSON('/history/' + getAppId() + '/customevents/api/timeline', function(data) {
              renderTimeline(data);
            });
          }

          function fetchMetricData() {
            $.getJSON('/history/' + getAppId() + '/customevents/api/metrics', function(data) {
              renderMetricCharts(data);
            });
          }

          function getAppId() {
            // Extract app ID from URL
            var path = window.location.pathname;
            var match = path.match(/\/history\/([^\/]+)/);
            return match ? match[1] : '';
          }

          function renderTimeline(data) {
            // Simple timeline rendering
            var chartDiv = $('#timeline-chart');
            if (data && data.events && data.events.length > 0) {
              var html = '<table class="table table-bordered table-condensed">';
              html += '<thead><tr><th>Time</th><th>Event Type</th><th>Count</th></tr></thead><tbody>';
              
              var eventCounts = {};
              data.events.forEach(function(event) {
                var key = event.eventType;
                eventCounts[key] = (eventCounts[key] || 0) + 1;
              });

              for (var eventType in eventCounts) {
                html += '<tr><td>-</td><td>' + eventType + '</td><td>' + eventCounts[eventType] + '</td></tr>';
              }
              
              html += '</tbody></table>';
              chartDiv.html(html);
            } else {
              chartDiv.html('<p>No timeline data available</p>');
            }
          }

          function renderMetricCharts(data) {
            if (!window.Highcharts || !data || !data.series) {
              return;
            }

            function getSeries(name) {
              return (data.series && data.series[name]) ? data.series[name] : [];
            }

            Highcharts.chart('jvm-offheap-chart', {
              title: { text: 'JVM / Off-heap Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              series: [
                { name: 'jvmTotal', data: getSeries('jvmTotal') },
                { name: 'jvmFree', data: getSeries('jvmFree') },
                { name: 'offHeapPinned', data: getSeries('offHeapPinned') },
                { name: 'offHeapPageable', data: getSeries('offHeapPageable') }
              ]
            });

            Highcharts.chart('gpu-chart', {
              title: { text: 'GPU Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              series: [
                { name: 'gpuMemUsed', data: getSeries('gpuMemUsed') }
              ]
            });
          }
        """)}
      </script>

    UIUtils.headerSparkPage(request, "Custom Events",
      content ++ highchartsScript ++ scriptContent, parent)
  }

  private def renderEventsTable(): Node = {
    <table class="table table-bordered table-striped table-condensed" id="events-table">
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Event Type</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>
        {customEvents.sortBy(-_.timestamp).take(100).map(renderEventRow)}
      </tbody>
    </table>
  }

  private def renderEventRow(event: CustomEventData): Node = {
    val formattedTime = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
      .format(new java.util.Date(event.timestamp))
    
    val details = event.eventData.map { case (k, v) => s"$k: $v" }.mkString(", ")
    
    <tr>
      <td>{formattedTime}</td>
      <td><span class="badge badge-info">{event.eventType}</span></td>
      <td style="font-size: 11px;">{details}</td>
    </tr>
  }
}

/**
 * REST API page for custom events data
 */
class CustomEventsApiPage(parent: CustomEventsTab, customEvents: List[CustomEventData]) 
    extends WebUIPage("api") {

  override def render(request: HttpServletRequest): Seq[Node] = {
    // Return empty sequence - we'll use renderJson instead
    Seq.empty
  }

  override def renderJson(request: HttpServletRequest): org.json4s.JsonAST.JValue = {
    //import org.json4s.JsonDSL._
    import org.json4s.jackson.JsonMethods._
    
    val endpoint = Option(request.getPathInfo).getOrElse("")
    
    val jsonString = endpoint match {
      case "/timeline" => generateTimelineJson()
      case "/summary" => generateSummaryJson()
      case "/events" => generateEventsJson(request)
      case "/metrics" => generateMetricsJson()
      case _ => generateApiIndexJson()
    }
    
    parse(jsonString)
  }

/*
  private def renderTimelineJson(): Seq[Node] = {
    scala.xml.Unparsed(generateTimelineJson())
  }

  private def renderSummaryJson(): Seq[Node] = {
    scala.xml.Unparsed(generateSummaryJson())
  }

  private def renderEventsJson(request: HttpServletRequest): Seq[Node] = {
    scala.xml.Unparsed(generateEventsJson(request))
  }

  private def renderApiIndexJson(): Seq[Node] = {
    scala.xml.Unparsed(generateApiIndexJson())
  }
  */

  private def generateTimelineJson(): String = {
    val eventsJson = customEvents.map { event =>
      s"""{
        "timestamp": ${event.timestamp},
        "eventType": "${event.eventType}",
        "applicationId": "${event.applicationId}"
      }"""
    }.mkString(",")
    
    s"""{"events": [$eventsJson]}"""
  }

  private def generateSummaryJson(): String = {
    val totalEvents = customEvents.size
    val eventTypes = customEvents.map(_.eventType).distinct
    val eventTypeCounts = eventTypes.map { eventType =>
      val count = customEvents.count(_.eventType == eventType)
      s""""$eventType": $count"""
    }.mkString(",")
    
    s"""{
      "totalEvents": $totalEvents,
      "eventTypeCount": ${eventTypes.size},
      "eventTypeCounts": {$eventTypeCounts}
    }"""
  }

  private def generateEventsJson(request: HttpServletRequest): String = {
    val eventType = Option(request.getParameter("type"))
    val limit = Option(request.getParameter("limit")).map(_.toInt).getOrElse(100)
    
    val filteredEvents = eventType match {
      case Some(et) => customEvents.filter(_.eventType == et)
      case None => customEvents
    }
    
    val eventsJson = filteredEvents.sortBy(-_.timestamp).take(limit).map { event =>
      val eventDataJson = event.eventData.map { case (k, v) => 
        s""""$k": "$v""""
      }.mkString(",")
      
      s"""{
        "timestamp": ${event.timestamp},
        "eventType": "${event.eventType}",
        "applicationId": "${event.applicationId}",
        "eventData": {$eventDataJson}
      }"""
    }.mkString(",")
    
    s"""{"events": [$eventsJson], "count": ${filteredEvents.size}}"""
  }

  private def hexToBytes(hex: String): Array[Byte] = {
    val cleanHex = hex.trim
    val len = cleanHex.length
    if (len % 2 != 0) {
      return Array.emptyByteArray
    }
    val data = new Array[Byte](len / 2)
    var i = 0
    while (i < len) {
      val byteStr = cleanHex.substring(i, i + 2)
      data(i / 2) = Integer.parseInt(byteStr, 16).toByte
      i += 2
    }
    data
  }

  private def generateMetricsJson(): String = {
    // Build a mapping from executorId -> metric names (in positional order)
    val metricDefs: Map[String, Seq[String]] = customEvents
      .filter(_.eventType == "MetricDefinition")
      .flatMap { e =>
        for {
          execId <- e.eventData.get("executorId")
          namesStr <- e.eventData.get("metricNames")
        } yield execId -> namesStr.split(",").map(_.trim).filter(_.nonEmpty).toSeq
      }.groupBy(_._1).mapValues(_.last._2).toMap

    val series = mutable.Map[String, mutable.ArrayBuffer[(Long, Long)]]()

    // Process each MetricUpdates event: decode hex payload and expand into per-metric series
    customEvents
      .filter(_.eventType == "MetricUpdates")
      .foreach { e =>
        for {
          execId <- e.eventData.get("executorId")
          encoded <- e.eventData.get("encodedMetricsHex")
          metricNames <- metricDefs.get(execId)
        } {
          val bytes = hexToBytes(encoded)
          if (bytes.nonEmpty && metricNames.nonEmpty) {
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
            if (bb.remaining() >= Integer.BYTES) {
              val numUpdates = bb.getInt()
              val numMetricsPerUpdate = metricNames.length
              var u = 0
              while (u < numUpdates &&
                     bb.remaining() >= java.lang.Long.BYTES * (1 + numMetricsPerUpdate)) {
                val ts = bb.getLong()
                var m = 0
                while (m < numMetricsPerUpdate && bb.remaining() >= java.lang.Long.BYTES) {
                  val value = bb.getLong()
                  val name = metricNames(m)
                  val buf = series.getOrElseUpdate(name,
                    mutable.ArrayBuffer[(Long, Long)]())
                  buf += ((ts, value))
                  m += 1
                }
                u += 1
              }
            }
          }
        }
      }

    // Order series and restrict to the metrics we currently know about
    val knownMetrics = Seq("jvmTotal", "jvmFree", "offHeapPinned", "offHeapPageable", "gpuMemUsed")

    val seriesEntries = knownMetrics.map { name =>
      val points = series.getOrElse(name, mutable.ArrayBuffer.empty).sortBy(_._1)
      val ptsJson = points.map { case (ts, v) => s"[$ts,$v]" }.mkString(",")
      s""""$name": [$ptsJson]"""
    }.mkString(",")

    s"""{"series": {$seriesEntries}}"""
  }

  private def generateApiIndexJson(): String = {
    s"""{
      "endpoints": [
        "/api/timeline",
        "/api/summary",
        "/api/events?type=<eventType>&limit=<number>"
      ]
    }"""
  }
}

