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
          <p>This tab shows RAPIDS runtime metrics captured during application execution.</p>

          <div id="metric-charts">
            <h5>RAPIDS Metrics</h5>
            <div id="executor-filters" style="margin-bottom: 10px;"></div>
            <div id="jvm-chart" style="width: 100%; height: 300px; margin-top: 20px;"></div>
            <div id="offheap-chart" style="width: 100%; height: 300px; margin-top: 20px;"></div>
            <div id="sys-mem-chart" style="width: 100%; height: 300px; margin-top: 20px;"></div>
            <div id="cpu-chart" style="width: 100%; height: 250px; margin-top: 20px;"></div>
            <div id="gpu-tasks-chart" style="width: 100%; height: 250px; margin-top: 20px;"></div>
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
            // Fetch and render RAPIDS metric data
            fetchMetricData();
          });

          function fetchMetricData() {
            $.getJSON('/history/' + getAppId() + '/customevents/api/json/metrics', function(data) {
              window._rapidsMetricData = data;
              initExecutorFilters(data);
              renderMetricCharts();
            });
          }

          function getAppId() {
            // Extract app ID from URL
            var path = window.location.pathname;
            var match = path.match(/\/history\/([^\/]+)/);
            return match ? match[1] : '';
          }

          function initExecutorFilters(data) {
            var container = $('#executor-filters');
            container.empty();
            if (!data || !data.executors || data.executors.length === 0) {
              container.append('<span>No executor metric data available</span>');
              return;
            }

            container.append('<span style="margin-right: 8px;">Executors:</span>');

            // Initialize selected executors on first load
            if (!window._selectedExecutors) {
              window._selectedExecutors = new Set(data.executors);
            }

            data.executors.forEach(function(execId) {
              var checkboxId = 'exec-filter-' + execId;
              var checked = window._selectedExecutors.has(execId) ? 'checked' : '';
              container.append(
                '<label style="margin-right: 10px;">' +
                  '<input type="checkbox" id="' + checkboxId + '" data-exec="' + execId + '" ' +
                  checked + ' /> ' + execId +
                '</label>');
            });

            container.find('input[type=checkbox]').change(function() {
              var execId = $(this).data('exec');
              if (this.checked) {
                window._selectedExecutors.add(execId);
              } else {
                window._selectedExecutors.delete(execId);
              }
              renderMetricCharts();
            });
          }

          function renderMetricCharts() {
            var data = window._rapidsMetricData;
            if (!window.Highcharts || !data || !data.seriesByExecutor) {
              return;
            }

            var selected = window._selectedExecutors || new Set();

            function getSeriesForMetric(metricName) {
              var allSeries = [];
              selected.forEach(function(execId) {
                var execSeries = (data.seriesByExecutor[execId] &&
                  data.seriesByExecutor[execId][metricName]) || [];
                if (execSeries.length > 0) {
                  allSeries.push({
                    name: execId + ' ' + metricName,
                    data: execSeries
                  });
                }
              });
              return allSeries;
            }

            // JVM chart: jvmTotal, jvmUsed
            var jvmChart = Highcharts.chart('jvm-chart', {
              title: { text: 'JVM Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              series: []
            });

            if (jvmChart) {
              var jvmMetrics = ['jvmTotal', 'jvmUsed'];
              jvmMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  if (metricName === 'jvmTotal') {
                    // Draw total heap as a red shaded area so jvmUsed
                    // appears visually "within" it.
                    s.type = 'area';
                    s.color = '#ff0000';
                    s.fillOpacity = 0.15;
                    s.lineWidth = 1;
                    s.zIndex = 0;
                  } else if (metricName === 'jvmUsed') {
                    // Emphasize used heap as a solid line above the area.
                    s.zIndex = 1;
                    s.lineWidth = 2;
                  }
                  jvmChart.addSeries(s, false);
                });
              });
              jvmChart.redraw();
            }

            // Off-heap chart: pinned and pageable
            var offheapChart = Highcharts.chart('offheap-chart', {
              title: { text: 'Off-heap Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              series: []
            });

            if (offheapChart) {
              var offheapMetrics = ['offHeapPinned', 'offHeapPageable'];
              offheapMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  offheapChart.addSeries(s, false);
                });
              });
              offheapChart.redraw();
            }

            // System memory chart: system used and free
            var sysMemChart = Highcharts.chart('sys-mem-chart', {
              title: { text: 'System Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              series: []
            });

            if (sysMemChart) {
              var sysMemMetrics = ['sysMemUsed', 'sysMemFree'];
              sysMemMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  sysMemChart.addSeries(s, false);
                });
              });
              sysMemChart.redraw();
            }

            // CPU usage chart: cpuPercent
            var cpuChart = Highcharts.chart('cpu-chart', {
              title: { text: 'System CPU Usage' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: 'CPU %' },
                max: 100,
                min: 0
              },
              legend: { enabled: true },
              series: []
            });

            if (cpuChart) {
              var cpuMetrics = ['cpuPercent'];
              cpuMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  cpuChart.addSeries(s, false);
                });
              });
              cpuChart.redraw();
            }

            // GPU concurrent tasks chart
            var gpuTasksChart = Highcharts.chart('gpu-tasks-chart', {
              title: { text: 'GPU Concurrent Tasks' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: 'Tasks' },
                min: 0
              },
              legend: { enabled: true },
              series: []
            });

            if (gpuTasksChart) {
              var gpuTasksMetrics = ['gpuConcurrentTasks'];
              gpuTasksMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  gpuTasksChart.addSeries(s, false);
                });
              });
              gpuTasksChart.redraw();
            }

            // GPU chart
            var gpuChart = Highcharts.chart('gpu-chart', {
              title: { text: 'GPU Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              series: []
            });

            if (gpuChart) {
              var gpuMetrics = ['gpuMemUsed'];
              gpuMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  gpuChart.addSeries(s, false);
                });
              });
              gpuChart.redraw();
            }
          }
        """)}
      </script>

    UIUtils.headerSparkPage(request, "Custom Events",
      content ++ highchartsScript ++ scriptContent, parent)
  }

}

/**
 * REST API page for custom events data
 */
class CustomEventsApiPage(parent: CustomEventsTab, customEvents: List[CustomEventData]) 
    extends WebUIPage("api") {

  override def render(request: HttpServletRequest): Seq[Node] = {
    // Return JSON directly so that hitting /customevents/api/* without
    // specifying format=json still returns useful data instead of List().
    val endpoint = Option(request.getServletPath).getOrElse("")

    val jsonString = endpoint match {
      case "/timeline" => generateTimelineJson()
      case "/summary" => generateSummaryJson()
      case "/events" => generateEventsJson(request)
      case "/metrics" => generateMetricsJson()
      case _ => generateApiIndexJson()
    }

    scala.xml.Unparsed(jsonString)
  }

  override def renderJson(request: HttpServletRequest): org.json4s.JsonAST.JValue = {
    //import org.json4s.JsonDSL._
    import org.json4s.jackson.JsonMethods._
    
    val endpoint = Option(request.getServletPath).getOrElse("")
    
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

    // series((executorId, metricName)) -> points
    val series = mutable.Map[(String, String), mutable.ArrayBuffer[(Long, Long)]]()

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
                  val buf = series.getOrElseUpdate((execId, name),
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

    val knownMetrics = Seq(
      "jvmTotal",
      "jvmUsed",
      "offHeapPinned",
      "offHeapPageable",
      "gpuMemUsed",
      "sysMemUsed",
      "sysMemFree",
      "cpuPercent",
      "gpuConcurrentTasks")

    // Collect unique executor IDs
    val execIds = series.keys.map(_._1).toSet.toSeq.sorted

    // Build per-executor, per-metric series JSON
    val seriesByExecutorEntries = execIds.map { execId =>
      val metricEntries = knownMetrics.map { name =>
        name match {
          case "jvmUsed" =>
            // Derive jvmUsed = jvmTotal - jvmFree for each timestamp
            val totalPoints =
              series.getOrElse((execId, "jvmTotal"), mutable.ArrayBuffer.empty).sortBy(_._1)
            val freePoints =
              series.getOrElse((execId, "jvmFree"), mutable.ArrayBuffer.empty).sortBy(_._1)
            val usedPoints: Seq[(Long, Long)] =
              if (totalPoints.length == freePoints.length) {
                totalPoints.zip(freePoints).map {
                  case ((tsT, vT), (_, vF)) =>
                    // timestamps should match; use tsT
                    (tsT, vT - vF)
                }
              } else {
                Seq.empty
              }
            val ptsJson = usedPoints.map { case (ts, v) => s"[$ts,$v]" }.mkString(",")
            s""""jvmUsed": [$ptsJson]"""

          case other =>
            val points = series.getOrElse((execId, other), mutable.ArrayBuffer.empty).sortBy(_._1)
            val ptsJson = points.map { case (ts, v) => s"[$ts,$v]" }.mkString(",")
            s""""$other": [$ptsJson]"""
        }
      }.mkString(",")
      s""""$execId": {$metricEntries}"""
    }.mkString(",")

    val execIdsJson = execIds.map(id => s""""$id"""").mkString(",")

    s"""{"executors": [$execIdsJson], "seriesByExecutor": {$seriesByExecutorEntries}}"""
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

