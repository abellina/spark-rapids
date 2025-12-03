package org.apache.spark.status

import java.nio.{ByteBuffer, ByteOrder}
import java.util.Base64

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

            <div class="row-fluid">
              <div class="span6">
                <div class="rapids-section">
                  <h5>
                    Memory
                    <a href="#" class="rapids-section-toggle" data-target="section-memory-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-memory-body">
                    <div id="mem-composition-chart" style="width: 100%; height: 300px; margin-top: 10px;"></div>
                    <div id="jvm-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="offheap-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="sys-mem-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                  </div>
                </div>
              </div>

              <div class="span6">
                <div class="rapids-section">
                  <h5>
                    CPU / Tasks / Retries
                    <a href="#" class="rapids-section-toggle" data-target="section-cpu-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-cpu-body">
                    <div id="cpu-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="gpu-tasks-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="retries-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                  </div>
                </div>
              </div>
            </div>

            <div class="row-fluid" style="margin-top: 20px;">
              <div class="span6">
                <div class="rapids-section">
                  <h5>
                    Disk
                    <a href="#" class="rapids-section-toggle" data-target="section-disk-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-disk-body">
                    <div id="disk-io-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="disk-util-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                  </div>
                </div>
              </div>

              <div class="span6">
                <div class="rapids-section">
                  <h5>
                    GPU Spill
                    <a href="#" class="rapids-section-toggle" data-target="section-spill-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-spill-body">
                    <div id="spill-time-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="spill-bytes-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                  </div>
                </div>
              </div>
            </div>

            <div class="row-fluid" style="margin-top: 20px;">
              <div class="span12">
                <div class="rapids-section">
                  <h5>
                    GPU Memory
                    <a href="#" class="rapids-section-toggle" data-target="section-gpu-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-gpu-body">
                    <div id="gpu-chart" style="width: 100%; height: 300px; margin-top: 10px;"></div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>

    val highchartsScript =
      <script src="https://code.highcharts.com/highcharts.js"></script>

    val scriptContent = 
      <script type="text/javascript">
        {scala.xml.Unparsed("""
          $(document).ready(function() {
            // Initialize collapsible sections
            initSectionToggles();
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

          function initSectionToggles() {
            $('.rapids-section-toggle').off('click').on('click', function(e) {
              e.preventDefault();
              var targetId = $(this).data('target');
              var body = $('#' + targetId);
              if (!body.length) {
                return;
              }
              if (body.is(':visible')) {
                body.hide();
                $(this).text('[show]');
              } else {
                body.show();
                $(this).text('[hide]');
              }
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

            // Memory composition chart: jvmUsed, offHeapPinned, offHeapPageable, systemOtherUsed
            var memCompChart = Highcharts.chart('mem-composition-chart', {
              chart: { type: 'area' },
              title: { text: 'Memory Composition (Used)' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: 'Bytes' }
              },
              legend: { enabled: true },
              plotOptions: {
                area: {
                  stacking: 'normal',
                  marker: { enabled: false }
                }
              },
              series: []
            });

            if (memCompChart) {
              var compMetrics = ['jvmUsed', 'offHeapPinned', 'offHeapPageable', 'systemOtherUsed'];
              compMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  s.type = 'area';
                  memCompChart.addSeries(s, false);
                });
              });

              // Draw system total memory as a non-stacked background line.
              // Use the first selected executor (if any) to compute
              // sysMemTotal = sysMemUsed + sysMemFree.
              var anyExecId = null;
              selected.forEach(function(execId) {
                if (!anyExecId && data.seriesByExecutor[execId]) {
                  anyExecId = execId;
                }
              });
              if (anyExecId && data.seriesByExecutor[anyExecId]) {
                var used = data.seriesByExecutor[anyExecId]['sysMemUsed'] || [];
                var free = data.seriesByExecutor[anyExecId]['sysMemFree'] || [];
                if (used.length === free.length && used.length > 0) {
                  var totalData = [];
                  for (var i = 0; i < used.length; i++) {
                    var ts = used[i][0];
                    var total = used[i][1] + free[i][1];
                    totalData.push([ts, total]);
                  }
                  memCompChart.addSeries({
                    name: anyExecId + ' sysMemTotal',
                    type: 'line',
                    data: totalData,
                    color: '#888888',
                    lineWidth: 1,
                    zIndex: 0,
                    enableMouseTracking: false,
                    marker: { enabled: false }
                  }, false);
                }
              }

              memCompChart.redraw();
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

            // Retries chart: retryCount, splitRetryCount
            var retriesChart = Highcharts.chart('retries-chart', {
              title: { text: 'Retries' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: 'Count (per interval per executor)' },
                min: 0
              },
              legend: { enabled: true },
              series: []
            });

            if (retriesChart) {
              var retryMetrics = ['retryCount', 'splitRetryCount'];
              retryMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  retriesChart.addSeries(s, false);
                });
              });
              retriesChart.redraw();
            }

            // Disk IO chart: diskReadBytes, diskWriteBytes (per sample interval)
            var diskIoChart = Highcharts.chart('disk-io-chart', {
              title: { text: 'Disk IO (sample interval bytes)' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: 'Bytes per interval' },
                min: 0
              },
              legend: { enabled: true },
              series: []
            });

            if (diskIoChart) {
              var ioMetrics = ['diskReadBytes', 'diskWriteBytes'];
              ioMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  diskIoChart.addSeries(s, false);
                });
              });
              diskIoChart.redraw();
            }

            // Disk utilization chart: diskUtilPct
            var diskUtilChart = Highcharts.chart('disk-util-chart', {
              title: { text: 'Disk Utilization (spark.local.dir device)' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: '% busy' },
                max: 100,
                min: 0
              },
              legend: { enabled: true },
              series: []
            });

            if (diskUtilChart) {
              var utilMetrics = ['diskUtilPct'];
              utilMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  diskUtilChart.addSeries(s, false);
                });
              });
              diskUtilChart.redraw();
            }

            // Spill time chart: GPU spill times from GpuTaskMetrics (seconds, per interval)
            var spillTimeChart = Highcharts.chart('spill-time-chart', {
              title: { text: 'GPU Spill Time' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: 'Time (s, per interval per executor)' },
                min: 0
              },
              legend: { enabled: true },
              series: []
            });

            if (spillTimeChart) {
              var spillTimeMetrics = [
                'gpuSpillToHostTimeNs',
                'gpuSpillToDiskTimeNs',
                'gpuReadSpillFromHostTimeNs',
                'gpuReadSpillFromDiskTimeNs'
              ];
              spillTimeMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  spillTimeChart.addSeries(s, false);
                });
              });
              spillTimeChart.redraw();
            }

            // Spill bytes chart: GPU spill bytes from GpuTaskMetrics (per interval)
            var spillBytesChart = Highcharts.chart('spill-bytes-chart', {
              title: { text: 'GPU Spill Bytes' },
              xAxis: { type: 'datetime' },
              yAxis: {
                title: { text: 'Bytes (per interval per executor)' },
                min: 0
              },
              legend: { enabled: true },
              series: []
            });

            if (spillBytesChart) {
              var spillByteMetrics = ['gpuSpillHostBytes', 'gpuSpillDiskBytes'];
              spillByteMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
                  spillBytesChart.addSeries(s, false);
                });
              });
              spillBytesChart.redraw();
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

  private def base64ToBytes(encoded: String): Array[Byte] = {
    try {
      Base64.getDecoder.decode(encoded.trim)
    } catch {
      case _: IllegalArgumentException =>
        Array.emptyByteArray
    }
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
          encoded <- e.eventData.get("encodedMetricsB64")
          metricNames <- metricDefs.get(execId)
        } {
          val bytes = base64ToBytes(encoded)
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

    val timeMetrics = Set(
      "gpuSpillToHostTimeNs",
      "gpuSpillToDiskTimeNs",
      "gpuReadSpillFromHostTimeNs",
      "gpuReadSpillFromDiskTimeNs")

    val knownMetrics = Seq(
      "jvmTotal",
      "jvmUsed",
      "offHeapPinned",
      "offHeapPageable",
      "gpuMemUsed",
      "sysMemUsed",
      "sysMemFree",
      "systemOtherUsed",
      "cpuPercent",
      "gpuConcurrentTasks",
      "retryCount",
      "splitRetryCount",
      "diskReadBytes",
      "diskWriteBytes",
      "diskUtilPct",
      "gpuSpillToHostTimeNs",
      "gpuSpillToDiskTimeNs",
      "gpuReadSpillFromHostTimeNs",
      "gpuReadSpillFromDiskTimeNs",
      "gpuSpillHostBytes",
      "gpuSpillDiskBytes")

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

          case "systemOtherUsed" =>
            // systemOtherUsed = sysMemUsed - (jvmUsed + offHeapPinned + offHeapPageable)
            val sysUsedPoints =
              series.getOrElse((execId, "sysMemUsed"), mutable.ArrayBuffer.empty).sortBy(_._1)
            val jvmTotalPoints =
              series.getOrElse((execId, "jvmTotal"), mutable.ArrayBuffer.empty).sortBy(_._1)
            val jvmFreePoints =
              series.getOrElse((execId, "jvmFree"), mutable.ArrayBuffer.empty).sortBy(_._1)
            val pinnedPoints =
              series.getOrElse((execId, "offHeapPinned"), mutable.ArrayBuffer.empty).sortBy(_._1)
            val pageablePoints =
              series.getOrElse((execId, "offHeapPageable"), mutable.ArrayBuffer.empty).sortBy(_._1)

            val otherPoints: Seq[(Long, Long)] =
              if (sysUsedPoints.length == jvmTotalPoints.length &&
                  jvmTotalPoints.length == jvmFreePoints.length &&
                  jvmFreePoints.length == pinnedPoints.length &&
                  pinnedPoints.length == pageablePoints.length) {
                sysUsedPoints
                  .zip(jvmTotalPoints)
                  .zip(jvmFreePoints)
                  .zip(pinnedPoints)
                  .zip(pageablePoints)
                  .map {
                    case ((((sysUsedPair, jvmTotalPair), jvmFreePair), pinnedPair), pageablePair) =>
                      val (ts, sysUsed) = sysUsedPair
                      val jvmUsed = jvmTotalPair._2 - jvmFreePair._2
                      val offHeapPinned = pinnedPair._2
                      val offHeapPageable = pageablePair._2
                      val rapidsUsed = jvmUsed + offHeapPinned + offHeapPageable
                      val otherUsed = math.max(0L, sysUsed - rapidsUsed)
                      (ts, otherUsed)
                  }
              } else {
                Seq.empty
              }

            val ptsJson = otherPoints.map { case (ts, v) => s"[$ts,$v]" }.mkString(",")
            s""""systemOtherUsed": [$ptsJson]"""

          case timeMetric if timeMetrics.contains(timeMetric) =>
            // Convert nanoseconds to fractional seconds for display
            val points =
              series.getOrElse((execId, timeMetric), mutable.ArrayBuffer.empty).sortBy(_._1)
            val ptsJson = points.map { case (ts, v) =>
              val secs = v.toDouble / 1e9
              s"[$ts,$secs]"
            }.mkString(",")
            s""""$timeMetric": [$ptsJson]"""

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

