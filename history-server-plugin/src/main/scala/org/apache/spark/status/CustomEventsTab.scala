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

  override val name: String = "🚀 RAPIDS"

  attachPage(new CustomEventsPage(this, customEvents))
  attachPage(new CustomEventsApiPage(this, customEvents))
}

/**
 * Main page for the custom events tab
 */
class CustomEventsPage(tab: CustomEventsTab, customEvents: List[CustomEventData])
    extends WebUIPage("") {

  override def render(request: HttpServletRequest): Seq[Node] = {
    // Extract latest driver-level SparkRapidsBuildInfo event (no executorId) if present.
    val latestBuildInfo = customEvents.reverse.find { e =>
      e.eventType == "SparkRapidsBuildInfo" &&
        !e.eventData.get("sparkRapidsBuildInfo.executorId").exists(_.nonEmpty)
    }

    def getBuildValue(prefix: String, key: String): Option[String] =
      latestBuildInfo.flatMap(_.eventData.get(s"$prefix.$key")).filter(_.nonEmpty)

    val pluginVersion = getBuildValue("sparkRapidsBuildInfo", "version")
    val pluginRevision = getBuildValue("sparkRapidsBuildInfo", "revision")
    val jniVersion = getBuildValue("sparkRapidsJniBuildInfo", "version")
    val jniRevision = getBuildValue("sparkRapidsJniBuildInfo", "revision")
    val gpuModel = getBuildValue("sparkRapidsJniBuildInfo", "gpuModel")
    // GPU arch is not standardized across all builds, so fall back to any matching key.
    val jniArch = latestBuildInfo.flatMap { info =>
      info.eventData.collectFirst {
        case (k, v) if k.startsWith("sparkRapidsJniBuildInfo.") &&
          (k.toLowerCase.contains("arch") || k.toLowerCase.contains("compute")) => v
      }
    }

    // Monitored disk device and measured disk bandwidth.
    // NOTE: Disk bandwidth values are executor-local and are populated via
    // the /metrics JSON endpoint. We intentionally leave the initial values
    // for bandwidth empty here so the UI always reflects the currently
    // selected executor.
    val monitoredDiskDevice =
      getBuildValue("sparkRapidsBuildInfo", "monitoredDiskDevice")
    val diskWriteBw =
      Option.empty[String]
    val diskReadBw =
      Option.empty[String]

    val content = 
      <div class="row-fluid">
        <div class="span12">
          <h4>Custom Events Analysis</h4>
          <p>This tab shows RAPIDS runtime metrics captured during application execution.</p>

          <div id="rapids-build-info" style="margin-bottom: 15px;">
            <table class="table table-condensed" style="width:auto;">
              <tbody>
                <tr><th>RAPIDS Plugin Version</th>
                  <td id="rapids-plugin-version">{pluginVersion.getOrElse("")}</td></tr>
                <tr><th>RAPIDS Plugin Revision</th>
                  <td id="rapids-plugin-revision">{pluginRevision.getOrElse("")}</td></tr>
                <tr><th>spark-rapids-jni Version</th>
                  <td id="rapids-jni-version">{jniVersion.getOrElse("")}</td></tr>
                <tr><th>spark-rapids-jni Revision</th>
                  <td id="rapids-jni-revision">{jniRevision.getOrElse("")}</td></tr>
                <tr><th>GPU Model (NVML)</th>
                  <td id="rapids-gpu-model">{gpuModel.getOrElse("")}</td></tr>
                <tr><th>JNI GPU Arch (from build info)</th>
                  <td id="rapids-jni-arch">{jniArch.getOrElse("")}</td></tr>
                <tr><th>Monitored Disk Device (spark.local.dir)</th>
                  <td id="rapids-disk-device">{monitoredDiskDevice.getOrElse("")}</td></tr>
                <tr><th>Disk Write Bandwidth (bytes/s)</th>
                  <td id="rapids-disk-write-bw">{diskWriteBw.getOrElse("")}</td></tr>
                <tr><th>Disk Read Bandwidth (bytes/s)</th>
                  <td id="rapids-disk-read-bw">{diskReadBw.getOrElse("")}</td></tr>
              </tbody>
            </table>
          </div>

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
                    <div id="cpu-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="system-stage-detail-group" class="rapids-stage-detail" style="margin-top: 4px;"></div>
                  </div>
                </div>
              </div>
            </div>

            <div class="row-fluid" style="margin-top: 20px;">
              <div class="span6">
                <div class="rapids-section">
                  <h5>
                    Disk / Network
                    <a href="#" class="rapids-section-toggle" data-target="section-disk-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-disk-body">
                    <div id="disk-io-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="disk-util-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="net-io-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="io-stage-detail-group" class="rapids-stage-detail" style="margin-top: 4px;"></div>
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
                    <div id="spill-stage-detail-group" class="rapids-stage-detail" style="margin-top: 4px;"></div>
                  </div>
                </div>
              </div>
          </div>

            <div class="row-fluid" style="margin-top: 20px;">
              <div class="span12">
                <div class="rapids-section">
                  <h5>
                    GPU
                    <a href="#" class="rapids-section-toggle" data-target="section-gpu-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-gpu-body">
                    <div id="gpu-chart" style="width: 100%; height: 260px; margin-top: 10px;"></div>
                    <div id="gpu-tasks-chart" style="width: 100%; height: 220px; margin-top: 10px;"></div>
                    <div id="retries-chart" style="width: 100%; height: 230px; margin-top: 10px;"></div>
                    <div id="gpu-util-chart" style="width: 100%; height: 240px; margin-top: 10px;"></div>
                    <div id="gpu-stage-detail-group" class="rapids-stage-detail" style="margin-top: 4px;"></div>
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
            // Fetch executor list first, then metrics for the initial executor
            fetchExecutors();
          });

          function fetchExecutors() {
            $.getJSON('/history/' + getAppId() + '/customevents/api/json/metrics/executors', function(data) {
              if (!data || !data.executors || data.executors.length === 0) {
                $('#executor-filters').text('No executor metric data available');
                return;
              }
              if (!window._selectedExecutorId) {
                window._selectedExecutorId = data.executors[0];
              }
              initExecutorFilters(data.executors);
              fetchMetricData(window._selectedExecutorId);
            });
          }

          function fetchMetricData(executorId) {
            var url = '/history/' + getAppId() + '/customevents/api/json/metrics';
            if (executorId) {
              url += '?executorId=' + encodeURIComponent(executorId);
            }
            $.getJSON(url, function(data) {
              window._rapidsMetricData = data;
              if (executorId) {
                window._selectedExecutorId = executorId;
              } else if (data.selectedExecutor && !window._selectedExecutorId) {
                window._selectedExecutorId = data.selectedExecutor;
              }
              if (data.executorBuildInfo) {
                updateExecutorBuildInfo(data.executorBuildInfo);
              }
              renderMetricCharts();
            });
          }

          function getAppId() {
            // Extract app ID from URL
            var path = window.location.pathname;
            var match = path.match(/\/history\/([^\/]+)/);
            return match ? match[1] : '';
          }

          function initExecutorFilters(executors) {
            var container = $('#executor-filters');
            container.empty();
            if (!executors || executors.length === 0) {
              container.append('<span>No executor metric data available</span>');
              return;
            }

            if (!window._selectedExecutorId) {
              window._selectedExecutorId = executors[0];
            }

            container.append('<span style="margin-right: 8px;">Executor:</span>');

            executors.forEach(function(execId) {
              var isActive = (execId === window._selectedExecutorId);
              var btn = $('<button type="button" class="btn btn-mini"></button>');
              btn.text(execId);
              if (isActive) {
                btn.addClass('btn-primary');
              }
              btn.css('margin-right', '6px');
              btn.on('click', function() {
                if (window._selectedExecutorId === execId) {
                  return;
                }
                window._selectedExecutorId = execId;
                fetchMetricData(execId);
              });
              container.append(btn);
            });
          }

          function updateExecutorBuildInfo(info) {
            if (!info) return;
            if (info.pluginVersion !== undefined) {
              $('#rapids-plugin-version').text(info.pluginVersion);
            }
            if (info.pluginRevision !== undefined) {
              $('#rapids-plugin-revision').text(info.pluginRevision);
            }
            if (info.jniVersion !== undefined) {
              $('#rapids-jni-version').text(info.jniVersion);
            }
            if (info.jniRevision !== undefined) {
              $('#rapids-jni-revision').text(info.jniRevision);
            }
            if (info.gpuModel !== undefined) {
              $('#rapids-gpu-model').text(info.gpuModel);
            }
            if (info.jniArch !== undefined) {
              $('#rapids-jni-arch').text(info.jniArch);
            }
            if (info.diskDevice !== undefined) {
              $('#rapids-disk-device').text(info.diskDevice);
            }
            if (info.diskWriteBwBytesPerSec !== undefined) {
              $('#rapids-disk-write-bw').text(info.diskWriteBwBytesPerSec);
            }
            if (info.diskReadBwBytesPerSec !== undefined) {
              $('#rapids-disk-read-bw').text(info.diskReadBwBytesPerSec);
            }
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

            var stages = data.stages || [];
            // Global grouping of related charts that should share tooltip content
            // and, where wired, x-axis range. Each entry maps a group name to an
            // array of Highcharts chart instances.
            window._rapidsChartGroups = window._rapidsChartGroups || {};
            var selectedExecId = window._selectedExecutorId;
            if (!selectedExecId) {
              var keys = Object.keys(data.seriesByExecutor || {});
              if (keys.length > 0) {
                selectedExecId = keys[0];
                window._selectedExecutorId = selectedExecId;
              }
            }
            if (!selectedExecId) {
              return;
            }

            function getSeriesForMetric(metricName) {
              var allSeries = [];
              var execSeries = (data.seriesByExecutor[selectedExecId] &&
                data.seriesByExecutor[selectedExecId][metricName]) || [];
              if (execSeries.length > 0) {
                allSeries.push({
                  name: selectedExecId + ' ' + metricName,
                  data: execSeries
                });
              }
              return allSeries;
            }

          function getActiveStagesAt(ts) {
              if (!stages || stages.length === 0) {
                return [];
              }
              var active = [];
              for (var i = 0; i < stages.length; i++) {
                var s = stages[i];
                if (s.startTime != null && s.endTime != null &&
                    s.startTime <= ts && ts <= s.endTime) {
                  active.push(s);
                }
              }
              return active;
            }

            function escapeHtml(str) {
              if (str == null) return '';
              return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
            }

            function updateStageDetailTable(containerId, ts) {
              var container = $('#' + containerId);
              if (!container.length) {
                return;
              }
              // Center the stage detail "card" within the available width.
              container.css('text-align', 'center');
              var active = getActiveStagesAt(ts);
              if (!active || active.length === 0) {
                container.html(
                  '<div style="display:inline-block;max-width:700px;margin:0 auto;' +
                  'border:1px solid #ccc;border-radius:6px;padding:6px 10px;' +
                  'background-color:#fafafa;font-size:11px;color:#777;">' +
                  '<span style="float:right;cursor:pointer;color:#999;" ' +
                  'onclick="$(\'#' + escapeHtml(containerId) +
                  '\').empty();">&times;</span>' +
                  'No stages active at the selected time.' +
                  '</div>');
                return;
              }

              var headerTime = Highcharts.dateFormat('%Y-%m-%d %H:%M:%S', ts);
              var html = '';
              html += '<div style="display:inline-block;max-width:900px;margin:0 auto;' +
                'border:1px solid #ccc;border-radius:6px;padding:6px 10px;' +
                'background-color:#fafafa;">';
              html += '<div style="font-size:11px;margin-bottom:4px;overflow:hidden;">' +
                '<span>Active stages at ' + escapeHtml(headerTime) + '</span>' +
                '<span style="float:right;cursor:pointer;color:#999;" ' +
                'onclick="$(\'#' + escapeHtml(containerId) +
                '\').empty();">&times;</span>' +
                '</div>';
              html += '<table class="table table-condensed" ' +
                'style="font-size:11px;margin-bottom:0;background-color:white;border-radius:4px;">';
              html += '<thead><tr>' +
                '<th>Stage ID</th>' +
                '<th>Attempt</th>' +
                '<th>Name</th>' +
                '<th>Start Time</th>' +
                '<th>End Time</th>' +
                '<th>Link</th>' +
                '</tr></thead><tbody>';

              active.forEach(function(st) {
                var id = st.id;
                var attemptId = (st.attemptId != null) ? st.attemptId : 0;
                var start = st.startTime != null
                  ? Highcharts.dateFormat('%Y-%m-%d %H:%M:%S', st.startTime)
                  : '';
                var end = st.endTime != null
                  ? Highcharts.dateFormat('%Y-%m-%d %H:%M:%S', st.endTime)
                  : '';
                var href = '/history/' + getAppId() + '/stages/stage/?id=' +
                  encodeURIComponent(id) + '&attempt=' + encodeURIComponent(attemptId);
                html += '<tr>' +
                  '<td>' + escapeHtml(id) + '</td>' +
                  '<td>' + escapeHtml(attemptId) + '</td>' +
                  '<td>' + escapeHtml(st.name) + '</td>' +
                  '<td>' + escapeHtml(start) + '</td>' +
                  '<td>' + escapeHtml(end) + '</td>' +
                  '<td><a href="' + href + '">View</a></td>' +
                  '</tr>';
              });

              html += '</tbody></table>';
              container.html(html);
            }

            function makeLinkedXAxis(groupName, options) {
              var cfg = options || {};
              if (!cfg.type) {
                cfg.type = 'datetime';
              }
              var existingEvents = cfg.events || {};
              cfg.events = existingEvents;
              var prevSetExtremes = existingEvents.setExtremes;
              cfg.events.setExtremes = function (e) {
                if (prevSetExtremes) {
                  prevSetExtremes.call(this, e);
                }
                if (!e || e.trigger === 'syncExtremes') {
                  return;
                }
                var groups = window._rapidsChartGroups || {};
                var charts = (groups[groupName] || []);
                var me = this.chart;
                charts.forEach(function (c) {
                  if (c && c !== me) {
                    c.xAxis[0].setExtremes(
                      e.min, e.max, false, false, { trigger: 'syncExtremes' });
                    c.redraw();
                  }
                });
              };
              return cfg;
            }

            // Memory composition chart: jvmUsed, offHeapPinned, offHeapPageable, systemOtherUsed
            var memCompChart = Highcharts.chart('mem-composition-chart', {
              chart: { type: 'area' },
              title: { text: 'Memory Composition (Used)', align: 'left' },
              xAxis: makeLinkedXAxis('system', { type: 'datetime' }),
              yAxis: {
                title: { text: 'Bytes' }
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('system-stage-detail-group', this.x);
                      }
                    }
                  }
                },
                area: {
                  stacking: 'normal',
                  marker: { enabled: false }
                }
              },
              tooltip: { enabled: false },
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

              // Draw system total memory as a non-stacked background line for the
              // currently selected executor: sysMemTotal = sysMemUsed + sysMemFree.
              var anyExecId = selectedExecId;
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
              title: { text: 'JVM Memory', align: 'left' },
              xAxis: makeLinkedXAxis('system', { type: 'datetime' }),
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('system-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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
              title: { text: 'Off-heap Memory', align: 'left' },
              xAxis: makeLinkedXAxis('system', { type: 'datetime' }),
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('system-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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
              title: { text: 'System Memory', align: 'left' },
              xAxis: makeLinkedXAxis('system', { type: 'datetime' }),
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('system-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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
              title: { text: 'System CPU Usage', align: 'left' },
              xAxis: makeLinkedXAxis('system', { type: 'datetime' }),
              yAxis: {
                title: { text: 'CPU %' },
                max: 100,
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('system-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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
              title: { text: 'GPU Concurrent Tasks', align: 'left' },
              xAxis: makeLinkedXAxis('gpu', { type: 'datetime' }),
              yAxis: {
                title: { text: 'Tasks' },
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('gpu-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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
              title: { text: 'Retries', align: 'left' },
              xAxis: makeLinkedXAxis('gpu', { type: 'datetime' }),
              yAxis: {
                title: { text: 'Count (per interval per executor)' },
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('gpu-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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

            // Group: System (CPU + Memory charts) with linked stage table
            window._rapidsChartGroups.system = [
              memCompChart,
              jvmChart,
              offheapChart,
              sysMemChart,
              cpuChart
            ];

            // Disk IO chart: diskReadBytes, diskWriteBytes (per sample interval)
            var diskIoChart = Highcharts.chart('disk-io-chart', {
              title: { text: 'Disk IO (sample interval bytes)', align: 'left' },
              xAxis: makeLinkedXAxis('io', { type: 'datetime' }),
              yAxis: {
                title: { text: 'Bytes per interval' },
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('io-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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
              title: { text: 'Disk Utilization (spark.local.dir device)', align: 'left' },
              xAxis: makeLinkedXAxis('io', { type: 'datetime' }),
              yAxis: {
                title: { text: '% busy' },
                max: 100,
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('io-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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

            // Network IO chart: netReadBytes, netWriteBytes (per sample interval)
            var netIoChart = Highcharts.chart('net-io-chart', {
              title: { text: 'Network IO (sample interval bytes)', align: 'left' },
              xAxis: makeLinkedXAxis('io', { type: 'datetime' }),
              yAxis: {
                title: { text: 'Bytes per interval' },
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('io-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
              series: []
            });

            if (netIoChart) {
              var netMetrics = ['netReadBytes', 'netWriteBytes'];
              netMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
              netIoChart.addSeries(s, false);
                });
              });
              netIoChart.redraw();
            }

            // Group: IO (disk IO, disk util, net IO)
            window._rapidsChartGroups.io = [
              diskIoChart,
              diskUtilChart,
              netIoChart
            ];

            // Spill time chart: GPU spill times from GpuTaskMetrics (seconds, per interval)
            var spillTimeChart = Highcharts.chart('spill-time-chart', {
              title: { text: 'GPU Spill Time', align: 'left' },
              xAxis: makeLinkedXAxis('spill', { type: 'datetime' }),
              yAxis: {
                title: { text: 'Time (s, per interval per executor)' },
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('spill-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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
              title: { text: 'GPU Spill Bytes', align: 'left' },
              xAxis: makeLinkedXAxis('spill', { type: 'datetime' }),
              yAxis: {
                title: { text: 'Bytes (per interval per executor)' },
                min: 0
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('spill-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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

            // Group: Spill (time + bytes)
            window._rapidsChartGroups.spill = [
              spillTimeChart,
              spillBytesChart
            ];

            // GPU memory chart (top) - shares x-axis range with GPU SM util chart.
            // We hide the x-axis labels here so that only the bottom chart shows
            // the time axis, but both charts stay synchronized.
            var gpuChart = Highcharts.chart('gpu-chart', {
              title: { text: 'GPU Memory', align: 'left' },
              xAxis: makeLinkedXAxis('gpu', {
                type: 'datetime',
                labels: { enabled: false },
                tickLength: 0,
                lineWidth: 0
              }),
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('gpu-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
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

            // GPU SM utilization chart
            var gpuUtilChart = Highcharts.chart('gpu-util-chart', {
              title: { text: 'GPU SM Utilization', align: 'left' },
              xAxis: makeLinkedXAxis('gpu', { type: 'datetime' }),
              yAxis: {
                title: { text: 'SM Utilization (%)' },
                min: 0,
                max: 100
              },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        updateStageDetailTable('gpu-stage-detail-group', this.x);
                      }
                    }
                  }
                }
              },
              tooltip: { enabled: false },
              series: []
            });

            if (gpuUtilChart) {
              var gpuUtilMetrics = ['gpuSmUtilPct'];
              gpuUtilMetrics.forEach(function(metricName) {
                var metricSeries = getSeriesForMetric(metricName);
                metricSeries.forEach(function(s) {
              gpuUtilChart.addSeries(s, false);
                });
              });
              gpuUtilChart.redraw();
            }

            // Register GPU charts as a group for synchronized behavior (x-axis, tooltip).
            window._rapidsChartGroups.gpu = [
              gpuChart,
              gpuUtilChart,
              gpuTasksChart,
              retriesChart
            ];

            // Highcharts synchronized-charts style behavior for grouped charts.
            // Adapted from the Highcharts demo: https://www.highcharts.com/demo/highcharts/synchronized-charts
            if (!Highcharts.Point.prototype.highlight) {
              Highcharts.Point.prototype.highlight = function (event) {
                var chart = this.series && this.series.chart;
                this.onMouseOver(); // Show hover marker
                if (chart && chart.xAxis && chart.xAxis.length > 0) {
                  chart.xAxis[0].drawCrosshair(event, this);
                }
              };
            }

            if (!Highcharts.Pointer.prototype._rapidsResetPatched) {
              // Do not hide tooltips/crosshairs on mouse out; we control them via sync.
              Highcharts.Pointer.prototype.reset = function () {
                return undefined;
              };
              Highcharts.Pointer.prototype._rapidsResetPatched = true;
            }

            if (!window._rapidsPointerSyncBound) {
              window._rapidsPointerSyncBound = {};
            }

            function updateChartSummaryLabel(chart, metricValues) {
              try {
                var parts = [];
                Object.keys(metricValues).sort().forEach(function (name) {
                  var v = metricValues[name];
                  parts.push(escapeHtml(name) + '=' + escapeHtml(v));
                });
                var text = parts.join('<br/>');
                if (!text) {
                  if (chart._rapidsSummaryLabel) {
                    chart._rapidsSummaryLabel.destroy();
                    chart._rapidsSummaryLabel = null;
                  }
                  return;
                }
                var label = chart._rapidsSummaryLabel;
                if (!label) {
                  label = chart.renderer.label(
                    text,
                    chart.plotWidth - 10,
                    10,
                    null,
                    null,
                    null,
                    true
                  ).attr({
                    align: 'right',
                    zIndex: 5
                  }).css({
                    fontSize: '11px',
                    textAlign: 'right',
                    pointerEvents: 'none',
                    backgroundColor: 'rgba(255,255,255,0.90)',
                    padding: '4px 6px',
                    borderRadius: '4px',
                    boxShadow: '0 0 2px rgba(0,0,0,0.25)'
                  }).add();
                  chart._rapidsSummaryLabel = label;
                } else {
                  label.attr({ text: text });
                }
                // Reposition to top-right inside the plot area, roughly aligned with the title
                var yOffset = 10;
                try {
                  if (chart.title && chart.title.element && chart.title.getBBox) {
                    yOffset = chart.title.getBBox().y || yOffset;
                  }
                } catch (ignore) {}
                label.align({
                  align: 'right',
                  verticalAlign: 'top',
                  x: -10,
                  y: yOffset
                }, null, 'spacingBox');
              } catch (e) {
                // best-effort; don't break charts on error
                if (window.console && console.log) {
                  console.log('[RAPIDS metrics] failed to update summary label', e);
                }
              }
            }

            function bindPointerSyncForGroup(groupName, containerId) {
              if (!containerId || window._rapidsPointerSyncBound[groupName]) {
                return;
              }
              window._rapidsPointerSyncBound[groupName] = true;
              ['mousemove', 'touchmove', 'touchstart'].forEach(function (eventType) {
                var container = document.getElementById(containerId);
                if (!container) {
                  return;
                }
                container.addEventListener(eventType, function (e) {
                  var charts = (window._rapidsChartGroups &&
                    window._rapidsChartGroups[groupName]) || [];
                  if (!charts || !charts.length) {
                    return;
                  }
                  charts.forEach(function (chart) {
                    if (!chart || !chart.pointer) {
                      return;
                    }
                    var event = chart.pointer.normalize(e);
                    var bestPoint = null;
                    var metricValues = {};
                    chart.series.forEach(function (s) {
                      if (!s.visible) {
                        return;
                      }
                      var p = s.searchPoint(event, true);
                      if (!p) {
                        return;
                      }
                      if (!bestPoint || Math.abs(p.x - event.chartX) <
                          Math.abs(bestPoint.x - event.chartX)) {
                        bestPoint = p;
                      }
                      // series.name is like "<execId> <metricName>"
                      var name = s.name || '';
                      var idx = name.indexOf(' ');
                      var metricName = (idx >= 0) ? name.substring(idx + 1) : name;
                      metricValues[metricName] = p.y;
                    });
                    if (bestPoint) {
                      bestPoint.highlight(event);
                    }
                    updateChartSummaryLabel(chart, metricValues);
                  });
                });
              });
            }

            // Bind synchronized pointer behavior for all chart groups.
            bindPointerSyncForGroup('system', 'section-memory-body');
            bindPointerSyncForGroup('io', 'section-disk-body');
            bindPointerSyncForGroup('spill', 'section-spill-body');
            bindPointerSyncForGroup('gpu', 'section-gpu-body');
          }
        """)}
      </script>

    UIUtils.headerSparkPage(request, "Custom Events",
      content ++ highchartsScript ++ scriptContent, tab)
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
      case "/metrics" => generateMetricsJson(request)
      case "/metrics/executors" => generateMetricsExecutorsJson()
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
      case "/metrics" => generateMetricsJson(request)
      case "/metrics/executors" => generateMetricsExecutorsJson()
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

  private def generateMetricsJson(request: HttpServletRequest): String = {
    // Build a mapping from executorId -> metric names (in positional order)
    val metricDefs: Map[String, Seq[String]] = customEvents
      .filter(_.eventType == "MetricDefinition")
      .flatMap { e =>
        for {
          execId <- e.eventData.get("executorId")
          namesStr <- e.eventData.get("metricNames")
        } yield execId -> namesStr.split(",").map(_.trim).filter(_.nonEmpty).toSeq
      }.groupBy(_._1).mapValues(_.last._2).toMap

    // Determine which executor we should materialize metrics for.
    val allExecIds = metricDefs.keys.toSeq.sorted
    val requestedExecId = Option(request.getParameter("executorId")).filter(_.nonEmpty)
    val targetExecIdOpt = requestedExecId.orElse(allExecIds.headOption)

    // series((executorId, metricName)) -> points, but we only populate for targetExecIdOpt.
    val series = mutable.Map[(String, String), mutable.ArrayBuffer[(Long, Long)]]()

    // Process each MetricUpdates event: decode hex payload and expand into per-metric series
    customEvents
      .filter(_.eventType == "MetricUpdates")
      .foreach { e =>
        for {
          execId <- e.eventData.get("executorId")
          encoded <- e.eventData.get("encodedMetricsB64")
          metricNames <- metricDefs.get(execId)
          if targetExecIdOpt.forall(_ == execId)
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
      "netReadBytes",
      "netWriteBytes",
      "gpuSpillToHostTimeNs",
      "gpuSpillToDiskTimeNs",
      "gpuReadSpillFromHostTimeNs",
      "gpuReadSpillFromDiskTimeNs",
      "gpuSpillHostBytes",
      "gpuSpillDiskBytes",
      "gpuSmUtilPct")

    // Collect executor IDs to return in this response
    val execIds =
      if (requestedExecId.isDefined) {
        targetExecIdOpt.toSeq
      } else {
        allExecIds
      }

    // Derive stage ranges from StageCompleted events captured by the listener.
    import scala.util.Try
    val stageEvents = customEvents.filter(_.eventType == "StageCompleted")
    val stages = stageEvents.flatMap { e =>
      val data = e.eventData
      for {
        idStr <- data.get("stageId")
        id <- Try(idStr.toInt).toOption
      } yield {
        val name = data.getOrElse("stageName", s"Stage $id")
        val attemptId = data
          .get("stageAttemptId")
          .flatMap(s => Try(s.toInt).toOption)
          .getOrElse(0)
        val start = data
          .get("stageStartTime")
          .flatMap(s => Try(s.toLong).toOption)
          .getOrElse(e.timestamp)
        val end = data
          .get("stageEndTime")
          .flatMap(s => Try(s.toLong).toOption)
          .getOrElse(e.timestamp)
        (id, attemptId, name, start, end)
      }
    }

    def escapeJsonString(s: String): String =
      s.replace("\\", "\\\\").replace("\"", "\\\"")

    // Extract per-executor build info for the selected executor, if available.
    // IMPORTANT: We *only* use per-executor SparkRapidsBuildInfo events here.
    // We intentionally do NOT fall back to the driver-level build info because
    // that would make executor-specific fields (like disk bandwidth) appear
    // to vary per executor when they are actually global.
    val executorBuildInfoFields: String = targetExecIdOpt.flatMap { execId =>
      val maybeEvent = customEvents.reverse.find { e =>
        e.eventType == "SparkRapidsBuildInfo" &&
          e.eventData.get("sparkRapidsBuildInfo.executorId").contains(execId)
      }
      maybeEvent.map { e =>
        val data = e.eventData
        val pluginVersion = data.get("sparkRapidsBuildInfo.version")
        val pluginRevision = data.get("sparkRapidsBuildInfo.revision")
        val jniVersion = data.get("sparkRapidsJniBuildInfo.version")
        val jniRevision = data.get("sparkRapidsJniBuildInfo.revision")
        val gpuModel = data.get("sparkRapidsJniBuildInfo.gpuModel")
        val jniArch = data.collectFirst {
          case (k, v) if k.startsWith("sparkRapidsJniBuildInfo.") &&
            (k.toLowerCase.contains("arch") || k.toLowerCase.contains("compute")) => v
        }
        val diskDevice = data.get("sparkRapidsBuildInfo.monitoredDiskDevice")
        val diskWriteBw = data.get("sparkRapidsBuildInfo.diskWriteBwBytesPerSec")
        val diskReadBw = data.get("sparkRapidsBuildInfo.diskReadBwBytesPerSec")

        val fields = Seq(
          pluginVersion.map(v => s""""pluginVersion":"${escapeJsonString(v)}""""),
          pluginRevision.map(v => s""""pluginRevision":"${escapeJsonString(v)}""""),
          jniVersion.map(v => s""""jniVersion":"${escapeJsonString(v)}""""),
          jniRevision.map(v => s""""jniRevision":"${escapeJsonString(v)}""""),
          gpuModel.map(v => s""""gpuModel":"${escapeJsonString(v)}""""),
          jniArch.map(v => s""""jniArch":"${escapeJsonString(v)}""""),
          diskDevice.map(v => s""""diskDevice":"${escapeJsonString(v)}""""),
          diskWriteBw.map(v => s""""diskWriteBwBytesPerSec":"${escapeJsonString(v)}""""),
          diskReadBw.map(v => s""""diskReadBwBytesPerSec":"${escapeJsonString(v)}"""")
        ).flatten.mkString(",")
        fields
      }
    }.getOrElse("")

    // Build per-executor, per-metric series JSON for the selected executor only
    val visibleExecIds = targetExecIdOpt.toSeq
    val seriesByExecutorEntries = visibleExecIds.map { execId =>
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

    val stagesJson = stages.map {
      case (id, attemptId, name, start, end) =>
        s"""{"id":$id,"attemptId":$attemptId,"name":"${escapeJsonString(name)}","startTime":$start,"endTime":$end}"""
    }.mkString(",")
    val selectedExecutorJson = targetExecIdOpt.map(id => s""""selectedExecutor":"$id",""")
      .getOrElse("")
    val executorBuildInfoJson =
      if (executorBuildInfoFields.nonEmpty) s""""executorBuildInfo":{$executorBuildInfoFields},""" else ""

    s"""{"executors": [$execIdsJson], $selectedExecutorJson $executorBuildInfoJson "seriesByExecutor": {$seriesByExecutorEntries}, "stages": [$stagesJson]}"""
  }

  /**
   * Lightweight endpoint to list executor IDs that have metric definitions.
   * This avoids returning any time-series data.
   */
  private def generateMetricsExecutorsJson(): String = {
    val execIds =
      customEvents
        .filter(_.eventType == "MetricDefinition")
        .flatMap(_.eventData.get("executorId"))
        .distinct
        .sorted
    val execIdsJson = execIds.map(id => s""""$id"""").mkString(",")
    s"""{"executors": [$execIdsJson]}"""
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

