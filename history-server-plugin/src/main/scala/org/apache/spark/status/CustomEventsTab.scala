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
    // Extract latest SparkRapidsBuildInfo event if present
    val latestBuildInfo = customEvents.reverse.find(_.eventType == "SparkRapidsBuildInfo")

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

    // Monitored disk device and measured disk bandwidth (driver-side test)
    val monitoredDiskDevice =
      getBuildValue("sparkRapidsBuildInfo", "monitoredDiskDevice")
    val diskWriteBw =
      getBuildValue("sparkRapidsBuildInfo", "diskWriteBwBytesPerSec")
    val diskReadBw =
      getBuildValue("sparkRapidsBuildInfo", "diskReadBwBytesPerSec")

    val content = 
      <div class="row-fluid">
        <div class="span12">
          <h4>Custom Events Analysis</h4>
          <p>This tab shows RAPIDS runtime metrics captured during application execution.</p>

          <div id="rapids-build-info" style="margin-bottom: 15px;">
            <table class="table table-condensed" style="width:auto;">
              <tbody>
                {
                  Seq(
                    pluginVersion.map(v => <tr><th>RAPIDS Plugin Version</th><td>{v}</td></tr>),
                    pluginRevision.map(v => <tr><th>RAPIDS Plugin Revision</th><td>{v}</td></tr>),
                    jniVersion.map(v => <tr><th>spark-rapids-jni Version</th><td>{v}</td></tr>),
                    jniRevision.map(v => <tr><th>spark-rapids-jni Revision</th><td>{v}</td></tr>),
                    gpuModel.map(v => <tr><th>GPU Model (NVML)</th><td>{v}</td></tr>),
                    jniArch.map(v => <tr><th>JNI GPU Arch (from build info)</th><td>{v}</td></tr>),
                    monitoredDiskDevice.map(v =>
                      <tr><th>Monitored Disk Device (spark.local.dir)</th><td>{v}</td></tr>),
                    diskWriteBw.map(v =>
                      <tr><th>Disk Write Bandwidth (bytes/s)</th><td>{v}</td></tr>),
                    diskReadBw.map(v =>
                      <tr><th>Disk Read Bandwidth (bytes/s)</th><td>{v}</td></tr>)
                  ).flatten
                }
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
                    Disk / Network
                    <a href="#" class="rapids-section-toggle" data-target="section-disk-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-disk-body">
                    <div id="disk-io-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="disk-util-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
                    <div id="net-io-chart" style="width: 100%; height: 250px; margin-top: 10px;"></div>
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
                    GPU
                    <a href="#" class="rapids-section-toggle" data-target="section-gpu-body"
                       style="margin-left: 8px; font-size: 11px;">[hide]</a>
                  </h5>
                  <div id="section-gpu-body">
                    <div id="gpu-chart" style="width: 100%; height: 260px; margin-top: 10px;"></div>
                    <div id="gpu-util-chart" style="width: 100%; height: 240px; margin-top: 10px;"></div>
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

            function makeTooltipFormatter() {
              return function() {
                var x = this.x;
                var active = getActiveStagesAt(x);
                var header = Highcharts.dateFormat('%Y-%m-%d %H:%M:%S', x);
                var s = '<span style="font-size:10px;">' + header + '</span>';
                if (this.points && this.points.length) {
                  this.points.forEach(function(p) {
                    s += '<br/><span style="color:' + p.color +
                      '">\u25CF</span> ' + escapeHtml(p.series.name) +
                      ': <b>' + p.y + '</b>';
                  });
                } else if (this.point) {
                  s += '<br/><span style="color:' + this.point.color +
                    '">\u25CF</span> ' + escapeHtml(this.point.series.name) +
                    ': <b>' + this.point.y + '</b>';
                }
                if (active.length > 0) {
                  var stageLinks = active.map(function(st) {
                    var id = st.id;
                    var href = '/history/' + getAppId() + '/stages/stage/?id=' + id;
                    return '<a href="' + href + '">' + escapeHtml(id) + '</a>';
                  }).join(', ');
                  s += '<br/><span style="font-size:10px;">Stages: ' + stageLinks + '</span>';
                }
                return s;
              };
            }

            var commonTooltipFormatter = makeTooltipFormatter();

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
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                },
                area: {
                  stacking: 'normal',
                  marker: { enabled: false }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
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
              title: { text: 'JVM Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(jvmChart);
              jvmChart.redraw();
            }

            // Off-heap chart: pinned and pageable
            var offheapChart = Highcharts.chart('offheap-chart', {
              title: { text: 'Off-heap Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(offheapChart);
              offheapChart.redraw();
            }

            // System memory chart: system used and free
            var sysMemChart = Highcharts.chart('sys-mem-chart', {
              title: { text: 'System Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(sysMemChart);
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
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(cpuChart);
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
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(gpuTasksChart);
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
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(retriesChart);
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
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(diskIoChart);
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
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(diskUtilChart);
              diskUtilChart.redraw();
            }

            // Network IO chart: netReadBytes, netWriteBytes (per sample interval)
            var netIoChart = Highcharts.chart('net-io-chart', {
              title: { text: 'Network IO (sample interval bytes)' },
              xAxis: { type: 'datetime' },
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
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(netIoChart);
              netIoChart.redraw();
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
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(spillTimeChart);
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
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(spillBytesChart);
              spillBytesChart.redraw();
            }

            // GPU memory chart
            var gpuChart = Highcharts.chart('gpu-chart', {
              title: { text: 'GPU Memory' },
              xAxis: { type: 'datetime' },
              yAxis: { title: { text: 'Bytes' } },
              legend: { enabled: true },
              plotOptions: {
                series: {
                  point: {
                    events: {
                      click: function () {
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(gpuChart);
              gpuChart.redraw();
            }

            // GPU SM utilization chart
            var gpuUtilChart = Highcharts.chart('gpu-util-chart', {
              title: { text: 'GPU SM Utilization' },
              xAxis: { type: 'datetime' },
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
                        var chart = this.series.chart;
                        if (chart.pinnedX === this.x) {
                          chart.pinnedX = null;
                          chart.tooltip.hide();
                        } else {
                          chart.pinnedX = this.x;
                          chart.tooltip.refresh(this);
                        }
                      }
                    }
                  }
                }
              },
              tooltip: {
                shared: true,
                useHTML: true,
                formatter: commonTooltipFormatter
              },
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
              enableTooltipPinning(gpuUtilChart);
              gpuUtilChart.redraw();
            }
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
        val start = data
          .get("stageStartTime")
          .flatMap(s => Try(s.toLong).toOption)
          .getOrElse(e.timestamp)
        val end = data
          .get("stageEndTime")
          .flatMap(s => Try(s.toLong).toOption)
          .getOrElse(e.timestamp)
        (id, name, start, end)
      }
    }

    def escapeJsonString(s: String): String =
      s.replace("\\", "\\\\").replace("\"", "\\\"")

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
      case (id, name, start, end) =>
        s"""{"id":$id,"name":"${escapeJsonString(name)}","startTime":$start,"endTime":$end}"""
    }.mkString(",")
    val selectedExecutorJson = targetExecIdOpt.map(id => s""""selectedExecutor":"$id",""")
      .getOrElse("")

    s"""{"executors": [$execIdsJson], $selectedExecutorJson "seriesByExecutor": {$seriesByExecutorEntries}, "stages": [$stagesJson]}"""
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

