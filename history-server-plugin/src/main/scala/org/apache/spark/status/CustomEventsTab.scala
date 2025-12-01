package org.apache.spark.status

import org.apache.spark.ui.{SparkUI, SparkUITab, UIUtils, WebUIPage}

import javax.servlet.http.HttpServletRequest
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
            <div id="timeline-chart" style="width: 100%; height: 400px;">
              <p>Timeline visualization would be rendered here with JavaScript</p>
            </div>
          </div>
        </div>
      </div>

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
          });

          function fetchTimelineData() {
            $.getJSON('/history/' + getAppId() + '/customevents/api/timeline', function(data) {
              renderTimeline(data);
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
        """)}
      </script>

    UIUtils.headerSparkPage(request, "Custom Events", content ++ scriptContent, parent)
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

