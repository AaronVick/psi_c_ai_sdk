"""
Trace Viewer Web Interface

This module provides a web-based interface for visualizing and exploring
trace events from the introspection logger. It renders timelines, graphs,
and summary reports of cognitive processes.
"""

import os
import json
import webbrowser
import tempfile
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .introspection_log import IntrospectionLogger, EventType, get_logger
from .trace_consumer import TraceConsumer, MemoryTraceConsumer, JsonFileTraceConsumer, TraceVisualizer

# HTML template for the trace viewer
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ΨC Trace Viewer</title>
    <style>
        :root {
            --primary-color: #4a86e8;
            --secondary-color: #6aa84f;
            --accent-color: #f44336;
            --bg-color: #f9f9f9;
            --card-bg: #ffffff;
            --text-color: #333333;
            --border-color: #dddddd;
            --header-bg: #333333;
            --header-text: #ffffff;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        
        header {
            background-color: var(--header-bg);
            color: var(--header-text);
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        header h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }
        
        .tab {
            padding: 0.5rem 1rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }
        
        .tab.active {
            border-bottom: 2px solid var(--primary-color);
            font-weight: bold;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .timeline {
            position: relative;
            margin: 2rem 0;
            padding-top: 2rem;
        }
        
        .timeline-ruler {
            height: 2px;
            background-color: var(--border-color);
            position: relative;
        }
        
        .timeline-marker {
            position: absolute;
            height: 10px;
            width: 10px;
            border-radius: 50%;
            background-color: var(--primary-color);
            transform: translate(-50%, -50%);
            cursor: pointer;
        }
        
        .timeline-marker:hover {
            background-color: var(--accent-color);
        }
        
        .timeline-event {
            position: absolute;
            top: 15px;
            padding: 0.25rem;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 0.8rem;
            transform: translateX(-50%);
            white-space: nowrap;
            cursor: pointer;
        }
        
        .timeline-event:hover {
            background-color: #f0f0f0;
        }
        
        .timeline-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: #666;
        }
        
        .event-details {
            display: none;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 1rem;
            margin-top: 1rem;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .summary-section {
            margin-bottom: 1.5rem;
        }
        
        .summary-section h3 {
            margin-top: 0;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        
        .stat-item .label {
            font-weight: bold;
        }
        
        .event-type-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        
        .event-type-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            background-color: #e0e0e0;
            cursor: pointer;
        }
        
        /* Graph styles */
        #graph-container {
            width: 100%;
            height: 600px;
            border: 1px solid var(--border-color);
            margin-top: 1rem;
        }
        
        .node {
            cursor: pointer;
            stroke: #fff;
            stroke-width: 1.5px;
        }
        
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        
        .event-filter {
            margin-bottom: 1rem;
        }
        
        .event-filter label {
            margin-right: 0.5rem;
        }
        
        /* Color coding for event types */
        .event-SYSTEM_STARTUP { background-color: #4285f4; color: white; }
        .event-MEMORY_ADDED { background-color: #34a853; color: white; }
        .event-MEMORY_ACCESSED { background-color: #fbbc05; color: black; }
        .event-SCHEMA_UPDATED { background-color: #ea4335; color: white; }
        .event-COHERENCE_CALCULATED { background-color: #9c27b0; color: white; }
        .event-CONTRADICTION_DETECTED { background-color: #f44336; color: white; }
        .event-PSIC_ACTIVATION { background-color: #2196f3; color: white; }
        .event-PSIC_STATE_CHANGE { background-color: #673ab7; color: white; }
        .event-REFLECTION_INSIGHT { background-color: #ff9800; color: black; }
        .event-COLLAPSE_DECISION { background-color: #795548; color: white; }
        .event-USER_COMMAND { background-color: #607d8b; color: white; }
        .event-RECURSION_LIMIT { background-color: #e91e63; color: white; }
    </style>
</head>
<body>
    <header>
        <h1>ΨC Trace Viewer</h1>
        <div>
            <span id="trace-id-display"></span>
            <span id="event-count-display"></span>
        </div>
    </header>
    
    <div class="container">
        <div class="card">
            <div class="tabs">
                <div class="tab active" data-tab="timeline">Timeline</div>
                <div class="tab" data-tab="graph">Trace Graph</div>
                <div class="tab" data-tab="summary">Summary</div>
                <div class="tab" data-tab="raw">Raw Data</div>
            </div>
            
            <div class="tab-content active" id="timeline-tab">
                <div class="event-filter">
                    <label>Filter by event type:</label>
                    <select id="event-type-filter">
                        <option value="all">All Events</option>
                        <!-- Event type options will be populated dynamically -->
                    </select>
                </div>
                
                <div class="timeline">
                    <div class="timeline-ruler" id="timeline-ruler"></div>
                    <div class="timeline-labels">
                        <span id="start-time"></span>
                        <span id="end-time"></span>
                    </div>
                </div>
                
                <div class="event-details" id="event-details">
                    <h3>Event Details</h3>
                    <pre id="event-details-content"></pre>
                </div>
            </div>
            
            <div class="tab-content" id="graph-tab">
                <div id="graph-container"></div>
            </div>
            
            <div class="tab-content" id="summary-tab">
                <div class="summary-section">
                    <h3>Overview</h3>
                    <div class="stat-item">
                        <span class="label">Total Events:</span>
                        <span id="total-events">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Time Range:</span>
                        <span id="time-range">N/A</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Duration:</span>
                        <span id="duration">N/A</span>
                    </div>
                </div>
                
                <div class="summary-section">
                    <h3>Event Types</h3>
                    <div class="event-type-list" id="event-type-list">
                        <!-- Will be populated dynamically -->
                    </div>
                </div>
                
                <div class="summary-section">
                    <h3>ΨC State Changes</h3>
                    <div id="psic-state-changes">
                        <!-- Will be populated dynamically -->
                    </div>
                </div>
                
                <div class="summary-section">
                    <h3>Reflection Insights</h3>
                    <div id="reflection-insights">
                        <!-- Will be populated dynamically -->
                    </div>
                </div>
            </div>
            
            <div class="tab-content" id="raw-tab">
                <pre id="raw-data"></pre>
            </div>
        </div>
    </div>
    
    <script>
        // Will be replaced with actual trace data
        const traceData = __TRACE_DATA_PLACEHOLDER__;
        
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Deactivate all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    
                    // Activate selected tab
                    tab.classList.add('active');
                    
                    // Hide all tab contents
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Show selected tab content
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(tabId + '-tab').classList.add('active');
                    
                    // Initialize specific tab if needed
                    if (tabId === 'graph' && !graphInitialized) {
                        initializeGraph();
                        graphInitialized = true;
                    }
                });
            });
            
            // Display trace info
            if (traceData.trace_ids && traceData.trace_ids.length > 0) {
                document.getElementById('trace-id-display').textContent = 'Trace: ' + traceData.trace_ids[0];
            }
            document.getElementById('event-count-display').textContent = traceData.event_count + ' events';
            
            // Initialize timeline
            initializeTimeline();
            
            // Initialize summary
            initializeSummary();
            
            // Initialize raw data
            document.getElementById('raw-data').textContent = JSON.stringify(traceData, null, 2);
        });
        
        let graphInitialized = false;
        
        function initializeTimeline() {
            const timelineRuler = document.getElementById('timeline-ruler');
            const startTimeEl = document.getElementById('start-time');
            const endTimeEl = document.getElementById('end-time');
            const eventTypeFilter = document.getElementById('event-type-filter');
            
            // Clear timeline
            timelineRuler.innerHTML = '';
            
            // Add event type options to filter
            const eventTypes = Object.keys(traceData.event_types || {});
            eventTypes.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type + ' (' + traceData.event_types[type] + ')';
                eventTypeFilter.appendChild(option);
            });
            
            // Event filter change handler
            eventTypeFilter.addEventListener('change', () => {
                renderTimeline(eventTypeFilter.value);
            });
            
            // Format timestamps
            if (traceData.time_range) {
                const startDate = new Date(traceData.time_range.start * 1000);
                const endDate = new Date(traceData.time_range.end * 1000);
                
                startTimeEl.textContent = formatDate(startDate);
                endTimeEl.textContent = formatDate(endDate);
            }
            
            // Render timeline
            renderTimeline('all');
        }
        
        function renderTimeline(eventTypeFilter) {
            const timelineRuler = document.getElementById('timeline-ruler');
            timelineRuler.innerHTML = '';
            
            // Get all events
            let events = [];
            if (traceData.events) {
                traceData.events.forEach(eventGroup => {
                    if (eventTypeFilter === 'all' || eventGroup.type === eventTypeFilter) {
                        events = events.concat(eventGroup.items.map(item => {
                            return {
                                ...item,
                                type: eventGroup.type
                            };
                        }));
                    }
                });
            }
            
            // Sort events by time
            events.sort((a, b) => a.time - b.time);
            
            if (events.length === 0) {
                timelineRuler.innerHTML = '<div style="text-align: center; padding: 2rem;">No events to display</div>';
                return;
            }
            
            // Calculate positions
            const startTime = traceData.time_range.start;
            const endTime = traceData.time_range.end;
            const timeRange = endTime - startTime;
            
            // Create timeline events
            events.forEach((event, index) => {
                const position = ((event.time - startTime) / timeRange) * 100;
                
                // Create marker
                const marker = document.createElement('div');
                marker.className = 'timeline-marker';
                marker.style.left = position + '%';
                marker.setAttribute('data-event-index', index);
                marker.title = event.type;
                timelineRuler.appendChild(marker);
                
                // Create event label
                if (index % 5 === 0) { // Only show every 5th label to avoid overcrowding
                    const label = document.createElement('div');
                    label.className = 'timeline-event event-' + event.type;
                    label.textContent = event.type;
                    label.style.left = position + '%';
                    label.setAttribute('data-event-index', index);
                    timelineRuler.appendChild(label);
                }
                
                // Add click handler
                marker.addEventListener('click', () => showEventDetails(event));
            });
        }
        
        function showEventDetails(event) {
            const eventDetails = document.getElementById('event-details');
            const eventDetailsContent = document.getElementById('event-details-content');
            
            eventDetailsContent.textContent = JSON.stringify(event, null, 2);
            eventDetails.style.display = 'block';
        }
        
        function initializeSummary() {
            // Total events
            document.getElementById('total-events').textContent = traceData.event_count;
            
            // Time range
            if (traceData.time_range) {
                const startDate = new Date(traceData.time_range.start * 1000);
                const endDate = new Date(traceData.time_range.end * 1000);
                
                document.getElementById('time-range').textContent = 
                    formatDate(startDate) + ' to ' + formatDate(endDate);
                
                // Duration
                const duration = traceData.time_range.duration;
                document.getElementById('duration').textContent = formatDuration(duration);
            }
            
            // Event types
            const eventTypeList = document.getElementById('event-type-list');
            eventTypeList.innerHTML = '';
            
            if (traceData.event_types) {
                Object.entries(traceData.event_types).forEach(([type, count]) => {
                    const badge = document.createElement('div');
                    badge.className = 'event-type-badge event-' + type;
                    badge.textContent = type + ' (' + count + ')';
                    eventTypeList.appendChild(badge);
                });
            }
            
            // PsiC state changes
            const stateChangesEl = document.getElementById('psic-state-changes');
            stateChangesEl.innerHTML = '';
            
            if (traceData.psic_state_changes && traceData.psic_state_changes.length > 0) {
                const ul = document.createElement('ul');
                traceData.psic_state_changes.forEach(change => {
                    const li = document.createElement('li');
                    li.textContent = `${formatDate(new Date(change.time * 1000))}: ${change.old_state} → ${change.new_state} (Score: ${change.activation_score.toFixed(3)})`;
                    ul.appendChild(li);
                });
                stateChangesEl.appendChild(ul);
            } else {
                stateChangesEl.textContent = 'No state changes recorded';
            }
            
            // Reflection insights
            const insightsEl = document.getElementById('reflection-insights');
            insightsEl.innerHTML = '';
            
            if (traceData.reflection_insights && traceData.reflection_insights.length > 0) {
                const ul = document.createElement('ul');
                traceData.reflection_insights.forEach(insight => {
                    const li = document.createElement('li');
                    li.textContent = `${formatDate(new Date(insight.time * 1000))}: ${insight.insight} (Confidence: ${insight.confidence.toFixed(2)})`;
                    ul.appendChild(li);
                });
                insightsEl.appendChild(ul);
            } else {
                insightsEl.textContent = 'No reflection insights recorded';
            }
        }
        
        function initializeGraph() {
            // This would use D3.js or a similar library to render a graph
            // of event relationships. For simplicity, we're just showing a placeholder.
            const graphContainer = document.getElementById('graph-container');
            graphContainer.innerHTML = 'Graph visualization requires D3.js library. Not included in this basic viewer.';
        }
        
        function formatDate(date) {
            return date.toLocaleTimeString();
        }
        
        function formatDuration(seconds) {
            if (seconds < 60) {
                return seconds.toFixed(2) + ' seconds';
            } else if (seconds < 3600) {
                return (seconds / 60).toFixed(2) + ' minutes';
            } else {
                return (seconds / 3600).toFixed(2) + ' hours';
            }
        }
    </script>
</body>
</html>
"""


class TraceViewerWeb:
    """
    Web-based trace viewer for visualizing and exploring trace events.
    
    This class generates an HTML file with interactive visualizations of
    trace events, including timelines, graphs, and summary reports.
    """
    
    def __init__(self, logger: Optional[IntrospectionLogger] = None):
        """
        Initialize a web-based trace viewer.
        
        Args:
            logger: Optional introspection logger to use (uses global logger if None)
        """
        self.logger = logger or get_logger()
        self.visualizer = TraceVisualizer(self.logger)
    
    def generate_html(self, trace_id: Optional[str] = None) -> str:
        """
        Generate HTML for the trace viewer.
        
        Args:
            trace_id: Optional trace ID to filter events by
            
        Returns:
            HTML content for the trace viewer
        """
        # Generate trace data
        summary_data = self.visualizer.generate_summary_report(trace_id)
        timeline_data = self.visualizer.generate_timeline(trace_id)
        
        if trace_id:
            graph_data = self.visualizer.generate_trace_graph(trace_id)
            trace_data = {
                **summary_data,
                "events": timeline_data.get("events", []),
                "graph": graph_data
            }
        else:
            trace_data = {
                **summary_data,
                "events": timeline_data.get("events", [])
            }
        
        # Insert trace data into HTML template
        html = HTML_TEMPLATE.replace('__TRACE_DATA_PLACEHOLDER__', json.dumps(trace_data))
        return html
    
    def view_trace(self, trace_id: Optional[str] = None, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate HTML and open it in a web browser.
        
        Args:
            trace_id: Optional trace ID to filter events by
            output_path: Optional path to save the HTML file
            
        Returns:
            Path to the generated HTML file
        """
        html = self.generate_html(trace_id)
        
        # Determine output path
        if output_path:
            html_path = Path(output_path)
        else:
            # Create temporary file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trace_viewer_{timestamp}.html"
            html_path = Path(tempfile.gettempdir()) / filename
        
        # Write HTML to file
        with open(html_path, 'w') as f:
            f.write(html)
        
        # Open in web browser
        webbrowser.open(f"file://{html_path.absolute()}")
        
        return str(html_path)
    
    def export_trace(self, output_path: Union[str, Path], trace_id: Optional[str] = None) -> str:
        """
        Export trace data to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            trace_id: Optional trace ID to filter events by
            
        Returns:
            Path to the generated JSON file
        """
        # Get trace data
        if trace_id:
            events = self.logger.get_trace_events(trace_id)
        else:
            events = self.logger.events
        
        # Convert to dictionaries
        event_dicts = [event.to_dict() for event in events]
        
        # Create JSON data
        json_data = {
            "trace_version": "1.0",
            "generated_at": datetime.datetime.now().isoformat(),
            "event_count": len(event_dicts),
            "events": event_dicts
        }
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return str(output_path)


def view_trace(trace_id: Optional[str] = None, logger: Optional[IntrospectionLogger] = None) -> str:
    """
    Convenient function to view a trace in a web browser.
    
    Args:
        trace_id: Optional trace ID to filter events by
        logger: Optional introspection logger to use (uses global logger if None)
        
    Returns:
        Path to the generated HTML file
    """
    viewer = TraceViewerWeb(logger)
    return viewer.view_trace(trace_id)


def export_trace(output_path: Union[str, Path], trace_id: Optional[str] = None, 
                 logger: Optional[IntrospectionLogger] = None) -> str:
    """
    Convenient function to export a trace to a JSON file.
    
    Args:
        output_path: Path to save the JSON file
        trace_id: Optional trace ID to filter events by
        logger: Optional introspection logger to use (uses global logger if None)
        
    Returns:
        Path to the generated JSON file
    """
    viewer = TraceViewerWeb(logger)
    return viewer.export_trace(output_path, trace_id) 