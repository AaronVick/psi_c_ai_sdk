"""
Trace Consumer Interface

This module defines interfaces and implementations for consuming trace events
from the introspection logger, allowing for visualization, analysis, and
export of cognitive process traces.
"""

import abc
from typing import List, Dict, Any, Optional, Set, Union, Callable
import json
import time
import datetime
from pathlib import Path

from .introspection_log import TraceEvent, EventType, get_logger, IntrospectionLogger


class TraceConsumer(abc.ABC):
    """
    Base abstract class for trace consumers.
    
    Trace consumers process trace events from the introspection logger,
    allowing for visualization, analysis, and export of cognitive traces.
    """
    
    @abc.abstractmethod
    def consume_event(self, event: TraceEvent) -> None:
        """
        Process a single trace event.
        
        Args:
            event: The trace event to process
        """
        pass
    
    @abc.abstractmethod
    def consume_events(self, events: List[TraceEvent]) -> None:
        """
        Process multiple trace events.
        
        Args:
            events: The trace events to process
        """
        pass
    
    @abc.abstractmethod
    def get_trace_summary(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the consumed trace(s).
        
        Args:
            trace_id: Optional trace ID to filter by
            
        Returns:
            A summary of the trace(s)
        """
        pass
    
    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all consumed traces."""
        pass


class JsonFileTraceConsumer(TraceConsumer):
    """
    Trace consumer that writes trace events to a JSON file.
    """
    
    def __init__(self, output_path: Union[str, Path], append: bool = False):
        """
        Initialize a JSON file trace consumer.
        
        Args:
            output_path: Path to the output JSON file
            append: Whether to append to the file (True) or overwrite it (False)
        """
        self.output_path = Path(output_path)
        self.append = append
        self.consumed_events: List[Dict[str, Any]] = []
        
        # Create parent directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the file if not appending
        if not append and not self.output_path.exists():
            with open(self.output_path, 'w') as f:
                json.dump({
                    "trace_version": "1.0",
                    "generated_at": datetime.datetime.now().isoformat(),
                    "events": []
                }, f)
    
    def consume_event(self, event: TraceEvent) -> None:
        """
        Process a single trace event by writing it to the JSON file.
        
        Args:
            event: The trace event to process
        """
        event_dict = event.to_dict()
        self.consumed_events.append(event_dict)
        
        if self.append:
            with open(self.output_path, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
        else:
            # Read existing data
            with open(self.output_path, 'r') as f:
                data = json.load(f)
            
            # Update data
            data["events"].append(event_dict)
            data["event_count"] = len(data["events"])
            data["last_updated"] = datetime.datetime.now().isoformat()
            
            # Write updated data
            with open(self.output_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def consume_events(self, events: List[TraceEvent]) -> None:
        """
        Process multiple trace events.
        
        Args:
            events: The trace events to process
        """
        for event in events:
            self.consume_event(event)
    
    def get_trace_summary(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the consumed traces.
        
        Args:
            trace_id: Optional trace ID to filter by
            
        Returns:
            A summary of the traces
        """
        if trace_id:
            events = [e for e in self.consumed_events if e.get("trace_id") == trace_id]
        else:
            events = self.consumed_events
        
        event_types = {}
        for event in events:
            event_type = event.get("event_type", "UNKNOWN")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": len(events),
            "event_types": event_types,
            "trace_ids": list(set(e.get("trace_id") for e in events)),
            "time_range": {
                "start": min((e.get("timestamp", 0) for e in events), default=0),
                "end": max((e.get("timestamp", 0) for e in events), default=0)
            }
        }
    
    def clear(self) -> None:
        """Clear all consumed traces."""
        self.consumed_events = []
        
        if not self.append:
            with open(self.output_path, 'w') as f:
                json.dump({
                    "trace_version": "1.0",
                    "generated_at": datetime.datetime.now().isoformat(),
                    "events": []
                }, f, indent=2)


class MemoryTraceConsumer(TraceConsumer):
    """
    Trace consumer that keeps trace events in memory for analysis.
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize a memory trace consumer.
        
        Args:
            max_events: Maximum number of events to keep in memory
        """
        self.max_events = max_events
        self.events: List[TraceEvent] = []
        self.trace_ids: Set[str] = set()
        self.event_types: Dict[EventType, int] = {}
        
    def consume_event(self, event: TraceEvent) -> None:
        """
        Process a single trace event.
        
        Args:
            event: The trace event to process
        """
        self.events.append(event)
        self.trace_ids.add(event.trace_id)
        
        # Update event type counts
        self.event_types[event.event_type] = self.event_types.get(event.event_type, 0) + 1
        
        # Trim events if necessary
        if len(self.events) > self.max_events:
            removed = self.events.pop(0)
            
            # Update trace IDs and event types
            self._recalculate_stats()
    
    def consume_events(self, events: List[TraceEvent]) -> None:
        """
        Process multiple trace events.
        
        Args:
            events: The trace events to process
        """
        for event in events:
            self.consume_event(event)
    
    def get_trace_summary(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the consumed traces.
        
        Args:
            trace_id: Optional trace ID to filter by
            
        Returns:
            A summary of the traces
        """
        if trace_id:
            filtered_events = [e for e in self.events if e.trace_id == trace_id]
        else:
            filtered_events = self.events
        
        if not filtered_events:
            return {
                "total_events": 0,
                "event_types": {},
                "trace_ids": [],
                "time_range": {"start": 0, "end": 0}
            }
        
        event_types = {}
        for event in filtered_events:
            event_type = event.event_type.name
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": len(filtered_events),
            "event_types": event_types,
            "trace_ids": list(set(e.trace_id for e in filtered_events)),
            "time_range": {
                "start": min(e.timestamp for e in filtered_events),
                "end": max(e.timestamp for e in filtered_events)
            }
        }
    
    def get_events_by_type(self, event_type: EventType) -> List[TraceEvent]:
        """
        Get all events of the specified type.
        
        Args:
            event_type: Event type to filter by
            
        Returns:
            List of events of the specified type
        """
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_by_trace(self, trace_id: str) -> List[TraceEvent]:
        """
        Get all events with the specified trace ID.
        
        Args:
            trace_id: Trace ID to filter by
            
        Returns:
            List of events with the specified trace ID
        """
        return [event for event in self.events if event.trace_id == trace_id]
    
    def get_events_by_time_range(self, start_time: float, end_time: float) -> List[TraceEvent]:
        """
        Get all events within the specified time range.
        
        Args:
            start_time: Start time in seconds since epoch
            end_time: End time in seconds since epoch
            
        Returns:
            List of events within the specified time range
        """
        return [
            event for event in self.events
            if start_time <= event.timestamp <= end_time
        ]
    
    def clear(self) -> None:
        """Clear all consumed traces."""
        self.events = []
        self.trace_ids = set()
        self.event_types = {}
    
    def _recalculate_stats(self) -> None:
        """Recalculate trace IDs and event type counts."""
        self.trace_ids = set()
        self.event_types = {}
        
        for event in self.events:
            self.trace_ids.add(event.trace_id)
            self.event_types[event.event_type] = self.event_types.get(event.event_type, 0) + 1


class TraceVisualizer:
    """
    Base class for trace visualizers that can render trace data.
    
    This class provides methods to transform trace events into various visual
    representations for analysis and debugging.
    """
    
    def __init__(self, logger: Optional[IntrospectionLogger] = None):
        """
        Initialize a trace visualizer.
        
        Args:
            logger: Optional introspection logger to use (uses global logger if None)
        """
        self.logger = logger or get_logger()
    
    def generate_timeline(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a timeline visualization data for the specified trace.
        
        Args:
            trace_id: Optional trace ID to filter by
            
        Returns:
            Timeline data for visualization
        """
        if trace_id:
            events = self.logger.get_trace_events(trace_id)
        else:
            events = self.logger.events
        
        if not events:
            return {"events": []}
        
        # Sort events by timestamp
        events = sorted(events, key=lambda e: e.timestamp)
        
        # Group events by type
        grouped_events = {}
        for event in events:
            event_type = event.event_type.name
            if event_type not in grouped_events:
                grouped_events[event_type] = []
            
            grouped_events[event_type].append({
                "id": event.event_id,
                "time": event.timestamp,
                "data": event.data,
                "parent_id": event.parent_id
            })
        
        # Calculate time ranges
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration = end_time - start_time
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "events": [
                {
                    "type": event_type,
                    "count": len(event_list),
                    "items": event_list
                }
                for event_type, event_list in grouped_events.items()
            ]
        }
    
    def generate_trace_graph(self, trace_id: str) -> Dict[str, Any]:
        """
        Generate a graph visualization of event relationships for the specified trace.
        
        Args:
            trace_id: Trace ID to visualize
            
        Returns:
            Graph data for visualization
        """
        events = self.logger.get_trace_events(trace_id)
        
        if not events:
            return {"nodes": [], "edges": []}
        
        nodes = []
        edges = []
        event_map = {}
        
        # Create nodes
        for i, event in enumerate(events):
            event_map[event.event_id] = i
            nodes.append({
                "id": i,
                "event_id": event.event_id,
                "type": event.event_type.name,
                "time": event.timestamp,
                "data": event.data
            })
        
        # Create edges
        for event in events:
            if event.parent_id and event.parent_id in event_map:
                edges.append({
                    "source": event_map[event.parent_id],
                    "target": event_map[event.event_id],
                    "type": "parent_child"
                })
        
        # Create sequential edges
        events = sorted(events, key=lambda e: e.timestamp)
        for i in range(len(events) - 1):
            if events[i].trace_id == events[i+1].trace_id:
                edges.append({
                    "source": event_map[events[i].event_id],
                    "target": event_map[events[i+1].event_id],
                    "type": "sequential"
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def generate_summary_report(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a summary report of the specified trace.
        
        Args:
            trace_id: Optional trace ID to filter by
            
        Returns:
            Summary report data
        """
        if trace_id:
            events = self.logger.get_trace_events(trace_id)
        else:
            events = self.logger.events
        
        if not events:
            return {
                "event_count": 0,
                "event_types": {},
                "time_range": {"start": 0, "end": 0, "duration": 0},
                "trace_ids": []
            }
        
        # Count events by type
        event_types = {}
        for event in events:
            event_type = event.event_type.name
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        # Get time range
        timestamps = [event.timestamp for event in events]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        # Get unique trace IDs
        trace_ids = list(set(event.trace_id for event in events))
        
        # Get PsiC state changes
        psic_state_changes = [
            event for event in events 
            if event.event_type == EventType.PSIC_STATE_CHANGE
        ]
        
        # Get reflection insights
        reflection_insights = [
            event for event in events
            if event.event_type == EventType.REFLECTION_INSIGHT
        ]
        
        return {
            "event_count": len(events),
            "event_types": event_types,
            "time_range": {
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time
            },
            "trace_ids": trace_ids,
            "psic_state_changes": [
                {
                    "time": event.timestamp,
                    "old_state": event.data.get("old_state"),
                    "new_state": event.data.get("new_state"),
                    "activation_score": event.data.get("activation_score")
                }
                for event in psic_state_changes
            ],
            "reflection_insights": [
                {
                    "time": event.timestamp,
                    "insight": event.data.get("insight"),
                    "confidence": event.data.get("confidence")
                }
                for event in reflection_insights
            ]
        }


# Register trace consumers with a logger
def register_consumer(logger: IntrospectionLogger, consumer: TraceConsumer) -> Callable:
    """
    Register a trace consumer with an introspection logger.
    
    This function sets up a trace consumer to receive events from the logger
    in real-time. It returns a function that can be called to unregister the consumer.
    
    Args:
        logger: The introspection logger to register with
        consumer: The trace consumer to register
        
    Returns:
        Function to unregister the consumer
    """
    
    # Define callback to consume events
    def event_callback(event: TraceEvent) -> None:
        consumer.consume_event(event)
    
    # Store original log_event method
    original_log_event = logger.log_event
    
    # Override log_event method to call the callback
    def wrapped_log_event(event_type, data, trace_id=None, parent_id=None):
        event_id = original_log_event(event_type, data, trace_id, parent_id)
        
        # Find the event and pass it to the consumer
        for event in logger.events:
            if hasattr(event, 'event_id') and event.event_id == event_id:
                event_callback(event)
                break
        
        return event_id
    
    # Apply the wrapped method
    logger.log_event = wrapped_log_event
    
    # Define unregister function
    def unregister():
        logger.log_event = original_log_event
    
    return unregister 