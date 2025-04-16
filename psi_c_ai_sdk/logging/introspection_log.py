# psi_c_ai_sdk/logging/introspection_log.py

"""Introspection Logger for ΨC System.

This module provides tools for logging introspection events and managing traces
of cognitive processes within the ΨC system. It enables detailed monitoring of
memory operations, coherence scoring, and ΨC state changes.
"""

import time
import json
import uuid
import enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class EventType(enum.Enum):
    """Types of events that can be logged."""
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGE = "config_change"
    
    # Memory events
    MEMORY_ADD = "memory_add"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    
    # ΨC events
    PSI_C_STATE_CHANGE = "psi_c_state_change"
    COHERENCE_SCORE = "coherence_score"
    REFLECTION_CYCLE = "reflection_cycle"
    RECURSIVE_DEPTH_LIMIT = "recursive_depth_limit"
    COLLAPSE_EVENT = "collapse_event"
    
    # Error events
    ERROR = "error"
    WARNING = "warning"


class Event:
    """Represents a single introspection event."""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        """Initialize an introspection event.
        
        Args:
            event_type: Type of the event
            data: Event specific data
            timestamp: Event timestamp (defaults to current time)
        """
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create an event from dictionary data."""
        event = cls(
            event_type=EventType(data["event_type"]),
            data=data["data"],
            timestamp=data["timestamp"]
        )
        event.id = data["id"]
        return event


class Trace:
    """A collection of related events forming a cognitive process trace."""
    
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a new trace.
        
        Args:
            name: Name of the trace
            metadata: Additional information about the trace
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.metadata = metadata or {}
        self.events: List[Event] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
    
    def add_event(self, event: Event) -> None:
        """Add an event to the trace.
        
        Args:
            event: The event to add
        """
        self.events.append(event)
    
    def end(self) -> None:
        """Mark the trace as complete."""
        self.end_time = time.time()
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate the duration of the trace in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "events": [event.to_dict() for event in self.events]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trace':
        """Create a trace from dictionary data."""
        trace = cls(name=data["name"], metadata=data["metadata"])
        trace.id = data["id"]
        trace.start_time = data["start_time"]
        trace.end_time = data["end_time"]
        
        for event_data in data["events"]:
            trace.events.append(Event.from_dict(event_data))
        
        return trace


class IntrospectionLogger:
    """Main logger class for introspection events."""
    
    def __init__(self, max_traces: int = 100):
        """Initialize the introspection logger.
        
        Args:
            max_traces: Maximum number of traces to keep in memory
        """
        self.traces: Dict[str, Trace] = {}
        self.current_trace: Optional[Trace] = None
        self.max_traces = max_traces
    
    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Trace:
        """Start a new trace.
        
        Args:
            name: Name of the trace
            metadata: Additional information about the trace
            
        Returns:
            The newly created trace
        """
        trace = Trace(name=name, metadata=metadata)
        self.traces[trace.id] = trace
        self.current_trace = trace
        
        # Manage trace limit
        if len(self.traces) > self.max_traces:
            oldest_id = min(self.traces.keys(), key=lambda k: self.traces[k].start_time)
            del self.traces[oldest_id]
            
        return trace
    
    def end_trace(self, trace_id: Optional[str] = None) -> None:
        """End the specified trace or the current trace if none specified.
        
        Args:
            trace_id: ID of the trace to end
        """
        if trace_id:
            if trace_id in self.traces:
                self.traces[trace_id].end()
                if self.current_trace and self.current_trace.id == trace_id:
                    self.current_trace = None
        elif self.current_trace:
            self.current_trace.end()
            self.current_trace = None
    
    def log_event(
        self, 
        event_type: EventType, 
        data: Dict[str, Any], 
        trace_id: Optional[str] = None
    ) -> Event:
        """Log an introspection event.
        
        Args:
            event_type: Type of the event
            data: Event specific data
            trace_id: ID of trace to add the event to (defaults to current)
            
        Returns:
            The created event
        """
        event = Event(event_type=event_type, data=data)
        
        # Add to specified trace or current trace
        if trace_id and trace_id in self.traces:
            self.traces[trace_id].add_event(event)
        elif self.current_trace:
            self.current_trace.add_event(event)
            
        return event
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by its ID.
        
        Args:
            trace_id: ID of the trace to retrieve
            
        Returns:
            The trace if found, None otherwise
        """
        return self.traces.get(trace_id)
    
    def get_traces(self) -> List[Trace]:
        """Get all traces in chronological order.
        
        Returns:
            List of traces
        """
        return sorted(self.traces.values(), key=lambda t: t.start_time)
    
    def export_traces(self, filepath: str) -> None:
        """Export all traces to a JSON file.
        
        Args:
            filepath: Path to save the traces
        """
        traces_data = {
            trace_id: trace.to_dict() 
            for trace_id, trace in self.traces.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(traces_data, f, indent=2)
    
    def import_traces(self, filepath: str) -> None:
        """Import traces from a JSON file.
        
        Args:
            filepath: Path to load traces from
        """
        with open(filepath, 'r') as f:
            traces_data = json.load(f)
        
        for trace_id, trace_data in traces_data.items():
            self.traces[trace_id] = Trace.from_dict(trace_data)
    
    def clear_traces(self) -> None:
        """Clear all stored traces."""
        self.traces = {}
        self.current_trace = None
