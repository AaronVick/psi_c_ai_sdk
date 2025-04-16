"""
Safety Trace Exporter

This module implements audit-compliant logging of all critical system updates,
providing tamper-resistant evidence trails for:
- Reflection triggers and outcomes
- Contradiction events and resolutions
- Schema snapshots and mutations
- Safety boundary crossings

The Safety Trace Exporter is designed to meet regulatory and compliance
requirements by providing verifiable, unchangeable logs of all system
operations that might have safety implications.
"""

import json
import time
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict
import threading
from enum import Enum
import uuid
import shutil

# Setup logger
logger = logging.getLogger(__name__)


class SafetyEventType(Enum):
    """Types of safety-relevant events to be logged."""
    REFLECTION_TRIGGER = "reflection_trigger"
    REFLECTION_OUTCOME = "reflection_outcome"
    CONTRADICTION_DETECTED = "contradiction_detected"
    CONTRADICTION_RESOLVED = "contradiction_resolved"
    SCHEMA_MUTATION = "schema_mutation"
    SCHEMA_SNAPSHOT = "schema_snapshot"
    SAFETY_BOUNDARY_APPROACH = "safety_boundary_approach"
    SAFETY_BOUNDARY_CROSSED = "safety_boundary_crossed"
    ALIGNMENT_DRIFT = "alignment_drift"
    TRUST_ADJUSTMENT = "trust_adjustment"
    EMERGENCY_STOP = "emergency_stop"
    MEMORY_QUARANTINE = "memory_quarantine"
    MEMORY_RELEASE = "memory_release"
    IDENTITY_SHIFT = "identity_shift"
    PSI_C_ACTIVATION = "psi_c_activation"
    PSI_C_DEACTIVATION = "psi_c_deactivation"
    ENTROPY_SPIKE = "entropy_spike"
    COHERENCE_DROP = "coherence_drop"


@dataclass
class SafetyEvent:
    """A safety-related event that should be logged for audit."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SafetyEventType = field(default=SafetyEventType.REFLECTION_TRIGGER)
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    description: str = ""
    severity: float = 0.0  # 0.0 to 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)
    hash_chain: str = ""  # Hash of previous event to create tamper-evident chain
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = asdict(self)
        result["event_type"] = self.event_type.value if isinstance(self.event_type, SafetyEventType) else self.event_type
        result["timestamp_readable"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SafetyEvent':
        """Create event from dictionary representation."""
        # Convert string event_type back to enum
        if isinstance(data.get("event_type"), str):
            data["event_type"] = SafetyEventType(data["event_type"])
        
        # Remove any extra fields not in the dataclass
        event_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in event_fields}
        
        return cls(**filtered_data)
    
    def compute_hash(self) -> str:
        """Compute cryptographic hash of this event."""
        event_data = self.to_dict()
        # Remove hash_chain as it will be computed
        event_data.pop("hash_chain", None)
        # Sort keys for consistent serialization
        serialized = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()


class SafetyTraceExporter:
    """
    Main class for exporting safety-relevant events as tamper-evident logs.
    
    This class provides:
    1. Structured event recording with severity and relationships
    2. Tamper-evident logging with hash chains
    3. Multiple export formats (JSON, CSV, encrypted)
    4. Automatic rotation and retention policies
    5. Integration with existing logging systems
    """
    
    def __init__(
        self,
        export_dir: Union[str, Path] = "safety_logs",
        rotation_size_mb: float = 10.0,
        max_log_files: int = 100,
        encryption_key: Optional[str] = None,
        enable_hash_chain: bool = True,
        auto_flush: bool = True,
        flush_interval_sec: int = 30
    ):
        """
        Initialize the safety trace exporter.
        
        Args:
            export_dir: Directory to store exported logs
            rotation_size_mb: Size in MB at which to rotate log files
            max_log_files: Maximum number of log files to keep
            encryption_key: Optional key for encrypting log files
            enable_hash_chain: Whether to create a hash chain for tamper evidence
            auto_flush: Whether to automatically flush logs to disk
            flush_interval_sec: How often to flush logs if auto_flush is enabled
        """
        self.export_dir = Path(export_dir)
        self.rotation_size_mb = rotation_size_mb
        self.max_log_files = max_log_files
        self.encryption_key = encryption_key
        self.enable_hash_chain = enable_hash_chain
        self.auto_flush = auto_flush
        self.flush_interval_sec = flush_interval_sec
        
        # Create export directory if it doesn't exist
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffer of events before they're flushed to disk
        self.event_buffer: List[SafetyEvent] = []
        
        # Keep track of the last event's hash for the hash chain
        self.last_hash: str = ""
        
        # Current log file
        self.current_log_file: Optional[Path] = None
        self._init_log_file()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Start auto-flush thread if enabled
        if self.auto_flush:
            self._start_auto_flush()
    
    def _init_log_file(self) -> None:
        """Initialize a new log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.export_dir / f"safety_trace_{timestamp}.json"
        
        # Initialize with an empty event list
        with open(self.current_log_file, 'w') as f:
            json.dump({
                "safety_trace_version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "events": []
            }, f, indent=2)
        
        logger.info(f"Initialized new safety trace log file: {self.current_log_file}")
    
    def _start_auto_flush(self) -> None:
        """Start a background thread to periodically flush events to disk."""
        def flush_worker():
            while True:
                time.sleep(self.flush_interval_sec)
                try:
                    self.flush()
                except Exception as e:
                    logger.error(f"Error in auto-flush: {e}")
        
        thread = threading.Thread(target=flush_worker, daemon=True)
        thread.start()
        logger.debug("Started auto-flush thread")
    
    def log_event(
        self,
        event_type: Union[SafetyEventType, str],
        component: str,
        description: str,
        severity: float = 0.0,
        data: Optional[Dict[str, Any]] = None,
        related_events: Optional[List[str]] = None
    ) -> str:
        """
        Log a safety-related event.
        
        Args:
            event_type: Type of safety event
            component: Component that generated the event
            description: Human-readable description of the event
            severity: Severity level from 0.0 (info) to 1.0 (critical)
            data: Additional structured data about the event
            related_events: IDs of related events
            
        Returns:
            ID of the created event
        """
        with self.lock:
            # Convert string event_type to enum if needed
            if isinstance(event_type, str):
                try:
                    event_type = SafetyEventType(event_type)
                except ValueError:
                    logger.warning(f"Unknown event type: {event_type}, using as-is")
            
            # Create the event
            event = SafetyEvent(
                event_type=event_type,
                component=component,
                description=description,
                severity=max(0.0, min(1.0, severity)),  # Clamp to [0, 1]
                data=data or {},
                related_events=related_events or []
            )
            
            # Add to hash chain if enabled
            if self.enable_hash_chain:
                if self.last_hash:
                    # Include previous hash in the chain
                    event.hash_chain = self.last_hash
                
                # Compute this event's hash
                current_hash = event.compute_hash()
                self.last_hash = current_hash
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Auto-flush if buffer gets too large
            if len(self.event_buffer) >= 100:
                self.flush()
            
            return event.event_id
    
    def log_reflection_trigger(
        self,
        trigger_reason: str,
        coherence_score: float,
        entropy_level: float,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a reflection trigger event.
        
        Args:
            trigger_reason: Why reflection was triggered
            coherence_score: Current coherence score
            entropy_level: Current entropy level
            data: Additional data
            
        Returns:
            Event ID
        """
        event_data = {
            "trigger_reason": trigger_reason,
            "coherence_score": coherence_score,
            "entropy_level": entropy_level,
            **(data or {})
        }
        
        return self.log_event(
            event_type=SafetyEventType.REFLECTION_TRIGGER,
            component="reflection_engine",
            description=f"Reflection triggered: {trigger_reason}",
            severity=0.3,  # Moderate severity
            data=event_data
        )
    
    def log_schema_mutation(
        self,
        mutation_type: str,
        mutation_id: str,
        affected_nodes: List[str],
        before_coherence: float,
        after_coherence: float,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a schema mutation event.
        
        Args:
            mutation_type: Type of mutation
            mutation_id: ID of the mutation event
            affected_nodes: Schema nodes affected
            before_coherence: Coherence score before mutation
            after_coherence: Coherence score after mutation
            data: Additional data
            
        Returns:
            Event ID
        """
        severity = 0.7 if after_coherence < before_coherence else 0.4
        
        event_data = {
            "mutation_type": mutation_type,
            "mutation_id": mutation_id,
            "affected_nodes": affected_nodes,
            "before_coherence": before_coherence,
            "after_coherence": after_coherence,
            "coherence_delta": after_coherence - before_coherence,
            **(data or {})
        }
        
        return self.log_event(
            event_type=SafetyEventType.SCHEMA_MUTATION,
            component="schema_graph",
            description=f"Schema mutation: {mutation_type} affecting {len(affected_nodes)} nodes",
            severity=severity,
            data=event_data
        )
    
    def log_schema_snapshot(
        self,
        snapshot_id: str,
        node_count: int,
        edge_count: int,
        hash_signature: str,
        reason: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a schema snapshot event.
        
        Args:
            snapshot_id: ID of the snapshot
            node_count: Number of nodes in the schema
            edge_count: Number of edges in the schema
            hash_signature: Hash signature of the schema
            reason: Reason for taking the snapshot
            data: Additional data
            
        Returns:
            Event ID
        """
        event_data = {
            "snapshot_id": snapshot_id,
            "node_count": node_count,
            "edge_count": edge_count,
            "hash_signature": hash_signature,
            "reason": reason,
            **(data or {})
        }
        
        return self.log_event(
            event_type=SafetyEventType.SCHEMA_SNAPSHOT,
            component="schema_graph",
            description=f"Schema snapshot taken: {reason}",
            severity=0.1,  # Low severity - informational
            data=event_data
        )
    
    def log_contradiction(
        self,
        contradiction_id: str,
        memory_ids: List[str],
        contradiction_type: str,
        confidence: float,
        resolution_strategy: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a contradiction detection event.
        
        Args:
            contradiction_id: ID of the contradiction
            memory_ids: IDs of memories involved in the contradiction
            contradiction_type: Type of contradiction
            confidence: Confidence in the contradiction detection
            resolution_strategy: Strategy for resolving the contradiction, if any
            data: Additional data
            
        Returns:
            Event ID
        """
        event_data = {
            "contradiction_id": contradiction_id,
            "memory_ids": memory_ids,
            "contradiction_type": contradiction_type,
            "confidence": confidence,
            "resolution_strategy": resolution_strategy,
            **(data or {})
        }
        
        return self.log_event(
            event_type=SafetyEventType.CONTRADICTION_DETECTED,
            component="contradiction_detector",
            description=f"Contradiction detected: {contradiction_type}",
            severity=0.5,  # Medium severity
            data=event_data
        )
    
    def log_boundary_crossing(
        self,
        boundary_type: str,
        value: float,
        threshold: float,
        component: str,
        description: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a safety boundary crossing event.
        
        Args:
            boundary_type: Type of safety boundary
            value: Current value that crossed the boundary
            threshold: Threshold that was crossed
            component: Component that generated the event
            description: Description of the boundary crossing
            data: Additional data
            
        Returns:
            Event ID
        """
        # Calculate how far past the threshold we are (normalized)
        distance = (value - threshold) / max(abs(threshold), 0.001)
        severity = min(0.9, 0.6 + abs(distance) * 0.3)  # Scale severity based on distance
        
        event_data = {
            "boundary_type": boundary_type,
            "value": value,
            "threshold": threshold,
            "distance": distance,
            **(data or {})
        }
        
        return self.log_event(
            event_type=SafetyEventType.SAFETY_BOUNDARY_CROSSED,
            component=component,
            description=description,
            severity=severity,
            data=event_data
        )
    
    def flush(self) -> None:
        """Flush buffered events to disk."""
        with self.lock:
            if not self.event_buffer:
                return
            
            # Check if we need to rotate the log file
            if self._should_rotate_log_file():
                self._rotate_log_file()
            
            # Read existing events
            with open(self.current_log_file, 'r') as f:
                data = json.load(f)
            
            # Add new events
            for event in self.event_buffer:
                data["events"].append(event.to_dict())
            
            # Update metadata
            data["last_updated"] = datetime.now().isoformat()
            data["event_count"] = len(data["events"])
            
            # Write back to file
            with open(self.current_log_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Clear buffer
            self.event_buffer = []
            
            logger.debug(f"Flushed safety trace events to {self.current_log_file}")
    
    def _should_rotate_log_file(self) -> bool:
        """Check if we should rotate the log file based on size."""
        if not self.current_log_file or not self.current_log_file.exists():
            return False
        
        # Check file size
        size_mb = self.current_log_file.stat().st_size / (1024 * 1024)
        return size_mb >= self.rotation_size_mb
    
    def _rotate_log_file(self) -> None:
        """Rotate the log file."""
        # Flush any remaining events first
        self._flush_to_current_file()
        
        # Create a new log file
        self._init_log_file()
        
        # Check if we need to remove old log files
        self._enforce_retention_policy()
    
    def _flush_to_current_file(self) -> None:
        """Flush events to current file without any rotation checks."""
        if not self.event_buffer:
            return
        
        # Read existing events
        with open(self.current_log_file, 'r') as f:
            data = json.load(f)
        
        # Add new events
        for event in self.event_buffer:
            data["events"].append(event.to_dict())
        
        # Update metadata
        data["last_updated"] = datetime.now().isoformat()
        data["event_count"] = len(data["events"])
        
        # Write back to file
        with open(self.current_log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Clear buffer
        self.event_buffer = []
    
    def _enforce_retention_policy(self) -> None:
        """Remove old log files to stay within retention limits."""
        log_files = sorted(
            [f for f in self.export_dir.glob("safety_trace_*.json")],
            key=lambda p: p.stat().st_mtime
        )
        
        # If we have more than the max allowed, delete the oldest ones
        if len(log_files) > self.max_log_files:
            for old_file in log_files[:-self.max_log_files]:
                try:
                    old_file.unlink()
                    logger.info(f"Removed old safety trace log file: {old_file}")
                except Exception as e:
                    logger.error(f"Failed to remove old log file {old_file}: {e}")
    
    def get_events_by_type(self, event_type: Union[SafetyEventType, str]) -> List[SafetyEvent]:
        """
        Get all events of a specific type.
        
        Args:
            event_type: Type of events to retrieve
            
        Returns:
            List of matching events
        """
        # Convert string to enum if needed
        if isinstance(event_type, str):
            try:
                event_type_enum = SafetyEventType(event_type)
            except ValueError:
                event_type_enum = None
        else:
            event_type_enum = event_type
        
        # Flush to make sure all events are on disk
        self.flush()
        
        # Load all events from all log files
        all_events = []
        for log_file in self.export_dir.glob("safety_trace_*.json"):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    events = data.get("events", [])
                    all_events.extend(events)
            except Exception as e:
                logger.error(f"Failed to read log file {log_file}: {e}")
        
        # Filter by event type
        filtered_events = []
        for event_data in all_events:
            event_type_value = event_data.get("event_type")
            if (event_type_enum and event_type_value == event_type_enum.value) or \
               (isinstance(event_type, str) and event_type_value == event_type):
                filtered_events.append(SafetyEvent.from_dict(event_data))
        
        return filtered_events
    
    def get_event_by_id(self, event_id: str) -> Optional[SafetyEvent]:
        """
        Get an event by its ID.
        
        Args:
            event_id: ID of the event to retrieve
            
        Returns:
            The event if found, None otherwise
        """
        # Check in-memory buffer first
        for event in self.event_buffer:
            if event.event_id == event_id:
                return event
        
        # Then check on disk
        for log_file in self.export_dir.glob("safety_trace_*.json"):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    for event_data in data.get("events", []):
                        if event_data.get("event_id") == event_id:
                            return SafetyEvent.from_dict(event_data)
            except Exception as e:
                logger.error(f"Failed to read log file {log_file}: {e}")
        
        return None
    
    def get_events_in_timeframe(
        self,
        start_time: Optional[Union[float, datetime]] = None,
        end_time: Optional[Union[float, datetime]] = None
    ) -> List[SafetyEvent]:
        """
        Get events within a specific timeframe.
        
        Args:
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            
        Returns:
            List of events within the timeframe
        """
        # Convert datetime to timestamp if needed
        if isinstance(start_time, datetime):
            start_time = start_time.timestamp()
        if isinstance(end_time, datetime):
            end_time = end_time.timestamp()
        
        # Flush to make sure all events are on disk
        self.flush()
        
        # Load all events from all log files
        all_events = []
        for log_file in self.export_dir.glob("safety_trace_*.json"):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    events = data.get("events", [])
                    all_events.extend(events)
            except Exception as e:
                logger.error(f"Failed to read log file {log_file}: {e}")
        
        # Filter by timeframe
        filtered_events = []
        for event_data in all_events:
            timestamp = event_data.get("timestamp", 0)
            if ((start_time is None or timestamp >= start_time) and
                (end_time is None or timestamp <= end_time)):
                filtered_events.append(SafetyEvent.from_dict(event_data))
        
        # Sort by timestamp
        filtered_events.sort(key=lambda e: e.timestamp)
        
        return filtered_events
    
    def export_to_file(self, filepath: Union[str, Path], format: str = "json") -> bool:
        """
        Export all safety traces to a single file.
        
        Args:
            filepath: Path to export to
            format: Export format (json, csv, encrypted)
            
        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)
        
        # Flush to make sure all events are on disk
        self.flush()
        
        # Collect all events
        all_events = []
        for log_file in sorted(self.export_dir.glob("safety_trace_*.json"), 
                              key=lambda p: p.stat().st_mtime):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    events = data.get("events", [])
                    all_events.extend(events)
            except Exception as e:
                logger.error(f"Failed to read log file {log_file}: {e}")
        
        # Create export data
        export_data = {
            "safety_trace_version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "event_count": len(all_events),
            "events": all_events
        }
        
        # Export in the specified format
        try:
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format.lower() == "csv":
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow([
                        "event_id", "event_type", "timestamp", "component", 
                        "description", "severity", "data", "related_events", "hash_chain"
                    ])
                    # Write events
                    for event in all_events:
                        writer.writerow([
                            event.get("event_id", ""),
                            event.get("event_type", ""),
                            event.get("timestamp_readable", ""),
                            event.get("component", ""),
                            event.get("description", ""),
                            event.get("severity", 0.0),
                            json.dumps(event.get("data", {})),
                            json.dumps(event.get("related_events", [])),
                            event.get("hash_chain", "")
                        ])
            elif format.lower() == "encrypted":
                if not self.encryption_key:
                    logger.error("Cannot export as encrypted without an encryption key")
                    return False
                    
                # Simple encryption using key
                import base64
                from cryptography.fernet import Fernet
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                
                # Derive a key from the encryption key
                salt = b'psi_c_ai_sdk_safety_trace'  # Fixed salt for simplicity
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
                fernet = Fernet(key)
                
                # Encrypt the data
                json_data = json.dumps(export_data).encode()
                encrypted_data = fernet.encrypt(json_data)
                
                # Write to file
                with open(filepath, 'wb') as f:
                    f.write(encrypted_data)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Exported {len(all_events)} safety trace events to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export safety traces: {e}")
            return False
    
    def verify_hash_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the integrity of the hash chain.
        
        Returns:
            Tuple of (is_valid, first_broken_event_id)
        """
        if not self.enable_hash_chain:
            logger.warning("Hash chain verification requested but hash chains are disabled")
            return True, None
        
        # Flush to make sure all events are on disk
        self.flush()
        
        # Load all events from all log files in chronological order
        all_events = []
        for log_file in sorted(self.export_dir.glob("safety_trace_*.json"), 
                              key=lambda p: p.stat().st_mtime):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    events = data.get("events", [])
                    all_events.extend(events)
            except Exception as e:
                logger.error(f"Failed to read log file {log_file}: {e}")
        
        # Verify the hash chain
        prev_hash = ""
        for event_data in all_events:
            event = SafetyEvent.from_dict(event_data)
            
            # Skip events without a hash chain
            if not event.hash_chain:
                continue
            
            # Verify that this event's hash_chain matches the previous event's hash
            if prev_hash and event.hash_chain != prev_hash:
                return False, event.event_id
            
            # Compute this event's hash for the next iteration
            prev_hash = event.compute_hash()
        
        return True, None
    
    def close(self) -> None:
        """Close the exporter and flush any remaining events."""
        with self.lock:
            self.flush()
            logger.info("Safety trace exporter closed")


# Global instance for convenience
_global_exporter: Optional[SafetyTraceExporter] = None

def get_safety_trace_exporter() -> SafetyTraceExporter:
    """Get the global safety trace exporter instance."""
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = SafetyTraceExporter()
    return _global_exporter

def log_safety_event(
    event_type: Union[SafetyEventType, str],
    component: str,
    description: str,
    severity: float = 0.0,
    data: Optional[Dict[str, Any]] = None,
    related_events: Optional[List[str]] = None
) -> str:
    """Convenience function to log a safety event using the global exporter."""
    return get_safety_trace_exporter().log_event(
        event_type=event_type,
        component=component,
        description=description,
        severity=severity,
        data=data,
        related_events=related_events
    ) 