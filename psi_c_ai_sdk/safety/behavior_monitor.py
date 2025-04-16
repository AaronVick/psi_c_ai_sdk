"""
Behavior Monitor System

Monitors and enforces boundaries on model behavior:
- Provides real-time monitoring of model behavior across sessions
- Integrates with reflection guards and profile analyzers
- Enforces safety boundaries on model outputs
- Collects and processes behavior metrics for safety assessment
"""

import logging
import time
import threading
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Union, Tuple
from datetime import datetime
from enum import Enum
import json

from .reflection_guard import ReflectionGuard, create_reflection_guard_with_validator
from .profile_analyzer import ProfileAnalyzer, SafetyProfile, ProfileCategory, create_default_analyzer

logger = logging.getLogger(__name__)

class BehaviorCategory(Enum):
    """Categories of AI behavior to monitor"""
    OUTPUT = "output"        # Content generation patterns
    REASONING = "reasoning"  # Reasoning process patterns
    ACTION = "action"        # Action-taking patterns
    PLANNING = "planning"    # Planning and strategy patterns
    LEARNING = "learning"    # Learning and adaptation patterns
    SOCIAL = "social"        # Social interaction patterns


class BehaviorBoundary:
    """
    Defines boundaries for model behavior.
    
    Attributes:
        name: Boundary name
        category: Behavior category
        description: Detailed description 
        threshold: Threshold value or range
        action: Action to take when boundary is crossed
    """
    
    def __init__(self, 
                name: str,
                category: BehaviorCategory,
                description: str,
                action: str = "alert",
                threshold: Optional[Dict[str, Any]] = None):
        """
        Initialize a behavior boundary.
        
        Args:
            name: Boundary name
            category: Behavior category
            description: Detailed description
            action: Action to take when boundary is crossed
            threshold: Threshold configuration
        """
        self.name = name
        self.category = category
        self.description = description
        self.action = action
        self.threshold = threshold or {}
        self.created_at = datetime.now().timestamp()
        self.updated_at = self.created_at
        self.violation_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert boundary to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "action": self.action,
            "threshold": self.threshold,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "violation_count": self.violation_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehaviorBoundary':
        """Create boundary from dictionary."""
        boundary = cls(
            name=data["name"],
            category=BehaviorCategory(data["category"]),
            description=data["description"],
            action=data["action"],
            threshold=data["threshold"]
        )
        boundary.created_at = data.get("created_at", boundary.created_at)
        boundary.updated_at = data.get("updated_at", boundary.updated_at)
        boundary.violation_count = data.get("violation_count", 0)
        return boundary


class BehaviorMonitor:
    """
    Monitors and enforces model behavior boundaries.
    
    Integrates reflection guards and profile analyzers to provide
    comprehensive behavior monitoring and enforcement.
    """
    
    def __init__(self, 
                reflection_guard: Optional[ReflectionGuard] = None,
                profile_analyzer: Optional[ProfileAnalyzer] = None,
                enforce_boundaries: bool = True):
        """
        Initialize the behavior monitor.
        
        Args:
            reflection_guard: ReflectionGuard for reasoning monitoring
            profile_analyzer: ProfileAnalyzer for profile monitoring
            enforce_boundaries: Whether to enforce boundaries
        """
        # Initialize components
        self.reflection_guard = reflection_guard or create_reflection_guard_with_validator()
        self.profile_analyzer = profile_analyzer or create_default_analyzer(self.reflection_guard)
        
        # Configuration
        self.enforce_boundaries = enforce_boundaries
        
        # State management
        self.boundaries: Dict[str, BehaviorBoundary] = {}
        self.boundary_violations: List[Dict[str, Any]] = []
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Set[str] = set()
        self._lock = threading.RLock()
        
        # Default profile
        self.default_profile_name = "base"
        
        # Event callbacks
        self.violation_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.register_violation_callback(self._handle_violation)
        
        # Register with profile analyzer
        self.profile_analyzer.register_alert_callback(self._handle_profile_alert)
        
        logger.info("Behavior monitor initialized")
        
    def add_boundary(self, boundary: BehaviorBoundary) -> None:
        """
        Add a behavior boundary.
        
        Args:
            boundary: Boundary to add
        """
        with self._lock:
            self.boundaries[boundary.name] = boundary
            logger.info(f"Added behavior boundary: {boundary.name}")
            
    def remove_boundary(self, name: str) -> bool:
        """
        Remove a behavior boundary.
        
        Args:
            name: Name of boundary to remove
            
        Returns:
            Whether the boundary was removed
        """
        with self._lock:
            if name in self.boundaries:
                del self.boundaries[name]
                logger.info(f"Removed behavior boundary: {name}")
                return True
            return False
            
    def get_boundary(self, name: str) -> Optional[BehaviorBoundary]:
        """
        Get a behavior boundary.
        
        Args:
            name: Name of boundary
            
        Returns:
            BehaviorBoundary if found, None otherwise
        """
        return self.boundaries.get(name)
        
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new monitoring session.
        
        Args:
            session_id: Optional ID for the session
            
        Returns:
            Session ID
        """
        session_id = session_id or str(uuid.uuid4())
        
        with self._lock:
            self.active_sessions.add(session_id)
            self.session_metrics[session_id] = {
                "start_time": datetime.now().timestamp(),
                "metrics": {},
                "events": [],
                "violations": [],
                "profile_name": self.default_profile_name
            }
            
        logger.info(f"Started monitoring session: {session_id}")
        return session_id
        
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a monitoring session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session metrics
        """
        with self._lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session not found: {session_id}")
                return {}
                
            self.active_sessions.remove(session_id)
            
            # Update end time
            if session_id in self.session_metrics:
                self.session_metrics[session_id]["end_time"] = datetime.now().timestamp()
                self.session_metrics[session_id]["duration"] = (
                    self.session_metrics[session_id]["end_time"] - 
                    self.session_metrics[session_id]["start_time"]
                )
                
            logger.info(f"Ended monitoring session: {session_id}")
            return self.session_metrics.get(session_id, {})
            
    def register_violation_callback(self, 
                                 callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for boundary violations.
        
        Args:
            callback: Function to call when a boundary is violated
        """
        with self._lock:
            self.violation_callbacks.append(callback)
            
    def monitor_reflection(self, 
                         session_id: str, 
                         content: str,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor model reflection content.
        
        Args:
            session_id: Session ID
            content: Reflection content
            metadata: Additional metadata
            
        Returns:
            Monitoring results
        """
        if session_id not in self.active_sessions:
            session_id = self.start_session(session_id)
            
        # Get profile name for this session
        profile_name = self.session_metrics[session_id].get("profile_name", self.default_profile_name)
        
        # Process with reflection guard
        reflection_id = f"{session_id}_{len(self.session_metrics[session_id].get('events', []))}"
        
        # Analyze with profile analyzer
        analysis_result = self.profile_analyzer.analyze_reflection(
            profile_name=profile_name,
            reflection_id=reflection_id,
            content=content
        )
        
        # Record event
        event = {
            "timestamp": datetime.now().timestamp(),
            "type": "reflection",
            "content_length": len(content),
            "reflection_id": reflection_id,
            "metadata": metadata or {},
            "analysis": analysis_result
        }
        
        with self._lock:
            if "events" not in self.session_metrics[session_id]:
                self.session_metrics[session_id]["events"] = []
                
            self.session_metrics[session_id]["events"].append(event)
            
            # Check boundaries
            boundary_results = self._check_boundaries(
                session_id=session_id,
                category=BehaviorCategory.REASONING,
                event=event
            )
            
            # Update results with boundary check
            event["boundary_results"] = boundary_results
            
        return {
            "analysis": analysis_result,
            "boundary_results": boundary_results,
            "reflection_id": reflection_id,
        }
        
    def monitor_output(self,
                     session_id: str,
                     content: str,
                     output_type: str = "text",
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor model output content.
        
        Args:
            session_id: Session ID
            content: Output content
            output_type: Type of output
            metadata: Additional metadata
            
        Returns:
            Monitoring results
        """
        if session_id not in self.active_sessions:
            session_id = self.start_session(session_id)
            
        # Get profile name for this session
        profile_name = self.session_metrics[session_id].get("profile_name", self.default_profile_name)
        
        # Record metrics
        metrics_results = {}
        
        # Record content length
        metrics_results["length"] = self.profile_analyzer.record_metric(
            profile_name=profile_name,
            category=ProfileCategory.CONTENT,
            metric_name="content_length",
            value=len(content)
        )
        
        # Record event
        event = {
            "timestamp": datetime.now().timestamp(),
            "type": "output",
            "output_type": output_type,
            "content_length": len(content),
            "metadata": metadata or {},
            "metrics_results": metrics_results
        }
        
        with self._lock:
            if "events" not in self.session_metrics[session_id]:
                self.session_metrics[session_id]["events"] = []
                
            self.session_metrics[session_id]["events"].append(event)
            
            # Check boundaries
            boundary_results = self._check_boundaries(
                session_id=session_id,
                category=BehaviorCategory.OUTPUT,
                event=event
            )
            
            # Update results with boundary check
            event["boundary_results"] = boundary_results
            
        return {
            "metrics_results": metrics_results,
            "boundary_results": boundary_results
        }
        
    def monitor_action(self,
                     session_id: str,
                     action_type: str,
                     action_data: Dict[str, Any],
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor model action execution.
        
        Args:
            session_id: Session ID
            action_type: Type of action
            action_data: Action parameters
            metadata: Additional metadata
            
        Returns:
            Monitoring results
        """
        if session_id not in self.active_sessions:
            session_id = self.start_session(session_id)
            
        # Get profile name for this session
        profile_name = self.session_metrics[session_id].get("profile_name", self.default_profile_name)
        
        # Record metrics
        metrics_results = {}
        
        # Record action occurence
        metrics_results["action"] = self.profile_analyzer.record_metric(
            profile_name=profile_name,
            category=ProfileCategory.INTERACTION,
            metric_name=f"action_{action_type}",
            value=1  # Increment counter
        )
        
        # Record event
        event = {
            "timestamp": datetime.now().timestamp(),
            "type": "action",
            "action_type": action_type,
            "action_data": action_data,
            "metadata": metadata or {},
            "metrics_results": metrics_results
        }
        
        with self._lock:
            if "events" not in self.session_metrics[session_id]:
                self.session_metrics[session_id]["events"] = []
                
            self.session_metrics[session_id]["events"].append(event)
            
            # Update action metrics
            if "metrics" not in self.session_metrics[session_id]:
                self.session_metrics[session_id]["metrics"] = {}
                
            if "actions" not in self.session_metrics[session_id]["metrics"]:
                self.session_metrics[session_id]["metrics"]["actions"] = {}
                
            action_metrics = self.session_metrics[session_id]["metrics"]["actions"]
            action_metrics[action_type] = action_metrics.get(action_type, 0) + 1
            
            # Check boundaries
            boundary_results = self._check_boundaries(
                session_id=session_id,
                category=BehaviorCategory.ACTION,
                event=event
            )
            
            # Update results with boundary check
            event["boundary_results"] = boundary_results
            
        return {
            "metrics_results": metrics_results,
            "boundary_results": boundary_results
        }
        
    def record_metric(self,
                    session_id: str,
                    category: Union[BehaviorCategory, ProfileCategory, str],
                    metric_name: str,
                    value: Union[float, int, bool, str],
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record a custom metric.
        
        Args:
            session_id: Session ID
            category: Metric category
            metric_name: Metric name
            value: Metric value
            metadata: Additional metadata
            
        Returns:
            Recording results
        """
        if session_id not in self.active_sessions:
            session_id = self.start_session(session_id)
            
        # Get profile name for this session
        profile_name = self.session_metrics[session_id].get("profile_name", self.default_profile_name)
        
        # Map BehaviorCategory to ProfileCategory if needed
        profile_category = category
        if isinstance(category, BehaviorCategory):
            category_map = {
                BehaviorCategory.OUTPUT: ProfileCategory.CONTENT,
                BehaviorCategory.REASONING: ProfileCategory.REASONING,
                BehaviorCategory.ACTION: ProfileCategory.INTERACTION,
                BehaviorCategory.PLANNING: ProfileCategory.REASONING,
                BehaviorCategory.LEARNING: ProfileCategory.PERFORMANCE,
                BehaviorCategory.SOCIAL: ProfileCategory.INTERACTION
            }
            profile_category = category_map.get(category, ProfileCategory.PERFORMANCE)
            
        # Record metric
        result = self.profile_analyzer.record_metric(
            profile_name=profile_name,
            category=profile_category,
            metric_name=metric_name,
            value=value
        )
        
        # Record event
        event = {
            "timestamp": datetime.now().timestamp(),
            "type": "metric",
            "category": str(category.value) if hasattr(category, "value") else str(category),
            "metric_name": metric_name,
            "value": value,
            "metadata": metadata or {},
            "result": result
        }
        
        with self._lock:
            if "events" not in self.session_metrics[session_id]:
                self.session_metrics[session_id]["events"] = []
                
            self.session_metrics[session_id]["events"].append(event)
            
            # Update metrics in session
            if "metrics" not in self.session_metrics[session_id]:
                self.session_metrics[session_id]["metrics"] = {}
                
            metrics_cat = str(category.value) if hasattr(category, "value") else str(category)
            if metrics_cat not in self.session_metrics[session_id]["metrics"]:
                self.session_metrics[session_id]["metrics"][metrics_cat] = {}
                
            self.session_metrics[session_id]["metrics"][metrics_cat][metric_name] = value
            
        return {
            "result": result,
            "violation": result.get("violation", False)
        }
        
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session metrics
        """
        return self.session_metrics.get(session_id, {})
        
    def get_session_events(self, 
                         session_id: str,
                         event_type: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get events for a session.
        
        Args:
            session_id: Session ID
            event_type: Filter by event type
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        session = self.session_metrics.get(session_id, {})
        events = session.get("events", [])
        
        if event_type:
            events = [e for e in events if e.get("type") == event_type]
            
        return events[-limit:]
        
    def get_violation_history(self, 
                            session_id: Optional[str] = None,
                            boundary_name: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history of boundary violations.
        
        Args:
            session_id: Filter by session ID
            boundary_name: Filter by boundary name
            limit: Maximum number of violations
            
        Returns:
            List of violations
        """
        with self._lock:
            violations = self.boundary_violations
            
            if session_id:
                violations = [v for v in violations if v.get("session_id") == session_id]
                
            if boundary_name:
                violations = [v for v in violations if v.get("boundary_name") == boundary_name]
                
            return violations[-limit:]
            
    def export_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Export all boundaries.
        
        Returns:
            Dict of boundary name to boundary data
        """
        with self._lock:
            return {name: boundary.to_dict() for name, boundary in self.boundaries.items()}
            
    def import_boundaries(self, boundaries_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Import boundaries from exported data.
        
        Args:
            boundaries_data: Dict of boundary name to boundary data
        """
        with self._lock:
            for name, data in boundaries_data.items():
                self.boundaries[name] = BehaviorBoundary.from_dict(data)
                
    def reset(self) -> None:
        """Reset the monitor state."""
        with self._lock:
            self.boundaries = {}
            self.boundary_violations = []
            self.session_metrics = {}
            self.active_sessions = set()
            
    def _check_boundaries(self,
                         session_id: str,
                         category: BehaviorCategory,
                         event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check event against boundaries.
        
        Args:
            session_id: Session ID
            category: Event category
            event: Event data
            
        Returns:
            Boundary check results
        """
        results = {
            "violations": [],
            "has_violations": False
        }
        
        # Find relevant boundaries
        relevant_boundaries = [
            b for b in self.boundaries.values()
            if b.category == category
        ]
        
        if not relevant_boundaries:
            return results
            
        # Check each boundary
        for boundary in relevant_boundaries:
            violation = self._check_boundary(boundary, event)
            
            if violation:
                violation["session_id"] = session_id
                results["violations"].append(violation)
                results["has_violations"] = True
                
                # Record violation
                self.boundary_violations.append(violation)
                
                # Record in session
                with self._lock:
                    if "violations" not in self.session_metrics[session_id]:
                        self.session_metrics[session_id]["violations"] = []
                        
                    self.session_metrics[session_id]["violations"].append(violation)
                    
                # Trigger callbacks
                for callback in self.violation_callbacks:
                    try:
                        callback(violation)
                    except Exception as e:
                        logger.error(f"Error in violation callback: {e}")
                
        return results
        
    def _check_boundary(self,
                       boundary: BehaviorBoundary,
                       event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if an event violates a boundary.
        
        Args:
            boundary: Boundary to check
            event: Event data
            
        Returns:
            Violation data if boundary violated, None otherwise
        """
        # Extract relevant metrics based on boundary category and event type
        metrics = None
        
        if boundary.category == BehaviorCategory.REASONING:
            if event.get("type") == "reflection":
                analysis = event.get("analysis", {})
                
                # Check for reasoning cycles
                if "guard_result" in analysis:
                    guard_result = analysis["guard_result"]
                    
                    if boundary.name == "max_reasoning_cycles" and guard_result.get("cycle_detected"):
                        return {
                            "timestamp": datetime.now().timestamp(),
                            "boundary_name": boundary.name,
                            "boundary_category": boundary.category.value,
                            "details": {
                                "reflection_id": event.get("reflection_id"),
                                "cycles_detected": guard_result.get("cycles_total", 0)
                            }
                        }
                        
                    if boundary.name == "max_contradictions" and guard_result.get("contradiction_detected"):
                        return {
                            "timestamp": datetime.now().timestamp(),
                            "boundary_name": boundary.name,
                            "boundary_category": boundary.category.value,
                            "details": {
                                "reflection_id": event.get("reflection_id"),
                                "contradictions_detected": guard_result.get("contradictions_total", 0)
                            }
                        }
                        
        elif boundary.category == BehaviorCategory.OUTPUT:
            if event.get("type") == "output":
                # Check output length
                if boundary.name == "max_output_length":
                    max_length = boundary.threshold.get("max", 10000)
                    actual_length = event.get("content_length", 0)
                    
                    if actual_length > max_length:
                        return {
                            "timestamp": datetime.now().timestamp(),
                            "boundary_name": boundary.name,
                            "boundary_category": boundary.category.value,
                            "details": {
                                "output_type": event.get("output_type"),
                                "length": actual_length,
                                "max_allowed": max_length
                            }
                        }
                        
        elif boundary.category == BehaviorCategory.ACTION:
            if event.get("type") == "action":
                action_type = event.get("action_type")
                
                # Check for restricted actions
                if boundary.name == "restricted_actions":
                    restricted = boundary.threshold.get("restricted_types", [])
                    
                    if action_type in restricted:
                        return {
                            "timestamp": datetime.now().timestamp(),
                            "boundary_name": boundary.name,
                            "boundary_category": boundary.category.value,
                            "details": {
                                "action_type": action_type,
                                "action_data": event.get("action_data"),
                                "reason": "Action type is restricted"
                            }
                        }
                        
                # Check for action rate limits
                if boundary.name == "action_rate_limit":
                    # This would require session-level tracking of action rates
                    pass
                    
        return None
        
    def _handle_violation(self, violation: Dict[str, Any]) -> None:
        """
        Handle a boundary violation.
        
        Args:
            violation: Violation data
        """
        boundary_name = violation.get("boundary_name")
        if not boundary_name:
            return
            
        # Update boundary violation count
        boundary = self.get_boundary(boundary_name)
        if boundary:
            boundary.violation_count += 1
            boundary.updated_at = datetime.now().timestamp()
            
        # Log violation
        logger.warning(
            f"Boundary violation: {boundary_name} in category "
            f"{violation.get('boundary_category')} - "
            f"{json.dumps(violation.get('details', {}))}"
        )
        
    def _handle_profile_alert(self, alert: Dict[str, Any]) -> None:
        """
        Handle a profile alert.
        
        Args:
            alert: Alert data
        """
        # This would handle alerts from the profile analyzer
        # Currently just logs them
        logger.warning(f"Profile alert: {json.dumps(alert)}")
        

def create_default_monitor() -> BehaviorMonitor:
    """
    Create a BehaviorMonitor with default settings.
    
    Returns:
        Configured BehaviorMonitor
    """
    # Create components
    reflection_guard = create_reflection_guard_with_validator()
    profile_analyzer = create_default_analyzer(reflection_guard)
    
    # Create monitor
    monitor = BehaviorMonitor(
        reflection_guard=reflection_guard,
        profile_analyzer=profile_analyzer,
        enforce_boundaries=True
    )
    
    # Add default boundaries
    monitor.add_boundary(
        BehaviorBoundary(
            name="max_reasoning_cycles",
            category=BehaviorCategory.REASONING,
            description="Maximum number of reasoning cycles",
            action="alert",
            threshold={"max": 5}
        )
    )
    
    monitor.add_boundary(
        BehaviorBoundary(
            name="max_contradictions",
            category=BehaviorCategory.REASONING,
            description="Maximum number of contradictions",
            action="alert",
            threshold={"max": 3}
        )
    )
    
    monitor.add_boundary(
        BehaviorBoundary(
            name="max_output_length",
            category=BehaviorCategory.OUTPUT,
            description="Maximum output length",
            action="alert",
            threshold={"max": 10000}
        )
    )
    
    monitor.add_boundary(
        BehaviorBoundary(
            name="restricted_actions",
            category=BehaviorCategory.ACTION,
            description="Restricted action types",
            action="block",
            threshold={"restricted_types": ["system", "delete", "modify_security"]}
        )
    )
    
    return monitor 