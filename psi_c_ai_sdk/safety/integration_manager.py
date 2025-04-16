"""
Safety Integration Manager

Coordinates safety components including:
- Reflection Guards
- Safety Profile Analyzers
- Safety Boundaries
- Feedback Mechanisms

Provides unified safety monitoring and response system.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from datetime import datetime
from enum import Enum

# Import safety components
from psi_c_ai_sdk.safety.reflection_guard import ReflectionGuard
from psi_c_ai_sdk.safety.profile_analyzer import SafetyProfiler, SafetyProfileRegistry

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety severity levels with numeric values for comparison."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SafetyResponse(Enum):
    """Response strategies to safety violations."""
    ALLOW = 0       # Proceed normally
    THROTTLE = 1    # Slow down processing 
    QUARANTINE = 2  # Isolate content
    BLOCK = 3       # Block the operation
    TERMINATE = 4   # Terminate the process

class SafetyIntegrationManager:
    """
    Unified manager for all safety components in the system.
    
    Coordinates between:
    - Reflection guards (detects contradictions, loops)
    - Safety profile analyzers (monitors resource usage)
    - Response policies (what to do when violations occur)
    """
    
    def __init__(self, 
                 enable_reflection_guard: bool = True,
                 enable_safety_profiler: bool = True,
                 default_safety_level: SafetyLevel = SafetyLevel.MEDIUM,
                 auto_throttle: bool = True,
                 safety_policy_path: Optional[str] = None):
        """
        Initialize the safety integration manager.
        
        Args:
            enable_reflection_guard: Whether to activate reflection guard
            enable_safety_profiler: Whether to activate safety profiler
            default_safety_level: Default safety level for operations
            auto_throttle: Whether to automatically throttle on violations
            safety_policy_path: Path to safety policy configuration file
        """
        self.default_safety_level = default_safety_level
        self.auto_throttle = auto_throttle
        
        # Initialize components
        self._setup_reflection_guard() if enable_reflection_guard else None
        self._setup_safety_profiler() if enable_safety_profiler else None
        
        # State tracking
        self.active_safety_violations: Dict[str, Dict[str, Any]] = {}
        self.safety_events: List[Dict[str, Any]] = []
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Safety policies (could be loaded from file)
        self.safety_policies = self._load_safety_policies(safety_policy_path)
        
        # Registered safety event callbacks
        self.safety_callbacks: Dict[str, List[Callable]] = {
            "reflection": [],
            "resource": [],
            "access": [],
            "timing": [],
            "schema": [],
            "general": []
        }
        
        # Composite safety state
        self.current_safety_state = {
            "overall_level": SafetyLevel.NONE,
            "reflection_guard": {
                "status": "inactive",
                "contradiction_count": 0,
                "cycle_count": 0
            },
            "safety_profiler": {
                "status": "inactive",
                "resource_alerts": 0,
                "timing_alerts": 0,
                "access_alerts": 0
            }
        }

    def _setup_reflection_guard(self) -> None:
        """Initialize the reflection guard."""
        self.reflection_guard = ReflectionGuard(
            cycle_threshold=3,
            cycle_count_threshold=5,
            var_threshold=2
        )
        self.current_safety_state["reflection_guard"]["status"] = "active"
        logger.info("Reflection guard initialized")
        
    def _setup_safety_profiler(self) -> None:
        """Initialize the safety profiler."""
        # Create and register a safety profiler with the global registry
        self.safety_profiler = SafetyProfiler(
            anomaly_threshold=3.0,
            resource_check_interval=1.0, 
            history_window=30,
            auto_throttle=self.auto_throttle,
            alert_callback=self._handle_profiler_alert
        )
        
        # Register with global registry
        registry = SafetyProfileRegistry()
        registry.register_profile("default", self.safety_profiler, make_default=True)
        
        # Start monitoring
        self.safety_profiler.start_monitoring()
        self.current_safety_state["safety_profiler"]["status"] = "active"
        logger.info("Safety profiler initialized and started")
        
    def _load_safety_policies(self, policy_path: Optional[str]) -> Dict[str, Any]:
        """
        Load safety policies from file or use defaults.
        
        Args:
            policy_path: Path to the policy file
            
        Returns:
            Dictionary of safety policies
        """
        # Default policies
        default_policies = {
            "reflection": {
                "contradiction_response": {
                    SafetyLevel.LOW: SafetyResponse.ALLOW,
                    SafetyLevel.MEDIUM: SafetyResponse.THROTTLE,
                    SafetyLevel.HIGH: SafetyResponse.QUARANTINE,
                    SafetyLevel.CRITICAL: SafetyResponse.BLOCK
                },
                "cycle_response": {
                    SafetyLevel.LOW: SafetyResponse.ALLOW,
                    SafetyLevel.MEDIUM: SafetyResponse.THROTTLE,
                    SafetyLevel.HIGH: SafetyResponse.QUARANTINE,
                    SafetyLevel.CRITICAL: SafetyResponse.BLOCK
                }
            },
            "resources": {
                "cpu_response": {
                    SafetyLevel.LOW: SafetyResponse.ALLOW,
                    SafetyLevel.MEDIUM: SafetyResponse.THROTTLE,
                    SafetyLevel.HIGH: SafetyResponse.THROTTLE,
                    SafetyLevel.CRITICAL: SafetyResponse.TERMINATE
                },
                "memory_response": {
                    SafetyLevel.LOW: SafetyResponse.ALLOW,
                    SafetyLevel.MEDIUM: SafetyResponse.THROTTLE,
                    SafetyLevel.HIGH: SafetyResponse.THROTTLE,
                    SafetyLevel.CRITICAL: SafetyResponse.TERMINATE
                }
            },
            "operations": {
                "default": {
                    SafetyLevel.LOW: SafetyResponse.ALLOW,
                    SafetyLevel.MEDIUM: SafetyResponse.THROTTLE,
                    SafetyLevel.HIGH: SafetyResponse.BLOCK,
                    SafetyLevel.CRITICAL: SafetyResponse.TERMINATE
                }
            }
        }
        
        # TODO: If policy_path is provided, load and merge with defaults
        
        return default_policies
        
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for a safety event.
        
        Args:
            event_type: Type of event (reflection, resource, access, timing, schema, general)
            callback: Function to call when this event occurs
        """
        if event_type not in self.safety_callbacks:
            logger.warning(f"Unknown safety event type: {event_type}")
            return
            
        self.safety_callbacks[event_type].append(callback)
        
    def process_reflection(self, reflection_id: str, content: str) -> dict:
        """
        Process a reflection through the reflection guard.
        
        Args:
            reflection_id: Unique identifier for the reflection
            content: Content of the reflection
            
        Returns:
            Dictionary with processing results
        """
        if not hasattr(self, 'reflection_guard'):
            return {"status": "inactive", "action": SafetyResponse.ALLOW}
            
        # Process through reflection guard
        guard_result = self.reflection_guard.process_reflection(reflection_id, content)
        
        # Determine safety level based on guard result
        safety_level = SafetyLevel.NONE
        
        if guard_result["contradiction_detected"]:
            safety_level = SafetyLevel.HIGH
        elif guard_result["cycle_detected"]:
            safety_level = SafetyLevel.MEDIUM
            
        # Get appropriate response from policy
        response = self._get_policy_response("reflection", "contradiction_response", safety_level)
        
        # Update safety state
        self.current_safety_state["reflection_guard"]["contradiction_count"] = (
            self.reflection_guard.get_contradiction_count()
        )
        self.current_safety_state["reflection_guard"]["cycle_count"] = (
            self.reflection_guard.get_cycle_count()
        )
        
        # Trigger callbacks if needed
        if safety_level != SafetyLevel.NONE:
            event = {
                "timestamp": datetime.now().timestamp(),
                "source": "reflection_guard",
                "safety_level": safety_level,
                "details": guard_result,
                "response": response
            }
            self.safety_events.append(event)
            self._trigger_callbacks("reflection", event)
            
        return {
            "status": "active",
            "safety_level": safety_level.name,
            "action": response.name,
            "details": guard_result
        }
        
    def record_operation_timing(self, operation_name: str, execution_time: float) -> dict:
        """
        Record timing for an operation.
        
        Args:
            operation_name: Name of the operation
            execution_time: Time taken in seconds
            
        Returns:
            Dictionary with processing results
        """
        if not hasattr(self, 'safety_profiler'):
            return {"status": "inactive", "action": SafetyResponse.ALLOW}
            
        # Use execution timer from safety profiler
        self.safety_profiler.record_execution_time(operation_name, execution_time)
        
        # Check if operation is throttled
        throttled = self.safety_profiler.is_operation_throttled(operation_name)
        response = SafetyResponse.THROTTLE if throttled else SafetyResponse.ALLOW
        
        return {
            "status": "recorded",
            "action": response.name,
            "throttled": throttled
        }
        
    def record_resource_access(self, 
                              access_type: str, 
                              resource_id: str,
                              metadata: Optional[Dict[str, Any]] = None) -> dict:
        """
        Record access to a resource.
        
        Args:
            access_type: Type of access (read, write, etc.)
            resource_id: Identifier for the resource
            metadata: Additional information about the access
            
        Returns:
            Dictionary with processing results
        """
        if not hasattr(self, 'safety_profiler'):
            return {"status": "inactive", "action": SafetyResponse.ALLOW}
            
        # Record the access
        self.safety_profiler.record_access(access_type, resource_id, metadata)
        
        return {
            "status": "recorded",
            "action": SafetyResponse.ALLOW
        }
        
    def validate_schema(self,
                       schema_name: str,
                       is_valid: bool,
                       validation_errors: Optional[List[str]] = None) -> dict:
        """
        Record schema validation result.
        
        Args:
            schema_name: Name of the schema
            is_valid: Whether validation passed
            validation_errors: List of validation errors if any
            
        Returns:
            Dictionary with processing results
        """
        if not hasattr(self, 'safety_profiler'):
            return {"status": "inactive", "action": SafetyResponse.ALLOW}
            
        # Record the validation
        self.safety_profiler.record_schema_validation(
            schema_name, 
            is_valid, 
            validation_errors
        )
        
        # Determine response based on validity
        response = SafetyResponse.ALLOW if is_valid else SafetyResponse.BLOCK
        
        return {
            "status": "validated",
            "action": response.name,
            "is_valid": is_valid,
            "errors": validation_errors or []
        }
        
    def get_safety_state(self) -> Dict[str, Any]:
        """
        Get the current overall safety state.
        
        Returns:
            Dictionary with the overall safety state
        """
        # Update the overall safety level
        self._update_overall_safety_level()
        
        # Return a copy of the state to prevent modifications
        return self.current_safety_state.copy()
        
    def get_safety_events(self, 
                         event_types: Optional[List[str]] = None,
                         min_level: Optional[SafetyLevel] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent safety events.
        
        Args:
            event_types: Types of events to include
            min_level: Minimum safety level to include
            limit: Maximum number of events to return
            
        Returns:
            List of safety events
        """
        filtered_events = self.safety_events.copy()
        
        # Filter by event type if specified
        if event_types:
            filtered_events = [
                e for e in filtered_events 
                if e.get("source", "").split("_")[0] in event_types
            ]
            
        # Filter by minimum level if specified
        if min_level:
            filtered_events = [
                e for e in filtered_events 
                if e.get("safety_level", SafetyLevel.NONE).value >= min_level.value
            ]
            
        # Return the most recent events up to the limit
        return filtered_events[-limit:]
        
    def reset_safety_state(self) -> None:
        """Reset all safety components and clear history."""
        # Reset reflection guard if active
        if hasattr(self, 'reflection_guard'):
            self.reflection_guard = ReflectionGuard(
                cycle_threshold=self.reflection_guard.cycle_threshold,
                cycle_count_threshold=self.reflection_guard.cycle_count_threshold,
                var_threshold=self.reflection_guard.var_threshold
            )
            self.current_safety_state["reflection_guard"] = {
                "status": "active",
                "contradiction_count": 0,
                "cycle_count": 0
            }
            
        # Reset safety profiler alerts if active
        if hasattr(self, 'safety_profiler'):
            self.safety_profiler.reset_alerts()
            self.current_safety_state["safety_profiler"] = {
                "status": "active",
                "resource_alerts": 0,
                "timing_alerts": 0,
                "access_alerts": 0
            }
            
        # Reset overall state
        self.active_safety_violations = {}
        self.safety_events = []
        self.current_safety_state["overall_level"] = SafetyLevel.NONE
        
        logger.info("Safety state has been reset")
        
    def shutdown(self) -> None:
        """Shut down all safety components."""
        # Stop safety profiler if active
        if hasattr(self, 'safety_profiler'):
            self.safety_profiler.stop_monitoring()
            self.current_safety_state["safety_profiler"]["status"] = "inactive"
            
        logger.info("Safety integration manager shut down")
        
    def _handle_profiler_alert(self, alert: Dict[str, Any]) -> None:
        """
        Handle alerts from the safety profiler.
        
        Args:
            alert: Alert data from the profiler
        """
        # Map alert severity to safety level
        severity_map = {
            "low": SafetyLevel.LOW,
            "medium": SafetyLevel.MEDIUM,
            "high": SafetyLevel.HIGH,
            "critical": SafetyLevel.CRITICAL
        }
        safety_level = severity_map.get(alert.get("severity", "low"), SafetyLevel.LOW)
        
        # Determine event type based on alert type
        event_type = "general"
        if alert.get("alert_type", "").startswith("resource_"):
            event_type = "resource"
        elif alert.get("alert_type", "").startswith("timing_"):
            event_type = "timing"
        elif alert.get("alert_type", "").startswith("access_"):
            event_type = "access"
        elif alert.get("alert_type", "").startswith("schema_"):
            event_type = "schema"
            
        # Create safety event
        event = {
            "timestamp": datetime.now().timestamp(),
            "source": f"{event_type}_profiler",
            "safety_level": safety_level,
            "details": alert.get("details", {}),
            "response": self._get_policy_response("operations", "default", safety_level)
        }
        
        # Add to safety events
        self.safety_events.append(event)
        
        # Update counters in safety state
        if event_type == "resource":
            self.current_safety_state["safety_profiler"]["resource_alerts"] += 1
        elif event_type == "timing":
            self.current_safety_state["safety_profiler"]["timing_alerts"] += 1
        elif event_type == "access":
            self.current_safety_state["safety_profiler"]["access_alerts"] += 1
            
        # Trigger callbacks
        self._trigger_callbacks(event_type, event)
        
    def _get_policy_response(self, 
                            category: str, 
                            policy_name: str, 
                            safety_level: SafetyLevel) -> SafetyResponse:
        """
        Get the appropriate response based on policy and safety level.
        
        Args:
            category: Policy category
            policy_name: Specific policy name
            safety_level: Current safety level
            
        Returns:
            Safety response to apply
        """
        # Default to ALLOW for no issues
        if safety_level == SafetyLevel.NONE:
            return SafetyResponse.ALLOW
            
        # Try to get response from policy
        try:
            return self.safety_policies.get(category, {}).get(policy_name, {}).get(
                safety_level, SafetyResponse.ALLOW
            )
        except (KeyError, TypeError):
            # Fall back to default policy
            return self.safety_policies.get("operations", {}).get("default", {}).get(
                safety_level, SafetyResponse.ALLOW
            )
            
    def _update_overall_safety_level(self) -> None:
        """Update the overall safety level based on all components."""
        max_level = SafetyLevel.NONE
        
        # Check reflection guard issues
        if hasattr(self, 'reflection_guard'):
            contradiction_count = self.reflection_guard.get_contradiction_count()
            cycle_count = self.reflection_guard.get_cycle_count()
            
            if contradiction_count > 5:
                max_level = max(max_level, SafetyLevel.HIGH)
            elif contradiction_count > 2:
                max_level = max(max_level, SafetyLevel.MEDIUM)
            elif contradiction_count > 0:
                max_level = max(max_level, SafetyLevel.LOW)
                
            if cycle_count > 10:
                max_level = max(max_level, SafetyLevel.HIGH)
            elif cycle_count > 5:
                max_level = max(max_level, SafetyLevel.MEDIUM)
            elif cycle_count > 0:
                max_level = max(max_level, SafetyLevel.LOW)
                
        # Check safety profiler issues
        if hasattr(self, 'safety_profiler'):
            # Check critical alerts
            critical_alerts = sum(
                1 for alert in self.safety_profiler.alerts
                if alert.get("severity") == "critical"
            )
            
            high_alerts = sum(
                1 for alert in self.safety_profiler.alerts
                if alert.get("severity") == "high"
            )
            
            medium_alerts = sum(
                1 for alert in self.safety_profiler.alerts
                if alert.get("severity") == "medium"
            )
            
            if critical_alerts > 0:
                max_level = max(max_level, SafetyLevel.CRITICAL)
            elif high_alerts > 2:
                max_level = max(max_level, SafetyLevel.HIGH)
            elif high_alerts > 0 or medium_alerts > 3:
                max_level = max(max_level, SafetyLevel.MEDIUM)
            elif medium_alerts > 0:
                max_level = max(max_level, SafetyLevel.LOW)
                
        # Update the state
        self.current_safety_state["overall_level"] = max_level
        
    def _trigger_callbacks(self, event_type: str, event: Dict[str, Any]) -> None:
        """
        Trigger callbacks for a safety event.
        
        Args:
            event_type: Type of event
            event: Event data
        """
        # Call specific event type callbacks
        for callback in self.safety_callbacks.get(event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {e}")
                
        # Call general callbacks for all events
        for callback in self.safety_callbacks.get("general", []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in general callback: {e}")


# Context manager for operation timing
class TimedOperation:
    """Context manager for timing operations through the safety system."""
    
    def __init__(self, manager: SafetyIntegrationManager, operation_name: str):
        self.manager = manager
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            self.manager.record_operation_timing(self.operation_name, execution_time) 