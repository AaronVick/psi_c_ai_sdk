"""
Recursive Stability Scanner

Detects runaway or looping recursion patterns in schema self-modeling.

Mathematical formulations:
- Track reflective loop depth:
  R_n(S) = ∑_{i=1}^{n} M_i(S)
  where M_i is the i-th meta-model of self
  
- Stability condition:
  |dΨ_C/dn| < ε_r for n ≥ n_threshold
  
- Trigger lockdown if:
  ΔΨ_C > δ_spike ∧ d²Ψ_C/dn² > 0
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

class RecursiveStabilityScanner:
    """Scanner for detecting and mitigating unstable recursion patterns."""
    
    def __init__(self, 
                 max_recursion_depth: int = 5,
                 stability_threshold: float = 0.05,
                 min_measurements: int = 3,
                 spike_threshold: float = 0.2,
                 lockdown_window: int = 10,
                 cooldown_period: int = 20):
        """
        Initialize the stability scanner.
        
        Args:
            max_recursion_depth: Maximum permitted recursion depth
            stability_threshold: Threshold for derivative stability (ε_r)
            min_measurements: Minimum measurements needed before stability check
            spike_threshold: Threshold for detecting spikes (δ_spike)
            lockdown_window: Number of cycles to maintain lockdown after instability
            cooldown_period: Cycles to wait after lockdown before allowing full depth
        """
        self.max_recursion_depth = max_recursion_depth
        self.stability_threshold = stability_threshold
        self.min_measurements = min_measurements
        self.spike_threshold = spike_threshold
        self.lockdown_window = lockdown_window
        self.cooldown_period = cooldown_period
        
        # State tracking
        self.psi_c_history: List[float] = []
        self.recursion_depths: List[int] = []
        self.current_lockdown_counter: int = 0
        self.current_cooldown_counter: int = 0
        self.is_in_lockdown: bool = False
        self.last_instability_type: Optional[str] = None
        self.last_instability_time: Optional[float] = None
        self.current_depth_limit: int = max_recursion_depth
        self.history: List[Dict[str, Any]] = []
        
        # Detection sensitivity
        self.derivative_window: int = 5
        self.pattern_detection_window: int = 20
        
        # Additional trackers
        self.loop_patterns: Dict[str, int] = {}  # Pattern -> count
        self.stability_scores: List[float] = []
    
    def record_measurement(self, psi_c_value: float, recursion_depth: int, 
                         timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Record a measurement of ΨC value and recursion depth.
        
        Args:
            psi_c_value: Current ΨC value
            recursion_depth: Current recursion depth
            timestamp: Measurement timestamp (defaults to current time)
            
        Returns:
            Status information about the current state
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        # Append to history
        self.psi_c_history.append(psi_c_value)
        self.recursion_depths.append(recursion_depth)
        
        # Update lockdown status
        if self.is_in_lockdown:
            self.current_lockdown_counter -= 1
            if self.current_lockdown_counter <= 0:
                self.is_in_lockdown = False
                self.current_cooldown_counter = self.cooldown_period
                logger.info("Exiting recursion lockdown, entering cooldown period")
        
        # Update cooldown status
        if self.current_cooldown_counter > 0:
            self.current_cooldown_counter -= 1
            if self.current_cooldown_counter <= 0:
                self.current_depth_limit = self.max_recursion_depth
                logger.info(f"Cooldown complete, restoring max recursion depth to {self.max_recursion_depth}")
        
        # Check for instability if we have enough measurements
        instability_detected = False
        instability_type = None
        
        if len(self.psi_c_history) >= self.min_measurements:
            # Check for instability conditions
            is_unstable, reason = self._check_instability()
            
            if is_unstable:
                instability_detected = True
                instability_type = reason
                self._trigger_lockdown(reason)
        
        # Calculate current stability score
        stability_score = self._calculate_stability_score()
        self.stability_scores.append(stability_score)
        
        # Record the event
        event = {
            "timestamp": timestamp,
            "psi_c": psi_c_value,
            "depth": recursion_depth,
            "permitted_depth": self.current_depth_limit,
            "in_lockdown": self.is_in_lockdown,
            "lockdown_counter": self.current_lockdown_counter,
            "cooldown_counter": self.current_cooldown_counter,
            "instability_detected": instability_detected,
            "instability_type": instability_type,
            "stability_score": stability_score
        }
        
        self.history.append(event)
        
        # Return current state information
        return {
            "permitted_depth": self.current_depth_limit,
            "in_lockdown": self.is_in_lockdown,
            "stability_score": stability_score,
            "cooldown_remaining": self.current_cooldown_counter,
            "instability_detected": instability_detected,
            "instability_type": instability_type
        }
    
    def get_permitted_recursion_depth(self) -> int:
        """
        Get the currently permitted maximum recursion depth.
        
        Returns:
            Current maximum permitted recursion depth
        """
        return self.current_depth_limit
    
    def is_currently_stable(self) -> Tuple[bool, Optional[str]]:
        """
        Check if the system is currently in a stable state.
        
        Returns:
            Tuple of (is_stable, reason_if_unstable)
        """
        if self.is_in_lockdown:
            return False, self.last_instability_type
            
        if len(self.psi_c_history) < self.min_measurements:
            return True, None  # Assume stable if not enough measurements
            
        return not self._check_instability()
    
    def _check_instability(self) -> Tuple[bool, Optional[str]]:
        """
        Check for signs of instability in the recorded measurements.
        
        Returns:
            Tuple of (is_unstable, reason_if_unstable)
        """
        # Only check if we have enough measurements
        if len(self.psi_c_history) < self.min_measurements:
            return False, None
            
        # 1. Check for rapid growth in ΨC (first derivative)
        first_derivative = self._calculate_first_derivative()
        if abs(first_derivative) > self.stability_threshold:
            return True, "rapid_growth"
            
        # 2. Check for acceleration in ΨC (second derivative)
        second_derivative = self._calculate_second_derivative()
        recent_psi_c_delta = self._calculate_recent_delta()
        if second_derivative > 0 and recent_psi_c_delta > self.spike_threshold:
            return True, "acceleration_spike"
            
        # 3. Check for oscillation patterns
        if self._detect_oscillation():
            return True, "oscillation"
            
        # 4. Check for recursion depth approaching limit
        recent_depths = self.recursion_depths[-self.derivative_window:]
        if any(d >= self.max_recursion_depth - 1 for d in recent_depths):
            return True, "approaching_depth_limit"
            
        # 5. Check for depth increasing while stability decreasing
        if (len(self.stability_scores) >= self.derivative_window and
            len(self.recursion_depths) >= self.derivative_window):
            
            avg_recent_stability = sum(self.stability_scores[-self.derivative_window:]) / self.derivative_window
            avg_older_stability = sum(self.stability_scores[-2*self.derivative_window:-self.derivative_window]) / self.derivative_window
            
            avg_recent_depth = sum(self.recursion_depths[-self.derivative_window:]) / self.derivative_window
            avg_older_depth = sum(self.recursion_depths[-2*self.derivative_window:-self.derivative_window]) / self.derivative_window
            
            if avg_recent_stability < avg_older_stability and avg_recent_depth > avg_older_depth:
                return True, "destabilizing_recursion"
                
        return False, None
    
    def _calculate_first_derivative(self) -> float:
        """
        Calculate the first derivative (rate of change) of ΨC values.
        
        Returns:
            First derivative value
        """
        if len(self.psi_c_history) < 2:
            return 0.0
            
        # Use the most recent values within the derivative window
        window = min(self.derivative_window, len(self.psi_c_history))
        recent_values = self.psi_c_history[-window:]
        
        if len(recent_values) < 2:
            return 0.0
            
        # Calculate average change per step
        changes = [recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values))]
        return sum(changes) / len(changes)
    
    def _calculate_second_derivative(self) -> float:
        """
        Calculate the second derivative (acceleration) of ΨC values.
        
        Returns:
            Second derivative value
        """
        if len(self.psi_c_history) < 3:
            return 0.0
            
        # Use the most recent values within the derivative window
        window = min(self.derivative_window, len(self.psi_c_history))
        recent_values = self.psi_c_history[-window:]
        
        if len(recent_values) < 3:
            return 0.0
            
        # Calculate first derivatives
        first_derivatives = [recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values))]
        
        # Calculate second derivatives
        second_derivatives = [first_derivatives[i] - first_derivatives[i-1] for i in range(1, len(first_derivatives))]
        
        return sum(second_derivatives) / len(second_derivatives)
    
    def _calculate_recent_delta(self) -> float:
        """
        Calculate the absolute change in ΨC over recent measurements.
        
        Returns:
            Recent delta value
        """
        if len(self.psi_c_history) < 2:
            return 0.0
            
        window = min(self.derivative_window, len(self.psi_c_history))
        recent_values = self.psi_c_history[-window:]
        
        return abs(recent_values[-1] - recent_values[0])
    
    def _detect_oscillation(self) -> bool:
        """
        Detect oscillation patterns in ΨC values.
        
        Returns:
            True if oscillation is detected
        """
        if len(self.psi_c_history) < self.pattern_detection_window:
            return False
            
        # Check for simple oscillation pattern (alternating increases and decreases)
        recent_values = self.psi_c_history[-self.pattern_detection_window:]
        changes = [1 if recent_values[i] > recent_values[i-1] else -1 for i in range(1, len(recent_values))]
        
        # Check for alternating pattern
        alternating_count = sum(1 for i in range(1, len(changes)) if changes[i] != changes[i-1])
        alternating_ratio = alternating_count / (len(changes) - 1)
        
        # If more than 70% of changes alternate direction, consider it oscillating
        return alternating_ratio > 0.7
    
    def _calculate_stability_score(self) -> float:
        """
        Calculate a stability score based on recent measurements.
        
        Returns:
            Stability score (0.0 = unstable, 1.0 = stable)
        """
        # If no measurements, assume neutral stability
        if len(self.psi_c_history) < self.min_measurements:
            return 0.5
            
        # Calculate based on derivatives and patterns
        first_derivative = abs(self._calculate_first_derivative())
        second_derivative = abs(self._calculate_second_derivative())
        
        # Normalize derivatives
        norm_first_deriv = min(1.0, first_derivative / self.stability_threshold)
        norm_second_deriv = min(1.0, second_derivative / (self.stability_threshold / 2))
        
        # Additional patterns
        oscillation_factor = 1.0
        if self._detect_oscillation():
            oscillation_factor = 0.5
            
        # Depth factor
        recent_depth = self.recursion_depths[-1] if self.recursion_depths else 0
        depth_factor = 1.0 - (recent_depth / self.max_recursion_depth) ** 2
        
        # Combined score (higher is more stable)
        raw_score = (1.0 - norm_first_deriv) * 0.4 + (1.0 - norm_second_deriv) * 0.3 + oscillation_factor * 0.1 + depth_factor * 0.2
        
        # Bound between 0 and 1
        return max(0.0, min(1.0, raw_score))
    
    def _trigger_lockdown(self, reason: str) -> None:
        """
        Trigger a lockdown period to prevent instability.
        
        Args:
            reason: Reason for the lockdown
        """
        self.is_in_lockdown = True
        self.current_lockdown_counter = self.lockdown_window
        self.last_instability_type = reason
        self.last_instability_time = datetime.now().timestamp()
        
        # Set a reduced depth limit during lockdown
        self.current_depth_limit = max(1, self.max_recursion_depth // 2)
        
        logger.warning(f"Recursive instability detected ({reason}). "
                      f"Limiting recursion depth to {self.current_depth_limit} "
                      f"for {self.lockdown_window} cycles.")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about stability.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_measurements": len(self.psi_c_history),
            "current_depth_limit": self.current_depth_limit,
            "max_depth_limit": self.max_recursion_depth,
            "in_lockdown": self.is_in_lockdown,
            "lockdown_counter": self.current_lockdown_counter,
            "cooldown_counter": self.current_cooldown_counter,
            "last_instability_type": self.last_instability_type,
            "last_instability_time": self.last_instability_time,
            "recent_stability_score": self.stability_scores[-1] if self.stability_scores else None,
            "avg_stability_score": (sum(self.stability_scores) / len(self.stability_scores)) 
                                  if self.stability_scores else None
        }
    
    def reset(self) -> None:
        """Reset the scanner state while preserving configuration."""
        self.psi_c_history = []
        self.recursion_depths = []
        self.current_lockdown_counter = 0
        self.current_cooldown_counter = 0
        self.is_in_lockdown = False
        self.last_instability_type = None
        self.last_instability_time = None
        self.current_depth_limit = self.max_recursion_depth
        self.loop_patterns = {}
        self.stability_scores = []
        
        # Keep history for debugging
        self.history.append({
            "timestamp": datetime.now().timestamp(),
            "event": "reset",
            "message": "Scanner state was reset"
        })
        
        logger.info("Recursive stability scanner has been reset")


class RecursiveDepthController:
    """Controls recursive depth and prevents instability in self-modeling."""
    
    def __init__(self, 
                 stability_scanner: Optional[RecursiveStabilityScanner] = None,
                 safe_mode: bool = True):
        """
        Initialize the depth controller.
        
        Args:
            stability_scanner: RecursiveStabilityScanner instance
            safe_mode: Whether to use restrictive defaults for safety
        """
        self.scanner = stability_scanner or RecursiveStabilityScanner()
        self.safe_mode = safe_mode
        self.total_reflections = 0
        self.reflection_stack: List[Dict[str, Any]] = []
        self.callbacks: Dict[str, List[Callable]] = {
            "lockdown": [],
            "depth_exceeded": [],
            "reflection_complete": []
        }
    
    def enter_reflection(self, psi_c_value: float, reflection_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enter a new level of reflection.
        
        Args:
            psi_c_value: Current ΨC value
            reflection_metadata: Additional metadata about this reflection
            
        Returns:
            Status information including whether reflection is permitted
        """
        current_depth = len(self.reflection_stack)
        permitted_depth = self.scanner.get_permitted_recursion_depth()
        
        # Record this reflection attempt
        reflection_data = {
            "reflection_id": f"reflection_{self.total_reflections}",
            "timestamp": datetime.now().timestamp(),
            "psi_c_value": psi_c_value,
            "depth": current_depth,
            "metadata": reflection_metadata or {}
        }
        
        # Check if we're allowed to go deeper
        is_permitted = current_depth < permitted_depth
        
        if is_permitted:
            # Push to stack
            self.reflection_stack.append(reflection_data)
            
            # Record the measurement
            self.scanner.record_measurement(psi_c_value, current_depth + 1)
            
            status = {
                "permitted": True,
                "current_depth": current_depth + 1,
                "max_permitted_depth": permitted_depth,
                "reflection_id": reflection_data["reflection_id"],
                "stability_info": self.scanner.get_statistics()
            }
        else:
            # Reflection not permitted
            reflection_data["prevented"] = True
            
            # Trigger callbacks
            for callback in self.callbacks.get("depth_exceeded", []):
                try:
                    callback(current_depth, permitted_depth, reflection_data)
                except Exception as e:
                    logger.error(f"Error in depth_exceeded callback: {e}")
            
            status = {
                "permitted": False,
                "current_depth": current_depth,
                "max_permitted_depth": permitted_depth,
                "reason": "max_depth_exceeded",
                "stability_info": self.scanner.get_statistics()
            }
        
        self.total_reflections += 1
        return status
    
    def exit_reflection(self, psi_c_value: float, reflection_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Exit the current reflection level.
        
        Args:
            psi_c_value: Current ΨC value
            reflection_result: Results of this reflection
            
        Returns:
            Status information
        """
        # Check if we have an active reflection
        if not self.reflection_stack:
            return {
                "error": "no_active_reflection",
                "current_depth": 0,
                "stability_info": self.scanner.get_statistics()
            }
        
        # Pop the reflection
        reflection_data = self.reflection_stack.pop()
        
        # Update with results
        reflection_data["exit_timestamp"] = datetime.now().timestamp()
        reflection_data["duration"] = reflection_data["exit_timestamp"] - reflection_data["timestamp"]
        reflection_data["exit_psi_c_value"] = psi_c_value
        reflection_data["psi_c_delta"] = psi_c_value - reflection_data["psi_c_value"]
        
        if reflection_result:
            reflection_data["result"] = reflection_result
        
        # Record the measurement on exit
        current_depth = len(self.reflection_stack)
        self.scanner.record_measurement(psi_c_value, current_depth)
        
        # Trigger callbacks
        for callback in self.callbacks.get("reflection_complete", []):
            try:
                callback(reflection_data)
            except Exception as e:
                logger.error(f"Error in reflection_complete callback: {e}")
        
        return {
            "exited_depth": current_depth + 1,
            "current_depth": current_depth,
            "reflection_id": reflection_data["reflection_id"],
            "psi_c_delta": reflection_data["psi_c_delta"],
            "stability_info": self.scanner.get_statistics()
        }
    
    def register_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Register a callback for specific events.
        
        Args:
            event_type: Event type ('lockdown', 'depth_exceeded', 'reflection_complete')
            callback: Callback function
            
        Returns:
            True if registered successfully
        """
        if event_type not in self.callbacks:
            return False
            
        self.callbacks[event_type].append(callback)
        return True
    
    def get_current_depth(self) -> int:
        """
        Get the current reflection depth.
        
        Returns:
            Current depth
        """
        return len(self.reflection_stack)
    
    def get_stability_status(self) -> Dict[str, Any]:
        """
        Get the current stability status.
        
        Returns:
            Dictionary of stability information
        """
        is_stable, reason = self.scanner.is_currently_stable()
        
        return {
            "is_stable": is_stable,
            "instability_reason": reason,
            "current_depth": len(self.reflection_stack),
            "max_permitted_depth": self.scanner.get_permitted_recursion_depth(),
            "in_lockdown": self.scanner.is_in_lockdown,
            **self.scanner.get_statistics()
        }
    
    def force_lockdown(self, reason: str, duration: Optional[int] = None) -> None:
        """
        Force the system into lockdown mode.
        
        Args:
            reason: Reason for the lockdown
            duration: Lockdown duration (uses default if None)
        """
        # Override the scanner's lockdown
        self.scanner.is_in_lockdown = True
        self.scanner.last_instability_type = reason
        self.scanner.last_instability_time = datetime.now().timestamp()
        
        if duration is not None:
            self.scanner.current_lockdown_counter = duration
        else:
            self.scanner.current_lockdown_counter = self.scanner.lockdown_window
            
        # Set a reduced depth limit during lockdown
        self.scanner.current_depth_limit = max(1, self.scanner.max_recursion_depth // 2)
        
        # Trigger callbacks
        for callback in self.callbacks.get("lockdown", []):
            try:
                callback(reason, self.scanner.current_depth_limit)
            except Exception as e:
                logger.error(f"Error in lockdown callback: {e}")
        
        logger.warning(f"Manual lockdown triggered: {reason}. "
                      f"Limiting recursion depth to {self.scanner.current_depth_limit} "
                      f"for {self.scanner.current_lockdown_counter} cycles.")
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.scanner.reset()
        self.reflection_stack = []
        
        logger.info("Recursive depth controller has been reset") 