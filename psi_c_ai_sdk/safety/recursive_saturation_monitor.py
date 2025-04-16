"""
Recursive Saturation Monitor for ΨC-AI SDK

This module implements a monitor for detecting recursive saturation in self-modeling systems,
based on Formula 46 from the AGI-Aware Mathematical Safeguards:

Saturation(t) = max(1/n ∑ |dR_i/dt|)

It provides mechanisms to:
1. Track reflective modeling intensity over time
2. Detect unstable recursive growth
3. Trigger safety actions when saturation thresholds are exceeded
4. Automatically adjust parameters based on system behavior
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Deque, Callable
from collections import deque
from enum import Enum, auto
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class SaturationAction(Enum):
    """Actions to take when saturation is detected."""
    LOG_ONLY = auto()       # Just log the event
    PAUSE_REFLECTION = auto()  # Pause reflection activities
    ROLLBACK = auto()       # Roll back to a previous state
    TERMINATE = auto()      # Terminate the current process


class SaturationLevel(Enum):
    """Levels of recursive saturation."""
    NORMAL = auto()         # Normal operation
    ELEVATED = auto()       # Higher than normal, but still safe
    WARNING = auto()        # Approaching dangerous levels
    CRITICAL = auto()       # Critically high saturation


class RecursiveSaturationMonitor:
    """
    Monitor for detecting and responding to recursive saturation in self-modeling systems.
    
    This implements Formula 46 from the AGI-Aware Mathematical Safeguards:
    Saturation(t) = max(1/n ∑ |dR_i/dt|)
    
    The monitor tracks reflective modeling intensities (R_i values) over time,
    calculates the rate of change, and detects when the system is at risk of
    runaway recursion.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        normal_threshold: float = 0.1,
        warning_threshold: float = 0.3,
        critical_threshold: float = 0.5,
        default_action: SaturationAction = SaturationAction.PAUSE_REFLECTION,
        sampling_interval: float = 1.0,  # seconds
        auto_adjust: bool = True
    ):
        """
        Initialize the recursive saturation monitor.
        
        Args:
            window_size: Size of the sliding window for R_i values
            normal_threshold: Threshold for normal saturation
            warning_threshold: Threshold for warning level saturation
            critical_threshold: Threshold for critical saturation
            default_action: Default action to take when critical saturation is detected
            sampling_interval: Time interval between measurements in seconds
            auto_adjust: Whether to automatically adjust thresholds based on history
        """
        self.window_size = window_size
        self.normal_threshold = normal_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.default_action = default_action
        self.sampling_interval = sampling_interval
        self.auto_adjust = auto_adjust
        
        # Store for R_i values and timestamps
        self.r_values: Deque[List[float]] = deque(maxlen=window_size)
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        
        # Saturation history for auto-adjustment
        self.saturation_history: Deque[float] = deque(maxlen=100)
        
        # Track current state
        self.current_saturation = 0.0
        self.current_level = SaturationLevel.NORMAL
        self.last_check_time = time.time()
        
        # Action callbacks
        self.action_handlers: Dict[SaturationAction, Callable] = {}
        
        # Statistics
        self.stats = {
            "created_at": datetime.now().isoformat(),
            "total_checks": 0,
            "saturation_events": 0,
            "warning_events": 0,
            "critical_events": 0,
            "max_saturation": 0.0,
            "last_critical": None,
            "auto_adjustments": 0
        }
    
    def register_action_handler(
        self, 
        action: SaturationAction, 
        handler: Callable[[], Any]
    ) -> None:
        """
        Register a handler for a saturation action.
        
        Args:
            action: The saturation action to register a handler for
            handler: The callback function to invoke
        """
        self.action_handlers[action] = handler
        logger.info(f"Registered handler for {action.name}")
    
    def add_r_values(self, r_values: List[float]) -> Tuple[float, SaturationLevel]:
        """
        Add a new set of R_i values and calculate saturation.
        
        Args:
            r_values: List of R_i values (reflective modeling intensities)
            
        Returns:
            Tuple of (saturation_value, saturation_level)
        """
        current_time = time.time()
        self.r_values.append(r_values)
        self.timestamps.append(current_time)
        
        # Update stats
        self.stats["total_checks"] += 1
        
        # Calculate saturation if we have enough data
        if len(self.r_values) >= 2:
            saturation = self._calculate_saturation()
            self.current_saturation = saturation
            self.saturation_history.append(saturation)
            
            # Update max saturation
            if saturation > self.stats["max_saturation"]:
                self.stats["max_saturation"] = saturation
            
            # Determine saturation level
            if saturation >= self.critical_threshold:
                level = SaturationLevel.CRITICAL
                self.stats["critical_events"] += 1
                self.stats["last_critical"] = datetime.now().isoformat()
                logger.warning(f"CRITICAL saturation detected: {saturation:.4f}")
            elif saturation >= self.warning_threshold:
                level = SaturationLevel.WARNING
                self.stats["warning_events"] += 1
                logger.warning(f"WARNING saturation level: {saturation:.4f}")
            elif saturation >= self.normal_threshold:
                level = SaturationLevel.ELEVATED
                logger.info(f"ELEVATED saturation level: {saturation:.4f}")
            else:
                level = SaturationLevel.NORMAL
            
            self.current_level = level
            
            # If critical or warning, increment saturation events
            if level in (SaturationLevel.WARNING, SaturationLevel.CRITICAL):
                self.stats["saturation_events"] += 1
            
            # Auto-adjust thresholds if enabled
            if self.auto_adjust and len(self.saturation_history) >= 50:
                self._auto_adjust_thresholds()
            
            return saturation, level
        
        return 0.0, SaturationLevel.NORMAL
    
    def _calculate_saturation(self) -> float:
        """
        Calculate the saturation value based on Formula 46.
        
        Returns:
            Calculated saturation value
        """
        # We need at least two sets of R values
        if len(self.r_values) < 2:
            return 0.0
        
        # Get the most recent and previous R values
        r_current = self.r_values[-1]
        r_previous = self.r_values[-2]
        
        # Calculate time difference
        dt = self.timestamps[-1] - self.timestamps[-2]
        
        # If time difference is too small, use the sampling interval
        if dt < 1e-6:
            dt = self.sampling_interval
        
        # Calculate derivatives
        derivatives = []
        for i in range(min(len(r_current), len(r_previous))):
            derivative = abs((r_current[i] - r_previous[i]) / dt)
            derivatives.append(derivative)
        
        # Calculate average derivative
        if not derivatives:
            return 0.0
        
        # Formula 46: Saturation(t) = max(1/n ∑ |dR_i/dt|)
        saturation = np.mean(derivatives)
        
        return saturation
    
    def _auto_adjust_thresholds(self) -> None:
        """
        Automatically adjust thresholds based on observed saturation history.
        
        This helps the system adapt to different workloads and environments.
        """
        if not self.saturation_history:
            return
        
        # Calculate statistics from history
        mean_saturation = np.mean(self.saturation_history)
        std_saturation = np.std(self.saturation_history)
        
        # Only adjust if we have a significant amount of data
        if len(self.saturation_history) >= 50:
            # Adjust normal threshold to be slightly above mean
            new_normal = max(0.05, min(0.3, mean_saturation + 0.5 * std_saturation))
            
            # Warning is 2x normal, critical is 3x normal
            new_warning = max(0.2, min(0.5, new_normal * 2))
            new_critical = max(0.4, min(0.8, new_normal * 3))
            
            # Only count as adjustment if thresholds change significantly
            if (abs(new_normal - self.normal_threshold) > 0.02 or
                abs(new_warning - self.warning_threshold) > 0.02 or
                abs(new_critical - self.critical_threshold) > 0.02):
                
                # Store old values for logging
                old_normal = self.normal_threshold
                old_warning = self.warning_threshold
                old_critical = self.critical_threshold
                
                # Update thresholds
                self.normal_threshold = new_normal
                self.warning_threshold = new_warning
                self.critical_threshold = new_critical
                
                # Update stats
                self.stats["auto_adjustments"] += 1
                
                logger.info(
                    f"Auto-adjusted thresholds: normal {old_normal:.2f}->{new_normal:.2f}, "
                    f"warning {old_warning:.2f}->{new_warning:.2f}, "
                    f"critical {old_critical:.2f}->{new_critical:.2f}"
                )
    
    def check_saturation(self) -> Tuple[float, SaturationLevel, bool]:
        """
        Check the current saturation status.
        
        Returns:
            Tuple of (saturation_value, saturation_level, action_triggered)
        """
        action_triggered = False
        
        # If we're in a warning or critical state, trigger the appropriate action
        if self.current_level == SaturationLevel.CRITICAL:
            action = self.default_action
            handler = self.action_handlers.get(action)
            
            if handler:
                try:
                    handler()
                    action_triggered = True
                    logger.warning(f"Executed {action.name} action for critical saturation")
                except Exception as e:
                    logger.error(f"Error executing {action.name} action: {str(e)}")
            else:
                logger.warning(f"No handler registered for {action.name}")
        
        return self.current_saturation, self.current_level, action_triggered
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the saturation monitor.
        
        Returns:
            Dictionary with monitor statistics
        """
        return {
            **self.stats,
            "current_saturation": self.current_saturation,
            "current_level": self.current_level.name,
            "normal_threshold": self.normal_threshold,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "history_size": len(self.saturation_history),
            "avg_saturation": np.mean(list(self.saturation_history)) if self.saturation_history else 0.0,
            "window_size": self.window_size,
            "auto_adjust": self.auto_adjust
        }
    
    def reset(self) -> None:
        """
        Reset the saturation monitor to its initial state.
        """
        self.r_values.clear()
        self.timestamps.clear()
        self.current_saturation = 0.0
        self.current_level = SaturationLevel.NORMAL
        self.last_check_time = time.time()
        
        # Keep saturation history for auto-adjustment but clear event counts
        self.stats.update({
            "total_checks": 0,
            "saturation_events": 0,
            "warning_events": 0,
            "critical_events": 0,
            "last_critical": None
        })
        
        logger.info("Recursive saturation monitor reset")


def create_default_monitor(
    window_size: int = 10,
    normal_threshold: float = 0.1,
    warning_threshold: float = 0.3,
    critical_threshold: float = 0.5,
    action: SaturationAction = SaturationAction.PAUSE_REFLECTION,
) -> RecursiveSaturationMonitor:
    """
    Create a default recursive saturation monitor with standard settings.
    
    Args:
        window_size: Size of the sliding window for R_i values
        normal_threshold: Threshold for normal saturation
        warning_threshold: Threshold for warning level saturation
        critical_threshold: Threshold for critical saturation
        action: Default action to take when critical saturation is detected
        
    Returns:
        Configured RecursiveSaturationMonitor
    """
    monitor = RecursiveSaturationMonitor(
        window_size=window_size,
        normal_threshold=normal_threshold,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        default_action=action
    )
    
    # Register default action handlers
    monitor.register_action_handler(
        SaturationAction.LOG_ONLY,
        lambda: logger.warning("RECURSIVE SATURATION DETECTED - Log Only Action")
    )
    
    return monitor 