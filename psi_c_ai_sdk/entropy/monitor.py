"""
Entropy Monitoring for ΨC-AI SDK

This module implements entropy monitoring capabilities for ΨC-based systems,
providing real-time tracking of entropy levels, alert thresholds, and
a subscription-based event system for entropy-related events.

Key components:
- EntropyMonitor: Continuously tracks entropy levels using configurable thresholds
- EntropyAlert: Represents different levels of entropy alerts
- EntropySubscriber: Interface for components that need entropy notifications
"""

import time
import threading
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta

from psi_c_ai_sdk.memory.memory import MemoryStore
from psi_c_ai_sdk.entropy.entropy import EntropyCalculator

logger = logging.getLogger(__name__)


class EntropyAlert(Enum):
    """Entropy alert levels indicating system stability state."""
    NORMAL = 0       # Normal entropy levels
    ELEVATED = 1     # Slightly elevated entropy
    HIGH = 2         # High entropy, potential instability
    CRITICAL = 3     # Critical entropy, system at risk


class EntropySubscriber:
    """Interface for components that subscribe to entropy alerts."""
    
    def on_entropy_alert(self, alert_level: EntropyAlert, entropy_value: float,
                         details: Dict[str, Any]) -> None:
        """
        Called when an entropy alert occurs.
        
        Args:
            alert_level: Current alert level
            entropy_value: Current entropy value
            details: Additional details about the alert
        """
        raise NotImplementedError("Subscribers must implement on_entropy_alert")
    
    def on_termination_decision(self, entropy_value: float,
                                details: Dict[str, Any]) -> bool:
        """
        Called when entropy reaches termination threshold.
        
        Args:
            entropy_value: Current entropy value
            details: Additional details about the termination
            
        Returns:
            True if termination should proceed, False to override
        """
        raise NotImplementedError("Subscribers must implement on_termination_decision")


class EntropyMonitor:
    """
    Monitors entropy levels in a memory system and provides alerts.
    
    This class implements a continuous monitoring system that periodically
    calculates entropy levels, compares them against configurable thresholds,
    and notifies subscribers when alert conditions are met.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        entropy_calculator: Optional[EntropyCalculator] = None,
        elevated_threshold: float = 0.6,
        high_threshold: float = 0.75,
        critical_threshold: float = 0.85,
        termination_threshold: float = 0.95,
        check_interval: float = 5.0,
        window_size: int = 5
    ):
        """
        Initialize the entropy monitor.
        
        Args:
            memory_store: Memory store to monitor
            entropy_calculator: Calculator to use for entropy measurement
            elevated_threshold: Threshold for ELEVATED alerts
            high_threshold: Threshold for HIGH alerts
            critical_threshold: Threshold for CRITICAL alerts
            termination_threshold: Threshold for system termination decisions
            check_interval: Seconds between entropy checks
            window_size: Number of measurements to use for trend analysis
        """
        self.memory_store = memory_store
        self.entropy_calculator = entropy_calculator or EntropyCalculator()
        
        # Thresholds
        self.thresholds = {
            EntropyAlert.ELEVATED: elevated_threshold,
            EntropyAlert.HIGH: high_threshold,
            EntropyAlert.CRITICAL: critical_threshold
        }
        self.termination_threshold = termination_threshold
        
        # Monitoring settings
        self.check_interval = check_interval
        self.window_size = window_size
        self.running = False
        self.monitor_thread = None
        
        # Tracking state
        self.current_alert = EntropyAlert.NORMAL
        self.measurements = []  # Recent entropy measurements
        self.measurement_times = []  # Timestamps of measurements
        self.subscribers: List[EntropySubscriber] = []
        
        # Callbacks
        self.alert_callbacks: List[Callable[[EntropyAlert, float, Dict[str, Any]], None]] = []
        self.termination_callbacks: List[Callable[[float, Dict[str, Any]], bool]] = []
    
    def start(self) -> None:
        """Start the entropy monitoring thread."""
        if self.running:
            logger.warning("Entropy monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="entropy-monitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Entropy monitoring started")
    
    def stop(self) -> None:
        """Stop the entropy monitoring thread."""
        if not self.running:
            logger.warning("Entropy monitor is not running")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Entropy monitoring stopped")
    
    def add_subscriber(self, subscriber: EntropySubscriber) -> None:
        """
        Add a subscriber to receive entropy alerts.
        
        Args:
            subscriber: EntropySubscriber to receive alerts
        """
        if subscriber not in self.subscribers:
            self.subscribers.append(subscriber)
    
    def remove_subscriber(self, subscriber: EntropySubscriber) -> None:
        """
        Remove a subscriber from entropy alerts.
        
        Args:
            subscriber: EntropySubscriber to remove
        """
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)
    
    def add_alert_callback(self, callback: Callable[[EntropyAlert, float, Dict[str, Any]], None]) -> None:
        """
        Add a callback for entropy alerts.
        
        Args:
            callback: Function to call on alerts
        """
        self.alert_callbacks.append(callback)
    
    def add_termination_callback(self, callback: Callable[[float, Dict[str, Any]], bool]) -> None:
        """
        Add a callback for termination decisions.
        
        Args:
            callback: Function to call for termination decisions
        """
        self.termination_callbacks.append(callback)
    
    def get_current_entropy(self) -> float:
        """
        Get the current entropy level.
        
        Returns:
            Current entropy value (0.0-1.0)
        """
        return self.entropy_calculator.calculate_memory_store_entropy(self.memory_store)
    
    def get_detailed_metrics(self) -> Dict[str, float]:
        """
        Get detailed entropy metrics from all measures.
        
        Returns:
            Dictionary with entropy metrics by category
        """
        return self.entropy_calculator.get_detailed_entropy_metrics(self.memory_store)
    
    def get_entropy_history(self) -> Tuple[List[datetime], List[float]]:
        """
        Get the history of entropy measurements.
        
        Returns:
            Tuple of (timestamps, entropy values)
        """
        return self.measurement_times.copy(), self.measurements.copy()
    
    def get_entropy_trend(self) -> float:
        """
        Calculate the recent trend in entropy values.
        
        Returns:
            Rate of change in entropy per minute (positive = increasing)
        """
        if len(self.measurements) < 2 or len(self.measurement_times) < 2:
            return 0.0
        
        # Use simple linear regression to calculate trend
        x = [(t - self.measurement_times[0]).total_seconds() / 60.0 
             for t in self.measurement_times[-self.window_size:]]
        y = self.measurements[-self.window_size:]
        
        if len(x) < 2:
            return 0.0
        
        # Calculate slope using least squares method
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        # Avoid division by zero
        if n * sum_xx - sum_x * sum_x == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that periodically checks entropy levels."""
        while self.running:
            try:
                # Calculate current entropy
                entropy = self.get_current_entropy()
                now = datetime.now()
                
                # Store measurement history
                self.measurements.append(entropy)
                self.measurement_times.append(now)
                
                # Keep only the most recent measurements
                if len(self.measurements) > self.window_size * 3:
                    self.measurements = self.measurements[-self.window_size * 3:]
                    self.measurement_times = self.measurement_times[-self.window_size * 3:]
                
                # Check for termination threshold
                if entropy >= self.termination_threshold:
                    self._handle_termination(entropy)
                    
                # Check for alert level changes
                self._check_alert_levels(entropy)
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in entropy monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_alert_levels(self, entropy: float) -> None:
        """
        Check if the current entropy level triggers an alert level change.
        
        Args:
            entropy: Current entropy value
        """
        # Determine current alert level based on thresholds
        new_alert = EntropyAlert.NORMAL
        for alert, threshold in self.thresholds.items():
            if entropy >= threshold:
                new_alert = alert
        
        # Only notify if alert level changed
        if new_alert != self.current_alert:
            old_alert = self.current_alert
            self.current_alert = new_alert
            
            # Get detailed metrics for the alert
            details = {
                'metrics': self.get_detailed_metrics(),
                'trend': self.get_entropy_trend(),
                'previous_level': old_alert,
                'timestamp': datetime.now()
            }
            
            # Notify subscribers
            self._notify_alert(new_alert, entropy, details)
    
    def _notify_alert(self, alert: EntropyAlert, entropy: float, details: Dict[str, Any]) -> None:
        """
        Notify all subscribers about an entropy alert.
        
        Args:
            alert: Current alert level
            entropy: Current entropy value
            details: Additional alert details
        """
        logger.info(f"Entropy alert: {alert.name} (value: {entropy:.4f})")
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber.on_entropy_alert(alert, entropy, details)
            except Exception as e:
                logger.error(f"Error notifying subscriber {subscriber}: {e}")
        
        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert, entropy, details)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _handle_termination(self, entropy: float) -> None:
        """
        Handle entropy exceeding termination threshold.
        
        Args:
            entropy: Current entropy value
        """
        details = {
            'metrics': self.get_detailed_metrics(),
            'trend': self.get_entropy_trend(),
            'alert_level': self.current_alert,
            'timestamp': datetime.now()
        }
        
        logger.warning(
            f"Entropy reached termination threshold: {entropy:.4f} "
            f"(threshold: {self.termination_threshold:.4f})"
        )
        
        # Check if any subscriber or callback wants to override termination
        should_terminate = True
        
        # Ask subscribers first
        for subscriber in self.subscribers:
            try:
                if not subscriber.on_termination_decision(entropy, details):
                    should_terminate = False
                    logger.info(f"Termination overridden by subscriber {subscriber}")
            except Exception as e:
                logger.error(f"Error in termination decision from {subscriber}: {e}")
        
        # Check callbacks if subscribers didn't override
        if should_terminate:
            for callback in self.termination_callbacks:
                try:
                    if not callback(entropy, details):
                        should_terminate = False
                        logger.info("Termination overridden by callback")
                        break
                except Exception as e:
                    logger.error(f"Error in termination callback: {e}")
        
        # Log the final decision
        if should_terminate:
            logger.critical(
                f"Initiating system termination due to critical entropy: {entropy:.4f}"
            )
            # The actual termination action would be implemented by subscribers
            # This could include safe shutdown, entropy reduction, or other mitigations 