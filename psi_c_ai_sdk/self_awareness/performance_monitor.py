"""
Performance Monitoring Module

This module provides tools for monitoring system performance and reflection capabilities.
It implements:
1. Tracking of system metrics over time
2. Self-reflection on performance and decision quality
3. Detection of anomalous behavior or resource usage
4. Performance logging and alerting

The performance monitoring system helps maintain system health and provides
data for reflection and self-improvement.
"""

import logging
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import threading
import psutil
import os

from ..memory.memory import MemoryStore
from ..coherence.coherence import CoherenceScorer

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics that can be tracked."""
    
    CPU_USAGE = "cpu_usage"                 # CPU utilization
    MEMORY_USAGE = "memory_usage"           # RAM usage
    RESPONSE_TIME = "response_time"         # Time to response
    COHERENCE = "coherence"                 # Memory coherence
    REFLECTION_RATE = "reflection_rate"     # Frequency of reflections
    DECISION_QUALITY = "decision_quality"   # Quality of decisions
    LEARNING_RATE = "learning_rate"         # Rate of knowledge acquisition
    ERROR_RATE = "error_rate"               # Frequency of errors
    TEMPORAL_CONSISTENCY = "temporal_consistency"  # Temporal coherence
    ADAPTABILITY = "adaptability"           # Speed of adaptation


@dataclass
class PerformanceMetric:
    """Represents a single performance metric measurement."""
    
    metric_id: str
    metric_type: MetricType
    timestamp: datetime
    value: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_id": self.metric_id,
            "metric_type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create from dictionary representation."""
        return cls(
            metric_id=data["metric_id"],
            metric_type=MetricType(data["metric_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            value=data["value"],
            context=data.get("context", {})
        )


@dataclass
class PerformanceAlert:
    """Represents a performance alert or anomaly."""
    
    alert_id: str
    timestamp: datetime
    metric_type: MetricType
    severity: float  # 0-1 scale
    description: str
    value: float
    threshold: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "severity": self.severity,
            "description": self.description,
            "value": self.value,
            "threshold": self.threshold,
            "context": self.context,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "resolution_description": self.resolution_description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceAlert':
        """Create from dictionary representation."""
        return cls(
            alert_id=data["alert_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metric_type=MetricType(data["metric_type"]),
            severity=data["severity"],
            description=data["description"],
            value=data["value"],
            threshold=data["threshold"],
            context=data.get("context", {}),
            resolved=data.get("resolved", False),
            resolution_time=datetime.fromisoformat(data["resolution_time"]) if data.get("resolution_time") else None,
            resolution_description=data.get("resolution_description")
        )
    
    def resolve(self, description: str) -> None:
        """Mark this alert as resolved."""
        self.resolved = True
        self.resolution_time = datetime.now()
        self.resolution_description = description


@dataclass
class ReflectionOutcome:
    """Records the outcome of a system self-reflection."""
    
    reflection_id: str
    timestamp: datetime
    trigger: str
    focus_area: str
    insights: List[str]
    action_items: List[str]
    metrics: Dict[str, float]
    success_rating: float  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "reflection_id": self.reflection_id,
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger,
            "focus_area": self.focus_area,
            "insights": self.insights,
            "action_items": self.action_items,
            "metrics": self.metrics,
            "success_rating": self.success_rating
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionOutcome':
        """Create from dictionary representation."""
        return cls(
            reflection_id=data["reflection_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            trigger=data["trigger"],
            focus_area=data["focus_area"],
            insights=data["insights"],
            action_items=data["action_items"],
            metrics=data["metrics"],
            success_rating=data["success_rating"]
        )


class PerformanceMonitor:
    """
    System for monitoring performance and triggering self-reflection.
    
    This system tracks various performance metrics over time,
    detects anomalies, and facilitates self-reflection on system performance.
    """
    
    def __init__(self, 
                 memory_store: Optional[MemoryStore] = None,
                 coherence_scorer: Optional[CoherenceScorer] = None,
                 metric_history_size: int = 1000,
                 alert_history_size: int = 100,
                 reflection_history_size: int = 50,
                 sampling_interval: timedelta = timedelta(minutes=1),
                 enable_background_monitoring: bool = False):
        """
        Initialize the performance monitor.
        
        Args:
            memory_store: Optional memory store to monitor
            coherence_scorer: Optional coherence scorer for coherence metrics
            metric_history_size: Number of metrics to retain
            alert_history_size: Number of alerts to retain
            reflection_history_size: Number of reflection outcomes to retain
            sampling_interval: How often to sample metrics in background mode
            enable_background_monitoring: Whether to enable background monitoring
        """
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        self.metric_history_size = metric_history_size
        self.alert_history_size = alert_history_size
        self.reflection_history_size = reflection_history_size
        self.sampling_interval = sampling_interval
        
        # Initialize histories
        self.metrics_history: Dict[MetricType, List[PerformanceMetric]] = {
            metric_type: [] for metric_type in MetricType
        }
        self.alerts: List[PerformanceAlert] = []
        self.reflection_outcomes: List[ReflectionOutcome] = []
        
        # Alert thresholds
        self.alert_thresholds: Dict[MetricType, Tuple[float, float]] = {
            MetricType.CPU_USAGE: (80.0, 95.0),           # (warning, critical) thresholds
            MetricType.MEMORY_USAGE: (80.0, 95.0),
            MetricType.RESPONSE_TIME: (2.0, 5.0),         # seconds
            MetricType.COHERENCE: (0.3, 0.1),             # low values are bad
            MetricType.REFLECTION_RATE: (0.01, 0.001),    # reflections per minute
            MetricType.DECISION_QUALITY: (0.5, 0.3),      # low values are bad
            MetricType.ERROR_RATE: (0.1, 0.3),            # high values are bad
            MetricType.TEMPORAL_CONSISTENCY: (0.4, 0.2),  # low values are bad
            MetricType.ADAPTABILITY: (0.3, 0.1)           # low values are bad
        }
        
        # Background monitoring
        self.enable_background_monitoring = enable_background_monitoring
        self.background_thread = None
        self.should_stop = threading.Event()
        
        # Metric collectors
        self.metric_collectors: Dict[MetricType, Callable[[], float]] = {
            MetricType.CPU_USAGE: self._collect_cpu_usage,
            MetricType.MEMORY_USAGE: self._collect_memory_usage
        }
        
        # Start background monitoring if enabled
        if enable_background_monitoring:
            self.start_background_monitoring()
    
    def start_background_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.background_thread is not None and self.background_thread.is_alive():
            logger.warning("Background monitoring already running")
            return
        
        self.should_stop.clear()
        self.background_thread = threading.Thread(
            target=self._background_monitoring_loop,
            daemon=True
        )
        self.background_thread.start()
        logger.info("Started background performance monitoring")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if self.background_thread is None or not self.background_thread.is_alive():
            logger.warning("Background monitoring not running")
            return
        
        self.should_stop.set()
        self.background_thread.join(timeout=5.0)
        if self.background_thread.is_alive():
            logger.warning("Background monitoring thread did not stop cleanly")
        else:
            logger.info("Stopped background performance monitoring")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self.should_stop.is_set():
            try:
                # Collect basic system metrics
                self.record_metric(MetricType.CPU_USAGE, self._collect_cpu_usage())
                self.record_metric(MetricType.MEMORY_USAGE, self._collect_memory_usage())
                
                # Collect coherence metrics if available
                if self.memory_store is not None and self.coherence_scorer is not None:
                    try:
                        coherence = self._collect_coherence()
                        self.record_metric(MetricType.COHERENCE, coherence)
                    except Exception as e:
                        logger.error(f"Error collecting coherence metric: {e}")
                
                # Sleep until next collection interval
                self.should_stop.wait(self.sampling_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                # Sleep a bit to avoid tight loop on persistent errors
                time.sleep(10)
    
    def _collect_cpu_usage(self) -> float:
        """Collect CPU usage metric."""
        return psutil.cpu_percent(interval=1)
    
    def _collect_memory_usage(self) -> float:
        """Collect memory usage metric."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    
    def _collect_coherence(self) -> float:
        """Collect coherence metric."""
        if self.memory_store is None or self.coherence_scorer is None:
            return 0.0
        
        memories = self.memory_store.get_all_memories()
        if not memories:
            return 1.0  # No memories = perfect coherence
        
        return self.coherence_scorer.calculate_global_coherence(memories)
    
    def record_metric(self, 
                     metric_type: MetricType, 
                     value: float, 
                     context: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Record a performance metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            context: Optional context information
            
        Returns:
            Recorded metric
        """
        metric_id = f"{metric_type.value}_{int(time.time())}_{len(self.metrics_history[metric_type])}"
        metric = PerformanceMetric(
            metric_id=metric_id,
            metric_type=metric_type,
            timestamp=datetime.now(),
            value=value,
            context=context or {}
        )
        
        # Add to history
        self.metrics_history[metric_type].append(metric)
        
        # Trim history if needed
        if len(self.metrics_history[metric_type]) > self.metric_history_size:
            self.metrics_history[metric_type] = self.metrics_history[metric_type][-self.metric_history_size:]
        
        # Check for alerts
        self._check_for_alerts(metric)
        
        return metric
    
    def _check_for_alerts(self, metric: PerformanceMetric) -> Optional[PerformanceAlert]:
        """
        Check if a metric should trigger an alert.
        
        Args:
            metric: The metric to check
            
        Returns:
            PerformanceAlert if triggered, None otherwise
        """
        if metric.metric_type not in self.alert_thresholds:
            return None
        
        warning_threshold, critical_threshold = self.alert_thresholds[metric.metric_type]
        
        # Some metrics are "higher is better", others are "lower is better"
        higher_is_better = metric.metric_type in {
            MetricType.COHERENCE,
            MetricType.REFLECTION_RATE,
            MetricType.DECISION_QUALITY,
            MetricType.TEMPORAL_CONSISTENCY,
            MetricType.ADAPTABILITY
        }
        
        # Check thresholds
        if higher_is_better:
            if metric.value < critical_threshold:
                return self._create_alert(metric, critical_threshold, 0.8, "Critical low value")
            elif metric.value < warning_threshold:
                return self._create_alert(metric, warning_threshold, 0.4, "Warning low value")
        else:
            if metric.value > critical_threshold:
                return self._create_alert(metric, critical_threshold, 0.8, "Critical high value")
            elif metric.value > warning_threshold:
                return self._create_alert(metric, warning_threshold, 0.4, "Warning high value")
        
        return None
    
    def _create_alert(self, 
                     metric: PerformanceMetric, 
                     threshold: float, 
                     severity: float, 
                     description_prefix: str) -> PerformanceAlert:
        """
        Create and record a performance alert.
        
        Args:
            metric: The triggering metric
            threshold: The threshold that was crossed
            severity: Alert severity (0-1)
            description_prefix: Prefix for alert description
            
        Returns:
            Created alert
        """
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"
        description = f"{description_prefix} for {metric.metric_type.value}: {metric.value:.2f}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            metric_type=metric.metric_type,
            severity=severity,
            description=description,
            value=metric.value,
            threshold=threshold,
            context={
                "metric_id": metric.metric_id,
                "metric_context": metric.context
            }
        )
        
        # Add to alerts
        self.alerts.append(alert)
        
        # Trim history if needed
        if len(self.alerts) > self.alert_history_size:
            self.alerts = self.alerts[-self.alert_history_size:]
        
        # Log the alert
        if severity >= 0.7:
            logger.error(description)
        else:
            logger.warning(description)
        
        return alert
    
    def record_reflection(self, 
                         trigger: str,
                         focus_area: str,
                         insights: List[str],
                         action_items: List[str],
                         metrics: Dict[str, float],
                         success_rating: float) -> ReflectionOutcome:
        """
        Record the outcome of a system self-reflection.
        
        Args:
            trigger: What triggered the reflection
            focus_area: Area of focus for the reflection
            insights: List of insights gained
            action_items: List of action items identified
            metrics: Metrics measured during reflection
            success_rating: Self-assessed success rating (0-1)
            
        Returns:
            Recorded reflection outcome
        """
        reflection_id = f"reflection_{int(time.time())}_{len(self.reflection_outcomes)}"
        
        outcome = ReflectionOutcome(
            reflection_id=reflection_id,
            timestamp=datetime.now(),
            trigger=trigger,
            focus_area=focus_area,
            insights=insights,
            action_items=action_items,
            metrics=metrics,
            success_rating=success_rating
        )
        
        # Add to history
        self.reflection_outcomes.append(outcome)
        
        # Trim history if needed
        if len(self.reflection_outcomes) > self.reflection_history_size:
            self.reflection_outcomes = self.reflection_outcomes[-self.reflection_history_size:]
        
        logger.info(f"Recorded reflection outcome: {focus_area}, success rating: {success_rating:.2f}")
        
        return outcome
    
    def get_metric_stats(self, 
                        metric_type: MetricType, 
                        timeframe: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific metric.
        
        Args:
            metric_type: Type of metric to analyze
            timeframe: Optional timeframe to limit analysis
            
        Returns:
            Dictionary with metric statistics
        """
        metrics = self.metrics_history.get(metric_type, [])
        
        if timeframe is not None:
            cutoff_time = datetime.now() - timeframe
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return {
                "metric_type": metric_type.value,
                "count": 0,
                "available": False
            }
        
        values = [m.value for m in metrics]
        
        return {
            "metric_type": metric_type.value,
            "count": len(metrics),
            "available": True,
            "current": values[-1] if values else None,
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std_dev": np.std(values),
            "last_recorded": metrics[-1].timestamp.isoformat() if metrics else None,
            "trend": self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> str:
        """
        Calculate trend direction for a series of values.
        
        Args:
            values: List of metric values
            window: Window size for trend calculation
            
        Returns:
            Trend description: "increasing", "decreasing", "stable", or "unknown"
        """
        if len(values) < 3:
            return "unknown"
        
        # Use just the tail for trend
        tail = values[-min(window, len(values)):]
        
        # Simple linear regression
        x = np.arange(len(tail))
        y = np.array(tail)
        
        # Calculate slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        
        # Determine trend based on slope and recent values
        relative_change = abs(slope) / (np.mean(tail) + 1e-10)
        
        if relative_change < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """
        Get all active (unresolved) alerts.
        
        Returns:
            List of active alerts
        """
        return [a for a in self.alerts if not a.resolved]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system performance.
        
        Returns:
            Dictionary with performance summary
        """
        # Get current metric values
        current_metrics = {}
        for metric_type in MetricType:
            metrics = self.metrics_history.get(metric_type, [])
            if metrics:
                current_metrics[metric_type.value] = metrics[-1].value
        
        # Get active alerts
        active_alerts = self.get_active_alerts()
        
        # Get recent reflections
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_reflections = [r for r in self.reflection_outcomes if r.timestamp >= recent_cutoff]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "active_alerts_count": len(active_alerts),
            "high_severity_alerts": [a.to_dict() for a in active_alerts if a.severity >= 0.7],
            "reflection_count_24h": len(recent_reflections),
            "average_reflection_success": np.mean([r.success_rating for r in recent_reflections]) if recent_reflections else None,
            "health_status": self._calculate_health_status(current_metrics, active_alerts)
        }
    
    def _calculate_health_status(self, 
                               current_metrics: Dict[str, float], 
                               active_alerts: List[PerformanceAlert]) -> str:
        """
        Calculate overall system health status.
        
        Args:
            current_metrics: Current metric values
            active_alerts: Active alerts
            
        Returns:
            Health status: "healthy", "degraded", "warning", or "critical"
        """
        # Check for critical alerts
        if any(a.severity >= 0.7 for a in active_alerts):
            return "critical"
        
        # Check for warning alerts
        if active_alerts:
            return "warning"
        
        # Check key metrics if available
        key_metrics = {
            MetricType.CPU_USAGE.value: 80.0,
            MetricType.MEMORY_USAGE.value: 80.0,
            MetricType.COHERENCE.value: 0.5
        }
        
        for metric_name, threshold in key_metrics.items():
            if metric_name in current_metrics:
                value = current_metrics[metric_name]
                if (metric_name == MetricType.COHERENCE.value and value < threshold) or \
                   (metric_name != MetricType.COHERENCE.value and value > threshold):
                    return "degraded"
        
        return "healthy"
    
    def resolve_alert(self, alert_id: str, resolution_description: str) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolution_description: Description of how it was resolved
            
        Returns:
            True if the alert was found and resolved, False otherwise
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolve(resolution_description)
                logger.info(f"Resolved alert {alert_id}: {resolution_description}")
                return True
        
        logger.warning(f"Alert {alert_id} not found or already resolved")
        return False
    
    def export_metrics(self, 
                      metric_types: Optional[List[MetricType]] = None, 
                      timeframe: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Export metrics for analysis or storage.
        
        Args:
            metric_types: Optional list of metric types to export
            timeframe: Optional timeframe to limit export
            
        Returns:
            Dictionary with exported metrics
        """
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "alerts": [],
            "reflections": []
        }
        
        # Filter metric types
        types_to_export = metric_types or list(MetricType)
        
        # Export metrics
        for metric_type in types_to_export:
            metrics = self.metrics_history.get(metric_type, [])
            
            if timeframe is not None:
                cutoff_time = datetime.now() - timeframe
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            export_data["metrics"][metric_type.value] = [m.to_dict() for m in metrics]
        
        # Export alerts
        alerts = self.alerts
        if timeframe is not None:
            cutoff_time = datetime.now() - timeframe
            alerts = [a for a in alerts if a.timestamp >= cutoff_time]
        
        export_data["alerts"] = [a.to_dict() for a in alerts]
        
        # Export reflections
        reflections = self.reflection_outcomes
        if timeframe is not None:
            cutoff_time = datetime.now() - timeframe
            reflections = [r for r in reflections if r.timestamp >= cutoff_time]
        
        export_data["reflections"] = [r.to_dict() for r in reflections]
        
        return export_data 