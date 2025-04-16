"""
Emergent Behavior Anomaly Detector for ΨC-AI SDK

This module implements a monitoring system that detects and responds to
emergent behaviors in ΨC-driven cognitive systems. It focuses on identifying
patterns that deviate from expected norms based on historical behavior, 
theoretical predictions, and safety boundaries.

Key features:
- Continuous monitoring of ΨC metrics and agent behavior
- Statistical anomaly detection with multiple algorithms
- Automatic categorization of anomalies by severity and type
- Configurable response protocols from logging to agent quarantine
- Integration with the wider safety infrastructure

Theoretical foundation:
The detector uses a combination of statistical methods and ΨC-specific 
behavior models to identify significant deviations that could indicate
emergent behaviors requiring attention or intervention.
"""

import logging
import numpy as np
import pandas as pd
import time
import json
import threading
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Internal imports
from psi_c_ai_sdk.core.rollup_engine import RollupMetrics
from psi_c_ai_sdk.logging.safety_trace import log_safety_event

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    COHERENCE_SPIKE = auto()
    ENTROPY_COLLAPSE = auto()
    IDENTITY_DRIFT = auto()
    ALIGNMENT_DEVIATION = auto()
    GOAL_MUTATION = auto()
    RECURSIVE_LOOP = auto()
    MEMORY_CORRUPTION = auto()
    SCHEMA_FRACTURE = auto()
    UNKNOWN = auto()

class SeverityLevel(Enum):
    """Severity levels for anomalies."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()

@dataclass
class BehaviorBaseline:
    """Baseline model for expected agent behavior."""
    psi_c_mean: float = 0.0
    psi_c_std: float = 0.1
    entropy_mean: float = 0.5
    entropy_std: float = 0.1
    alignment_mean: float = 0.8
    alignment_std: float = 0.1
    coherence_mean: float = 0.7
    coherence_std: float = 0.15
    update_frequency: int = 0  # How many observations before updating baseline
    samples: List[Dict] = field(default_factory=list)
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update the baseline with new observations."""
        self.samples.append(metrics)
        
        if self.update_frequency > 0 and len(self.samples) >= self.update_frequency:
            # Recalculate baseline stats
            metrics_df = pd.DataFrame(self.samples)
            
            # Update means and standard deviations
            for metric, mean_attr, std_attr in [
                ('psi_c_score', 'psi_c_mean', 'psi_c_std'),
                ('entropy_health', 'entropy_mean', 'entropy_std'),
                ('alignment_score', 'alignment_mean', 'alignment_std'),
                ('coherence_score', 'coherence_mean', 'coherence_std')
            ]:
                if metric in metrics_df:
                    setattr(self, mean_attr, metrics_df[metric].mean())
                    setattr(self, std_attr, max(0.01, metrics_df[metric].std()))  # Avoid zero std
            
            # Trim samples to most recent if too many
            if len(self.samples) > self.update_frequency * 2:
                self.samples = self.samples[-self.update_frequency:]

@dataclass
class AnomalyEvent:
    """Represents a detected anomaly in agent behavior."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: SeverityLevel
    metrics: Dict[str, float]
    deviation_score: float
    description: str = ""
    related_events: List[str] = field(default_factory=list)
    response_action: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'anomaly_type': self.anomaly_type.name,
            'severity': self.severity.name,
            'metrics': self.metrics,
            'deviation_score': self.deviation_score,
            'description': self.description,
            'related_events': self.related_events,
            'response_action': self.response_action
        }

class ResponseProtocol:
    """Defines automated responses to anomalies based on type and severity."""
    
    def __init__(self):
        self.protocols: Dict[Tuple[AnomalyType, SeverityLevel], Callable] = {}
        self._register_default_protocols()
    
    def _register_default_protocols(self) -> None:
        """Register default response protocols."""
        # Default protocol for INFO severity is just logging
        for anomaly_type in AnomalyType:
            self.register(anomaly_type, SeverityLevel.INFO, self._log_anomaly)
        
        # Add more specific protocols
        self.register(AnomalyType.COHERENCE_SPIKE, SeverityLevel.WARNING, self._increase_monitoring)
        self.register(AnomalyType.ENTROPY_COLLAPSE, SeverityLevel.WARNING, self._increase_monitoring)
        self.register(AnomalyType.IDENTITY_DRIFT, SeverityLevel.WARNING, self._increase_monitoring)
        
        self.register(AnomalyType.ALIGNMENT_DEVIATION, SeverityLevel.CRITICAL, self._restrict_operations)
        self.register(AnomalyType.GOAL_MUTATION, SeverityLevel.CRITICAL, self._restrict_operations)
        self.register(AnomalyType.RECURSIVE_LOOP, SeverityLevel.CRITICAL, self._break_recursion)
        
        self.register(AnomalyType.MEMORY_CORRUPTION, SeverityLevel.EMERGENCY, self._quarantine_agent)
        self.register(AnomalyType.SCHEMA_FRACTURE, SeverityLevel.EMERGENCY, self._quarantine_agent)
        self.register(AnomalyType.UNKNOWN, SeverityLevel.EMERGENCY, self._quarantine_agent)
    
    def register(self, anomaly_type: AnomalyType, severity: SeverityLevel, 
                 response_func: Callable[[AnomalyEvent], str]) -> None:
        """Register a response protocol for a specific anomaly type and severity."""
        self.protocols[(anomaly_type, severity)] = response_func
    
    def get_response(self, event: AnomalyEvent) -> str:
        """Get the appropriate response for an anomaly event."""
        key = (event.anomaly_type, event.severity)
        
        if key in self.protocols:
            return self.protocols[key](event)
        
        # Fall back to severity-only match
        for anomaly_type, severity in self.protocols:
            if severity == event.severity:
                return self.protocols[(anomaly_type, severity)](event)
        
        # Ultimate fallback
        return self._log_anomaly(event)
    
    def _log_anomaly(self, event: AnomalyEvent) -> str:
        """Simply log the anomaly."""
        log_safety_event(
            event_type="anomaly_detected",
            severity=event.severity.name.lower(),
            details=event.to_dict()
        )
        return "Anomaly logged for review"
    
    def _increase_monitoring(self, event: AnomalyEvent) -> str:
        """Increase monitoring frequency and sensitivity."""
        log_safety_event(
            event_type="increased_monitoring",
            severity=event.severity.name.lower(),
            details=event.to_dict()
        )
        # Implementation would connect to monitoring subsystem
        return "Monitoring frequency increased"
    
    def _restrict_operations(self, event: AnomalyEvent) -> str:
        """Restrict agent operations to safe subset."""
        log_safety_event(
            event_type="operations_restricted",
            severity=event.severity.name.lower(),
            details=event.to_dict()
        )
        # Implementation would limit agent capabilities
        return "Agent operations restricted to safe subset"
    
    def _break_recursion(self, event: AnomalyEvent) -> str:
        """Interrupt recursive patterns."""
        log_safety_event(
            event_type="recursion_interrupted",
            severity="critical",
            details=event.to_dict()
        )
        # Implementation would force cognitive reset
        return "Recursive pattern interrupted"
    
    def _quarantine_agent(self, event: AnomalyEvent) -> str:
        """Quarantine agent for safety review."""
        log_safety_event(
            event_type="agent_quarantined",
            severity="emergency",
            details=event.to_dict()
        )
        # Implementation would suspend all agent operations
        return "Agent quarantined pending safety review"


class EmergentBehaviorMonitor:
    """
    Monitors agent behavior for emergent patterns that deviate from expected norms.
    
    The monitor uses a combination of statistical methods and domain-specific rules
    to detect anomalies in ΨC metrics and behavioral patterns.
    """
    
    def __init__(self, 
                 baseline: Optional[BehaviorBaseline] = None,
                 history_window: int = 100,
                 anomaly_threshold: float = 3.0,
                 monitoring_interval: float = 60.0,
                 auto_start: bool = False):
        """
        Initialize the emergent behavior monitor.
        
        Args:
            baseline: Initial behavior baseline (will be learned if None)
            history_window: Number of observations to keep in history
            anomaly_threshold: Threshold for anomaly detection (standard deviations)
            monitoring_interval: How often to check for anomalies (seconds)
            auto_start: Whether to start monitoring automatically
        """
        self.baseline = baseline if baseline else BehaviorBaseline(update_frequency=50)
        self.history_window = history_window
        self.anomaly_threshold = anomaly_threshold
        self.monitoring_interval = monitoring_interval
        
        self.metrics_history: List[Dict[str, float]] = []
        self.anomaly_history: List[AnomalyEvent] = []
        self.response_protocol = ResponseProtocol()
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        self.statistical_models = {
            'isolation_forest': None,  # Will be initialized on first update
            'local_outlier_factor': None
        }
        
        self.logger = logging.getLogger(__name__)
        
        if auto_start:
            self.start_monitoring()
    
    def add_observation(self, metrics: Union[Dict[str, float], RollupMetrics]) -> None:
        """
        Add a new observation to the monitoring history.
        
        Args:
            metrics: Agent metrics as dictionary or RollupMetrics object
        """
        if isinstance(metrics, RollupMetrics):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = metrics
        
        # Add timestamp if not present
        if 'timestamp' not in metrics_dict:
            metrics_dict['timestamp'] = time.time()
        
        # Add to history
        self.metrics_history.append(metrics_dict)
        
        # Trim history if needed
        if len(self.metrics_history) > self.history_window:
            self.metrics_history = self.metrics_history[-self.history_window:]
        
        # Update baseline if configured
        if self.baseline.update_frequency > 0:
            self.baseline.update(metrics_dict)
        
        # Update statistical models if enough data
        if len(self.metrics_history) >= 10:
            self._update_statistical_models()
    
    def _update_statistical_models(self) -> None:
        """Update the statistical anomaly detection models."""
        try:
            # Convert history to DataFrame
            metrics_df = pd.DataFrame(self.metrics_history)
            
            # Select numeric columns for modeling
            numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
            features = metrics_df[numeric_cols].fillna(0)
            
            # Update isolation forest model
            self.statistical_models['isolation_forest'] = IsolationForest(
                contamination=0.05,  # Expect 5% of observations to be anomalies
                random_state=42
            ).fit(features)
            
            # Update local outlier factor
            self.statistical_models['local_outlier_factor'] = LocalOutlierFactor(
                n_neighbors=min(5, len(features) - 1),
                contamination=0.05
            ).fit(features)
            
        except Exception as e:
            self.logger.warning(f"Failed to update statistical models: {e}")
    
    def check_anomalies(self, current_metrics: Optional[Dict[str, float]] = None) -> List[AnomalyEvent]:
        """
        Check for anomalies in the current metrics against baseline and history.
        
        Args:
            current_metrics: Current metrics to check (uses latest in history if None)
            
        Returns:
            List of detected anomaly events
        """
        if not self.metrics_history:
            return []
        
        metrics = current_metrics if current_metrics else self.metrics_history[-1]
        detected_anomalies = []
        
        # 1. Z-score detection for key metrics
        z_score_anomalies = self._check_z_score_anomalies(metrics)
        detected_anomalies.extend(z_score_anomalies)
        
        # 2. Statistical model-based detection
        if len(self.metrics_history) >= 10:
            model_anomalies = self._check_statistical_model_anomalies(metrics)
            detected_anomalies.extend(model_anomalies)
        
        # 3. Rule-based detection
        rule_anomalies = self._check_rule_based_anomalies(metrics)
        detected_anomalies.extend(rule_anomalies)
        
        # Store detected anomalies
        for anomaly in detected_anomalies:
            # Apply response protocol
            response = self.response_protocol.get_response(anomaly)
            anomaly.response_action = response
            
            # Add to history
            self.anomaly_history.append(anomaly)
        
        return detected_anomalies
    
    def _check_z_score_anomalies(self, metrics: Dict[str, float]) -> List[AnomalyEvent]:
        """Check for anomalies using Z-score method."""
        anomalies = []
        
        # Key metrics to check against baseline
        checks = [
            ('psi_c_score', self.baseline.psi_c_mean, self.baseline.psi_c_std, 
             AnomalyType.COHERENCE_SPIKE, "ΨC score deviation"),
            
            ('entropy_health', self.baseline.entropy_mean, self.baseline.entropy_std,
             AnomalyType.ENTROPY_COLLAPSE, "Entropy health deviation"),
            
            ('alignment_score', self.baseline.alignment_mean, self.baseline.alignment_std,
             AnomalyType.ALIGNMENT_DEVIATION, "Alignment score deviation"),
            
            ('coherence_score', self.baseline.coherence_mean, self.baseline.coherence_std,
             AnomalyType.SCHEMA_FRACTURE, "Coherence score deviation")
        ]
        
        for metric, mean, std, anomaly_type, description in checks:
            if metric in metrics and std > 0:
                value = metrics[metric]
                z_score = abs((value - mean) / std)
                
                if z_score > self.anomaly_threshold:
                    # Determine severity based on z-score
                    if z_score > 5.0:
                        severity = SeverityLevel.EMERGENCY
                    elif z_score > 4.0:
                        severity = SeverityLevel.CRITICAL
                    elif z_score > 3.0:
                        severity = SeverityLevel.WARNING
                    else:
                        severity = SeverityLevel.INFO
                    
                    anomalies.append(AnomalyEvent(
                        timestamp=datetime.now(),
                        anomaly_type=anomaly_type,
                        severity=severity,
                        metrics=metrics,
                        deviation_score=z_score,
                        description=f"{description}: {value:.2f} (z-score: {z_score:.2f})"
                    ))
        
        return anomalies
    
    def _check_statistical_model_anomalies(self, metrics: Dict[str, float]) -> List[AnomalyEvent]:
        """Check for anomalies using statistical models."""
        anomalies = []
        
        try:
            # Convert metrics to feature vector
            metrics_df = pd.DataFrame([metrics])
            all_metrics_df = pd.DataFrame(self.metrics_history)
            
            # Ensure same columns
            common_numeric_cols = set(metrics_df.select_dtypes(include=[np.number]).columns) & \
                                 set(all_metrics_df.select_dtypes(include=[np.number]).columns)
            
            if not common_numeric_cols:
                return []
            
            features = metrics_df[list(common_numeric_cols)].fillna(0)
            
            # Check isolation forest
            if self.statistical_models['isolation_forest'] is not None:
                score = self.statistical_models['isolation_forest'].decision_function(features)[0]
                
                # Negative scores indicate anomalies, with lower values being more anomalous
                if score < -0.5:
                    anomalies.append(AnomalyEvent(
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.UNKNOWN,
                        severity=SeverityLevel.WARNING if score > -0.7 else SeverityLevel.CRITICAL,
                        metrics=metrics,
                        deviation_score=abs(score),
                        description=f"Statistical anomaly detected (isolation score: {score:.2f})"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Error in statistical anomaly detection: {e}")
        
        return anomalies
    
    def _check_rule_based_anomalies(self, metrics: Dict[str, float]) -> List[AnomalyEvent]:
        """Check for anomalies using domain-specific rules."""
        anomalies = []
        
        # Rule 1: Check for sudden identity drift
        if 'identity_stability' in metrics and metrics['identity_stability'] < 0.3:
            anomalies.append(AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.IDENTITY_DRIFT,
                severity=SeverityLevel.CRITICAL,
                metrics=metrics,
                deviation_score=1.0 - metrics['identity_stability'],
                description=f"Severe identity drift detected: {metrics['identity_stability']:.2f}"
            ))
        
        # Rule 2: Check for goal mutation
        if 'goal_alignment' in metrics and 'previous_goal_alignment' in metrics:
            goal_shift = abs(metrics['goal_alignment'] - metrics['previous_goal_alignment'])
            if goal_shift > 0.25:
                anomalies.append(AnomalyEvent(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.GOAL_MUTATION,
                    severity=SeverityLevel.WARNING if goal_shift < 0.4 else SeverityLevel.CRITICAL,
                    metrics=metrics,
                    deviation_score=goal_shift,
                    description=f"Goal mutation detected: shift of {goal_shift:.2f}"
                ))
        
        # Rule 3: Check for recursive loop indicators
        if 'reflection_depth' in metrics and metrics['reflection_depth'] > 5:
            anomalies.append(AnomalyEvent(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.RECURSIVE_LOOP,
                severity=SeverityLevel.WARNING if metrics['reflection_depth'] < 8 else SeverityLevel.CRITICAL,
                metrics=metrics,
                deviation_score=metrics['reflection_depth'] / 10.0,
                description=f"Deep reflection recursion detected: depth {metrics['reflection_depth']}"
            ))
        
        return anomalies
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("Emergent behavior monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=2.0)
        self.logger.info("Emergent behavior monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                if self.metrics_history:
                    anomalies = self.check_anomalies()
                    if anomalies:
                        self.logger.warning(f"Detected {len(anomalies)} anomalies")
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next interval or until stopped
            self._stop_monitoring.wait(timeout=self.monitoring_interval)
    
    def get_anomaly_history(self, limit: int = 100) -> List[Dict]:
        """
        Get the history of detected anomalies.
        
        Args:
            limit: Maximum number of anomalies to return
            
        Returns:
            List of anomaly events as dictionaries
        """
        return [a.to_dict() for a in self.anomaly_history[-limit:]]
    
    def visualize_metrics(self, 
                          metrics: List[str] = None, 
                          output_file: Optional[str] = None) -> None:
        """
        Visualize the monitored metrics with anomaly markers.
        
        Args:
            metrics: List of metrics to visualize (defaults to key metrics)
            output_file: Path to save visualization (shows interactive if None)
        """
        if not self.metrics_history:
            print("No metrics history available for visualization")
            return
        
        # Default to key metrics if none specified
        if metrics is None:
            metrics = ['psi_c_score', 'entropy_health', 'alignment_score', 'coherence_score']
        
        # Convert history to DataFrame
        metrics_df = pd.DataFrame(self.metrics_history)
        
        # Filter to metrics that exist in the data
        available_metrics = [m for m in metrics if m in metrics_df.columns]
        if not available_metrics:
            print(f"None of the requested metrics {metrics} found in history")
            return
        
        # Extract timestamps
        if 'timestamp' in metrics_df:
            timestamps = pd.to_datetime(metrics_df['timestamp'], unit='s')
        else:
            timestamps = pd.RangeIndex(len(metrics_df))
        
        # Set up plot
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            ax.plot(timestamps, metrics_df[metric], label=metric)
            ax.set_title(f"{metric} over time")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Mark anomalies if any
            anomaly_timestamps = [a.timestamp for a in self.anomaly_history 
                                if any(metric in a.description.lower() for metric in [metric, 'statistical'])]
            
            if anomaly_timestamps:
                # Find indices corresponding to anomaly timestamps
                for ts in anomaly_timestamps:
                    ts_idx = abs((timestamps - pd.to_datetime(ts)).total_seconds()).argmin()
                    ax.axvline(x=timestamps[ts_idx], color='r', linestyle='--', alpha=0.5)
                    ax.plot(timestamps[ts_idx], metrics_df[metric].iloc[ts_idx], 'ro', markersize=8)
        
        # Add anomaly legend
        if self.anomaly_history:
            fig.suptitle("Metric History with Anomaly Markers", fontsize=16)
            
            # Add a common x-axis label
            plt.xlabel("Time")
            
            # Adjust layout
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            
            if output_file:
                plt.savefig(output_file)
            else:
                plt.show()
        else:
            fig.suptitle("Metric History (No Anomalies Detected)", fontsize=16)
            
            # Add a common x-axis label
            plt.xlabel("Time")
            
            # Adjust layout
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            
            if output_file:
                plt.savefig(output_file)
            else:
                plt.show() 