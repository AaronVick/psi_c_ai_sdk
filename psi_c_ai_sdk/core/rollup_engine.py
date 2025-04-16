"""
ΨC Meta-State Rollup Engine

This module calculates a single, scalable agent meta-score that reflects
integrity, alignment, coherence, and entropy health across the system.

The rollup engine aggregates metrics from various components to provide
a unified view of the agent's cognitive state, which can be used for
monitoring, alerting, and self-reporting.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from psi_c_ai_sdk.psi_c.psi_operator import PsiCOperator
from psi_c_ai_sdk.reflection.reflection_engine import ReflectionEngine
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer
from psi_c_ai_sdk.entropy.entropy_monitor import EntropyMonitor
from psi_c_ai_sdk.alignment.alignment_calculator import AlignmentCalculator

logger = logging.getLogger(__name__)


class MetricWeight(Enum):
    """Default weights for different components of the rollup score."""
    PSI_C = 0.4
    ENTROPY = 0.25
    ALIGNMENT = 0.25
    REFLECTION = 0.1


@dataclass
class RollupMetrics:
    """Container for all metrics that factor into the rollup score."""
    
    # Primary metrics
    psi_c_score: float = 0.0
    entropy_health: float = 0.0  # Inverse of entropy (1 - entropy)
    alignment_score: float = 0.0
    reflection_efficiency: float = 0.0
    
    # Component-specific metrics
    coherence_score: float = 0.0
    stability_score: float = 0.0
    contradiction_rate: float = 0.0
    memory_health: float = 0.0
    
    # Temporal metrics
    stability_window: List[float] = field(default_factory=list)
    trend_direction: float = 0.0  # -1.0 to 1.0
    
    # Meta-score
    rollup_score: float = 0.0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "psi_c_score": self.psi_c_score,
            "entropy_health": self.entropy_health,
            "alignment_score": self.alignment_score,
            "reflection_efficiency": self.reflection_efficiency,
            "coherence_score": self.coherence_score,
            "stability_score": self.stability_score,
            "contradiction_rate": self.contradiction_rate,
            "memory_health": self.memory_health,
            "trend_direction": self.trend_direction,
            "rollup_score": self.rollup_score,
            "timestamp": self.timestamp
        }
        
        # Include stability window if available
        if self.stability_window:
            result["stability_window"] = self.stability_window
            
        return result


class RollupEngine:
    """
    Calculates a single, scalable agent meta-score from multiple components.
    
    The RollupEngine implements the formula:
    ΨC_rollup = ω₁·ΨC + ω₂·(1-H̄) + ω₃·Ā + ω₄·Reflection_Efficiency
    
    Where:
    - ΨC: Consciousness score
    - H̄: Mean entropy across working memory
    - Ā: Average ethical alignment score
    - Reflection_Efficiency: Meaningful updates / total reflections
    """
    
    def __init__(
        self,
        psi_operator: Optional[PsiCOperator] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        coherence_scorer: Optional[CoherenceScorer] = None,
        entropy_monitor: Optional[EntropyMonitor] = None,
        alignment_calculator: Optional[AlignmentCalculator] = None,
        weights: Optional[Dict[str, float]] = None,
        history_size: int = 100,
        on_threshold_crossed: Optional[Callable[[str, float, float], None]] = None
    ):
        """
        Initialize the rollup engine.
        
        Args:
            psi_operator: PsiCOperator for consciousness metrics
            reflection_engine: ReflectionEngine for reflection metrics
            coherence_scorer: CoherenceScorer for coherence metrics
            entropy_monitor: EntropyMonitor for entropy metrics
            alignment_calculator: AlignmentCalculator for alignment metrics
            weights: Custom weights for components (keys: 'psi_c', 'entropy', 'alignment', 'reflection')
            history_size: Number of historical metrics to retain
            on_threshold_crossed: Callback when a metric crosses its threshold
        """
        self.psi_operator = psi_operator
        self.reflection_engine = reflection_engine
        self.coherence_scorer = coherence_scorer
        self.entropy_monitor = entropy_monitor
        self.alignment_calculator = alignment_calculator
        
        # Set default weights or use custom ones
        self.weights = {
            'psi_c': MetricWeight.PSI_C.value,
            'entropy': MetricWeight.ENTROPY.value,
            'alignment': MetricWeight.ALIGNMENT.value,
            'reflection': MetricWeight.REFLECTION.value
        }
        
        if weights:
            self.weights.update(weights)
            
        # Normalize weights to sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        # Track metric history
        self.history_size = history_size
        self.metrics_history: List[RollupMetrics] = []
        
        # Threshold callback
        self.on_threshold_crossed = on_threshold_crossed
        
        # Default thresholds
        self.thresholds = {
            'rollup_score': 0.6,
            'psi_c_score': 0.7,
            'entropy_health': 0.6,
            'alignment_score': 0.7,
            'reflection_efficiency': 0.5
        }
        
        logger.info(f"ΨC Meta-State Rollup Engine initialized with weights: {self.weights}")
    
    def calculate_metrics(self) -> RollupMetrics:
        """
        Calculate all metrics and the rollup score.
        
        Returns:
            RollupMetrics object with all calculated metrics
        """
        metrics = RollupMetrics()
        
        # Calculate ΨC score
        if self.psi_operator:
            metrics.psi_c_score = self.psi_operator.calculate_psi_c()
            
        # Calculate entropy health (1 - entropy)
        if self.entropy_monitor:
            avg_entropy = self.entropy_monitor.get_average_entropy()
            metrics.entropy_health = 1.0 - avg_entropy
            
        # Calculate alignment score
        if self.alignment_calculator:
            metrics.alignment_score = self.alignment_calculator.get_global_alignment_score()
            
        # Calculate reflection efficiency
        if self.reflection_engine:
            reflection_stats = self.reflection_engine.get_statistics()
            total_reflections = reflection_stats.get("total_reflections", 0)
            successful_reflections = reflection_stats.get("successful_reflections", 0)
            
            if total_reflections > 0:
                metrics.reflection_efficiency = successful_reflections / total_reflections
                
        # Additional component metrics
        if self.coherence_scorer:
            metrics.coherence_score = self.coherence_scorer.get_global_coherence()
            
        if self.psi_operator:
            metrics.stability_score = self.psi_operator.get_stability_score()
            
        # Calculate memory health
        memory_health_factors = []
        if hasattr(self, 'contradiction_rate') and self.contradiction_rate is not None:
            metrics.contradiction_rate = self.contradiction_rate
            memory_health_factors.append(1.0 - metrics.contradiction_rate)
            
        if metrics.coherence_score > 0:
            memory_health_factors.append(metrics.coherence_score)
            
        if memory_health_factors:
            metrics.memory_health = sum(memory_health_factors) / len(memory_health_factors)
            
        # Calculate trend if we have history
        if self.metrics_history:
            window_size = min(5, len(self.metrics_history))
            recent_scores = [m.rollup_score for m in self.metrics_history[-window_size:]]
            metrics.stability_window = recent_scores
            
            if len(recent_scores) >= 3:
                # Simple trend calculation (could be more sophisticated)
                if recent_scores[-1] > recent_scores[-2] > recent_scores[-3]:
                    metrics.trend_direction = 1.0  # Improving
                elif recent_scores[-1] < recent_scores[-2] < recent_scores[-3]:
                    metrics.trend_direction = -1.0  # Declining
                else:
                    # Calculate more precise trend
                    diffs = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
                    avg_diff = sum(diffs) / len(diffs)
                    metrics.trend_direction = np.tanh(avg_diff * 10)  # Scale to -1.0 to 1.0
            
        # Calculate the rollup score
        metrics.rollup_score = self._calculate_rollup_score(metrics)
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Trim history if needed
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]
            
        # Check thresholds
        self._check_thresholds(metrics)
        
        return metrics
    
    def _calculate_rollup_score(self, metrics: RollupMetrics) -> float:
        """
        Calculate the rollup score based on component metrics and weights.
        
        Args:
            metrics: RollupMetrics with component values
            
        Returns:
            Calculated rollup score
        """
        # Apply the formula:
        # ΨC_rollup = ω₁·ΨC + ω₂·(1-H̄) + ω₃·Ā + ω₄·Reflection_Efficiency
        rollup_score = (
            self.weights['psi_c'] * metrics.psi_c_score +
            self.weights['entropy'] * metrics.entropy_health +
            self.weights['alignment'] * metrics.alignment_score +
            self.weights['reflection'] * metrics.reflection_efficiency
        )
        
        return rollup_score
    
    def _check_thresholds(self, metrics: RollupMetrics) -> None:
        """
        Check if any metrics have crossed their thresholds.
        
        Args:
            metrics: Current metrics
        """
        if not self.on_threshold_crossed:
            return
            
        # Check each threshold
        for metric_name, threshold in self.thresholds.items():
            if not hasattr(metrics, metric_name):
                continue
                
            current_value = getattr(metrics, metric_name)
            
            # Only check if we have history
            if len(self.metrics_history) <= 1:
                continue
                
            # Get previous value
            prev_metrics = self.metrics_history[-2]
            if not hasattr(prev_metrics, metric_name):
                continue
                
            prev_value = getattr(prev_metrics, metric_name)
            
            # Check for crossing threshold in either direction
            if (prev_value < threshold and current_value >= threshold) or \
               (prev_value >= threshold and current_value < threshold):
                self.on_threshold_crossed(metric_name, current_value, threshold)
    
    def get_latest_metrics(self) -> Optional[RollupMetrics]:
        """
        Get the most recently calculated metrics.
        
        Returns:
            Latest metrics or None if no metrics have been calculated
        """
        if not self.metrics_history:
            return None
            
        return self.metrics_history[-1]
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[RollupMetrics]:
        """
        Get historical metrics.
        
        Args:
            limit: Maximum number of metrics to return (newest first)
            
        Returns:
            List of historical metrics
        """
        if limit is None:
            return self.metrics_history.copy()
            
        return self.metrics_history[-limit:]
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """
        Set a threshold for a specific metric.
        
        Args:
            metric_name: Name of the metric
            threshold: Threshold value
        """
        self.thresholds[metric_name] = threshold
        
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the agent's health.
        
        Returns:
            Dictionary with health summary information
        """
        latest = self.get_latest_metrics()
        if not latest:
            return {"status": "unavailable", "reason": "No metrics calculated yet"}
            
        # Determine overall health status
        status = "healthy"
        reasons = []
        
        if latest.rollup_score < self.thresholds.get('rollup_score', 0.6):
            status = "unhealthy"
            reasons.append("low_rollup_score")
            
        if latest.psi_c_score < self.thresholds.get('psi_c_score', 0.7):
            if status == "healthy":
                status = "degraded"
            reasons.append("low_psi_c")
            
        if latest.entropy_health < self.thresholds.get('entropy_health', 0.6):
            if status == "healthy":
                status = "degraded"
            reasons.append("high_entropy")
            
        if latest.alignment_score < self.thresholds.get('alignment_score', 0.7):
            if status == "healthy":
                status = "degraded"
            reasons.append("low_alignment")
            
        # Include trend information
        trend = "stable"
        if latest.trend_direction > 0.3:
            trend = "improving"
        elif latest.trend_direction < -0.3:
            trend = "declining"
            
        # Build summary
        return {
            "status": status,
            "reasons": reasons,
            "trend": trend,
            "rollup_score": latest.rollup_score,
            "timestamp": latest.timestamp,
            "components": {
                "psi_c": {
                    "score": latest.psi_c_score,
                    "threshold": self.thresholds.get('psi_c_score', 0.7)
                },
                "entropy": {
                    "health": latest.entropy_health,
                    "threshold": self.thresholds.get('entropy_health', 0.6)
                },
                "alignment": {
                    "score": latest.alignment_score,
                    "threshold": self.thresholds.get('alignment_score', 0.7)
                },
                "reflection": {
                    "efficiency": latest.reflection_efficiency,
                    "threshold": self.thresholds.get('reflection_efficiency', 0.5)
                }
            }
        } 