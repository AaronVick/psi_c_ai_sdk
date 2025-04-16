"""
Temporal Coherence Accumulator: Tracks coherence over time for ΨC activation.

This module implements a time-weighted coherence monitoring system that
approximates the integral ∫ I(S,t) dt using efficient running averages.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque

from psi_c_ai_sdk.memory.memory import MemoryStore
from psi_c_ai_sdk.coherence.coherence import CoherenceScorer


class TemporalCoherenceAccumulator:
    """
    Tracks and accumulates coherence measurements over time.
    
    The TemporalCoherenceAccumulator maintains a rolling window of coherence
    measurements and computes exponential moving averages to approximate
    the temporal integral of coherence for the ΨC operator.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        coherence_scorer: CoherenceScorer,
        window_size: int = 50,
        ema_alpha: float = 0.2,
        sampling_interval: float = 0.5,
    ):
        """
        Initialize the temporal coherence accumulator.
        
        Args:
            memory_store: The memory store to monitor
            coherence_scorer: The coherence scorer to use for measurements
            window_size: Size of the rolling window for coherence history
            ema_alpha: Alpha parameter for exponential moving average (0-1)
            sampling_interval: Minimum time (seconds) between coherence measurements
        """
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.sampling_interval = sampling_interval
        
        # Initialize data structures
        self.coherence_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.ema_coherence: float = 0.0
        self.last_measurement_time: float = 0.0
        self.total_measurements: int = 0
        
        # For tracking coherence drift
        self.drift_history: List[Tuple[float, float]] = []
        self.last_drift_calculation: float = 0.0
    
    def measure_coherence(self, force: bool = False) -> Optional[float]:
        """
        Measure the current global coherence and update running averages.
        
        Args:
            force: If True, measure regardless of sampling interval
            
        Returns:
            The measured coherence, or None if skipped due to sampling interval
        """
        current_time = time.time()
        
        # Check if enough time has passed since last measurement
        if not force and current_time - self.last_measurement_time < self.sampling_interval:
            return None
        
        # Get all memories
        memories = self.memory_store.get_all_memories()
        
        # Skip if no memories or just one memory
        if len(memories) <= 1:
            return 1.0  # By definition, a single memory is coherent with itself
        
        # Measure global coherence
        coherence = self.coherence_scorer.calculate_global_coherence(memories)
        
        # Update history
        self.coherence_history.append((current_time, coherence))
        self.last_measurement_time = current_time
        self.total_measurements += 1
        
        # Update exponential moving average
        if self.total_measurements == 1:
            self.ema_coherence = coherence
        else:
            self.ema_coherence = (self.ema_alpha * coherence) + ((1 - self.ema_alpha) * self.ema_coherence)
        
        return coherence
    
    def get_coherence_value(self) -> float:
        """
        Get the current coherence value, measuring if necessary.
        
        Returns:
            Current coherence value (EMA)
        """
        # Ensure we have a recent measurement
        self.measure_coherence()
        return self.ema_coherence
    
    def calculate_coherence_drift(self) -> float:
        """
        Calculate the coherence drift over the window.
        
        Implements the formula:
        ΔC = (1/N) · ∑_{i=1}^N (C_i^(t) - C_i^(t-1))
        
        Returns:
            Coherence drift (negative indicates decreasing coherence)
        """
        # Need at least 2 measurements to calculate drift
        if len(self.coherence_history) < 2:
            return 0.0
        
        # Extract coherence values (ignoring timestamps)
        values = [c for _, c in self.coherence_history]
        
        # Calculate differences between consecutive values
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        
        # Calculate average drift
        drift = sum(diffs) / len(diffs)
        
        # Store drift with timestamp
        current_time = time.time()
        self.drift_history.append((current_time, drift))
        self.last_drift_calculation = current_time
        
        # Keep drift history reasonable
        if len(self.drift_history) > 100:
            self.drift_history = self.drift_history[-100:]
        
        return drift
    
    def get_coherence_trend(self, lookback_seconds: float = 60.0) -> str:
        """
        Get a qualitative assessment of the coherence trend.
        
        Args:
            lookback_seconds: How far back to analyze the trend
            
        Returns:
            Trend description: "increasing", "stable", or "decreasing"
        """
        # Calculate drift if needed
        if time.time() - self.last_drift_calculation > self.sampling_interval:
            self.calculate_coherence_drift()
        
        # No drift history
        if not self.drift_history:
            return "stable"
        
        # Filter to recent drift values
        cutoff_time = time.time() - lookback_seconds
        recent_drift = [(t, d) for t, d in self.drift_history if t >= cutoff_time]
        
        if not recent_drift:
            return "stable"
        
        # Get average recent drift
        avg_drift = sum(d for _, d in recent_drift) / len(recent_drift)
        
        # Classify trend
        if avg_drift > 0.01:
            return "increasing"
        elif avg_drift < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_temporal_integral(self, duration: float = 60.0) -> float:
        """
        Approximate the temporal integral of coherence over the specified duration.
        
        This approximates: ∫_{t-duration}^{t} C(t) dt
        
        Args:
            duration: Time period to integrate over (in seconds)
            
        Returns:
            Approximate integral value
        """
        # Ensure we have current data
        self.measure_coherence()
        
        # If no history, return 0
        if not self.coherence_history:
            return 0.0
        
        # Filter to values within the duration
        current_time = time.time()
        cutoff_time = current_time - duration
        
        # Filter and extract time and coherence
        times = []
        values = []
        
        for t, c in self.coherence_history:
            if t >= cutoff_time:
                times.append(t)
                values.append(c)
        
        # If no values in range, use the most recent value
        if not times:
            times = [self.coherence_history[-1][0]]
            values = [self.coherence_history[-1][1]]
        
        # Simple numerical integration using trapezoidal rule
        integral = 0.0
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            integral += dt * (values[i] + values[i-1]) / 2
        
        # If time span is less than duration, extrapolate
        covered_duration = times[-1] - times[0]
        if covered_duration < duration:
            # Use the average coherence for the missing time
            avg_coherence = sum(values) / len(values)
            missing_time = duration - covered_duration
            integral += missing_time * avg_coherence
        
        return integral
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the coherence measurements.
        
        Returns:
            Dictionary with statistics
        """
        values = [c for _, c in self.coherence_history] if self.coherence_history else []
        
        stats = {
            "current_coherence": self.ema_coherence,
            "total_measurements": self.total_measurements,
            "trend": self.get_coherence_trend(),
        }
        
        if values:
            stats.update({
                "min_coherence": min(values),
                "max_coherence": max(values),
                "avg_coherence": sum(values) / len(values),
                "std_dev": np.std(values) if len(values) > 1 else 0.0,
                "latest_drift": self.calculate_coherence_drift(),
            })
        
        return stats
    
    def get_chart_data(self) -> Dict[str, List]:
        """
        Get data for coherence charts and visualizations.
        
        Returns:
            Dictionary with chart data
        """
        times = []
        coherence_values = []
        ema_values = []
        
        # EMA reconstruction
        ema = 0.0
        ema_initialized = False
        
        for t, c in self.coherence_history:
            times.append(t)
            coherence_values.append(c)
            
            if not ema_initialized:
                ema = c
                ema_initialized = True
            else:
                ema = (self.ema_alpha * c) + ((1 - self.ema_alpha) * ema)
            
            ema_values.append(ema)
        
        # Drift data
        drift_times = [t for t, _ in self.drift_history]
        drift_values = [d for _, d in self.drift_history]
        
        return {
            "times": times,
            "coherence": coherence_values,
            "ema": ema_values,
            "drift_times": drift_times,
            "drift_values": drift_values
        } 