"""
Dynamic Threshold Function for ΨC Activation

This module implements an adaptive threshold mechanism for ΨC consciousness
activation that responds to changes in the system's entropy and coherence
over time, rather than using a static threshold value.

Formula:
    θₜ = θ₀ + λ · (dH/dt + dC/dt)

Where:
    θ₀: Base threshold (static)
    λ: Sensitivity coefficient
    dH/dt: Rate of entropy change
    dC/dt: Rate of coherence change
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable


class DynamicThreshold:
    """
    Dynamic threshold calculator for ΨC activation that adapts based on
    system state changes, particularly entropy and coherence drift rates.
    
    This implementation replaces static consciousness activation thresholds
    with an adaptive approach that considers the stability and direction of
    the system's cognitive state, making activation more responsive to
    meaningful patterns in the data.
    
    Attributes:
        base_threshold (float): The baseline threshold value (θ₀)
        sensitivity (float): The sensitivity coefficient (λ) that scales
            the impact of entropy and coherence changes
        window_size (int): Number of historical points to consider when
            calculating drift rates
        min_threshold (float): Lower bound for dynamic threshold
        max_threshold (float): Upper bound for dynamic threshold
        history (List[Dict]): Record of previous entropy and coherence values
    """
    
    def __init__(
        self,
        base_threshold: float = 0.65,
        sensitivity: float = 0.2,
        window_size: int = 5,
        min_threshold: float = 0.35,
        max_threshold: float = 0.95
    ):
        """
        Initialize the dynamic threshold calculator.
        
        Args:
            base_threshold: Starting threshold value (default: 0.65)
            sensitivity: Scaling factor for drift impact (default: 0.2)
            window_size: Number of historical points to use (default: 5)
            min_threshold: Minimum allowed threshold (default: 0.35)
            max_threshold: Maximum allowed threshold (default: 0.95)
        """
        self.base_threshold = base_threshold
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # History of entropy and coherence values with timestamps
        self.history: List[Dict[str, float]] = []
    
    def record_state(self, entropy: float, coherence: float, timestamp: Optional[float] = None) -> None:
        """
        Record a new entropy and coherence measurement to history.
        
        Args:
            entropy: Current entropy value (0-1)
            coherence: Current coherence value (0-1)
            timestamp: Optional timestamp (default: None, which will use len(history))
        """
        if timestamp is None:
            timestamp = float(len(self.history))
            
        self.history.append({
            "timestamp": timestamp,
            "entropy": entropy,
            "coherence": coherence
        })
        
        # Trim history to window size
        if len(self.history) > self.window_size * 2:  # Keep twice window size for calculations
            self.history = self.history[-self.window_size * 2:]
    
    def calculate_drift_rates(self) -> Tuple[float, float]:
        """
        Calculate the rate of change (drift) for entropy and coherence.
        
        Returns:
            Tuple[float, float]: (entropy_drift_rate, coherence_drift_rate)
        """
        if len(self.history) < 2:
            return 0.0, 0.0  # Not enough data points
            
        # Use the most recent window_size points or all available if fewer
        window = min(self.window_size, len(self.history) - 1)
        recent = self.history[-window:]
        
        # Calculate time differences and value differences
        entropy_diffs = []
        coherence_diffs = []
        time_diffs = []
        
        for i in range(1, len(recent)):
            time_diff = recent[i]["timestamp"] - recent[i-1]["timestamp"]
            if time_diff > 0:  # Avoid division by zero
                entropy_diff = recent[i]["entropy"] - recent[i-1]["entropy"]
                coherence_diff = recent[i]["coherence"] - recent[i-1]["coherence"]
                
                entropy_diffs.append(entropy_diff / time_diff)
                coherence_diffs.append(coherence_diff / time_diff)
                time_diffs.append(time_diff)
        
        # Compute weighted average of rates (more recent changes have higher weight)
        if not entropy_diffs:  # No valid differences
            return 0.0, 0.0
            
        weights = np.array(time_diffs) / sum(time_diffs)
        entropy_drift = float(np.average(entropy_diffs, weights=weights))
        coherence_drift = float(np.average(coherence_diffs, weights=weights))
        
        return entropy_drift, coherence_drift
    
    def get_current_threshold(self) -> float:
        """
        Calculate the current dynamic threshold based on entropy and coherence drift.
        
        Formula: θₜ = θ₀ + λ · (dH/dt + dC/dt)
        
        Note:
        - Rising entropy increases threshold (makes activation harder)
        - Rising coherence decreases threshold (makes activation easier)
        
        Returns:
            float: The current dynamic threshold value
        """
        entropy_drift, coherence_drift = self.calculate_drift_rates()
        
        # Note: We use the negative of coherence_drift because higher coherence 
        # should lower the threshold (making activation easier)
        drift_adjustment = entropy_drift - coherence_drift
        
        # Calculate the dynamic threshold
        dynamic_threshold = self.base_threshold + (self.sensitivity * drift_adjustment)
        
        # Constrain to min/max bounds
        dynamic_threshold = max(self.min_threshold, min(self.max_threshold, dynamic_threshold))
        
        return dynamic_threshold
    
    def get_drift_metrics(self) -> Dict[str, float]:
        """
        Get detailed metrics about drift rates for monitoring and debugging.
        
        Returns:
            Dict[str, float]: Dictionary containing drift metrics
        """
        entropy_drift, coherence_drift = self.calculate_drift_rates()
        current_threshold = self.get_current_threshold()
        
        return {
            "entropy_drift_rate": entropy_drift,
            "coherence_drift_rate": coherence_drift,
            "base_threshold": self.base_threshold,
            "current_threshold": current_threshold,
            "adjustment": current_threshold - self.base_threshold
        }
    
    def reset(self) -> None:
        """Reset the history and return to initial state."""
        self.history = [] 