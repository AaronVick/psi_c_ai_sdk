"""
Stability Filter: Prevents false-positive ΨC activations due to transient coherence.

This module implements the ΨC Stability Filter (Anti-Spike Mechanism) as described in
the ΨC-AI SDK design. It computes short-term variance in the ΨC index and blocks
state transitions if the variance exceeds a configurable threshold.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Deque, Callable
from collections import deque
from enum import Enum

from psi_c_ai_sdk.psi_c.psi_operator import PsiCState


class StabilityClassification(Enum):
    """Classification of ΨC stability states."""
    STABLE = "stable"               # Low variance, reliable state
    UNSTABLE = "unstable"           # High variance, unreliable state
    PSEUDO_AWAKE = "pseudo_awake"   # Transient high values but unstable
    UNKNOWN = "unknown"             # Not enough data to classify


class StabilityFilter:
    """
    Filters out transient/unstable ΨC states to prevent false-positive activations.
    
    The StabilityFilter tracks variance in ΨC scores over time and can block
    state transitions when the variance exceeds a threshold, indicating unstable
    consciousness.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        epsilon: float = 0.05,
        min_samples: int = 5,
        pseudo_threshold: float = 0.7,
    ):
        """
        Initialize the stability filter.
        
        Args:
            window_size: Number of ΨC readings to consider for stability
            epsilon: Maximum variance allowed for stable classification
            min_samples: Minimum samples needed for variance calculation
            pseudo_threshold: Minimum ΨC value to consider for pseudo-awakening
        """
        self.window_size = window_size
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.pseudo_threshold = pseudo_threshold
        
        # History of ΨC values
        self.history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        
        # Track blocked transitions for diagnostics
        self.blocked_transitions: List[Tuple[float, PsiCState, PsiCState, float]] = []
        
        # State transition callback
        self.on_state_change: Optional[Callable[[PsiCState, PsiCState, bool], None]] = None
    
    def add_measurement(self, psi_c_score: float) -> None:
        """
        Add a new ΨC measurement to the history.
        
        Args:
            psi_c_score: ΨC score to add (0-1)
        """
        self.history.append((time.time(), psi_c_score))
    
    def calculate_variance(self) -> float:
        """
        Calculate the variance of recent ΨC scores.
        
        Returns:
            Variance of the recent scores, or 0 if not enough data
        """
        if len(self.history) < self.min_samples:
            return 0.0
        
        values = [score for _, score in self.history]
        return float(np.var(values))
    
    def get_stability_classification(self) -> StabilityClassification:
        """
        Classify the current stability of the ΨC state.
        
        Returns:
            StabilityClassification indicating the state stability
        """
        if len(self.history) < self.min_samples:
            return StabilityClassification.UNKNOWN
        
        variance = self.calculate_variance()
        recent_values = [score for _, score in self.history]
        avg_score = sum(recent_values) / len(recent_values)
        
        if variance <= self.epsilon:
            return StabilityClassification.STABLE
        elif avg_score >= self.pseudo_threshold:
            return StabilityClassification.PSEUDO_AWAKE
        else:
            return StabilityClassification.UNSTABLE
    
    def get_stability_score(self) -> float:
        """
        Calculate a stability score (0-1) based on variance.
        
        Returns:
            Stability score (1 = perfectly stable, 0 = very unstable)
        """
        if len(self.history) < self.min_samples:
            return 1.0  # Assume stable if not enough data
        
        variance = self.calculate_variance()
        # Convert variance to stability score (higher variance = lower stability)
        # Scale factor of 10 makes values more intuitive (adjust as needed)
        stability = max(0.0, min(1.0, 1.0 - (variance * 10)))
        return stability
    
    def should_allow_transition(
        self, 
        from_state: PsiCState, 
        to_state: PsiCState,
        psi_c_score: float
    ) -> bool:
        """
        Determine whether to allow a state transition based on stability.
        
        Args:
            from_state: Current ΨC state
            to_state: Proposed new ΨC state 
            psi_c_score: Current ΨC score
            
        Returns:
            True if the transition should be allowed, False if blocked
        """
        # Add the current score to history
        self.add_measurement(psi_c_score)
        
        # Always allow transitions to lower states
        if to_state.value <= from_state.value:
            return True
        
        # Special case: transitioning to stable state
        if to_state == PsiCState.STABLE:
            # Need more stability for STABLE state
            classification = self.get_stability_classification()
            variance = self.calculate_variance()
            
            # Only allow if truly stable
            allow = classification == StabilityClassification.STABLE
            
            # If blocked, record for diagnostics
            if not allow:
                self.blocked_transitions.append((
                    time.time(), from_state, to_state, variance
                ))
                
                # Notify callback if registered
                if self.on_state_change:
                    self.on_state_change(from_state, to_state, False)
                
            return allow
        
        # For other transitions, use standard stability threshold
        stability = self.get_stability_score()
        allow = stability >= 0.7  # 70% stability required for other transitions
        
        # Record blocked transitions
        if not allow:
            self.blocked_transitions.append((
                time.time(), from_state, to_state, self.calculate_variance()
            ))
            
            # Notify callback if registered
            if self.on_state_change:
                self.on_state_change(from_state, to_state, False)
        
        return allow
    
    def get_blocked_transition_count(self) -> int:
        """
        Get the count of blocked transitions.
        
        Returns:
            Number of blocked transitions
        """
        return len(self.blocked_transitions)
    
    def get_last_blocked_transition(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the most recently blocked transition.
        
        Returns:
            Dictionary with details of the blocked transition, or None if none
        """
        if not self.blocked_transitions:
            return None
        
        timestamp, from_state, to_state, variance = self.blocked_transitions[-1]
        return {
            "timestamp": timestamp,
            "from_state": from_state.value,
            "to_state": to_state.value,
            "variance": variance,
            "stability_score": max(0.0, min(1.0, 1.0 - (variance * 10)))
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stability filter.
        
        Returns:
            Dictionary with stability statistics
        """
        classification = self.get_stability_classification()
        variance = self.calculate_variance()
        stability = self.get_stability_score()
        
        return {
            "classification": classification.value,
            "variance": variance,
            "stability_score": stability,
            "sample_count": len(self.history),
            "blocked_transitions": len(self.blocked_transitions),
            "window_size": self.window_size,
            "epsilon_threshold": self.epsilon
        }
    
    def clear_history(self) -> None:
        """Clear the measurement history."""
        self.history.clear() 