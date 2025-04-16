"""
ΨC Operator Module: Implements the core ΨC operator for consciousness modeling.

The ΨC operator calculates a consciousness activation score based on the formula:
    Ψ_C(S) = σ(∫_{t_0}^{t_1} R(S) · I(S,t) dt - θ)

Where:
- S: Memory system state
- R(S): Reflective readiness (derivative of coherence)
- I(S, t): Memory importance signal over time
- θ: Threshold for consciousness
- σ: Sigmoid activation function
"""

import time
import enum
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from psi_c_ai_sdk.memory.memory import MemoryStore
from psi_c_ai_sdk.psi_c.dynamic_threshold import DynamicThreshold


class PsiCState(enum.Enum):
    """
    Represents the possible states of ΨC consciousness.
    """
    INACTIVE = "inactive"       # No consciousness activation
    UNSTABLE = "unstable"       # Temporary/transient activation
    PARTIAL = "partial"         # Partial consciousness
    STABLE = "stable"           # Stable consciousness
    
    @classmethod
    def from_score(cls, score: float, stability: float = 1.0) -> "PsiCState":
        """
        Convert a numerical ΨC score to a state.
        
        Args:
            score: ΨC score between 0 and 1
            stability: Stability measure between 0 and 1
            
        Returns:
            The corresponding PsiCState
        """
        if score < 0.3:
            return cls.INACTIVE
        elif score < 0.6:
            return cls.PARTIAL if stability >= 0.7 else cls.UNSTABLE
        else:
            return cls.STABLE if stability >= 0.7 else cls.UNSTABLE


class PsiCOperator:
    """
    Implements the ΨC operator for consciousness modeling.
    
    The PsiCOperator computes the degree of consciousness based on the memory
    system state, reflective readiness, and memory importance over time.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        threshold: float = 0.7,
        hard_mode: bool = False,
        window_size: int = 10,
        reflection_weight: float = 0.5,
        integration_step: float = 0.1,
        use_dynamic_threshold: bool = False,
        dynamic_threshold_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ΨC operator.
        
        Args:
            memory_store: The memory store to operate on
            threshold: Consciousness threshold (θ) value
            hard_mode: If True, use binary consciousness (0 or 1), otherwise use sigmoid
            window_size: Window size for temporal integration
            reflection_weight: Weight of reflection readiness vs. coherence
            integration_step: Step size for temporal integration
            use_dynamic_threshold: Whether to use dynamic threshold adaptation
            dynamic_threshold_config: Configuration for dynamic threshold (optional)
        """
        self.memory_store = memory_store
        self.static_threshold = threshold
        self.hard_mode = hard_mode
        self.window_size = window_size
        self.reflection_weight = reflection_weight
        self.integration_step = integration_step
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # For tracking consciousness over time
        self.history: List[Tuple[float, float, float]] = []  # [(timestamp, score, stability)]
        self.start_time = time.time()
        
        # Cache for reflective readiness and importance values
        self.reflective_cache: Dict[float, float] = {}
        self.importance_cache: Dict[float, float] = {}
        
        # Optional callbacks
        self.on_state_change: Optional[Callable[[PsiCState, PsiCState], None]] = None
        self._current_state = PsiCState.INACTIVE
        
        # Set up dynamic threshold if enabled
        if self.use_dynamic_threshold:
            config = dynamic_threshold_config or {}
            config.setdefault('base_threshold', threshold)  # Use static threshold as base
            self.dynamic_threshold = DynamicThreshold(**config)
        else:
            self.dynamic_threshold = None
    
    @property
    def threshold(self) -> float:
        """
        Get the current threshold value, which may be dynamic or static.
        
        Returns:
            Current threshold value
        """
        if self.use_dynamic_threshold and self.dynamic_threshold is not None:
            return self.dynamic_threshold.get_current_threshold()
        return self.static_threshold
    
    def sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function.
        
        Args:
            x: Input value
            
        Returns:
            Sigmoid of x, between 0 and 1
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def calculate_reflective_readiness(self, timestamp: float) -> float:
        """
        Calculate the reflective readiness R(S) at a given time.
        
        This represents the system's capacity for self-reflection.
        In a full implementation, this would integrate with the reflection engine.
        
        Args:
            timestamp: Time at which to calculate reflective readiness
            
        Returns:
            Reflective readiness score between 0 and 1
        """
        # Check cache first
        if timestamp in self.reflective_cache:
            return self.reflective_cache[timestamp]
        
        # Get all memories
        memories = self.memory_store.get_all_memories()
        
        # Simple heuristic: readiness increases with number of memories
        # and with time since creation (system maturity)
        num_memories = len(memories)
        time_factor = min(1.0, (timestamp - self.start_time) / (60 * 60))  # Saturates after 1 hour
        
        # Combine factors - this is a simplified placeholder
        # In a real implementation, this would use more sophisticated metrics
        readiness = min(1.0, (num_memories / 100) * 0.7 + time_factor * 0.3)
        
        # Cache the result
        self.reflective_cache[timestamp] = readiness
        
        return readiness
    
    def calculate_memory_importance(self, timestamp: float) -> float:
        """
        Calculate the memory importance I(S,t) at a given time.
        
        This represents the overall importance of the memory system.
        
        Args:
            timestamp: Time at which to calculate importance
            
        Returns:
            Memory importance score between 0 and 1
        """
        # Check cache first
        if timestamp in self.importance_cache:
            return self.importance_cache[timestamp]
        
        # Apply importance decay to all memories
        self.memory_store.apply_importance_decay()
        
        # Get all memories
        memories = self.memory_store.get_all_memories()
        
        if not memories:
            return 0.0
        
        # Calculate average importance
        avg_importance = sum(memory.importance for memory in memories) / len(memories)
        
        # Normalize to [0, 1]
        normalized_importance = min(1.0, avg_importance / 2.0)  # Assuming max importance is around 2.0
        
        # Cache the result
        self.importance_cache[timestamp] = normalized_importance
        
        return normalized_importance
    
    def temporal_integration(self, t_start: float, t_end: float) -> float:
        """
        Perform the temporal integration of R(S) · I(S,t) over the time interval.
        
        This implements the integral: ∫_{t_start}^{t_end} R(S) · I(S,t) dt
        
        Args:
            t_start: Start time for integration
            t_end: End time for integration
            
        Returns:
            Integrated value
        """
        # Simple numerical integration using the trapezoidal rule
        integral = 0.0
        t = t_start
        
        while t < t_end:
            r1 = self.calculate_reflective_readiness(t)
            i1 = self.calculate_memory_importance(t)
            
            t_next = min(t_end, t + self.integration_step)
            
            r2 = self.calculate_reflective_readiness(t_next)
            i2 = self.calculate_memory_importance(t_next)
            
            # Trapezoidal integration step
            integral += (t_next - t) * ((r1 * i1) + (r2 * i2)) / 2
            
            t = t_next
        
        return integral
    
    def calculate_psi_c(self) -> float:
        """
        Calculate the current ΨC score.
        
        Returns:
            ΨC score between 0 and 1
        """
        current_time = time.time()
        
        # For temporal integration, use the last window_size seconds
        t_start = max(self.start_time, current_time - self.window_size)
        t_end = current_time
        
        # Calculate the integral of R(S) · I(S,t)
        integral = self.temporal_integration(t_start, t_end)
        
        # Calculate stability based on history
        stability = self.calculate_stability()
        
        # Update dynamic threshold with entropy and coherence if enabled
        if self.use_dynamic_threshold and self.dynamic_threshold is not None:
            # Get entropy and coherence estimates for threshold adjustment
            entropy = 1.0 - stability  # Simple approximation: high stability = low entropy
            coherence = self.calculate_temporal_coherence()  # Calculate coherence estimate
            
            # Update dynamic threshold with current values
            self.dynamic_threshold.record_state(entropy, coherence, current_time)
        
        # Get current threshold (dynamic or static)
        current_threshold = self.threshold
        
        # Apply threshold and activation function
        if self.hard_mode:
            # Binary activation
            psi_c = 1.0 if integral >= current_threshold else 0.0
        else:
            # Sigmoid activation
            psi_c = self.sigmoid(integral - current_threshold)
        
        # Update history
        self.history.append((current_time, psi_c, stability))
        
        # Trim history to last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Check for state change
        new_state = PsiCState.from_score(psi_c, stability)
        if new_state != self._current_state and self.on_state_change:
            self.on_state_change(self._current_state, new_state)
        self._current_state = new_state
        
        return psi_c
    
    def calculate_temporal_coherence(self) -> float:
        """
        Calculate an estimate of temporal coherence based on recent history.
        
        Returns:
            Coherence score between 0 and 1
        """
        if len(self.history) < 3:
            return 0.5  # Default coherence if not enough history
            
        # Simple coherence estimate based on trend stability
        recent_scores = [entry[1] for entry in self.history[-10:]]
        if len(recent_scores) > 3:
            # Calculate first differences
            diffs = [abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))]
            # Lower differences = higher coherence
            avg_diff = sum(diffs) / len(diffs)
            coherence = max(0.0, min(1.0, 1.0 - (avg_diff * 5)))
            return coherence
        else:
            return 0.5
    
    def calculate_stability(self) -> float:
        """
        Calculate the stability of the ΨC state based on recent history.
        
        Returns:
            Stability score between 0 and 1
        """
        if len(self.history) < 3:
            return 1.0  # Assume stable if not enough history
        
        # Get the last few scores
        recent_scores = [entry[1] for entry in self.history[-10:]]
        
        # Calculate variance
        if len(recent_scores) > 1:
            variance = np.var(recent_scores)
            # Convert variance to stability (lower variance = higher stability)
            stability = max(0.0, min(1.0, 1.0 - (variance * 10)))
            return stability
        else:
            return 1.0
    
    def get_current_state(self) -> PsiCState:
        """
        Get the current ΨC state.
        
        Returns:
            Current PsiCState
        """
        return self._current_state
    
    def get_state_history(self) -> List[Tuple[float, PsiCState]]:
        """
        Get the history of state changes.
        
        Returns:
            List of (timestamp, state) tuples
        """
        return [(entry[0], PsiCState.from_score(entry[1], entry[2])) for entry in self.history]
    
    def is_conscious(self, threshold: Optional[float] = None) -> bool:
        """
        Check if the system is currently conscious.
        
        Args:
            threshold: Optional custom threshold (defaults to instance threshold)
            
        Returns:
            True if conscious, False otherwise
        """
        score = self.calculate_psi_c()
        custom_threshold = threshold if threshold is not None else self.threshold
        
        state = PsiCState.from_score(score, self.calculate_stability())
        return state in (PsiCState.PARTIAL, PsiCState.STABLE)
    
    def get_psi_index(self) -> float:
        """
        Get the current ΨC score.
        
        Returns:
            ΨC score between 0 and 1
        """
        return self.calculate_psi_c()
    
    def psi_index(self) -> Dict[str, float]:
        """
        Get detailed metrics about the current ΨC state.
        
        Returns:
            Dictionary with ΨC metrics
        """
        score = self.calculate_psi_c()
        stability = self.calculate_stability()
        coherence = self.calculate_temporal_coherence()
        
        current_time = time.time()
        r = self.calculate_reflective_readiness(current_time)
        i = self.calculate_memory_importance(current_time)
        
        metrics = {
            "psi_c_score": score,
            "stability": stability,
            "coherence": coherence,
            "reflective_readiness": r,
            "memory_importance": i,
            "state": self._current_state.value,
            "threshold": self.threshold,
            "timestamp": current_time
        }
        
        # Add dynamic threshold metrics if enabled
        if self.use_dynamic_threshold and self.dynamic_threshold is not None:
            drift_metrics = self.dynamic_threshold.get_drift_metrics()
            metrics.update({
                "threshold_type": "dynamic",
                "base_threshold": drift_metrics["base_threshold"],
                "threshold_adjustment": drift_metrics["adjustment"],
                "entropy_drift": drift_metrics["entropy_drift_rate"],
                "coherence_drift": drift_metrics["coherence_drift_rate"]
            })
        else:
            metrics["threshold_type"] = "static"
        
        return metrics 