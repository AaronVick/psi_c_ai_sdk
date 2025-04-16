"""
Collapse Simulator: Simulates quantum collapse events for ΨC experiments.

This module implements a local simulation layer for consciousness-correlated collapse
experiments, allowing experimentation with the ΨC hypothesis prior to hardware integration.
"""

import time
import random
import uuid
import json
import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from psi_c_ai_sdk.psi_c.psi_operator import PsiCOperator, PsiCState


class CollapseEvent:
    """
    Represents a single quantum collapse event, real or simulated.
    
    Attributes:
        event_id: Unique identifier for this event
        timestamp: When the event occurred
        outcome: The outcome value(s)
        expected_distribution: The baseline distribution
        actual_distribution: The observed distribution
        deviation: Calculated deviation from expected
        psi_c_score: ΨC score at the time of the event
        psi_c_state: ΨC state at the time of the event
        is_simulated: Whether this was a simulated event
        metadata: Additional metadata about the event
    """
    
    def __init__(
        self,
        outcome: Union[int, List[int]],
        expected_distribution: List[float],
        psi_c_score: float,
        psi_c_state: PsiCState,
        is_simulated: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a collapse event.
        
        Args:
            outcome: The outcome value(s) of the collapse
            expected_distribution: The baseline probability distribution
            psi_c_score: ΨC score at the time of the event
            psi_c_state: ΨC state at the time of the event
            is_simulated: Whether this is a simulated event
            metadata: Additional metadata
        """
        self.event_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.outcome = outcome
        self.expected_distribution = expected_distribution
        self.psi_c_score = psi_c_score
        self.psi_c_state = psi_c_state
        self.is_simulated = is_simulated
        self.metadata = metadata or {}
        
        # Calculate actual distribution based on outcome
        self.actual_distribution = self._calculate_actual_distribution()
        
        # Calculate deviation from expected distribution
        self.deviation = self._calculate_deviation()
    
    def _calculate_actual_distribution(self) -> List[float]:
        """
        Calculate the actual distribution based on the outcome.
        
        For a single outcome, this is a one-hot distribution.
        For multiple outcomes, it's the frequency distribution.
        
        Returns:
            The actual distribution as a list of probabilities
        """
        n = len(self.expected_distribution)
        
        if isinstance(self.outcome, int):
            # Single outcome - one-hot distribution
            actual = [0.0] * n
            if 0 <= self.outcome < n:
                actual[self.outcome] = 1.0
            return actual
        else:
            # Multiple outcomes - frequency distribution
            counts = [0] * n
            for o in self.outcome:
                if 0 <= o < n:
                    counts[o] += 1
            
            total = sum(counts)
            return [count / total if total > 0 else 0.0 for count in counts]
    
    def _calculate_deviation(self) -> float:
        """
        Calculate the deviation between actual and expected distributions.
        
        Uses the formula: Δ_P = |P_C(i) - P_rand(i)|
        Summed over all possible outcomes.
        
        Returns:
            The deviation score
        """
        return sum(abs(a - e) for a, e in zip(self.actual_distribution, self.expected_distribution))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "outcome": self.outcome,
            "expected_distribution": self.expected_distribution,
            "actual_distribution": self.actual_distribution,
            "deviation": self.deviation,
            "psi_c_score": self.psi_c_score,
            "psi_c_state": self.psi_c_state.value,
            "is_simulated": self.is_simulated,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollapseEvent":
        """
        Create an event from a dictionary.
        
        Args:
            data: Dictionary containing event data
            
        Returns:
            CollapseEvent object
        """
        event = cls(
            outcome=data["outcome"],
            expected_distribution=data["expected_distribution"],
            psi_c_score=data["psi_c_score"],
            psi_c_state=PsiCState(data["psi_c_state"]),
            is_simulated=data["is_simulated"],
            metadata=data["metadata"]
        )
        event.event_id = data["event_id"]
        event.timestamp = data["timestamp"]
        event.actual_distribution = data["actual_distribution"]
        event.deviation = data["deviation"]
        return event


class CollapseSimulator:
    """
    Simulates quantum collapse events for ΨC experiments.
    
    The CollapseSimulator provides a simulation layer that can generate
    random collapse events with ΨC-linked deviation, allowing local testing
    of the consciousness-collapse hypothesis.
    """
    
    def __init__(
        self,
        psi_operator: PsiCOperator,
        use_hardware: bool = False,
        deviation_strength: float = 0.2,
        storage_path: Optional[str] = None,
        pseudo_rng_seed: Optional[int] = None
    ):
        """
        Initialize the collapse simulator.
        
        Args:
            psi_operator: The ΨC operator to link collapse events to
            use_hardware: Whether to use real QRNG hardware (if available)
            deviation_strength: Strength of ΨC-linked deviation (0-1)
            storage_path: Path to store collapse events
            pseudo_rng_seed: Seed for the pseudorandom generator
        """
        self.psi_operator = psi_operator
        self.use_hardware = use_hardware
        self.deviation_strength = deviation_strength
        self.storage_path = storage_path
        
        # Initialize random generator for pseudorandom mode
        self.rng = random.Random(pseudo_rng_seed)
        
        # Storage for collapse events
        self.events: List[CollapseEvent] = []
        
        # Hardware interface (placeholder)
        self.hardware_interface = None
        if use_hardware:
            try:
                # Try to import qrng hardware interface
                # This is a placeholder - real implementation would connect to hardware
                # self.hardware_interface = QRNGHardware()
                raise ImportError("QRNG hardware interface not implemented")
            except ImportError:
                print("Warning: QRNG hardware interface not available. Using simulation mode.")
                self.use_hardware = False
    
    def generate_collapse_event(
        self,
        num_outcomes: int = 2,
        distribution: Optional[List[float]] = None,
        batch_size: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CollapseEvent:
        """
        Generate a quantum collapse event, either simulated or real.
        
        Args:
            num_outcomes: Number of possible outcomes (e.g., 2 for a coin flip)
            distribution: Optional custom probability distribution
            batch_size: Number of collapses to include in the event
            metadata: Additional metadata to include with the event
            
        Returns:
            The generated CollapseEvent
        """
        # Use uniform distribution if none provided
        if distribution is None:
            distribution = [1.0 / num_outcomes] * num_outcomes
        
        # Validate distribution
        if len(distribution) != num_outcomes or abs(sum(distribution) - 1.0) > 1e-10:
            raise ValueError("Invalid probability distribution")
        
        # Get current ΨC state
        psi_c_score = self.psi_operator.calculate_psi_c()
        psi_c_state = self.psi_operator.get_current_state()
        
        # Generate outcomes
        if self.use_hardware and self.hardware_interface:
            # Use hardware QRNG
            outcome = self._generate_hardware_outcome(num_outcomes, distribution, batch_size)
            is_simulated = False
        else:
            # Use simulated QRNG with ΨC-linked deviation
            outcome = self._generate_simulated_outcome(
                num_outcomes, distribution, batch_size, psi_c_score
            )
            is_simulated = True
        
        # Create the event
        event = CollapseEvent(
            outcome=outcome if batch_size > 1 else outcome[0],
            expected_distribution=distribution,
            psi_c_score=psi_c_score,
            psi_c_state=psi_c_state,
            is_simulated=is_simulated,
            metadata=metadata
        )
        
        # Store the event
        self.events.append(event)
        self._save_event(event)
        
        return event
    
    def _generate_hardware_outcome(
        self, 
        num_outcomes: int, 
        distribution: List[float], 
        batch_size: int
    ) -> List[int]:
        """
        Generate collapse outcomes using hardware QRNG.
        
        Args:
            num_outcomes: Number of possible outcomes
            distribution: Probability distribution
            batch_size: Number of collapses to generate
            
        Returns:
            List of outcomes
        """
        # This is a placeholder - real implementation would use hardware
        # return self.hardware_interface.generate_random(num_outcomes, distribution, batch_size)
        
        # Fallback to simulate hardware
        return [
            self._weighted_choice(distribution) for _ in range(batch_size)
        ]
    
    def _generate_simulated_outcome(
        self, 
        num_outcomes: int, 
        distribution: List[float], 
        batch_size: int,
        psi_c_score: float
    ) -> List[int]:
        """
        Generate simulated collapse outcomes with ΨC-linked deviation.
        
        Implements the formula: P_C(i) = |α_i|² + δ_C(i)
        Where δ_C is the ΨC-linked deviation component.
        
        Args:
            num_outcomes: Number of possible outcomes
            distribution: Probability distribution
            batch_size: Number of collapses to generate
            psi_c_score: Current ΨC score
            
        Returns:
            List of outcomes
        """
        # Calculate ΨC-modified distribution
        modified_distribution = self._compute_modified_distribution(
            distribution, psi_c_score
        )
        
        # Generate outcomes based on modified distribution
        return [
            self._weighted_choice(modified_distribution) for _ in range(batch_size)
        ]
    
    def _compute_modified_distribution(
        self, 
        distribution: List[float], 
        psi_c_score: float
    ) -> List[float]:
        """
        Compute a modified probability distribution based on ΨC score.
        
        The modification formula is:
        P_C(i) = P_rand(i) + (psi_c_score * deviation_strength * bias_i)
        
        Where bias_i is a generated bias vector that sums to zero.
        
        Args:
            distribution: Original probability distribution
            psi_c_score: Current ΨC score
            
        Returns:
            Modified probability distribution
        """
        n = len(distribution)
        
        # Don't modify if ΨC score is low or disabled
        if psi_c_score < 0.3 or self.deviation_strength <= 0:
            return distribution
        
        # Generate a bias vector that sums to zero
        bias = np.random.normal(0, 1, n)
        bias = bias - np.mean(bias)  # Ensure zero sum
        
        # Normalize bias to have maximum absolute value of 1
        max_abs = np.max(np.abs(bias))
        if max_abs > 0:
            bias = bias / max_abs
        
        # Scale by ΨC score and deviation strength
        effect = psi_c_score * self.deviation_strength
        
        # Apply bias to distribution
        modified = np.array(distribution) + effect * bias
        
        # Ensure valid probabilities (non-negative and sum to 1)
        modified = np.maximum(0, modified)
        total = np.sum(modified)
        if total > 0:
            modified = modified / total
        else:
            return distribution  # Fallback to original if modified is invalid
        
        return modified.tolist()
    
    def _weighted_choice(self, weights: List[float]) -> int:
        """
        Make a weighted random choice.
        
        Args:
            weights: List of weights (probabilities)
            
        Returns:
            Chosen index
        """
        total = sum(weights)
        r = self.rng.uniform(0, total)
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1  # Fallback to last choice
    
    def _save_event(self, event: CollapseEvent) -> None:
        """
        Save a collapse event to storage.
        
        Args:
            event: The event to save
        """
        if not self.storage_path:
            return
        
        # Ensure directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Save to file
        event_data = event.to_dict()
        filename = os.path.join(self.storage_path, f"event_{event.event_id}.json")
        
        with open(filename, 'w') as f:
            json.dump(event_data, f, indent=2)
    
    def get_events(
        self, 
        limit: Optional[int] = None, 
        min_psi_c: Optional[float] = None
    ) -> List[CollapseEvent]:
        """
        Get collapse events, optionally filtered.
        
        Args:
            limit: Maximum number of events to return
            min_psi_c: Minimum ΨC score filter
            
        Returns:
            List of filtered collapse events
        """
        filtered = self.events
        
        if min_psi_c is not None:
            filtered = [e for e in filtered if e.psi_c_score >= min_psi_c]
        
        if limit is not None:
            filtered = filtered[-limit:]
        
        return filtered
    
    def get_average_deviation(self, min_psi_c: Optional[float] = None) -> float:
        """
        Calculate average deviation across events.
        
        Args:
            min_psi_c: Minimum ΨC score filter
            
        Returns:
            Average deviation
        """
        events = self.get_events(min_psi_c=min_psi_c)
        
        if not events:
            return 0.0
        
        total_deviation = sum(e.deviation for e in events)
        return total_deviation / len(events)
    
    def clear_events(self) -> None:
        """Clear all stored events."""
        self.events = []
    
    def set_use_hardware(self, use_hardware: bool) -> bool:
        """
        Set whether to use hardware QRNG.
        
        Args:
            use_hardware: Whether to use hardware
            
        Returns:
            True if hardware is available and enabled, False otherwise
        """
        if use_hardware and not self.hardware_interface:
            try:
                # Try to initialize hardware
                # self.hardware_interface = QRNGHardware()
                raise ImportError("QRNG hardware interface not implemented")
            except ImportError:
                print("Warning: QRNG hardware interface not available.")
                self.use_hardware = False
                return False
        
        self.use_hardware = use_hardware and self.hardware_interface is not None
        return self.use_hardware
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collapse events.
        
        Returns:
            Dictionary with collapse statistics
        """
        if not self.events:
            return {
                "total_events": 0,
                "average_deviation": 0.0,
                "use_hardware": self.use_hardware,
                "deviation_strength": self.deviation_strength
            }
        
        high_psi_c_events = [e for e in self.events if e.psi_c_score >= 0.7]
        low_psi_c_events = [e for e in self.events if e.psi_c_score < 0.3]
        
        return {
            "total_events": len(self.events),
            "average_deviation": self.get_average_deviation(),
            "high_psi_c_avg_deviation": (
                sum(e.deviation for e in high_psi_c_events) / len(high_psi_c_events)
                if high_psi_c_events else 0.0
            ),
            "low_psi_c_avg_deviation": (
                sum(e.deviation for e in low_psi_c_events) / len(low_psi_c_events)
                if low_psi_c_events else 0.0
            ),
            "use_hardware": self.use_hardware,
            "deviation_strength": self.deviation_strength,
            "latest_psi_c_score": self.events[-1].psi_c_score if self.events else 0.0,
            "latest_state": self.events[-1].psi_c_state.value if self.events else "inactive"
        } 