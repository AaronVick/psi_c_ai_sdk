"""
Complexity Controller for ΨC-AI SDK.

This module implements a bounded cognitive runtime that monitors and controls
the complexity of ΨC operations, preventing feedback loops, runaway recursion,
and unbounded activation of high-tier features.

The controller uses a complexity budget function that takes into account:
- Number of active memories
- Schema complexity
- Recent contradictions and reflections

It regulates feature activation using sigmoid-gated thresholds, implements
adaptive parameter tuning to prevent oscillations, and uses temporal dampening
to prevent recursion-on-recursion spirals.
"""

import math
import time
import random
import logging
import collections
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, Deque, Set, Union

import numpy as np

from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.reflection import ReflectionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityTier(Enum):
    """Cognitive complexity tiers that gate feature activation."""
    TIER_0 = 0  # Basic memory + coherence
    TIER_1 = 1  # Reflection + value decay
    TIER_2 = 2  # Schema mutation
    TIER_3 = 3  # Self-awareness, identity, legacy


class FeatureActivation(Enum):
    """Features that can be activated or deactivated based on complexity budget."""
    MEMORY_STORAGE = auto()
    CONTRADICTION_DETECTION = auto()
    COHERENCE_TRACKING = auto()
    REFLECTION = auto()
    VALUE_DECAY = auto()
    SCHEMA_MUTATION = auto()
    IDENTITY_MODELING = auto()
    LEGACY_CREATION = auto()


class ComplexityController:
    """
    Controller for managing cognitive complexity and feature activation.
    
    The ComplexityController implements a bounded cognitive runtime that:
    1. Monitors system complexity based on memory, schema, and reflection metrics
    2. Controls feature activation using complexity tiers and sigmoid gating
    3. Prevents feedback loops and runaway complexity growth
    4. Adaptively tunes parameters to stabilize behavior
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        schema_graph: Optional[SchemaGraph] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        tier_thresholds: Dict[ComplexityTier, float] = None,
        alpha_weights: Tuple[float, float, float] = (1.0, 2.0, 3.0),
        interaction_coefficient: float = 0.2,
        min_interval: float = 30.0,
        history_window: int = 10,
    ):
        """
        Initialize the complexity controller.
        
        Args:
            memory_store: The memory store to monitor
            schema_graph: Optional schema graph to monitor
            reflection_engine: Optional reflection engine to monitor
            tier_thresholds: Complexity thresholds for each tier
            alpha_weights: Weights for memory, schema, and contradictions
            interaction_coefficient: Weight for cross-feature interactions
            min_interval: Minimum time between high-complexity operations
            history_window: Size of history window for stability tracking
        """
        self.memory_store = memory_store
        self.schema_graph = schema_graph
        self.reflection_engine = reflection_engine
        
        # Set default tier thresholds if not provided
        self.tier_thresholds = tier_thresholds or {
            ComplexityTier.TIER_0: 0.0,    # Always active
            ComplexityTier.TIER_1: 50.0,   # Reflection, decay
            ComplexityTier.TIER_2: 100.0,  # Schema mutation
            ComplexityTier.TIER_3: 200.0,  # Identity, legacy
        }
        
        # Weights for different complexity components
        self.alpha_memory, self.alpha_schema, self.alpha_contradiction = alpha_weights
        self.interaction_coefficient = interaction_coefficient
        
        # Temporal tracking
        self.min_interval = min_interval
        self.history_window = history_window
        
        # Sigmoid activation parameters
        self.activation_params: Dict[FeatureActivation, Dict[str, float]] = {
            feature: {"tau": 0.5, "k": 10.0} for feature in FeatureActivation
        }
        
        # History tracking
        self.complexity_history: Deque[float] = collections.deque(maxlen=history_window)
        self.tier_history: Deque[ComplexityTier] = collections.deque(maxlen=history_window)
        
        # Feature last activation times
        self.last_activation: Dict[FeatureActivation, float] = {
            feature: 0.0 for feature in FeatureActivation
        }
        
        # Energy budget tracking
        self.total_energy_budget = 100.0
        self.feature_energy_costs: Dict[FeatureActivation, float] = {
            FeatureActivation.MEMORY_STORAGE: 1.0,
            FeatureActivation.CONTRADICTION_DETECTION: 2.0,
            FeatureActivation.COHERENCE_TRACKING: 1.0,
            FeatureActivation.REFLECTION: 5.0,
            FeatureActivation.VALUE_DECAY: 2.0,
            FeatureActivation.SCHEMA_MUTATION: 10.0,
            FeatureActivation.IDENTITY_MODELING: 15.0,
            FeatureActivation.LEGACY_CREATION: 20.0,
        }
        
        # Recent feature activations
        self.recent_activations: Dict[FeatureActivation, int] = {
            feature: 0 for feature in FeatureActivation
        }
        
        # Initialize
        self._update_complexity()
    
    def calculate_complexity(self) -> float:
        """
        Calculate the current complexity score based on memory, schema, and contradiction state.
        
        Formula: C_t = α₁ · M_t + α₂ · S_t + α₃ · D_t
        
        Returns:
            Current complexity score
        """
        # Memory complexity
        memory_count = len(self.memory_store.memories)
        memory_complexity = self.alpha_memory * memory_count
        
        # Schema complexity
        schema_complexity = 0.0
        if self.schema_graph:
            schema_size = len(self.schema_graph.graph.nodes)
            schema_edges = len(self.schema_graph.graph.edges)
            schema_depth = 1.0
            
            # Estimate schema depth by calculating average path length
            if schema_size > 1 and schema_edges > 0:
                # This is an approximation - a full graph traversal would be more accurate
                # but more computationally expensive
                schema_depth = max(1.0, schema_edges / schema_size)
            
            schema_complexity = self.alpha_schema * schema_size * schema_depth
        
        # Contradiction and reflection complexity
        contradiction_count = 0
        reflection_count = 0
        
        if self.reflection_engine:
            contradiction_count = len(self.reflection_engine.get_recent_contradictions())
            reflection_count = self.reflection_engine.get_reflection_count()
        
        contradiction_complexity = self.alpha_contradiction * (contradiction_count + reflection_count)
        
        # Base complexity
        base_complexity = memory_complexity + schema_complexity + contradiction_complexity
        
        # Apply interaction effects if we have enough data
        if self.schema_graph and schema_size > 0:
            # Cross-feature interaction model: C'_t = C_t · (1 + λ · (M_t · D_t)/(S_t + ε))
            interaction_term = (
                self.interaction_coefficient * 
                (memory_count * (contradiction_count + reflection_count)) /
                (schema_size + 1e-5)  # Epsilon to avoid division by zero
            )
            adjusted_complexity = base_complexity * (1.0 + interaction_term)
            return adjusted_complexity
        
        return base_complexity
    
    def _update_complexity(self) -> Tuple[float, ComplexityTier]:
        """
        Update internal complexity state and determine current tier.
        
        Returns:
            Tuple of (complexity_score, current_tier)
        """
        # Calculate current complexity
        complexity = self.calculate_complexity()
        
        # Determine current tier
        current_tier = ComplexityTier.TIER_0
        for tier, threshold in sorted(self.tier_thresholds.items(), key=lambda x: x[1]):
            if complexity >= threshold:
                current_tier = tier
        
        # Update history
        self.complexity_history.append(complexity)
        self.tier_history.append(current_tier)
        
        return complexity, current_tier
    
    def get_current_tier(self) -> ComplexityTier:
        """
        Get the current complexity tier.
        
        Returns:
            Current complexity tier
        """
        complexity, tier = self._update_complexity()
        return tier
    
    def is_tier_stable(self) -> bool:
        """
        Check if the complexity tier has been stable in recent history.
        
        Returns:
            True if tier has been stable, False otherwise
        """
        if len(self.tier_history) < 3:
            return True
        
        # Consider stable if all recent tiers are the same
        return len(set(list(self.tier_history)[-3:])) == 1
    
    def is_feedback_loop_detected(self) -> bool:
        """
        Detect potential feedback loops in complexity growth.
        
        Returns:
            True if a feedback loop is detected, False otherwise
        """
        if len(self.tier_history) < 3:
            return False
        
        # Check for rapid tier escalation pattern
        recent_tiers = list(self.tier_history)[-3:]
        tier_values = [tier.value for tier in recent_tiers]
        
        # Check for strictly increasing tier values
        return tier_values[0] < tier_values[1] < tier_values[2]
    
    def adjust_activation_parameters(self, feature: FeatureActivation):
        """
        Adaptively tune activation parameters based on stability.
        
        Args:
            feature: The feature to adjust parameters for
        """
        params = self.activation_params[feature]
        
        if not self.is_tier_stable():
            # In unstable regimes, make activation more conservative
            params["tau"] = min(0.8, params["tau"] + 0.01)
            params["k"] = max(5.0, params["k"] - 0.5)
        else:
            # In stable regimes, gradually relax parameters
            params["tau"] = max(0.2, params["tau"] - 0.005)
            params["k"] = min(15.0, params["k"] + 0.25)
    
    def sigmoid_activation(
        self, 
        metric: float, 
        feature: FeatureActivation
    ) -> float:
        """
        Calculate sigmoid-gated activation probability.
        
        Formula: A(x) = 1/(1 + e^(-k(x - τ)))
        
        Args:
            metric: Input metric value (e.g., coherence, entropy)
            feature: Feature to calculate activation for
            
        Returns:
            Activation probability [0, 1]
        """
        params = self.activation_params[feature]
        k = params["k"]
        tau = params["tau"]
        
        return 1.0 / (1.0 + math.exp(-k * (metric - tau)))
    
    def available_energy(self) -> float:
        """
        Calculate available energy based on the energy budget.
        
        Formula: E_available = B - ∑_i E_i
        
        Returns:
            Available energy for feature activation
        """
        # Sum energy used by recently activated features
        used_energy = sum([
            self.feature_energy_costs[feature] * count 
            for feature, count in self.recent_activations.items()
        ])
        
        return max(0.0, self.total_energy_budget - used_energy)
    
    def calculate_cooldown(self, feature: FeatureActivation) -> float:
        """
        Calculate temporal cooldown for recursive features.
        
        Formula: T_cool = min_interval + θ · log(1 + R_recent)
        
        Args:
            feature: Feature to calculate cooldown for
            
        Returns:
            Cooldown time in seconds
        """
        # Get count of recent activations of this feature
        recent_count = self.recent_activations.get(feature, 0)
        
        # Apply logarithmic dampening
        dampening_factor = 2.0  # θ parameter
        cooldown = self.min_interval + dampening_factor * math.log(1.0 + recent_count)
        
        return cooldown
    
    def is_feature_allowed(
        self, 
        feature: FeatureActivation, 
        metric_value: Optional[float] = None
    ) -> bool:
        """
        Determine if a feature is allowed to activate based on complexity constraints.
        
        Args:
            feature: Feature to check activation for
            metric_value: Optional specific metric value to use for activation
            
        Returns:
            True if feature can be activated, False otherwise
        """
        # Update complexity state
        complexity, current_tier = self._update_complexity()
        
        # Map features to required tiers
        feature_tiers = {
            FeatureActivation.MEMORY_STORAGE: ComplexityTier.TIER_0,
            FeatureActivation.CONTRADICTION_DETECTION: ComplexityTier.TIER_0,
            FeatureActivation.COHERENCE_TRACKING: ComplexityTier.TIER_0,
            FeatureActivation.REFLECTION: ComplexityTier.TIER_1,
            FeatureActivation.VALUE_DECAY: ComplexityTier.TIER_1,
            FeatureActivation.SCHEMA_MUTATION: ComplexityTier.TIER_2,
            FeatureActivation.IDENTITY_MODELING: ComplexityTier.TIER_3,
            FeatureActivation.LEGACY_CREATION: ComplexityTier.TIER_3,
        }
        
        required_tier = feature_tiers.get(feature, ComplexityTier.TIER_3)
        
        # Check if we're at the required tier
        if current_tier.value < required_tier.value:
            logger.debug(f"Feature {feature.name} denied - current tier {current_tier.name} below required {required_tier.name}")
            return False
        
        # Check for feedback loop - restrict high-tier features
        if self.is_feedback_loop_detected() and feature in [
            FeatureActivation.SCHEMA_MUTATION, 
            FeatureActivation.IDENTITY_MODELING,
            FeatureActivation.LEGACY_CREATION
        ]:
            logger.warning(f"Feature {feature.name} blocked due to feedback loop detection")
            return False
        
        # Check cooldown period
        last_activation_time = self.last_activation.get(feature, 0.0)
        cooldown = self.calculate_cooldown(feature)
        time_since_last = time.time() - last_activation_time
        
        if time_since_last < cooldown:
            logger.debug(f"Feature {feature.name} in cooldown period: {time_since_last:.1f}s < {cooldown:.1f}s")
            return False
        
        # Check energy budget
        energy_required = self.feature_energy_costs.get(feature, 0.0)
        if energy_required > self.available_energy():
            logger.debug(f"Feature {feature.name} denied due to insufficient energy budget")
            return False
        
        # Apply sigmoid probability based on metric if provided
        if metric_value is not None:
            # Adjust activation parameters based on stability
            self.adjust_activation_parameters(feature)
            
            # Calculate activation probability
            activation_prob = self.sigmoid_activation(metric_value, feature)
            
            # Probabilistic activation
            if random.random() > activation_prob:
                logger.debug(f"Feature {feature.name} probabilistically denied: {activation_prob:.2f}")
                return False
        
        return True
    
    def activate_feature(
        self, 
        feature: FeatureActivation, 
        metric_value: Optional[float] = None
    ) -> bool:
        """
        Try to activate a feature based on complexity constraints.
        
        Args:
            feature: Feature to activate
            metric_value: Optional metric value to use for probabilistic activation
            
        Returns:
            True if feature was activated, False otherwise
        """
        if self.is_feature_allowed(feature, metric_value):
            # Update activation time
            self.last_activation[feature] = time.time()
            
            # Increment recent activations counter
            self.recent_activations[feature] = self.recent_activations.get(feature, 0) + 1
            
            logger.info(f"Feature {feature.name} activated")
            return True
        
        return False
    
    def update_energy_budget(self, new_budget: float):
        """
        Update the total energy budget available for feature activation.
        
        Args:
            new_budget: New energy budget value
        """
        self.total_energy_budget = max(0.0, new_budget)
    
    def decay_recent_activations(self, decay_factor: float = 0.5):
        """
        Decay the count of recent activations to prevent accumulation.
        
        Args:
            decay_factor: Factor to decay counts by (applied hourly)
        """
        # Calculate hours since last decay
        current_time = time.time()
        
        # Apply decay to all counts
        for feature in self.recent_activations:
            current_count = self.recent_activations[feature]
            if current_count > 0:
                # Apply decay
                self.recent_activations[feature] = max(0, int(current_count * decay_factor))
    
    def get_complexity_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current complexity state.
        
        Returns:
            Dictionary of complexity statistics
        """
        complexity, current_tier = self._update_complexity()
        
        return {
            "complexity_score": complexity,
            "current_tier": current_tier.name,
            "tier_stable": self.is_tier_stable(),
            "feedback_loop_detected": self.is_feedback_loop_detected(),
            "available_energy": self.available_energy(),
            "memory_count": len(self.memory_store.memories),
            "schema_size": len(self.schema_graph.graph.nodes) if self.schema_graph else 0,
            "recent_activations": {feature.name: count for feature, count in self.recent_activations.items()},
            "activation_parameters": {
                feature.name: self.activation_params[feature] for feature in FeatureActivation
            }
        }


# Decorator for complexity-controlled functions
def complexity_controlled(
    feature: FeatureActivation,
    controller_attr: str = "complexity_controller",
    metric_getter: Optional[Callable] = None
):
    """
    Decorator to make a function controlled by complexity constraints.
    
    Args:
        feature: Feature type this function represents
        controller_attr: Attribute name to access complexity controller
        metric_getter: Optional function to extract metric from args/kwargs
    
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Get controller from instance
            controller = getattr(self, controller_attr, None)
            if not controller or not isinstance(controller, ComplexityController):
                # No controller, just call the function
                return func(self, *args, **kwargs)
            
            # Get metric value if provided
            metric_value = None
            if metric_getter:
                metric_value = metric_getter(self, *args, **kwargs)
            
            # Check if feature is allowed
            if controller.activate_feature(feature, metric_value):
                # Feature allowed, call the function
                return func(self, *args, **kwargs)
            else:
                # Feature denied
                logger.warning(f"Function {func.__name__} blocked by complexity controller")
                return None
                
        return wrapper
    return decorator 