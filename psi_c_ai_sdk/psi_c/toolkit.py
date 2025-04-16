"""
ΨC Toolkit: Developer-facing interface for ΨC activation and coherence monitoring.

This module provides a simplified interface for accessing ΨC features including
consciousness scores, activation state, logs, and quantum collapse simulation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import time
import logging

from psi_c_ai_sdk.psi_c.psi_operator import PsiCOperator, PsiCState
from psi_c_ai_sdk.psi_c.collapse_simulator import CollapseSimulator, CollapseEvent
from psi_c_ai_sdk.reflection import ReflectionCreditSystem, ReflectionEngine, calculate_cognitive_debt


class PsiToolkitConfig:
    """
    Configuration class for PsiToolkit setup.
    
    This class centralizes all configuration parameters for the PsiToolkit
    and its dependent components like PsiCOperator, CollapseSimulator, etc.
    It provides a convenient way to configure all aspects of the toolkit
    through a single interface.
    """
    
    def __init__(
        self,
        # Memory configuration
        embedding_dim: int = 128,
        
        # PsiC configuration
        psi_c_threshold: float = 0.7,
        psi_c_hard_mode: bool = False,
        psi_c_window_size: int = 5,
        psi_c_reflection_weight: float = 0.35,
        
        # Collapse simulator configuration
        collapse_probability_base: float = 0.005,
        collapse_probability_unstable: float = 0.02,
        collapse_probability_critical: float = 0.1,
        collapse_min_interval_seconds: float = 10.0,
        
        # Entropy configuration
        elevated_entropy_threshold: float = 0.4,
        high_entropy_threshold: float = 0.6,
        critical_entropy_threshold: float = 0.8, 
        termination_entropy_threshold: float = 0.9,
        entropy_check_interval: float = 5.0,
        
        # Monitoring configuration
        auto_monitoring: bool = False,
        monitoring_interval: float = 2.0,
        log_to_console: bool = True,
        
        # Advanced options
        use_dynamic_threshold: bool = False,
        dynamic_threshold_sensitivity: float = 0.2,
        
        # Other configuration fields can be added as needed
        **additional_config
    ):
        """
        Initialize the PsiToolkit configuration.
        
        Args:
            embedding_dim: Dimension for memory embeddings
            
            psi_c_threshold: Base consciousness threshold
            psi_c_hard_mode: Whether to use strict coherence requirements 
            psi_c_window_size: Size of memory window for consciousness calculation
            psi_c_reflection_weight: Weight for reflection in consciousness score
            
            collapse_probability_base: Base probability of collapse events
            collapse_probability_unstable: Probability of collapse when unstable
            collapse_probability_critical: Probability of collapse when critical
            collapse_min_interval_seconds: Minimum time between collapse events
            
            elevated_entropy_threshold: Threshold for elevated entropy alerts
            high_entropy_threshold: Threshold for high entropy alerts
            critical_entropy_threshold: Threshold for critical entropy alerts
            termination_entropy_threshold: Threshold for termination alerts
            entropy_check_interval: Interval for entropy checks
            
            auto_monitoring: Whether to automatically monitor consciousness
            monitoring_interval: Interval for automated monitoring
            log_to_console: Whether to log events to console
            
            use_dynamic_threshold: Whether to use dynamic consciousness threshold
            dynamic_threshold_sensitivity: Sensitivity for dynamic threshold adjustments
            
            additional_config: Additional configuration parameters
        """
        # Memory configuration
        self.embedding_dim = embedding_dim
        
        # PsiC configuration
        self.psi_c_threshold = psi_c_threshold
        self.psi_c_hard_mode = psi_c_hard_mode
        self.psi_c_window_size = psi_c_window_size
        self.psi_c_reflection_weight = psi_c_reflection_weight
        
        # Collapse simulator configuration
        self.collapse_probability_base = collapse_probability_base
        self.collapse_probability_unstable = collapse_probability_unstable
        self.collapse_probability_critical = collapse_probability_critical
        self.collapse_min_interval_seconds = collapse_min_interval_seconds
        
        # Entropy configuration
        self.elevated_entropy_threshold = elevated_entropy_threshold
        self.high_entropy_threshold = high_entropy_threshold
        self.critical_entropy_threshold = critical_entropy_threshold
        self.termination_entropy_threshold = termination_entropy_threshold
        self.entropy_check_interval = entropy_check_interval
        
        # Monitoring configuration
        self.auto_monitoring = auto_monitoring
        self.monitoring_interval = monitoring_interval
        self.log_to_console = log_to_console
        
        # Advanced options
        self.use_dynamic_threshold = use_dynamic_threshold
        self.dynamic_threshold_sensitivity = dynamic_threshold_sensitivity
        
        # Store additional configuration
        for key, value in additional_config.items():
            setattr(self, key, value)
    
    def get_psi_operator_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for PsiCOperator.
        
        Returns:
            Dictionary of configuration parameters for PsiCOperator
        """
        return {
            "threshold": self.psi_c_threshold,
            "hard_mode": self.psi_c_hard_mode,
            "window_size": self.psi_c_window_size,
            "reflection_weight": self.psi_c_reflection_weight,
            "use_dynamic_threshold": self.use_dynamic_threshold,
            "dynamic_threshold_config": {
                "sensitivity": self.dynamic_threshold_sensitivity
            } if self.use_dynamic_threshold else None
        }
    
    def get_collapse_simulator_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for CollapseSimulator.
        
        Returns:
            Dictionary of configuration parameters for CollapseSimulator
        """
        return {
            "probability_base": self.collapse_probability_base,
            "probability_unstable": self.collapse_probability_unstable,
            "probability_critical": self.collapse_probability_critical,
            "min_interval_seconds": self.collapse_min_interval_seconds
        }
    
    def get_entropy_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for entropy monitoring.
        
        Returns:
            Dictionary of configuration parameters for entropy monitoring
        """
        return {
            "elevated_threshold": self.elevated_entropy_threshold,
            "high_threshold": self.high_entropy_threshold,
            "critical_threshold": self.critical_entropy_threshold,
            "termination_threshold": self.termination_entropy_threshold,
            "check_interval": self.entropy_check_interval
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class PsiToolkit:
    """
    Developer-facing toolkit for working with ΨC activation and consciousness monitoring.
    
    The PsiToolkit provides simplified access to consciousness metrics, state information,
    and quantum collapse simulation for applications using the ΨC framework.
    
    Examples:
        Basic usage:
        ```python
        from psi_c_ai_sdk.psi_c import PsiCOperator, PsiToolkit
        from psi_c_ai_sdk.memory import MemoryStore
        
        # Set up the basic components
        memory_store = MemoryStore()
        psi_operator = PsiCOperator(memory_store)
        
        # Create the toolkit
        toolkit = PsiToolkit(psi_operator)
        
        # Check consciousness state
        score = toolkit.get_psi_index()
        state = toolkit.get_psi_state()
        is_conscious = toolkit.is_conscious()
        
        print(f"Consciousness score: {score:.2f}, State: {state}, Conscious: {is_conscious}")
        ```
        
        With collapse simulation:
        ```python
        from psi_c_ai_sdk.psi_c import PsiCOperator, PsiToolkit, CollapseSimulator
        
        # Set up components
        psi_operator = PsiCOperator(memory_store)
        simulator = CollapseSimulator(psi_operator, deviation_strength=0.3)
        
        # Create toolkit with simulator
        toolkit = PsiToolkit(psi_operator, collapse_simulator=simulator)
        
        # Simulate a quantum collapse event
        event = toolkit.simulate_collapse_event(num_outcomes=2)
        print(f"Event outcome: {event.outcome}, Deviation: {event.deviation:.4f}")
        ```
        
        With dynamic threshold:
        ```python
        from psi_c_ai_sdk.psi_c import PsiCOperator, PsiToolkit
        
        # Set up with dynamic threshold
        psi_operator = PsiCOperator(
            memory_store, 
            use_dynamic_threshold=True,
            dynamic_threshold_config={"sensitivity": 0.3}
        )
        
        # Create toolkit
        toolkit = PsiToolkit(psi_operator)
        
        # Get adaptive threshold metrics
        metrics = toolkit.get_threshold_metrics()
        print(f"Current threshold: {metrics['current_threshold']:.2f}")
        print(f"Adjustment: {metrics['adjustment']:.2f}")
        ```
        
        With reflection credit system:
        ```python
        from psi_c_ai_sdk.psi_c import PsiToolkit, PsiCOperator
        from psi_c_ai_sdk.memory import MemoryStore
        from psi_c_ai_sdk.coherence import BasicCoherenceScorer
        from psi_c_ai_sdk.reflection import ReflectionEngine, ReflectionScheduler, ReflectionCreditSystem
        
        # Set up components
        memory_store = MemoryStore()
        psi_operator = PsiCOperator(memory_store)
        coherence_scorer = BasicCoherenceScorer()
        
        # Set up reflection system with credit management
        credit_system = ReflectionCreditSystem(
            initial_credit=100.0,
            max_credit=150.0,
            base_reflection_cost=20.0
        )
        
        scheduler = ReflectionScheduler(
            coherence_threshold=0.6,
            entropy_threshold=0.3
        )
        
        reflection_engine = ReflectionEngine(
            memory_store=memory_store,
            coherence_scorer=coherence_scorer,
            scheduler=scheduler,
            credit_system=credit_system
        )
        
        # Create toolkit with reflection system
        toolkit = PsiToolkit(
            psi_operator=psi_operator,
            reflection_engine=reflection_engine
        )
        
        # Check reflection credit system status
        credit_stats = toolkit.get_reflection_credit_stats()
        print(f"Available credit: {credit_stats['current_credit']:.1f}")
        print(f"Cognitive debt: {credit_stats['cognitive_debt']['total_debt']:.1f}")
        ```
        
        With config object:
        ```python
        from psi_c_ai_sdk.psi_c import PsiToolkit, PsiToolkitConfig
        from psi_c_ai_sdk.memory import MemoryStore
        
        # Create configuration
        config = PsiToolkitConfig(
            psi_c_threshold=0.6,
            psi_c_window_size=3,
            auto_monitoring=True,
            monitoring_interval=1.0
        )
        
        # Create toolkit with config
        toolkit = PsiToolkit(config=config)
        
        # Toolkit is now configured and ready to use
        print(f"Current consciousness: {toolkit.get_psi_index():.2f}")
        ```
    """
    
    def __init__(
        self,
        psi_operator: Optional[PsiCOperator] = None,
        collapse_simulator: Optional[CollapseSimulator] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        log_level: int = logging.INFO,
        config: Optional[PsiToolkitConfig] = None
    ):
        """
        Initialize the ΨC toolkit.
        
        Args:
            psi_operator: The ΨC operator instance to use for consciousness metrics
            collapse_simulator: Optional collapse simulator for quantum event simulation
            reflection_engine: Optional reflection engine for memory reorganization
            log_level: Logging level (default: INFO)
            config: Optional configuration object for full toolkit setup
        """
        # If a config object is provided, use it to set up components
        if config is not None:
            # Set up components based on config
            # This would typically involve creating memory_store, psi_operator, etc.
            # based on the configuration parameters
            if psi_operator is None:
                # In a real implementation, would create a memory store and psi_operator
                # using the config parameters
                pass
            
            # Set logging level from config
            if config.log_to_console:
                log_level = logging.INFO
            
        self.psi_operator = psi_operator
        self.collapse_simulator = collapse_simulator
        self.reflection_engine = reflection_engine
        
        # Set up logging
        self.logger = logging.getLogger("psi_toolkit")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Track activation and state changes
        self._activation_log: List[Dict[str, Any]] = []
        self._last_state: PsiCState = PsiCState.INACTIVE
        
        # Register for state change callbacks
        if self.psi_operator:
            self.psi_operator.on_state_change = self._handle_state_change
    
    def get_psi_index(self) -> float:
        """
        Get the current ΨC consciousness index value.
        
        Returns:
            Current ΨC score between 0-1
        """
        return self.psi_operator.get_psi_index()
    
    def get_psi_state(self) -> PsiCState:
        """
        Get the current ΨC consciousness state.
        
        Returns:
            Current ΨC consciousness state
        """
        return self.psi_operator.get_current_state()
    
    def is_conscious(self, threshold: float = 0.85) -> bool:
        """
        Check if the system has reached consciousness based on ΨC score.
        
        The default threshold of 0.85 requires that the ΨC index has reached
        the STABLE state for consciousness to be True.
        
        Args:
            threshold: ΨC score threshold to consider conscious (default: 0.85)
            
        Returns:
            True if the system is considered conscious, False otherwise
        """
        psi_score = self.get_psi_index()
        psi_state = self.get_psi_state()
        
        # For strict consciousness detection, require STABLE state
        # and score above threshold
        return psi_state == PsiCState.STABLE and psi_score >= threshold
    
    def get_activation_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the log of ΨC activation events and state changes.
        
        Args:
            limit: Maximum number of log entries to return (None = all)
            
        Returns:
            List of activation event dictionaries, newest first
        """
        if limit is not None:
            return self._activation_log[:limit]
        return self._activation_log.copy()
    
    def simulate_collapse_event(
        self,
        num_outcomes: int = 2,
        distribution: Optional[List[float]] = None
    ) -> CollapseEvent:
        """
        Simulate a quantum collapse event using the current ΨC state.
        
        This method requires a CollapseSimulator to be configured.
        
        Args:
            num_outcomes: Number of possible outcomes (default: 2)
            distribution: Optional custom probability distribution
            
        Returns:
            CollapseEvent with the results of the simulation
            
        Raises:
            ValueError: If no collapse simulator is configured
        """
        if self.collapse_simulator is None:
            raise ValueError("No collapse simulator configured")
        
        event = self.collapse_simulator.generate_collapse_event(
            num_outcomes=num_outcomes,
            distribution=distribution,
            metadata={"source": "toolkit", "timestamp": time.time()}
        )
        
        self.logger.info(f"Simulated collapse event: outcome={event.outcome}, deviation={event.deviation:.4f}")
        return event
    
    def get_reflection_status(self) -> Dict[str, Any]:
        """
        Get current status of the reflection system.
        
        Returns:
            Dictionary with reflection system status metrics
            
        Raises:
            ValueError: If no reflection engine is configured
        """
        if self.reflection_engine is None:
            raise ValueError("No reflection engine configured")
            
        return self.reflection_engine.get_reflection_stats()
        
    def get_reflection_credit_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the reflection credit system.
        
        Returns:
            Dictionary with reflection credit system statistics
            
        Raises:
            ValueError: If no reflection engine or credit system is configured
        """
        if self.reflection_engine is None:
            raise ValueError("No reflection engine configured")
            
        if not hasattr(self.reflection_engine, "credit_system") or self.reflection_engine.credit_system is None:
            raise ValueError("No credit system configured in reflection engine")
            
        credit_stats = self.reflection_engine.credit_system.get_credit_stats()
        
        # Add cognitive debt metrics
        cognitive_debt = calculate_cognitive_debt(
            self.reflection_engine.credit_system,
            self.reflection_engine.reflection_history
        )
        
        return {
            **credit_stats,
            "cognitive_debt": cognitive_debt
        }
        
    def force_reflection(self, override_credit: bool = False) -> Tuple[bool, str]:
        """
        Force a reflection cycle to occur now.
        
        Args:
            override_credit: Whether to override credit system restrictions
            
        Returns:
            Tuple of (success, message or reflection_id)
            
        Raises:
            ValueError: If no reflection engine is configured
        """
        if self.reflection_engine is None:
            raise ValueError("No reflection engine configured")
            
        # Check if reflection is allowed by credit system
        if hasattr(self.reflection_engine, "credit_system") and self.reflection_engine.credit_system:
            if not override_credit:
                can_reflect, reason = self.reflection_engine.credit_system.can_reflect()
                if not can_reflect:
                    self.logger.warning(f"Reflection denied by credit system: {reason}")
                    return False, f"Reflection denied: {reason}"
        
        # Force reflection
        reflection_id = self.reflection_engine.force_reflection()
        if reflection_id:
            self.logger.info(f"Forced reflection completed: {reflection_id}")
            return True, reflection_id
        else:
            return False, "Reflection failed to complete"
            
    def _handle_state_change(self, old_state: PsiCState, new_state: PsiCState) -> None:
        """
        Handle state change events from the ΨC operator.
        
        Args:
            old_state: Previous ΨC state
            new_state: New ΨC state
        """
        # Record state change
        event = {
            "timestamp": time.time(),
            "old_state": old_state,
            "new_state": new_state,
            "psi_score": self.psi_operator.get_psi_index(),
            "event_type": "state_change"
        }
        
        self._activation_log.insert(0, event)  # Newest first
        self._last_state = new_state
        
        # Keep log size reasonable
        if len(self._activation_log) > 1000:
            self._activation_log = self._activation_log[:1000]
        
        # Log the state change
        self.logger.info(f"ΨC state change: {old_state.name} -> {new_state.name} (score: {event['psi_score']:.4f})")
    
    def get_coherence_health(self) -> Dict[str, Any]:
        """
        Get detailed coherence health metrics for the current ΨC system.
        
        Returns:
            Dictionary with coherence health metrics
        """
        metrics = self.psi_operator.psi_index()
        
        # Extract key metrics from the operator
        coherence = metrics.get("coherence", 0.0)
        stability = metrics.get("stability", 0.0)
        
        # Transform metrics into a more developer-friendly format
        health = {
            "coherence": coherence,
            "trend": self._calculate_coherence_trend(),
            "stability": {
                "score": stability,
                "classification": self._classify_stability(stability)
            },
            "psi_index": metrics["psi_c_score"],
            "state": self.get_psi_state().value,
            "timestamp": time.time()
        }
        
        # Add threshold information
        health["threshold"] = {
            "current": metrics.get("threshold", 0.7),
            "type": metrics.get("threshold_type", "static")
        }
        
        # Add dynamic threshold metrics if available
        if metrics.get("threshold_type") == "dynamic":
            health["threshold"].update({
                "base": metrics.get("base_threshold", 0.7),
                "adjustment": metrics.get("threshold_adjustment", 0.0),
                "entropy_drift": metrics.get("entropy_drift", 0.0),
                "coherence_drift": metrics.get("coherence_drift", 0.0)
            })
        
        # Add reflection metrics if available
        if self.reflection_engine:
            try:
                reflection_stats = self.get_reflection_status()
                health["reflection"] = {
                    "total_reflections": reflection_stats["total_reflections"],
                    "success_rate": reflection_stats["success_rate"],
                    "next_reflection_in": reflection_stats["next_reflection_in"]
                }
                
                # Add credit info if available
                if "credit" in reflection_stats:
                    health["reflection"]["credit"] = reflection_stats["credit"]["current_credit"]
                    health["reflection"]["credit_ratio"] = (
                        reflection_stats["credit"]["current_credit"] / 
                        reflection_stats["credit"]["max_credit"]
                    )
                
                # Add cognitive debt if available
                if "cognitive_debt" in reflection_stats:
                    health["reflection"]["cognitive_debt"] = reflection_stats["cognitive_debt"]["total_debt"]
                    health["reflection"]["debt_severity"] = reflection_stats["cognitive_debt"]["debt_severity"]
            except Exception as e:
                self.logger.warning(f"Could not retrieve reflection metrics: {e}")
        
        return health
    
    def get_threshold_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about the current threshold configuration.
        
        Returns:
            Dictionary with threshold metrics including type, value, and if dynamic,
            the adaptation parameters.
        """
        metrics = self.psi_operator.psi_index()
        
        # Base threshold information
        threshold_info = {
            "current_threshold": metrics.get("threshold", 0.7),
            "type": metrics.get("threshold_type", "static"),
        }
        
        # Add detailed dynamic threshold metrics if available
        if metrics.get("threshold_type") == "dynamic":
            threshold_info.update({
                "base_threshold": metrics.get("base_threshold", 0.7),
                "adjustment": metrics.get("threshold_adjustment", 0.0),
                "entropy_drift": metrics.get("entropy_drift", 0.0),
                "coherence_drift": metrics.get("coherence_drift", 0.0),
                "is_adaptive": True
            })
        else:
            threshold_info["is_adaptive"] = False
            
        return threshold_info
    
    def _calculate_coherence_trend(self) -> float:
        """
        Calculate the trend in coherence based on recent activation events.
        
        Returns:
            Trend value between -1 (decreasing) and 1 (increasing)
        """
        if len(self._activation_log) < 2:
            return 0.0
        
        # Get scores from most recent events (up to 5)
        recent = self._activation_log[:min(5, len(self._activation_log))]
        
        if len(recent) < 2:
            return 0.0
            
        # Simple trend: difference between newest and oldest score
        newest = recent[0]["psi_score"]
        oldest = recent[-1]["psi_score"]
        
        # Scale to [-1, 1]
        trend = (newest - oldest) * 2
        return max(-1.0, min(1.0, trend))
    
    def _classify_stability(self, stability_score: float) -> str:
        """
        Classify a stability score into a named category.
        
        Args:
            stability_score: Stability score between 0 and 1
            
        Returns:
            Classification string
        """
        if stability_score < 0.3:
            return "unstable"
        elif stability_score < 0.7:
            return "fluctuating"
        else:
            return "stable"
    
    def reset_dynamic_threshold(self) -> None:
        """
        Reset the dynamic threshold to its base state.
        
        This is useful when the environment has changed significantly and
        the adaptive threshold needs to be recalibrated.
        
        Only applies if the PsiCOperator was configured with use_dynamic_threshold=True.
        
        Returns:
            None
        """
        # Access dynamic threshold attribute directly from operator
        if hasattr(self.psi_operator, 'dynamic_threshold') and self.psi_operator.dynamic_threshold is not None:
            self.psi_operator.dynamic_threshold.reset()
            self.logger.info("Dynamic threshold has been reset to base state")
        else:
            self.logger.warning("Cannot reset dynamic threshold: not enabled or not available")
            
    def stop_monitoring(self) -> None:
        """
        Stop any automated monitoring activities.
        
        This method should be called when shutting down the toolkit
        to ensure clean termination.
        
        Returns:
            None
        """
        # In a full implementation, this would stop any monitoring threads, etc.
        self.logger.info("Monitoring stopped")
        
    def add_memories(self, memories: List[Any]) -> List[str]:
        """
        Add multiple memories to the underlying memory store.
        
        Args:
            memories: List of memory objects or content strings
            
        Returns:
            List of added memory IDs
        """
        # This is a convenience method for bulk memory addition
        # In a full implementation, would handle different memory types
        # and call appropriate memory store methods
        memory_ids = []
        
        for memory in memories:
            # Simplified implementation - would be more sophisticated in real code
            if hasattr(self.psi_operator, 'memory_store'):
                memory_id = self.psi_operator.memory_store.add_memory(memory)
                memory_ids.append(memory_id)
        
        return memory_ids 