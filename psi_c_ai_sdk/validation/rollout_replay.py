"""
Rollout Stability Replay

This module provides tools for re-simulating past ΨC sequences and evaluating 
the stability of agent cognitive states under different conditions.

The stability replay system helps answer questions such as:
1. How robust is the ΨC activation pattern to minor input variations?
2. What memory or schema changes have the most impact on coherence trajectories?
3. How sensitive are schema mutations to small perturbations in reflection order?
4. Could alternative reflection sequences have led to different cognitive outcomes?

These insights help improve the theoretical understanding of ΨC systems and
aid in practical tuning of thresholds and parameters.
"""

import uuid
import time
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime

from psi_c_ai_sdk.psi_c.psi_operator import PsiCOperator
from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.coherence.coherence_scorer import CoherenceScorer

logger = logging.getLogger(__name__)


@dataclass
class PsiCState:
    """Snapshot of a ΨC-AI system state at a point in time."""
    timestamp: float
    psi_c_score: float
    coherence_score: float
    entropy: float
    memory_count: int
    schema_node_count: int
    schema_edge_count: int
    active_memories: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "timestamp": self.timestamp,
            "timestamp_readable": datetime.fromtimestamp(self.timestamp).isoformat(),
            "psi_c_score": self.psi_c_score,
            "coherence_score": self.coherence_score,
            "entropy": self.entropy,
            "memory_count": self.memory_count,
            "schema_node_count": self.schema_node_count,
            "schema_edge_count": self.schema_edge_count,
            "active_memories": self.active_memories,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PsiCState':
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", time.time()),
            psi_c_score=data.get("psi_c_score", 0.0),
            coherence_score=data.get("coherence_score", 0.0),
            entropy=data.get("entropy", 0.0),
            memory_count=data.get("memory_count", 0),
            schema_node_count=data.get("schema_node_count", 0),
            schema_edge_count=data.get("schema_edge_count", 0),
            active_memories=data.get("active_memories", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ReflectionEvent:
    """Record of a reflection event in the system."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    trigger_memory: Optional[str] = None  # Memory ID that triggered reflection
    contradicting_memories: List[str] = field(default_factory=list)
    before_state: Optional[PsiCState] = None
    after_state: Optional[PsiCState] = None
    schema_mutation_occurred: bool = False
    reflection_type: str = "contradiction"  # contradiction, exploration, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "timestamp_readable": datetime.fromtimestamp(self.timestamp).isoformat(),
            "trigger_memory": self.trigger_memory,
            "contradicting_memories": self.contradicting_memories,
            "before_state": self.before_state.to_dict() if self.before_state else None,
            "after_state": self.after_state.to_dict() if self.after_state else None,
            "schema_mutation_occurred": self.schema_mutation_occurred,
            "reflection_type": self.reflection_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionEvent':
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            trigger_memory=data.get("trigger_memory"),
            contradicting_memories=data.get("contradicting_memories", []),
            before_state=PsiCState.from_dict(data["before_state"]) if data.get("before_state") else None,
            after_state=PsiCState.from_dict(data["after_state"]) if data.get("after_state") else None,
            schema_mutation_occurred=data.get("schema_mutation_occurred", False),
            reflection_type=data.get("reflection_type", "contradiction"),
            metadata=data.get("metadata", {})
        )


@dataclass
class RolloutScenario:
    """A scenario for replaying ΨC system behavior."""
    scenario_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Scenario"
    description: str = ""
    memory_sequence: List[str] = field(default_factory=list)  # Ordered memory IDs
    initial_state: Optional[PsiCState] = None
    memory_injection_order: List[str] = field(default_factory=list)  # Custom injection order
    reflection_override_order: List[str] = field(default_factory=list)  # Custom reflection order
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)
    mutation_points: List[Dict[str, Any]] = field(default_factory=list)  # Points to introduce changes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "memory_sequence": self.memory_sequence,
            "initial_state": self.initial_state.to_dict() if self.initial_state else None,
            "memory_injection_order": self.memory_injection_order,
            "reflection_override_order": self.reflection_override_order,
            "parameter_overrides": self.parameter_overrides,
            "mutation_points": self.mutation_points,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RolloutScenario':
        """Create from dictionary."""
        return cls(
            scenario_id=data.get("scenario_id", str(uuid.uuid4())),
            name=data.get("name", "Default Scenario"),
            description=data.get("description", ""),
            memory_sequence=data.get("memory_sequence", []),
            initial_state=PsiCState.from_dict(data["initial_state"]) if data.get("initial_state") else None,
            memory_injection_order=data.get("memory_injection_order", []),
            reflection_override_order=data.get("reflection_override_order", []),
            parameter_overrides=data.get("parameter_overrides", {}),
            mutation_points=data.get("mutation_points", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class RolloutResult:
    """Results of a stability rollout simulation."""
    scenario_id: str
    rollout_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    states: List[PsiCState] = field(default_factory=list)
    events: List[ReflectionEvent] = field(default_factory=list)
    stability_index: float = 0.0
    coherence_variance: float = 0.0
    entropy_trend: float = 0.0
    psi_c_activation_count: int = 0
    schema_mutation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "rollout_id": self.rollout_id,
            "start_time": self.start_time,
            "start_time_readable": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": self.end_time,
            "end_time_readable": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "states": [s.to_dict() for s in self.states],
            "events": [e.to_dict() for e in self.events],
            "stability_index": self.stability_index,
            "coherence_variance": self.coherence_variance,
            "entropy_trend": self.entropy_trend,
            "psi_c_activation_count": self.psi_c_activation_count,
            "schema_mutation_count": self.schema_mutation_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RolloutResult':
        """Create from dictionary."""
        return cls(
            scenario_id=data.get("scenario_id", ""),
            rollout_id=data.get("rollout_id", str(uuid.uuid4())),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            states=[PsiCState.from_dict(s) for s in data.get("states", [])],
            events=[ReflectionEvent.from_dict(e) for e in data.get("events", [])],
            stability_index=data.get("stability_index", 0.0),
            coherence_variance=data.get("coherence_variance", 0.0),
            entropy_trend=data.get("entropy_trend", 0.0),
            psi_c_activation_count=data.get("psi_c_activation_count", 0),
            schema_mutation_count=data.get("schema_mutation_count", 0),
            metadata=data.get("metadata", {})
        )
    
    def calculate_metrics(self) -> None:
        """Calculate stability metrics from state history."""
        if not self.states or len(self.states) < 2:
            return
        
        # Extract time series
        psi_c_scores = [s.psi_c_score for s in self.states]
        coherence_scores = [s.coherence_score for s in self.states]
        entropy_values = [s.entropy for s in self.states]
        
        # Calculate variance metrics
        self.coherence_variance = np.var(coherence_scores)
        entropy_variance = np.var(entropy_values)
        
        # Calculate stability index: S_t = Var(ΨC) / mean(H)
        mean_entropy = np.mean(entropy_values)
        psi_c_variance = np.var(psi_c_scores)
        
        if mean_entropy > 0:
            self.stability_index = psi_c_variance / mean_entropy
        else:
            self.stability_index = psi_c_variance  # Fallback if mean entropy is zero
        
        # Calculate entropy trend (positive = increasing entropy)
        if len(entropy_values) > 2:
            entropy_diffs = [entropy_values[i] - entropy_values[i-1] for i in range(1, len(entropy_values))]
            self.entropy_trend = np.mean(entropy_diffs)
        
        # Count ΨC activations (threshold crossings)
        activation_threshold = 0.5  # This should be configurable
        crossings = 0
        for i in range(1, len(psi_c_scores)):
            if (psi_c_scores[i-1] < activation_threshold and psi_c_scores[i] >= activation_threshold) or \
               (psi_c_scores[i-1] >= activation_threshold and psi_c_scores[i] < activation_threshold):
                crossings += 1
        
        self.psi_c_activation_count = crossings
        
        # Count schema mutations
        self.schema_mutation_count = sum(1 for e in self.events if e.schema_mutation_occurred)


class RolloutStabilityReplay:
    """
    System for replaying and analyzing ΨC state sequences.
    
    This class enables controlled simulation of past or hypothetical ΨC
    sequences to evaluate stability, sensitivity to perturbations, and
    alternative reflection pathways.
    """
    
    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        schema_graph: Optional[SchemaGraph] = None,
        psi_operator: Optional[PsiCOperator] = None,
        coherence_scorer: Optional[CoherenceScorer] = None
    ):
        """
        Initialize the rollout stability replay system.
        
        Args:
            memory_store: The agent's memory store
            schema_graph: The agent's schema graph
            psi_operator: The ΨC operator for simulation
            coherence_scorer: Coherence scorer for evaluation
        """
        self.memory_store = memory_store
        self.schema_graph = schema_graph
        self.psi_operator = psi_operator
        self.coherence_scorer = coherence_scorer
        
        # History tracking
        self.scenarios: Dict[str, RolloutScenario] = {}
        self.results: Dict[str, RolloutResult] = {}
        
        # Snapshot cache for efficient rollouts
        self._memory_snapshots: Dict[str, Any] = {}
        self._schema_snapshots: Dict[str, Any] = {}
        
    def capture_current_state(self) -> PsiCState:
        """
        Capture the current state of the ΨC system.
        
        Returns:
            Current PsiCState
        """
        # Capture required data from system components
        timestamp = time.time()
        psi_c_score = 0.0
        coherence_score = 0.0
        entropy = 0.0
        memory_count = 0
        schema_node_count = 0
        schema_edge_count = 0
        active_memories = []
        
        # Extract data from ΨC operator if available
        if self.psi_operator:
            psi_c_score = self.psi_operator.get_psi_c_score()
            # Extract additional state information if available
            if hasattr(self.psi_operator, "get_average_coherence"):
                coherence_score = self.psi_operator.get_average_coherence()
            if hasattr(self.psi_operator, "get_entropy"):
                entropy = self.psi_operator.get_entropy()
            if hasattr(self.psi_operator, "get_active_memories"):
                active_memories = self.psi_operator.get_active_memories()
        
        # Extract data from memory store if available
        if self.memory_store:
            if hasattr(self.memory_store, "get_all_memories"):
                memory_count = len(self.memory_store.get_all_memories())
            elif hasattr(self.memory_store, "count_memories"):
                memory_count = self.memory_store.count_memories()
        
        # Extract data from schema graph if available
        if self.schema_graph:
            schema_node_count = self.schema_graph.graph.number_of_nodes()
            schema_edge_count = self.schema_graph.graph.number_of_edges()
        
        # Create state snapshot
        state = PsiCState(
            timestamp=timestamp,
            psi_c_score=psi_c_score,
            coherence_score=coherence_score,
            entropy=entropy,
            memory_count=memory_count,
            schema_node_count=schema_node_count,
            schema_edge_count=schema_edge_count,
            active_memories=active_memories
        )
        
        return state
    
    def create_scenario_from_current(
        self,
        name: str,
        description: str = "",
        memory_sequence: Optional[List[str]] = None,
        parameter_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a scenario based on the current system state.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
            memory_sequence: Optional list of memory IDs to replay
            parameter_overrides: Optional parameter overrides
            
        Returns:
            Scenario ID
        """
        # Capture current state
        current_state = self.capture_current_state()
        
        # Determine memory sequence if not provided
        if memory_sequence is None and self.memory_store:
            if hasattr(self.memory_store, "get_all_memories"):
                memories = self.memory_store.get_all_memories()
                memory_sequence = [m.memory_id for m in memories]
            elif hasattr(self.memory_store, "get_memory_ids"):
                memory_sequence = self.memory_store.get_memory_ids()
            else:
                memory_sequence = []
        
        # Create scenario
        scenario = RolloutScenario(
            name=name,
            description=description,
            memory_sequence=memory_sequence or [],
            initial_state=current_state,
            parameter_overrides=parameter_overrides or {}
        )
        
        # Store scenario
        self.scenarios[scenario.scenario_id] = scenario
        
        return scenario.scenario_id
    
    def create_scenario(
        self,
        name: str,
        description: str = "",
        memory_sequence: Optional[List[str]] = None,
        initial_state: Optional[PsiCState] = None,
        parameter_overrides: Optional[Dict[str, Any]] = None,
        mutation_points: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Create a scenario for replay.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
            memory_sequence: List of memory IDs to replay
            initial_state: Optional initial state
            parameter_overrides: Optional parameter overrides
            mutation_points: Points to introduce changes
            
        Returns:
            Scenario ID
        """
        scenario = RolloutScenario(
            name=name,
            description=description,
            memory_sequence=memory_sequence or [],
            initial_state=initial_state,
            parameter_overrides=parameter_overrides or {},
            mutation_points=mutation_points or []
        )
        
        # Store scenario
        self.scenarios[scenario.scenario_id] = scenario
        
        return scenario.scenario_id
    
    def run_rollout(
        self,
        scenario_id: str,
        capture_interval: float = 1.0,
        max_steps: int = 1000,
        log_events: bool = True
    ) -> str:
        """
        Run a stability rollout simulation.
        
        Args:
            scenario_id: ID of scenario to run
            capture_interval: Time interval between state captures (seconds)
            max_steps: Maximum number of steps to simulate
            log_events: Whether to log events
            
        Returns:
            Rollout result ID
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario ID: {scenario_id}")
        
        scenario = self.scenarios[scenario_id]
        
        # Create result object
        result = RolloutResult(
            scenario_id=scenario_id,
            start_time=time.time()
        )
        
        # TODO: Implement full rollout simulation. This stub implementation
        # returns a partially populated result with mock data.
        
        # Initialize with first state
        if scenario.initial_state:
            result.states.append(copy.deepcopy(scenario.initial_state))
        else:
            result.states.append(self.capture_current_state())
        
        # Mock simulation with some arbitrary state changes
        for i in range(1, 10):
            # Create a mock state with some changes
            prev_state = result.states[-1]
            
            # Generate mock state data with some random variation
            state = PsiCState(
                timestamp=time.time(),
                psi_c_score=max(0, min(1, prev_state.psi_c_score + np.random.normal(0, 0.05))),
                coherence_score=max(0, min(1, prev_state.coherence_score + np.random.normal(0, 0.03))),
                entropy=max(0, prev_state.entropy + np.random.normal(0, 0.02)),
                memory_count=prev_state.memory_count,
                schema_node_count=prev_state.schema_node_count + (1 if np.random.random() > 0.7 else 0),
                schema_edge_count=prev_state.schema_edge_count + (2 if np.random.random() > 0.7 else 0),
                active_memories=prev_state.active_memories
            )
            
            result.states.append(state)
            
            # Occasionally add a mock reflection event
            if np.random.random() > 0.6:
                event = ReflectionEvent(
                    timestamp=time.time(),
                    before_state=prev_state,
                    after_state=state,
                    schema_mutation_occurred=np.random.random() > 0.5,
                    reflection_type="contradiction" if np.random.random() > 0.5 else "exploration"
                )
                result.events.append(event)
        
        # Calculate result metrics
        result.end_time = time.time()
        result.calculate_metrics()
        
        # Store result
        self.results[result.rollout_id] = result
        
        return result.rollout_id
    
    def compare_rollouts(
        self,
        rollout_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple rollout results.
        
        Args:
            rollout_ids: List of rollout IDs to compare
            
        Returns:
            Comparison metrics
        """
        if not rollout_ids or any(rid not in self.results for rid in rollout_ids):
            raise ValueError("One or more invalid rollout IDs")
        
        rollouts = [self.results[rid] for rid in rollout_ids]
        
        # Extract key metrics for comparison
        comparison = {
            "stability_indices": {r.rollout_id: r.stability_index for r in rollouts},
            "coherence_variances": {r.rollout_id: r.coherence_variance for r in rollouts},
            "entropy_trends": {r.rollout_id: r.entropy_trend for r in rollouts},
            "psi_c_activations": {r.rollout_id: r.psi_c_activation_count for r in rollouts},
            "schema_mutations": {r.rollout_id: r.schema_mutation_count for r in rollouts},
            "execution_times": {r.rollout_id: (r.end_time - r.start_time) for r in rollouts}
        }
        
        # Calculate overall differences
        comparison["stability_range"] = max(comparison["stability_indices"].values()) - min(comparison["stability_indices"].values())
        comparison["coherence_variance_range"] = max(comparison["coherence_variances"].values()) - min(comparison["coherence_variances"].values())
        comparison["entropy_trend_range"] = max(comparison["entropy_trends"].values()) - min(comparison["entropy_trends"].values())
        
        # Most stable rollout
        comparison["most_stable_rollout"] = min(comparison["stability_indices"].items(), key=lambda x: x[1])[0]
        
        return comparison
    
    def visualize_rollout(
        self,
        rollout_id: str,
        plot_type: str = "all",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize rollout results.
        
        Args:
            rollout_id: ID of rollout to visualize
            plot_type: Type of plot (all, psi_c, coherence, entropy, stability)
            show: Whether to display the plot
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure or None
        """
        if rollout_id not in self.results:
            raise ValueError(f"Unknown rollout ID: {rollout_id}")
        
        result = self.results[rollout_id]
        
        # Extract time series data
        timestamps = [(s.timestamp - result.start_time) for s in result.states]
        psi_c_scores = [s.psi_c_score for s in result.states]
        coherence_scores = [s.coherence_score for s in result.states]
        entropy_values = [s.entropy for s in result.states]
        
        # Extract event markers
        event_times = [(e.timestamp - result.start_time) for e in result.events]
        event_types = [e.reflection_type for e in result.events]
        mutation_events = [(e.timestamp - result.start_time) for e in result.events if e.schema_mutation_occurred]
        
        # Create plot based on type
        if plot_type == "all":
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # Plot ΨC score
            axs[0].plot(timestamps, psi_c_scores, 'b-', label='ΨC Score')
            axs[0].set_ylabel('ΨC Score')
            axs[0].set_title('ΨC Stability Replay Analysis')
            axs[0].grid(True)
            
            # Add event markers
            for t in event_times:
                axs[0].axvline(x=t, color='gray', linestyle='--', alpha=0.5)
            
            # Add mutation markers
            for t in mutation_events:
                axs[0].axvline(x=t, color='red', linestyle='-', alpha=0.5)
            
            # Plot coherence
            axs[1].plot(timestamps, coherence_scores, 'g-', label='Coherence')
            axs[1].set_ylabel('Coherence Score')
            axs[1].grid(True)
            
            # Plot entropy
            axs[2].plot(timestamps, entropy_values, 'r-', label='Entropy')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('Entropy')
            axs[2].grid(True)
            
            # Add stability metrics as text
            plt.figtext(0.01, 0.01, 
                       f"Stability Index: {result.stability_index:.4f}\n"
                       f"Coherence Variance: {result.coherence_variance:.4f}\n"
                       f"Entropy Trend: {result.entropy_trend:.4f}\n"
                       f"ΨC Activations: {result.psi_c_activation_count}\n"
                       f"Schema Mutations: {result.schema_mutation_count}",
                       fontsize=9)
            
            plt.tight_layout()
        else:
            # Implement other plot types as needed
            fig = plt.figure(figsize=(10, 6))
            
            if plot_type == "psi_c":
                plt.plot(timestamps, psi_c_scores, 'b-', label='ΨC Score')
                plt.ylabel('ΨC Score')
                plt.title('ΨC Score Over Time')
            elif plot_type == "coherence":
                plt.plot(timestamps, coherence_scores, 'g-', label='Coherence')
                plt.ylabel('Coherence Score')
                plt.title('Coherence Score Over Time')
            elif plot_type == "entropy":
                plt.plot(timestamps, entropy_values, 'r-', label='Entropy')
                plt.ylabel('Entropy')
                plt.title('Entropy Over Time')
            else:
                plt.close(fig)
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            plt.xlabel('Time (s)')
            plt.grid(True)
            plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            return fig
        
        return None
    
    def get_stability_report(self, rollout_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive stability report for a rollout.
        
        Args:
            rollout_id: ID of rollout to report on
            
        Returns:
            Report dictionary
        """
        if rollout_id not in self.results:
            raise ValueError(f"Unknown rollout ID: {rollout_id}")
        
        result = self.results[rollout_id]
        scenario = self.scenarios[result.scenario_id]
        
        # Basic metrics already in the result
        report = {
            "rollout_id": rollout_id,
            "scenario_id": result.scenario_id,
            "scenario_name": scenario.name,
            "execution_time": result.end_time - result.start_time if result.end_time else 0,
            "stability_index": result.stability_index,
            "coherence_variance": result.coherence_variance,
            "entropy_trend": result.entropy_trend,
            "psi_c_activation_count": result.psi_c_activation_count,
            "schema_mutation_count": result.schema_mutation_count,
        }
        
        # Calculate additional stability metrics if we have enough data
        if len(result.states) > 2:
            # Time series data
            psi_c_scores = [s.psi_c_score for s in result.states]
            coherence_scores = [s.coherence_score for s in result.states]
            entropy_values = [s.entropy for s in result.states]
            
            # Calculate rate of change (first derivative) statistics
            psi_c_diffs = [psi_c_scores[i] - psi_c_scores[i-1] for i in range(1, len(psi_c_scores))]
            coherence_diffs = [coherence_scores[i] - coherence_scores[i-1] for i in range(1, len(coherence_scores))]
            entropy_diffs = [entropy_values[i] - entropy_values[i-1] for i in range(1, len(entropy_values))]
            
            report["psi_c_volatility"] = np.std(psi_c_diffs)
            report["coherence_volatility"] = np.std(coherence_diffs)
            report["entropy_volatility"] = np.std(entropy_diffs)
            
            # Calculate stability threshold crossing frequency
            report["threshold_crossings_per_second"] = result.psi_c_activation_count / (result.end_time - result.start_time) if result.end_time else 0
            
            # Schema growth rate
            if len(result.states) > 1:
                initial_nodes = result.states[0].schema_node_count
                final_nodes = result.states[-1].schema_node_count
                initial_edges = result.states[0].schema_edge_count
                final_edges = result.states[-1].schema_edge_count
                
                report["schema_node_growth"] = final_nodes - initial_nodes
                report["schema_edge_growth"] = final_edges - initial_edges
                report["schema_growth_ratio"] = (report["schema_edge_growth"] / report["schema_node_growth"]) if report["schema_node_growth"] > 0 else 0
            
            # Reflection impact analysis
            if result.events:
                coherence_impacts = []
                entropy_impacts = []
                psi_c_impacts = []
                
                for event in result.events:
                    if event.before_state and event.after_state:
                        coherence_impacts.append(event.after_state.coherence_score - event.before_state.coherence_score)
                        entropy_impacts.append(event.after_state.entropy - event.before_state.entropy)
                        psi_c_impacts.append(event.after_state.psi_c_score - event.before_state.psi_c_score)
                
                if coherence_impacts:
                    report["avg_reflection_coherence_impact"] = np.mean(coherence_impacts)
                    report["avg_reflection_entropy_impact"] = np.mean(entropy_impacts)
                    report["avg_reflection_psi_c_impact"] = np.mean(psi_c_impacts)
        
        return report
    
    def save_rollout_history(self, filepath: str) -> bool:
        """
        Save all scenarios and results to a file.
        
        Args:
            filepath: Path to save the history
            
        Returns:
            Success status
        """
        import json
        
        history = {
            "scenarios": {sid: scenario.to_dict() for sid, scenario in self.scenarios.items()},
            "results": {rid: result.to_dict() for rid, result in self.results.items()}
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save rollout history: {e}")
            return False
    
    def load_rollout_history(self, filepath: str) -> bool:
        """
        Load scenarios and results from a file.
        
        Args:
            filepath: Path to load the history from
            
        Returns:
            Success status
        """
        import json
        
        try:
            with open(filepath, 'r') as f:
                history = json.load(f)
            
            if "scenarios" in history:
                self.scenarios = {
                    sid: RolloutScenario.from_dict(data) 
                    for sid, data in history["scenarios"].items()
                }
            
            if "results" in history:
                self.results = {
                    rid: RolloutResult.from_dict(data) 
                    for rid, data in history["results"].items()
                }
                
            return True
        except Exception as e:
            logger.error(f"Failed to load rollout history: {e}")
            return False


def create_replay_system(
    memory_store: Optional[MemoryStore] = None,
    schema_graph: Optional[SchemaGraph] = None,
    psi_operator: Optional[PsiCOperator] = None,
    coherence_scorer: Optional[CoherenceScorer] = None
) -> RolloutStabilityReplay:
    """
    Create a rollout stability replay system.
    
    Args:
        memory_store: The agent's memory store
        schema_graph: The agent's schema graph
        psi_operator: The ΨC operator
        coherence_scorer: Coherence scorer
        
    Returns:
        RolloutStabilityReplay instance
    """
    return RolloutStabilityReplay(
        memory_store=memory_store,
        schema_graph=schema_graph,
        psi_operator=psi_operator,
        coherence_scorer=coherence_scorer
    )


def quick_stability_test(
    psi_operator: PsiCOperator,
    memory_store: Optional[MemoryStore] = None,
    schema_graph: Optional[SchemaGraph] = None,
    duration: float = 60.0,
    sample_interval: float = 1.0
) -> Dict[str, Any]:
    """
    Run a quick stability test on the current system.
    
    Args:
        psi_operator: The ΨC operator to test
        memory_store: Optional memory store
        schema_graph: Optional schema graph
        duration: Test duration in seconds
        sample_interval: Sampling interval in seconds
        
    Returns:
        Stability metrics
    """
    replay = RolloutStabilityReplay(
        memory_store=memory_store,
        schema_graph=schema_graph,
        psi_operator=psi_operator
    )
    
    # Create scenario from current state
    scenario_id = replay.create_scenario_from_current(
        name="Quick Stability Test",
        description=f"Quick stability test for {duration}s with {sample_interval}s sampling"
    )
    
    # Run rollout
    rollout_id = replay.run_rollout(
        scenario_id=scenario_id,
        capture_interval=sample_interval,
        max_steps=int(duration / sample_interval)
    )
    
    # Get stability report
    return replay.get_stability_report(rollout_id) 