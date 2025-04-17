#!/usr/bin/env python3
"""
ΨC-AI SDK Master Orchestration Script

This module implements the core orchestration cycle for the ΨC system,
integrating all five pillars into a coherent runtime:
1. Memory-Based Identity
2. Reflection Engine
3. Contradiction Detection
4. Coherence Calculation
5. Falsifiability Verification

The CycleController class manages the flow through the following stages:
- Input ingestion
- Reflection processing
- Contradiction detection and resolution
- Memory updates
- Coherence calculation and logging

This implements the full system equation:
S(t+1) = S(t) + [f(Reflection(t), Δ(t))] + g(Δ(t))
where ΨC(t) = σ(∫(t₀ᵗ¹) R(S(t)) · I(S(t), t) dt - θ)
"""

import os
import time
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from psi_c_ai_sdk.memory import MemoryStore, Memory
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.coherence import CoherenceCalculator
from psi_c_ai_sdk.reflection import ReflectionEngine
from psi_c_ai_sdk.contradiction import ContradictionDetector
from psi_c_ai_sdk.identity import self_entropy
from psi_c_ai_sdk.runtime.complexity_controller import ComplexityController, ComplexityTier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputProcessor:
    """
    Handles the ingestion of new information with rich metadata tagging.
    
    Features:
    - Support for multiple input types (text, structured data)
    - Automatic metadata enrichment
    - Configurable preprocessing pipelines
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        source_trust_levels: Optional[Dict[str, float]] = None,
        preprocessors: Optional[List[Callable]] = None
    ):
        """
        Initialize the input processor.
        
        Args:
            memory_store: The memory store to add inputs to
            source_trust_levels: Dictionary mapping sources to trust levels (0.0-1.0)
            preprocessors: Optional list of preprocessing functions
        """
        self.memory_store = memory_store
        self.source_trust_levels = source_trust_levels or {}
        self.preprocessors = preprocessors or []
        
        # Stats tracking
        self.stats = {
            "inputs_processed": 0,
            "total_entropy": 0.0,
            "avg_entropy": 0.0,
            "by_source": {},
            "by_domain": {}
        }
    
    def ingest(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        domain: Optional[str] = None,
        importance: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Ingest new information into the system.
        
        Args:
            content: The content to ingest
            metadata: Optional additional metadata
            source: Optional source of the information
            domain: Optional domain classification
            importance: Optional importance score (0.0-1.0)
            timestamp: Optional custom timestamp
            
        Returns:
            Memory ID of the ingested content
        """
        # Apply preprocessors if any
        for preprocessor in self.preprocessors:
            content = preprocessor(content)
        
        # Prepare metadata
        metadata = metadata or {}
        if source:
            metadata["source"] = source
            metadata["trust_level"] = self.source_trust_levels.get(source, 0.5)
        
        if domain:
            metadata["domain"] = domain
            
        if importance is not None:
            metadata["importance"] = max(0.0, min(1.0, importance))
        
        if timestamp:
            metadata["timestamp"] = timestamp.isoformat()
        else:
            metadata["timestamp"] = datetime.now().isoformat()
            
        # Add entropy estimation
        # Simple implementation - in real system use information theory metrics
        entropy = min(1.0, len(content) / 1000)
        metadata["entropy"] = entropy
        
        # Add memory to store
        memory_id = self.memory_store.add_memory(content, metadata)
        
        # Update stats
        self.stats["inputs_processed"] += 1
        self.stats["total_entropy"] += entropy
        self.stats["avg_entropy"] = self.stats["total_entropy"] / self.stats["inputs_processed"]
        
        if source:
            if source not in self.stats["by_source"]:
                self.stats["by_source"][source] = 0
            self.stats["by_source"][source] += 1
            
        if domain:
            if domain not in self.stats["by_domain"]:
                self.stats["by_domain"][domain] = 0
            self.stats["by_domain"][domain] += 1
            
        logger.info(f"Ingested new memory with ID {memory_id}")
        return memory_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about processed inputs."""
        return self.stats.copy()


class CycleController:
    """
    Core orchestration controller for the ΨC system cycle.
    
    Manages the flow between components:
    1. Input ingestion
    2. Reflection processing
    3. Contradiction detection and resolution
    4. Memory updates
    5. Coherence calculation and logging
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        schema_graph: SchemaGraph,
        reflection_engine: ReflectionEngine,
        contradiction_detector: ContradictionDetector,
        coherence_calculator: CoherenceCalculator,
        input_processor: Optional[InputProcessor] = None,
        complexity_controller: Optional[ComplexityController] = None,
        cycle_frequency: Optional[float] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize the cycle controller.
        
        Args:
            memory_store: Memory storage system
            schema_graph: Schema graph representation
            reflection_engine: Reflection processor
            contradiction_detector: Contradiction detection system
            coherence_calculator: Coherence calculation system
            input_processor: Optional custom input processor
            complexity_controller: Optional complexity controller
            cycle_frequency: Optional fixed cycle interval in seconds
            log_dir: Directory to store cycle logs
        """
        self.memory_store = memory_store
        self.schema_graph = schema_graph
        self.reflection_engine = reflection_engine
        self.contradiction_detector = contradiction_detector
        self.coherence_calculator = coherence_calculator
        self.input_processor = input_processor or InputProcessor(memory_store)
        self.complexity_controller = complexity_controller
        self.cycle_frequency = cycle_frequency
        
        # Create log directory if specified
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Cycle tracking
        self.cycle_history = []
        self.current_cycle = None
        self.cycle_count = 0
        self.last_cycle_time = None
        
        # Coherence history
        self.coherence_history = []
        
        logger.info("CycleController initialized")
    
    def ingest(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        domain: Optional[str] = None,
        importance: Optional[float] = None
    ) -> str:
        """
        Ingest new information into the system.
        
        Args:
            content: The content to ingest
            metadata: Optional additional metadata
            source: Optional source of the information
            domain: Optional domain classification
            importance: Optional importance score (0.0-1.0)
            
        Returns:
            Memory ID of the ingested content
        """
        return self.input_processor.ingest(
            content=content,
            metadata=metadata,
            source=source,
            domain=domain,
            importance=importance
        )
    
    def run_cycle(self, memory_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete ΨC cycle.
        
        Args:
            memory_id: Optional specific memory to focus on
            
        Returns:
            Dictionary with cycle results
        """
        cycle_id = str(uuid.uuid4())
        cycle_start = time.time()
        
        logger.info(f"Starting cycle {self.cycle_count} (ID: {cycle_id})")
        
        # Create cycle record
        cycle_record = {
            "id": cycle_id,
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now().isoformat(),
            "focused_memory": memory_id,
            "stages": {},
            "metrics": {},
            "duration": None
        }
        
        # Check if complexity controller allows this cycle
        if self.complexity_controller:
            # Calculate current complexity
            complexity = self.complexity_controller.get_complexity()
            cycle_record["metrics"]["complexity"] = complexity
            
            # Check if we can run reflection
            if not self.complexity_controller.can_activate_feature(
                FeatureActivation.REFLECTION
            ):
                logger.info("Cycle aborted: complexity budget exceeded")
                cycle_record["aborted"] = True
                cycle_record["abort_reason"] = "complexity_exceeded"
                
                # Minimal cycle - just update coherence
                coherence = self.coherence_calculator.calculate_coherence()
                cycle_record["metrics"]["coherence"] = coherence
                self.coherence_history.append((time.time(), coherence))
                
                # Log the abbreviated cycle
                self._log_cycle(cycle_record)
                self.cycle_history.append(cycle_record)
                self.cycle_count += 1
                
                return cycle_record
        
        # 1. Reflection phase
        logger.info("Starting reflection phase")
        reflection_start = time.time()
        
        reflection_result = self.reflection_engine.reflect(memory_id=memory_id)
        reflection_duration = time.time() - reflection_start
        
        cycle_record["stages"]["reflection"] = {
            "duration": reflection_duration,
            "memory_count": len(reflection_result.get("activated_memories", [])),
            "activated_memories": reflection_result.get("activated_memories", []),
            "insights": reflection_result.get("insights", [])
        }
        
        # 2. Contradiction detection phase
        logger.info("Starting contradiction detection phase")
        contradiction_start = time.time()
        
        contradictions = self.contradiction_detector.detect_contradictions(
            memories=reflection_result.get("activated_memories", []),
            threshold=0.5  # Configurable threshold
        )
        contradiction_duration = time.time() - contradiction_start
        
        cycle_record["stages"]["contradiction"] = {
            "duration": contradiction_duration,
            "contradiction_count": len(contradictions),
            "contradictions": [
                {"memory1": c[0], "memory2": c[1], "score": c[2]}
                for c in contradictions
            ]
        }
        
        # 3. Schema update phase
        logger.info("Starting schema update phase")
        schema_start = time.time()
        
        # Only update schema if contradictions found
        if contradictions:
            self.schema_graph.update_from_contradictions(contradictions)
            
        schema_duration = time.time() - schema_start
        
        cycle_record["stages"]["schema_update"] = {
            "duration": schema_duration,
            "schema_version": self.schema_graph.get_version(),
            "node_count": len(self.schema_graph.get_nodes()),
            "edge_count": len(self.schema_graph.get_edges())
        }
        
        # 4. Coherence calculation phase
        logger.info("Starting coherence calculation phase")
        coherence_start = time.time()
        
        coherence = self.coherence_calculator.calculate_coherence()
        coherence_duration = time.time() - coherence_start
        
        cycle_record["stages"]["coherence"] = {
            "duration": coherence_duration,
            "coherence_score": coherence,
            "coherence_components": self.coherence_calculator.get_component_scores()
        }
        
        # 5. Memory reinforcement phase - implement reinforce_beliefs
        logger.info("Starting memory reinforcement phase")
        reinforcement_start = time.time()
        
        # Call the reinforce_beliefs function
        reinforcement_result = self.reinforce_beliefs(
            reflection_result=reflection_result,
            contradictions=contradictions
        )
        
        reinforcement_duration = time.time() - reinforcement_start
        
        cycle_record["stages"]["reinforcement"] = {
            "duration": reinforcement_duration,
            "memories_reinforced": reinforcement_result.get("memories_reinforced", 0),
            "weights_adjusted": reinforcement_result.get("weights_adjusted", 0)
        }
        
        # Record cycle metrics
        cycle_record["metrics"] = {
            "memory_count": self.memory_store.get_memory_count(),
            "schema_size": len(self.schema_graph.get_nodes()),
            "contradiction_count": len(contradictions),
            "coherence": coherence,
            "cycle_duration": time.time() - cycle_start
        }
        
        # Record identity metrics if available
        try:
            identity_metrics = self_entropy.measure_identity_metrics(
                self.schema_graph, self.memory_store
            )
            cycle_record["metrics"].update({
                "identity_entropy": identity_metrics.get("entropy", 0),
                "identity_stability": identity_metrics.get("stability", 0)
            })
        except Exception as e:
            logger.warning(f"Failed to calculate identity metrics: {e}")
        
        # Finalize cycle
        cycle_duration = time.time() - cycle_start
        cycle_record["duration"] = cycle_duration
        
        logger.info(f"Cycle {self.cycle_count} completed in {cycle_duration:.2f}s with coherence {coherence:.4f}")
        
        # Update coherence history
        self.coherence_history.append((time.time(), coherence))
        
        # Log cycle data
        self._log_cycle(cycle_record)
        
        # Store cycle in history
        self.cycle_history.append(cycle_record)
        self.last_cycle_time = time.time()
        self.cycle_count += 1
        
        return cycle_record
    
    def run_continuous(
        self,
        max_cycles: Optional[int] = None,
        duration: Optional[float] = None,
        coherence_target: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Run cycles continuously until a stopping condition is met.
        
        Args:
            max_cycles: Maximum number of cycles to run (None for no limit)
            duration: Maximum duration in seconds (None for no limit)
            coherence_target: Stop when coherence reaches this target
            
        Returns:
            List of cycle records
        """
        start_time = time.time()
        cycles_run = 0
        results = []
        
        while True:
            # Check stopping conditions
            if max_cycles is not None and cycles_run >= max_cycles:
                logger.info(f"Reached maximum cycles: {max_cycles}")
                break
                
            if duration is not None and (time.time() - start_time) >= duration:
                logger.info(f"Reached maximum duration: {duration}s")
                break
                
            if coherence_target is not None and self.coherence_history and self.coherence_history[-1][1] >= coherence_target:
                logger.info(f"Reached coherence target: {coherence_target}")
                break
            
            # Run a cycle
            cycle_result = self.run_cycle()
            results.append(cycle_result)
            cycles_run += 1
            
            # Sleep if cycle frequency is set
            if self.cycle_frequency:
                time.sleep(self.cycle_frequency)
        
        logger.info(f"Continuous run completed: {cycles_run} cycles in {time.time() - start_time:.2f}s")
        return results
    
    def reinforce_beliefs(
        self,
        reflection_result: Dict[str, Any],
        contradictions: List[Tuple[str, str, float]]
    ) -> Dict[str, Any]:
        """
        Implement the reinforce_beliefs function to update memory weights 
        based on reflection and contradiction resolution.
        
        Args:
            reflection_result: Result from the reflection engine
            contradictions: List of detected contradictions
            
        Returns:
            Dictionary with reinforcement statistics
        """
        reinforced_count = 0
        weights_adjusted = 0
        
        # 1. Process memories activated during reflection
        activated_memories = reflection_result.get("activated_memories", [])
        
        # Get memory objects
        memories = [
            self.memory_store.get_memory(memory_id) 
            for memory_id in activated_memories
            if self.memory_store.has_memory(memory_id)
        ]
        
        # 2. Apply Bayesian-inspired weight updates
        for memory in memories:
            # Skip memories involved in contradictions (handled separately)
            if any(memory.id in (c[0], c[1]) for c in contradictions):
                continue
                
            # Calculate evidence strength from reflection
            evidence_strength = reflection_result.get("memory_scores", {}).get(memory.id, 0.5)
            
            # Get current importance/confidence
            current_importance = getattr(memory, "importance", 0.5)
            
            # Bayesian update: new_belief ∝ prior * likelihood
            # Simplified implementation
            updated_importance = (current_importance + evidence_strength) / 2
            
            # Apply a small reinforcement boost for consistent memories
            reinforcement_factor = 1.02  # Small boost for confirmed memories
            updated_importance = min(1.0, updated_importance * reinforcement_factor)
            
            # Update the memory
            memory.importance = updated_importance
            
            # Add reinforcement metadata
            memory.metadata["reinforced"] = True
            memory.metadata["reinforcement_time"] = datetime.now().isoformat()
            memory.metadata["prior_importance"] = current_importance
            memory.metadata["evidence_strength"] = evidence_strength
            
            # Save the updated memory
            self.memory_store.update_memory(memory)
            
            reinforced_count += 1
            weights_adjusted += 1
        
        # 3. Handle contradicted memories
        contradiction_memory_ids = set()
        for m1_id, m2_id, strength in contradictions:
            contradiction_memory_ids.add(m1_id)
            contradiction_memory_ids.add(m2_id)
        
        for memory_id in contradiction_memory_ids:
            if not self.memory_store.has_memory(memory_id):
                continue
                
            memory = self.memory_store.get_memory(memory_id)
            
            # Contradicted memories get confidence reduction
            current_importance = getattr(memory, "importance", 0.5)
            
            # Calculate contradiction penalty
            contradiction_factor = 0.9  # Reduce importance for contradicted memories
            updated_importance = max(0.1, current_importance * contradiction_factor)
            
            # Update the memory
            memory.importance = updated_importance
            
            # Add contradiction metadata
            memory.metadata["contradicted"] = True
            memory.metadata["contradiction_time"] = datetime.now().isoformat()
            memory.metadata["prior_importance"] = current_importance
            
            # Save the updated memory
            self.memory_store.update_memory(memory)
            
            weights_adjusted += 1
        
        # Return statistics
        return {
            "memories_reinforced": reinforced_count,
            "weights_adjusted": weights_adjusted,
            "contradiction_memories": len(contradiction_memory_ids)
        }
    
    def _log_cycle(self, cycle_record: Dict[str, Any]) -> None:
        """
        Log cycle data to file if log_dir is configured.
        
        Args:
            cycle_record: Cycle data to log
        """
        if not self.log_dir:
            return
            
        # Create filename with timestamp and cycle ID
        filename = f"cycle_{cycle_record['cycle_number']}_{cycle_record['id']}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(cycle_record, f, indent=2)
                
            # Also update a latest.json for easy access
            latest_path = os.path.join(self.log_dir, "latest_cycle.json")
            with open(latest_path, 'w') as f:
                json.dump(cycle_record, f, indent=2)
                
            # Update coherence history file
            coherence_path = os.path.join(self.log_dir, "coherence_history.json")
            coherence_data = {
                "timestamps": [c[0] for c in self.coherence_history],
                "values": [c[1] for c in self.coherence_history]
            }
            with open(coherence_path, 'w') as f:
                json.dump(coherence_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log cycle data: {e}")
    
    def get_coherence_history(self) -> List[Tuple[float, float]]:
        """Get the coherence history as (timestamp, value) pairs."""
        return self.coherence_history.copy()
    
    def get_cycle_stats(self) -> Dict[str, Any]:
        """Get statistics about all cycles run."""
        if not self.cycle_history:
            return {
                "cycles_run": 0,
                "avg_duration": 0,
                "avg_coherence": 0,
                "coherence_trend": 0
            }
            
        # Calculate statistics
        durations = [c["duration"] for c in self.cycle_history if "duration" in c]
        coherence_values = [c["metrics"]["coherence"] for c in self.cycle_history if "metrics" in c and "coherence" in c["metrics"]]
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0
        
        # Calculate coherence trend (positive = improving)
        coherence_trend = 0
        if len(coherence_values) > 1:
            # Simple linear regression slope
            n = len(coherence_values)
            indices = list(range(n))
            x_mean = sum(indices) / n
            y_mean = sum(coherence_values) / n
            
            numerator = sum((indices[i] - x_mean) * (coherence_values[i] - y_mean) for i in range(n))
            denominator = sum((indices[i] - x_mean) ** 2 for i in range(n))
            
            coherence_trend = numerator / denominator if denominator != 0 else 0
        
        return {
            "cycles_run": len(self.cycle_history),
            "avg_duration": avg_duration,
            "avg_coherence": avg_coherence,
            "coherence_trend": coherence_trend,
            "latest_coherence": coherence_values[-1] if coherence_values else 0,
            "initial_coherence": coherence_values[0] if coherence_values else 0
        } 