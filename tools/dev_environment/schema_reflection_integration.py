#!/usr/bin/env python3
"""
Schema-Reflection Integration Module for ΨC-AI SDK

This module implements the integration between the schema system and 
the reflection engine, enabling cognitive processes to update and evolve
the schema structure based on mathematical principles.

The implementation directly aligns with the ΨC-AI SDK mathematical foundations,
particularly formulas #7 (Reflection Cycle Activation), #13 (Reflective Control Formula),
and #14 (Meta-Objective Function).
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)

# Try to import required components
try:
    from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration
    from tools.dev_environment.schema_math import (
        reflective_control_score,
        meta_objective_function,
        cognitive_debt
    )
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    logger.warning("Required dependencies not available. Schema-Reflection integration disabled.")

class SchemaReflectionIntegrator:
    """
    Integration between schema system and reflection engine.
    
    This class serves as the bridge between the schema graph system and 
    the cognitive reflection engine, implementing core mathematical principles
    of the ΨC-AI SDK.
    
    Attributes:
        schema_integration: The MemorySchemaIntegration instance
        reflection_engine: The ReflectionEngine instance (if available)
        reflection_threshold: Threshold for triggering reflection
        reflection_log: Log of reflection events
    """
    
    def __init__(
        self, 
        schema_integration,
        reflection_engine=None,
        reflection_threshold=0.7,
        max_reflections_per_hour=5
    ):
        """
        Initialize the schema-reflection integrator.
        
        Args:
            schema_integration: MemorySchemaIntegration instance
            reflection_engine: ReflectionEngine instance (optional)
            reflection_threshold: Threshold for triggering reflection
            max_reflections_per_hour: Maximum reflections allowed per hour
        """
        self.schema_integration = schema_integration
        self.reflection_engine = reflection_engine
        self.reflection_threshold = reflection_threshold
        self.max_reflections_per_hour = max_reflections_per_hour
        self.reflection_log = []
        self.last_reflection_time = None
        self.reflection_count_last_hour = 0
        
    def check_reflection_needed(self) -> Tuple[bool, str, List[str]]:
        """
        Check if reflection is needed based on formula #7:
        
        Reflect = {
            1 if C̄ < Ψ_threshold
            0 otherwise
        }
        
        Returns:
            Tuple of (reflection_needed, trigger_type, memory_ids)
        """
        if not HAS_DEPENDENCIES:
            return False, "dependencies_missing", []
            
        # Get schema graph
        graph = self.schema_integration.graph
        if not graph or graph.number_of_nodes() == 0:
            return False, "empty_graph", []
            
        # Check if we've exceeded reflection rate limit
        current_time = datetime.datetime.now()
        if self.last_reflection_time:
            time_diff = current_time - self.last_reflection_time
            if time_diff.total_seconds() < 3600:  # Within the last hour
                if self.reflection_count_last_hour >= self.max_reflections_per_hour:
                    return False, "rate_limited", []
            else:
                # Reset counter if more than an hour has passed
                self.reflection_count_last_hour = 0
            
        # Calculate mean coherence across memory nodes
        memory_nodes = [n for n, attrs in graph.nodes(data=True) 
                       if attrs.get('node_type') == 'memory']
        
        if not memory_nodes:
            return False, "no_memories", []
            
        # Calculate average coherence
        total_coherence = 0.0
        edge_count = 0
        for node in memory_nodes:
            for _, _, data in graph.edges(node, data=True):
                total_coherence += data.get('weight', 0)
                edge_count += 1
                
        mean_coherence = total_coherence / max(1, edge_count)
        
        # Check if mean coherence is below threshold
        if mean_coherence < self.reflection_threshold:
            # Find memories with lowest coherence
            memory_coherence = {}
            for node in memory_nodes:
                node_edges = list(graph.edges(node, data=True))
                if node_edges:
                    node_coherence = sum(data.get('weight', 0) for _, _, data in node_edges) / len(node_edges)
                    memory_coherence[node] = node_coherence
                
            # Sort memories by coherence and get bottom 3
            sorted_memories = sorted(memory_coherence.items(), key=lambda x: x[1])
            target_memories = [m[0] for m in sorted_memories[:3]]
            
            return True, "low_coherence", target_memories
            
        # Check for contradictions (edges with negative weights or marked as contradictions)
        contradictions = []
        for source, target, data in graph.edges(data=True):
            if (data.get('edge_type') == 'contradiction' or 
                data.get('weight', 0) < 0):
                contradictions.append((source, target))
                
        if contradictions:
            # Get unique memory IDs involved in contradictions
            contradiction_memories = set()
            for source, target in contradictions:
                if graph.nodes[source].get('node_type') == 'memory':
                    contradiction_memories.add(source)
                if graph.nodes[target].get('node_type') == 'memory':
                    contradiction_memories.add(target)
                    
            return True, "contradiction", list(contradiction_memories)
            
        # Check for high cognitive debt
        try:
            debt = self.schema_integration.calculate_cognitive_debt()
            if debt > 3.0:  # Threshold for high cognitive debt
                # Select most important memories
                important_memories = sorted(
                    [(n, graph.nodes[n].get('importance', 0)) 
                     for n in memory_nodes if graph.nodes[n].get('importance', 0) > 0.7],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                target_memories = [m[0] for m in important_memories[:3]]
                return True, "cognitive_debt", target_memories
        except:
            pass
            
        # No reflection needed
        return False, "none", []
    
    def trigger_reflection(self, trigger_type="manual", memory_ids=None):
        """
        Trigger a reflection cycle and update the schema accordingly.
        
        Args:
            trigger_type: Type of reflection trigger
            memory_ids: List of memory IDs to focus reflection on
            
        Returns:
            dict: Results of reflection process
        """
        if not HAS_DEPENDENCIES:
            return {"status": "error", "message": "Dependencies missing"}
            
        # Log reflection event
        current_time = datetime.datetime.now()
        reflection_event = {
            "timestamp": current_time.isoformat(),
            "trigger_type": trigger_type,
            "memory_ids": memory_ids or []
        }
        
        # Update reflection rate tracking
        if self.last_reflection_time:
            time_diff = current_time - self.last_reflection_time
            if time_diff.total_seconds() < 3600:  # Within the last hour
                self.reflection_count_last_hour += 1
            else:
                self.reflection_count_last_hour = 1
        else:
            self.reflection_count_last_hour = 1
            
        self.last_reflection_time = current_time
        
        # If reflection engine is available, use it
        reflection_result = {}
        if self.reflection_engine:
            try:
                # Call reflection engine with memory IDs
                reflection_result = self.reflection_engine.reflect(memory_ids)
            except Exception as e:
                logger.error(f"Error during reflection: {e}")
                reflection_result = {"status": "error", "message": str(e)}
        else:
            # Simulated reflection without engine
            reflection_result = self._simulate_reflection(memory_ids)
            
        # Update schema based on reflection result
        schema_changes = self.schema_integration.process_reflection(
            trigger_type, memory_ids, reflection_result
        )
        
        # Complete reflection event log
        reflection_event["result"] = reflection_result
        reflection_event["schema_changes"] = schema_changes
        self.reflection_log.append(reflection_event)
        
        # Keep log size reasonable
        max_log_size = 100
        if len(self.reflection_log) > max_log_size:
            self.reflection_log = self.reflection_log[-max_log_size:]
            
        return {
            "status": "success",
            "trigger_type": trigger_type,
            "reflection_result": reflection_result,
            "schema_changes": schema_changes
        }
    
    def _simulate_reflection(self, memory_ids):
        """
        Simulate a reflection process when no reflection engine is available.
        
        Args:
            memory_ids: List of memory IDs to focus reflection on
            
        Returns:
            dict: Simulated reflection results
        """
        # Get memory details
        graph = self.schema_integration.graph
        memories = {}
        for memory_id in memory_ids:
            if memory_id in graph.nodes():
                node = graph.nodes[memory_id]
                memories[memory_id] = {
                    "content": node.get("content", ""),
                    "type": node.get("memory_type", "unknown"),
                    "importance": node.get("importance", 0.5)
                }
                
        # Generate simulated concepts based on memory content
        concepts = {}
        if memories:
            # Simple concept generation based on memory content
            all_content = " ".join([m.get("content", "") for m in memories.values()])
            words = all_content.lower().split()
            word_count = {}
            
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_count[word] = word_count.get(word, 0) + 1
                    
            # Find potential concept words
            concept_words = sorted(
                [(word, count) for word, count in word_count.items() if count > 1],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Generate concepts
            for i, (word, count) in enumerate(concept_words):
                concept_id = f"concept_{i}"
                concepts[concept_id] = {
                    "concept_name": word.capitalize(),
                    "keywords": [word],
                    "memory_ids": memory_ids,
                    "importance": min(0.5 + (count / 10), 0.9),
                    "dominant_type": max(
                        [(m.get("type", "unknown"), m.get("importance", 0)) 
                         for m in memories.values()],
                        key=lambda x: x[1]
                    )[0]
                }
                
        return {
            "memories_processed": list(memories.keys()),
            "concepts": concepts,
            "simulated": True
        }
    
    def calculate_schema_concept_alignment(self):
        """
        Calculate alignment between schema concepts and system goals.
        
        Implements formula #34: Relevance(S_i) = cos(vec(S_i), vec(G))
        
        Returns:
            dict: Schema concept alignment scores
        """
        # Get concept nodes
        graph = self.schema_integration.graph
        concept_nodes = [n for n, attrs in graph.nodes(data=True) 
                        if attrs.get('node_type') == 'concept']
        
        # If no concepts, return empty result
        if not concept_nodes:
            return {}
            
        # Calculate alignment score for each concept
        alignment_scores = {}
        for concept_node in concept_nodes:
            # Get concept details
            concept_name = graph.nodes[concept_node].get('name', concept_node)
            
            # Get connected memories
            connected_memories = []
            for target in graph.neighbors(concept_node):
                if graph.nodes[target].get('node_type') == 'memory':
                    connected_memories.append(target)
                    
            # Calculate importance based on connected memories
            memory_importance = [
                graph.nodes[m].get('importance', 0.5) 
                for m in connected_memories
            ]
            avg_importance = sum(memory_importance) / len(memory_importance) if memory_importance else 0.5
            
            # Calculate alignment score
            # In a real system, this would compare to system goals
            # Here we use a simple heuristic based on importance
            alignment = avg_importance
            
            alignment_scores[concept_name] = {
                "alignment_score": alignment,
                "connected_memories": len(connected_memories),
                "importance": avg_importance
            }
            
        return alignment_scores
    
    def detect_schema_drift(self, time_window=24):
        """
        Detect schema drift over a time window.
        
        Implements formula #5: ΔC = (1/N) · ∑_{i=1}^N (C_i^(t) - C_i^(t-1))
        
        Args:
            time_window: Time window in hours
            
        Returns:
            dict: Schema drift metrics
        """
        # Need at least 2 schema snapshots to calculate drift
        if len(self.schema_integration.schema_history) < 2:
            return {"status": "insufficient_data", "drift": 0}
            
        # Get current time
        current_time = datetime.datetime.now()
        
        # Find schema snapshot within time window
        comparison_snapshot = None
        comparison_idx = -1
        
        for i, timestamp in enumerate(self.schema_integration.schema_timestamps):
            try:
                snapshot_time = datetime.datetime.fromisoformat(timestamp)
                time_diff = current_time - snapshot_time
                
                if time_diff.total_seconds() / 3600 <= time_window:
                    comparison_snapshot = self.schema_integration.schema_history[i]
                    comparison_idx = i
                    break
            except:
                pass
                
        if comparison_snapshot is None:
            return {"status": "no_snapshot_in_window", "drift": 0}
            
        # Get current schema
        current_schema = self.schema_integration.graph
        
        # Calculate node changes
        current_nodes = set(current_schema.nodes())
        comparison_nodes = set(comparison_snapshot.nodes())
        
        added_nodes = current_nodes - comparison_nodes
        removed_nodes = comparison_nodes - current_nodes
        common_nodes = current_nodes.intersection(comparison_nodes)
        
        # Calculate edge changes
        current_edges = set(current_schema.edges())
        comparison_edges = set(comparison_snapshot.edges())
        
        added_edges = current_edges - comparison_edges
        removed_edges = comparison_edges - current_edges
        
        # Calculate coherence change for common nodes
        coherence_changes = []
        for node in common_nodes:
            current_edges = list(current_schema.edges(node, data=True))
            comparison_edges = list(comparison_snapshot.edges(node, data=True))
            
            if current_edges and comparison_edges:
                current_coherence = sum(data.get('weight', 0) for _, _, data in current_edges) / len(current_edges)
                comparison_coherence = sum(data.get('weight', 0) for _, _, data in comparison_edges) / len(comparison_edges)
                
                coherence_changes.append(current_coherence - comparison_coherence)
                
        # Calculate average coherence change (drift)
        avg_coherence_change = sum(coherence_changes) / len(coherence_changes) if coherence_changes else 0
        
        return {
            "status": "success",
            "drift": avg_coherence_change,
            "node_changes": {
                "added": len(added_nodes),
                "removed": len(removed_nodes),
                "common": len(common_nodes)
            },
            "edge_changes": {
                "added": len(added_edges),
                "removed": len(removed_edges)
            },
            "time_window_hours": time_window,
            "comparison_snapshot_age_hours": 
                (current_time - datetime.datetime.fromisoformat(
                    self.schema_integration.schema_timestamps[comparison_idx]
                )).total_seconds() / 3600
        }
    
    def get_reflection_health_metrics(self):
        """
        Calculate reflection health metrics.
        
        Returns:
            dict: Reflection health metrics
        """
        # Get metrics from schema integration
        cognitive_debt = self.schema_integration.calculate_cognitive_debt()
        schema_health = self.schema_integration.calculate_schema_health()
        
        # Calculate reflection rate
        current_time = datetime.datetime.now()
        reflections_last_24h = 0
        
        for event in self.reflection_log:
            try:
                event_time = datetime.datetime.fromisoformat(event["timestamp"])
                time_diff = current_time - event_time
                
                if time_diff.total_seconds() / 3600 <= 24:
                    reflections_last_24h += 1
            except:
                pass
                
        # Calculate reflection effectiveness
        effective_reflections = 0
        for event in self.reflection_log:
            schema_changes = event.get("schema_changes", {})
            total_changes = (
                schema_changes.get("nodes_added", 0) +
                schema_changes.get("nodes_modified", 0) +
                schema_changes.get("edges_added", 0) +
                schema_changes.get("edges_modified", 0)
            )
            
            if total_changes > 0:
                effective_reflections += 1
                
        effectiveness_rate = effective_reflections / max(1, len(self.reflection_log))
        
        return {
            "cognitive_debt": cognitive_debt,
            "schema_health": schema_health,
            "reflections_last_24h": reflections_last_24h,
            "reflection_effectiveness": effectiveness_rate,
            "total_reflections": len(self.reflection_log)
        }
    
    def create_reflection_report(self, output_path=None):
        """
        Create a comprehensive reflection report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            dict: Reflection report
        """
        # Generate report data
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": self.get_reflection_health_metrics(),
            "schema_stats": self.schema_integration.calculate_schema_statistics(),
            "schema_drift": self.detect_schema_drift(),
            "concept_alignment": self.calculate_schema_concept_alignment(),
            "energy_usage": self.schema_integration.calculate_memory_energy_usage(),
            "reflection_history": self.reflection_log[-10:]  # Include last 10 reflection events
        }
        
        # Save report to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report 