"""
Identity Recognition Module

This module implements identity recognition based on schema fingerprinting.
It provides tools for:
1. Creating and comparing schema fingerprints
2. Detecting identity changes over time
3. Monitoring system self-consistency and coherence
4. Logging identity-related events

The identity recognition system helps maintain a stable sense of self
while allowing for natural evolution of the schema over time.
"""

import logging
import time
import hashlib
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..schema.schema import SchemaGraph
from ..schema.annealing.annealing import ConvergenceTracker
from ..memory.memory import MemoryStore
from ..coherence.coherence import CoherenceScorer

logger = logging.getLogger(__name__)


class IdentityChangeType(Enum):
    """Types of identity changes that can be detected."""
    
    GRADUAL_EVOLUTION = "gradual_evolution"       # Slow, natural changes over time
    STRUCTURAL_SHIFT = "structural_shift"         # Major reorganization of schema
    BELIEF_REVISION = "belief_revision"           # Changes to core beliefs
    GOAL_ALTERATION = "goal_alteration"           # Changes to goals or objectives
    COHERENCE_LOSS = "coherence_loss"             # Loss of internal consistency
    EXTERNAL_INFLUENCE = "external_influence"     # Changes due to external factors
    RECOVERY = "recovery"                         # Return to previous identity state


@dataclass
class IdentityFingerprint:
    """Represents a fingerprint of the system's identity state."""
    
    fingerprint_id: str
    timestamp: datetime
    hash_value: str
    node_count: int
    edge_count: int
    core_nodes: List[str]
    belief_hashes: Dict[str, str]
    coherence_score: float
    stability_score: float
    structural_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    
    @classmethod
    def from_schema(cls, 
                   schema_graph: SchemaGraph,
                   memory_store: MemoryStore,
                   coherence_scorer: CoherenceScorer) -> 'IdentityFingerprint':
        """
        Create an identity fingerprint from the current schema.
        
        Args:
            schema_graph: Current schema graph
            memory_store: Memory store
            coherence_scorer: Coherence scorer
            
        Returns:
            New identity fingerprint
        """
        # Generate unique ID
        fingerprint_id = f"fp_{int(time.time())}_{hash(schema_graph)}"
        
        # Get core nodes (highest importance)
        node_importances = {}
        for node_id, data in schema_graph.graph.nodes(data=True):
            node_importances[node_id] = data.get("importance", 0.0)
        
        # Sort by importance and take top 10%
        sorted_nodes = sorted(node_importances.items(), key=lambda x: x[1], reverse=True)
        num_core = max(3, int(len(sorted_nodes) * 0.1))
        core_nodes = [node_id for node_id, _ in sorted_nodes[:num_core]]
        
        # Create belief hashes
        belief_hashes = {}
        for node_id, data in schema_graph.graph.nodes(data=True):
            if data.get("node_type") == "belief":
                node_str = json.dumps({k: v for k, v in data.items() if k != "embedding"}, sort_keys=True)
                belief_hashes[node_id] = hashlib.sha256(node_str.encode()).hexdigest()[:16]
        
        # Calculate coherence score
        coherence_score = 0.0
        if memory_store.get_all_memories():
            memories = memory_store.get_all_memories()
            coherence_score = coherence_scorer.calculate_global_coherence(memories)
        
        # Extract structural metrics
        structural_metrics = cls._extract_structural_metrics(schema_graph)
        
        # Calculate stability score
        stability_score = cls._calculate_stability_score(structural_metrics)
        
        # Create overall hash
        graph_data = {
            "nodes": len(schema_graph.graph.nodes),
            "edges": len(schema_graph.graph.edges),
            "core_nodes": core_nodes,
            "belief_hashes": belief_hashes,
            "structural_metrics": structural_metrics
        }
        hash_value = hashlib.sha256(json.dumps(graph_data, sort_keys=True).encode()).hexdigest()
        
        return cls(
            fingerprint_id=fingerprint_id,
            timestamp=datetime.now(),
            hash_value=hash_value,
            node_count=len(schema_graph.graph.nodes),
            edge_count=len(schema_graph.graph.edges),
            core_nodes=core_nodes,
            belief_hashes=belief_hashes,
            coherence_score=coherence_score,
            stability_score=stability_score,
            structural_metrics=structural_metrics,
            metadata={}
        )
    
    @staticmethod
    def _extract_structural_metrics(schema_graph: SchemaGraph) -> Dict[str, Any]:
        """Extract structural metrics from the schema graph."""
        graph = schema_graph.graph
        metrics = {}
        
        if len(graph.nodes) == 0:
            return {"empty_graph": True}
        
        # Basic graph metrics
        metrics["density"] = nx.density(graph)
        
        try:
            metrics["clustering_coefficient"] = nx.average_clustering(graph)
        except:
            metrics["clustering_coefficient"] = 0.0
        
        # Node type distribution
        node_types = {}
        for _, data in graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        metrics["node_type_distribution"] = node_types
        
        # Degree statistics
        degrees = [d for _, d in graph.degree()]
        metrics["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0
        metrics["max_degree"] = max(degrees) if degrees else 0
        
        # Connectivity
        try:
            components = list(nx.connected_components(graph.to_undirected()))
            metrics["num_components"] = len(components)
            metrics["largest_component_size"] = len(max(components, key=len))
        except:
            metrics["num_components"] = 1
            metrics["largest_component_size"] = len(graph.nodes)
        
        return metrics
    
    @staticmethod
    def _calculate_stability_score(metrics: Dict[str, Any]) -> float:
        """Calculate a stability score from structural metrics."""
        stability = 0.5  # Default mid-point
        
        # More connected graphs are more stable
        if "density" in metrics:
            stability += 0.1 * metrics["density"]
        
        # Higher clustering coefficient indicates more stability
        if "clustering_coefficient" in metrics:
            stability += 0.1 * metrics["clustering_coefficient"]
        
        # Single connected component is more stable
        if "num_components" in metrics and metrics["num_components"] > 0:
            component_ratio = metrics.get("largest_component_size", 0) / sum(
                metrics.get("node_type_distribution", {}).values())
            stability += 0.2 * component_ratio
        
        # Clip to valid range
        return max(0.0, min(1.0, stability))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "timestamp": self.timestamp.isoformat(),
            "hash_value": self.hash_value,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "core_nodes": self.core_nodes,
            "belief_hashes": self.belief_hashes,
            "coherence_score": self.coherence_score,
            "stability_score": self.stability_score,
            "structural_metrics": self.structural_metrics,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentityFingerprint':
        """Create from dictionary representation."""
        return cls(
            fingerprint_id=data["fingerprint_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            hash_value=data["hash_value"],
            node_count=data["node_count"],
            edge_count=data["edge_count"],
            core_nodes=data["core_nodes"],
            belief_hashes=data["belief_hashes"],
            coherence_score=data["coherence_score"],
            stability_score=data["stability_score"],
            structural_metrics=data["structural_metrics"],
            metadata=data.get("metadata", {})
        )


@dataclass
class IdentityChange:
    """Records a significant change in the system's identity."""
    
    change_id: str
    timestamp: datetime
    change_type: IdentityChangeType
    before_fingerprint_id: str
    after_fingerprint_id: str
    description: str
    magnitude: float
    affected_beliefs: List[str]
    affected_components: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "change_type": self.change_type.value,
            "before_fingerprint_id": self.before_fingerprint_id,
            "after_fingerprint_id": self.after_fingerprint_id,
            "description": self.description,
            "magnitude": self.magnitude,
            "affected_beliefs": self.affected_beliefs,
            "affected_components": self.affected_components,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentityChange':
        """Create from dictionary representation."""
        return cls(
            change_id=data["change_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            change_type=IdentityChangeType(data["change_type"]),
            before_fingerprint_id=data["before_fingerprint_id"],
            after_fingerprint_id=data["after_fingerprint_id"],
            description=data["description"],
            magnitude=data["magnitude"],
            affected_beliefs=data["affected_beliefs"],
            affected_components=data["affected_components"],
            metadata=data.get("metadata", {})
        )


class IdentityRecognitionSystem:
    """
    System for tracking and monitoring identity over time.
    
    This system maintains a history of identity fingerprints,
    detects changes, and logs significant identity events.
    """
    
    def __init__(self, 
                 schema_graph: SchemaGraph,
                 memory_store: MemoryStore,
                 coherence_scorer: CoherenceScorer,
                 convergence_tracker: Optional[ConvergenceTracker] = None,
                 fingerprint_history_size: int = 100,
                 change_detection_threshold: float = 0.15,
                 identity_check_interval: timedelta = timedelta(minutes=10)):
        """
        Initialize the identity recognition system.
        
        Args:
            schema_graph: The schema graph to monitor
            memory_store: Memory store to analyze
            coherence_scorer: Coherence scorer for evaluating memories
            convergence_tracker: Optional schema convergence tracker
            fingerprint_history_size: Number of fingerprints to retain
            change_detection_threshold: Threshold for significant identity changes
            identity_check_interval: How often to check for identity changes
        """
        self.schema_graph = schema_graph
        self.memory_store = memory_store
        self.coherence_scorer = coherence_scorer
        self.convergence_tracker = convergence_tracker
        self.fingerprint_history_size = fingerprint_history_size
        self.change_detection_threshold = change_detection_threshold
        self.identity_check_interval = identity_check_interval
        
        # Initialize fingerprint history
        self.fingerprint_history: List[IdentityFingerprint] = []
        self.fingerprints_by_id: Dict[str, IdentityFingerprint] = {}
        
        # Change history
        self.identity_changes: List[IdentityChange] = []
        
        # Tracking
        self.last_check_time = datetime.min
        self.change_counter = 0
        
        # Create initial fingerprint
        self._create_current_fingerprint()
    
    def _create_current_fingerprint(self) -> IdentityFingerprint:
        """
        Create a fingerprint of the current identity state.
        
        Returns:
            New identity fingerprint
        """
        fingerprint = IdentityFingerprint.from_schema(
            schema_graph=self.schema_graph,
            memory_store=self.memory_store,
            coherence_scorer=self.coherence_scorer
        )
        
        # Add to history
        self.fingerprint_history.append(fingerprint)
        self.fingerprints_by_id[fingerprint.fingerprint_id] = fingerprint
        
        # Maintain history size
        if len(self.fingerprint_history) > self.fingerprint_history_size:
            old_fingerprint = self.fingerprint_history.pop(0)
            self.fingerprints_by_id.pop(old_fingerprint.fingerprint_id, None)
        
        logger.info(f"Created identity fingerprint: {fingerprint.fingerprint_id} with {fingerprint.node_count} nodes")
        return fingerprint
    
    def check_identity(self, force: bool = False) -> Optional[IdentityChange]:
        """
        Check for identity changes.
        
        Args:
            force: Force check regardless of interval
            
        Returns:
            IdentityChange if significant change detected, None otherwise
        """
        current_time = datetime.now()
        
        # Check if enough time has passed since last check
        if not force and (current_time - self.last_check_time) < self.identity_check_interval:
            return None
        
        # Update last check time
        self.last_check_time = current_time
        
        # Need at least two fingerprints to compare
        if len(self.fingerprint_history) < 1:
            # Create first fingerprint if none exists
            self._create_current_fingerprint()
            return None
        
        # Get previous fingerprint
        previous_fingerprint = self.fingerprint_history[-1]
        
        # Create current fingerprint
        current_fingerprint = self._create_current_fingerprint()
        
        # Compare fingerprints
        change = self._compare_fingerprints(previous_fingerprint, current_fingerprint)
        
        if change:
            # Record the change
            self.identity_changes.append(change)
            logger.warning(f"Identity change detected: {change.change_type.value}, magnitude: {change.magnitude:.2f}")
            return change
        
        return None
    
    def _compare_fingerprints(self, 
                              fp1: IdentityFingerprint, 
                              fp2: IdentityFingerprint) -> Optional[IdentityChange]:
        """
        Compare two fingerprints to detect identity changes.
        
        Args:
            fp1: Previous fingerprint
            fp2: Current fingerprint
            
        Returns:
            IdentityChange if significant change detected, None otherwise
        """
        # Simple case: exact same hash
        if fp1.hash_value == fp2.hash_value:
            return None
        
        # Calculate overall change magnitude
        magnitude = self._calculate_change_magnitude(fp1, fp2)
        
        # Not significant enough
        if magnitude < self.change_detection_threshold:
            return None
        
        # Determine change type
        change_type, description = self._determine_change_type(fp1, fp2, magnitude)
        
        # Find affected beliefs
        affected_beliefs = []
        for belief_id in set(fp1.belief_hashes.keys()) | set(fp2.belief_hashes.keys()):
            hash1 = fp1.belief_hashes.get(belief_id)
            hash2 = fp2.belief_hashes.get(belief_id)
            if hash1 != hash2:
                affected_beliefs.append(belief_id)
        
        # Check affected components
        affected_components = self._identify_affected_components(fp1, fp2)
        
        # Create change record
        self.change_counter += 1
        change_id = f"change_{int(time.time())}_{self.change_counter}"
        
        return IdentityChange(
            change_id=change_id,
            timestamp=datetime.now(),
            change_type=change_type,
            before_fingerprint_id=fp1.fingerprint_id,
            after_fingerprint_id=fp2.fingerprint_id,
            description=description,
            magnitude=magnitude,
            affected_beliefs=affected_beliefs,
            affected_components=affected_components,
            metadata={}
        )
    
    def _calculate_change_magnitude(self, fp1: IdentityFingerprint, fp2: IdentityFingerprint) -> float:
        """
        Calculate the magnitude of change between fingerprints.
        
        Args:
            fp1: Previous fingerprint
            fp2: Current fingerprint
            
        Returns:
            Change magnitude (0-1)
        """
        factors = []
        
        # 1. Core node changes
        core_overlap = len(set(fp1.core_nodes) & set(fp2.core_nodes))
        core_union = len(set(fp1.core_nodes) | set(fp2.core_nodes))
        core_change = 1.0 - (core_overlap / core_union if core_union > 0 else 0.0)
        factors.append((core_change, 0.35))  # Weight: 35%
        
        # 2. Belief hash changes
        all_beliefs = set(fp1.belief_hashes.keys()) | set(fp2.belief_hashes.keys())
        changed_beliefs = 0
        for belief_id in all_beliefs:
            hash1 = fp1.belief_hashes.get(belief_id)
            hash2 = fp2.belief_hashes.get(belief_id)
            if hash1 != hash2:
                changed_beliefs += 1
        
        belief_change = changed_beliefs / max(1, len(all_beliefs))
        factors.append((belief_change, 0.3))  # Weight: 30%
        
        # 3. Structural changes
        node_change = abs(fp2.node_count - fp1.node_count) / max(1, max(fp1.node_count, fp2.node_count))
        edge_change = abs(fp2.edge_count - fp1.edge_count) / max(1, max(fp1.edge_count, fp2.edge_count))
        structural_change = (node_change + edge_change) / 2
        factors.append((structural_change, 0.2))  # Weight: 20%
        
        # 4. Coherence and stability changes
        coherence_change = abs(fp2.coherence_score - fp1.coherence_score)
        stability_change = abs(fp2.stability_score - fp1.stability_score)
        consistency_change = (coherence_change + stability_change) / 2
        factors.append((consistency_change, 0.15))  # Weight: 15%
        
        # Combine factors with weights
        magnitude = sum(value * weight for value, weight in factors)
        
        return min(1.0, magnitude)
    
    def _determine_change_type(self, 
                               fp1: IdentityFingerprint, 
                               fp2: IdentityFingerprint, 
                               magnitude: float) -> Tuple[IdentityChangeType, str]:
        """
        Determine the type of identity change.
        
        Args:
            fp1: Previous fingerprint
            fp2: Current fingerprint
            magnitude: Change magnitude
            
        Returns:
            Tuple of (change type, description)
        """
        # Check for coherence loss
        if fp2.coherence_score < fp1.coherence_score * 0.8:
            return (IdentityChangeType.COHERENCE_LOSS, 
                    f"Significant coherence drop: {fp1.coherence_score:.2f} to {fp2.coherence_score:.2f}")
        
        # Check for recovery (return to previous state)
        if magnitude < 0.3 and fp2.stability_score > fp1.stability_score * 1.2:
            return (IdentityChangeType.RECOVERY,
                    f"Recovery detected with increased stability: {fp1.stability_score:.2f} to {fp2.stability_score:.2f}")
        
        # Check for major structural shift
        node_change = abs(fp2.node_count - fp1.node_count) / max(1, fp1.node_count)
        edge_change = abs(fp2.edge_count - fp1.edge_count) / max(1, fp1.edge_count)
        if node_change > 0.2 or edge_change > 0.3:
            return (IdentityChangeType.STRUCTURAL_SHIFT,
                    f"Major structural reorganization: {fp1.node_count} to {fp2.node_count} nodes, " +
                    f"{fp1.edge_count} to {fp2.edge_count} edges")
        
        # Check for belief revision
        belief_changes = []
        for belief_id in set(fp1.belief_hashes.keys()) | set(fp2.belief_hashes.keys()):
            hash1 = fp1.belief_hashes.get(belief_id)
            hash2 = fp2.belief_hashes.get(belief_id)
            if hash1 != hash2:
                belief_changes.append(belief_id)
        
        if len(belief_changes) > 0 and set(belief_changes) & set(fp1.core_nodes):
            return (IdentityChangeType.BELIEF_REVISION,
                    f"Core belief revision affecting {len(belief_changes)} beliefs")
        
        # Default to gradual evolution for smaller changes
        return (IdentityChangeType.GRADUAL_EVOLUTION,
                f"Gradual identity evolution with magnitude {magnitude:.2f}")
    
    def _identify_affected_components(self, 
                                     fp1: IdentityFingerprint, 
                                     fp2: IdentityFingerprint) -> List[str]:
        """
        Identify schema components affected by the change.
        
        Args:
            fp1: Previous fingerprint
            fp2: Current fingerprint
            
        Returns:
            List of affected component names
        """
        components = []
        
        # Check node type distribution
        type_dist1 = fp1.structural_metrics.get("node_type_distribution", {})
        type_dist2 = fp2.structural_metrics.get("node_type_distribution", {})
        
        for node_type in set(type_dist1.keys()) | set(type_dist2.keys()):
            count1 = type_dist1.get(node_type, 0)
            count2 = type_dist2.get(node_type, 0)
            if abs(count2 - count1) / max(1, max(count1, count2)) > 0.1:
                components.append(f"node_type:{node_type}")
        
        # Check connectivity
        if (fp1.structural_metrics.get("num_components", 1) != 
            fp2.structural_metrics.get("num_components", 1)):
            components.append("connectivity")
        
        # Check density and clustering
        if (abs(fp2.structural_metrics.get("density", 0) - 
                fp1.structural_metrics.get("density", 0)) > 0.1):
            components.append("density")
        
        if (abs(fp2.structural_metrics.get("clustering_coefficient", 0) - 
                fp1.structural_metrics.get("clustering_coefficient", 0)) > 0.1):
            components.append("clustering")
        
        return components
    
    def get_identity_fingerprint(self, fingerprint_id: Optional[str] = None) -> Optional[IdentityFingerprint]:
        """
        Get a specific identity fingerprint.
        
        Args:
            fingerprint_id: ID of fingerprint to retrieve (None for most recent)
            
        Returns:
            IdentityFingerprint if found, None otherwise
        """
        if fingerprint_id is None:
            return self.fingerprint_history[-1] if self.fingerprint_history else None
        
        return self.fingerprints_by_id.get(fingerprint_id)
    
    def get_identity_changes(self, 
                            since: Optional[datetime] = None,
                            change_type: Optional[IdentityChangeType] = None) -> List[IdentityChange]:
        """
        Get identity changes, optionally filtered.
        
        Args:
            since: Only return changes since this time
            change_type: Only return changes of this type
            
        Returns:
            List of matching identity changes
        """
        filtered_changes = self.identity_changes
        
        if since is not None:
            filtered_changes = [c for c in filtered_changes if c.timestamp >= since]
        
        if change_type is not None:
            filtered_changes = [c for c in filtered_changes if c.change_type == change_type]
        
        return filtered_changes
    
    def get_identity_stability(self, timeframe: timedelta = timedelta(days=1)) -> float:
        """
        Calculate the stability of the identity over a timeframe.
        
        Args:
            timeframe: Time period to analyze
            
        Returns:
            Stability score (0-1, higher is more stable)
        """
        # Filter recent fingerprints
        cutoff_time = datetime.now() - timeframe
        recent_fingerprints = [fp for fp in self.fingerprint_history if fp.timestamp >= cutoff_time]
        
        if len(recent_fingerprints) < 2:
            # Not enough data points
            return 1.0 if recent_fingerprints else 0.5
        
        # Calculate average change magnitude between consecutive fingerprints
        changes = []
        for i in range(1, len(recent_fingerprints)):
            fp1 = recent_fingerprints[i-1]
            fp2 = recent_fingerprints[i]
            magnitude = self._calculate_change_magnitude(fp1, fp2)
            changes.append(magnitude)
        
        avg_change = sum(changes) / len(changes)
        
        # Count significant identity changes
        cutoff_time = datetime.now() - timeframe
        significant_changes = [
            c for c in self.identity_changes 
            if c.timestamp >= cutoff_time and c.magnitude > self.change_detection_threshold
        ]
        
        # More changes = less stability
        change_penalty = min(0.5, len(significant_changes) * 0.1)
        
        # Calculate stability (inverse of change)
        stability = 1.0 - avg_change - change_penalty
        
        return max(0.0, min(1.0, stability))
    
    def get_identity_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the identity system.
        
        Returns:
            Dictionary with identity stats
        """
        current_fingerprint = self.get_identity_fingerprint()
        
        stats = {
            "fingerprint_count": len(self.fingerprint_history),
            "change_count": len(self.identity_changes),
            "current_stability": current_fingerprint.stability_score if current_fingerprint else 0.0,
            "current_coherence": current_fingerprint.coherence_score if current_fingerprint else 0.0,
            "identity_stability": self.get_identity_stability(),
            "last_check_time": self.last_check_time.isoformat(),
            "recent_changes": []
        }
        
        # Add recent changes
        recent_changes = self.get_identity_changes(
            since=datetime.now() - timedelta(days=1)
        )
        stats["recent_changes"] = [
            {
                "timestamp": c.timestamp.isoformat(),
                "type": c.change_type.value,
                "magnitude": c.magnitude,
                "description": c.description
            }
            for c in recent_changes
        ]
        
        return stats 