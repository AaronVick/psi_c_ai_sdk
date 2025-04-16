"""
Reproducibility Framework
------------------------

This module ensures deterministic behavior when needed for scientific verification,
while maintaining stochastic capabilities.

Components:
- Seed management for RNGs
- Snapshot/restore primitives
- State fingerprinting
- Drift detection between runs

Mathematical basis:
- State hash function: H_state(S) = hash(params + memories + schema)
- Reproducibility score: R_score = 1 - dist(S_1, S_2)/max_dist
"""

import os
import sys
import time
import hashlib
import json
import pickle
import logging
import random
import numpy as np
import uuid
from typing import Dict, List, Set, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)


class SeedManager:
    """
    Manages random seeds for reproducible experiments.
    
    This class provides utilities for setting, tracking, and restoring
    random seeds across different libraries (Python random, NumPy, etc.)
    """
    
    _tracked_seeds: Dict[str, int] = {}
    _seed_history: List[Dict[str, Any]] = []
    
    @classmethod
    def set_seed(cls, seed: Optional[int] = None, name: str = "global") -> int:
        """
        Set a random seed for a specific source.
        
        Args:
            seed: Specific seed to use, or None for a random seed
            name: Name of the seed source
            
        Returns:
            The seed that was set
        """
        if seed is None:
            # Generate a random seed
            seed = int(time.time() * 1000) % (2**32 - 1)
        
        # Track the seed
        cls._tracked_seeds[name] = seed
        cls._seed_history.append({
            "timestamp": time.time(),
            "name": name,
            "seed": seed
        })
        
        # Set seed for appropriate library
        if name == "global" or name == "random":
            random.seed(seed)
            logger.debug(f"Set random seed to {seed}")
            
        if name == "global" or name == "numpy":
            np.random.seed(seed)
            logger.debug(f"Set numpy seed to {seed}")
            
        if name == "global" or name == "torch":
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.debug(f"Set torch seed to {seed}")
            except ImportError:
                pass  # Torch not available
        
        if name == "global" or name == "tensorflow":
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
                logger.debug(f"Set tensorflow seed to {seed}")
            except ImportError:
                pass  # TensorFlow not available
        
        return seed
    
    @classmethod
    def get_seed(cls, name: str = "global") -> Optional[int]:
        """Get the current seed for a specific source."""
        return cls._tracked_seeds.get(name)
    
    @classmethod
    def get_all_seeds(cls) -> Dict[str, int]:
        """Get all tracked seeds."""
        return cls._tracked_seeds.copy()
    
    @classmethod
    def restore_seeds(cls, seeds: Dict[str, int]) -> None:
        """
        Restore seeds from a previous state.
        
        Args:
            seeds: Dictionary mapping seed names to seed values
        """
        for name, seed in seeds.items():
            cls.set_seed(seed, name)
            
    @classmethod
    def get_seed_history(cls) -> List[Dict[str, Any]]:
        """Get the history of seed changes."""
        return cls._seed_history.copy()
    
    @classmethod
    def reset(cls) -> None:
        """Reset all seeds to new random values."""
        for name in list(cls._tracked_seeds.keys()):
            cls.set_seed(None, name)


@dataclass
class StateFingerprint:
    """
    Fingerprint of a system state for reproducibility comparisons.
    
    This class encapsulates a hash of the system state, including components
    like memory, schema, and parameters.
    """
    
    hash: str
    timestamp: float = field(default_factory=time.time)
    components: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateFingerprint':
        """Create fingerprint from dictionary."""
        return cls(
            hash=data["hash"],
            timestamp=data.get("timestamp", time.time()),
            components=data.get("components", {}),
            metadata=data.get("metadata", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary."""
        return {
            "hash": self.hash,
            "timestamp": self.timestamp,
            "components": self.components,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert fingerprint to JSON."""
        return json.dumps(self.to_dict())
    
    def __eq__(self, other: Any) -> bool:
        """
        Check if this fingerprint equals another.
        
        Args:
            other: Another fingerprint
            
        Returns:
            True if the fingerprints are equal
        """
        if not isinstance(other, StateFingerprint):
            return False
        return self.hash == other.hash


class StateSnapshot:
    """
    Snapshot of a system state for reproducibility.
    
    This class captures the complete state of the system at a point in time,
    including memory, schema, parameters, and random seeds.
    """
    
    def __init__(
        self,
        state: Dict[str, Any],
        seeds: Dict[str, int],
        fingerprint: StateFingerprint,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a snapshot.
        
        Args:
            state: Complete state dictionary
            seeds: Dictionary of random seeds
            fingerprint: State fingerprint
            metadata: Additional metadata
        """
        self.state = state
        self.seeds = seeds
        self.fingerprint = fingerprint
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.snapshot_id = str(uuid.uuid4())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create snapshot from dictionary."""
        return cls(
            state=data["state"],
            seeds=data["seeds"],
            fingerprint=StateFingerprint.from_dict(data["fingerprint"]),
            metadata=data.get("metadata", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "state": self.state,
            "seeds": self.seeds,
            "fingerprint": self.fingerprint.to_dict(),
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert snapshot to JSON."""
        return json.dumps(self.to_dict())
    
    def save(self, filepath: str) -> None:
        """
        Save the snapshot to a file.
        
        Args:
            filepath: Path to save the snapshot
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'StateSnapshot':
        """
        Load a snapshot from a file.
        
        Args:
            filepath: Path to load the snapshot from
            
        Returns:
            Loaded snapshot
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ReproducibilityError(Exception):
    """Exception raised for reproducibility issues."""
    pass


class ReproducibilityFramework:
    """
    Framework for ensuring reproducible experiments.
    
    This class provides utilities for capturing, restoring, and comparing
    system states for reproducibility.
    """
    
    _snapshots: Dict[str, StateSnapshot] = {}
    _fingerprints: List[StateFingerprint] = []
    _restore_callbacks: Dict[str, Callable] = {}
    _fingerprint_callbacks: Dict[str, Callable] = {}
    
    @classmethod
    def register_restore_callback(cls, component_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for restoring a component state.
        
        Args:
            component_id: ID of the component
            callback: Function that takes a state dictionary and restores the component
        """
        cls._restore_callbacks[component_id] = callback
        logger.debug(f"Registered restore callback for {component_id}")
    
    @classmethod
    def register_fingerprint_callback(cls, component_id: str, callback: Callable[[], str]) -> None:
        """
        Register a callback for fingerprinting a component.
        
        Args:
            component_id: ID of the component
            callback: Function that returns a hash string for the component
        """
        cls._fingerprint_callbacks[component_id] = callback
        logger.debug(f"Registered fingerprint callback for {component_id}")
    
    @classmethod
    def create_fingerprint(cls, **extra_metadata) -> StateFingerprint:
        """
        Create a fingerprint of the current system state.
        
        Args:
            **extra_metadata: Additional metadata to include
            
        Returns:
            State fingerprint
        """
        # Collect component hashes
        component_hashes = {}
        for component_id, callback in cls._fingerprint_callbacks.items():
            try:
                component_hash = callback()
                component_hashes[component_id] = component_hash
            except Exception as e:
                logger.warning(f"Error fingerprinting {component_id}: {e}")
                component_hashes[component_id] = "error"
        
        # Create combined hash
        combined = "".join(sorted([f"{k}:{v}" for k, v in component_hashes.items()]))
        combined_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        # Create fingerprint
        fingerprint = StateFingerprint(
            hash=combined_hash,
            components=component_hashes,
            metadata=extra_metadata
        )
        
        # Track fingerprint
        cls._fingerprints.append(fingerprint)
        
        return fingerprint
    
    @classmethod
    def create_snapshot(cls, snapshot_id: Optional[str] = None, **extra_metadata) -> StateSnapshot:
        """
        Create a snapshot of the current system state.
        
        Args:
            snapshot_id: ID for the snapshot, or None for auto-generated
            **extra_metadata: Additional metadata to include
            
        Returns:
            State snapshot
        """
        # Create state dictionary
        state = {}
        for component_id, callback in cls._restore_callbacks.items():
            # This will be filled during restoration
            state[component_id] = None
        
        # Get seeds
        seeds = SeedManager.get_all_seeds()
        
        # Create fingerprint
        fingerprint = cls.create_fingerprint(**extra_metadata)
        
        # Create snapshot
        snapshot = StateSnapshot(
            state=state,
            seeds=seeds,
            fingerprint=fingerprint,
            metadata=extra_metadata
        )
        
        # Generate snapshot ID if needed
        if snapshot_id is None:
            snapshot_id = snapshot.snapshot_id
        else:
            snapshot.snapshot_id = snapshot_id
        
        # Store snapshot
        cls._snapshots[snapshot_id] = snapshot
        
        logger.info(f"Created snapshot {snapshot_id}")
        return snapshot
    
    @classmethod
    def restore_snapshot(cls, snapshot: Union[str, StateSnapshot]) -> None:
        """
        Restore a system state from a snapshot.
        
        Args:
            snapshot: Snapshot object or ID
        """
        # Get snapshot object
        if isinstance(snapshot, str):
            if snapshot not in cls._snapshots:
                raise ValueError(f"Snapshot {snapshot} not found")
            snapshot = cls._snapshots[snapshot]
        
        # Restore seeds
        SeedManager.restore_seeds(snapshot.seeds)
        
        # Call restore callbacks
        for component_id, callback in cls._restore_callbacks.items():
            try:
                callback(snapshot.state.get(component_id))
            except Exception as e:
                logger.error(f"Error restoring {component_id}: {e}")
                raise ReproducibilityError(f"Failed to restore {component_id}: {e}")
        
        # Verify restored state
        new_fingerprint = cls.create_fingerprint()
        if new_fingerprint.hash != snapshot.fingerprint.hash:
            logger.warning(
                f"Fingerprint mismatch after restoration: "
                f"{new_fingerprint.hash} != {snapshot.fingerprint.hash}"
            )
            # Detailed component comparison
            for component_id, original_hash in snapshot.fingerprint.components.items():
                if component_id in new_fingerprint.components:
                    new_hash = new_fingerprint.components[component_id]
                    if new_hash != original_hash:
                        logger.warning(f"Component {component_id} hash mismatch: {new_hash} != {original_hash}")
        
        logger.info(f"Restored snapshot {snapshot.snapshot_id}")
    
    @classmethod
    def save_snapshot(cls, snapshot: Union[str, StateSnapshot], filepath: str) -> None:
        """
        Save a snapshot to a file.
        
        Args:
            snapshot: Snapshot object or ID
            filepath: Path to save the snapshot
        """
        # Get snapshot object
        if isinstance(snapshot, str):
            if snapshot not in cls._snapshots:
                raise ValueError(f"Snapshot {snapshot} not found")
            snapshot = cls._snapshots[snapshot]
        
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save snapshot
        snapshot.save(filepath)
        logger.info(f"Saved snapshot {snapshot.snapshot_id} to {filepath}")
    
    @classmethod
    def load_snapshot(cls, filepath: str) -> StateSnapshot:
        """
        Load a snapshot from a file.
        
        Args:
            filepath: Path to load the snapshot from
            
        Returns:
            Loaded snapshot
        """
        snapshot = StateSnapshot.load(filepath)
        cls._snapshots[snapshot.snapshot_id] = snapshot
        logger.info(f"Loaded snapshot {snapshot.snapshot_id} from {filepath}")
        return snapshot
    
    @classmethod
    def get_snapshot(cls, snapshot_id: str) -> Optional[StateSnapshot]:
        """Get a snapshot by ID."""
        return cls._snapshots.get(snapshot_id)
    
    @classmethod
    def list_snapshots(cls) -> List[Dict[str, Any]]:
        """
        Get a list of all snapshots.
        
        Returns:
            List of snapshot metadata
        """
        return [
            {
                "id": sid,
                "timestamp": snapshot.timestamp,
                "fingerprint": snapshot.fingerprint.hash,
                "metadata": snapshot.metadata
            }
            for sid, snapshot in cls._snapshots.items()
        ]
    
    @classmethod
    def calculate_reproducibility_score(
        cls, fingerprint1: StateFingerprint, fingerprint2: StateFingerprint
    ) -> float:
        """
        Calculate reproducibility score between two states.
        
        Args:
            fingerprint1: First state fingerprint
            fingerprint2: Second state fingerprint
            
        Returns:
            Reproducibility score between 0 and 1
        """
        # If hashes match exactly, score is 1.0
        if fingerprint1.hash == fingerprint2.hash:
            return 1.0
        
        # Component-wise comparison
        common_components = set(fingerprint1.components.keys()) & set(fingerprint2.components.keys())
        
        if not common_components:
            return 0.0  # No common components
            
        # Count matching components
        matches = 0
        for component_id in common_components:
            if fingerprint1.components[component_id] == fingerprint2.components[component_id]:
                matches += 1
        
        # Calculate score
        score = matches / len(common_components)
        return score
    
    @classmethod
    def detect_drift(cls, reference_fingerprint: StateFingerprint, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect drift from a reference state.
        
        Args:
            reference_fingerprint: Reference state fingerprint
            threshold: Threshold for considering a change significant
            
        Returns:
            Drift detection report
        """
        current_fingerprint = cls.create_fingerprint()
        score = cls.calculate_reproducibility_score(reference_fingerprint, current_fingerprint)
        
        # Component-wise comparison
        component_drift = {}
        all_components = set(reference_fingerprint.components.keys()) | set(current_fingerprint.components.keys())
        
        for component_id in all_components:
            ref_hash = reference_fingerprint.components.get(component_id)
            cur_hash = current_fingerprint.components.get(component_id)
            
            if ref_hash is None:
                component_drift[component_id] = "added"
            elif cur_hash is None:
                component_drift[component_id] = "removed"
            elif ref_hash != cur_hash:
                component_drift[component_id] = "changed"
            else:
                component_drift[component_id] = "unchanged"
        
        # Create report
        report = {
            "score": score,
            "significant_drift": score < (1.0 - threshold),
            "reference_hash": reference_fingerprint.hash,
            "current_hash": current_fingerprint.hash,
            "component_drift": component_drift,
            "drift_time": time.time() - reference_fingerprint.timestamp
        }
        
        return report


def make_reproducible(seed: Optional[int] = None) -> Dict[str, int]:
    """
    Make the current environment reproducible.
    
    This is a convenience function that sets seeds for common libraries.
    
    Args:
        seed: Seed to use, or None for a random seed
        
    Returns:
        Dictionary of seeds that were set
    """
    # Set the global seed
    global_seed = SeedManager.set_seed(seed, "global")
    
    # Set individual library seeds
    seeds = {
        "global": global_seed,
        "random": SeedManager.set_seed(global_seed, "random"),
        "numpy": SeedManager.set_seed(global_seed, "numpy")
    }
    
    # Try to set torch seed
    try:
        import torch
        seeds["torch"] = SeedManager.set_seed(global_seed, "torch")
    except ImportError:
        pass  # Torch not available
    
    # Try to set tensorflow seed
    try:
        import tensorflow as tf
        seeds["tensorflow"] = SeedManager.set_seed(global_seed, "tensorflow")
    except ImportError:
        pass  # TensorFlow not available
    
    # Try to set JAX seed
    try:
        import jax.random
        key = jax.random.PRNGKey(global_seed)
        # JAX doesn't have a global seed, so we just track it
        seeds["jax"] = global_seed
    except ImportError:
        pass  # JAX not available
    
    logger.info(f"Environment made reproducible with seed {global_seed}")
    return seeds


def compare_runs(run1: str, run2: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare two experimental runs.
    
    Args:
        run1: Path to first run's snapshot
        run2: Path to second run's snapshot
        output_file: Path to save comparison report
        
    Returns:
        Comparison report
    """
    # Load snapshots
    snapshot1 = ReproducibilityFramework.load_snapshot(run1)
    snapshot2 = ReproducibilityFramework.load_snapshot(run2)
    
    # Calculate reproducibility score
    score = ReproducibilityFramework.calculate_reproducibility_score(
        snapshot1.fingerprint, snapshot2.fingerprint
    )
    
    # Component comparison
    component_comparison = {}
    all_components = set(snapshot1.fingerprint.components.keys()) | set(snapshot2.fingerprint.components.keys())
    
    for component_id in all_components:
        hash1 = snapshot1.fingerprint.components.get(component_id)
        hash2 = snapshot2.fingerprint.components.get(component_id)
        
        if hash1 is None:
            status = "only_in_run2"
        elif hash2 is None:
            status = "only_in_run1"
        elif hash1 == hash2:
            status = "identical"
        else:
            status = "different"
        
        component_comparison[component_id] = {
            "status": status,
            "run1_hash": hash1,
            "run2_hash": hash2
        }
    
    # Create report
    report = {
        "reproducibility_score": score,
        "run1": {
            "id": snapshot1.snapshot_id,
            "timestamp": snapshot1.timestamp,
            "fingerprint": snapshot1.fingerprint.hash
        },
        "run2": {
            "id": snapshot2.snapshot_id,
            "timestamp": snapshot2.timestamp,
            "fingerprint": snapshot2.fingerprint.hash
        },
        "component_comparison": component_comparison,
        "seed_comparison": {
            "match": snapshot1.seeds == snapshot2.seeds,
            "run1_seeds": snapshot1.seeds,
            "run2_seeds": snapshot2.seeds
        }
    }
    
    # Save report if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved comparison report to {output_file}")
    
    return report


# Example usage
def example_usage():
    """Example of how to use the reproducibility framework."""
    # Make environment reproducible
    seed = 42
    seeds = make_reproducible(seed)
    print(f"Environment made reproducible with seed {seed}")
    
    # Register components for fingerprinting and restoration
    class MemorySystem:
        def __init__(self):
            self.memories = []
            
        def add_memory(self, memory):
            self.memories.append(memory)
            
        def get_state(self):
            return {"memories": self.memories}
            
        def restore_state(self, state):
            if state and "memories" in state:
                self.memories = state["memories"]
                
        def fingerprint(self):
            # Create a hash of the memories
            hash_input = str([str(m) for m in self.memories])
            return hashlib.sha256(hash_input.encode()).hexdigest()
    
    class SchemaSystem:
        def __init__(self):
            self.schema = {"nodes": [], "edges": []}
            
        def add_node(self, node):
            self.schema["nodes"].append(node)
            
        def add_edge(self, edge):
            self.schema["edges"].append(edge)
            
        def get_state(self):
            return {"schema": self.schema}
            
        def restore_state(self, state):
            if state and "schema" in state:
                self.schema = state["schema"]
                
        def fingerprint(self):
            # Create a hash of the schema
            hash_input = str(self.schema)
            return hashlib.sha256(hash_input.encode()).hexdigest()
    
    # Create components
    memory_system = MemorySystem()
    schema_system = SchemaSystem()
    
    # Register fingerprint callbacks
    ReproducibilityFramework.register_fingerprint_callback(
        "memory_system", memory_system.fingerprint
    )
    ReproducibilityFramework.register_fingerprint_callback(
        "schema_system", schema_system.fingerprint
    )
    
    # Register restore callbacks
    ReproducibilityFramework.register_restore_callback(
        "memory_system", memory_system.restore_state
    )
    ReproducibilityFramework.register_restore_callback(
        "schema_system", schema_system.restore_state
    )
    
    # Add some data
    memory_system.add_memory({"id": 1, "content": "A memory", "importance": 0.8})
    schema_system.add_node({"id": 1, "type": "concept", "name": "A concept"})
    schema_system.add_edge({"source": 1, "target": 2, "type": "related"})
    
    # Create baseline snapshot
    baseline_snapshot = ReproducibilityFramework.create_snapshot(
        snapshot_id="baseline",
        experiment="example",
        researcher="Jane Doe"
    )
    
    # Save the baseline snapshot
    ReproducibilityFramework.save_snapshot(baseline_snapshot, "baseline_snapshot.pkl")
    print(f"Created and saved baseline snapshot: {baseline_snapshot.snapshot_id}")
    
    # Add more data (modifies the state)
    memory_system.add_memory({"id": 2, "content": "Another memory", "importance": 0.5})
    schema_system.add_node({"id": 2, "type": "concept", "name": "Another concept"})
    
    # Create modified snapshot
    modified_snapshot = ReproducibilityFramework.create_snapshot(
        snapshot_id="modified",
        experiment="example",
        researcher="Jane Doe"
    )
    
    # Save the modified snapshot
    ReproducibilityFramework.save_snapshot(modified_snapshot, "modified_snapshot.pkl")
    print(f"Created and saved modified snapshot: {modified_snapshot.snapshot_id}")
    
    # Reset memory system and schema system
    memory_system.memories = []
    schema_system.schema = {"nodes": [], "edges": []}
    
    # Restore baseline snapshot
    ReproducibilityFramework.restore_snapshot("baseline")
    print(f"Restored baseline snapshot")
    
    # Check if restoration was successful
    print(f"Memory system has {len(memory_system.memories)} memories")
    print(f"Schema system has {len(schema_system.schema['nodes'])} nodes")
    
    # Compare runs
    comparison_report = compare_runs("baseline_snapshot.pkl", "modified_snapshot.pkl", "comparison_report.json")
    
    print(f"Comparison report:")
    print(f"  Reproducibility score: {comparison_report['reproducibility_score']}")
    print(f"  Component comparison:")
    for component_id, comparison in comparison_report["component_comparison"].items():
        print(f"    {component_id}: {comparison['status']}")
    
    # Detect drift
    drift_report = ReproducibilityFramework.detect_drift(baseline_snapshot.fingerprint)
    
    print(f"Drift report:")
    print(f"  Score: {drift_report['score']}")
    print(f"  Significant drift: {drift_report['significant_drift']}")
    print(f"  Component drift:")
    for component_id, status in drift_report["component_drift"].items():
        print(f"    {component_id}: {status}")


if __name__ == "__main__":
    example_usage() 