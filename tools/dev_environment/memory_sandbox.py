"""
Memory Manipulation Sandbox for ΨC-AI SDK Development Environment

This module provides a sandbox for manipulating and testing agent memory structures.
It allows developers to:
1. Load, manipulate, and save agent memory states
2. Create synthetic memories for testing
3. Visualize memory activation patterns
4. Test memory recall and association mechanisms
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from psi_c_ai_sdk.tools.dev_environment.base_tool import BaseTool
from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.memory.memory_types import Memory, MemoryType

class MemorySandbox(BaseTool):
    """Sandbox for manipulating and testing agent memory structures."""
    
    def __init__(self, memory_store: Optional[MemoryStore] = None, snapshot_dir: str = "./memory_snapshots"):
        """
        Initialize the Memory Manipulation Sandbox.
        
        Args:
            memory_store: Optional MemoryStore to manipulate
            snapshot_dir: Directory to store memory snapshots
        """
        super().__init__(name="Memory Manipulation Sandbox")
        self.memory_store = memory_store
        self.snapshot_dir = snapshot_dir
        self.history = []
        self.snapshots = {}
        
        os.makedirs(snapshot_dir, exist_ok=True)
        
    def set_memory_store(self, memory_store: MemoryStore) -> None:
        """Set the memory store to manipulate."""
        self.take_snapshot("before_change")
        self.memory_store = memory_store
        
    def take_snapshot(self, name: str) -> Dict[str, Any]:
        """
        Take a snapshot of the current memory state.
        
        Args:
            name: Name of the snapshot
            
        Returns:
            Dictionary representation of the memory state
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"{name}_{timestamp}"
        
        # Serialize the memory store state
        memories = self.memory_store.get_all_memories()
        snapshot = {
            "timestamp": timestamp,
            "memory_count": len(memories),
            "memories": [memory.to_dict() for memory in memories]
        }
        
        self.snapshots[snapshot_name] = snapshot
        
        # Save to disk
        snapshot_path = os.path.join(self.snapshot_dir, f"{snapshot_name}.json")
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)
            
        print(f"Snapshot '{snapshot_name}' saved with {len(memories)} memories")
        return snapshot
    
    def load_snapshot(self, snapshot_name: str) -> None:
        """
        Load a memory snapshot.
        
        Args:
            snapshot_name: Name of the snapshot to load
        """
        if snapshot_name not in self.snapshots:
            # Try to load from disk
            snapshot_path = os.path.join(self.snapshot_dir, f"{snapshot_name}.json")
            if not os.path.exists(snapshot_path):
                raise ValueError(f"Snapshot '{snapshot_name}' not found")
                
            with open(snapshot_path, "r") as f:
                snapshot = json.load(f)
                self.snapshots[snapshot_name] = snapshot
        
        # Create a new memory store with the snapshot data
        if self.memory_store is None:
            from psi_c_ai_sdk.memory.memory_store import MemoryStore
            self.memory_store = MemoryStore()
            
        # Clear current memories
        self.memory_store.clear()
        
        # Load memories from snapshot
        for memory_dict in self.snapshots[snapshot_name]["memories"]:
            memory = Memory.from_dict(memory_dict)
            self.memory_store.add_memory(memory)
            
        print(f"Loaded snapshot '{snapshot_name}' with {len(self.snapshots[snapshot_name]['memories'])} memories")
    
    def create_synthetic_memory(self, content: str, memory_type: MemoryType, 
                              importance: float = 0.5, 
                              embedding: Optional[List[float]] = None) -> Memory:
        """
        Create a synthetic memory for testing.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            embedding: Optional embedding vector
            
        Returns:
            The created Memory object
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        # Generate a random embedding if none provided
        if embedding is None:
            embedding = list(np.random.randn(128))  # Default embedding size
            
        # Create memory
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            timestamp=datetime.now()
        )
        
        # Add to memory store
        self.memory_store.add_memory(memory)
        return memory
    
    def batch_create_memories(self, count: int, memory_type: MemoryType, 
                             template: str = "Synthetic memory {index}",
                             importance_range: Tuple[float, float] = (0.1, 0.9)) -> List[Memory]:
        """
        Create multiple synthetic memories at once.
        
        Args:
            count: Number of memories to create
            memory_type: Type of memories
            template: Template string for content
            importance_range: Range of importance scores
            
        Returns:
            List of created Memory objects
        """
        memories = []
        for i in range(count):
            importance = importance_range[0] + (importance_range[1] - importance_range[0]) * np.random.random()
            content = template.format(index=i+1)
            memory = self.create_synthetic_memory(content, memory_type, importance)
            memories.append(memory)
        
        return memories
    
    def visualize_memory_activation(self, n_recent: int = 50, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize memory activation patterns.
        
        Args:
            n_recent: Number of recent memories to visualize
            figsize: Figure size
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        memories = sorted(self.memory_store.get_all_memories(), 
                          key=lambda m: m.last_accessed if m.last_accessed else m.timestamp,
                          reverse=True)[:n_recent]
        
        if not memories:
            print("No memories to visualize")
            return
            
        # Extract data for visualization
        labels = [f"{m.memory_type.name}: {m.content[:30]}..." for m in memories]
        importance = [m.importance for m in memories]
        recency = [(datetime.now() - (m.last_accessed or m.timestamp)).total_seconds() / 3600 for m in memories]
        memory_types = [m.memory_type.name for m in memories]
        
        # Create colormap based on memory types
        unique_types = list(set(memory_types))
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_types)))
        color_map = {t: colors[i] for i, t in enumerate(unique_types)}
        
        # Visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot memories as scatter points
        for i, memory in enumerate(memories):
            ax.scatter(recency[i], importance[i], 
                     s=100 + importance[i] * 200,
                     color=color_map[memory.memory_type.name],
                     alpha=0.7,
                     label=memory.memory_type.name)
            
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Memory Types")
        
        ax.set_xlabel("Recency (hours)")
        ax.set_ylabel("Importance")
        ax.set_title("Memory Activation Pattern")
        ax.grid(True, alpha=0.3)
        
        # Invert x-axis so most recent are on the right
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.show()
    
    def test_memory_recall(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Test memory recall mechanism.
        
        Args:
            query: Query string
            top_k: Number of top memories to return
            
        Returns:
            List of recalled memories
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        recalled_memories = self.memory_store.retrieve_relevant_memories(query, top_k)
        
        print(f"Retrieved {len(recalled_memories)} memories for query: '{query}'")
        for i, memory in enumerate(recalled_memories):
            print(f"{i+1}. [{memory.memory_type.name}] {memory.content[:50]}... (Relevance: {memory.relevance:.3f})")
            
        return recalled_memories
    
    def analyze_memory_distribution(self) -> Dict[str, int]:
        """
        Analyze distribution of memories by type.
        
        Returns:
            Dictionary with counts by memory type
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        memories = self.memory_store.get_all_memories()
        distribution = {}
        
        for memory in memories:
            memory_type = memory.memory_type.name
            if memory_type not in distribution:
                distribution[memory_type] = 0
            distribution[memory_type] += 1
            
        # Print analysis
        print("Memory Distribution:")
        for memory_type, count in distribution.items():
            print(f"  {memory_type}: {count} ({count/len(memories)*100:.1f}%)")
            
        return distribution
            
    def compare_memories(self, memory_id1: str, memory_id2: str) -> Dict[str, Any]:
        """
        Compare two memories and analyze their similarity.
        
        Args:
            memory_id1: ID of first memory to compare
            memory_id2: ID of second memory to compare
            
        Returns:
            Dictionary with comparison metrics
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        memories = self.memory_store.get_all_memories()
        
        # Find memories by ID
        memory1 = next((m for m in memories if getattr(m, 'id', None) == memory_id1), None)
        memory2 = next((m for m in memories if getattr(m, 'id', None) == memory_id2), None)
        
        if not memory1 or not memory2:
            raise ValueError(f"Could not find memories with IDs {memory_id1} and {memory_id2}")
            
        # Calculate cosine similarity between embeddings
        if hasattr(memory1, 'embedding') and hasattr(memory2, 'embedding'):
            from sklearn.metrics.pairwise import cosine_similarity
            
            embedding1 = np.array(memory1.embedding).reshape(1, -1)
            embedding2 = np.array(memory2.embedding).reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
        else:
            similarity = None
            
        # Compare other attributes
        same_type = memory1.memory_type == memory2.memory_type
        importance_diff = abs(memory1.importance - memory2.importance)
        time_diff = abs((memory1.timestamp - memory2.timestamp).total_seconds())
        
        return {
            "semantic_similarity": similarity,
            "same_type": same_type,
            "importance_difference": importance_diff,
            "time_difference_seconds": time_diff,
            "memory1": {
                "id": memory_id1,
                "type": memory1.memory_type.name,
                "importance": memory1.importance,
                "timestamp": memory1.timestamp.isoformat()
            },
            "memory2": {
                "id": memory_id2,
                "type": memory2.memory_type.name,
                "importance": memory2.importance,
                "timestamp": memory2.timestamp.isoformat()
            }
        }
        
    def compare_snapshots(self, snapshot1_name: str, snapshot2_name: str) -> Dict[str, Any]:
        """
        Compare two memory snapshots and analyze differences.
        
        Args:
            snapshot1_name: Name of first snapshot
            snapshot2_name: Name of second snapshot
            
        Returns:
            Dictionary with comparison metrics
        """
        # Load snapshots
        if snapshot1_name not in self.snapshots:
            snapshot_path = os.path.join(self.snapshot_dir, f"{snapshot1_name}.json")
            if not os.path.exists(snapshot_path):
                raise ValueError(f"Snapshot '{snapshot1_name}' not found")
                
            with open(snapshot_path, "r") as f:
                snapshot1 = json.load(f)
                self.snapshots[snapshot1_name] = snapshot1
        else:
            snapshot1 = self.snapshots[snapshot1_name]
            
        if snapshot2_name not in self.snapshots:
            snapshot_path = os.path.join(self.snapshot_dir, f"{snapshot2_name}.json")
            if not os.path.exists(snapshot_path):
                raise ValueError(f"Snapshot '{snapshot2_name}' not found")
                
            with open(snapshot_path, "r") as f:
                snapshot2 = json.load(f)
                self.snapshots[snapshot2_name] = snapshot2
        else:
            snapshot2 = self.snapshots[snapshot2_name]
            
        # Extract memory IDs
        memories1 = snapshot1.get("memories", [])
        memories2 = snapshot2.get("memories", [])
        
        memory_ids1 = set(m.get("id", i) for i, m in enumerate(memories1))
        memory_ids2 = set(m.get("id", i) for i, m in enumerate(memories2))
        
        # Find common, added, and removed memories
        common_ids = memory_ids1.intersection(memory_ids2)
        added_ids = memory_ids2 - memory_ids1
        removed_ids = memory_ids1 - memory_ids2
        
        # Create type distribution for each snapshot
        type_dist1 = {}
        type_dist2 = {}
        
        for memory in memories1:
            memory_type = memory.get("memory_type", "unknown")
            type_dist1[memory_type] = type_dist1.get(memory_type, 0) + 1
            
        for memory in memories2:
            memory_type = memory.get("memory_type", "unknown")
            type_dist2[memory_type] = type_dist2.get(memory_type, 0) + 1
        
        # Calculate average importance for each snapshot
        avg_importance1 = sum(m.get("importance", 0) for m in memories1) / len(memories1) if memories1 else 0
        avg_importance2 = sum(m.get("importance", 0) for m in memories2) / len(memories2) if memories2 else 0
        
        return {
            "snapshot1": {
                "name": snapshot1_name,
                "memory_count": len(memories1),
                "type_distribution": type_dist1,
                "avg_importance": avg_importance1
            },
            "snapshot2": {
                "name": snapshot2_name,
                "memory_count": len(memories2),
                "type_distribution": type_dist2,
                "avg_importance": avg_importance2
            },
            "comparison": {
                "common_memory_count": len(common_ids),
                "added_memory_count": len(added_ids),
                "removed_memory_count": len(removed_ids),
                "importance_change": avg_importance2 - avg_importance1
            }
        }
        
    def delete_memory(self, memory_id: Optional[str] = None, index: Optional[int] = None) -> bool:
        """
        Delete a memory by ID or index.
        
        Args:
            memory_id: ID of the memory to delete (takes precedence over index)
            index: Index of the memory to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        memories = self.memory_store.get_all_memories()
        
        if not memories:
            print("No memories to delete")
            return False
            
        # Take snapshot before deletion for undo capability
        self.take_snapshot("before_deletion")
        
        # Delete by ID if provided
        if memory_id is not None:
            memory_to_delete = next((m for m in memories if getattr(m, 'id', None) == memory_id), None)
            
            if memory_to_delete is None:
                print(f"No memory found with ID: {memory_id}")
                return False
                
            # If MemoryStore has a delete method, use it
            if hasattr(self.memory_store, 'delete_memory') and callable(getattr(self.memory_store, 'delete_memory')):
                self.memory_store.delete_memory(memory_id)
                print(f"Deleted memory with ID: {memory_id}")
                return True
            else:
                # Otherwise, recreate the memory store without the deleted memory
                new_memories = [m for m in memories if getattr(m, 'id', None) != memory_id]
                self._replace_memories(new_memories)
                print(f"Deleted memory with ID: {memory_id}")
                return True
                
        # Delete by index if ID not provided
        elif index is not None:
            if index < 0 or index >= len(memories):
                print(f"Invalid index: {index}. Valid range is 0-{len(memories)-1}")
                return False
                
            memory_to_delete = memories[index]
            
            # If MemoryStore has a delete method and memory has an ID, use it
            memory_id = getattr(memory_to_delete, 'id', None)
            if memory_id and hasattr(self.memory_store, 'delete_memory') and callable(getattr(self.memory_store, 'delete_memory')):
                self.memory_store.delete_memory(memory_id)
                print(f"Deleted memory at index {index}")
                return True
            else:
                # Otherwise, recreate the memory store without the deleted memory
                new_memories = memories.copy()
                new_memories.pop(index)
                self._replace_memories(new_memories)
                print(f"Deleted memory at index {index}")
                return True
        
        print("Either memory_id or index must be provided")
        return False
        
    def _replace_memories(self, new_memories):
        """
        Replace all memories in the memory store with the given list.
        
        Args:
            new_memories: List of memories to use
        """
        # Clear the memory store
        if hasattr(self.memory_store, 'clear') and callable(getattr(self.memory_store, 'clear')):
            self.memory_store.clear()
        else:
            # Create a new memory store if clear method not available
            from psi_c_ai_sdk.memory.memory_store import MemoryStore
            self.memory_store = MemoryStore()
            
        # Add all memories
        for memory in new_memories:
            self.memory_store.add_memory(memory)
            
    def simulate_memory_dynamics(self, 
                               duration_days: int = 30, 
                               memory_decay_rate: float = 0.05,
                               memory_creation_rate: float = 5,
                               importance_drift: float = 0.02,
                               memory_types_distribution: Optional[Dict[str, float]] = None) -> None:
        """
        Simulate memory dynamics over time.
        
        This simulates:
        - Memory creation with realistic patterns
        - Memory importance decay/strengthening
        - Memory recall effects
        - Forgetting of low-importance memories
        
        Args:
            duration_days: Number of days to simulate
            memory_decay_rate: Rate at which memory importance decays per day
            memory_creation_rate: Average number of new memories per day
            importance_drift: Random fluctuation in importance per day
            memory_types_distribution: Distribution of memory types to create (defaults to equal)
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        # Create a snapshot before simulation
        self.take_snapshot("before_simulation")
        
        # Default distribution if none provided
        if memory_types_distribution is None:
            memory_types_distribution = {
                "EPISODIC": 0.4,
                "SEMANTIC": 0.3, 
                "PROCEDURAL": 0.2,
                "EMOTIONAL": 0.1
            }
            
        # Get current memories
        current_memories = self.memory_store.get_all_memories()
        print(f"Starting simulation with {len(current_memories)} memories over {duration_days} days")
            
        # Import required libraries
        import random
        from datetime import datetime, timedelta
        from psi_c_ai_sdk.memory.memory_types import MemoryType
        
        # Simulation start time
        start_time = datetime.now() - timedelta(days=duration_days)
        
        # Templates for synthetic memories
        templates = {
            "EPISODIC": [
                "Experienced {event} at {location}",
                "Met with {person} to discuss {topic}",
                "Observed {object} while at {location}",
                "Participated in {activity} with {person}"
            ],
            "SEMANTIC": [
                "Learned that {fact}",
                "Understood the concept of {concept}",
                "Realized that {insight}",
                "Discovered information about {topic}"
            ],
            "PROCEDURAL": [
                "Learned how to {action} using {tool}",
                "Practiced {skill} technique",
                "Developed method for {task}",
                "Established routine for {activity}"
            ],
            "EMOTIONAL": [
                "Felt {emotion} about {subject}",
                "Experienced {emotion} when thinking about {topic}",
                "Had strong {emotion} reaction to {stimulus}",
                "Developed {emotion} association with {subject}"
            ]
        }
        
        # Placeholder data for templates
        placeholders = {
            "event": ["a meeting", "a presentation", "a celebration", "a discussion", "an interview"],
            "location": ["the office", "home", "a conference room", "downtown", "the park"],
            "person": ["a colleague", "a friend", "a family member", "a client", "a stranger"],
            "topic": ["project goals", "recent developments", "future plans", "technical challenges", "personal interests"],
            "object": ["an interesting artifact", "a new product", "unusual behavior", "a pattern", "a document"],
            "activity": ["brainstorming", "planning", "analyzing data", "problem-solving", "decision making"],
            "fact": ["certain systems exhibit emergent properties", "human cognition relies on pattern recognition", 
                    "complex systems often have simple rules", "adaptation is key to intelligence", "communication requires shared context"],
            "concept": ["emergence", "abstraction", "recursion", "feedback loops", "self-organization"],
            "insight": ["patterns repeat at different scales", "small changes can have large effects", 
                       "most systems have implicit assumptions", "observation changes the observed", "boundaries are often arbitrary"],
            "action": ["analyze data", "solve complex problems", "communicate effectively", "make decisions", "learn continuously"],
            "tool": ["algorithms", "frameworks", "visualization techniques", "formal methods", "modeling tools"],
            "skill": ["pattern recognition", "systematic thinking", "creative problem-solving", "critical analysis", "abstract reasoning"],
            "task": ["information processing", "knowledge representation", "decision optimization", "pattern detection", "prediction"],
            "emotion": ["curiosity", "satisfaction", "concern", "excitement", "confidence"],
            "subject": ["learning outcomes", "problem complexity", "system behavior", "unexpected results", "future possibilities"],
            "stimulus": ["new information", "a challenging problem", "an elegant solution", "a surprising connection", "a novel perspective"]
        }
        
        # Function to generate a memory content from template
        def generate_memory_content(memory_type):
            template = random.choice(templates[memory_type])
            
            # Replace each placeholder with a random choice
            for placeholder, options in placeholders.items():
                if "{" + placeholder + "}" in template:
                    template = template.replace("{" + placeholder + "}", random.choice(options))
                    
            return template
        
        # Simulation day by day
        for day in range(duration_days):
            current_time = start_time + timedelta(days=day)
            
            # 1. Simulate memory decay and random importance changes
            for memory in current_memories:
                # Decay importance
                memory.importance *= (1 - memory_decay_rate)
                
                # Random importance drift (can increase or decrease)
                memory.importance += random.uniform(-importance_drift, importance_drift)
                
                # Ensure importance stays in [0, 1] range
                memory.importance = max(0, min(1, memory.importance))
                
                # Update last accessed for some memories (simulating recall)
                if random.random() < 0.1:  # 10% chance of recall per day
                    memory.last_accessed = current_time
                    # Strengthen remembered memories slightly
                    memory.importance = min(1.0, memory.importance * 1.05)
            
            # 2. Remove "forgotten" memories (very low importance)
            current_memories = [m for m in current_memories if m.importance > 0.05]
            
            # 3. Create new memories
            num_new_memories = int(random.normalvariate(memory_creation_rate, memory_creation_rate/3))
            for _ in range(max(0, num_new_memories)):
                # Select memory type based on distribution
                memory_type_name = random.choices(
                    list(memory_types_distribution.keys()),
                    weights=list(memory_types_distribution.values())
                )[0]
                
                mem_type = getattr(MemoryType, memory_type_name)
                
                # Generate content
                content = generate_memory_content(memory_type_name)
                
                # Create memory with timestamp during this day
                time_offset = random.uniform(0, 86400)  # seconds in a day
                timestamp = current_time + timedelta(seconds=time_offset)
                
                # Create with random importance, higher for emotional memories
                base_importance = random.uniform(0.3, 0.8)
                if memory_type_name == "EMOTIONAL":
                    base_importance = random.uniform(0.5, 0.9)
                
                # Create and add the memory
                memory = Memory(
                    content=content,
                    memory_type=mem_type,
                    importance=base_importance,
                    embedding=list(np.random.randn(128)),
                    timestamp=timestamp
                )
                current_memories.append(memory)
                
            # Progress update every 10% of simulation
            if day % max(1, duration_days // 10) == 0 or day == duration_days - 1:
                print(f"Day {day+1}/{duration_days}: {len(current_memories)} memories")
        
        # Replace memory store with simulated memories
        self._replace_memories(current_memories)
        
        # Create a snapshot after simulation
        self.take_snapshot("after_simulation")
        print(f"Simulation complete. Memory count: {len(current_memories)}")
        
        return current_memories
        
    def calculate_memory_statistics(self) -> Dict[str, Any]:
        """
        Calculate advanced statistics about the memory store.
        
        Returns:
            Dictionary with various memory statistics
        """
        if self.memory_store is None:
            raise ValueError("No memory store is set")
            
        memories = self.memory_store.get_all_memories()
        
        if not memories:
            return {"error": "No memories available"}
            
        # Calculate basic statistics
        memory_count = len(memories)
        memory_types = {}
        total_importance = 0
        oldest_timestamp = None
        newest_timestamp = None
        
        importance_values = []
        recency_values = []  # In hours
        
        now = datetime.now()
        
        for memory in memories:
            # Count by type
            memory_type = memory.memory_type.name
            if memory_type not in memory_types:
                memory_types[memory_type] = 0
            memory_types[memory_type] += 1
            
            # Importance stats
            importance = memory.importance
            total_importance += importance
            importance_values.append(importance)
            
            # Timestamp stats
            timestamp = memory.timestamp
            if oldest_timestamp is None or timestamp < oldest_timestamp:
                oldest_timestamp = timestamp
            if newest_timestamp is None or timestamp > newest_timestamp:
                newest_timestamp = timestamp
                
            # Calculate recency (hours)
            recency = (now - timestamp).total_seconds() / 3600
            recency_values.append(recency)
            
        # Calculate advanced statistics
        avg_importance = total_importance / memory_count
        
        # Calculate importance distribution
        importance_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        importance_distribution = {f"{importance_bins[i]:.1f}-{importance_bins[i+1]:.1f}": 0 
                                  for i in range(len(importance_bins)-1)}
                                  
        for importance in importance_values:
            for i in range(len(importance_bins)-1):
                if importance_bins[i] <= importance < importance_bins[i+1] or \
                   (i == len(importance_bins)-2 and importance == importance_bins[i+1]):
                    key = f"{importance_bins[i]:.1f}-{importance_bins[i+1]:.1f}"
                    importance_distribution[key] += 1
                    break
        
        # Calculate time-based memory creation patterns
        memory_age_days = [(now - m.timestamp).total_seconds() / 86400 for m in memories]
        
        # Create histogram of memory creation
        max_age = max(memory_age_days)
        age_bins = np.linspace(0, max_age, min(20, int(max_age)+1))
        age_counts, _ = np.histogram(memory_age_days, bins=age_bins)
        age_distribution = {f"{age_bins[i]:.1f}-{age_bins[i+1]:.1f} days": int(age_counts[i]) 
                           for i in range(len(age_counts))}
        
        # Calculate correlation between importance and recency
        if len(importance_values) > 1:
            from scipy.stats import pearsonr
            importance_recency_correlation, _ = pearsonr(importance_values, recency_values)
        else:
            importance_recency_correlation = None
            
        # Calculate memory retrieval probability distribution
        # Simple model: P(retrieval) ~ importance * recency_factor
        retrieval_probs = []
        for m in memories:
            recency_hours = (now - m.timestamp).total_seconds() / 3600
            # Apply a logarithmic decay to recency
            recency_factor = 1.0 / (1.0 + np.log1p(recency_hours / 24.0))  # normalize by days
            retrieval_prob = m.importance * recency_factor
            retrieval_probs.append(retrieval_prob)
            
        # Normalize probabilities
        total_prob = sum(retrieval_probs)
        if total_prob > 0:
            retrieval_probs = [p / total_prob for p in retrieval_probs]
        
        # Return all calculated statistics
        return {
            "basic": {
                "memory_count": memory_count,
                "memory_types": memory_types,
                "avg_importance": avg_importance,
                "time_span_days": (newest_timestamp - oldest_timestamp).total_seconds() / 86400 if oldest_timestamp and newest_timestamp else 0
            },
            "importance": {
                "distribution": importance_distribution,
                "min": min(importance_values),
                "max": max(importance_values),
                "median": sorted(importance_values)[len(importance_values)//2],
                "std_dev": np.std(importance_values)
            },
            "temporal": {
                "oldest_memory": oldest_timestamp.isoformat() if oldest_timestamp else None,
                "newest_memory": newest_timestamp.isoformat() if newest_timestamp else None,
                "age_distribution": age_distribution
            },
            "correlations": {
                "importance_recency": importance_recency_correlation
            },
            "retrieval": {
                "top_memories": [
                    {
                        "content": memories[i].content[:50] + "..." if len(memories[i].content) > 50 else memories[i].content,
                        "type": memories[i].memory_type.name,
                        "probability": retrieval_probs[i]
                    }
                    for i in sorted(range(len(retrieval_probs)), key=lambda i: retrieval_probs[i], reverse=True)[:5]
                ]
            }
        } 