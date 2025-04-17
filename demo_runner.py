#!/usr/bin/env python3
"""
ΨC Schema Integration Demo Runner
---------------------------------
The central component of the demo system that provides a safe, isolated interface
to the ΨC-AI SDK. This module handles schema management, coherence calculation,
and state persistence while keeping demo code entirely separate from core SDK code.
"""

import os
import sys
import json
import pickle
import logging
import time
import random  # For mock implementations
import uuid    # For generating unique IDs
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Add the parent directory to the Python path for imports
parent_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent_dir)

# Mock classes for SDK components in case they're not available
class MockSchemaGraph:
    """Mock implementation of SchemaGraph for testing and fallback."""
    def __init__(self):
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id, **attr):
        self.nodes[node_id] = attr
    
    def add_edge(self, source, target, **attr):
        self.edges.append((source, target))
    
    def get_edge_data(self, source, target):
        return {"weight": 0.5, "type": "relation"}

class MockMemory:
    """Mock implementation of Memory for testing and fallback."""
    def __init__(self, content, metadata=None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }

class MockMemoryStore:
    """Mock implementation of MemoryStore for testing and fallback."""
    def __init__(self):
        self.memories = {}
    
    def add_memory(self, memory_dict):
        memory = MockMemory(memory_dict["content"], memory_dict.get("metadata", {}))
        self.memories[memory.id] = memory
        return memory.id
    
    def add_memory_from_text(self, content, metadata=None):
        memory = MockMemory(content, metadata)
        self.memories[memory.id] = memory
        return memory.id
    
    def get_memory_by_id(self, memory_id):
        return self.memories.get(memory_id)
    
    def get_all_memories(self):
        return list(self.memories.values())
    
    def get_related_memories(self, memory_id, limit=5):
        """Get memories related to the given memory ID."""
        all_memories = list(self.memories.values())
        target_memory = self.get_memory_by_id(memory_id)
        
        if not target_memory or len(all_memories) <= 1:
            return []
        
        # In a real implementation, this would use semantic similarity
        # Here we just return random memories other than the target
        other_memories = [m for m in all_memories if m.id != memory_id]
        result = other_memories[:min(limit, len(other_memories))]
        return result

class MockMemorySchemaIntegration:
    """Mock implementation of MemorySchemaIntegration for testing and fallback."""
    def __init__(self, schema_graph, memory_store):
        self.schema_graph = schema_graph
        self.memory_store = memory_store
    
    def integrate_memory(self, memory_id):
        """Simulate integrating a memory into the schema graph."""
        memory = self.memory_store.get_memory_by_id(memory_id)
        if not memory:
            return False
        
        # Add the memory as a node in the graph
        self.schema_graph.add_node(memory_id, 
                                  label=memory.content[:20] + "...", 
                                  type="memory",
                                  importance=memory.metadata.get("confidence", 0.5))
        
        # Connect to some existing nodes if they exist
        existing_nodes = list(self.schema_graph.nodes.keys())
        if existing_nodes:
            for _ in range(min(3, len(existing_nodes))):
                target = random.choice(existing_nodes)
                if target != memory_id:
                    self.schema_graph.add_edge(memory_id, target, 
                                             weight=random.uniform(0.3, 0.9),
                                             type="relation")
        
        return True

class MockCoherenceCalculator:
    """Mock implementation of CoherenceCalculator for testing and fallback."""
    def __init__(self, schema_graph):
        self.schema_graph = schema_graph
        self._base_coherence = random.uniform(0.5, 0.7)
        self._base_entropy = random.uniform(0.3, 0.5)
    
    def calculate_coherence(self):
        """Calculate a simulated coherence score based on graph size."""
        node_count = len(self.schema_graph.nodes)
        edge_count = len(self.schema_graph.edges)
        
        if node_count == 0:
            return self._base_coherence
        
        # More edges relative to nodes = higher coherence
        edge_to_node_ratio = edge_count / node_count if node_count > 0 else 0
        coherence = self._base_coherence + (edge_to_node_ratio * 0.1)
        return min(max(coherence, 0.1), 0.95)  # Keep between 0.1 and 0.95
    
    def calculate_entropy(self):
        """Calculate a simulated entropy score based on graph size."""
        node_count = len(self.schema_graph.nodes)
        edge_count = len(self.schema_graph.edges)
        
        if node_count == 0:
            return self._base_entropy
        
        # More edges relative to nodes = lower entropy
        edge_to_node_ratio = edge_count / node_count if node_count > 0 else 0
        entropy = self._base_entropy - (edge_to_node_ratio * 0.05)
        return min(max(entropy, 0.05), 0.9)  # Keep between 0.05 and 0.9

class MockContradiction:
    """Mock implementation of Contradiction for testing and fallback."""
    def __init__(self, memory_id, severity=0.5):
        self.memory_id = memory_id
        self.severity = severity
        self.description = "Possible contradiction detected"

class MockContradictionDetector:
    """Mock implementation of ContradictionDetector for testing and fallback."""
    def __init__(self, schema_graph):
        self.schema_graph = schema_graph
    
    def detect_contradictions(self, memory):
        """Detect simulated contradictions for a memory."""
        # Simulate a 20% chance of finding a contradiction
        if random.random() < 0.2:
            return [MockContradiction(memory.id, random.uniform(0.3, 0.8))]
        return []

class MockReflectionEngine:
    """Mock implementation of ReflectionEngine for testing and fallback."""
    def __init__(self, memory_store, depth=3):
        self.memory_store = memory_store
        self.depth = depth
    
    def reflect_on_memory(self, memory_id):
        """Simulate reflection on a memory."""
        memory = self.memory_store.get_memory_by_id(memory_id)
        if not memory:
            return "No memory found for reflection."
        
        return f"Reflected on memory: {memory.content[:50]}..." 

class MockHierarchicalKnowledgeManager:
    """Mock implementation of HierarchicalKnowledgeManager for testing and fallback."""
    def __init__(self, schema_graph):
        self.schema_graph = schema_graph
    
    def get_lineage(self, concept_id):
        """Get simulated lineage for a concept."""
        return []
    
    def find_common_ancestor(self, concept_id1, concept_id2):
        """Find simulated common ancestor for two concepts."""
        return None
    
    def get_concept_impact(self, concept_id):
        """Get simulated impact for a concept."""
        return 0.5

# Try to import SDK components, fall back to mocks if not available
try:
    from psi_c_ai_sdk.schema.schema_graph import SchemaGraph
    from psi_c_ai_sdk.schema.schema_math import calculate_coherence
    from psi_c_ai_sdk.memory.memory_store import MemoryStore
    from psi_c_ai_sdk.core.memory_schema_integration import MemorySchemaIntegration
    from psi_c_ai_sdk.schema.coherence.coherence_calculator import CoherenceCalculator
    from psi_c_ai_sdk.knowledge.hierarchical_manager import HierarchicalKnowledgeManager
    from psi_c_ai_sdk.schema.contradiction.contradiction_detector import ContradictionDetector
    from psi_c_ai_sdk.reflection.reflection_engine import ReflectionEngine
    
    logger = logging.getLogger("demo_runner")
    logger.info("Successfully imported SDK components")
except ImportError as e:
    # Fall back to mock implementations
    logger = logging.getLogger("demo_runner")
    logger.warning(f"SDK import error: {e}")
    logger.info("Falling back to mock implementations for demo")
    
    SchemaGraph = MockSchemaGraph
    MemoryStore = MockMemoryStore
    MemorySchemaIntegration = MockMemorySchemaIntegration
    CoherenceCalculator = MockCoherenceCalculator
    ContradictionDetector = MockContradictionDetector
    ReflectionEngine = MockReflectionEngine
    HierarchicalKnowledgeManager = MockHierarchicalKnowledgeManager

# Import LLM Bridge for enhanced capabilities if available
try:
    from llm_bridge import LLMBridge
    has_llm_bridge = True
except ImportError:
    has_llm_bridge = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_runner")

# Constants
DEFAULT_CONFIG = {
    "alpha": 0.5,  # Reflection weight
    "beta": 0.3,   # Information relevance weight
    "theta": 0.2,  # Minimum coherence pressure threshold
    "epsilon": 0.1,  # Identity continuity threshold
    "coherence_window": 50,  # Number of historical coherence values to track
    "contradiction_threshold": 0.3,  # Threshold for triggering schema updates
    "reflection_depth": 3,  # Depth of reflection process
    "use_llm": False,  # Whether to use LLM for enhanced explanations
}

STATE_DIR = Path(__file__).parent / "state"
CONFIG_DIR = Path(__file__).parent / "demo_config"


class DemoRunner:
    """
    The main demo runner that safely interfaces with the ΨC-AI SDK.
    
    This class handles:
    1. Schema graph management
    2. Memory integration
    3. Reflection and contradiction detection
    4. Coherence calculation
    5. State persistence
    6. Optional LLM-enhanced explanations
    """
    
    def __init__(self, profile: str = "default"):
        """
        Initialize the demo runner.
        
        Args:
            profile: Name of the demo profile to load (e.g., "healthcare", "legal")
        """
        STATE_DIR.mkdir(exist_ok=True)
        CONFIG_DIR.mkdir(exist_ok=True)
        
        self.profile = profile
        self.config = self._load_config()
        
        # Initialize SDK components
        self.schema_graph = self._load_schema_graph()
        self.memory_store = self._load_memory_store()
        self.integration = MemorySchemaIntegration(self.schema_graph, self.memory_store)
        self.coherence_calculator = CoherenceCalculator(self.schema_graph)
        self.contradiction_detector = ContradictionDetector(self.schema_graph)
        self.reflection_engine = ReflectionEngine(
            memory_store=self.memory_store,
            depth=self.config["reflection_depth"]
        )
        self.knowledge_manager = HierarchicalKnowledgeManager(self.schema_graph)
        
        # Initialize LLM bridge if available and enabled
        self.llm_bridge = None
        if has_llm_bridge and self.config.get("use_llm", False):
            self.llm_bridge = LLMBridge()
            logger.info(f"LLM Bridge initialized. Enabled: {self.llm_bridge.is_enabled()}")
        
        # Initialize demo state
        self.coherence_history = self._load_coherence_history()
        self.entropy_history = self._load_entropy_history()
        self.session_log = self._load_session_log()
        self.schema_version = len(self.session_log)
        
        logger.info(f"Demo runner initialized with profile: {profile}")
        logger.info(f"Current schema version: {self.schema_version}")
        logger.info(f"Current coherence: {self.get_current_coherence():.4f}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from the profile or use defaults."""
        config_path = CONFIG_DIR / f"{self.profile}_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**DEFAULT_CONFIG, **config}
        else:
            # Create default config if none exists
            with open(CONFIG_DIR / "default_config.json", 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            return DEFAULT_CONFIG
    
    def _load_schema_graph(self) -> SchemaGraph:
        """Load the schema graph from disk or create a new one."""
        schema_path = STATE_DIR / f"{self.profile}_schema_graph.pkl"
        
        if schema_path.exists():
            try:
                with open(schema_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading schema graph: {e}")
                logger.info("Creating new schema graph")
                return SchemaGraph()
        else:
            return SchemaGraph()
    
    def _load_memory_store(self) -> MemoryStore:
        """Load the memory store from disk or create a new one."""
        memory_path = STATE_DIR / f"{self.profile}_memory.json"
        
        if memory_path.exists():
            try:
                memory_store = MemoryStore()
                with open(memory_path, 'r') as f:
                    memories = json.load(f)
                
                for mem in memories:
                    memory_store.add_memory(mem)
                
                return memory_store
            except Exception as e:
                logger.error(f"Error loading memory store: {e}")
                logger.info("Creating new memory store")
                return MemoryStore()
        else:
            # Check if there's a preloaded case
            memory_store = MemoryStore()
            case_path = CONFIG_DIR / f"{self.profile}_case.json"
            
            if case_path.exists():
                try:
                    with open(case_path, 'r') as f:
                        case_data = json.load(f)
                    
                    # Load preloaded memories
                    if "memories" in case_data:
                        for mem in case_data["memories"]:
                            content = mem["content"]
                            metadata = mem["metadata"]
                            memory_store.add_memory_from_text(content, metadata)
                        
                        logger.info(f"Loaded {len(case_data['memories'])} preloaded memories from {self.profile}_case.json")
                except Exception as e:
                    logger.error(f"Error loading preloaded case: {e}")
            
            return memory_store
    
    def _load_coherence_history(self) -> List[float]:
        """Load coherence history from disk or create a new one."""
        coherence_path = STATE_DIR / f"{self.profile}_coherence_history.json"
        
        if coherence_path.exists():
            try:
                with open(coherence_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading coherence history: {e}")
                return []
        else:
            initial_coherence = self.coherence_calculator.calculate_coherence()
            return [initial_coherence]
    
    def _load_entropy_history(self) -> List[float]:
        """Load entropy history from disk or create a new one."""
        entropy_path = STATE_DIR / f"{self.profile}_entropy_history.json"
        
        if entropy_path.exists():
            try:
                with open(entropy_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading entropy history: {e}")
                return []
        else:
            initial_entropy = self.coherence_calculator.calculate_entropy()
            return [initial_entropy]
    
    def _load_session_log(self) -> List[Dict[str, Any]]:
        """Load session log from disk or create a new one."""
        log_path = STATE_DIR / f"{self.profile}_session_log.json"
        
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading session log: {e}")
                return []
        else:
            return []
    
    def save_state(self):
        """Save the current state to disk."""
        try:
            # Save schema graph
            schema_path = STATE_DIR / f"{self.profile}_schema_graph.pkl"
            with open(schema_path, 'wb') as f:
                pickle.dump(self.schema_graph, f)
            
            # Save memory store
            memory_path = STATE_DIR / f"{self.profile}_memory.json"
            with open(memory_path, 'w') as f:
                json.dump([mem.to_dict() for mem in self.memory_store.get_all_memories()], f, indent=2)
            
            # Save coherence history
            coherence_path = STATE_DIR / f"{self.profile}_coherence_history.json"
            with open(coherence_path, 'w') as f:
                json.dump(self.coherence_history, f, indent=2)
            
            # Save entropy history
            entropy_path = STATE_DIR / f"{self.profile}_entropy_history.json"
            with open(entropy_path, 'w') as f:
                json.dump(self.entropy_history, f, indent=2)
            
            # Save session log
            log_path = STATE_DIR / f"{self.profile}_session_log.json"
            with open(log_path, 'w') as f:
                json.dump(self.session_log, f, indent=2)
            
            logger.info("State saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a new memory to the system and process it.
        
        Args:
            content: The text content of the memory
            metadata: Optional metadata for the memory
        
        Returns:
            Dict containing the processing results
        """
        # Default metadata if none provided
        if metadata is None:
            metadata = {
                "source": "user_input",
                "confidence": 0.9,
                "timestamp": time.time()
            }
        
        # Add memory to store
        memory_id = self.memory_store.add_memory_from_text(content, metadata)
        memory = self.memory_store.get_memory_by_id(memory_id)
        
        # Perform reflection
        reflection_result = self.reflection_engine.reflect_on_memory(memory_id)
        
        # Get relevant memories for context
        relevant_memories = self.memory_store.get_related_memories(memory_id, limit=5)
        relevant_memory_texts = [mem.content for mem in relevant_memories]
        
        # Detect contradictions
        contradictions = self.contradiction_detector.detect_contradictions(memory)
        contradiction_level = sum(c.severity for c in contradictions) / max(1, len(contradictions))
        
        # Update schema if needed
        schema_updated = False
        if contradiction_level > self.config["contradiction_threshold"]:
            self.integration.integrate_memory(memory_id)
            schema_updated = True
            self.schema_version += 1
        
        # Calculate coherence and entropy
        old_coherence = self.get_current_coherence()
        new_coherence = self.coherence_calculator.calculate_coherence()
        self.coherence_history.append(new_coherence)
        
        old_entropy = self.get_current_entropy()
        new_entropy = self.coherence_calculator.calculate_entropy()
        self.entropy_history.append(new_entropy)
        
        # Check for phase transitions
        phase_transition = self._detect_phase_transition(old_entropy, new_entropy)
        
        # Log the event
        log_entry = {
            "timestamp": time.time(),
            "memory_id": memory_id,
            "content": content,
            "contradictions": len(contradictions),
            "contradiction_level": contradiction_level,
            "schema_updated": schema_updated,
            "old_coherence": old_coherence,
            "new_coherence": new_coherence,
            "old_entropy": old_entropy,
            "new_entropy": new_entropy,
            "schema_version": self.schema_version,
            "phase_transition": phase_transition
        }
        self.session_log.append(log_entry)
        
        # Generate enhanced explanations with LLM if available
        enhanced_reflection = None
        change_summary = None
        
        if self.llm_bridge and self.llm_bridge.is_enabled():
            enhanced_reflection = self.llm_bridge.enhance_reflection(
                content, relevant_memory_texts
            )
            
            change_summary = self.llm_bridge.generate_change_summary(
                content, 
                old_coherence, 
                new_coherence,
                old_entropy, 
                new_entropy,
                len(contradictions),
                schema_updated, 
                phase_transition
            )
        
        # Save state
        self.save_state()
        
        result = {
            "memory_id": memory_id,
            "reflection": reflection_result,
            "contradictions": len(contradictions),
            "contradiction_level": contradiction_level,
            "schema_updated": schema_updated,
            "coherence_change": new_coherence - old_coherence,
            "entropy_change": new_entropy - old_entropy,
            "phase_transition": phase_transition
        }
        
        # Add enhanced explanations if available
        if enhanced_reflection:
            result["enhanced_reflection"] = enhanced_reflection
        
        if change_summary:
            result["change_summary"] = change_summary
        
        return result
    
    def get_current_coherence(self) -> float:
        """Get the current coherence score."""
        if not self.coherence_history:
            return self.coherence_calculator.calculate_coherence()
        return self.coherence_history[-1]
    
    def get_current_entropy(self) -> float:
        """Get the current entropy value."""
        if not self.entropy_history:
            return self.coherence_calculator.calculate_entropy()
        return self.entropy_history[-1]
    
    def get_coherence_history(self) -> List[float]:
        """Get the history of coherence values."""
        return self.coherence_history
    
    def get_entropy_history(self) -> List[float]:
        """Get the history of entropy values."""
        return self.entropy_history
    
    def get_schema_graph_data(self) -> Dict[str, Any]:
        """
        Get the schema graph data in a format suitable for visualization.
        
        Returns:
            Dict with nodes and edges for visualization
        """
        nodes = []
        edges = []
        
        # Extract nodes and edges from schema graph
        for node_id, node_data in self.schema_graph.nodes.items():
            nodes.append({
                "id": node_id,
                "label": node_data.get("label", node_id),
                "type": node_data.get("type", "concept"),
                "importance": node_data.get("importance", 0.5)
            })
        
        for edge in self.schema_graph.edges:
            source, target = edge[:2]
            edge_data = self.schema_graph.get_edge_data(source, target)
            edges.append({
                "source": source,
                "target": target,
                "weight": edge_data.get("weight", 0.5),
                "type": edge_data.get("type", "relation")
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def get_latest_memories(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent memories.
        
        Args:
            count: Number of memories to retrieve
        
        Returns:
            List of memory dictionaries
        """
        memories = self.memory_store.get_all_memories()
        memories.sort(key=lambda x: x.metadata.get("timestamp", 0), reverse=True)
        return [mem.to_dict() for mem in memories[:count]]
    
    def process_user_query(self, query: str) -> str:
        """
        Process a direct user query using the LLM if available.
        
        Args:
            query: User's question about the system
        
        Returns:
            Response to the user's query
        """
        if not self.llm_bridge or not self.llm_bridge.is_enabled():
            return "LLM integration is not enabled. Please enable it to use this feature."
        
        # Prepare context for the LLM
        context = {
            "coherence": self.get_current_coherence(),
            "entropy": self.get_current_entropy(),
            "memory_count": len(self.memory_store.get_all_memories()),
            "node_count": len(self.schema_graph.nodes),
            "recent_memories": [
                mem["content"] for mem in self.get_latest_memories(5)
            ]
        }
        
        return self.llm_bridge.process_user_query(query, context)
    
    def _detect_phase_transition(self, old_entropy: float, new_entropy: float) -> bool:
        """
        Detect if a phase transition has occurred based on entropy change.
        
        A phase transition is detected when there's a significant change in entropy.
        
        Args:
            old_entropy: Previous entropy value
            new_entropy: Current entropy value
        
        Returns:
            True if a phase transition is detected, False otherwise
        """
        if abs(new_entropy - old_entropy) > 0.2:  # Threshold for significant change
            return True
        return False
    
    def reset_system(self):
        """Reset the entire system to initial state."""
        self.schema_graph = SchemaGraph()
        self.memory_store = MemoryStore()
        self.integration = MemorySchemaIntegration(self.schema_graph, self.memory_store)
        self.coherence_calculator = CoherenceCalculator(self.schema_graph)
        self.contradiction_detector = ContradictionDetector(self.schema_graph)
        self.reflection_engine = ReflectionEngine(
            memory_store=self.memory_store,
            depth=self.config["reflection_depth"]
        )
        self.knowledge_manager = HierarchicalKnowledgeManager(self.schema_graph)
        
        self.coherence_history = [self.coherence_calculator.calculate_coherence()]
        self.entropy_history = [self.coherence_calculator.calculate_entropy()]
        self.session_log = []
        self.schema_version = 0
        
        self.save_state()
        logger.info("System reset complete")
    
    def export_session_summary(self, format: str = "json") -> str:
        """
        Export a summary of the current session.
        
        Args:
            format: Output format ("json" or "markdown")
        
        Returns:
            String containing the session summary
        """
        summary = {
            "profile": self.profile,
            "schema_version": self.schema_version,
            "final_coherence": self.get_current_coherence(),
            "final_entropy": self.get_current_entropy(),
            "memory_count": len(self.memory_store.get_all_memories()),
            "schema_node_count": len(self.schema_graph.nodes),
            "schema_edge_count": len(self.schema_graph.edges),
            "session_log": self.session_log,
            "coherence_history": self.coherence_history,
            "entropy_history": self.entropy_history
        }
        
        if format == "json":
            return json.dumps(summary, indent=2)
        elif format == "markdown":
            md = f"# ΨC Session Summary\n\n"
            md += f"**Profile:** {self.profile}\n"
            md += f"**Schema Version:** {self.schema_version}\n"
            md += f"**Final Coherence:** {self.get_current_coherence():.4f}\n"
            md += f"**Final Entropy:** {self.get_current_entropy():.4f}\n"
            md += f"**Memory Count:** {len(self.memory_store.get_all_memories())}\n"
            md += f"**Schema Nodes:** {len(self.schema_graph.nodes)}\n"
            md += f"**Schema Edges:** {len(self.schema_graph.edges)}\n\n"
            
            md += "## Session Log\n\n"
            for entry in self.session_log:
                md += f"- **Memory:** {entry['content']}\n"
                md += f"  - Contradictions: {entry['contradictions']}\n"
                md += f"  - Schema Updated: {entry['schema_updated']}\n"
                md += f"  - Coherence: {entry['old_coherence']:.4f} → {entry['new_coherence']:.4f}\n"
                md += f"  - Entropy: {entry['old_entropy']:.4f} → {entry['new_entropy']:.4f}\n"
                if entry['phase_transition']:
                    md += f"  - **Phase Transition Detected!**\n"
                md += "\n"
            
            return md
        else:
            return "Unsupported format"


if __name__ == "__main__":
    # Simple test code
    demo = DemoRunner()
    result = demo.add_memory("The sky is blue.")
    print(f"Memory added. Coherence: {demo.get_current_coherence():.4f}")
    print(f"Contradictions: {result['contradictions']}")
    print(f"Schema updated: {result['schema_updated']}")
    
    # Test LLM integration if available
    if hasattr(demo, 'llm_bridge') and demo.llm_bridge and demo.llm_bridge.is_enabled():
        print("\nLLM integration is enabled.")
        if 'enhanced_reflection' in result:
            print(f"\nEnhanced reflection:\n{result['enhanced_reflection']}")
        if 'change_summary' in result:
            print(f"\nChange summary:\n{result['change_summary']}")
    else:
        print("\nLLM integration is not enabled.") 