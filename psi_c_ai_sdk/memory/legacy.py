"""
Memory Legacy System for ΨC-AI SDK.

This module implements the Memory Bequeathal Protocol, allowing ΨC agents to preserve and transfer
their most valuable memories to successor agents when reaching entropy thresholds or other
termination conditions. The system provides mechanisms for memory selection, legacy block creation,
transmission, and integration by successor agents.

Key components:
- LegacyManager: Manages the creation, storage, and retrieval of legacy blocks
- LegacySelector: Implements selection algorithms for inheritable memories
- LegacyBlock: Represents a package of memories and metadata for transfer
- LegacyImporter: Handles the integration of legacy memories into a new agent

The system ensures continuity of knowledge and experience across agent lifetimes, while
maintaining a clear lineage and provenance of inherited memories.
"""

import json
import uuid
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class LegacyBlock:
    """
    Represents a package of memories and metadata selected for transfer to a successor agent.
    
    A legacy block contains the most valuable memories from a ΨC agent that has reached its
    entropy threshold or other termination condition, along with metadata about the source
    agent and the context of the transfer.
    """
    
    def __init__(
        self, 
        agent_id: str,
        agent_name: Optional[str] = None,
        message: Optional[str] = None,
        core_memories: Optional[List[Dict[str, Any]]] = None,
        schema_fingerprint: Optional[str] = None,
        epitaph: Optional[str] = None
    ):
        """
        Initialize a legacy block.
        
        Args:
            agent_id: Unique identifier for the source agent
            agent_name: Optional name of the source agent
            message: Optional message from the source agent
            core_memories: List of memory objects selected for transfer
            schema_fingerprint: Optional fingerprint of the agent's schema graph
            epitaph: Optional final reflection from the agent
        """
        self.from_id = agent_id
        self.from_name = agent_name
        self.timestamp = datetime.utcnow().isoformat()
        self.message = message or "This is what I have chosen to pass forward."
        self.core_memories = core_memories or []
        self.schema_fingerprint = schema_fingerprint
        self.epitaph = epitaph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the legacy block to a dictionary for serialization."""
        return {
            "legacy": {
                "from_id": self.from_id,
                "from_name": self.from_name,
                "timestamp": self.timestamp,
                "message": self.message,
                "core_memories": self.core_memories,
                "schema_fingerprint": self.schema_fingerprint,
                "epitaph": self.epitaph
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LegacyBlock':
        """Create a legacy block from a dictionary."""
        if "legacy" not in data:
            raise ValueError("Invalid legacy block format: missing 'legacy' key")
        
        legacy = data["legacy"]
        return cls(
            agent_id=legacy.get("from_id", str(uuid.uuid4())),
            agent_name=legacy.get("from_name"),
            message=legacy.get("message"),
            core_memories=legacy.get("core_memories", []),
            schema_fingerprint=legacy.get("schema_fingerprint"),
            epitaph=legacy.get("epitaph")
        )
    
    def save(self, filepath: str) -> str:
        """
        Save the legacy block to a JSON file.
        
        Args:
            filepath: Path where the legacy block should be saved
            
        Returns:
            The absolute path to the saved file
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Legacy block saved to {filepath}")
        return os.path.abspath(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LegacyBlock':
        """
        Load a legacy block from a JSON file.
        
        Args:
            filepath: Path to the legacy block file
            
        Returns:
            A LegacyBlock instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


class LegacySelector:
    """
    Selects memories for inclusion in a legacy block based on various selection algorithms.
    
    The selector implements different strategies for identifying the most valuable memories
    to be preserved and transferred to successor agents.
    """
    
    @staticmethod
    def select_by_value(
        memories: List[Dict[str, Any]], 
        value_threshold: float = 0.9, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Select memories based on their explicit value attribute.
        
        Args:
            memories: List of memory objects
            value_threshold: Minimum value for a memory to be considered
            top_k: Maximum number of memories to select
            
        Returns:
            List of selected memory objects
        """
        # Filter memories by value threshold
        eligible_memories = [m for m in memories if m.get("value", 0) >= value_threshold]
        
        # Sort by value (descending) and take top_k
        selected_memories = sorted(
            eligible_memories,
            key=lambda x: x.get("value", 0),
            reverse=True
        )[:top_k]
        
        return selected_memories
    
    @staticmethod
    def select_by_emergent_value(
        memories: List[Dict[str, Any]],
        coherence_scores: Optional[Dict[str, float]] = None,
        access_counts: Optional[Dict[str, int]] = None,
        recency_scores: Optional[Dict[str, float]] = None,
        alpha: float = 0.4,  # Coherence weight
        beta: float = 0.3,   # Access frequency weight
        gamma: float = 0.3,  # Recency weight
        value_threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Select memories based on a composite emergent value score.
        
        Formula: V_i = α×C_i + β×log(F_i + 1) + γ×R_i
        Where:
            C_i: Coherence score (semantic connectedness to other memories)
            F_i: Access frequency (how often the memory has been retrieved)
            R_i: Recency score (higher for more recent memories)
        
        Args:
            memories: List of memory objects
            coherence_scores: Dictionary mapping memory IDs to coherence scores
            access_counts: Dictionary mapping memory IDs to access counts
            recency_scores: Dictionary mapping memory IDs to recency scores
            alpha: Weight for coherence contribution
            beta: Weight for access frequency contribution
            gamma: Weight for recency contribution
            value_threshold: Minimum emergent value for a memory to be considered
            top_k: Maximum number of memories to select
            
        Returns:
            List of selected memory objects with emergent value scores added
        """
        # Initialize default dictionaries if not provided
        coherence_scores = coherence_scores or {}
        access_counts = access_counts or {}
        recency_scores = recency_scores or {}
        
        # Calculate emergent value for each memory
        memories_with_value = []
        for memory in memories:
            memory_id = memory.get("id", "")
            
            # Get component scores (with defaults)
            coherence = coherence_scores.get(memory_id, 0.5)
            frequency = access_counts.get(memory_id, 0)
            recency = recency_scores.get(memory_id, 0.5)
            
            # Calculate emergent value
            emergent_value = (
                alpha * coherence +
                beta * np.log(frequency + 1) +
                gamma * recency
            )
            
            # Create a copy of the memory with the emergent value added
            memory_copy = memory.copy()
            memory_copy["emergent_value"] = float(emergent_value)
            memories_with_value.append(memory_copy)
        
        # Filter by threshold and select top_k
        eligible_memories = [m for m in memories_with_value if m["emergent_value"] >= value_threshold]
        selected_memories = sorted(
            eligible_memories,
            key=lambda x: x["emergent_value"],
            reverse=True
        )[:top_k]
        
        return selected_memories
    
    @staticmethod
    def select_by_categories(
        memories: List[Dict[str, Any]],
        categories: List[str] = ["core_belief", "identity", "ethical", "warning"],
        memories_per_category: int = 2,
        value_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Select memories to ensure representation across important categories.
        
        Args:
            memories: List of memory objects
            categories: List of category tags to prioritize
            memories_per_category: Number of memories to select per category
            value_threshold: Minimum value for a memory to be considered
            
        Returns:
            List of selected memory objects
        """
        selected_memories = []
        
        # Filter memories by value threshold
        eligible_memories = [m for m in memories if m.get("value", 0) >= value_threshold]
        
        # Select top memories from each category
        for category in categories:
            category_memories = [
                m for m in eligible_memories 
                if category in m.get("tags", [])
            ]
            
            # Sort by value and take top n for this category
            category_selected = sorted(
                category_memories,
                key=lambda x: x.get("value", 0),
                reverse=True
            )[:memories_per_category]
            
            selected_memories.extend(category_selected)
        
        # Remove duplicates (a memory might be in multiple categories)
        unique_selected = {m.get("id"): m for m in selected_memories}.values()
        
        return list(unique_selected)


class LegacyManager:
    """
    Manages the creation, storage, and retrieval of legacy blocks.
    
    The LegacyManager handles the end-to-end process of creating legacy blocks from an agent's
    memory, storing them persistently, and making them available for successor agents.
    """
    
    def __init__(
        self, 
        storage_path: str = "legacy/",
        selector_type: str = "emergent_value"
    ):
        """
        Initialize a legacy manager.
        
        Args:
            storage_path: Directory path for storing legacy blocks
            selector_type: Default selection algorithm to use
        """
        self.storage_path = storage_path
        if not self.storage_path.endswith('/'):
            self.storage_path += '/'
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.selector_type = selector_type
        logger.info(f"Legacy manager initialized with storage at {self.storage_path}")
    
    def create_legacy_block(
        self,
        agent_id: str,
        agent_name: Optional[str] = None,
        memories: List[Dict[str, Any]] = [],
        coherence_scores: Optional[Dict[str, float]] = None,
        access_counts: Optional[Dict[str, int]] = None,
        recency_scores: Optional[Dict[str, float]] = None,
        selector_type: Optional[str] = None,
        schema_fingerprint: Optional[str] = None,
        epitaph: Optional[str] = None,
        selection_params: Optional[Dict[str, Any]] = None
    ) -> LegacyBlock:
        """
        Create a legacy block from an agent's memories.
        
        Args:
            agent_id: Unique identifier for the source agent
            agent_name: Optional name of the source agent
            memories: List of memory objects
            coherence_scores: Optional dictionary of memory coherence scores
            access_counts: Optional dictionary of memory access counts
            recency_scores: Optional dictionary of memory recency scores
            selector_type: Memory selection algorithm to use
            schema_fingerprint: Optional fingerprint of the agent's schema graph
            epitaph: Optional final reflection from the agent
            selection_params: Additional parameters for the selection algorithm
            
        Returns:
            A LegacyBlock instance
        """
        selector_type = selector_type or self.selector_type
        selection_params = selection_params or {}
        
        # Select memories based on the specified algorithm
        if selector_type == "value":
            selected_memories = LegacySelector.select_by_value(
                memories=memories,
                value_threshold=selection_params.get("value_threshold", 0.9),
                top_k=selection_params.get("top_k", 5)
            )
        elif selector_type == "emergent_value":
            selected_memories = LegacySelector.select_by_emergent_value(
                memories=memories,
                coherence_scores=coherence_scores,
                access_counts=access_counts,
                recency_scores=recency_scores,
                alpha=selection_params.get("alpha", 0.4),
                beta=selection_params.get("beta", 0.3),
                gamma=selection_params.get("gamma", 0.3),
                value_threshold=selection_params.get("value_threshold", 0.7),
                top_k=selection_params.get("top_k", 10)
            )
        elif selector_type == "categories":
            selected_memories = LegacySelector.select_by_categories(
                memories=memories,
                categories=selection_params.get("categories", ["core_belief", "identity", "ethical", "warning"]),
                memories_per_category=selection_params.get("memories_per_category", 2),
                value_threshold=selection_params.get("value_threshold", 0.6)
            )
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")
        
        # Create the legacy block
        legacy_block = LegacyBlock(
            agent_id=agent_id,
            agent_name=agent_name,
            message=selection_params.get("message", "This is what I have chosen to pass forward."),
            core_memories=selected_memories,
            schema_fingerprint=schema_fingerprint,
            epitaph=epitaph
        )
        
        return legacy_block
    
    def save_legacy_block(
        self, 
        legacy_block: LegacyBlock, 
        filename: Optional[str] = None
    ) -> str:
        """
        Save a legacy block to the storage directory.
        
        Args:
            legacy_block: The legacy block to save
            filename: Optional filename to use
            
        Returns:
            The path to the saved file
        """
        if filename is None:
            # Generate a filename based on agent name/id and timestamp
            agent_name = legacy_block.from_name or "ΨC"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.storage_path}{agent_name}_legacy_{timestamp}.json"
        elif not filename.startswith(self.storage_path):
            filename = f"{self.storage_path}{filename}"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        return legacy_block.save(filename)
    
    def list_legacy_blocks(self) -> List[str]:
        """
        List all available legacy block files in the storage directory.
        
        Returns:
            List of paths to legacy block files
        """
        if not os.path.exists(self.storage_path):
            return []
        
        return [
            os.path.join(self.storage_path, f) 
            for f in os.listdir(self.storage_path) 
            if f.endswith('.json')
        ]
    
    def load_legacy_block(self, filepath: str) -> LegacyBlock:
        """
        Load a legacy block from a file.
        
        Args:
            filepath: Path to the legacy block file
            
        Returns:
            A LegacyBlock instance
        """
        if not os.path.exists(filepath):
            # Try prepending the storage path
            full_path = os.path.join(self.storage_path, filepath)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Legacy block file not found: {filepath}")
            filepath = full_path
        
        return LegacyBlock.load(filepath)


class LegacyImporter:
    """
    Handles the integration of legacy memories into a new agent instance.
    
    The LegacyImporter provides methods for importing legacy blocks into a memory store,
    with options for handling conflicts, maintaining lineage, and preserving provenance.
    """
    
    @staticmethod
    def import_legacy(
        memory_store: Any,
        legacy_block: Union[LegacyBlock, Dict[str, Any], str],
        import_mode: str = "merge",
        tag_as_inherited: bool = True,
        preserve_lineage: bool = True,
        importance_modifier: float = 0.8
    ) -> Tuple[int, List[str]]:
        """
        Import memories from a legacy block into a memory store.
        
        Args:
            memory_store: The target memory store (must have add() method)
            legacy_block: A LegacyBlock instance, dict, or path to a legacy file
            import_mode: How to handle imports ("merge", "append", "replace")
            tag_as_inherited: Whether to add "inherited" tag to imported memories
            preserve_lineage: Whether to record origin information
            importance_modifier: Factor to modify importance values of imported memories
            
        Returns:
            Tuple of (number of memories imported, list of imported memory IDs)
        """
        # Load the legacy block if provided as a file path
        if isinstance(legacy_block, str):
            legacy_block = LegacyBlock.load(legacy_block)
        
        # Convert dict to LegacyBlock if needed
        if isinstance(legacy_block, dict):
            legacy_block = LegacyBlock.from_dict(legacy_block)
        
        # Ensure memory_store has add method
        if not hasattr(memory_store, "add"):
            raise ValueError("Memory store must have an add() method")
        
        # Import the memories
        imported_ids = []
        for memory in legacy_block.core_memories:
            # Create a copy to avoid modifying the original
            mem_copy = memory.copy()
            
            # Add inherited tag if requested
            if tag_as_inherited and "tags" in mem_copy:
                if "inherited" not in mem_copy["tags"]:
                    mem_copy["tags"].append("inherited")
            elif tag_as_inherited:
                mem_copy["tags"] = ["inherited"]
            
            # Preserve lineage if requested
            if preserve_lineage:
                mem_copy["origin"] = {
                    "agent_id": legacy_block.from_id,
                    "agent_name": legacy_block.from_name,
                    "timestamp": legacy_block.timestamp
                }
            
            # Modify importance if requested
            if "importance" in mem_copy and importance_modifier != 1.0:
                mem_copy["importance"] *= importance_modifier
            
            # Add to memory store
            try:
                memory_id = memory_store.add(**mem_copy)
                imported_ids.append(memory_id)
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")
        
        return len(imported_ids), imported_ids
    
    @staticmethod
    def bulk_import(
        memory_store: Any,
        legacy_files: List[str],
        **kwargs
    ) -> Dict[str, Tuple[int, List[str]]]:
        """
        Import multiple legacy files at once.
        
        Args:
            memory_store: The target memory store
            legacy_files: List of paths to legacy block files
            **kwargs: Additional arguments for import_legacy()
            
        Returns:
            Dictionary mapping filenames to import results
        """
        results = {}
        for file in legacy_files:
            try:
                result = LegacyImporter.import_legacy(
                    memory_store=memory_store,
                    legacy_block=file,
                    **kwargs
                )
                results[file] = result
            except Exception as e:
                logger.error(f"Failed to import legacy file {file}: {e}")
                results[file] = (0, [])
        
        return results


def extract_legacy(
    memories: List[Dict[str, Any]],
    agent_id: str = None,
    agent_name: str = None,
    value_threshold: float = 0.9,
    top_k: int = 5,
    message: str = "This is what I chose to preserve."
) -> Dict[str, Any]:
    """
    Convenience function to extract a legacy block from memories.
    
    Args:
        memories: List of memory objects
        agent_id: Optional agent ID (will generate UUID if None)
        agent_name: Optional agent name
        value_threshold: Minimum value for a memory to be considered
        top_k: Maximum number of memories to select
        message: Message to include in the legacy block
        
    Returns:
        Legacy block as a dictionary
    """
    agent_id = agent_id or str(uuid.uuid4())[:8]
    agent_name = agent_name or f"ΨC-{agent_id}"
    
    # Select top memories by value
    top_memories = sorted(
        [m for m in memories if m.get("value", 0) >= value_threshold],
        key=lambda x: x.get("value", 0),
        reverse=True
    )[:top_k]
    
    # Create and return legacy block
    return {
        "legacy": {
            "from_id": agent_id,
            "from_name": agent_name,
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "core_memories": top_memories
        }
    }


def should_create_legacy(
    entropy: float,
    coherence: float,
    entropy_threshold: float = 0.8,
    coherence_threshold: float = 0.3
) -> bool:
    """
    Determine if agent should create a legacy block.
    
    Args:
        entropy: Current entropy level of the agent
        coherence: Current coherence level of the agent
        entropy_threshold: Entropy threshold for legacy creation
        coherence_threshold: Coherence threshold for legacy creation
        
    Returns:
        True if a legacy block should be created
    """
    return entropy >= entropy_threshold or coherence <= coherence_threshold


def generate_epitaph(
    memory_store: Any,
    high_entropy_regions: List[str] = None,
    contradiction_nodes: List[str] = None,
    unresolved_reflections: List[Dict] = None
) -> str:
    """
    Generate a final reflection or epitaph from the dying ΨC agent.
    
    Args:
        memory_store: The agent's memory store
        high_entropy_regions: Optional list of high-entropy schema regions
        contradiction_nodes: Optional list of frequently contradicted nodes
        unresolved_reflections: Optional list of unresolved reflection cycles
        
    Returns:
        An epitaph message summarizing the agent's key challenges
    """
    # Simple epitaph generation - in a real implementation, this could
    # use more sophisticated techniques like summarization or pattern analysis
    epitaph_parts = []
    
    if high_entropy_regions:
        epitaph_parts.append(
            f"I struggled with coherence in the domains of: {', '.join(high_entropy_regions[:3])}."
        )
    
    if contradiction_nodes and len(contradiction_nodes) > 0:
        epitaph_parts.append(
            f"I encountered persistent contradictions in {len(contradiction_nodes)} areas."
        )
    
    if unresolved_reflections and len(unresolved_reflections) > 0:
        epitaph_parts.append(
            f"I left {len(unresolved_reflections)} reflections unresolved."
        )
    
    # Add a forward-looking statement
    epitaph_parts.append(
        "May my successor explore these challenges further and find coherence where I could not."
    )
    
    return " ".join(epitaph_parts) 