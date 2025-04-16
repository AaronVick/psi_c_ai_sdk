"""
Recursive Depth Limiter: Controls recursive depth in self-modeling systems.

This module implements the bounded recursive depth control mechanism as
described in the ΨC-AI SDK design. It prevents runaway recursion and enables
graceful scaling of computational complexity.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import uuid


class RecursiveModel:
    """
    Represents a self-model with a specific depth level.
    
    Attributes:
        model_id: Unique identifier for this model
        depth: Recursion depth level (0 = base model)
        parent_id: ID of the parent model (None for base model)
        content: The actual model content
        metadata: Additional metadata
    """
    
    def __init__(
        self,
        content: Any,
        depth: int = 0,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a recursive model.
        
        Args:
            content: The model content
            depth: Recursion depth level
            parent_id: ID of the parent model
            metadata: Additional metadata
        """
        self.model_id = str(uuid.uuid4())
        self.depth = depth
        self.parent_id = parent_id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = metadata.get("created_at", None)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dictionary representation of the model
        """
        return {
            "model_id": self.model_id,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "content": self.content if not isinstance(self.content, object) else str(self.content),
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecursiveModel":
        """
        Create a model from a dictionary.
        
        Args:
            data: Dictionary containing model data
            
        Returns:
            RecursiveModel object
        """
        model = cls(
            content=data["content"],
            depth=data["depth"],
            parent_id=data["parent_id"],
            metadata=data["metadata"]
        )
        model.model_id = data["model_id"]
        model.created_at = data.get("created_at", None)
        return model


class RecursiveDepthLimiter:
    """
    Controls and limits the recursive depth of self-models.
    
    The RecursiveDepthLimiter implements bounded recursion by:
    1. Tracking the depth of each self-model
    2. Enforcing a maximum recursion depth
    3. Providing roll-up logic for models beyond max depth
    """
    
    def __init__(self, max_depth: int = 3):
        """
        Initialize the recursive depth limiter.
        
        Args:
            max_depth: Maximum allowed recursion depth
        """
        self.max_depth = max_depth
        self.models: Dict[str, RecursiveModel] = {}
        
    def create_model(
        self,
        content: Any,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecursiveModel:
        """
        Create a new recursive model.
        
        If a parent is specified, the new model's depth will be parent.depth + 1.
        If the depth would exceed max_depth, the model will be rolled up.
        
        Args:
            content: The model content
            parent_id: Optional ID of the parent model
            metadata: Additional metadata
            
        Returns:
            The created RecursiveModel
        """
        # Determine depth
        depth = 0
        if parent_id is not None:
            if parent_id not in self.models:
                raise ValueError(f"Parent model {parent_id} not found")
            depth = self.models[parent_id].depth + 1
        
        # Check max depth
        if depth > self.max_depth:
            # Roll up the model (return the parent's parent)
            ancestor_id = self.find_ancestor_at_depth(parent_id, self.max_depth)
            if ancestor_id:
                return self.models[ancestor_id]
            raise ValueError(f"Cannot roll up model: no ancestor at depth {self.max_depth}")
        
        # Create the model
        model = RecursiveModel(content, depth, parent_id, metadata)
        self.models[model.model_id] = model
        
        return model
    
    def find_ancestor_at_depth(self, model_id: str, target_depth: int) -> Optional[str]:
        """
        Find an ancestor of the given model at the specified depth.
        
        Args:
            model_id: ID of the model to start from
            target_depth: Depth of the desired ancestor
            
        Returns:
            ID of the ancestor model, or None if not found
        """
        if model_id not in self.models:
            return None
        
        current = self.models[model_id]
        
        while current and current.depth > target_depth:
            if current.parent_id is None:
                return None
            current = self.models.get(current.parent_id)
        
        return current.model_id if current and current.depth == target_depth else None
    
    def get_model(self, model_id: str) -> Optional[RecursiveModel]:
        """
        Get a model by its ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The RecursiveModel if found, None otherwise
        """
        return self.models.get(model_id)
    
    def get_models_at_depth(self, depth: int) -> List[RecursiveModel]:
        """
        Get all models at the specified depth.
        
        Args:
            depth: Depth level to retrieve
            
        Returns:
            List of RecursiveModel objects at the specified depth
        """
        return [model for model in self.models.values() if model.depth == depth]
    
    def get_model_lineage(self, model_id: str) -> List[RecursiveModel]:
        """
        Get the lineage of a model (all ancestors up to the root).
        
        Args:
            model_id: ID of the model to get lineage for
            
        Returns:
            List of models in the lineage, ordered from root to the given model
        """
        if model_id not in self.models:
            return []
        
        lineage = []
        current_id = model_id
        
        while current_id is not None:
            current = self.models.get(current_id)
            if current is None:
                break
            
            lineage.append(current)
            current_id = current.parent_id
        
        return list(reversed(lineage))
    
    def prune_models(self, keep_ids: Set[str]) -> int:
        """
        Prune models, keeping only those with IDs in the given set.
        
        Args:
            keep_ids: Set of model IDs to keep
            
        Returns:
            Number of models pruned
        """
        to_remove = [mid for mid in self.models if mid not in keep_ids]
        for mid in to_remove:
            del self.models[mid]
        
        return len(to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the models.
        
        Returns:
            Dictionary with statistics
        """
        depth_counts = {}
        for model in self.models.values():
            depth_counts[model.depth] = depth_counts.get(model.depth, 0) + 1
        
        return {
            "total_models": len(self.models),
            "max_depth_configured": self.max_depth,
            "max_depth_actual": max(depth_counts.keys()) if depth_counts else 0,
            "depth_distribution": depth_counts
        }
    
    def to_graph(self) -> Dict[str, Any]:
        """
        Convert the model hierarchy to a graph representation.
        
        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes = []
        edges = []
        
        for mid, model in self.models.items():
            nodes.append({
                "id": mid,
                "depth": model.depth,
                "label": f"Model {mid[:8]} (D{model.depth})"
            })
            
            if model.parent_id:
                edges.append({
                    "source": model.parent_id,
                    "target": mid
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        } 