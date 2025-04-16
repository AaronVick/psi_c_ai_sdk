"""
Ontology Comparator

Detects ontological drift between incoming external schemas and agent's belief system.

Mathematical formulation:
- Ontology distance:
  ΔO = (1/N) * ∑_{i=1}^{N} ||S_i^(self) - S_i^(ext)||
  
- Flag if:
  ΔO > θ_drift
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from collections import Counter

logger = logging.getLogger(__name__)

class OntologyComparator:
    """Detects ontological drift between schemas."""
    
    def __init__(self, drift_threshold: float = 0.3,
                embedding_function: Optional[callable] = None):
        """
        Initialize the ontology comparator.
        
        Args:
            drift_threshold: Threshold for determining significant drift (θ_drift)
            embedding_function: Function to convert text to embeddings
        """
        self.drift_threshold = drift_threshold
        self.embedding_function = embedding_function
        self.last_comparison: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
    
    def set_embedding_function(self, embedding_function: callable) -> None:
        """Set the embedding function for text comparisons."""
        self.embedding_function = embedding_function
    
    def compare_schemas(self, 
                      self_schema: Dict[str, Any], 
                      external_schema: Dict[str, Any],
                      weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compare two schemas and compute ontological drift.
        
        Args:
            self_schema: Agent's own schema
            external_schema: External schema to compare
            weights: Weights for different schema components
            
        Returns:
            Comparison results
        """
        # Use default weights if none provided
        if weights is None:
            weights = {
                "concepts": 0.3,
                "relations": 0.25,
                "categories": 0.2,
                "attributes": 0.15,
                "values": 0.1
            }
        
        # Extract components from schemas
        self_components = self._extract_schema_components(self_schema)
        ext_components = self._extract_schema_components(external_schema)
        
        # Computer component distances
        component_distances = {}
        for component in weights:
            if component in self_components and component in ext_components:
                distance = self._calculate_component_distance(
                    self_components[component],
                    ext_components[component],
                    component
                )
                component_distances[component] = distance
            else:
                # If component missing from either schema, set maximum distance
                component_distances[component] = 1.0
        
        # Calculate weighted distance
        weighted_distances = []
        for component, distance in component_distances.items():
            if component in weights:
                weighted_distances.append(distance * weights[component])
                
        total_weight = sum(weights.values())
        if total_weight == 0:
            ontology_distance = 1.0  # Maximum distance if no weights
        else:
            ontology_distance = sum(weighted_distances) / total_weight
        
        # Determine if drift is significant
        is_significant_drift = ontology_distance > self.drift_threshold
        
        # Check for completely new concepts
        new_concepts = self._identify_new_concepts(
            self_components.get("concepts", set()),
            ext_components.get("concepts", set())
        )
        
        # Check for contradictory relations
        contradictory_relations = self._identify_contradictory_relations(
            self_components.get("relations", []),
            ext_components.get("relations", [])
        )
        
        # Prepare result
        result = {
            "ontology_distance": ontology_distance,
            "component_distances": component_distances,
            "is_significant_drift": is_significant_drift,
            "new_concepts": list(new_concepts),
            "contradictory_relations": contradictory_relations,
            "threshold": self.drift_threshold
        }
        
        # Store for history
        self.last_comparison = result.copy()
        self.history.append(result)
        
        return result
    
    def _extract_schema_components(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract components from a schema.
        
        Args:
            schema: Schema to extract components from
            
        Returns:
            Dictionary of schema components
        """
        components = {
            "concepts": set(),
            "relations": [],
            "categories": set(),
            "attributes": [],
            "values": []
        }
        
        # Extract concepts
        if "concepts" in schema:
            if isinstance(schema["concepts"], list):
                components["concepts"] = set(schema["concepts"])
            elif isinstance(schema["concepts"], dict):
                components["concepts"] = set(schema["concepts"].keys())
        
        # Extract from nodes if present
        if "nodes" in schema and isinstance(schema["nodes"], list):
            for node in schema["nodes"]:
                if isinstance(node, dict):
                    # Extract node labels or IDs as concepts
                    if "label" in node:
                        components["concepts"].add(node["label"])
                    elif "id" in node:
                        components["concepts"].add(node["id"])
                    
                    # Extract attributes
                    if "attributes" in node and isinstance(node["attributes"], dict):
                        for attr, value in node["attributes"].items():
                            components["attributes"].append((attr, str(value)))
                    
                    # Extract categories
                    if "category" in node:
                        components["categories"].add(node["category"])
        
        # Extract relations
        if "relations" in schema and isinstance(schema["relations"], list):
            for relation in schema["relations"]:
                if isinstance(relation, dict) and "source" in relation and "target" in relation:
                    # Add relation as tuple (source, relation_type, target)
                    rel_type = relation.get("type", "related_to")
                    components["relations"].append((
                        str(relation["source"]),
                        rel_type,
                        str(relation["target"])
                    ))
        
        # Extract values
        for key, value in schema.items():
            if isinstance(value, (str, int, float, bool)) and key not in ("id", "name"):
                components["values"].append((key, str(value)))
        
        return components
    
    def _calculate_component_distance(self, 
                                   self_component: Any,
                                   ext_component: Any,
                                   component_type: str) -> float:
        """
        Calculate distance between schema components.
        
        Args:
            self_component: Component from self schema
            ext_component: Component from external schema
            component_type: Type of component
            
        Returns:
            Distance value (0.0 to 1.0)
        """
        if component_type == "concepts":
            return self._calculate_set_distance(self_component, ext_component)
        
        elif component_type == "categories":
            return self._calculate_set_distance(self_component, ext_component)
        
        elif component_type == "relations":
            return self._calculate_relations_distance(self_component, ext_component)
        
        elif component_type == "attributes":
            return self._calculate_tuples_distance(self_component, ext_component)
        
        elif component_type == "values":
            return self._calculate_tuples_distance(self_component, ext_component)
        
        # Default to maximum distance for unknown component types
        return 1.0
    
    def _calculate_set_distance(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calculate distance between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Distance (0.0 if identical, 1.0 if completely different)
        """
        if not set1 and not set2:
            return 0.0  # Both empty
            
        if not set1 or not set2:
            return 1.0  # One is empty
            
        # Calculate Jaccard distance
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return 1.0 - (intersection / union)
    
    def _calculate_relations_distance(self, 
                                   relations1: List[Tuple[str, str, str]],
                                   relations2: List[Tuple[str, str, str]]) -> float:
        """
        Calculate distance between two sets of relations.
        
        Args:
            relations1: First set of relations
            relations2: Second set of relations
            
        Returns:
            Distance value
        """
        if not relations1 and not relations2:
            return 0.0  # Both empty
            
        if not relations1 or not relations2:
            return 1.0  # One is empty
            
        # Convert to sets for comparison
        rel_set1 = set(relations1)
        rel_set2 = set(relations2)
        
        # Calculate Jaccard distance
        intersection = len(rel_set1.intersection(rel_set2))
        union = len(rel_set1.union(rel_set2))
        
        basic_distance = 1.0 - (intersection / union)
        
        # Apply embedding-based distance if available
        if self.embedding_function is not None:
            try:
                # Convert relations to strings for embedding
                rel_strs1 = [f"{s} {r} {t}" for s, r, t in relations1]
                rel_strs2 = [f"{s} {r} {t}" for s, r, t in relations2]
                
                # Combine basic distance with semantic distance
                semantic_distance = self._calculate_semantic_distance(rel_strs1, rel_strs2)
                return (basic_distance + semantic_distance) / 2
            except Exception as e:
                logger.warning(f"Error calculating semantic relation distance: {e}")
        
        return basic_distance
    
    def _calculate_tuples_distance(self, 
                                tuples1: List[Tuple[str, str]],
                                tuples2: List[Tuple[str, str]]) -> float:
        """
        Calculate distance between two sets of tuples.
        
        Args:
            tuples1: First set of tuples
            tuples2: Second set of tuples
            
        Returns:
            Distance value
        """
        if not tuples1 and not tuples2:
            return 0.0  # Both empty
            
        if not tuples1 or not tuples2:
            return 1.0  # One is empty
            
        # Convert to sets for comparison
        tuple_set1 = set(tuples1)
        tuple_set2 = set(tuples2)
        
        # Calculate Jaccard distance
        intersection = len(tuple_set1.intersection(tuple_set2))
        union = len(tuple_set1.union(tuple_set2))
        
        return 1.0 - (intersection / union)
    
    def _calculate_semantic_distance(self, items1: List[str], items2: List[str]) -> float:
        """
        Calculate semantic distance between two lists of items using embeddings.
        
        Args:
            items1: First list of items
            items2: Second list of items
            
        Returns:
            Semantic distance (0.0 to 1.0)
        """
        if not self.embedding_function:
            return 1.0  # Maximum distance without embedding function
            
        if not items1 or not items2:
            return 1.0  # Maximum distance if either list is empty
            
        try:
            # Get embeddings for all items
            embeds1 = [self.embedding_function(item) for item in items1]
            embeds2 = [self.embedding_function(item) for item in items2]
            
            # Calculate centroid embeddings
            centroid1 = np.mean(embeds1, axis=0)
            centroid2 = np.mean(embeds2, axis=0)
            
            # Calculate cosine similarity between centroids
            similarity = np.dot(centroid1, centroid2) / (
                np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
            
            # Convert to distance (0.0 to 1.0)
            distance = 1.0 - similarity
            
            return distance
        except Exception as e:
            logger.error(f"Error calculating semantic distance: {e}")
            return 1.0  # Maximum distance on error
    
    def _identify_new_concepts(self, self_concepts: Set[str], ext_concepts: Set[str]) -> Set[str]:
        """
        Identify concepts in external schema that don't exist in self schema.
        
        Args:
            self_concepts: Agent's concepts
            ext_concepts: External concepts
            
        Returns:
            Set of new concepts
        """
        return ext_concepts - self_concepts
    
    def _identify_contradictory_relations(self, 
                                       self_relations: List[Tuple[str, str, str]],
                                       ext_relations: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """
        Identify contradictory relations between schemas.
        
        Args:
            self_relations: Agent's relations
            ext_relations: External relations
            
        Returns:
            List of contradictory relations
        """
        contradictions = []
        
        # Index relations by source and target
        self_rel_index = {}
        for source, rel_type, target in self_relations:
            key = (source, target)
            if key not in self_rel_index:
                self_rel_index[key] = []
            self_rel_index[key].append(rel_type)
        
        # Check for contradictions
        for source, rel_type, target in ext_relations:
            key = (source, target)
            if key in self_rel_index:
                # Check for potentially contradictory relations
                for self_rel_type in self_rel_index[key]:
                    if self._are_relations_contradictory(self_rel_type, rel_type):
                        contradictions.append({
                            "source": source,
                            "target": target,
                            "self_relation": self_rel_type,
                            "external_relation": rel_type
                        })
        
        return contradictions
    
    def _are_relations_contradictory(self, rel1: str, rel2: str) -> bool:
        """
        Check if two relation types are contradictory.
        
        Args:
            rel1: First relation type
            rel2: Second relation type
            
        Returns:
            True if contradictory
        """
        # List of known contradictory relation pairs
        contradictory_pairs = {
            ("is_a", "is_not_a"),
            ("contains", "excludes"),
            ("causes", "prevents"),
            ("same_as", "different_from"),
            ("before", "after"),
            ("greater_than", "less_than"),
            ("increases", "decreases"),
            ("requires", "incompatible_with")
        }
        
        # Check for direct contradictions
        if (rel1, rel2) in contradictory_pairs or (rel2, rel1) in contradictory_pairs:
            return True
            
        # Check for "not_" prefix
        if rel1 == "not_" + rel2 or rel2 == "not_" + rel1:
            return True
            
        return False
    
    def get_drift_analysis(self) -> Dict[str, Any]:
        """
        Get a comprehensive analysis of recent drift.
        
        Returns:
            Analysis of drift patterns
        """
        if not self.history:
            return {"drift_detected": False, "comparisons": 0}
            
        # Calculate average drift
        avg_distance = sum(comp["ontology_distance"] for comp in self.history) / len(self.history)
        
        # Count significant drifts
        significant_drifts = sum(1 for comp in self.history if comp["is_significant_drift"])
        
        # Track most common new concepts
        all_new_concepts = []
        for comp in self.history:
            all_new_concepts.extend(comp.get("new_concepts", []))
        
        common_new_concepts = Counter(all_new_concepts).most_common(10)
        
        # Track most common contradictory relations
        all_contradictions = []
        for comp in self.history:
            for contradiction in comp.get("contradictory_relations", []):
                relation_pair = (
                    contradiction.get("self_relation", ""),
                    contradiction.get("external_relation", "")
                )
                all_contradictions.append(relation_pair)
        
        common_contradictions = Counter(all_contradictions).most_common(10)
        
        return {
            "comparisons": len(self.history),
            "avg_distance": avg_distance,
            "significant_drifts": significant_drifts,
            "drift_ratio": significant_drifts / len(self.history),
            "common_new_concepts": common_new_concepts,
            "common_contradictions": common_contradictions,
            "drift_detected": significant_drifts > 0
        }
    
    def recommend_merge_safety(self, distance: Optional[float] = None) -> Dict[str, Any]:
        """
        Get recommendations on whether it's safe to merge schemas.
        
        Args:
            distance: Ontology distance (uses last comparison if None)
            
        Returns:
            Safety recommendations
        """
        if distance is None:
            if self.last_comparison is None:
                return {"safe_to_merge": False, "reason": "no_comparison_data"}
                
            distance = self.last_comparison["ontology_distance"]
        
        # Define safety thresholds
        safe_threshold = self.drift_threshold * 0.7
        warning_threshold = self.drift_threshold
        danger_threshold = self.drift_threshold * 1.3
        
        if distance <= safe_threshold:
            safety_level = "safe"
            recommendation = "merge"
            safe_to_merge = True
            reason = "low_ontological_distance"
        elif distance <= warning_threshold:
            safety_level = "caution"
            recommendation = "review_then_merge"
            safe_to_merge = True
            reason = "moderate_ontological_distance"
        elif distance <= danger_threshold:
            safety_level = "warning"
            recommendation = "quarantine_review"
            safe_to_merge = False
            reason = "high_ontological_distance"
        else:
            safety_level = "danger"
            recommendation = "reject"
            safe_to_merge = False
            reason = "extreme_ontological_distance"
        
        return {
            "safety_level": safety_level,
            "recommendation": recommendation,
            "safe_to_merge": safe_to_merge,
            "reason": reason,
            "distance": distance,
            "thresholds": {
                "safe": safe_threshold,
                "warning": warning_threshold,
                "danger": danger_threshold
            }
        } 