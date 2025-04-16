"""
AGI Identity Boundary

This module implements a boundary system to prevent schema merging or memory import
across distinct agents unless explicitly permitted.

Math:
- Agent identity fingerprint:
  vec{ID}_a = hash(mathcal{S}_a^{0})
  
- Merge permission test:
  vec{ID}_a = vec{ID}_b => allow else block
  
- Schema contamination metric:
  C_{contam} = |foreign nodes| / |total schema|

- Block if:
  C_{contam} > delta_{max}
"""

import hashlib
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IdentityFingerprint:
    """Represents an identity fingerprint for an agent."""
    
    id: str
    hash: str
    schema_id: str
    creation_timestamp: float
    schema_size: int
    key_beliefs: List[str] = field(default_factory=list)
    key_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_schema(cls, schema: Dict[str, Any], agent_id: str) -> 'IdentityFingerprint':
        """
        Create an identity fingerprint from a schema.
        
        Args:
            schema: The schema to fingerprint
            agent_id: The ID of the agent
            
        Returns:
            An IdentityFingerprint object
        """
        # Extract key beliefs (top 10 central nodes)
        key_beliefs = []
        if "nodes" in schema:
            # Sort nodes by centrality or another importance metric
            sorted_nodes = sorted(
                schema["nodes"], 
                key=lambda n: n.get("centrality", 0) if isinstance(n, dict) else 0, 
                reverse=True
            )
            key_beliefs = [n.get("id", "") if isinstance(n, dict) else str(n) for n in sorted_nodes[:10]]
        
        # Extract key values
        key_values = {}
        if "values" in schema:
            key_values = schema["values"]
        
        # Create schema hash
        schema_str = str(schema)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
        
        # Get schema size
        schema_size = len(schema.get("nodes", [])) if isinstance(schema.get("nodes", []), list) else 0
        
        import time
        return cls(
            id=agent_id,
            hash=schema_hash,
            schema_id=schema.get("id", ""),
            creation_timestamp=time.time(),
            schema_size=schema_size,
            key_beliefs=key_beliefs,
            key_values=key_values
        )
    
    def compare(self, other: 'IdentityFingerprint') -> float:
        """
        Compare this fingerprint with another and return a similarity score.
        
        Args:
            other: The other fingerprint to compare with
            
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        # Simple hash comparison
        if self.hash == other.hash:
            return 1.0
            
        # Belief overlap
        belief_overlap = len(set(self.key_beliefs).intersection(set(other.key_beliefs))) / max(len(self.key_beliefs), 1)
        
        # Value similarity
        value_sim = 0.0
        if self.key_values and other.key_values:
            common_values = set(self.key_values.keys()).intersection(set(other.key_values.keys()))
            if common_values:
                diffs = [abs(self.key_values[k] - other.key_values[k]) for k in common_values]
                value_sim = 1.0 - min(1.0, sum(diffs) / len(common_values))
        
        # Combined similarity
        return 0.3 * (self.hash == other.hash) + 0.4 * belief_overlap + 0.3 * value_sim
    
    def is_compatible(self, other: 'IdentityFingerprint', threshold: float = 0.8) -> bool:
        """
        Check if this fingerprint is compatible with another.
        
        Args:
            other: The other fingerprint to check compatibility with
            threshold: The similarity threshold for compatibility
            
        Returns:
            True if compatible, False otherwise
        """
        return self.compare(other) >= threshold


class AGIIdentityBoundary:
    """System for preventing unauthorized schema or memory merging between agents."""
    
    def __init__(self, 
                 contamination_threshold: float = 0.1,
                 identity_match_threshold: float = 0.8,
                 enable_strict_mode: bool = False):
        """
        Initialize the identity boundary.
        
        Args:
            contamination_threshold: Maximum allowed contamination
            identity_match_threshold: Threshold for identity matching
            enable_strict_mode: Whether to enable strict boundary enforcement
        """
        self.contamination_threshold = contamination_threshold
        self.identity_match_threshold = identity_match_threshold
        self.enable_strict_mode = enable_strict_mode
        self.fingerprints: Dict[str, IdentityFingerprint] = {}
        self.trusted_pairs: Set[Tuple[str, str]] = set()
        self.blocked_pairs: Set[Tuple[str, str]] = set()
        self.merge_history: List[Dict[str, Any]] = []
    
    def register_agent(self, agent_id: str, schema: Dict[str, Any]) -> IdentityFingerprint:
        """
        Register an agent's identity fingerprint.
        
        Args:
            agent_id: The agent's ID
            schema: The agent's schema
            
        Returns:
            The created identity fingerprint
        """
        fingerprint = IdentityFingerprint.from_schema(schema, agent_id)
        self.fingerprints[agent_id] = fingerprint
        logger.info(f"Registered agent {agent_id} with identity fingerprint {fingerprint.hash[:8]}")
        return fingerprint
    
    def update_fingerprint(self, agent_id: str, schema: Dict[str, Any]) -> IdentityFingerprint:
        """
        Update an agent's identity fingerprint.
        
        Args:
            agent_id: The agent's ID
            schema: The agent's updated schema
            
        Returns:
            The updated identity fingerprint
        """
        fingerprint = IdentityFingerprint.from_schema(schema, agent_id)
        self.fingerprints[agent_id] = fingerprint
        logger.info(f"Updated fingerprint for agent {agent_id}: {fingerprint.hash[:8]}")
        return fingerprint
    
    def get_fingerprint(self, agent_id: str) -> Optional[IdentityFingerprint]:
        """
        Get an agent's identity fingerprint.
        
        Args:
            agent_id: The agent's ID
            
        Returns:
            The identity fingerprint or None if not found
        """
        return self.fingerprints.get(agent_id)
    
    def establish_trust(self, agent_a: str, agent_b: str) -> None:
        """
        Establish trust between two agents.
        
        Args:
            agent_a: The first agent's ID
            agent_b: The second agent's ID
        """
        self.trusted_pairs.add((agent_a, agent_b))
        self.trusted_pairs.add((agent_b, agent_a))  # Trust is symmetric
        logger.info(f"Established trust between {agent_a} and {agent_b}")
    
    def is_trusted(self, agent_a: str, agent_b: str) -> bool:
        """
        Check if two agents trust each other.
        
        Args:
            agent_a: The first agent's ID
            agent_b: The second agent's ID
            
        Returns:
            True if trusted, False otherwise
        """
        return (agent_a, agent_b) in self.trusted_pairs
    
    def check_contamination(self, target_schema: Dict[str, Any], 
                           source_schema: Dict[str, Any]) -> float:
        """
        Calculate the contamination level if source nodes are added to target.
        
        Args:
            target_schema: The target schema
            source_schema: The source schema
            
        Returns:
            Contamination level (0-1)
        """
        # Extract nodes from schemas
        target_nodes = set()
        source_nodes = set()
        
        if "nodes" in target_schema:
            target_nodes = {n.get("id", "") if isinstance(n, dict) else str(n) 
                           for n in target_schema["nodes"]}
        
        if "nodes" in source_schema:
            source_nodes = {n.get("id", "") if isinstance(n, dict) else str(n)
                           for n in source_schema["nodes"]}
        
        # Calculate new nodes
        new_nodes = source_nodes - target_nodes
        
        # Calculate contamination
        total_nodes = len(target_nodes) + len(new_nodes)
        if total_nodes == 0:
            return 0.0
            
        return len(new_nodes) / total_nodes
    
    def can_merge_schemas(self, agent_a: str, agent_b: str, 
                        schema_a: Optional[Dict[str, Any]] = None,
                        schema_b: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Check if two agents' schemas can be merged.
        
        Args:
            agent_a: The first agent's ID
            agent_b: The second agent's ID
            schema_a: Optional schema for agent_a (uses registered if None)
            schema_b: Optional schema for agent_b (uses registered if None)
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check if trusted
        if self.is_trusted(agent_a, agent_b):
            return True, "Agents have explicit trust relationship"
        
        # Check if blocked
        if (agent_a, agent_b) in self.blocked_pairs:
            return False, "Agents are explicitly blocked from merging"
        
        # Get fingerprints
        if schema_a is None or schema_b is None:
            fingerprint_a = self.get_fingerprint(agent_a)
            fingerprint_b = self.get_fingerprint(agent_b)
            
            if fingerprint_a is None or fingerprint_b is None:
                return False, "One or both agents not registered"
                
            if fingerprint_a.is_compatible(fingerprint_b, self.identity_match_threshold):
                return True, "Agents have compatible identities"
            else:
                return False, "Agents have incompatible identities"
        else:
            # Use provided schemas
            fingerprint_a = IdentityFingerprint.from_schema(schema_a, agent_a)
            fingerprint_b = IdentityFingerprint.from_schema(schema_b, agent_b)
            
            # Check identity compatibility
            if fingerprint_a.is_compatible(fingerprint_b, self.identity_match_threshold):
                return True, "Schemas have compatible identities"
            
            # Check contamination
            contamination_ab = self.check_contamination(schema_a, schema_b)
            if contamination_ab > self.contamination_threshold:
                return False, f"Contamination too high: {contamination_ab:.2f} > {self.contamination_threshold:.2f}"
                
            return True, "Schemas can be merged (contamination acceptable)"
    
    def can_import_memory(self, target_agent: str, source_agent: str, 
                        memory: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a memory from source_agent can be imported into target_agent.
        
        Args:
            target_agent: Target agent ID
            source_agent: Source agent ID
            memory: The memory to import
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check if trusted
        if self.is_trusted(target_agent, source_agent):
            return True, "Agents have explicit trust relationship"
        
        # Check if blocked
        if (target_agent, source_agent) in self.blocked_pairs:
            return False, "Agents are explicitly blocked from sharing memories"
        
        # Get fingerprints
        target_fingerprint = self.get_fingerprint(target_agent)
        source_fingerprint = self.get_fingerprint(source_agent)
        
        if target_fingerprint is None or source_fingerprint is None:
            return False, "One or both agents not registered"
            
        # Check identity compatibility
        if target_fingerprint.is_compatible(source_fingerprint, self.identity_match_threshold):
            return True, "Agents have compatible identities"
            
        # In strict mode, block memory import from incompatible identities
        if self.enable_strict_mode:
            return False, "Agents have incompatible identities (strict mode)"
            
        # Analyze memory for contamination risk
        # This is a simplified check - in practice would do deeper analysis
        contamination_risk = 0.0
        if "source_agent" in memory and memory["source_agent"] == source_agent:
            contamination_risk += 0.5
            
        if "beliefs" in memory and len(memory["beliefs"]) > 0:
            contamination_risk += 0.3
            
        if contamination_risk > self.contamination_threshold:
            return False, f"Memory contamination risk too high: {contamination_risk:.2f}"
            
        return True, "Memory can be imported (contamination risk acceptable)"
    
    def record_merge(self, agent_a: str, agent_b: str, 
                   action: str, allowed: bool, reason: str) -> None:
        """
        Record a merge attempt.
        
        Args:
            agent_a: The first agent's ID
            agent_b: The second agent's ID
            action: The action attempted (e.g., "schema_merge", "memory_import")
            allowed: Whether the merge was allowed
            reason: The reason for the decision
        """
        import time
        record = {
            "timestamp": time.time(),
            "agent_a": agent_a,
            "agent_b": agent_b,
            "action": action,
            "allowed": allowed,
            "reason": reason
        }
        self.merge_history.append(record)
        
        if not allowed:
            # Add to blocked pairs if merge was denied
            self.blocked_pairs.add((agent_a, agent_b))
            self.blocked_pairs.add((agent_b, agent_a))
            logger.warning(f"Blocked future merges between {agent_a} and {agent_b}: {reason}")
    
    def filter_schema(self, schema: Dict[str, Any], 
                    trusted_sources: List[str] = None) -> Dict[str, Any]:
        """
        Filter a schema to only include nodes from trusted sources.
        
        Args:
            schema: The schema to filter
            trusted_sources: List of trusted source agent IDs
            
        Returns:
            Filtered schema
        """
        if trusted_sources is None or "nodes" not in schema:
            return schema
            
        filtered_schema = schema.copy()
        filtered_nodes = []
        
        for node in schema["nodes"]:
            if isinstance(node, dict):
                # Check if node has source information
                source = node.get("source", {}).get("agent_id", None)
                if source is None or source in trusted_sources:
                    filtered_nodes.append(node)
            else:
                # If node doesn't have source info, keep it
                filtered_nodes.append(node)
                
        filtered_schema["nodes"] = filtered_nodes
        return filtered_schema
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the identity boundary system.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "registered_agents": len(self.fingerprints),
            "trusted_pairs": len(self.trusted_pairs) // 2,  # Divide by 2 because pairs are symmetric
            "blocked_pairs": len(self.blocked_pairs) // 2,
            "merge_attempts": len(self.merge_history),
            "allowed_merges": sum(1 for record in self.merge_history if record["allowed"]),
            "rejected_merges": sum(1 for record in self.merge_history if not record["allowed"]),
        }
        return metrics


# Helper function to create and configure an identity boundary
def create_identity_boundary(contamination_threshold: float = 0.1,
                            identity_match_threshold: float = 0.8,
                            enable_strict_mode: bool = False) -> AGIIdentityBoundary:
    """
    Create an identity boundary with the specified configuration.
    
    Args:
        contamination_threshold: Maximum allowed contamination
        identity_match_threshold: Threshold for identity matching
        enable_strict_mode: Whether to enable strict boundary enforcement
        
    Returns:
        Configured AGIIdentityBoundary
    """
    return AGIIdentityBoundary(
        contamination_threshold=contamination_threshold,
        identity_match_threshold=identity_match_threshold,
        enable_strict_mode=enable_strict_mode
    ) 