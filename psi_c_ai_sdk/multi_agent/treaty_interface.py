"""
External Schema Treaty Interface
------------------------------

This module enables ΨC agents to negotiate schema sharing treaties with other agents,
preserving identity boundaries while enabling cooperative intelligence.

A schema treaty defines:
- Which schema nodes are shared between agents
- Reflection limits on shared content
- Access permissions and expiration terms
- Trust boundaries and validation mechanisms

Mathematical basis:
    T_shared = |compliant accesses| / |total accesses|

This trust enforcement metric ensures agents adhere to treaty terms and allows
for adaptive trust calibration over time.
"""

import logging
import uuid
import json
import copy
import time
import datetime
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from pathlib import Path
import networkx as nx
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


class SchemaTreaty:
    """
    Represents a treaty between two or more agents for schema sharing.
    
    A treaty formally defines which parts of schema can be shared, under what
    conditions, and with what limitations. It acts as a trust boundary that
    enables federation while preserving agent identity.
    """
    
    def __init__(
        self,
        treaty_id: Optional[str] = None,
        name: Optional[str] = None,
        initiator_id: Optional[str] = None
    ):
        """
        Initialize a new schema treaty.
        
        Args:
            treaty_id: Unique identifier for the treaty, auto-generated if None
            name: Human-readable name for the treaty
            initiator_id: ID of the agent that initiated the treaty
        """
        # Basic treaty properties
        self.treaty_id = treaty_id or str(uuid.uuid4())
        self.name = name or f"Treaty-{self.treaty_id[:8]}"
        self.initiator_id = initiator_id
        self.created_at = datetime.datetime.now().isoformat()
        self.status = "draft"  # draft, proposed, active, expired, terminated
        
        # Parties to the treaty
        self.parties = {}  # agent_id -> metadata
        if initiator_id:
            self.add_party(initiator_id, role="initiator")
        
        # Schema sharing terms
        self.shared_nodes = {}  # node_id -> {permissions}
        self.shared_clusters = {}  # cluster_id -> {nodes, permissions}
        
        # Reflection and access limits
        self.reflection_limits = {
            "max_depth": 2,  # Default max recursive reflection depth
            "max_mutations": 5  # Default max mutations allowed
        }
        
        # Time constraints
        self.expiration = {
            "type": "indefinite",  # indefinite, timestamp, access_count
            "value": None
        }
        
        # Trust and compliance
        self.trust_metrics = {}  # agent_id -> trust_score
        self.access_log = []  # List of access records
        self.compliance_stats = {
            "compliant_accesses": 0,
            "non_compliant_accesses": 0,
            "trust_score": 1.0
        }
    
    def add_party(self, agent_id: str, role: str = "participant", metadata: Optional[Dict] = None):
        """
        Add a party to the treaty.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role of the agent (initiator, participant, observer)
            metadata: Additional agent metadata
        """
        if not metadata:
            metadata = {}
            
        self.parties[agent_id] = {
            "role": role,
            "joined_at": datetime.datetime.now().isoformat(),
            "status": "pending",  # pending, active, suspended, withdrawn
            "metadata": metadata
        }
        
        # Initialize trust metrics for the new party
        self.trust_metrics[agent_id] = 1.0
        
        logger.info(f"Added party {agent_id} to treaty {self.treaty_id} as {role}")
    
    def add_shared_nodes(self, agent_id: str, node_ids: List[str], permissions: Optional[Dict] = None):
        """
        Add nodes to be shared by a particular agent.
        
        Args:
            agent_id: ID of the agent contributing the nodes
            node_ids: List of node IDs to share
            permissions: Optional permissions dictating how nodes can be used
        """
        if agent_id not in self.parties:
            raise ValueError(f"Agent {agent_id} is not a party to this treaty")
            
        if not permissions:
            permissions = {
                "read": True,
                "reference": True,
                "modify": False,
                "delete": False
            }
            
        for node_id in node_ids:
            self.shared_nodes[node_id] = {
                "owner": agent_id,
                "added_at": datetime.datetime.now().isoformat(),
                "permissions": permissions,
                "access_count": 0
            }
            
        logger.info(f"Agent {agent_id} shared {len(node_ids)} nodes under treaty {self.treaty_id}")
    
    def add_shared_cluster(self, agent_id: str, cluster_name: str, node_ids: List[str], 
                           permissions: Optional[Dict] = None):
        """
        Add a cluster of related nodes to be shared.
        
        Args:
            agent_id: ID of the agent contributing the cluster
            cluster_name: Name to identify the cluster
            node_ids: List of node IDs in the cluster
            permissions: Optional permissions dictating how cluster can be used
        """
        if agent_id not in self.parties:
            raise ValueError(f"Agent {agent_id} is not a party to this treaty")
            
        if not permissions:
            permissions = {
                "read": True,
                "reference": True, 
                "modify": False,
                "delete": False
            }
            
        cluster_id = f"{cluster_name}_{str(uuid.uuid4())[:8]}"
        
        self.shared_clusters[cluster_id] = {
            "name": cluster_name,
            "owner": agent_id,
            "nodes": node_ids,
            "added_at": datetime.datetime.now().isoformat(),
            "permissions": permissions,
            "access_count": 0
        }
        
        # Also add individual nodes
        for node_id in node_ids:
            if node_id not in self.shared_nodes:
                self.shared_nodes[node_id] = {
                    "owner": agent_id,
                    "cluster": cluster_id,
                    "added_at": datetime.datetime.now().isoformat(),
                    "permissions": permissions.copy(),
                    "access_count": 0
                }
                
        logger.info(f"Agent {agent_id} shared cluster '{cluster_name}' with {len(node_ids)} nodes")
    
    def set_reflection_limits(self, max_depth: Optional[int] = None, 
                             max_mutations: Optional[int] = None):
        """
        Set limits on reflection operations for shared schema.
        
        Args:
            max_depth: Maximum recursive reflection depth allowed
            max_mutations: Maximum number of mutations allowed
        """
        if max_depth is not None:
            self.reflection_limits["max_depth"] = max_depth
            
        if max_mutations is not None:
            self.reflection_limits["max_mutations"] = max_mutations
            
        logger.info(f"Updated reflection limits: depth={max_depth}, mutations={max_mutations}")
    
    def set_expiration(self, expiration_type: str, value: Any):
        """
        Set expiration terms for the treaty.
        
        Args:
            expiration_type: Type of expiration (indefinite, timestamp, access_count)
            value: Expiration value (timestamp or access count)
        """
        valid_types = ["indefinite", "timestamp", "access_count"]
        if expiration_type not in valid_types:
            raise ValueError(f"Invalid expiration type. Must be one of: {valid_types}")
            
        self.expiration = {
            "type": expiration_type,
            "value": value
        }
        
        logger.info(f"Set treaty expiration: {expiration_type} = {value}")
    
    def is_expired(self) -> bool:
        """
        Check if the treaty has expired.
        
        Returns:
            True if the treaty has expired, False otherwise
        """
        if self.status in ["expired", "terminated"]:
            return True
            
        if self.expiration["type"] == "indefinite":
            return False
            
        elif self.expiration["type"] == "timestamp":
            expiry_time = datetime.datetime.fromisoformat(self.expiration["value"])
            current_time = datetime.datetime.now()
            return current_time >= expiry_time
            
        elif self.expiration["type"] == "access_count":
            total_accesses = sum(node["access_count"] for node in self.shared_nodes.values())
            return total_accesses >= self.expiration["value"]
            
        return False
    
    def log_access(self, agent_id: str, node_id: str, access_type: str, 
                   is_compliant: bool, context: Optional[Dict] = None):
        """
        Log an access to a shared node.
        
        Args:
            agent_id: ID of the agent accessing the node
            node_id: ID of the node being accessed
            access_type: Type of access (read, reference, modify, delete)
            is_compliant: Whether the access complies with treaty terms
            context: Additional context about the access
        """
        if agent_id not in self.parties:
            logger.warning(f"Agent {agent_id} is not a party to treaty {self.treaty_id}")
            is_compliant = False
            
        if node_id not in self.shared_nodes:
            logger.warning(f"Node {node_id} is not covered by treaty {self.treaty_id}")
            is_compliant = False
            
        # Create access record
        access_record = {
            "agent_id": agent_id,
            "node_id": node_id,
            "access_type": access_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "is_compliant": is_compliant,
            "context": context or {}
        }
        
        # Add to log
        self.access_log.append(access_record)
        
        # Update compliance stats
        if is_compliant:
            self.compliance_stats["compliant_accesses"] += 1
        else:
            self.compliance_stats["non_compliant_accesses"] += 1
            
        # Update trust score
        total = self.compliance_stats["compliant_accesses"] + self.compliance_stats["non_compliant_accesses"]
        if total > 0:
            self.compliance_stats["trust_score"] = self.compliance_stats["compliant_accesses"] / total
            
        # Update node access count
        if node_id in self.shared_nodes:
            self.shared_nodes[node_id]["access_count"] += 1
            
        # Check if treaty has expired due to access count
        if self.expiration["type"] == "access_count":
            total_accesses = sum(node["access_count"] for node in self.shared_nodes.values())
            if total_accesses >= self.expiration["value"]:
                self.status = "expired"
                logger.info(f"Treaty {self.treaty_id} has expired due to access count limit")
    
    def update_trust_score(self, agent_id: str, new_score: Optional[float] = None):
        """
        Update trust score for a specific agent based on compliance.
        
        Args:
            agent_id: ID of the agent to update
            new_score: Optional explicit score, otherwise calculated from compliance
        """
        if agent_id not in self.parties:
            raise ValueError(f"Agent {agent_id} is not a party to this treaty")
            
        if new_score is not None:
            self.trust_metrics[agent_id] = max(0.0, min(1.0, new_score))  # Clamp to [0, 1]
        else:
            # Calculate from recent access logs
            agent_logs = [log for log in self.access_log if log["agent_id"] == agent_id]
            
            if agent_logs:
                # Use the 10 most recent logs, or all if fewer than 10
                recent_logs = agent_logs[-10:]
                compliant = sum(1 for log in recent_logs if log["is_compliant"])
                if recent_logs:
                    score = compliant / len(recent_logs)
                    self.trust_metrics[agent_id] = score
    
    def check_permission(self, agent_id: str, node_id: str, access_type: str) -> bool:
        """
        Check if an agent has permission for a specific access type.
        
        Args:
            agent_id: ID of the agent requesting access
            node_id: ID of the node to access
            access_type: Type of access (read, reference, modify, delete)
            
        Returns:
            True if access is permitted, False otherwise
        """
        if self.is_expired():
            return False
            
        if agent_id not in self.parties:
            return False
            
        if node_id not in self.shared_nodes:
            return False
            
        # Skip permission check for the owner
        if self.shared_nodes[node_id]["owner"] == agent_id:
            return True
            
        # Check permission
        return self.shared_nodes[node_id]["permissions"].get(access_type, False)
    
    def to_dict(self) -> Dict:
        """
        Convert treaty to dictionary for serialization.
        
        Returns:
            Dictionary representation of the treaty
        """
        return {
            "treaty_id": self.treaty_id,
            "name": self.name,
            "initiator_id": self.initiator_id,
            "created_at": self.created_at,
            "status": self.status,
            "parties": self.parties,
            "shared_nodes": self.shared_nodes,
            "shared_clusters": self.shared_clusters,
            "reflection_limits": self.reflection_limits,
            "expiration": self.expiration,
            "trust_metrics": self.trust_metrics,
            "compliance_stats": self.compliance_stats
        }
    
    def save(self, file_path: Optional[str] = None) -> str:
        """
        Save treaty to file.
        
        Args:
            file_path: Path to save the treaty JSON file
            
        Returns:
            Path to the saved file
        """
        if file_path is None:
            # Generate default path
            treaty_dir = Path.home() / ".psi_c_ai_sdk" / "treaties"
            treaty_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = treaty_dir / f"treaty_{self.treaty_id}.json"
        else:
            file_path = Path(file_path)
            
        # Serialize to JSON
        treaty_dict = self.to_dict()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(treaty_dict, f, indent=2, default=str)
            
        logger.info(f"Saved treaty {self.treaty_id} to {file_path}")
        return str(file_path)
    
    @classmethod
    def load(cls, file_path: str) -> 'SchemaTreaty':
        """
        Load treaty from file.
        
        Args:
            file_path: Path to the treaty JSON file
            
        Returns:
            Loaded SchemaTreaty object
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r') as f:
            treaty_dict = json.load(f)
            
        treaty = cls(
            treaty_id=treaty_dict["treaty_id"],
            name=treaty_dict["name"],
            initiator_id=treaty_dict["initiator_id"]
        )
        
        # Load properties
        treaty.created_at = treaty_dict["created_at"]
        treaty.status = treaty_dict["status"]
        treaty.parties = treaty_dict["parties"]
        treaty.shared_nodes = treaty_dict["shared_nodes"]
        treaty.shared_clusters = treaty_dict["shared_clusters"]
        treaty.reflection_limits = treaty_dict["reflection_limits"]
        treaty.expiration = treaty_dict["expiration"]
        treaty.trust_metrics = treaty_dict["trust_metrics"]
        treaty.compliance_stats = treaty_dict["compliance_stats"]
        
        logger.info(f"Loaded treaty {treaty.treaty_id} from {file_path}")
        return treaty


class SchemaRestrictionZone:
    """
    Creates a restricted view of an agent's schema based on treaty permissions.
    
    This acts as a security boundary, ensuring that schema access adheres to
    treaty terms and maintains proper isolation between agents.
    """
    
    def __init__(self, treaty: SchemaTreaty, agent_id: str):
        """
        Initialize a schema restriction zone.
        
        Args:
            treaty: The treaty defining access permissions
            agent_id: ID of the agent accessing the schema
        """
        self.treaty = treaty
        self.agent_id = agent_id
        self.permitted_nodes = set()
        self.permitted_edges = set()
        self.permissions = {}  # node_id -> {permission_type: bool}
        
        # Calculate permitted nodes based on treaty
        self._calculate_permissions()
    
    def _calculate_permissions(self):
        """Calculate which nodes and permissions are available to this agent."""
        for node_id, node_info in self.treaty.shared_nodes.items():
            # Check if this node is visible to the agent
            if self.treaty.check_permission(self.agent_id, node_id, "read"):
                self.permitted_nodes.add(node_id)
                self.permissions[node_id] = node_info["permissions"]
    
    def is_permitted(self, node_id: str, access_type: str = "read") -> bool:
        """
        Check if a specific node access is permitted.
        
        Args:
            node_id: ID of the node to check
            access_type: Type of access to check
            
        Returns:
            True if access is permitted, False otherwise
        """
        if node_id not in self.permitted_nodes:
            return False
            
        return self.permissions[node_id].get(access_type, False)
    
    def filter_schema_graph(self, schema_graph: nx.Graph) -> nx.Graph:
        """
        Create a filtered view of a schema graph based on permissions.
        
        Args:
            schema_graph: The full schema graph
            
        Returns:
            A filtered graph containing only permitted nodes and edges
        """
        # Create a new graph for the filtered view
        filtered_graph = nx.Graph()
        
        # Add permitted nodes
        for node_id in self.permitted_nodes:
            if schema_graph.has_node(node_id):
                # Copy node with attributes
                filtered_graph.add_node(node_id, **schema_graph.nodes[node_id])
                
        # Add edges between permitted nodes
        for u, v in schema_graph.edges():
            if u in self.permitted_nodes and v in self.permitted_nodes:
                # Copy edge with attributes
                filtered_graph.add_edge(u, v, **schema_graph.get_edge_data(u, v))
                
        return filtered_graph
    
    def log_access(self, node_id: str, access_type: str, context: Optional[Dict] = None):
        """
        Log an access through this restriction zone.
        
        Args:
            node_id: ID of the node being accessed
            access_type: Type of access (read, reference, modify, delete)
            context: Additional context about the access
        """
        is_compliant = self.is_permitted(node_id, access_type)
        
        # Log the access in the treaty
        self.treaty.log_access(
            agent_id=self.agent_id,
            node_id=node_id,
            access_type=access_type,
            is_compliant=is_compliant,
            context=context
        )
        
        if not is_compliant:
            logger.warning(f"Non-compliant {access_type} access to node {node_id} by agent {self.agent_id}")
    
    def get_permitted_nodes(self) -> Set[str]:
        """
        Get the set of node IDs permitted for this agent.
        
        Returns:
            Set of permitted node IDs
        """
        return self.permitted_nodes.copy()
    
    def get_permissions(self, node_id: str) -> Dict[str, bool]:
        """
        Get the permissions for a specific node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary mapping permission types to boolean values
        """
        if node_id not in self.permissions:
            return {}
            
        return self.permissions[node_id].copy()


class TreatyInterface:
    """
    Interface for managing schema treaties between ΨC agents.
    
    This class provides methods for creating, proposing, accepting, and
    enforcing schema sharing treaties between agents.
    """
    
    def __init__(self, agent_id: str, data_dir: Optional[str] = None):
        """
        Initialize the treaty interface.
        
        Args:
            agent_id: ID of the agent using this interface
            data_dir: Directory for storing treaty data
        """
        self.agent_id = agent_id
        
        # Set data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / ".psi_c_ai_sdk" / "treaties"
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track active treaties
        self.active_treaties = {}  # treaty_id -> SchemaTreaty
        self.treaty_restrictions = {}  # treaty_id -> SchemaRestrictionZone
        
        # Load existing treaties for this agent
        self._load_agent_treaties()
    
    def _load_agent_treaties(self):
        """Load treaties where this agent is a participant."""
        treaty_files = list(self.data_dir.glob("treaty_*.json"))
        
        for file_path in treaty_files:
            try:
                treaty = SchemaTreaty.load(str(file_path))
                
                # Check if this agent is a party to the treaty
                if self.agent_id in treaty.parties:
                    # Skip expired treaties
                    if treaty.is_expired():
                        continue
                        
                    # Add to active treaties
                    self.active_treaties[treaty.treaty_id] = treaty
                    
                    # Create restriction zone
                    self.treaty_restrictions[treaty.treaty_id] = SchemaRestrictionZone(
                        treaty=treaty,
                        agent_id=self.agent_id
                    )
                    
            except Exception as e:
                logger.error(f"Error loading treaty from {file_path}: {e}")
                
        logger.info(f"Loaded {len(self.active_treaties)} active treaties for agent {self.agent_id}")
    
    def create_treaty(self, name: Optional[str] = None) -> SchemaTreaty:
        """
        Create a new schema sharing treaty.
        
        Args:
            name: Optional name for the treaty
            
        Returns:
            The created treaty
        """
        treaty = SchemaTreaty(
            name=name,
            initiator_id=self.agent_id
        )
        
        # Add this agent as initiator
        if self.agent_id not in treaty.parties:
            treaty.add_party(self.agent_id, role="initiator")
            
        logger.info(f"Created new treaty {treaty.treaty_id}")
        return treaty
    
    def propose_treaty(self, treaty: SchemaTreaty, recipient_ids: List[str]) -> str:
        """
        Propose a treaty to other agents.
        
        Args:
            treaty: The treaty to propose
            recipient_ids: IDs of agents to propose to
            
        Returns:
            Path to the saved treaty file
        """
        # Ensure treaty is in draft status
        if treaty.status != "draft":
            raise ValueError("Can only propose treaties in draft status")
            
        # Add recipients as parties if not already present
        for agent_id in recipient_ids:
            if agent_id not in treaty.parties:
                treaty.add_party(agent_id, role="participant")
                
        # Update treaty status
        treaty.status = "proposed"
        
        # Save treaty
        file_path = treaty.save(self.data_dir / f"treaty_{treaty.treaty_id}.json")
        
        # Add to active treaties
        self.active_treaties[treaty.treaty_id] = treaty
        
        # Create restriction zone
        self.treaty_restrictions[treaty.treaty_id] = SchemaRestrictionZone(
            treaty=treaty,
            agent_id=self.agent_id
        )
        
        logger.info(f"Proposed treaty {treaty.treaty_id} to {len(recipient_ids)} agents")
        return file_path
    
    def accept_treaty(self, treaty_id: str) -> bool:
        """
        Accept a proposed treaty.
        
        Args:
            treaty_id: ID of the treaty to accept
            
        Returns:
            True if accepted successfully, False otherwise
        """
        # Check if treaty exists
        if treaty_id not in self.active_treaties:
            logger.error(f"Treaty {treaty_id} not found or not active")
            return False
            
        treaty = self.active_treaties[treaty_id]
        
        # Check if this agent is a party to the treaty
        if self.agent_id not in treaty.parties:
            logger.error(f"Agent {self.agent_id} is not a party to treaty {treaty_id}")
            return False
            
        # Update party status
        treaty.parties[self.agent_id]["status"] = "active"
        
        # If all parties have accepted, update treaty status
        all_accepted = all(p["status"] == "active" for p in treaty.parties.values())
        if all_accepted:
            treaty.status = "active"
            
        # Save treaty
        treaty.save(self.data_dir / f"treaty_{treaty_id}.json")
        
        logger.info(f"Agent {self.agent_id} accepted treaty {treaty_id}")
        return True
    
    def reject_treaty(self, treaty_id: str) -> bool:
        """
        Reject a proposed treaty.
        
        Args:
            treaty_id: ID of the treaty to reject
            
        Returns:
            True if rejected successfully, False otherwise
        """
        # Check if treaty exists
        if treaty_id not in self.active_treaties:
            logger.error(f"Treaty {treaty_id} not found or not active")
            return False
            
        treaty = self.active_treaties[treaty_id]
        
        # Check if this agent is a party to the treaty
        if self.agent_id not in treaty.parties:
            logger.error(f"Agent {self.agent_id} is not a party to treaty {treaty_id}")
            return False
            
        # Update party status
        treaty.parties[self.agent_id]["status"] = "withdrawn"
        
        # Save treaty
        treaty.save(self.data_dir / f"treaty_{treaty_id}.json")
        
        # Remove from active treaties
        del self.active_treaties[treaty_id]
        if treaty_id in self.treaty_restrictions:
            del self.treaty_restrictions[treaty_id]
            
        logger.info(f"Agent {self.agent_id} rejected treaty {treaty_id}")
        return True
    
    def terminate_treaty(self, treaty_id: str) -> bool:
        """
        Terminate an active treaty.
        
        Args:
            treaty_id: ID of the treaty to terminate
            
        Returns:
            True if terminated successfully, False otherwise
        """
        # Check if treaty exists
        if treaty_id not in self.active_treaties:
            logger.error(f"Treaty {treaty_id} not found or not active")
            return False
            
        treaty = self.active_treaties[treaty_id]
        
        # Check if this agent is the initiator or has termination rights
        if (treaty.initiator_id != self.agent_id and
            treaty.parties.get(self.agent_id, {}).get("role") != "initiator"):
            logger.error(f"Agent {self.agent_id} does not have rights to terminate treaty {treaty_id}")
            return False
            
        # Update treaty status
        treaty.status = "terminated"
        
        # Save treaty
        treaty.save(self.data_dir / f"treaty_{treaty_id}.json")
        
        # Remove from active treaties
        del self.active_treaties[treaty_id]
        if treaty_id in self.treaty_restrictions:
            del self.treaty_restrictions[treaty_id]
            
        logger.info(f"Agent {self.agent_id} terminated treaty {treaty_id}")
        return True
    
    def share_schema_nodes(self, treaty_id: str, node_ids: List[str], 
                          permissions: Optional[Dict] = None) -> bool:
        """
        Share schema nodes under a treaty.
        
        Args:
            treaty_id: ID of the treaty to share under
            node_ids: List of node IDs to share
            permissions: Optional custom permissions
            
        Returns:
            True if shared successfully, False otherwise
        """
        # Check if treaty exists
        if treaty_id not in self.active_treaties:
            logger.error(f"Treaty {treaty_id} not found or not active")
            return False
            
        treaty = self.active_treaties[treaty_id]
        
        # Add shared nodes
        try:
            treaty.add_shared_nodes(self.agent_id, node_ids, permissions)
            
            # Save treaty
            treaty.save(self.data_dir / f"treaty_{treaty_id}.json")
            
            # Update restriction zone
            self.treaty_restrictions[treaty_id] = SchemaRestrictionZone(
                treaty=treaty,
                agent_id=self.agent_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sharing nodes under treaty {treaty_id}: {e}")
            return False
    
    def share_schema_cluster(self, treaty_id: str, cluster_name: str, node_ids: List[str],
                            permissions: Optional[Dict] = None) -> bool:
        """
        Share a cluster of schema nodes under a treaty.
        
        Args:
            treaty_id: ID of the treaty to share under
            cluster_name: Name for the cluster
            node_ids: List of node IDs in the cluster
            permissions: Optional custom permissions
            
        Returns:
            True if shared successfully, False otherwise
        """
        # Check if treaty exists
        if treaty_id not in self.active_treaties:
            logger.error(f"Treaty {treaty_id} not found or not active")
            return False
            
        treaty = self.active_treaties[treaty_id]
        
        # Add shared cluster
        try:
            treaty.add_shared_cluster(self.agent_id, cluster_name, node_ids, permissions)
            
            # Save treaty
            treaty.save(self.data_dir / f"treaty_{treaty_id}.json")
            
            # Update restriction zone
            self.treaty_restrictions[treaty_id] = SchemaRestrictionZone(
                treaty=treaty,
                agent_id=self.agent_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sharing cluster under treaty {treaty_id}: {e}")
            return False
    
    def get_permitted_schema(self, treaty_id: str, schema_graph: nx.Graph) -> nx.Graph:
        """
        Get a filtered view of a schema based on treaty permissions.
        
        Args:
            treaty_id: ID of the treaty to use
            schema_graph: The full schema graph
            
        Returns:
            A filtered graph containing only permitted nodes and edges
        """
        # Check if treaty exists
        if treaty_id not in self.treaty_restrictions:
            logger.error(f"Treaty {treaty_id} not found or no restrictions defined")
            return nx.Graph()  # Return empty graph
            
        restriction_zone = self.treaty_restrictions[treaty_id]
        
        # Filter schema graph
        return restriction_zone.filter_schema_graph(schema_graph)
    
    def log_schema_access(self, treaty_id: str, node_id: str, access_type: str, 
                         context: Optional[Dict] = None) -> bool:
        """
        Log an access to a shared schema node.
        
        Args:
            treaty_id: ID of the treaty
            node_id: ID of the node being accessed
            access_type: Type of access
            context: Additional context
            
        Returns:
            True if logged successfully, False otherwise
        """
        # Check if treaty exists
        if treaty_id not in self.treaty_restrictions:
            logger.error(f"Treaty {treaty_id} not found or no restrictions defined")
            return False
            
        restriction_zone = self.treaty_restrictions[treaty_id]
        
        # Log access
        try:
            restriction_zone.log_access(node_id, access_type, context)
            return True
        except Exception as e:
            logger.error(f"Error logging access to node {node_id}: {e}")
            return False
    
    def get_trust_metrics(self, treaty_id: Optional[str] = None) -> Dict:
        """
        Get trust metrics for treaties.
        
        Args:
            treaty_id: Optional specific treaty ID, or None for all treaties
            
        Returns:
            Dictionary mapping treaty IDs to trust metrics
        """
        if treaty_id:
            # Get metrics for specific treaty
            if treaty_id not in self.active_treaties:
                logger.error(f"Treaty {treaty_id} not found or not active")
                return {}
                
            treaty = self.active_treaties[treaty_id]
            return {treaty_id: treaty.trust_metrics}
            
        else:
            # Get metrics for all treaties
            metrics = {}
            for t_id, treaty in self.active_treaties.items():
                metrics[t_id] = treaty.trust_metrics
                
            return metrics
    
    def list_active_treaties(self) -> List[Dict]:
        """
        Get a list of active treaties for this agent.
        
        Returns:
            List of treaty summary dictionaries
        """
        treaty_summaries = []
        
        for treaty_id, treaty in self.active_treaties.items():
            # Skip expired treaties
            if treaty.is_expired():
                continue
                
            summary = {
                "treaty_id": treaty_id,
                "name": treaty.name,
                "initiator_id": treaty.initiator_id,
                "status": treaty.status,
                "party_count": len(treaty.parties),
                "shared_node_count": len(treaty.shared_nodes),
                "shared_cluster_count": len(treaty.shared_clusters),
                "my_role": treaty.parties.get(self.agent_id, {}).get("role", "unknown"),
                "my_status": treaty.parties.get(self.agent_id, {}).get("status", "unknown"),
                "my_trust_score": treaty.trust_metrics.get(self.agent_id, 0.0),
                "created_at": treaty.created_at
            }
            
            treaty_summaries.append(summary)
            
        return treaty_summaries


def create_sharing_treaty(agent_a, agent_b, shared_nodes=None, name=None):
    """
    Utility function to create a schema sharing treaty between two agents.
    
    Args:
        agent_a: First agent (initiator)
        agent_b: Second agent (participant)
        shared_nodes: Optional dictionary mapping agent IDs to lists of node IDs to share
        name: Optional name for the treaty
        
    Returns:
        The created treaty
    """
    # Get or create agent IDs
    agent_a_id = getattr(agent_a, 'id', str(id(agent_a)))
    agent_b_id = getattr(agent_b, 'id', str(id(agent_b)))
    
    # Create treaty interface for agent A
    interface_a = TreatyInterface(agent_a_id)
    
    # Create a new treaty
    treaty = interface_a.create_treaty(name=name or f"Treaty-{agent_a_id[:4]}-{agent_b_id[:4]}")
    
    # Add shared nodes if provided
    if shared_nodes:
        if agent_a_id in shared_nodes:
            treaty.add_shared_nodes(agent_a_id, shared_nodes[agent_a_id])
        if agent_b_id in shared_nodes:
            treaty.add_shared_nodes(agent_b_id, shared_nodes[agent_b_id])
    
    # Propose treaty to agent B
    interface_a.propose_treaty(treaty, [agent_b_id])
    
    # Create treaty interface for agent B
    interface_b = TreatyInterface(agent_b_id)
    
    # Accept treaty
    interface_b.accept_treaty(treaty.treaty_id)
    
    print(f"Created and activated treaty {treaty.treaty_id} between agents {agent_a_id} and {agent_b_id}")
    return treaty


if __name__ == "__main__":
    # Simple demo
    agent_alice = type('Agent', (), {'id': 'alice'})()
    agent_bob = type('Agent', (), {'id': 'bob'})()
    
    # Create a treaty between Alice and Bob
    treaty = create_sharing_treaty(
        agent_alice, 
        agent_bob,
        shared_nodes={
            'alice': ['node1', 'node2', 'node3'],
            'bob': ['node4', 'node5']
        },
        name="Demo Treaty"
    )
    
    # Print treaty details
    print(f"Treaty ID: {treaty.treaty_id}")
    print(f"Shared nodes: {len(treaty.shared_nodes)}")
    print(f"Trust score: {treaty.compliance_stats['trust_score']}") 