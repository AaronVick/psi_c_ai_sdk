"""
Visual Schema Editor for the ΨC-AI SDK Development Environment.

This module provides tools to visualize, analyze, and edit the schema structures 
that define agent cognitive patterns and memory layouts.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

from tools.dev_environment.base_tool import BaseTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import psi-c specific modules
try:
    from psi_c_ai_sdk.memory.schema import Schema, SchemaNode, SchemaEdge, SchemaConfig
    from psi_c_ai_sdk.memory.schema_graph import SchemaGraph
    PSI_C_AVAILABLE = True
except ImportError:
    logger.warning("Could not import Schema classes from psi_c_ai_sdk - running in limited mode")
    PSI_C_AVAILABLE = False
    
    # Define mock classes for development without the SDK
    class SchemaNode:
        def __init__(self, id, type, properties=None):
            self.id = id
            self.type = type
            self.properties = properties or {}
            
    class SchemaEdge:
        def __init__(self, source, target, type, properties=None):
            self.source = source
            self.target = target
            self.type = type
            self.properties = properties or {}
            
    class Schema:
        def __init__(self, nodes=None, edges=None, config=None):
            self.nodes = nodes or []
            self.edges = edges or []
            self.config = config or {}
            
    class SchemaGraph:
        def __init__(self, schema=None):
            self.schema = schema or Schema()
            self.graph = nx.DiGraph()
            
        def add_node(self, node):
            self.schema.nodes.append(node)
            self.graph.add_node(node.id, **node.__dict__)
            
        def add_edge(self, edge):
            self.schema.edges.append(edge)
            self.graph.add_edge(edge.source, edge.target, **edge.__dict__)
            
    class SchemaConfig:
        def __init__(self, properties=None):
            self.properties = properties or {}


class SchemaEditor(BaseTool):
    """
    Visual editor for ΨC agent schemas.
    
    This tool allows for interactive visualization, analysis, and editing of
    agent schema structures. It provides capabilities for:
    
    - Visualizing schema as interactive graphs
    - Editing schema nodes and edges
    - Analyzing schema properties
    - Schema comparison and versioning
    - Validation against schema constraints
    """
    
    def __init__(self, schema_graph=None, config_path: Optional[str] = None):
        """
        Initialize the Schema Editor.
        
        Args:
            schema_graph: The SchemaGraph to edit (optional)
            config_path: Path to configuration file (optional)
        """
        super().__init__(name="Schema Editor", 
                         description="Visualize and edit agent schema structures")
        
        self.schema_graph = schema_graph
        self.config = self._load_config(config_path)
        self.history = []  # For undo/redo functionality
        self.current_version = 0
        
        # Initialize an empty schema if none provided
        if not self.schema_graph and PSI_C_AVAILABLE:
            self.schema_graph = SchemaGraph(Schema(
                nodes=[],
                edges=[],
                config=SchemaConfig()
            ))
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration
        """
        default_config = {
            "visualization": {
                "node_size": 800,
                "edge_width": 1.5,
                "font_size": 10,
                "node_color_map": {
                    "concept": "#66c2a5",
                    "belief": "#fc8d62",
                    "memory": "#8da0cb",
                    "goal": "#e78ac3",
                    "value": "#a6d854",
                    "default": "#cccccc"
                },
                "edge_color_map": {
                    "is_a": "#377eb8",
                    "has_property": "#ff7f00",
                    "relates_to": "#4daf4a",
                    "default": "#999999"
                }
            },
            "validation": {
                "enforce_constraints": True,
                "allowed_node_types": ["concept", "belief", "memory", "goal", "value"],
                "allowed_edge_types": ["is_a", "has_property", "relates_to"]
            },
            "autosave": {
                "enabled": True,
                "interval": 300,  # seconds
                "max_backups": 5
            }
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                return default_config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            return default_config
    
    def set_schema_graph(self, schema_graph):
        """
        Set the schema graph to edit.
        
        Args:
            schema_graph: SchemaGraph instance
        """
        # Save current state to history
        if self.schema_graph:
            self._save_to_history()
            
        self.schema_graph = schema_graph
        
    def _save_to_history(self):
        """Save current schema state to history for undo/redo."""
        if not self.schema_graph:
            return
        
        # If we've done some undos and then make a new change,
        # we should truncate the history
        if self.current_version < len(self.history):
            self.history = self.history[:self.current_version]
        
        # Create a deep copy of the current schema
        if PSI_C_AVAILABLE:
            # Use the Schema's serialization methods if available
            schema_dict = self.schema_graph.schema.to_dict()
            self.history.append(schema_dict)
        else:
            # Simple serialization for mock objects
            nodes = [{
                'id': node.id,
                'type': node.type,
                'properties': node.properties
            } for node in self.schema_graph.schema.nodes]
            
            edges = [{
                'source': edge.source,
                'target': edge.target,
                'type': edge.type,
                'properties': edge.properties
            } for edge in self.schema_graph.schema.edges]
            
            self.history.append({
                'nodes': nodes,
                'edges': edges,
                'config': self.schema_graph.schema.config
            })
        
        self.current_version = len(self.history)
    
    def undo(self):
        """
        Undo the last change.
        
        Returns:
            True if undo was successful, False otherwise
        """
        if self.current_version <= 1:
            logger.info("Nothing to undo")
            return False
        
        self.current_version -= 1
        self._restore_from_history()
        return True
    
    def redo(self):
        """
        Redo the last undone change.
        
        Returns:
            True if redo was successful, False otherwise
        """
        if self.current_version >= len(self.history):
            logger.info("Nothing to redo")
            return False
        
        self.current_version += 1
        self._restore_from_history()
        return True
    
    def _restore_from_history(self):
        """Restore schema from history at current_version."""
        if not self.history or self.current_version < 1 or self.current_version > len(self.history):
            return
        
        schema_dict = self.history[self.current_version - 1]
        
        if PSI_C_AVAILABLE:
            # Use the Schema's deserialization methods if available
            schema = Schema.from_dict(schema_dict)
            self.schema_graph = SchemaGraph(schema)
        else:
            # Simple deserialization for mock objects
            nodes = [
                SchemaNode(
                    id=node['id'], 
                    type=node['type'], 
                    properties=node.get('properties', {})
                ) for node in schema_dict.get('nodes', [])
            ]
            
            edges = [
                SchemaEdge(
                    source=edge['source'],
                    target=edge['target'],
                    type=edge['type'],
                    properties=edge.get('properties', {})
                ) for edge in schema_dict.get('edges', [])
            ]
            
            schema = Schema(nodes=nodes, edges=edges, config=schema_dict.get('config', {}))
            self.schema_graph = SchemaGraph(schema)
    
    def load_schema(self, filepath: str) -> bool:
        """
        Load schema from a file.
        
        Args:
            filepath: Path to the schema file (JSON format)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                schema_dict = json.load(f)
            
            if PSI_C_AVAILABLE:
                schema = Schema.from_dict(schema_dict)
                self.schema_graph = SchemaGraph(schema)
            else:
                # Simple deserialization for mock objects
                nodes = [
                    SchemaNode(
                        id=node['id'], 
                        type=node['type'], 
                        properties=node.get('properties', {})
                    ) for node in schema_dict.get('nodes', [])
                ]
                
                edges = [
                    SchemaEdge(
                        source=edge['source'],
                        target=edge['target'],
                        type=edge['type'],
                        properties=edge.get('properties', {})
                    ) for edge in schema_dict.get('edges', [])
                ]
                
                schema = Schema(nodes=nodes, edges=edges, config=schema_dict.get('config', {}))
                self.schema_graph = SchemaGraph(schema)
            
            # Clear history and save initial state
            self.history = []
            self._save_to_history()
            
            logger.info(f"Schema loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load schema from {filepath}: {str(e)}")
            return False
    
    def save_schema(self, filepath: str) -> bool:
        """
        Save schema to a file.
        
        Args:
            filepath: Path to save the schema file (JSON format)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema_graph:
            logger.error("No schema to save")
            return False
        
        try:
            if PSI_C_AVAILABLE:
                schema_dict = self.schema_graph.schema.to_dict()
            else:
                # Simple serialization for mock objects
                nodes = [{
                    'id': node.id,
                    'type': node.type,
                    'properties': node.properties
                } for node in self.schema_graph.schema.nodes]
                
                edges = [{
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge.type,
                    'properties': edge.properties
                } for edge in self.schema_graph.schema.edges]
                
                schema_dict = {
                    'nodes': nodes,
                    'edges': edges,
                    'config': self.schema_graph.schema.config
                }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(schema_dict, f, indent=2)
                
            logger.info(f"Schema saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save schema to {filepath}: {str(e)}")
            return False
    
    def add_node(self, id: str, type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add a node to the schema.
        
        Args:
            id: Node identifier
            type: Node type
            properties: Node properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema_graph:
            logger.error("No schema graph to edit")
            return False
        
        # Validate node type if constraints are enabled
        if (self.config['validation']['enforce_constraints'] and 
            type not in self.config['validation']['allowed_node_types']):
            logger.error(f"Invalid node type: {type}")
            return False
        
        # Check if node already exists
        if any(node.id == id for node in self.schema_graph.schema.nodes):
            logger.error(f"Node with id {id} already exists")
            return False
        
        try:
            # Save current state to history
            self._save_to_history()
            
            # Create and add node
            node = SchemaNode(id=id, type=type, properties=properties or {})
            self.schema_graph.add_node(node)
            
            logger.info(f"Added node: {id} ({type})")
            return True
        except Exception as e:
            logger.error(f"Failed to add node: {str(e)}")
            return False
    
    def update_node(self, id: str, type: str = None, properties: Dict[str, Any] = None) -> bool:
        """
        Update an existing node in the schema.
        
        Args:
            id: Node identifier
            type: New node type (None to keep existing)
            properties: New node properties (None to keep existing)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema_graph:
            logger.error("No schema graph to edit")
            return False
        
        # Find the node
        node = next((node for node in self.schema_graph.schema.nodes if node.id == id), None)
        if not node:
            logger.error(f"Node with id {id} not found")
            return False
        
        # Validate node type if constraints are enabled
        if (type is not None and 
            self.config['validation']['enforce_constraints'] and 
            type not in self.config['validation']['allowed_node_types']):
            logger.error(f"Invalid node type: {type}")
            return False
        
        try:
            # Save current state to history
            self._save_to_history()
            
            # Update node
            if type is not None:
                node.type = type
            
            if properties is not None:
                node.properties = properties
            
            # Update the node in the graph representation
            attrs = {**node.__dict__}
            self.schema_graph.graph.nodes[id].update(attrs)
            
            logger.info(f"Updated node: {id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update node: {str(e)}")
            return False
    
    def remove_node(self, id: str) -> bool:
        """
        Remove a node from the schema.
        
        Args:
            id: Node identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema_graph:
            logger.error("No schema graph to edit")
            return False
        
        # Find the node
        node = next((node for node in self.schema_graph.schema.nodes if node.id == id), None)
        if not node:
            logger.error(f"Node with id {id} not found")
            return False
        
        try:
            # Save current state to history
            self._save_to_history()
            
            # Remove all edges connected to this node
            self.schema_graph.schema.edges = [
                edge for edge in self.schema_graph.schema.edges 
                if edge.source != id and edge.target != id
            ]
            
            # Remove node from schema
            self.schema_graph.schema.nodes = [
                node for node in self.schema_graph.schema.nodes if node.id != id
            ]
            
            # Remove from graph representation
            self.schema_graph.graph.remove_node(id)
            
            logger.info(f"Removed node: {id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove node: {str(e)}")
            return False
    
    def add_edge(self, source: str, target: str, type: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add an edge to the schema.
        
        Args:
            source: Source node ID
            target: Target node ID
            type: Edge type
            properties: Edge properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema_graph:
            logger.error("No schema graph to edit")
            return False
        
        # Validate edge type if constraints are enabled
        if (self.config['validation']['enforce_constraints'] and 
            type not in self.config['validation']['allowed_edge_types']):
            logger.error(f"Invalid edge type: {type}")
            return False
        
        # Check if source and target nodes exist
        source_node = next((node for node in self.schema_graph.schema.nodes if node.id == source), None)
        target_node = next((node for node in self.schema_graph.schema.nodes if node.id == target), None)
        
        if not source_node:
            logger.error(f"Source node with id {source} not found")
            return False
        
        if not target_node:
            logger.error(f"Target node with id {target} not found")
            return False
        
        # Check if edge already exists
        if any(edge.source == source and edge.target == target and edge.type == type 
               for edge in self.schema_graph.schema.edges):
            logger.error(f"Edge from {source} to {target} of type {type} already exists")
            return False
        
        try:
            # Save current state to history
            self._save_to_history()
            
            # Create and add edge
            edge = SchemaEdge(source=source, target=target, type=type, properties=properties or {})
            self.schema_graph.add_edge(edge)
            
            logger.info(f"Added edge: {source} --({type})--> {target}")
            return True
        except Exception as e:
            logger.error(f"Failed to add edge: {str(e)}")
            return False
    
    def update_edge(self, source: str, target: str, type: str = None, 
                   new_type: str = None, properties: Dict[str, Any] = None) -> bool:
        """
        Update an existing edge in the schema.
        
        Args:
            source: Source node ID
            target: Target node ID
            type: Current edge type (needed to identify the edge)
            new_type: New edge type (None to keep existing)
            properties: New edge properties (None to keep existing)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema_graph:
            logger.error("No schema graph to edit")
            return False
        
        # Find the edge
        edge = next((edge for edge in self.schema_graph.schema.edges 
                    if edge.source == source and edge.target == target and 
                    (type is None or edge.type == type)), None)
        
        if not edge:
            logger.error(f"Edge from {source} to {target}" + 
                        (f" of type {type}" if type else "") + 
                        " not found")
            return False
        
        # Validate edge type if constraints are enabled
        if (new_type is not None and 
            self.config['validation']['enforce_constraints'] and 
            new_type not in self.config['validation']['allowed_edge_types']):
            logger.error(f"Invalid edge type: {new_type}")
            return False
        
        try:
            # Save current state to history
            self._save_to_history()
            
            # Update edge
            if new_type is not None:
                edge.type = new_type
            
            if properties is not None:
                edge.properties = properties
            
            # Update the edge in the graph representation
            self.schema_graph.graph[source][target].update(edge.__dict__)
            
            logger.info(f"Updated edge: {source} --({edge.type})--> {target}")
            return True
        except Exception as e:
            logger.error(f"Failed to update edge: {str(e)}")
            return False
    
    def remove_edge(self, source: str, target: str, type: str = None) -> bool:
        """
        Remove an edge from the schema.
        
        Args:
            source: Source node ID
            target: Target node ID
            type: Edge type (None to remove all edges between source and target)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema_graph:
            logger.error("No schema graph to edit")
            return False
        
        # Find edges to remove
        edges_to_remove = [
            edge for edge in self.schema_graph.schema.edges 
            if edge.source == source and edge.target == target and
            (type is None or edge.type == type)
        ]
        
        if not edges_to_remove:
            logger.error(f"No edges found from {source} to {target}" + 
                        (f" of type {type}" if type else ""))
            return False
        
        try:
            # Save current state to history
            self._save_to_history()
            
            # Remove edges from schema
            self.schema_graph.schema.edges = [
                edge for edge in self.schema_graph.schema.edges 
                if edge not in edges_to_remove
            ]
            
            # Remove from graph representation
            for edge in edges_to_remove:
                if self.schema_graph.graph.has_edge(source, target):
                    self.schema_graph.graph.remove_edge(source, target)
            
            logger.info(f"Removed {len(edges_to_remove)} edge(s) from {source} to {target}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove edge(s): {str(e)}")
            return False
    
    def get_node(self, id: str) -> Optional[SchemaNode]:
        """
        Get a node by ID.
        
        Args:
            id: Node identifier
            
        Returns:
            SchemaNode if found, None otherwise
        """
        if not self.schema_graph:
            return None
        
        return next((node for node in self.schema_graph.schema.nodes if node.id == id), None)
    
    def get_edge(self, source: str, target: str, type: str = None) -> Optional[SchemaEdge]:
        """
        Get an edge by source, target, and optionally type.
        
        Args:
            source: Source node ID
            target: Target node ID
            type: Edge type (None to get any edge type)
            
        Returns:
            SchemaEdge if found, None otherwise
        """
        if not self.schema_graph:
            return None
        
        return next((edge for edge in self.schema_graph.schema.edges 
                    if edge.source == source and edge.target == target and
                    (type is None or edge.type == type)), None)
    
    def get_connected_nodes(self, id: str, edge_type: str = None, 
                           direction: str = 'both') -> List[SchemaNode]:
        """
        Get nodes connected to a given node.
        
        Args:
            id: Node identifier
            edge_type: Type of edges to consider (None for all)
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of connected SchemaNode objects
        """
        if not self.schema_graph or direction not in ['outgoing', 'incoming', 'both']:
            return []
        
        connected_nodes = []
        
        if direction in ['outgoing', 'both']:
            # Get nodes connected by outgoing edges
            for edge in self.schema_graph.schema.edges:
                if edge.source == id and (edge_type is None or edge.type == edge_type):
                    node = self.get_node(edge.target)
                    if node and node not in connected_nodes:
                        connected_nodes.append(node)
        
        if direction in ['incoming', 'both']:
            # Get nodes connected by incoming edges
            for edge in self.schema_graph.schema.edges:
                if edge.target == id and (edge_type is None or edge.type == edge_type):
                    node = self.get_node(edge.source)
                    if node and node not in connected_nodes:
                        connected_nodes.append(node)
        
        return connected_nodes
    
    def get_node_path(self, start_id: str, end_id: str) -> List[Tuple[SchemaNode, SchemaEdge]]:
        """
        Find the shortest path between two nodes.
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            
        Returns:
            List of (node, edge) tuples representing the path, or empty list if no path exists
        """
        if not self.schema_graph:
            return []
        
        if not self.schema_graph.graph.has_node(start_id) or not self.schema_graph.graph.has_node(end_id):
            return []
        
        try:
            # Find shortest path
            path_nodes = nx.shortest_path(self.schema_graph.graph, start_id, end_id)
            
            # Construct result with nodes and edges
            result = []
            for i in range(len(path_nodes) - 1):
                node = self.get_node(path_nodes[i])
                next_node = path_nodes[i + 1]
                edge = self.get_edge(node.id, next_node)
                
                result.append((node, edge))
            
            # Add final node
            result.append((self.get_node(path_nodes[-1]), None))
            
            return result
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Error finding path: {str(e)}")
            return []
    
    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate the schema against constraints.
        
        Returns:
            Dict with validation results
        """
        if not self.schema_graph:
            return {"valid": False, "error": "No schema graph to validate"}
        
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check for node type violations
        if self.config['validation']['enforce_constraints']:
            for node in self.schema_graph.schema.nodes:
                if node.type not in self.config['validation']['allowed_node_types']:
                    results["valid"] = False
                    results["errors"].append(f"Node {node.id} has invalid type: {node.type}")
        
            # Check for edge type violations
            for edge in self.schema_graph.schema.edges:
                if edge.type not in self.config['validation']['allowed_edge_types']:
                    results["valid"] = False
                    results["errors"].append(
                        f"Edge from {edge.source} to {edge.target} has invalid type: {edge.type}"
                    )
        
        # Check for edges with missing endpoints
        node_ids = [node.id for node in self.schema_graph.schema.nodes]
        for edge in self.schema_graph.schema.edges:
            if edge.source not in node_ids:
                results["valid"] = False
                results["errors"].append(f"Edge source node does not exist: {edge.source}")
            
            if edge.target not in node_ids:
                results["valid"] = False
                results["errors"].append(f"Edge target node does not exist: {edge.target}")
        
        # Check for duplicate node IDs
        node_id_counts = {}
        for node in self.schema_graph.schema.nodes:
            node_id_counts[node.id] = node_id_counts.get(node.id, 0) + 1
        
        for node_id, count in node_id_counts.items():
            if count > 1:
                results["valid"] = False
                results["errors"].append(f"Duplicate node ID: {node_id} (appears {count} times)")
        
        # Check for isolated nodes
        for node in self.schema_graph.schema.nodes:
            if not self.get_connected_nodes(node.id):
                results["warnings"].append(f"Isolated node: {node.id}")
        
        return results
    
    def plot_schema(self, figsize=(12, 8), layout='spring'):
        """
        Plot the schema as a graph visualization.
        
        Args:
            figsize: Figure size as (width, height)
            layout: Layout algorithm ('spring', 'circular', 'random', etc.)
            
        Returns:
            Matplotlib figure
        """
        if not self.schema_graph or not self.schema_graph.schema.nodes:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No schema to visualize", 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            return fig
            
        G = self.schema_graph.graph
        
        # Get the appropriate layout
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw nodes with different colors based on type
        node_colors = []
        for node in G.nodes:
            node_type = G.nodes[node].get('type', 'default')
            color = self.config['visualization']['node_color_map'].get(
                node_type, self.config['visualization']['node_color_map']['default']
            )
            node_colors.append(color)
        
        nx.draw_networkx_nodes(
            G, pos, 
            ax=ax,
            node_size=self.config['visualization']['node_size'],
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            ax=ax,
            font_size=self.config['visualization']['font_size'],
            font_weight='bold'
        )
        
        # Draw edges with different colors based on type
        edge_colors = {}
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type', 'default')
            color = self.config['visualization']['edge_color_map'].get(
                edge_type, self.config['visualization']['edge_color_map']['default']
            )
            edge_colors[(u, v)] = color
        
        # Group edges by color for more efficient drawing
        edges_by_color = {}
        for (u, v), color in edge_colors.items():
            if color not in edges_by_color:
                edges_by_color[color] = []
            edges_by_color[color].append((u, v))
        
        # Draw each group of edges with its color
        for color, edge_list in edges_by_color.items():
            nx.draw_networkx_edges(
                G, pos,
                ax=ax,
                edgelist=edge_list,
                width=self.config['visualization']['edge_width'],
                edge_color=color,
                arrows=True,
                arrowsize=10,
                alpha=0.7
            )
        
        # Draw edge labels
        edge_labels = {(u, v): data.get('type', '') 
                      for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos,
            ax=ax,
            edge_labels=edge_labels,
            font_size=self.config['visualization']['font_size'] - 2
        )
        
        plt.title("Schema Visualization")
        plt.axis('off')
        plt.tight_layout()
        
        return fig
    
    def export_as_json(self) -> str:
        """
        Export the schema as a JSON string.
        
        Returns:
            JSON representation of the schema
        """
        if not self.schema_graph:
            return "{}"
        
        try:
            if PSI_C_AVAILABLE:
                schema_dict = self.schema_graph.schema.to_dict()
            else:
                # Simple serialization for mock objects
                nodes = [{
                    'id': node.id,
                    'type': node.type,
                    'properties': node.properties
                } for node in self.schema_graph.schema.nodes]
                
                edges = [{
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge.type,
                    'properties': edge.properties
                } for edge in self.schema_graph.schema.edges]
                
                schema_dict = {
                    'nodes': nodes,
                    'edges': edges,
                    'config': self.schema_graph.schema.config
                }
                
            return json.dumps(schema_dict, indent=2)
        except Exception as e:
            logger.error(f"Failed to export schema as JSON: {str(e)}")
            return "{}"
    
    def export_as_dot(self) -> str:
        """
        Export the schema in DOT format for use with Graphviz.
        
        Returns:
            DOT representation of the schema
        """
        if not self.schema_graph:
            return "digraph G {}"
        
        try:
            G = self.schema_graph.graph
            
            # Use NetworkX's built-in DOT export
            dot_str = "\n".join(nx.generate_graphviz(G))
            
            return dot_str
        except Exception as e:
            logger.error(f"Failed to export schema as DOT: {str(e)}")
            return "digraph G {}"
    
    def get_schema_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the schema.
        
        Returns:
            Dict containing schema statistics
        """
        if not self.schema_graph:
            return {
                "node_count": 0,
                "edge_count": 0,
                "node_types": {},
                "edge_types": {},
                "density": 0.0,
                "average_degree": 0.0
            }
        
        try:
            G = self.schema_graph.graph
            
            # Count node types
            node_types = {}
            for node in self.schema_graph.schema.nodes:
                node_types[node.type] = node_types.get(node.type, 0) + 1
            
            # Count edge types
            edge_types = {}
            for edge in self.schema_graph.schema.edges:
                edge_types[edge.type] = edge_types.get(edge.type, 0) + 1
            
            # Calculate network metrics
            density = nx.density(G)
            
            # Average degree
            degrees = [d for _, d in G.degree()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            
            return {
                "node_count": len(self.schema_graph.schema.nodes),
                "edge_count": len(self.schema_graph.schema.edges),
                "node_types": node_types,
                "edge_types": edge_types,
                "density": density,
                "average_degree": avg_degree
            }
        except Exception as e:
            logger.error(f"Failed to get schema stats: {str(e)}")
            return {
                "error": str(e),
                "node_count": len(self.schema_graph.schema.nodes),
                "edge_count": len(self.schema_graph.schema.edges)
            } 