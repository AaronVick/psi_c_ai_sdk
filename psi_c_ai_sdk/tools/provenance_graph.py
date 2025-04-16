"""
Reflection Provenance Graph

Shows causal chains of reflection → contradiction → schema change → ΨC delta.
Provides traceability for cognitive events in the ΨC system.
"""

import os
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ProvenanceEvent:
    """Represents a single event in the provenance chain."""
    
    def __init__(self, event_type: str, event_id: str, timestamp: Optional[float] = None,
                 data: Optional[Dict[str, Any]] = None, parent_ids: Optional[List[str]] = None):
        """
        Initialize a provenance event.
        
        Args:
            event_type: Type of event (e.g., 'reflection', 'contradiction', 'schema_change')
            event_id: Unique identifier for this event
            timestamp: Event timestamp (defaults to current time)
            data: Additional event data
            parent_ids: List of parent event IDs that caused this event
        """
        self.event_type = event_type
        self.event_id = event_id
        self.timestamp = timestamp or datetime.now().timestamp()
        self.data = data or {}
        self.parent_ids = parent_ids or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "parent_ids": self.parent_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceEvent':
        """Create event from dictionary representation."""
        return cls(
            event_type=data["event_type"],
            event_id=data["event_id"],
            timestamp=data.get("timestamp"),
            data=data.get("data", {}),
            parent_ids=data.get("parent_ids", [])
        )


class ProvenanceGraph:
    """Graph representation of causal chains in ΨC cognitive events."""
    
    def __init__(self):
        """Initialize the provenance graph."""
        self.graph = nx.DiGraph()
        self.events = {}  # Dict[event_id, ProvenanceEvent]
        
    def add_event(self, event: ProvenanceEvent) -> str:
        """
        Add an event to the provenance graph.
        
        Args:
            event: The event to add
            
        Returns:
            The event ID
        """
        # Add node to graph
        self.graph.add_node(event.event_id, 
                          event_type=event.event_type,
                          timestamp=event.timestamp,
                          data=event.data)
        
        # Add edges from parent events
        for parent_id in event.parent_ids:
            if parent_id in self.events or parent_id in self.graph:
                self.graph.add_edge(parent_id, event.event_id)
            else:
                logger.warning(f"Parent event {parent_id} not found when adding {event.event_id}")
        
        # Store event
        self.events[event.event_id] = event
        return event.event_id
    
    def add_reflection_event(self, reflection_id: str, reflection_data: Dict[str, Any],
                           parent_ids: Optional[List[str]] = None) -> str:
        """
        Convenience method to add a reflection event.
        
        Args:
            reflection_id: Unique identifier for the reflection
            reflection_data: Data associated with the reflection
            parent_ids: List of parent event IDs
            
        Returns:
            The event ID
        """
        event = ProvenanceEvent(
            event_type="reflection",
            event_id=reflection_id,
            data=reflection_data,
            parent_ids=parent_ids or []
        )
        return self.add_event(event)
    
    def add_contradiction_event(self, contradiction_id: str, contradiction_data: Dict[str, Any],
                              parent_ids: Optional[List[str]] = None) -> str:
        """
        Convenience method to add a contradiction event.
        
        Args:
            contradiction_id: Unique identifier for the contradiction
            contradiction_data: Data associated with the contradiction
            parent_ids: List of parent event IDs
            
        Returns:
            The event ID
        """
        event = ProvenanceEvent(
            event_type="contradiction",
            event_id=contradiction_id,
            data=contradiction_data,
            parent_ids=parent_ids or []
        )
        return self.add_event(event)
    
    def add_schema_change_event(self, change_id: str, change_data: Dict[str, Any],
                              parent_ids: Optional[List[str]] = None) -> str:
        """
        Convenience method to add a schema change event.
        
        Args:
            change_id: Unique identifier for the schema change
            change_data: Data associated with the schema change
            parent_ids: List of parent event IDs
            
        Returns:
            The event ID
        """
        event = ProvenanceEvent(
            event_type="schema_change",
            event_id=change_id,
            data=change_data,
            parent_ids=parent_ids or []
        )
        return self.add_event(event)
    
    def add_psi_c_delta_event(self, delta_id: str, delta_value: float,
                            parent_ids: Optional[List[str]] = None) -> str:
        """
        Convenience method to add a ΨC delta event.
        
        Args:
            delta_id: Unique identifier for the ΨC delta
            delta_value: The ΨC delta value
            parent_ids: List of parent event IDs
            
        Returns:
            The event ID
        """
        event = ProvenanceEvent(
            event_type="psi_c_delta",
            event_id=delta_id,
            data={"delta": delta_value},
            parent_ids=parent_ids or []
        )
        return self.add_event(event)
    
    def get_event(self, event_id: str) -> Optional[ProvenanceEvent]:
        """Get an event by ID."""
        return self.events.get(event_id)
    
    def get_event_chain(self, event_id: str) -> List[ProvenanceEvent]:
        """
        Get the chain of events leading to the specified event.
        
        Args:
            event_id: The event ID to trace
            
        Returns:
            List of events in the causal chain, ordered by timestamp
        """
        if event_id not in self.events:
            return []
        
        # Find all ancestors
        ancestors = nx.ancestors(self.graph, event_id)
        ancestors.add(event_id)  # Include the target event
        
        # Get the events and sort by timestamp
        chain = [self.events[eid] for eid in ancestors if eid in self.events]
        chain.sort(key=lambda e: e.timestamp)
        
        return chain
    
    def get_causal_path(self, start_id: str, end_id: str) -> List[ProvenanceEvent]:
        """
        Get the causal path between two events.
        
        Args:
            start_id: Starting event ID
            end_id: Ending event ID
            
        Returns:
            List of events in the path
        """
        if start_id not in self.events or end_id not in self.events:
            return []
        
        try:
            # Find the shortest path
            path_ids = nx.shortest_path(self.graph, start_id, end_id)
            return [self.events[eid] for eid in path_ids if eid in self.events]
        except nx.NetworkXNoPath:
            return []  # No path exists
    
    def get_descendants(self, event_id: str) -> List[ProvenanceEvent]:
        """
        Get all events caused by the specified event.
        
        Args:
            event_id: The event ID
            
        Returns:
            List of descendant events
        """
        if event_id not in self.events:
            return []
        
        # Find all descendants
        descendants = nx.descendants(self.graph, event_id)
        
        # Get the events and sort by timestamp
        result = [self.events[eid] for eid in descendants if eid in self.events]
        result.sort(key=lambda e: e.timestamp)
        
        return result
    
    def visualize(self, output_file: Optional[str] = None, 
                highlight_path: Optional[List[str]] = None,
                highlight_node: Optional[str] = None) -> None:
        """
        Visualize the provenance graph.
        
        Args:
            output_file: Path to save the visualization (if None, displays interactively)
            highlight_path: List of event IDs to highlight as a path
            highlight_node: Event ID to highlight
        """
        if not self.graph.nodes:
            logger.warning("Cannot visualize empty graph")
            return
            
        # Set up colors for event types
        colors = {
            "reflection": "lightblue",
            "contradiction": "salmon",
            "schema_change": "lightgreen",
            "psi_c_delta": "gold",
            "other": "lightgray"
        }
        
        # Prepare node colors
        node_colors = []
        for node in self.graph.nodes:
            event_type = self.graph.nodes[node].get("event_type", "other")
            if highlight_node and node == highlight_node:
                node_colors.append("red")  # Highlighted node
            elif highlight_path and node in highlight_path:
                node_colors.append("purple")  # Highlighted path
            else:
                node_colors.append(colors.get(event_type, colors["other"]))
        
        # Prepare edge colors
        edge_colors = []
        for u, v in self.graph.edges:
            if highlight_path and u in highlight_path and v in highlight_path:
                edge_colors.append("purple")  # Highlighted path
            else:
                edge_colors.append("gray")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, arrows=True)
        
        # Draw labels
        labels = {}
        for node in self.graph.nodes:
            event_type = self.graph.nodes[node].get("event_type", "")
            short_id = node.split("_")[-1] if "_" in node else node[:8]
            labels[node] = f"{event_type}_{short_id}"
        
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=event_type) 
            for event_type, color in colors.items()
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.title("ΨC Provenance Graph")
        plt.axis('off')
        
        # Save or show
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Saved provenance graph to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def export_graphml(self, output_file: str) -> None:
        """
        Export the graph to GraphML format.
        
        Args:
            output_file: Path to save the GraphML file
        """
        try:
            nx.write_graphml(self.graph, output_file)
            logger.info(f"Exported graph to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
    
    def export_dot(self, output_file: str) -> None:
        """
        Export the graph to DOT format for use with Graphviz.
        
        Args:
            output_file: Path to save the DOT file
        """
        try:
            nx.drawing.nx_pydot.write_dot(self.graph, output_file)
            logger.info(f"Exported graph to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
    
    def save_json(self, output_file: str) -> None:
        """
        Save the provenance graph to a JSON file.
        
        Args:
            output_file: Path to save the JSON file
        """
        data = {
            "events": [event.to_dict() for event in self.events.values()],
            "nodes": list(self.graph.nodes),
            "edges": list(self.graph.edges)
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved provenance data to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save provenance data: {e}")
    
    @classmethod
    def load_json(cls, input_file: str) -> 'ProvenanceGraph':
        """
        Load a provenance graph from a JSON file.
        
        Args:
            input_file: Path to the JSON file
            
        Returns:
            The loaded ProvenanceGraph
        """
        graph = cls()
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Load events first
            for event_data in data.get("events", []):
                event = ProvenanceEvent.from_dict(event_data)
                graph.events[event.event_id] = event
            
            # Reconstruct the graph
            for event in graph.events.values():
                graph.graph.add_node(event.event_id, 
                                   event_type=event.event_type,
                                   timestamp=event.timestamp,
                                   data=event.data)
                
                for parent_id in event.parent_ids:
                    if parent_id in graph.events:
                        graph.graph.add_edge(parent_id, event.event_id)
            
            logger.info(f"Loaded provenance graph with {len(graph.events)} events")
            return graph
        except Exception as e:
            logger.error(f"Failed to load provenance data: {e}")
            return cls()
    
    def merge(self, other_graph: 'ProvenanceGraph') -> None:
        """
        Merge another provenance graph into this one.
        
        Args:
            other_graph: The graph to merge
        """
        # Add events from the other graph
        for event_id, event in other_graph.events.items():
            if event_id not in self.events:
                self.events[event_id] = event
        
        # Merge the graphs
        self.graph = nx.compose(self.graph, other_graph.graph)
        
        logger.info(f"Merged graphs, now contains {len(self.events)} events")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the provenance graph.
        
        Returns:
            Dictionary of statistics
        """
        if not self.graph.nodes:
            return {"events": 0, "connections": 0}
        
        # Count event types
        event_types = {}
        for node, attrs in self.graph.nodes(data=True):
            event_type = attrs.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Count connections between different event types
        connections = {}
        for u, v in self.graph.edges:
            u_type = self.graph.nodes[u].get("event_type", "unknown")
            v_type = self.graph.nodes[v].get("event_type", "unknown")
            key = f"{u_type} → {v_type}"
            connections[key] = connections.get(key, 0) + 1
        
        # Calculate maximum chain length
        try:
            longest_path = max(nx.all_simple_paths(self.graph, 
                                                  source=list(self.graph.nodes)[0],
                                                  target=list(self.graph.nodes)[-1]), 
                              key=len, default=[])
            max_chain_length = len(longest_path)
        except nx.NetworkXError:
            # Fall back to approximation
            paths = []
            for node in self.graph.nodes:
                if self.graph.in_degree(node) == 0:  # Source node
                    for target in self.graph.nodes:
                        if self.graph.out_degree(target) == 0:  # Target node
                            try:
                                path = nx.shortest_path(self.graph, node, target)
                                paths.append(path)
                            except nx.NetworkXNoPath:
                                continue
            max_chain_length = max(len(path) for path in paths) if paths else 0
        
        return {
            "events": len(self.events),
            "event_types": event_types,
            "connections": connections,
            "max_chain_length": max_chain_length,
            "density": nx.density(self.graph)
        }
    
    def filter_by_time(self, start_time: Optional[float] = None, 
                     end_time: Optional[float] = None) -> 'ProvenanceGraph':
        """
        Create a new graph filtered by time range.
        
        Args:
            start_time: Start timestamp (None for no lower bound)
            end_time: End timestamp (None for no upper bound)
            
        Returns:
            New ProvenanceGraph with filtered events
        """
        filtered = ProvenanceGraph()
        
        # Filter events
        for event_id, event in self.events.items():
            if ((start_time is None or event.timestamp >= start_time) and
                (end_time is None or event.timestamp <= end_time)):
                
                # Copy the event but clear parent IDs (we'll reconstruct these)
                new_event = ProvenanceEvent(
                    event_type=event.event_type,
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                    data=event.data.copy(),
                    parent_ids=[]  # Clear parent IDs
                )
                
                filtered.events[event_id] = new_event
        
        # Reconstruct the graph with connections
        for event_id, event in self.events.items():
            if event_id in filtered.events:
                # Add this node to the graph
                filtered.graph.add_node(event_id,
                                      event_type=event.event_type,
                                      timestamp=event.timestamp,
                                      data=event.data)
                
                # Add valid edges to the graph
                for parent_id in event.parent_ids:
                    if parent_id in filtered.events:
                        filtered.graph.add_edge(parent_id, event_id)
                        filtered.events[event_id].parent_ids.append(parent_id)
        
        return filtered

def create_provenance_graph_from_reflection_history(
    reflection_history: List[Dict[str, Any]],
    schema_changes: Optional[List[Dict[str, Any]]] = None,
    psi_c_deltas: Optional[List[Dict[str, Any]]] = None
) -> ProvenanceGraph:
    """
    Create a provenance graph from reflection history and related events.
    
    Args:
        reflection_history: List of reflection events
        schema_changes: List of schema change events
        psi_c_deltas: List of ΨC delta events
        
    Returns:
        ProvenanceGraph object
    """
    graph = ProvenanceGraph()
    
    # Track the last event ID for each type to establish parent-child relationships
    last_reflection_id = None
    last_contradiction_id = None
    
    # Add reflection events
    for i, reflection in enumerate(reflection_history):
        reflection_id = f"reflection_{i}"
        graph.add_reflection_event(reflection_id, reflection)
        
        # Set this reflection as parent for the next one
        if last_reflection_id:
            # The current reflection was potentially influenced by the previous one
            reflection["parent_ids"] = [last_reflection_id]
        
        last_reflection_id = reflection_id
        
        # Add contradiction events if present in this reflection
        if "contradictions" in reflection and reflection["contradictions"]:
            for j, contradiction in enumerate(reflection["contradictions"]):
                contradiction_id = f"contradiction_{i}_{j}"
                graph.add_contradiction_event(
                    contradiction_id, 
                    contradiction, 
                    parent_ids=[reflection_id]
                )
                last_contradiction_id = contradiction_id
    
    # Add schema change events if provided
    if schema_changes:
        for i, change in enumerate(schema_changes):
            change_id = f"schema_change_{i}"
            
            # Link to contradiction if timestamp indicates it's related
            parent_ids = []
            if "timestamp" in change and last_contradiction_id:
                last_contradiction = graph.get_event(last_contradiction_id)
                if (last_contradiction and "timestamp" in last_contradiction.data and
                    change["timestamp"] > last_contradiction.data["timestamp"]):
                    parent_ids = [last_contradiction_id]
                elif last_reflection_id:
                    parent_ids = [last_reflection_id]
            
            graph.add_schema_change_event(change_id, change, parent_ids=parent_ids)
    
    # Add ΨC delta events if provided
    if psi_c_deltas:
        for i, delta in enumerate(psi_c_deltas):
            delta_id = f"psi_c_delta_{i}"
            
            # Link to schema change or reflection based on timestamp
            parent_ids = []
            if last_reflection_id:
                parent_ids = [last_reflection_id]
            
            graph.add_psi_c_delta_event(
                delta_id,
                delta.get("delta", 0.0),
                parent_ids=parent_ids
            )
    
    return graph 