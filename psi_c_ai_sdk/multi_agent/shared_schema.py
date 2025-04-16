# psi_c_ai_sdk/multi_agent/shared_schema.py

import networkx as nx
from typing import List, Dict, Any
from psi_c_ai_sdk.schema.schema import SchemaGraph

class SharedSchemaBuilder:
    def __init__(self):
        self.shared_graph = nx.Graph()
        self.agent_graphs: Dict[str, SchemaGraph] = {}

    def add_agent_schema(self, agent_id: str, schema_graph: SchemaGraph):
        """Add an agent's schema into the shared graph"""
        self.agent_graphs[agent_id] = schema_graph
        for node, data in schema_graph.graph.nodes(data=True):
            self.shared_graph.add_node(f"{agent_id}:{node}", **data)
        for u, v, data in schema_graph.graph.edges(data=True):
            self.shared_graph.add_edge(f"{agent_id}:{u}", f"{agent_id}:{v}", **data)

    def visualize_shared_schema(self):
        """Return NetworkX graph representing the shared schema space"""
        return self.shared_graph

    def find_overlap_nodes(self) -> List[str]:
        """Detect overlapping concept names across agents"""
        concept_counts = {}
        for agent_id, schema in self.agent_graphs.items():
            for node in schema.graph.nodes:
                concept = schema.graph.nodes[node].get("label", node)
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        return [c for c, count in concept_counts.items() if count > 1]
