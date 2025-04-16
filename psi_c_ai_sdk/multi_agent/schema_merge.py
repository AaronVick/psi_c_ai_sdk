# psi_c_ai_sdk/multi_agent/schema_merge.py

from psi_c_ai_sdk.schema.schema import SchemaGraph
from psi_c_ai_sdk.schema.fingerprint import fingerprint_schema

class SchemaMerger:
    @staticmethod
    def merge_schemas(
        schema_a: SchemaGraph, 
        schema_b: SchemaGraph,
        merge_policy: str = "conservative"
    ) -> SchemaGraph:
        """
        Merge two schemas based on specified policy:
        - 'conservative': keep common coherent beliefs
        - 'inclusive': allow all beliefs, resolve contradictions later
        """
        merged = SchemaGraph()

        for node in schema_a.get_all_nodes():
            merged.add_node(node)

        for node in schema_b.get_all_nodes():
            if merge_policy == "inclusive" or merged.has_node(node.id) is False:
                merged.add_node(node)

        for edge in schema_a.get_all_edges() + schema_b.get_all_edges():
            merged.add_edge(edge)

        merged.compute_fingerprint()
        return merged
