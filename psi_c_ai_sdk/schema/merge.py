# psi_c_ai_sdk/schema/merge.py

from psi_c_ai_sdk.schema.schema_graph import SchemaGraph

def merge_schemas(a: SchemaGraph, b: SchemaGraph) -> SchemaGraph:
    merged = SchemaGraph()

    # Combine nodes
    for node in a.get_all_nodes() + b.get_all_nodes():
        merged.add_node(node, overwrite=True)

    # Combine edges
    for edge in a.get_all_edges() + b.get_all_edges():
        merged.add_edge(edge.source, edge.target, edge.label)

    # Log conflicts
    for key in set(a.nodes).intersection(b.nodes):
        if a.nodes[key] != b.nodes[key]:
            merged.add_conflict(key, a.nodes[key], b.nodes[key])

    return merged
