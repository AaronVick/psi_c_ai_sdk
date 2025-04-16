"""
Schema Export Plugin for ΨC-AI SDK

This module provides a plugin for exporting schema graphs to various file formats,
including GraphML, GEXF, and JSON, for external analysis and visualization.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Union
import networkx as nx

from psi_c_ai_sdk.plugin.base import (
    PluginBase,
    PluginInfo,
    PluginHook,
    PluginType,
    create_plugin_id
)
from psi_c_ai_sdk.schema.schema import SchemaGraph


class SchemaExportPlugin(PluginBase):
    """
    Plugin for exporting schema graphs to various file formats.
    
    This plugin provides functionality to export schema graphs to common
    graph file formats for external analysis and visualization.
    """
    
    @classmethod
    def _get_plugin_info(cls) -> PluginInfo:
        """Get metadata about the plugin."""
        return PluginInfo(
            id=create_plugin_id("schema_export", "psi_c_example"),
            name="Schema Export Plugin",
            version="0.1.0",
            description="A plugin for exporting schema graphs to various file formats",
            author="ΨC-AI SDK Team",
            plugin_type=PluginType.SCHEMA,
            hooks={
                PluginHook.POST_SCHEMA_UPDATE,
                PluginHook.POST_REFLECTION
            },
            tags=["schema", "export", "graphml", "gexf", "json", "example"]
        )
    
    def _register_hooks(self) -> Dict[PluginHook, Any]:
        """Register the hooks that this plugin implements."""
        return {
            PluginHook.POST_SCHEMA_UPDATE: self.post_schema_update_handler,
            PluginHook.POST_REFLECTION: self.post_reflection_handler
        }
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        self.logger.info("Initializing Schema Export Plugin")
        
        # Set default config if not provided
        if not self.config:
            self.config = {
                "output_dir": "schema_exports",
                "export_formats": ["graphml", "gexf", "json"],
                "min_coherence": 0.2,
                "include_metadata": True,
                "auto_export": True,
                "metadata_fields": ["importance", "memory_type", "created_at", "updated_at"]
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Track statistics
        self.stats = {
            "exports_created": 0,
            "nodes_exported": 0,
            "edges_exported": 0
        }
        
        self.logger.info("Schema Export Plugin initialized")
        return True
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.logger.info(
            f"Schema Export Plugin shutdown. Stats: {self.stats}"
        )
    
    def post_schema_update_handler(
        self, 
        schema: SchemaGraph,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-schema-update events.
        
        This method is called after the schema graph is updated
        and exports the schema to the configured formats.
        
        Args:
            schema: Updated schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        self.logger.info("Post-schema-update: Exporting schema")
        
        if self.config.get("auto_export", True):
            export_results = self._export_schema(schema)
            return export_results
        
        return {"auto_export": False}
    
    def post_reflection_handler(
        self,
        result: Dict[str, Any],
        schema: Optional[SchemaGraph] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handler for post-reflection events.
        
        This method is called after the reflection cycle completes
        and can export the updated schema if available.
        
        Args:
            result: Results from the reflection cycle
            schema: Updated schema graph
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with handler results
        """
        if schema and self.config.get("auto_export", True):
            self.logger.info("Post-reflection: Exporting schema")
            export_results = self._export_schema(schema)
            return export_results
        
        return {"exported": False, "reason": "No schema provided or auto-export disabled"}
    
    def export_schema(self, schema: SchemaGraph) -> Dict[str, Any]:
        """
        Public method to manually export a schema.
        
        Args:
            schema: Schema graph to export
            
        Returns:
            Dictionary with export results
        """
        return self._export_schema(schema)
    
    def _export_schema(self, schema: SchemaGraph) -> Dict[str, Any]:
        """
        Export schema to various formats.
        
        Args:
            schema: Schema graph to export
            
        Returns:
            Dictionary with export results
        """
        self.logger.info("Exporting schema to configured formats")
        
        # Extract the graph from the schema
        G = schema.get_graph() if hasattr(schema, 'get_graph') else nx.Graph()
        
        # If the graph is empty or not available, return error
        if not G or len(G.nodes()) == 0:
            self.logger.warning("Schema graph is empty, nothing to export")
            return {"exported": False, "reason": "Empty schema graph"}
        
        # Filter edges by minimum coherence if configured
        min_coherence = self.config.get("min_coherence", 0.2)
        if min_coherence > 0:
            edges_to_remove = []
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 0)
                if weight < min_coherence:
                    edges_to_remove.append((u, v))
            
            G.remove_edges_from(edges_to_remove)
            self.logger.info(f"Removed {len(edges_to_remove)} edges with coherence < {min_coherence}")
        
        # Add schema metadata if needed
        if self.config.get("include_metadata", True):
            G.graph['exported_by'] = 'SchemaExportPlugin'
            G.graph['plugin_version'] = self._get_plugin_info().version
            
            # Add node degree statistics
            degrees = dict(G.degree())
            G.graph['max_degree'] = max(degrees.values()) if degrees else 0
            G.graph['avg_degree'] = sum(degrees.values()) / len(degrees) if degrees else 0
            
            # Add edge weight statistics
            weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
            G.graph['max_coherence'] = max(weights) if weights else 0
            G.graph['avg_coherence'] = sum(weights) / len(weights) if weights else 0
            G.graph['edge_count'] = len(G.edges())
            G.graph['node_count'] = len(G.nodes())
        
        # Ensure all nodes and edges have string IDs (required for some formats)
        G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
        
        # Generate timestamp for filenames
        timestamp = self._get_timestamp()
        
        # Export to each configured format
        export_paths = {}
        export_stats = {}
        
        export_formats = self.config.get("export_formats", ["graphml"])
        
        for export_format in export_formats:
            try:
                filepath = self._export_to_format(G, export_format, timestamp)
                if filepath:
                    export_paths[export_format] = filepath
            except Exception as e:
                self.logger.error(f"Error exporting to {export_format}: {str(e)}")
                export_stats[f"{export_format}_error"] = str(e)
        
        # Update stats
        self.stats["exports_created"] += len(export_paths)
        self.stats["nodes_exported"] += len(G.nodes())
        self.stats["edges_exported"] += len(G.edges())
        
        return {
            "exported": len(export_paths) > 0,
            "export_paths": export_paths,
            "node_count": len(G.nodes()),
            "edge_count": len(G.edges()),
            "export_stats": export_stats
        }
    
    def _export_to_format(
        self, 
        G: nx.Graph, 
        export_format: str,
        timestamp: str
    ) -> Optional[str]:
        """
        Export graph to a specific format.
        
        Args:
            G: Graph to export
            export_format: Format to export to
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the exported file, or None if export failed
        """
        filename = f"{self.config['output_dir']}/schema_{timestamp}"
        
        if export_format.lower() == "graphml":
            filepath = f"{filename}.graphml"
            nx.write_graphml(G, filepath)
            return filepath
            
        elif export_format.lower() == "gexf":
            filepath = f"{filename}.gexf"
            nx.write_gexf(G, filepath)
            return filepath
            
        elif export_format.lower() == "json":
            filepath = f"{filename}.json"
            # Convert to node-link format
            data = nx.node_link_data(G)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return filepath
            
        elif export_format.lower() == "gml":
            filepath = f"{filename}.gml"
            nx.write_gml(G, filepath)
            return filepath
            
        elif export_format.lower() == "pajek":
            filepath = f"{filename}.net"
            nx.write_pajek(G, filepath)
            return filepath
            
        elif export_format.lower() == "csv":
            # Export nodes
            nodes_filepath = f"{filename}_nodes.csv"
            edges_filepath = f"{filename}_edges.csv"
            
            # Write nodes
            with open(nodes_filepath, 'w') as f:
                f.write("id,")
                if len(G.nodes()) > 0:
                    # Write header based on attributes of first node
                    first_node = list(G.nodes(data=True))[0]
                    attrs = list(first_node[1].keys())
                    f.write(",".join(attrs))
                    f.write("\n")
                    
                    # Write node data
                    for node_id, attrs in G.nodes(data=True):
                        f.write(f"{node_id}")
                        for attr in attrs:
                            f.write(f",{attrs[attr]}")
                        f.write("\n")
            
            # Write edges
            with open(edges_filepath, 'w') as f:
                f.write("source,target,weight\n")
                for source, target, data in G.edges(data=True):
                    weight = data.get('weight', 0)
                    f.write(f"{source},{target},{weight}\n")
            
            return f"{filename}_nodes.csv, {filename}_edges.csv"
        
        else:
            self.logger.warning(f"Unsupported export format: {export_format}")
            return None
    
    def _get_timestamp(self) -> str:
        """Generate a timestamp for filenames."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# This allows the plugin to be loaded directly
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and test the plugin
    plugin = SchemaExportPlugin()
    plugin.initialize()
    
    # Create a simple schema graph for testing
    test_schema = SchemaGraph()
    
    # For testing purposes, create a sample graph
    try:
        G = nx.Graph()
        G.add_node("concept1", label="Important Concept", type="concept", importance=0.9)
        G.add_node("concept2", label="Related Concept", type="concept", importance=0.7)
        G.add_node("memory1", label="Memory 1", type="memory", importance=0.8, memory_type="episodic")
        G.add_edge("concept1", "concept2", weight=0.65)
        G.add_edge("concept1", "memory1", weight=0.8)
        
        # Attach the graph to the schema for testing
        if hasattr(test_schema, 'set_graph'):
            test_schema.set_graph(G)
        else:
            # Simulated approach if set_graph is not available
            setattr(test_schema, '_graph', G)
            setattr(test_schema, 'get_graph', lambda: getattr(test_schema, '_graph'))
    except Exception as e:
        print(f"Warning: Couldn't create test graph: {e}")
    
    # Test the export functionality
    result = plugin.export_schema(test_schema)
    print(f"Export result: {result}")
    
    plugin.shutdown() 