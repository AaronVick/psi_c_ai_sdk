#!/usr/bin/env python3
"""
End-to-End Test Script for ΨC-AI SDK Plugins

This script performs a comprehensive workflow test of all three plugins:
- Memory Visualization Plugin
- Schema Export Plugin
- Integration Plugin

It simulates a complete workflow from initialization to shutdown.
"""

import os
import sys
import time
import json
import logging
import random
import tempfile
import argparse
import requests
import networkx as nx
from datetime import datetime
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("end_to_end_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("end_to_end_test")

# Try to import SDK components - fall back to mocks if not available
try:
    from psi_c_ai_sdk.memory.memory import Memory
    from psi_c_ai_sdk.schema.schema import SchemaGraph
    from psi_c_ai_sdk.plugin.base import PluginBase, PluginInfo, PluginHook, PluginType
    from psi_c_ai_sdk.plugin.registry import PluginRegistry
    
    # Import the plugins
    from psi_c_ai_sdk.plugin.examples.memory_visualization_plugin import MemoryVisualizationPlugin
    from psi_c_ai_sdk.plugin.examples.schema_export_plugin import SchemaExportPlugin
    from psi_c_ai_sdk.plugin.examples.integration_plugin import IntegrationPlugin
    
    USING_MOCKS = False
    logger.info("Using real SDK components")
    
except ImportError as e:
    logger.warning(f"SDK components not found: {e}. Using mock implementations.")
    USING_MOCKS = True
    
    # Import mock implementations from test_visualizations.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from test_visualizations import (
            Memory, SchemaGraph, PluginBase, PluginInfo, 
            PluginHook, PluginType, MemoryVisualizationPlugin
        )
    except ImportError:
        logger.error("Failed to import mock classes from test_visualizations.py")
        sys.exit(1)

    # Mock Registry
    class PluginRegistry:
        def __init__(self):
            self.plugins = {}
        
        def register_plugin(self, plugin_cls):
            info = plugin_cls._get_plugin_info()
            self.plugins[info.id] = {
                'class': plugin_cls,
                'info': info,
                'instance': None
            }
            return True
        
        def load_plugin(self, plugin_id, config=None):
            if plugin_id not in self.plugins:
                return None
            
            plugin_cls = self.plugins[plugin_id]['class']
            instance = plugin_cls()
            if config:
                instance.config = config
            
            if instance.initialize():
                self.plugins[plugin_id]['instance'] = instance
                return instance
            return None
        
        def get_plugin(self, plugin_id):
            if plugin_id in self.plugins and 'instance' in self.plugins[plugin_id]:
                return self.plugins[plugin_id]['instance']
            return None
        
        def get_plugins_for_hook(self, hook):
            result = []
            for plugin_id, data in self.plugins.items():
                if 'instance' in data and data['instance'] and hook in data['info'].hooks:
                    result.append(data['instance'])
            return result
        
        def shutdown_all(self):
            for plugin_id, data in self.plugins.items():
                if 'instance' in data and data['instance']:
                    data['instance'].shutdown()
            return True
    
    # Create mock Schema Export Plugin
    class SchemaExportPlugin(PluginBase):
        def __init__(self):
            super().__init__()
            self.logger = logging.getLogger("plugin.SchemaExportPlugin")
            self.stats = {
                "exports_created": 0,
                "nodes_exported": 0,
                "edges_exported": 0
            }
        
        @classmethod
        def _get_plugin_info(cls):
            return PluginInfo(
                id="psi_c_example.schema_export",
                name="Schema Export Plugin",
                version="0.1.0",
                description="A plugin for exporting schema graphs to various formats",
                author="ΨC-AI SDK Team",
                plugin_type="SCHEMA",
                hooks={
                    "POST_REFLECTION",
                    "POST_SCHEMA_UPDATE"
                },
                tags=["schema", "export", "graphml", "gexf", "json", "example"]
            )
        
        def initialize(self):
            self.logger.info("Initializing Schema Export Plugin")
            
            # Set default config if not provided
            if not self.config:
                self.config = {
                    "output_dir": "schema_exports",
                    "export_formats": ["graphml", "gexf", "json"],
                    "min_coherence": 0.2,
                    "include_metadata": True,
                    "auto_export": True
                }
            
            # Create output directory if it doesn't exist
            os.makedirs(self.config["output_dir"], exist_ok=True)
            
            self.logger.info("Schema Export Plugin initialized")
            return True
        
        def shutdown(self):
            self.logger.info(f"Schema Export Plugin shutdown. Stats: {self.stats}")
        
        def post_reflection_handler(self, result=None, schema=None, **kwargs):
            if schema and self.config.get("auto_export", True):
                return self.export_schema(schema)
            return {"exported": False}
        
        def post_schema_update_handler(self, schema=None, **kwargs):
            if schema and self.config.get("auto_export", True):
                return self.export_schema(schema)
            return {"exported": False}
        
        def export_schema(self, schema):
            self.logger.info("Exporting schema...")
            G = schema.get_graph()
            nodes = len(G.nodes())
            edges = len(G.edges())
            
            timestamp = self._get_timestamp()
            export_formats = self.config.get("export_formats", ["graphml"])
            export_paths = {}
            
            for fmt in export_formats:
                path = self._export_to_format(G, fmt, timestamp)
                export_paths[fmt] = path
                
            # Update stats
            self.stats["exports_created"] += 1
            self.stats["nodes_exported"] += nodes
            self.stats["edges_exported"] += edges
            
            self.logger.info(f"Exported schema with {nodes} nodes and {edges} edges to {len(export_formats)} formats")
            
            return {
                "exported": True,
                "export_paths": export_paths,
                "node_count": nodes,
                "edge_count": edges
            }
        
        def _export_to_format(self, G, fmt, timestamp):
            """Export graph to the specified format and return the file path."""
            filename = f"{self.config['output_dir']}/schema_{timestamp}.{fmt}"
            
            # In mock implementation, just write a placeholder file
            with open(filename, 'w') as f:
                f.write(f"Mock schema export - Format: {fmt}\n")
                f.write(f"Nodes: {len(G.nodes())}\n")
                f.write(f"Edges: {len(G.edges())}\n")
                
            return filename
        
        def _get_timestamp(self):
            return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create mock Integration Plugin
    class IntegrationPlugin(PluginBase):
        def __init__(self):
            super().__init__()
            self.logger = logging.getLogger("plugin.IntegrationPlugin")
            self.stats = {
                "api_calls": 0,
                "webhook_calls": 0,
                "errors": 0,
                "last_api_call": None
            }
        
        @classmethod
        def _get_plugin_info(cls):
            return PluginInfo(
                id="psi_c_example.integration",
                name="Integration Plugin",
                version="0.1.0",
                description="A plugin for integrating with external services",
                author="ΨC-AI SDK Team",
                plugin_type="INTEGRATION",
                hooks={
                    "POST_REFLECTION",
                    "POST_MEMORY_ADD",
                    "POST_SCHEMA_UPDATE",
                    "RUNTIME_MONITORING"
                },
                tags=["integration", "api", "webhook", "external", "example"]
            )
        
        def initialize(self):
            self.logger.info("Initializing Integration Plugin")
            
            # Set default config if not provided
            if not self.config:
                self.config = {
                    "api_endpoints": {
                        "memory": "http://example.com/api/memory",
                        "schema": "http://example.com/api/schema",
                        "reflection": "http://example.com/api/reflection",
                        "monitoring": "http://example.com/api/monitoring"
                    },
                    "webhooks": {
                        "memory": "http://example.com/webhook/memory",
                        "schema": "http://example.com/webhook/schema",
                        "reflection": "http://example.com/webhook/reflection",
                        "monitoring": "http://example.com/webhook/monitoring"
                    },
                    "enabled_hooks": [
                        "POST_REFLECTION", 
                        "POST_MEMORY_ADD", 
                        "POST_SCHEMA_UPDATE", 
                        "RUNTIME_MONITORING"
                    ]
                }
            
            self.logger.info("Integration Plugin initialized")
            return True
        
        def shutdown(self):
            self.logger.info(f"Integration Plugin shutdown. Stats: {self.stats}")
        
        def post_reflection_handler(self, result=None, memories=None, schema=None, **kwargs):
            if "POST_REFLECTION" not in self.config.get("enabled_hooks", []):
                return {"skipped": True}
                
            payload = self._prepare_reflection_payload(result, memories, schema)
            
            api_result = self._call_api("reflection", payload)
            webhook_result = self._call_webhook("reflection", payload)
            
            return {
                "api_result": api_result,
                "webhook_result": webhook_result
            }
        
        def post_memory_add_handler(self, memory=None, **kwargs):
            if "POST_MEMORY_ADD" not in self.config.get("enabled_hooks", []):
                return {"skipped": True}
                
            payload = self._prepare_memory_payload(memory)
            
            api_result = self._call_api("memory", payload)
            webhook_result = self._call_webhook("memory", payload)
            
            return {
                "api_result": api_result,
                "webhook_result": webhook_result
            }
        
        def post_schema_update_handler(self, schema=None, **kwargs):
            if "POST_SCHEMA_UPDATE" not in self.config.get("enabled_hooks", []):
                return {"skipped": True}
                
            payload = self._prepare_schema_payload(schema)
            
            api_result = self._call_api("schema", payload)
            webhook_result = self._call_webhook("schema", payload)
            
            return {
                "api_result": api_result,
                "webhook_result": webhook_result
            }
        
        def runtime_monitoring_handler(self, stats=None, **kwargs):
            if "RUNTIME_MONITORING" not in self.config.get("enabled_hooks", []):
                return {"skipped": True}
                
            payload = self._prepare_monitoring_payload(stats)
            
            api_result = self._call_api("monitoring", payload)
            webhook_result = self._call_webhook("monitoring", payload)
            
            return {
                "api_result": api_result,
                "webhook_result": webhook_result
            }
        
        def _prepare_reflection_payload(self, result, memories, schema):
            """Prepare payload for reflection events."""
            # In mock implementation, just return a simple dict
            return {
                "event_type": "reflection",
                "timestamp": datetime.now().isoformat(),
                "coherence_before": result.get("coherence_before", 0) if result else 0,
                "coherence_after": result.get("coherence_after", 0) if result else 0,
                "memories_count": len(memories) if memories else 0,
                "schema_nodes": len(schema.get_graph().nodes()) if schema else 0,
                "schema_edges": len(schema.get_graph().edges()) if schema else 0
            }
        
        def _prepare_memory_payload(self, memory):
            """Prepare payload for memory events."""
            # In mock implementation, just return a simple dict
            return {
                "event_type": "memory_added",
                "timestamp": datetime.now().isoformat(),
                "memory_id": getattr(memory, "id", "unknown"),
                "memory_type": getattr(memory, "memory_type", "unknown"),
                "importance": getattr(memory, "importance", 0)
            }
        
        def _prepare_schema_payload(self, schema):
            """Prepare payload for schema events."""
            # In mock implementation, just return a simple dict
            G = schema.get_graph() if schema else None
            return {
                "event_type": "schema_updated",
                "timestamp": datetime.now().isoformat(),
                "nodes": len(G.nodes()) if G else 0,
                "edges": len(G.edges()) if G else 0
            }
        
        def _prepare_monitoring_payload(self, stats):
            """Prepare payload for monitoring events."""
            # In mock implementation, just return a simple dict
            return {
                "event_type": "monitoring",
                "timestamp": datetime.now().isoformat(),
                "memory_usage": stats.get("memory_usage", 0) if stats else 0,
                "cpu_usage": stats.get("cpu_usage", 0) if stats else 0,
                "uptime": stats.get("uptime", 0) if stats else 0
            }
        
        def _call_api(self, endpoint_type, payload):
            """Call API endpoint with payload."""
            self.logger.info(f"Calling {endpoint_type} API")
            
            # In mock implementation, just log and return success
            self.stats["api_calls"] += 1
            self.stats["last_api_call"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "status_code": 200,
                "endpoint_type": endpoint_type,
                "timestamp": datetime.now().isoformat()
            }
        
        def _call_webhook(self, webhook_type, payload):
            """Call webhook with payload."""
            self.logger.info(f"Calling {webhook_type} webhook")
            
            # In mock implementation, just log and return success
            self.stats["webhook_calls"] += 1
            
            return {
                "success": True,
                "status_code": 200,
                "webhook_type": webhook_type,
                "timestamp": datetime.now().isoformat()
            }


# Helper functions for creating test data
def create_test_memories(count=20):
    """Create test memory objects."""
    memory_types = ["episodic", "semantic", "procedural", "declarative", "unknown"]
    memories = []
    
    for i in range(count):
        importance = random.uniform(0.1, 1.0)
        memory_type = random.choice(memory_types)
        created_at = datetime.now().isoformat()
        tags = random.sample(["important", "personal", "work", "family", "travel", "idea"], 
                             k=random.randint(0, 3))
        
        content = f"Test memory {i} with {memory_type} type and {importance:.2f} importance."
        
        memories.append(Memory(
            id=f"memory_{i}",
            content=content,
            importance=importance,
            memory_type=memory_type,
            created_at=created_at,
            tags=tags
        ))
    
    return memories

def create_test_schema(nodes=30, edges=50):
    """Create a test schema graph."""
    schema = SchemaGraph()
    G = nx.Graph()
    
    # Define node types for the graph
    node_types = ["concept", "entity", "event", "relation", "memory"]
    
    # Create nodes
    for i in range(nodes):
        node_type = random.choice(node_types)
        G.add_node(f"node_{i}", 
                   label=f"{node_type.capitalize()} {i}",
                   type=node_type,
                   importance=random.uniform(0.1, 1.0))
    
    # Create edges
    edge_count = 0
    attempts = 0
    all_nodes = list(G.nodes())
    
    while edge_count < edges and attempts < edges * 10:
        attempts += 1
        u, v = random.sample(all_nodes, 2)
        
        # Avoid self-loops and existing edges
        if u != v and not G.has_edge(u, v):
            weight = random.uniform(0.1, 1.0)
            G.add_edge(u, v, weight=weight)
            edge_count += 1
    
    schema.set_graph(G)
    return schema

def simulate_reflection_cycle(memories, schema):
    """Simulate a reflection cycle."""
    # In a real implementation, this would perform actual reflection
    # For our test, we'll just modify the schema a bit
    G = schema.get_graph()
    
    # Add some edges based on memory types
    memory_map = {}
    for i, memory in enumerate(memories):
        memory_type = getattr(memory, "memory_type", "unknown")
        if memory_type not in memory_map:
            memory_map[memory_type] = []
        memory_map[memory_type].append(i)
    
    # Add schema nodes for each memory type
    for memory_type, indices in memory_map.items():
        # Add a concept node for this memory type
        concept_node = f"concept_{memory_type}"
        if not G.has_node(concept_node):
            G.add_node(concept_node, 
                      label=f"{memory_type.capitalize()} Concept",
                      type="concept",
                      importance=0.8)
        
        # Connect some memories to this concept
        for idx in indices[:min(3, len(indices))]:
            memory_node = f"memory_{idx}"
            if not G.has_node(memory_node):
                G.add_node(memory_node,
                          label=f"Memory {idx}",
                          type="memory",
                          importance=getattr(memories[idx], "importance", 0.5))
            
            G.add_edge(concept_node, memory_node, weight=0.7)
    
    # Connect concepts that might be related
    concepts = [n for n in G.nodes() if G.nodes[n].get('type') == 'concept']
    if len(concepts) >= 2:
        for i in range(min(3, len(concepts))):
            u, v = random.sample(concepts, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=0.6)
    
    # Return a simulated result
    return {
        "coherence_before": 0.4,
        "coherence_after": 0.7,
        "memories_processed": len(memories),
        "schema_nodes_added": 5,
        "schema_edges_added": 8
    }

def simulate_system_stats():
    """Simulate system stats for monitoring."""
    return {
        "memory_usage": random.uniform(100, 500),  # MB
        "cpu_usage": random.uniform(5, 30),  # %
        "uptime": random.randint(60, 3600),  # seconds
        "thread_count": random.randint(2, 8),
        "active_plugins": 3
    }

def measure_execution_time(func, *args, **kwargs):
    """Measure and log execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def run_end_to_end_test(test_dir, memory_count=20, schema_nodes=30, schema_edges=50):
    """Run a complete end-to-end test of all plugins."""
    logger.info("="*80)
    logger.info("STARTING END-TO-END PLUGIN TEST")
    logger.info("="*80)
    
    # Print system information
    logger.info(f"Test directory: {test_dir}")
    logger.info(f"Using mock implementations: {USING_MOCKS}")
    logger.info(f"Memory count: {memory_count}")
    logger.info(f"Schema nodes: {schema_nodes}, edges: {schema_edges}")
    
    test_metrics = {
        "initialization_time": 0,
        "hook_execution_times": {},
        "memory_usage": {},
        "plugin_stats": {}
    }
    
    # Create the plugin registry
    registry = PluginRegistry()
    
    # Register plugins
    logger.info("Registering plugins...")
    registry.register_plugin(MemoryVisualizationPlugin)
    registry.register_plugin(SchemaExportPlugin)
    registry.register_plugin(IntegrationPlugin)
    
    # Create plugin configurations
    logger.info("Creating plugin configurations...")
    
    viz_config = {
        "output_dir": os.path.join(test_dir, "visualizations"),
        "min_coherence": 0.3,
        "show_plots": False,
        "file_format": "png",
        "max_nodes": 50
    }
    
    export_config = {
        "output_dir": os.path.join(test_dir, "schema_exports"),
        "export_formats": ["graphml", "gexf", "json"],
        "min_coherence": 0.2,
        "include_metadata": True,
        "auto_export": True
    }
    
    integration_config = {
        "api_endpoints": {
            "memory": "http://example.com/api/memory",
            "schema": "http://example.com/api/schema",
            "reflection": "http://example.com/api/reflection",
            "monitoring": "http://example.com/api/monitoring"
        },
        "webhooks": {
            "memory": "http://example.com/webhook/memory",
            "schema": "http://example.com/webhook/schema",
            "reflection": "http://example.com/webhook/reflection",
            "monitoring": "http://example.com/webhook/monitoring"
        },
        "enabled_hooks": [
            "POST_REFLECTION", 
            "POST_MEMORY_ADD", 
            "POST_SCHEMA_UPDATE", 
            "RUNTIME_MONITORING"
        ]
    }
    
    # Load plugins with configuration
    logger.info("Loading plugins...")
    start_time = time.time()
    
    viz_plugin = registry.load_plugin("psi_c_example.memory_visualization", viz_config)
    export_plugin = registry.load_plugin("psi_c_example.schema_export", export_config)
    integration_plugin = registry.load_plugin("psi_c_example.integration", integration_config)
    
    test_metrics["initialization_time"] = time.time() - start_time
    logger.info(f"Plugins loaded in {test_metrics['initialization_time']:.2f} seconds")
    
    # Verify plugins loaded successfully
    if not viz_plugin or not export_plugin or not integration_plugin:
        logger.error("Failed to load one or more plugins")
        return False
    
    # Create test data
    logger.info("Creating test data...")
    memories = create_test_memories(memory_count)
    schema = create_test_schema(schema_nodes, schema_edges)
    logger.info(f"Created {len(memories)} memories and schema with {schema_nodes} nodes, {schema_edges} edges")
    
    # Test adding memories one by one
    logger.info("Testing memory addition hooks...")
    memory_add_times = []
    
    for i, memory in enumerate(memories[:5]):  # Test with first 5 memories
        logger.info(f"Adding memory {i}: {memory.id}")
        
        plugins = registry.get_plugins_for_hook("POST_MEMORY_ADD")
        for plugin in plugins:
            start_time = time.time()
            result = plugin.post_memory_add_handler(memory=memory)
            elapsed = time.time() - start_time
            memory_add_times.append(elapsed)
            logger.info(f"  {plugin.__class__.__name__} processed memory in {elapsed:.4f} seconds")
    
    test_metrics["hook_execution_times"]["post_memory_add"] = {
        "min": min(memory_add_times) if memory_add_times else 0,
        "max": max(memory_add_times) if memory_add_times else 0,
        "avg": sum(memory_add_times) / len(memory_add_times) if memory_add_times else 0
    }
    
    # Test schema update hooks
    logger.info("Testing schema update hooks...")
    
    plugins = registry.get_plugins_for_hook("POST_SCHEMA_UPDATE")
    schema_update_times = []
    
    for plugin in plugins:
        start_time = time.time()
        result = plugin.post_schema_update_handler(schema=schema)
        elapsed = time.time() - start_time
        schema_update_times.append(elapsed)
        logger.info(f"  {plugin.__class__.__name__} processed schema update in {elapsed:.4f} seconds")
    
    test_metrics["hook_execution_times"]["post_schema_update"] = {
        "min": min(schema_update_times) if schema_update_times else 0,
        "max": max(schema_update_times) if schema_update_times else 0,
        "avg": sum(schema_update_times) / len(schema_update_times) if schema_update_times else 0
    }
    
    # Test reflection cycle
    logger.info("Testing reflection cycle hooks...")
    
    # Simulate a reflection cycle
    reflection_result = simulate_reflection_cycle(memories, schema)
    
    plugins = registry.get_plugins_for_hook("POST_REFLECTION")
    reflection_times = []
    
    for plugin in plugins:
        start_time = time.time()
        result = plugin.post_reflection_handler(
            result=reflection_result,
            memories=memories,
            schema=schema
        )
        elapsed = time.time() - start_time
        reflection_times.append(elapsed)
        logger.info(f"  {plugin.__class__.__name__} processed reflection in {elapsed:.4f} seconds")
    
    test_metrics["hook_execution_times"]["post_reflection"] = {
        "min": min(reflection_times) if reflection_times else 0,
        "max": max(reflection_times) if reflection_times else 0,
        "avg": sum(reflection_times) / len(reflection_times) if reflection_times else 0
    }
    
    # Test monitoring hooks
    logger.info("Testing monitoring hooks...")
    
    system_stats = simulate_system_stats()
    
    plugins = registry.get_plugins_for_hook("RUNTIME_MONITORING")
    monitoring_times = []
    
    for plugin in plugins:
        start_time = time.time()
        result = plugin.runtime_monitoring_handler(stats=system_stats)
        elapsed = time.time() - start_time
        monitoring_times.append(elapsed)
        logger.info(f"  {plugin.__class__.__name__} processed monitoring in {elapsed:.4f} seconds")
    
    test_metrics["hook_execution_times"]["runtime_monitoring"] = {
        "min": min(monitoring_times) if monitoring_times else 0,
        "max": max(monitoring_times) if monitoring_times else 0,
        "avg": sum(monitoring_times) / len(monitoring_times) if monitoring_times else 0
    }
    
    # Gather plugin statistics
    logger.info("Gathering plugin statistics...")
    
    if hasattr(viz_plugin, 'stats'):
        test_metrics["plugin_stats"]["visualization"] = viz_plugin.stats
    
    if hasattr(export_plugin, 'stats'):
        test_metrics["plugin_stats"]["export"] = export_plugin.stats
    
    if hasattr(integration_plugin, 'stats'):
        test_metrics["plugin_stats"]["integration"] = integration_plugin.stats
    
    # Shutdown plugins
    logger.info("Shutting down plugins...")
    registry.shutdown_all()
    
    # Log test results
    logger.info("="*80)
    logger.info("END-TO-END TEST RESULTS")
    logger.info("="*80)
    
    logger.info(f"Initialization time: {test_metrics['initialization_time']:.4f} seconds")
    
    logger.info("Hook execution times:")
    for hook, times in test_metrics["hook_execution_times"].items():
        logger.info(f"  {hook}: min={times['min']:.4f}s, max={times['max']:.4f}s, avg={times['avg']:.4f}s")
    
    logger.info("Plugin statistics:")
    for plugin_name, stats in test_metrics["plugin_stats"].items():
        logger.info(f"  {plugin_name}: {stats}")
    
    # Save test metrics to file
    metrics_file = os.path.join(test_dir, "test_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"Test metrics saved to {metrics_file}")
    logger.info("End-to-end test completed successfully")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end tests for ΨC-AI SDK plugins")
    parser.add_argument("--test-dir", type=str, help="Directory for test outputs")
    parser.add_argument("--memories", type=int, default=20, help="Number of test memories to create")
    parser.add_argument("--nodes", type=int, default=30, help="Number of schema nodes to create")
    parser.add_argument("--edges", type=int, default=50, help="Number of schema edges to create")
    
    args = parser.parse_args()
    
    # Create a test directory if not specified
    test_dir = args.test_dir
    if not test_dir:
        test_dir = tempfile.mkdtemp(prefix="psi_c_plugin_test_")
    else:
        os.makedirs(test_dir, exist_ok=True)
    
    success = run_end_to_end_test(
        test_dir=test_dir,
        memory_count=args.memories,
        schema_nodes=args.nodes,
        schema_edges=args.edges
    )
    
    sys.exit(0 if success else 1) 