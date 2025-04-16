#!/usr/bin/env python3
"""
Comprehensive test script for ΨC-AI SDK plugins.

This script tests all three example plugins:
- MemoryVisualizationPlugin
- SchemaExportPlugin
- IntegrationPlugin

It runs a series of tests to verify functionality, error handling, and outputs.
"""

import os
import sys
import logging
import tempfile
import unittest
import shutil
import traceback
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("plugin_tests")

# Mock the imports that might be missing
sys.modules['psi_c_ai_sdk.memory.memory'] = MagicMock()
sys.modules['psi_c_ai_sdk.schema.schema'] = MagicMock()
sys.modules['psi_c_ai_sdk.plugin.base'] = MagicMock()
sys.modules['psi_c_ai_sdk.embeddings'] = MagicMock()
sys.modules['psi_c_ai_sdk.plugin.registry'] = MagicMock()

# Create mock classes for core SDK components
class MockMemory:
    """Mock Memory class for testing."""
    def __init__(self, id=None, content=None, importance=0.5, memory_type="unknown", created_at=None, tags=None):
        self.id = id or "mock_memory"
        self.content = content or "Mock memory content"
        self.importance = importance
        self.memory_type = memory_type
        self.created_at = created_at or "2023-01-01T00:00:00"
        self.tags = tags or []

class MockSchemaGraph:
    """Mock SchemaGraph class for testing."""
    def __init__(self):
        self._graph = nx.Graph()
    
    def get_graph(self):
        return self._graph
    
    def set_graph(self, graph):
        self._graph = graph

# Mock PluginBase and related classes
class MockPluginHook:
    """Mock PluginHook enum for testing."""
    POST_REFLECTION = "POST_REFLECTION"
    POST_MEMORY_ADD = "POST_MEMORY_ADD"
    POST_SCHEMA_UPDATE = "POST_SCHEMA_UPDATE"
    RUNTIME_MONITORING = "RUNTIME_MONITORING"

class MockPluginType:
    """Mock PluginType enum for testing."""
    VISUALIZATION = "VISUALIZATION"
    SCHEMA = "SCHEMA"
    INTEGRATION = "INTEGRATION"

class MockPluginInfo:
    """Mock PluginInfo class for testing."""
    def __init__(self, id, name, version, description, author, plugin_type, hooks, tags):
        self.id = id
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.plugin_type = plugin_type
        self.hooks = hooks
        self.tags = tags

class MockPluginBase:
    """Mock PluginBase class for testing."""
    def __init__(self):
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self.config = {}
    
    def initialize(self):
        return True
    
    def shutdown(self):
        pass

# Insert the mocks into the modules
sys.modules['psi_c_ai_sdk.memory.memory'].Memory = MockMemory
sys.modules['psi_c_ai_sdk.schema.schema'].SchemaGraph = MockSchemaGraph
sys.modules['psi_c_ai_sdk.plugin.base'].PluginBase = MockPluginBase
sys.modules['psi_c_ai_sdk.plugin.base'].PluginInfo = MockPluginInfo
sys.modules['psi_c_ai_sdk.plugin.base'].PluginHook = MockPluginHook
sys.modules['psi_c_ai_sdk.plugin.base'].PluginType = MockPluginType
sys.modules['psi_c_ai_sdk.plugin.base'].create_plugin_id = lambda a, b: f"{b}.{a}"

# Now try to import the plugins
try:
    # Use exec to dynamically execute the plugin source code
    # This is needed because we're mocking the imports
    
    # Load Memory Visualization Plugin
    with open('psi_c_ai_sdk/plugin/examples/memory_visualization_plugin.py', 'r') as f:
        memory_viz_source = f.read()
    memory_viz_globals = {
        'logging': logging,
        'os': os,
        'typing': __import__('typing'),
        'matplotlib': MagicMock(),
        'plt': MagicMock(),
        'nx': nx,
        'np': MagicMock(),
        'PluginBase': MockPluginBase,
        'PluginInfo': MockPluginInfo,
        'PluginHook': MockPluginHook,
        'PluginType': MockPluginType,
        'create_plugin_id': lambda a, b: f"{b}.{a}",
        'Memory': MockMemory,
        'SchemaGraph': MockSchemaGraph
    }
    exec(memory_viz_source, memory_viz_globals)
    MemoryVisualizationPlugin = memory_viz_globals['MemoryVisualizationPlugin']
    
    # Load Schema Export Plugin
    with open('psi_c_ai_sdk/plugin/examples/schema_export_plugin.py', 'r') as f:
        schema_export_source = f.read()
    schema_export_globals = {
        'json': __import__('json'),
        'logging': logging,
        'os': os,
        'typing': __import__('typing'),
        'nx': nx,
        'PluginBase': MockPluginBase,
        'PluginInfo': MockPluginInfo,
        'PluginHook': MockPluginHook,
        'PluginType': MockPluginType,
        'create_plugin_id': lambda a, b: f"{b}.{a}",
        'SchemaGraph': MockSchemaGraph
    }
    exec(schema_export_source, schema_export_globals)
    SchemaExportPlugin = schema_export_globals['SchemaExportPlugin']
    
    # Load Integration Plugin
    with open('psi_c_ai_sdk/plugin/examples/integration_plugin.py', 'r') as f:
        integration_source = f.read()
    integration_globals = {
        'json': __import__('json'),
        'logging': logging,
        'os': os,
        'requests': __import__('requests'),
        'time': __import__('time'),
        'typing': __import__('typing'),
        'urljoin': __import__('urllib.parse').parse.urljoin,
        'PluginBase': MockPluginBase,
        'PluginInfo': MockPluginInfo,
        'PluginHook': MockPluginHook,
        'PluginType': MockPluginType,
        'create_plugin_id': lambda a, b: f"{b}.{a}",
        'Memory': MockMemory,
        'SchemaGraph': MockSchemaGraph
    }
    exec(integration_source, integration_globals)
    IntegrationPlugin = integration_globals['IntegrationPlugin']
    
except Exception as e:
    logger.error(f"Failed to load plugins: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)


class PluginTestCase(unittest.TestCase):
    """Base test case with common setup/teardown for all plugin tests."""
    
    def setUp(self):
        """Set up temporary directories and common test data."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        logger.info(f"Created test directory: {self.test_dir}")
        
        # Create test memory objects
        self.test_memories = [
            MockMemory(id="mem1", content="Test memory 1", importance=0.8, memory_type="episodic"),
            MockMemory(id="mem2", content="Test memory 2", importance=0.6, memory_type="semantic"),
            MockMemory(id="mem3", content="Test memory 3", importance=0.7, memory_type="procedural"),
            MockMemory(id="mem4", content="Test memory 4", importance=0.5, memory_type="episodic"),
            MockMemory(id="mem5", content="Test memory 5", importance=0.9, memory_type="semantic")
        ]
        
        # Create a test schema graph
        self.schema_graph = MockSchemaGraph()
        
        # Set up a NetworkX graph and attach it to the schema
        self.G = nx.Graph()
        self.G.add_node("concept1", label="Important Concept", type="concept", importance=0.9)
        self.G.add_node("concept2", label="Related Concept", type="concept", importance=0.7)
        self.G.add_node("memory1", label="Memory 1", type="memory", importance=0.8, memory_type="episodic")
        self.G.add_edge("concept1", "concept2", weight=0.65)
        self.G.add_edge("concept1", "memory1", weight=0.8)
        
        # Attach the graph to the schema
        self.schema_graph.set_graph(self.G)
        
        # Standard reflection result for testing
        self.reflection_result = {
            "coherence_before": 0.5,
            "coherence_after": 0.7,
            "memories_processed": 5,
            "schema_nodes_added": 2,
            "schema_edges_added": 1
        }
    
    def tearDown(self):
        """Clean up temporary directories and resources."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        logger.info(f"Removed test directory: {self.test_dir}")


class TestMemoryVisualizationPlugin(PluginTestCase):
    """Tests for the Memory Visualization Plugin."""
    
    def setUp(self):
        """Set up the test case."""
        super().setUp()
        
        # Initialize the plugin
        self.plugin = MemoryVisualizationPlugin()
        
        # Mock the matplotlib and plot functions
        self.plugin._visualize_memory_network = MagicMock(return_value=os.path.join(self.test_dir, "memory_network.png"))
        self.plugin._create_contradiction_heatmap = MagicMock(return_value=os.path.join(self.test_dir, "heatmap.png"))
        self.plugin._visualize_schema = MagicMock(return_value=os.path.join(self.test_dir, "schema.png"))
        
        # Mock other methods
        self.plugin.initialize = MagicMock(return_value=True)
        self.plugin.shutdown = MagicMock()
        
        # Set up plugin
        self.plugin.config = {
            "output_dir": os.path.join(self.test_dir, "visualizations"),
            "show_plots": False
        }
        self.plugin.stats = {
            "visualizations_created": 0,
            "files_saved": 0
        }
        
        # Create output directory
        os.makedirs(self.plugin.config["output_dir"], exist_ok=True)
    
    def test_post_reflection_handler(self):
        """Test post-reflection handler."""
        # Test with all data
        result = self.plugin.post_reflection_handler(
            result=self.reflection_result,
            memories=self.test_memories,
            schema=self.schema_graph
        )
        
        # Verify handler was called with right parameters
        self.plugin._visualize_memory_network.assert_called_once_with(self.test_memories)
        self.plugin._create_contradiction_heatmap.assert_called_once_with(self.test_memories)
        self.plugin._visualize_schema.assert_called_once_with(self.schema_graph)
        
        # Verify handler results
        self.assertIn("visualization_paths", result)
        self.assertIn("memory_network", result["visualization_paths"])
        self.assertIn("contradiction_heatmap", result["visualization_paths"])
        self.assertIn("schema_graph", result["visualization_paths"])


class TestSchemaExportPlugin(PluginTestCase):
    """Tests for the Schema Export Plugin."""
    
    def setUp(self):
        """Set up the test case."""
        super().setUp()
        
        # Initialize the plugin
        self.plugin = SchemaExportPlugin()
        
        # Mock the export methods
        self.plugin._export_to_format = MagicMock(
            side_effect=lambda G, fmt, ts: os.path.join(self.test_dir, f"schema_{ts}.{fmt}")
        )
        self.plugin._get_timestamp = MagicMock(return_value="20230101_000000")
        
        # Mock other methods
        self.plugin.initialize = MagicMock(return_value=True)
        self.plugin.shutdown = MagicMock()
        
        # Set up plugin
        self.plugin.config = {
            "output_dir": os.path.join(self.test_dir, "exports"),
            "export_formats": ["graphml", "gexf", "json"],
            "min_coherence": 0.2,
            "include_metadata": True,
            "auto_export": True
        }
        self.plugin.stats = {
            "exports_created": 0,
            "nodes_exported": 0,
            "edges_exported": 0
        }
        
        # Create output directory
        os.makedirs(self.plugin.config["output_dir"], exist_ok=True)
    
    def test_export_schema(self):
        """Test schema export functionality."""
        # Test export
        result = self.plugin.export_schema(self.schema_graph)
        
        # Verify export was successful
        self.assertTrue(result["exported"])
        self.assertIn("export_paths", result)
        
        # Check mock was called for each format
        expected_calls = len(self.plugin.config["export_formats"])
        self.assertEqual(self.plugin._export_to_format.call_count, expected_calls)
    
    def test_post_schema_update_handler(self):
        """Test post-schema-update handler."""
        # Patch the _export_schema method
        with patch.object(self.plugin, '_export_schema') as mock_export:
            mock_export.return_value = {"exported": True, "export_paths": {}}
            
            # Test with valid schema
            result = self.plugin.post_schema_update_handler(schema=self.schema_graph)
            
            # Verify _export_schema was called with the schema
            mock_export.assert_called_once_with(self.schema_graph)
            
            # Verify handler results
            self.assertTrue(result["exported"])


class TestIntegrationPlugin(PluginTestCase):
    """Tests for the Integration Plugin."""
    
    def setUp(self):
        """Set up the test case."""
        super().setUp()
        
        # Initialize the plugin
        self.plugin = IntegrationPlugin()
        
        # Mock methods
        self.plugin._prepare_reflection_payload = MagicMock(
            return_value={"event_type": "reflection", "timestamp": "2023-01-01T00:00:00"}
        )
        self.plugin._prepare_memory_payload = MagicMock(
            return_value={"event_type": "memory_added", "timestamp": "2023-01-01T00:00:00"}
        )
        self.plugin._prepare_schema_payload = MagicMock(
            return_value={"event_type": "schema_updated", "timestamp": "2023-01-01T00:00:00"}
        )
        self.plugin._prepare_monitoring_payload = MagicMock(
            return_value={"event_type": "monitoring", "timestamp": "2023-01-01T00:00:00"}
        )
        
        # Mock other methods
        self.plugin.initialize = MagicMock(return_value=True)
        self.plugin.shutdown = MagicMock()
        
        # Set up plugin
        self.plugin.config = {
            "api_endpoints": {
                "memory": "http://test.example.com/api/memory",
                "schema": "http://test.example.com/api/schema",
                "reflection": "http://test.example.com/api/reflection",
                "monitoring": "http://test.example.com/api/monitoring"
            },
            "webhooks": {
                "memory": "http://test.example.com/webhook/memory",
                "schema": "http://test.example.com/webhook/schema",
                "reflection": "http://test.example.com/webhook/reflection",
                "monitoring": "http://test.example.com/webhook/monitoring"
            },
            "enabled_hooks": [
                "POST_REFLECTION", 
                "POST_MEMORY_ADD", 
                "POST_SCHEMA_UPDATE", 
                "RUNTIME_MONITORING"
            ]
        }
        self.plugin.stats = {
            "api_calls": 0,
            "webhook_calls": 0,
            "errors": 0,
            "last_api_call": None
        }
    
    @patch('requests.post')
    def test_post_reflection_handler(self, mock_post):
        """Test post-reflection handler with mocked requests."""
        # Mock successful API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "success"}
        
        # Mock call_api and call_webhook methods
        with patch.object(self.plugin, '_call_api') as mock_call_api, \
             patch.object(self.plugin, '_call_webhook') as mock_call_webhook:
            
            mock_call_api.return_value = {"success": True, "status_code": 200}
            mock_call_webhook.return_value = {"success": True, "status_code": 200}
            
            # Test handler
            result = self.plugin.post_reflection_handler(
                result=self.reflection_result,
                memories=self.test_memories,
                schema=self.schema_graph
            )
            
            # Verify prepare_payload was called
            self.plugin._prepare_reflection_payload.assert_called_once_with(
                self.reflection_result, self.test_memories, self.schema_graph
            )
            
            # Verify call_api and call_webhook were called
            mock_call_api.assert_called_once()
            mock_call_webhook.assert_called_once()
            
            # Verify handler results
            self.assertIn("api_result", result)
            self.assertIn("webhook_result", result)
            self.assertTrue(result["api_result"]["success"])
            self.assertTrue(result["webhook_result"]["success"])
    
    @patch('requests.post')
    def test_post_memory_add_handler(self, mock_post):
        """Test post-memory-add handler with mocked requests."""
        # Mock successful API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "success"}
        
        # Mock call_api and call_webhook methods
        with patch.object(self.plugin, '_call_api') as mock_call_api, \
             patch.object(self.plugin, '_call_webhook') as mock_call_webhook:
            
            mock_call_api.return_value = {"success": True, "status_code": 200}
            mock_call_webhook.return_value = {"success": True, "status_code": 200}
            
            # Test handler
            result = self.plugin.post_memory_add_handler(
                memory=self.test_memories[0]
            )
            
            # Verify prepare_payload was called
            self.plugin._prepare_memory_payload.assert_called_once_with(self.test_memories[0])
            
            # Verify call_api and call_webhook were called
            mock_call_api.assert_called_once()
            mock_call_webhook.assert_called_once()
            
            # Verify handler results
            self.assertIn("api_result", result)
            self.assertIn("webhook_result", result)
            self.assertTrue(result["api_result"]["success"])
            self.assertTrue(result["webhook_result"]["success"])


def run_all_tests():
    """Run all plugin tests and output results."""
    logger.info("Starting comprehensive plugin tests")
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryVisualizationPlugin))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaExportPlugin))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationPlugin))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Output summary
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    logger.info(f"Success rate: {(result.testsRun - len(result.errors) - len(result.failures)) / result.testsRun:.2%}")
    
    if result.wasSuccessful():
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests()) 