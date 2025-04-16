#!/usr/bin/env python3
"""
Unit tests for the launcher integration with memory schema functionality.
Tests that the launcher properly initializes and exposes the schema integration.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
from tools.dev_environment.launcher import DevEnvironmentLauncher
from tools.dev_environment.memory_sandbox import MemorySandbox
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration


class TestLauncherSchemaIntegration(unittest.TestCase):
    """Test cases for the launcher's integration with memory schema functionality."""

    def setUp(self):
        """Set up the test environment before each test."""
        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp()
        self.snapshot_dir = os.path.join(self.temp_dir, 'snapshots')
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Initialize launcher with test directories
        self.launcher = DevEnvironmentLauncher(
            data_dir=self.temp_dir,
            tools_enabled={
                'memory_sandbox': True,
                'schema_integration': True
            }
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory and all its contents
        shutil.rmtree(self.temp_dir)
    
    def test_launcher_init_with_schema(self):
        """Test that the launcher correctly initializes with schema integration."""
        # Check that memory sandbox was initialized
        self.assertIsNotNone(self.launcher.tools.get('memory_sandbox'))
        self.assertIsInstance(self.launcher.tools['memory_sandbox'], MemorySandbox)
        
        # Check that schema integration was initialized
        self.assertIsNotNone(self.launcher.tools.get('schema_integration'))
        self.assertIsInstance(self.launcher.tools['schema_integration'], MemorySchemaIntegration)
    
    def test_schema_integration_linked_to_sandbox(self):
        """Test that the schema integration is linked to the memory sandbox."""
        memory_sandbox = self.launcher.tools['memory_sandbox']
        schema_integration = self.launcher.tools['schema_integration']
        
        # Schema integration should reference the same memory sandbox
        self.assertEqual(schema_integration.memory_sandbox, memory_sandbox)
    
    def test_launcher_web_api_schema_endpoints(self):
        """Test that the launcher exposes schema-related API endpoints."""
        # Get the web API routes
        app = self.launcher.get_web_app()
        routes = [route.path for route in app.routes]
        
        # Check for schema-related endpoints
        self.assertIn('/api/schema/build', routes)
        self.assertIn('/api/schema/clusters', routes)
        self.assertIn('/api/schema/concepts', routes)
        self.assertIn('/api/schema/visualization', routes)
    
    def test_launcher_schema_functionality(self):
        """Test that the schema integration functionality works through the launcher."""
        memory_sandbox = self.launcher.tools['memory_sandbox']
        schema_integration = self.launcher.tools['schema_integration']
        
        # Create some test memories
        memory_sandbox.create_synthetic_memory(
            memory_type='semantic',
            content='Artificial intelligence is a branch of computer science',
            metadata={'tags': ['AI', 'computer science', 'technology']}
        )
        
        memory_sandbox.create_synthetic_memory(
            memory_type='semantic',
            content='Machine learning is a subset of artificial intelligence',
            metadata={'tags': ['ML', 'AI', 'technology']}
        )
        
        # Build schema graph
        schema_integration.build_schema_graph()
        
        # Verify that the graph was built
        self.assertGreater(len(schema_integration.schema_graph.nodes()), 0)
        self.assertGreater(len(schema_integration.schema_graph.edges()), 0)
        
        # Test cluster detection
        clusters = schema_integration.detect_memory_clusters(min_cluster_size=1)
        self.assertIsNotNone(clusters)
        self.assertGreater(len(clusters), 0)
    
    def test_launcher_cli_commands(self):
        """Test that the launcher exposes schema-related CLI commands."""
        # Get available commands
        commands = self.launcher.get_available_commands()
        
        # Check for schema-related commands
        schema_commands = [cmd for cmd in commands if 'schema' in cmd]
        self.assertGreater(len(schema_commands), 0)
        
        # Common schema commands should be available
        self.assertIn('build-schema', schema_commands)
        self.assertIn('detect-clusters', schema_commands)
        self.assertIn('suggest-concepts', schema_commands)
    
    def test_launcher_json_export(self):
        """Test that the launcher can export schema data as JSON."""
        memory_sandbox = self.launcher.tools['memory_sandbox']
        schema_integration = self.launcher.tools['schema_integration']
        
        # Create a test memory
        memory_sandbox.create_synthetic_memory(
            memory_type='semantic',
            content='Knowledge graphs represent relationships between entities',
            metadata={'tags': ['knowledge graph', 'semantic web', 'data representation']}
        )
        
        # Build schema graph
        schema_integration.build_schema_graph()
        
        # Export schema as JSON
        export_path = os.path.join(self.temp_dir, 'schema_export.json')
        schema_integration.export_schema_graph(export_path)
        
        # Verify that the exported file exists and contains valid JSON
        self.assertTrue(os.path.exists(export_path))
        with open(export_path, 'r') as f:
            schema_data = json.load(f)
        
        # Basic validation of the exported schema
        self.assertIn('nodes', schema_data)
        self.assertIn('edges', schema_data)
        self.assertGreater(len(schema_data['nodes']), 0)


if __name__ == '__main__':
    unittest.main() 