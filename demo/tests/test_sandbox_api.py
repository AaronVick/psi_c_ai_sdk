#!/usr/bin/env python3
"""
ΨC Demo Sandbox API Tests
------------------------
Tests for the sandbox API functionality of the ΨC-AI SDK demonstration.
These tests validate that the demo sandbox correctly interfaces with the
core SDK components and provides appropriate API endpoints.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add the parent directory to the Python path for imports
script_dir = Path(__file__).parent.absolute()
demo_dir = script_dir.parent.absolute()
sys.path.append(str(demo_dir))
project_dir = demo_dir.parent.absolute()
sys.path.append(str(project_dir))

class TestSandboxAPI(unittest.TestCase):
    """Test cases for the demo sandbox API."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock API dependencies
        self.mock_schema = MagicMock()
        self.mock_memory_store = MagicMock()
        self.mock_reasoning = MagicMock()
        self.mock_orchestrator = MagicMock()
        
        # Import sandbox API after patching dependencies
        with patch('psi_c_ai_sdk.schema.schema_graph.SchemaGraph', return_value=self.mock_schema), \
             patch('psi_c_ai_sdk.memory.memory_store.MemoryStore', return_value=self.mock_memory_store), \
             patch('psi_c_ai_sdk.reasoning.reasoning_engine.ReasoningEngine', return_value=self.mock_reasoning), \
             patch('psi_c_ai_sdk.orchestration.orchestrator.Orchestrator', return_value=self.mock_orchestrator):
            
            from sandbox_api import SandboxAPI
            self.api = SandboxAPI()
    
    def test_initialization(self):
        """Test API initialization."""
        # Verify components were initialized
        self.assertIsNotNone(self.api.schema)
        self.assertIsNotNone(self.api.memory_store)
        self.assertIsNotNone(self.api.reasoning)
        self.assertIsNotNone(self.api.orchestrator)
    
    def test_add_memory(self):
        """Test adding memory via API."""
        # Setup return value for memory addition
        self.mock_orchestrator.process_memory.return_value = {
            "success": True,
            "memory_id": "test_memory_123",
            "schema_changes": 3,
            "coherence_delta": 0.05
        }
        
        # Call API method
        result = self.api.add_memory("This is a test memory")
        
        # Verify orchestrator was called
        self.mock_orchestrator.process_memory.assert_called_once_with("This is a test memory")
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["memory_id"], "test_memory_123")
        self.assertEqual(result["schema_changes"], 3)
        self.assertEqual(result["coherence_delta"], 0.05)
    
    def test_process_query(self):
        """Test processing query via API."""
        # Setup return value for query processing
        self.mock_orchestrator.process_query.return_value = {
            "success": True,
            "response": "This is a test response",
            "confidence": 0.85,
            "reasoning_steps": [
                {"step": 1, "description": "Initial analysis"},
                {"step": 2, "description": "Retrieved relevant memories"},
                {"step": 3, "description": "Generated response"}
            ]
        }
        
        # Call API method
        result = self.api.process_query("What is the meaning of life?")
        
        # Verify orchestrator was called
        self.mock_orchestrator.process_query.assert_called_once_with("What is the meaning of life?")
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["response"], "This is a test response")
        self.assertEqual(result["confidence"], 0.85)
        self.assertEqual(len(result["reasoning_steps"]), 3)
    
    def test_get_coherence_score(self):
        """Test getting coherence score via API."""
        # Setup return value
        self.mock_schema.calculate_coherence.return_value = 0.92
        
        # Call API method
        score = self.api.get_coherence_score()
        
        # Verify schema method was called
        self.mock_schema.calculate_coherence.assert_called_once()
        
        # Verify result
        self.assertEqual(score, 0.92)
    
    def test_get_schema_graph(self):
        """Test getting schema graph via API."""
        # Setup mock schema data
        mock_nodes = [
            {"id": "node1", "label": "Concept 1", "type": "concept"},
            {"id": "node2", "label": "Concept 2", "type": "concept"}
        ]
        
        mock_edges = [
            {"source": "node1", "target": "node2", "type": "related_to", "weight": 0.8}
        ]
        
        self.mock_schema.get_nodes.return_value = mock_nodes
        self.mock_schema.get_edges.return_value = mock_edges
        
        # Call API method
        graph = self.api.get_schema_graph()
        
        # Verify schema methods were called
        self.mock_schema.get_nodes.assert_called_once()
        self.mock_schema.get_edges.assert_called_once()
        
        # Verify result
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["nodes"][0]["id"], "node1")
        self.assertEqual(graph["edges"][0]["source"], "node1")
    
    def test_get_all_memories(self):
        """Test getting all memories via API."""
        # Setup mock memories
        mock_memories = [
            {"id": "mem1", "content": "Memory 1", "created_at": "2023-01-01T12:00:00"},
            {"id": "mem2", "content": "Memory 2", "created_at": "2023-01-02T12:00:00"}
        ]
        
        self.mock_memory_store.get_all_memories.return_value = mock_memories
        
        # Call API method
        memories = self.api.get_all_memories()
        
        # Verify memory store method was called
        self.mock_memory_store.get_all_memories.assert_called_once()
        
        # Verify result
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["id"], "mem1")
        self.assertEqual(memories[1]["content"], "Memory 2")
    
    def test_search_memories(self):
        """Test searching memories via API."""
        # Setup mock search results
        mock_results = [
            {"id": "mem1", "content": "Test memory", "relevance": 0.95},
            {"id": "mem2", "content": "Another test", "relevance": 0.75}
        ]
        
        self.mock_memory_store.search.return_value = mock_results
        
        # Call API method
        results = self.api.search_memories("test")
        
        # Verify memory store method was called
        self.mock_memory_store.search.assert_called_once_with("test")
        
        # Verify result
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "mem1")
        self.assertEqual(results[0]["relevance"], 0.95)
    
    def test_get_contradiction_report(self):
        """Test getting contradiction report via API."""
        # Setup mock contradictions
        mock_contradictions = [
            {
                "id": "c1",
                "memory_pairs": [("mem1", "mem3")],
                "description": "Conflicting information about X",
                "severity": 0.8
            },
            {
                "id": "c2",
                "memory_pairs": [("mem2", "mem4")],
                "description": "Conflicting information about Y",
                "severity": 0.6
            }
        ]
        
        self.mock_schema.get_contradictions.return_value = mock_contradictions
        
        # Call API method
        report = self.api.get_contradiction_report()
        
        # Verify schema method was called
        self.mock_schema.get_contradictions.assert_called_once()
        
        # Verify result
        self.assertEqual(len(report), 2)
        self.assertEqual(report[0]["id"], "c1")
        self.assertEqual(report[0]["severity"], 0.8)
        self.assertEqual(len(report[0]["memory_pairs"]), 1)
    
    def test_reset_sandbox(self):
        """Test resetting sandbox via API."""
        # Call API method
        self.api.reset_sandbox()
        
        # Verify components were reset
        self.mock_schema.reset.assert_called_once()
        self.mock_memory_store.reset.assert_called_once()
        self.mock_reasoning.reset.assert_called_once()
        self.mock_orchestrator.reset.assert_called_once()

class TestSandboxAPIErrorHandling(unittest.TestCase):
    """Test cases for sandbox API error handling."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock API dependencies with error behavior
        self.mock_schema = MagicMock()
        self.mock_memory_store = MagicMock()
        self.mock_reasoning = MagicMock()
        self.mock_orchestrator = MagicMock()
        
        # Import sandbox API after patching dependencies
        with patch('psi_c_ai_sdk.schema.schema_graph.SchemaGraph', return_value=self.mock_schema), \
             patch('psi_c_ai_sdk.memory.memory_store.MemoryStore', return_value=self.mock_memory_store), \
             patch('psi_c_ai_sdk.reasoning.reasoning_engine.ReasoningEngine', return_value=self.mock_reasoning), \
             patch('psi_c_ai_sdk.orchestration.orchestrator.Orchestrator', return_value=self.mock_orchestrator):
            
            from sandbox_api import SandboxAPI
            self.api = SandboxAPI()
    
    def test_add_memory_error(self):
        """Test error handling when adding memory."""
        # Setup error behavior
        self.mock_orchestrator.process_memory.side_effect = Exception("Test error")
        
        # Call API method and catch exception
        result = self.api.add_memory("This is a test memory")
        
        # Verify error handling
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Test error")
    
    def test_process_query_error(self):
        """Test error handling when processing query."""
        # Setup error behavior
        self.mock_orchestrator.process_query.side_effect = Exception("Query processing error")
        
        # Call API method
        result = self.api.process_query("What is the meaning of life?")
        
        # Verify error handling
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Query processing error")
    
    def test_get_coherence_score_error(self):
        """Test error handling when getting coherence score."""
        # Setup error behavior
        self.mock_schema.calculate_coherence.side_effect = Exception("Coherence calculation error")
        
        # Call API method
        score = self.api.get_coherence_score()
        
        # Verify error handling - should return default value
        self.assertEqual(score, 0.0)
    
    def test_get_schema_graph_error(self):
        """Test error handling when getting schema graph."""
        # Setup error behavior
        self.mock_schema.get_nodes.side_effect = Exception("Graph retrieval error")
        
        # Call API method
        graph = self.api.get_schema_graph()
        
        # Verify error handling - should return empty graph
        self.assertEqual(len(graph["nodes"]), 0)
        self.assertEqual(len(graph["edges"]), 0)

class TestSandboxAPIIntegration(unittest.TestCase):
    """Test cases for sandbox API integration with core SDK components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create real dependencies but with minimized functionality
        with patch('psi_c_ai_sdk.schema.schema_graph.SchemaGraph.initialize_defaults'), \
             patch('psi_c_ai_sdk.memory.memory_store.MemoryStore.initialize_defaults'), \
             patch('psi_c_ai_sdk.reasoning.reasoning_engine.ReasoningEngine.initialize_defaults'), \
             patch('psi_c_ai_sdk.orchestration.orchestrator.Orchestrator.initialize_defaults'):
            
            from sandbox_api import SandboxAPI
            self.api = SandboxAPI(use_minimal_dependencies=True)
    
    @patch('sandbox_api.SandboxAPI._check_system_status')
    def test_system_status_check(self, mock_check):
        """Test system status check integration."""
        # Setup mock return
        mock_check.return_value = {
            "schema": "healthy",
            "memory_store": "healthy", 
            "reasoning": "healthy",
            "orchestrator": "healthy",
            "overall": "healthy"
        }
        
        # Call API method
        status = self.api.check_system_status()
        
        # Verify result
        self.assertEqual(status["overall"], "healthy")
        self.assertEqual(status["schema"], "healthy")
    
    @patch('sandbox_api.SandboxAPI.export_data')
    def test_export_integration(self, mock_export):
        """Test export functionality."""
        # Setup mock export file
        mock_file = "test_export.json"
        mock_export.return_value = mock_file
        
        # Call export
        result = self.api.export_data("json")
        
        # Verify export was called and returned expected result
        self.assertEqual(result, mock_file)
    
    @patch('sandbox_api.SandboxAPI.import_data')
    def test_import_integration(self, mock_import):
        """Test import functionality."""
        # Setup mock import success
        mock_import.return_value = {"success": True, "items_imported": 10}
        
        # Call import
        result = self.api.import_data("test_data.json")
        
        # Verify import was called and returned expected result
        self.assertTrue(result["success"])
        self.assertEqual(result["items_imported"], 10)

if __name__ == '__main__':
    unittest.main() 