#!/usr/bin/env python3
"""
ΨC Demo Interaction Components Tests
-----------------------------------
Tests for the interaction components of the ΨC-AI SDK demonstration.
These tests validate user interactions, state management, and data flow
between different components of the demo application.
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

# Mock the Streamlit module
sys.modules['streamlit'] = MagicMock()
import streamlit as st

class TestStateManagement(unittest.TestCase):
    """Test cases for the state management of the demo application."""
    
    def setUp(self):
        """Set up test environment."""
        # Import state management functions after patching
        from state_manager import (
            initialize_session_state,
            update_coherence_history,
            save_session_state,
            load_session_state,
            reset_session_state
        )
        
        self.initialize_session_state = initialize_session_state
        self.update_coherence_history = update_coherence_history
        self.save_session_state = save_session_state
        self.load_session_state = load_session_state
        self.reset_session_state = reset_session_state
        
        # Create a mock state
        self.mock_state = {}
    
    def test_initialize_session_state(self):
        """Test initializing the session state."""
        # Mock the session_state
        with patch('streamlit.session_state', self.mock_state):
            self.initialize_session_state()
        
        # Verify initialization
        self.assertIn('memories', self.mock_state)
        self.assertIn('schema', self.mock_state)
        self.assertIn('coherence_history', self.mock_state)
        self.assertIn('metrics', self.mock_state)
        self.assertIn('profile_name', self.mock_state)
    
    def test_update_coherence_history(self):
        """Test updating the coherence history."""
        # Setup initial state
        self.mock_state = {
            'coherence_history': [
                {"timestamp": "2023-01-01T12:00:00", "coherence": 0.75}
            ],
            'metrics': {
                'coherence': 0.8
            }
        }
        
        # Update coherence history
        with patch('streamlit.session_state', self.mock_state):
            self.update_coherence_history()
        
        # Verify update
        self.assertEqual(len(self.mock_state['coherence_history']), 2)
        self.assertEqual(self.mock_state['coherence_history'][1]['coherence'], 0.8)
    
    @patch('state_manager.open', new_callable=MagicMock)
    @patch('state_manager.json.dump')
    def test_save_session_state(self, mock_json_dump, mock_open):
        """Test saving the session state to a file."""
        # Setup state
        self.mock_state = {
            'memories': [{"id": "mem1", "content": "Memory 1"}],
            'schema': {"nodes": [], "edges": []},
            'coherence_history': [{"timestamp": "2023-01-01", "coherence": 0.75}],
            'metrics': {'coherence': 0.8},
            'profile_name': "Test Profile"
        }
        
        # Save state
        with patch('streamlit.session_state', self.mock_state):
            self.save_session_state("test_profile.json")
        
        # Verify save
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
    
    @patch('state_manager.open', new_callable=MagicMock)
    @patch('state_manager.json.load', return_value={
        'memories': [{"id": "mem1", "content": "Memory 1"}],
        'schema': {"nodes": [], "edges": []},
        'coherence_history': [{"timestamp": "2023-01-01", "coherence": 0.75}],
        'metrics': {'coherence': 0.8},
        'profile_name': "Test Profile"
    })
    def test_load_session_state(self, mock_json_load, mock_open):
        """Test loading the session state from a file."""
        # Setup empty state
        self.mock_state = {}
        
        # Load state
        with patch('streamlit.session_state', self.mock_state):
            self.load_session_state("test_profile.json")
        
        # Verify load
        mock_open.assert_called_once()
        mock_json_load.assert_called_once()
        self.assertIn('memories', self.mock_state)
        self.assertIn('schema', self.mock_state)
        self.assertEqual(self.mock_state['profile_name'], "Test Profile")
    
    def test_reset_session_state(self):
        """Test resetting the session state."""
        # Setup state with data
        self.mock_state = {
            'memories': [{"id": "mem1", "content": "Memory 1"}],
            'schema': {"nodes": [1, 2, 3], "edges": [4, 5, 6]},
            'coherence_history': [{"timestamp": "2023-01-01", "coherence": 0.75}],
            'metrics': {'coherence': 0.8},
            'profile_name': "Test Profile"
        }
        
        # Reset state
        with patch('streamlit.session_state', self.mock_state):
            self.reset_session_state()
        
        # Verify reset
        self.assertEqual(self.mock_state['memories'], [])
        self.assertEqual(self.mock_state['schema']['nodes'], [])
        self.assertEqual(self.mock_state['schema']['edges'], [])
        self.assertEqual(self.mock_state['coherence_history'], [])


class TestQueryProcessing(unittest.TestCase):
    """Test cases for processing user queries."""
    
    def setUp(self):
        """Set up test environment."""
        # Import query processing functions after patching
        with patch('psi_c_ai_sdk.orchestration.orchestrator.Orchestrator'):
            from query_processor import (
                process_user_query,
                update_schema_after_query,
                generate_response,
                extract_insights
            )
            
            self.process_user_query = process_user_query
            self.update_schema_after_query = update_schema_after_query
            self.generate_response = generate_response
            self.extract_insights = extract_insights
        
        # Mock orchestrator
        self.mock_orchestrator = MagicMock()
        self.mock_orchestrator.process_query.return_value = {
            "response": "This is a test response",
            "confidence": 0.85,
            "schema_updates": {
                "new_nodes": [{"id": "n3", "label": "New Concept"}],
                "new_edges": [{"source": "n1", "target": "n3", "type": "related_to"}]
            }
        }
    
    def test_process_user_query(self):
        """Test processing a user query."""
        # Setup state
        mock_state = {
            'schema': {
                "nodes": [
                    {"id": "n1", "label": "Concept 1"},
                    {"id": "n2", "label": "Concept 2"}
                ],
                "edges": [
                    {"source": "n1", "target": "n2", "type": "related_to"}
                ]
            }
        }
        
        # Process query
        with patch('streamlit.session_state', mock_state):
            response = self.process_user_query("What is the meaning of life?", self.mock_orchestrator)
        
        # Verify processing
        self.mock_orchestrator.process_query.assert_called_once()
        self.assertEqual(response["response"], "This is a test response")
        self.assertEqual(response["confidence"], 0.85)
    
    def test_update_schema_after_query(self):
        """Test updating the schema after a query."""
        # Setup state
        mock_state = {
            'schema': {
                "nodes": [
                    {"id": "n1", "label": "Concept 1"},
                    {"id": "n2", "label": "Concept 2"}
                ],
                "edges": [
                    {"source": "n1", "target": "n2", "type": "related_to"}
                ]
            }
        }
        
        # Setup query result
        query_result = {
            "schema_updates": {
                "new_nodes": [{"id": "n3", "label": "New Concept"}],
                "new_edges": [{"source": "n1", "target": "n3", "type": "related_to"}]
            }
        }
        
        # Update schema
        with patch('streamlit.session_state', mock_state):
            self.update_schema_after_query(query_result)
        
        # Verify update
        self.assertEqual(len(mock_state['schema']['nodes']), 3)
        self.assertEqual(len(mock_state['schema']['edges']), 2)
        self.assertEqual(mock_state['schema']['nodes'][2]["id"], "n3")
    
    def test_generate_response(self):
        """Test generating a response to a user query."""
        # Setup query and result
        query = "What is the meaning of life?"
        result = {
            "response": "This is a test response",
            "confidence": 0.85
        }
        
        # Generate response
        response = self.generate_response(query, result)
        
        # Verify response generation
        self.assertIn(query, response)
        self.assertIn(result["response"], response)
    
    def test_extract_insights(self):
        """Test extracting insights from a query result."""
        # Setup query result
        result = {
            "confidence": 0.85,
            "processing_time": 0.25,
            "schema_updates": {
                "new_nodes": [{"id": "n3", "label": "New Concept"}],
                "new_edges": [{"source": "n1", "target": "n3", "type": "related_to"}]
            }
        }
        
        # Extract insights
        insights = self.extract_insights(result)
        
        # Verify extraction
        self.assertIsInstance(insights, dict)
        self.assertIn("confidence", insights)
        self.assertIn("processing_time", insights)
        self.assertIn("schema_changes", insights)
        self.assertEqual(insights["schema_changes"]["new_concepts"], 1)
        self.assertEqual(insights["schema_changes"]["new_relations"], 1)


class TestMemoryManagement(unittest.TestCase):
    """Test cases for memory management in the demo application."""
    
    def setUp(self):
        """Set up test environment."""
        # Import memory management functions after patching
        with patch('psi_c_ai_sdk.memory.memory_store.MemoryStore'):
            from memory_manager import (
                add_memory,
                search_memories,
                delete_memory,
                update_memory_importance,
                get_contradictions
            )
            
            self.add_memory = add_memory
            self.search_memories = search_memories
            self.delete_memory = delete_memory
            self.update_memory_importance = update_memory_importance
            self.get_contradictions = get_contradictions
        
        # Mock memory store
        self.mock_memory_store = MagicMock()
        self.mock_memory_store.add_memory.return_value = {"id": "mem3", "content": "New Memory", "importance": 0.7}
        self.mock_memory_store.search.return_value = [{"id": "mem1", "content": "Memory 1", "importance": 0.8}]
        self.mock_memory_store.get_contradictions.return_value = [
            {"memory_pairs": [("mem1", "mem2")], "severity": 0.8}
        ]
    
    def test_add_memory(self):
        """Test adding a new memory."""
        # Setup state
        mock_state = {
            'memories': [
                {"id": "mem1", "content": "Memory 1", "importance": 0.8},
                {"id": "mem2", "content": "Memory 2", "importance": 0.6}
            ]
        }
        
        # Add memory
        with patch('streamlit.session_state', mock_state):
            result = self.add_memory("New Memory", self.mock_memory_store)
        
        # Verify addition
        self.mock_memory_store.add_memory.assert_called_once()
        self.assertEqual(len(mock_state['memories']), 3)
        self.assertEqual(mock_state['memories'][2]["id"], "mem3")
        self.assertEqual(result["id"], "mem3")
    
    def test_search_memories(self):
        """Test searching memories."""
        # Setup state
        mock_state = {
            'memories': [
                {"id": "mem1", "content": "Memory 1", "importance": 0.8},
                {"id": "mem2", "content": "Memory 2", "importance": 0.6}
            ]
        }
        
        # Search memories
        with patch('streamlit.session_state', mock_state):
            results = self.search_memories("Memory 1", self.mock_memory_store)
        
        # Verify search
        self.mock_memory_store.search.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mem1")
    
    def test_delete_memory(self):
        """Test deleting a memory."""
        # Setup state
        mock_state = {
            'memories': [
                {"id": "mem1", "content": "Memory 1", "importance": 0.8},
                {"id": "mem2", "content": "Memory 2", "importance": 0.6}
            ]
        }
        
        # Delete memory
        with patch('streamlit.session_state', mock_state):
            self.delete_memory("mem1")
        
        # Verify deletion
        self.assertEqual(len(mock_state['memories']), 1)
        self.assertEqual(mock_state['memories'][0]["id"], "mem2")
    
    def test_update_memory_importance(self):
        """Test updating memory importance."""
        # Setup state
        mock_state = {
            'memories': [
                {"id": "mem1", "content": "Memory 1", "importance": 0.8},
                {"id": "mem2", "content": "Memory 2", "importance": 0.6}
            ]
        }
        
        # Update importance
        with patch('streamlit.session_state', mock_state):
            result = self.update_memory_importance("mem1", 0.9)
        
        # Verify update
        self.assertEqual(mock_state['memories'][0]["importance"], 0.9)
        self.assertTrue(result)
    
    def test_get_contradictions(self):
        """Test getting contradictions between memories."""
        # Get contradictions
        contradictions = self.get_contradictions(self.mock_memory_store)
        
        # Verify retrieval
        self.mock_memory_store.get_contradictions.assert_called_once()
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0]["severity"], 0.8)


class TestIntegration(unittest.TestCase):
    """Integration tests for the demo application components."""
    
    @patch('psi_c_ai_sdk.orchestration.orchestrator.Orchestrator')
    @patch('psi_c_ai_sdk.memory.memory_store.MemoryStore')
    def test_full_interaction_flow(self, mock_memory_store_class, mock_orchestrator_class):
        """Test a full interaction flow through the system."""
        # Import main demo entry point
        from demo_app import (
            setup_environment,
            handle_user_interaction,
            update_state_and_ui
        )
        
        # Mock instances
        mock_orchestrator = mock_orchestrator_class.return_value
        mock_memory_store = mock_memory_store_class.return_value
        
        # Setup mock responses
        mock_orchestrator.process_query.return_value = {
            "response": "This is a test response",
            "confidence": 0.85,
            "schema_updates": {
                "new_nodes": [{"id": "n3", "label": "New Concept"}],
                "new_edges": [{"source": "n1", "target": "n3", "type": "related_to"}]
            }
        }
        
        mock_memory_store.add_memory.return_value = {
            "id": "mem1", 
            "content": "Memory 1",
            "importance": 0.8
        }
        
        # Setup state
        mock_state = {
            'memories': [],
            'schema': {"nodes": [], "edges": []},
            'coherence_history': [],
            'metrics': {'coherence': 0.0}
        }
        
        # Initialize environment
        with patch('streamlit.session_state', mock_state):
            env = setup_environment()
            
            # Add a memory
            mem_result = env['memory_manager'].add_memory("Memory 1")
            
            # Process a query
            query_result = handle_user_interaction("What is the meaning of life?", env)
            
            # Update state
            update_state_and_ui(query_result, env)
        
        # Verify the interaction flow
        self.assertEqual(mem_result["id"], "mem1")
        self.assertEqual(query_result["response"], "This is a test response")
        self.assertEqual(len(mock_state['schema']['nodes']), 1)
        self.assertEqual(len(mock_state['schema']['edges']), 1)
        self.assertGreater(len(mock_state['coherence_history']), 0)


if __name__ == '__main__':
    unittest.main() 