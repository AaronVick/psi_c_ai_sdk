#!/usr/bin/env python3
"""
ΨC Demo Integration Tests
------------------------
Tests for the integration between components of the ΨC-AI SDK demonstration.
These tests validate that the different parts of the demo interact properly
and that data flows correctly through the system.
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

# Import test helper classes
from test_demo_persistence import MockDemoRunner, MockSessionState

class TestDataFlow(unittest.TestCase):
    """Test cases for data flow through the demo system."""
    
    def setUp(self):
        """Set up test environment."""
        # Setup mocks and patches
        self.mock_st = MagicMock()
        self.mock_session_state = MockSessionState()
        self.patches = []
        
        # Patch the streamlit module
        self.st_patch = patch.dict('sys.modules', {'streamlit': self.mock_st})
        self.st_patch.start()
        self.patches.append(self.st_patch)
        
        # Patch session_state
        self.state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.state_patch.start()
        self.patches.append(self.state_patch)
        
        # Import functions
        from web_interface_demo import (
            add_memory, process_query, auto_save, create_schema_graph,
            create_schema_diff, export_summary
        )
        self.add_memory = add_memory
        self.process_query = process_query
        self.auto_save = auto_save
        self.create_schema_graph = create_schema_graph
        self.create_schema_diff = create_schema_diff
        self.export_summary = export_summary
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_memory_to_persistence_flow(self):
        """Test data flow from memory addition to persistence."""
        # Setup mocks
        with patch('web_interface_demo.auto_save') as mock_auto_save:
            with patch('web_interface_demo.save_agent_state') as mock_save_state:
                # Add a memory
                result = self.add_memory("Test memory")
                
                # Verify the memory was processed
                self.assertTrue(result)
                
                # Verify last_action was set
                self.assertEqual(self.mock_session_state.last_action, "add_memory")
                
                # Verify metrics were updated
                # For coherence_history, schema_complexity
                self.assertTrue(len(self.mock_session_state.coherence_history) > 0)
                self.assertTrue(len(self.mock_session_state.schema_complexity) > 0)
                
                # Verify detailed log was updated
                self.assertTrue(len(self.mock_session_state.detailed_log) > 0)
                recent_log = self.mock_session_state.detailed_log[-1]
                self.assertEqual(recent_log["operation"], "add_memory")
                self.assertEqual(recent_log["content"], "Test memory")
                
                # Verify auto_save would be called when triggered
                self.assertTrue(self.mock_session_state.auto_save_enabled)
    
    def test_query_to_visualization_flow(self):
        """Test data flow from query processing to visualization."""
        # Setup
        self.mock_session_state.last_action = None
        
        # Process a query
        with patch('web_interface_demo.auto_save') as mock_auto_save:
            result = self.process_query("Test query")
            
            # Verify query was processed
            self.assertTrue(result)
            
            # Verify response was stored
            self.assertEqual(self.mock_session_state.response, "Response to: Test query")
            
            # Verify last_action was set for visualization triggers
            self.assertEqual(self.mock_session_state.last_action, "process_query")
            
            # Verify auto_save was triggered
            mock_auto_save.assert_called_once()
        
        # Test visualization flow
        # Previous schema should be set for diff visualization
        self.assertIsNotNone(self.mock_session_state.previous_schema_state)
        
        # Create visualization
        fig = self.create_schema_diff()
        self.assertIsNotNone(fig)
    
    def test_memory_metrics_flow(self):
        """Test flow from memory addition to metrics updates."""
        # Setup initial metrics state
        initial_coherence_len = len(self.mock_session_state.coherence_history)
        initial_complexity_len = len(self.mock_session_state.schema_complexity)
        initial_perf_len = len(self.mock_session_state.performance_history)
        
        # Add memory to trigger metrics updates
        with patch('web_interface_demo.auto_save'):
            self.add_memory("Test memory for metrics")
        
        # Verify metrics were updated
        self.assertGreater(len(self.mock_session_state.coherence_history), initial_coherence_len)
        self.assertGreater(len(self.mock_session_state.schema_complexity), initial_complexity_len)
        self.assertGreater(len(self.mock_session_state.performance_history), initial_perf_len)
        
        # Verify specific metrics were calculated
        self.assertIsNotNone(self.mock_session_state.avg_memory_time)
        self.assertIsNotNone(self.mock_session_state.contradiction_rate)
        
        # Verify delta metrics were set for visualization
        self.assertIsNotNone(self.mock_session_state.delta_concepts)
        self.assertIsNotNone(self.mock_session_state.delta_connections)

class TestUIDataBinding(unittest.TestCase):
    """Test cases for data binding between UI and backend."""
    
    def setUp(self):
        """Set up test environment."""
        # Setup mocks and patches
        self.mock_st = MagicMock()
        self.mock_session_state = MockSessionState()
        self.patches = []
        
        # Patch the streamlit module
        self.st_patch = patch.dict('sys.modules', {'streamlit': self.mock_st})
        self.st_patch.start()
        self.patches.append(self.st_patch)
        
        # Patch session_state
        self.state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.state_patch.start()
        self.patches.append(self.state_patch)
        
        # Import rendering functions
        from web_interface_demo import (
            render_sidebar, render_graph, render_metrics, main
        )
        self.render_sidebar = render_sidebar
        self.render_graph = render_graph
        self.render_metrics = render_metrics
        self.main = main
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_sidebar_data_binding(self):
        """Test data binding in sidebar UI."""
        # Call render_sidebar
        self.render_sidebar()
        
        # Verify profile data binding
        self.mock_st.sidebar.selectbox.assert_any_call(
            "Active Profile",
            options=["default", "healthcare", "legal", "narrative"],
            index=0,  # Default profile is first
            key="profile_selector",
            help=unittest.mock.ANY  # Any help text
        )
        
        # Verify auto-save toggle binding
        self.mock_st.sidebar.checkbox.assert_any_call(
            "Auto-save Sessions", 
            value=True,  # Default is enabled
            help=unittest.mock.ANY
        )
        
        # Verify memory input binding
        self.mock_st.sidebar.text_area.assert_any_call(
            "Add New Memory", 
            help=unittest.mock.ANY
        )
        
        # Verify query input binding
        self.mock_st.sidebar.text_area.assert_any_call(
            "Ask a Question", 
            help=unittest.mock.ANY
        )
    
    def test_graph_data_binding(self):
        """Test data binding in graph visualization."""
        # Add mock for pyplot to prevent actual rendering
        with patch('matplotlib.pyplot.Figure', return_value=MagicMock()):
            with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())):
                # Call render_graph
                self.render_graph()
                
                # Verify metrics data binding
                self.mock_st.metric.assert_any_call(
                    label="Concepts", 
                    value=10,  # From MockSessionState
                    delta=self.mock_session_state.delta_concepts,
                    help=unittest.mock.ANY
                )
                
                self.mock_st.metric.assert_any_call(
                    label="Connections", 
                    value=9,  # From MockSessionState
                    delta=self.mock_session_state.delta_connections,
                    help=unittest.mock.ANY
                )
    
    def test_metrics_data_binding(self):
        """Test data binding in metrics visualization."""
        # Add mock for pyplot to prevent actual rendering
        with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())):
            # Call render_metrics
            self.render_metrics()
            
            # Verify contradiction rate metric binding
            self.mock_st.metric.assert_any_call(
                label="Contradiction Rate", 
                value=f"{self.mock_session_state.contradiction_rate:.2f}",
                delta=unittest.mock.ANY,
                delta_color="inverse",
                help=unittest.mock.ANY
            )
            
            # Verify memory time metric binding
            self.mock_st.metric.assert_any_call(
                label="Memory Integration (ms)", 
                value=f"{self.mock_session_state.avg_memory_time:.1f}",
                delta=unittest.mock.ANY,
                delta_color="inverse",
                help=unittest.mock.ANY
            )
            
            # Verify query time metric binding
            self.mock_st.metric.assert_any_call(
                label="Query Response (ms)", 
                value=f"{self.mock_session_state.avg_query_time:.1f}",
                delta=unittest.mock.ANY,
                delta_color="inverse",
                help=unittest.mock.ANY
            )

class TestProfileChanges(unittest.TestCase):
    """Test cases for profile changes and data isolation."""
    
    def setUp(self):
        """Set up test environment."""
        # Setup mocks and patches
        self.mock_st = MagicMock()
        self.mock_session_state = MockSessionState()
        self.patches = []
        
        # Patch the streamlit module
        self.st_patch = patch.dict('sys.modules', {'streamlit': self.mock_st})
        self.st_patch.start()
        self.patches.append(self.st_patch)
        
        # Patch session_state
        self.state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.state_patch.start()
        self.patches.append(self.state_patch)
        
        # Import profile-related functions
        from web_interface_demo import (
            change_profile, archive_current_session, save_agent_state,
            init_session_state, create_demo_runner
        )
        self.change_profile = change_profile
        self.archive_current_session = archive_current_session
        self.save_agent_state = save_agent_state
        self.init_session_state = init_session_state
        self.create_demo_runner = create_demo_runner
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_profile_change_data_isolation(self):
        """Test that profile changes maintain data isolation."""
        # Set up initial state
        initial_session_id = self.mock_session_state.current_session_id
        
        # Mock functions to verify calls
        with patch('web_interface_demo.archive_current_session') as mock_archive:
            mock_archive.return_value = "archived_session_id"
            
            with patch('web_interface_demo.create_demo_runner') as mock_create_runner:
                mock_create_runner.return_value = MockDemoRunner("healthcare")
                
                with patch('web_interface_demo.save_agent_state') as mock_save:
                    mock_save.return_value = "new_session_id"
                    
                    # Set the profile selector and trigger change
                    self.mock_session_state.profile_selector = "healthcare"
                    self.change_profile()
                    
                    # Verify the archive function was called to preserve old data
                    mock_archive.assert_called_once()
                    
                    # Verify profile was changed
                    self.assertEqual(self.mock_session_state.profile, "healthcare")
                    
                    # Verify demo runner was recreated with new profile
                    mock_create_runner.assert_called_once_with("healthcare")
                    
                    # Verify a new session ID was created
                    self.assertNotEqual(self.mock_session_state.current_session_id, initial_session_id)
                    
                    # Verify state data was reset
                    self.assertEqual(self.mock_session_state.reflection_log, [])
                    
                    # Verify new state was saved
                    mock_save.assert_called_once_with(
                        "healthcare", 
                        self.mock_session_state.current_session_id
                    )
    
    def test_profile_specific_runner_creation(self):
        """Test that demo runners are created with profile-specific configs."""
        # Mock DemoRunner import and constructor
        with patch.dict('sys.modules', {'demo_runner': MagicMock()}):
            demo_runner_module = sys.modules['demo_runner']
            demo_runner_class = MagicMock()
            demo_runner_module.DemoRunner = demo_runner_class
            
            # Create runners with different profiles
            self.create_demo_runner("default")
            self.create_demo_runner("healthcare")
            self.create_demo_runner("legal")
            
            # Verify correct profile was passed each time
            demo_runner_class.assert_has_calls([
                call(profile="default"),
                call(profile="healthcare"),
                call(profile="legal")
            ])

if __name__ == '__main__':
    unittest.main() 