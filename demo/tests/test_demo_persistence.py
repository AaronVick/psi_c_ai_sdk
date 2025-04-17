#!/usr/bin/env python3
"""
ΨC Demo Persistence Tests
------------------------
Tests for the persistence functionality of the ΨC-AI SDK demonstration.
These tests validate that session data is properly saved, loaded, and
maintained across user sessions.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the Python path for imports
script_dir = Path(__file__).parent.absolute()
demo_dir = script_dir.parent.absolute()
sys.path.append(str(demo_dir))
project_dir = demo_dir.parent.absolute()
sys.path.append(str(project_dir))

class MockSessionState(dict):
    """Mock implementation of Streamlit's session_state."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    
    def __getattr__(self, name):
        if name not in self:
            self[name] = None
        return self[name]
    
    def __setattr__(self, name, value):
        self[name] = value

class TestSessionPersistence(unittest.TestCase):
    """Test cases for session persistence."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Mock streamlit
        self.mock_st = MagicMock()
        self.mock_session_state = MockSessionState()
        
        # Initialize session state with test data
        self.mock_session_state.profile = "test_profile"
        self.mock_session_state.memory_log = ["Memory 1", "Memory 2"]
        self.mock_session_state.query_log = ["Query 1", "Query 2"]
        self.mock_session_state.results_log = ["Result 1", "Result 2"]
        self.mock_session_state.memory_metrics = {"coherence": 0.85, "stability": 0.92}
        self.mock_session_state.last_action = "query_processed"
        
        # Set up patches
        self.patches = []
        
        # Patch the streamlit module
        self.st_patch = patch.dict('sys.modules', {'streamlit': self.mock_st})
        self.st_patch.start()
        self.patches.append(self.st_patch)
        
        # Patch session_state
        self.state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.state_patch.start()
        self.patches.append(self.state_patch)
        
        # Patch Path.home() to use temp dir
        self.path_home_patch = patch('pathlib.Path.home', return_value=Path(self.temp_dir.name))
        self.path_home_patch.start()
        self.patches.append(self.path_home_patch)
        
        # Import persistence functions after patches
        from demo_persistence import (
            save_session_data, 
            load_session_data, 
            initialize_session_state,
            get_session_data_path,
            backup_session_data,
            reset_session_state
        )
        
        self.save_session_data = save_session_data
        self.load_session_data = load_session_data
        self.initialize_session_state = initialize_session_state
        self.get_session_data_path = get_session_data_path
        self.backup_session_data = backup_session_data
        self.reset_session_state = reset_session_state
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_session_data_path(self):
        """Test that session data path is correctly generated."""
        # Call the function
        path = self.get_session_data_path()
        
        # Check path format
        self.assertIsInstance(path, Path)
        self.assertTrue(str(path).endswith('.psi_c_demo_data.json'))
        
        # Check path relative to home
        self.assertTrue(str(path).startswith(self.temp_dir.name))
    
    def test_save_session_data(self):
        """Test saving session data to file."""
        # Mock open to capture written data
        m = mock_open()
        
        with patch('builtins.open', m):
            # Call save function
            self.save_session_data()
        
        # Verify file was opened for writing
        m.assert_called_once()
        
        # Get the data that would have been written
        written_data = m().write.call_args[0][0]
        
        # Parse the JSON data
        saved_data = json.loads(written_data)
        
        # Verify saved data matches session state
        self.assertEqual(saved_data["profile"], "test_profile")
        self.assertEqual(saved_data["memory_log"], ["Memory 1", "Memory 2"])
        self.assertEqual(saved_data["query_log"], ["Query 1", "Query 2"])
        self.assertEqual(saved_data["results_log"], ["Result 1", "Result 2"])
        self.assertAlmostEqual(saved_data["memory_metrics"]["coherence"], 0.85)
        self.assertAlmostEqual(saved_data["memory_metrics"]["stability"], 0.92)
        self.assertEqual(saved_data["last_action"], "query_processed")
    
    def test_load_session_data(self):
        """Test loading session data from file."""
        # Create test data to load
        test_data = {
            "profile": "loaded_profile",
            "memory_log": ["Loaded Memory 1", "Loaded Memory 2"],
            "query_log": ["Loaded Query 1", "Loaded Query 2"],
            "results_log": ["Loaded Result 1", "Loaded Result 2"],
            "memory_metrics": {"coherence": 0.75, "stability": 0.82},
            "last_action": "loaded_action"
        }
        
        # Mock open to return test data
        m = mock_open(read_data=json.dumps(test_data))
        
        with patch('builtins.open', m), patch('os.path.exists', return_value=True):
            # Call load function
            self.load_session_data()
        
        # Verify file was opened for reading
        m.assert_called_once()
        
        # Verify session state was updated with loaded data
        self.assertEqual(self.mock_session_state.profile, "loaded_profile")
        self.assertEqual(self.mock_session_state.memory_log, ["Loaded Memory 1", "Loaded Memory 2"])
        self.assertEqual(self.mock_session_state.query_log, ["Loaded Query 1", "Loaded Query 2"])
        self.assertEqual(self.mock_session_state.results_log, ["Loaded Result 1", "Loaded Result 2"])
        self.assertAlmostEqual(self.mock_session_state.memory_metrics["coherence"], 0.75)
        self.assertAlmostEqual(self.mock_session_state.memory_metrics["stability"], 0.82)
        self.assertEqual(self.mock_session_state.last_action, "loaded_action")
    
    def test_backup_session_data(self):
        """Test backing up session data."""
        # Setup mock functions
        mock_path = MagicMock()
        mock_json_dumps = MagicMock(return_value="mocked_json_data")
        
        with patch('pathlib.Path', return_value=mock_path), \
             patch('json.dumps', mock_json_dumps), \
             patch('builtins.open', mock_open()):
            
            # Call backup function
            backup_path = self.backup_session_data()
        
        # Verify backup path naming
        self.assertTrue("backup" in str(backup_path).lower())
        
        # Verify data was written
        mock_path.write_text.assert_called_once_with("mocked_json_data")
    
    def test_reset_session_state(self):
        """Test resetting session state."""
        # Call reset function
        self.reset_session_state()
        
        # Verify state was reset to defaults
        self.assertEqual(self.mock_session_state.profile, "default")
        self.assertEqual(self.mock_session_state.memory_log, [])
        self.assertEqual(self.mock_session_state.query_log, [])
        self.assertEqual(self.mock_session_state.results_log, [])
        self.assertEqual(self.mock_session_state.last_action, None)
    
    def test_initialize_session_state(self):
        """Test initializing session state with defaults."""
        # Reset session state to simulate fresh start
        for key in list(self.mock_session_state.keys()):
            del self.mock_session_state[key]
        
        # Call initialize function with no existing data
        with patch('os.path.exists', return_value=False):
            self.initialize_session_state()
        
        # Verify state has default values
        self.assertEqual(self.mock_session_state.profile, "default")
        self.assertEqual(self.mock_session_state.memory_log, [])
        self.assertEqual(self.mock_session_state.query_log, [])
        self.assertEqual(self.mock_session_state.results_log, [])
        self.assertEqual(self.mock_session_state.memory_metrics, {})
        self.assertEqual(self.mock_session_state.last_action, None)

class TestProfileManagement(unittest.TestCase):
    """Test cases for profile management."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Mock streamlit
        self.mock_st = MagicMock()
        self.mock_session_state = MockSessionState()
        
        # Initialize session state with test data
        self.mock_session_state.profile = "default"
        self.mock_session_state.memory_log = ["Memory 1", "Memory 2"]
        self.mock_session_state.query_log = ["Query 1", "Query 2"]
        self.mock_session_state.results_log = ["Result 1", "Result 2"]
        
        # Set up patches
        self.patches = []
        
        # Patch the streamlit module
        self.st_patch = patch.dict('sys.modules', {'streamlit': self.mock_st})
        self.st_patch.start()
        self.patches.append(self.st_patch)
        
        # Patch session_state
        self.state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.state_patch.start()
        self.patches.append(self.state_patch)
        
        # Patch Path.home() to use temp dir
        self.path_home_patch = patch('pathlib.Path.home', return_value=Path(self.temp_dir.name))
        self.path_home_patch.start()
        self.patches.append(self.path_home_patch)
        
        # Import profile management functions after patches
        from demo_persistence import (
            load_profile_data, 
            save_profile_data,
            switch_profile,
            get_profile_path
        )
        
        self.load_profile_data = load_profile_data
        self.save_profile_data = save_profile_data
        self.switch_profile = switch_profile
        self.get_profile_path = get_profile_path
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_profile_path(self):
        """Test that profile path is correctly generated."""
        # Call the function for a specific profile
        path = self.get_profile_path("test_profile")
        
        # Check path format
        self.assertIsInstance(path, Path)
        self.assertTrue(str(path).endswith('profile_test_profile.json'))
        
        # Check path relative to home
        self.assertTrue(str(path).startswith(self.temp_dir.name))
    
    def test_save_profile_data(self):
        """Test saving profile data to file."""
        # Mock open to capture written data
        m = mock_open()
        
        with patch('builtins.open', m):
            # Call save function
            self.save_profile_data("test_profile")
        
        # Verify file was opened for writing
        m.assert_called_once()
        
        # Get the data that would have been written
        written_data = m().write.call_args[0][0]
        
        # Parse the JSON data
        saved_data = json.loads(written_data)
        
        # Verify saved data matches session state
        self.assertEqual(saved_data["memory_log"], ["Memory 1", "Memory 2"])
        self.assertEqual(saved_data["query_log"], ["Query 1", "Query 2"])
        self.assertEqual(saved_data["results_log"], ["Result 1", "Result 2"])
    
    def test_load_profile_data(self):
        """Test loading profile data from file."""
        # Create test data to load
        test_data = {
            "memory_log": ["Profile Memory 1", "Profile Memory 2"],
            "query_log": ["Profile Query 1", "Profile Query 2"],
            "results_log": ["Profile Result 1", "Profile Result 2"],
            "memory_metrics": {"coherence": 0.65, "stability": 0.72}
        }
        
        # Mock open to return test data
        m = mock_open(read_data=json.dumps(test_data))
        
        with patch('builtins.open', m), patch('os.path.exists', return_value=True):
            # Call load function
            self.load_profile_data("test_profile")
        
        # Verify file was opened for reading
        m.assert_called_once()
        
        # Verify session state was updated with loaded data
        self.assertEqual(self.mock_session_state.memory_log, ["Profile Memory 1", "Profile Memory 2"])
        self.assertEqual(self.mock_session_state.query_log, ["Profile Query 1", "Profile Query 2"])
        self.assertEqual(self.mock_session_state.results_log, ["Profile Result 1", "Profile Result 2"])
        self.assertAlmostEqual(self.mock_session_state.memory_metrics["coherence"], 0.65)
        self.assertAlmostEqual(self.mock_session_state.memory_metrics["stability"], 0.72)
    
    def test_switch_profile(self):
        """Test switching between profiles."""
        # Setup mocks
        mock_save = MagicMock()
        mock_load = MagicMock()
        
        with patch('demo_persistence.save_profile_data', mock_save), \
             patch('demo_persistence.load_profile_data', mock_load):
            
            # Call switch function
            self.switch_profile("new_profile")
        
        # Verify old profile was saved
        mock_save.assert_called_once_with("default")
        
        # Verify new profile was loaded
        mock_load.assert_called_once_with("new_profile")
        
        # Verify current profile was updated
        self.assertEqual(self.mock_session_state.profile, "new_profile")

class TestAutosave(unittest.TestCase):
    """Test cases for autosave functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock streamlit
        self.mock_st = MagicMock()
        self.mock_session_state = MockSessionState()
        
        # Set up patches
        self.patches = []
        
        # Patch the streamlit module
        self.st_patch = patch.dict('sys.modules', {'streamlit': self.mock_st})
        self.st_patch.start()
        self.patches.append(self.st_patch)
        
        # Patch session_state
        self.state_patch = patch('streamlit.session_state', self.mock_session_state)
        self.state_patch.start()
        self.patches.append(self.state_patch)
        
        # Import autosave function after patches
        from demo_persistence import (
            autosave_on_change,
        )
        
        self.autosave_on_change = autosave_on_change
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_autosave_on_memory_added(self):
        """Test autosave when memory is added."""
        mock_save = MagicMock()
        
        with patch('demo_persistence.save_session_data', mock_save):
            # Call autosave with memory_added action
            self.mock_session_state.last_action = "memory_added"
            self.autosave_on_change()
        
        # Verify save was called
        mock_save.assert_called_once()
    
    def test_autosave_on_query_processed(self):
        """Test autosave when query is processed."""
        mock_save = MagicMock()
        
        with patch('demo_persistence.save_session_data', mock_save):
            # Call autosave with query_processed action
            self.mock_session_state.last_action = "query_processed"
            self.autosave_on_change()
        
        # Verify save was called
        mock_save.assert_called_once()
    
    def test_autosave_on_profile_changed(self):
        """Test autosave when profile is changed."""
        mock_save = MagicMock()
        
        with patch('demo_persistence.save_session_data', mock_save):
            # Call autosave with profile_changed action
            self.mock_session_state.last_action = "profile_changed"
            self.autosave_on_change()
        
        # Verify save was called
        mock_save.assert_called_once()
    
    def test_no_autosave_on_other_actions(self):
        """Test no autosave on non-triggering actions."""
        mock_save = MagicMock()
        
        with patch('demo_persistence.save_session_data', mock_save):
            # Call autosave with other actions
            self.mock_session_state.last_action = "view_changed"
            self.autosave_on_change()
        
        # Verify save was not called
        mock_save.assert_not_called()
    
    def test_no_autosave_when_no_action(self):
        """Test no autosave when no action is recorded."""
        mock_save = MagicMock()
        
        with patch('demo_persistence.save_session_data', mock_save):
            # Call autosave with no action
            self.mock_session_state.last_action = None
            self.autosave_on_change()
        
        # Verify save was not called
        mock_save.assert_not_called()

if __name__ == '__main__':
    unittest.main() 