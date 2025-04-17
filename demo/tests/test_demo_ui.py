#!/usr/bin/env python3
"""
ΨC Demo UI Tests
----------------
Tests for the UI components of the ΨC-AI SDK demonstration.
These tests validate the appearance and usability of UI components
without requiring a running Streamlit server.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add the parent directory to the Python path for imports
script_dir = Path(__file__).parent.absolute()
demo_dir = script_dir.parent.absolute()
sys.path.append(str(demo_dir))
project_dir = demo_dir.parent.absolute()
sys.path.append(str(project_dir))

# Import test helper class
from test_demo_persistence import MockSessionState

class TestUIComponents(unittest.TestCase):
    """Test cases for UI component rendering."""
    
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
        
        # Import UI components
        from web_interface_demo import (
            render_sidebar, render_header, render_tabs, render_graph, 
            render_metrics, render_logs, render_empirical_data,
            render_tooltip
        )
        self.render_sidebar = render_sidebar
        self.render_header = render_header
        self.render_tabs = render_tabs
        self.render_graph = render_graph
        self.render_metrics = render_metrics
        self.render_logs = render_logs
        self.render_empirical_data = render_empirical_data
        self.render_tooltip = render_tooltip
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_header_rendering(self):
        """Test header rendering with proper styling."""
        # Call header render
        self.render_header()
        
        # Verify title rendering
        self.mock_st.title.assert_called_once()
        
        # Verify description markdown
        self.mock_st.markdown.assert_called()
        
        # Verify proper styling with CSS
        css_calls = [call for call in self.mock_st.markdown.call_args_list 
                     if isinstance(call[0][0], str) and "<style>" in call[0][0]]
        self.assertTrue(len(css_calls) > 0, "Header CSS styling not found")
        
        # Verify header image
        self.mock_st.image.assert_called()
    
    def test_tab_rendering(self):
        """Test tab interface rendering."""
        # Mock the tabs container
        mock_tabs = MagicMock()
        self.mock_st.tabs.return_value = [
            mock_tabs, mock_tabs, mock_tabs, mock_tabs
        ]
        
        # Call tabs render
        self.render_tabs()
        
        # Verify tabs creation with correct labels
        self.mock_st.tabs.assert_called_once()
        tab_args = self.mock_st.tabs.call_args[0][0]
        
        # Check tab names
        self.assertEqual(len(tab_args), 4)
        self.assertIn("Schema Graph", tab_args)
        self.assertIn("Performance Metrics", tab_args)
        self.assertIn("System Logs", tab_args)
        self.assertIn("Empirical Data", tab_args)
    
    def test_tooltip_rendering(self):
        """Test tooltip rendering with correct style."""
        # Define test tooltip text
        tooltip_text = "Test tooltip"
        
        # Call tooltip render
        self.render_tooltip("Test", tooltip_text)
        
        # Verify markdown call with tooltip HTML
        markdown_calls = [call for call in self.mock_st.markdown.call_args_list 
                         if isinstance(call[0][0], str) and "tooltip" in call[0][0] 
                         and tooltip_text in call[0][0]]
        
        self.assertTrue(len(markdown_calls) > 0, "Tooltip HTML not found")
        tooltip_html = markdown_calls[0][0][0]
        
        # Verify tooltip contains required components
        self.assertIn("<div class=\"tooltip\"", tooltip_html)
        self.assertIn("<span class=\"tooltiptext\"", tooltip_html)
        self.assertIn(tooltip_text, tooltip_html)
        
        # Verify unsafe allow is enabled for HTML
        unsafe_calls = [call for call in self.mock_st.markdown.call_args_list 
                       if len(call[1]) > 0 and 'unsafe_allow_html' in call[1] 
                       and call[1]['unsafe_allow_html'] is True]
        
        self.assertTrue(len(unsafe_calls) > 0, "Unsafe allow HTML not enabled for tooltip")
    
    def test_sidebar_controls(self):
        """Test sidebar controls rendering and layout."""
        # Call sidebar render
        self.render_sidebar()
        
        # Verify sidebar header
        sidebar_title_calls = [call for call in self.mock_st.sidebar.markdown.call_args_list 
                              if isinstance(call[0][0], str) and "##" in call[0][0]]
        
        self.assertTrue(len(sidebar_title_calls) > 0, "Sidebar header not found")
        
        # Verify control sections
        # Profile selection
        self.mock_st.sidebar.selectbox.assert_any_call(
            "Active Profile",
            options=["default", "healthcare", "legal", "narrative"],
            index=0,
            key="profile_selector",
            help=unittest.mock.ANY
        )
        
        # Memory input
        self.mock_st.sidebar.text_area.assert_any_call(
            "Add New Memory", 
            help=unittest.mock.ANY
        )
        
        # Memory submit button
        memory_button_calls = [call for call in self.mock_st.sidebar.button.call_args_list 
                              if call[0][0] == "Store Memory"]
        
        self.assertTrue(len(memory_button_calls) > 0, "Memory button not found")
        
        # Query input
        self.mock_st.sidebar.text_area.assert_any_call(
            "Ask a Question", 
            help=unittest.mock.ANY
        )
        
        # Query submit button
        query_button_calls = [call for call in self.mock_st.sidebar.button.call_args_list 
                             if call[0][0] == "Submit Query"]
        
        self.assertTrue(len(query_button_calls) > 0, "Query button not found")
        
        # Session controls section
        session_header_calls = [call for call in self.mock_st.sidebar.markdown.call_args_list 
                               if isinstance(call[0][0], str) and "Session Controls" in call[0][0]]
        
        self.assertTrue(len(session_header_calls) > 0, "Session controls section not found")
    
    def test_metrics_visualization(self):
        """Test metrics visualization components."""
        # Mock matplotlib to avoid actual rendering
        with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())):
            # Call metrics render
            self.render_metrics()
            
            # Verify title
            metrics_title_calls = [call for call in self.mock_st.markdown.call_args_list 
                                  if isinstance(call[0][0], str) and "Performance Metrics" in call[0][0]]
            
            self.assertTrue(len(metrics_title_calls) > 0, "Metrics title not found")
            
            # Verify metrics display
            self.mock_st.metric.assert_called()
            
            # Verify columns for layout
            self.mock_st.columns.assert_called()
            
            # Verify line chart creation for time series data
            plot_calls = 0
            for name, args, kwargs in self.mock_st.method_calls:
                if name == 'pyplot':
                    plot_calls += 1
            
            self.assertTrue(plot_calls > 0, "No plot visualizations found")
    
    def test_empirical_data_tab(self):
        """Test empirical data tab rendering."""
        # Call empirical data render
        self.render_empirical_data()
        
        # Verify title
        data_title_calls = [call for call in self.mock_st.markdown.call_args_list 
                          if isinstance(call[0][0], str) and "Empirical Data" in call[0][0]]
        
        self.assertTrue(len(data_title_calls) > 0, "Empirical data title not found")
        
        # Verify export options
        self.mock_st.download_button.assert_called()
        
        # Verify data tables
        self.mock_st.dataframe.assert_called()
        
        # Verify expanders for detailed data
        self.mock_st.expander.assert_called()

class TestResponsiveDesign(unittest.TestCase):
    """Test cases for responsive design elements."""
    
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
        
        # Import UI components
        from web_interface_demo import render_graph, apply_responsive_css
        self.render_graph = render_graph
        self.apply_responsive_css = apply_responsive_css
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_responsive_css(self):
        """Test responsive CSS application."""
        # Call responsive CSS function
        self.apply_responsive_css()
        
        # Verify CSS injection
        css_calls = [call for call in self.mock_st.markdown.call_args_list 
                    if isinstance(call[0][0], str) and "<style>" in call[0][0]]
        
        self.assertTrue(len(css_calls) > 0, "Responsive CSS not injected")
        
        css_content = css_calls[0][0][0]
        
        # Verify media queries for responsiveness
        self.assertIn("@media", css_content, "No media queries found in CSS")
        
        # Verify unsafe allow is enabled for CSS
        for call in css_calls:
            kwargs = call[1] if len(call) > 1 else {}
            self.assertTrue(kwargs.get('unsafe_allow_html', False), 
                           "Unsafe HTML not enabled for CSS injection")
    
    def test_graph_sizing(self):
        """Test graph sizing for responsiveness."""
        # Mock matplotlib to avoid actual rendering
        with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())):
            with patch('matplotlib.pyplot.Figure.set_size_inches') as mock_set_size:
                # Call graph render
                self.render_graph()
                
                # Verify size is set on figure
                mock_set_size.assert_called()

class TestAccessibility(unittest.TestCase):
    """Test cases for accessibility features."""
    
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
        
        # Import UI components
        from web_interface_demo import render_sidebar, render_header, render_tooltip
        self.render_sidebar = render_sidebar
        self.render_header = render_header
        self.render_tooltip = render_tooltip
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_help_text_provided(self):
        """Test that help text is provided for UI elements."""
        # Call sidebar render where most controls are
        self.render_sidebar()
        
        # Check for help parameter in UI elements
        help_texts = []
        
        # Check selectbox calls
        for name, args, kwargs in self.mock_st.sidebar.selectbox.call_args_list:
            if 'help' in kwargs and kwargs['help'] is not None:
                help_texts.append(kwargs['help'])
        
        # Check text_area calls
        for name, args, kwargs in self.mock_st.sidebar.text_area.call_args_list:
            if 'help' in kwargs and kwargs['help'] is not None:
                help_texts.append(kwargs['help'])
        
        # Check button calls
        for name, args, kwargs in self.mock_st.sidebar.button.call_args_list:
            if 'help' in kwargs and kwargs['help'] is not None:
                help_texts.append(kwargs['help'])
        
        # Verify help texts are provided
        self.assertTrue(len(help_texts) > 0, "No help texts found for UI controls")
    
    def test_color_contrast(self):
        """Test color contrast in CSS styling."""
        # Call header render where CSS is applied
        self.render_header()
        
        # Check CSS for color definitions
        css_calls = [call for call in self.mock_st.markdown.call_args_list 
                    if isinstance(call[0][0], str) and "<style>" in call[0][0]]
        
        self.assertTrue(len(css_calls) > 0, "No CSS styling found")
        
        # Extract CSS content from first style block
        css_content = css_calls[0][0][0]
        
        # Check for color definitions that should have sufficient contrast
        contains_dark_colors = any(color in css_content for color in 
                                  ['#000', '#111', '#222', '#333', 'rgb(0,0,0)', 'rgba(0,0,0'])
        
        contains_light_bg = any(color in css_content for color in 
                               ['#fff', '#f8f8f8', '#eee', 'rgb(255,255,255)', 'rgba(255,255,255'])
        
        self.assertTrue(contains_dark_colors or contains_light_bg, 
                       "No standard contrast colors found in CSS")

if __name__ == '__main__':
    unittest.main() 