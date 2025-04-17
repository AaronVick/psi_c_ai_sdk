#!/usr/bin/env python3
"""
ΨC Demo Message Handlers Tests
-----------------------------
Tests for the message and event handling components of the ΨC-AI SDK demonstration.
These tests validate message processing, event dispatching, and callback mechanisms
used throughout the demo application.
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

# Mock the Streamlit module
sys.modules['streamlit'] = MagicMock()
import streamlit as st

class TestEventSystem(unittest.TestCase):
    """Test cases for the demo application's event system."""
    
    def setUp(self):
        """Set up test environment."""
        # Import event system classes and functions after patching
        from event_system import (
            EventType,
            Event,
            EventDispatcher,
            EventSubscriber
        )
        
        self.EventType = EventType
        self.Event = Event
        self.EventDispatcher = EventDispatcher
        self.EventSubscriber = EventSubscriber
        
        # Create test event subscribers and handlers
        self.test_handler = MagicMock()
        self.test_handler2 = MagicMock()
    
    def test_event_creation(self):
        """Test creating events with different types and payloads."""
        # Create a test event
        event = self.Event(self.EventType.QUERY_PROCESSED, {"response": "Test response"})
        
        # Verify event properties
        self.assertEqual(event.event_type, self.EventType.QUERY_PROCESSED)
        self.assertEqual(event.payload["response"], "Test response")
        self.assertIsNotNone(event.timestamp)
    
    def test_event_subscription(self):
        """Test subscribing to events and receiving notifications."""
        # Create dispatcher and subscriber
        dispatcher = self.EventDispatcher()
        subscriber = self.EventSubscriber()
        
        # Subscribe to events
        subscriber.subscribe(self.EventType.MEMORY_ADDED, self.test_handler)
        subscriber.subscribe(self.EventType.QUERY_PROCESSED, self.test_handler2)
        
        # Register subscriber with dispatcher
        dispatcher.register_subscriber(subscriber)
        
        # Dispatch events
        dispatcher.dispatch(self.Event(self.EventType.MEMORY_ADDED, {"memory_id": "mem1"}))
        dispatcher.dispatch(self.Event(self.EventType.QUERY_PROCESSED, {"query": "test"}))
        
        # Verify handlers were called
        self.test_handler.assert_called_once()
        self.test_handler2.assert_called_once()
    
    def test_event_filtering(self):
        """Test filtering events based on type."""
        # Create dispatcher and subscriber
        dispatcher = self.EventDispatcher()
        subscriber = self.EventSubscriber()
        
        # Subscribe to specific event type
        subscriber.subscribe(self.EventType.MEMORY_ADDED, self.test_handler)
        
        # Register subscriber with dispatcher
        dispatcher.register_subscriber(subscriber)
        
        # Dispatch events of different types
        dispatcher.dispatch(self.Event(self.EventType.MEMORY_ADDED, {"memory_id": "mem1"}))
        dispatcher.dispatch(self.Event(self.EventType.QUERY_PROCESSED, {"query": "test"}))
        
        # Verify only relevant handler was called
        self.test_handler.assert_called_once()
        self.assertEqual(self.test_handler.call_count, 1)
    
    def test_event_unsubscribe(self):
        """Test unsubscribing from events."""
        # Create dispatcher and subscriber
        dispatcher = self.EventDispatcher()
        subscriber = self.EventSubscriber()
        
        # Subscribe to events
        subscriber.subscribe(self.EventType.MEMORY_ADDED, self.test_handler)
        
        # Register subscriber with dispatcher
        dispatcher.register_subscriber(subscriber)
        
        # Dispatch an event and verify handler was called
        dispatcher.dispatch(self.Event(self.EventType.MEMORY_ADDED, {"memory_id": "mem1"}))
        self.test_handler.assert_called_once()
        
        # Unsubscribe from events
        subscriber.unsubscribe(self.EventType.MEMORY_ADDED, self.test_handler)
        
        # Dispatch another event and verify handler was not called again
        dispatcher.dispatch(self.Event(self.EventType.MEMORY_ADDED, {"memory_id": "mem2"}))
        self.assertEqual(self.test_handler.call_count, 1)
    
    def test_event_chaining(self):
        """Test event chaining and propagation."""
        # Create mock event handlers that dispatch new events
        def chain_handler(event):
            # Handler that dispatches a new event
            nonlocal dispatcher
            dispatcher.dispatch(self.Event(self.EventType.SCHEMA_UPDATED, {"triggered_by": event.event_type}))
        
        def final_handler(event):
            # Final handler in the chain
            self.assertEqual(event.payload["triggered_by"], self.EventType.MEMORY_ADDED)
        
        # Create dispatcher and subscribers
        dispatcher = self.EventDispatcher()
        
        subscriber1 = self.EventSubscriber()
        subscriber1.subscribe(self.EventType.MEMORY_ADDED, chain_handler)
        
        subscriber2 = self.EventSubscriber()
        subscriber2.subscribe(self.EventType.SCHEMA_UPDATED, final_handler)
        
        # Register subscribers
        dispatcher.register_subscriber(subscriber1)
        dispatcher.register_subscriber(subscriber2)
        
        # Start event chain
        dispatcher.dispatch(self.Event(self.EventType.MEMORY_ADDED, {"memory_id": "mem1"}))
        
        # Verification is done in the final_handler


class TestMessageHandlers(unittest.TestCase):
    """Test cases for message handling in the demo application."""
    
    def setUp(self):
        """Set up test environment."""
        # Import message handler functions after patching
        with patch('psi_c_ai_sdk.orchestration.orchestrator.Orchestrator'):
            from message_handlers import (
                process_input_message,
                handle_reflection_message,
                handle_error_message,
                format_message_for_display
            )
            
            self.process_input_message = process_input_message
            self.handle_reflection_message = handle_reflection_message
            self.handle_error_message = handle_error_message
            self.format_message_for_display = format_message_for_display
        
        # Mock orchestrator and event dispatcher
        self.mock_orchestrator = MagicMock()
        self.mock_dispatcher = MagicMock()
        self.mock_state = {
            'messages': [],
            'schema': {"nodes": [], "edges": []}
        }
    
    def test_process_input_message(self):
        """Test processing a user input message."""
        # Set up orchestrator mock response
        self.mock_orchestrator.process_query.return_value = {
            "response": "This is a response",
            "confidence": 0.85,
            "processing_time": 0.5
        }
        
        # Process a message
        with patch('streamlit.session_state', self.mock_state):
            result = self.process_input_message(
                "What is the meaning of life?",
                self.mock_orchestrator,
                self.mock_dispatcher
            )
        
        # Verify processing
        self.mock_orchestrator.process_query.assert_called_once()
        self.assertEqual(result["response"], "This is a response")
        self.assertEqual(result["confidence"], 0.85)
        self.assertEqual(len(self.mock_state['messages']), 2)  # User msg + response
    
    def test_handle_reflection_message(self):
        """Test handling a reflection message."""
        # Set up reflection data
        reflection_data = {
            "focus": "schema_coherence",
            "insight": "The schema shows improving coherence over time.",
            "action": "No changes needed at this time."
        }
        
        # Process a reflection message
        with patch('streamlit.session_state', self.mock_state):
            self.handle_reflection_message(
                reflection_data,
                self.mock_dispatcher
            )
        
        # Verify handling
        self.assertEqual(len(self.mock_state['messages']), 1)
        self.assertEqual(self.mock_state['messages'][0]['type'], 'reflection')
        self.assertEqual(self.mock_state['messages'][0]['content']['focus'], 'schema_coherence')
        self.mock_dispatcher.dispatch.assert_called_once()
    
    def test_handle_error_message(self):
        """Test handling an error message."""
        # Set up error data
        error_data = {
            "error_type": "processing_error",
            "message": "Failed to process query due to an internal error",
            "details": "Exception: Division by zero"
        }
        
        # Process an error message
        with patch('streamlit.session_state', self.mock_state):
            self.handle_error_message(
                error_data,
                self.mock_dispatcher
            )
        
        # Verify handling
        self.assertEqual(len(self.mock_state['messages']), 1)
        self.assertEqual(self.mock_state['messages'][0]['type'], 'error')
        self.assertEqual(self.mock_state['messages'][0]['content']['error_type'], 'processing_error')
        self.mock_dispatcher.dispatch.assert_called_once()
    
    def test_format_message_for_display(self):
        """Test formatting messages for display."""
        # Set up test messages
        user_message = {
            'type': 'user',
            'content': 'What is the meaning of life?',
            'timestamp': '2023-01-01T12:00:00'
        }
        
        response_message = {
            'type': 'response',
            'content': 'This is a response to your question.',
            'metadata': {
                'confidence': 0.85,
                'processing_time': 0.5
            },
            'timestamp': '2023-01-01T12:00:01'
        }
        
        reflection_message = {
            'type': 'reflection',
            'content': {
                'focus': 'schema_coherence',
                'insight': 'The schema shows improving coherence.',
                'action': 'No changes needed.'
            },
            'timestamp': '2023-01-01T12:00:02'
        }
        
        # Format messages
        user_format = self.format_message_for_display(user_message)
        response_format = self.format_message_for_display(response_message)
        reflection_format = self.format_message_for_display(reflection_message)
        
        # Verify formatting
        self.assertIn('What is the meaning of life?', user_format)
        self.assertIn('This is a response to your question.', response_format)
        self.assertIn('confidence: 0.85', response_format)
        self.assertIn('Schema Coherence', reflection_format)
        self.assertIn('The schema shows improving coherence.', reflection_format)


class TestNotificationSystem(unittest.TestCase):
    """Test cases for the notification system in the demo application."""
    
    def setUp(self):
        """Set up test environment."""
        # Import notification system functions after patching
        from notification_system import (
            NotificationType,
            create_notification,
            show_notification,
            dismiss_notification
        )
        
        self.NotificationType = NotificationType
        self.create_notification = create_notification
        self.show_notification = show_notification
        self.dismiss_notification = dismiss_notification
        
        # Mock state
        self.mock_state = {
            'notifications': []
        }
    
    def test_create_notification(self):
        """Test creating a notification."""
        # Create notifications of different types
        info_notification = self.create_notification(
            self.NotificationType.INFO,
            "Information notification",
            "This is an informational message."
        )
        
        warning_notification = self.create_notification(
            self.NotificationType.WARNING,
            "Warning notification",
            "This is a warning message."
        )
        
        error_notification = self.create_notification(
            self.NotificationType.ERROR,
            "Error notification",
            "This is an error message."
        )
        
        # Verify notification properties
        self.assertEqual(info_notification['type'], self.NotificationType.INFO)
        self.assertEqual(info_notification['title'], "Information notification")
        self.assertEqual(info_notification['message'], "This is an informational message.")
        self.assertIsNotNone(info_notification['id'])
        self.assertIsNotNone(info_notification['timestamp'])
        
        self.assertEqual(warning_notification['type'], self.NotificationType.WARNING)
        self.assertEqual(error_notification['type'], self.NotificationType.ERROR)
    
    def test_show_notification(self):
        """Test showing a notification."""
        # Create a notification
        notification = self.create_notification(
            self.NotificationType.INFO,
            "Test Notification",
            "This is a test notification."
        )
        
        # Show the notification
        with patch('streamlit.session_state', self.mock_state):
            self.show_notification(notification)
        
        # Verify notification was added to state
        self.assertEqual(len(self.mock_state['notifications']), 1)
        self.assertEqual(self.mock_state['notifications'][0]['title'], "Test Notification")
    
    def test_dismiss_notification(self):
        """Test dismissing a notification."""
        # Create and show notifications
        notification1 = self.create_notification(
            self.NotificationType.INFO,
            "Notification 1",
            "This is notification 1."
        )
        
        notification2 = self.create_notification(
            self.NotificationType.WARNING,
            "Notification 2",
            "This is notification 2."
        )
        
        self.mock_state = {
            'notifications': [notification1, notification2]
        }
        
        # Dismiss a notification
        with patch('streamlit.session_state', self.mock_state):
            self.dismiss_notification(notification1['id'])
        
        # Verify notification was removed
        self.assertEqual(len(self.mock_state['notifications']), 1)
        self.assertEqual(self.mock_state['notifications'][0]['title'], "Notification 2")


class TestCommandSystem(unittest.TestCase):
    """Test cases for the command system in the demo application."""
    
    def setUp(self):
        """Set up test environment."""
        # Import command system functions after patching
        from command_system import (
            Command,
            CommandHistory,
            execute_command,
            undo_command,
            redo_command
        )
        
        self.Command = Command
        self.CommandHistory = CommandHistory
        self.execute_command = execute_command
        self.undo_command = undo_command
        self.redo_command = redo_command
        
        # Create mock command functions
        self.mock_execute = MagicMock()
        self.mock_undo = MagicMock()
    
    def test_command_creation(self):
        """Test creating a command."""
        # Create a test command
        command = self.Command(
            name="test_command",
            execute_fn=self.mock_execute,
            undo_fn=self.mock_undo,
            args=("arg1", "arg2"),
            kwargs={"kwarg1": "value1"}
        )
        
        # Verify command properties
        self.assertEqual(command.name, "test_command")
        self.assertEqual(command.args, ("arg1", "arg2"))
        self.assertEqual(command.kwargs, {"kwarg1": "value1"})
    
    def test_command_execution(self):
        """Test executing a command."""
        # Create a command history
        history = self.CommandHistory()
        
        # Create and execute a command
        command = self.Command(
            name="test_command",
            execute_fn=self.mock_execute,
            undo_fn=self.mock_undo,
            args=("arg1",),
            kwargs={"kwarg1": "value1"}
        )
        
        self.execute_command(command, history)
        
        # Verify command was executed and added to history
        self.mock_execute.assert_called_once_with("arg1", kwarg1="value1")
        self.assertEqual(len(history.command_stack), 1)
        self.assertEqual(len(history.undo_stack), 0)
    
    def test_command_undo(self):
        """Test undoing a command."""
        # Create a command history
        history = self.CommandHistory()
        
        # Create and execute a command
        command = self.Command(
            name="test_command",
            execute_fn=self.mock_execute,
            undo_fn=self.mock_undo,
            args=("arg1",),
            kwargs={"kwarg1": "value1"}
        )
        
        history.command_stack.append(command)
        
        # Undo the command
        self.undo_command(history)
        
        # Verify command was undone and moved to undo stack
        self.mock_undo.assert_called_once_with("arg1", kwarg1="value1")
        self.assertEqual(len(history.command_stack), 0)
        self.assertEqual(len(history.undo_stack), 1)
    
    def test_command_redo(self):
        """Test redoing a command."""
        # Create a command history
        history = self.CommandHistory()
        
        # Create a command and add it to the undo stack
        command = self.Command(
            name="test_command",
            execute_fn=self.mock_execute,
            undo_fn=self.mock_undo,
            args=("arg1",),
            kwargs={"kwarg1": "value1"}
        )
        
        history.undo_stack.append(command)
        
        # Redo the command
        self.redo_command(history)
        
        # Verify command was re-executed and moved to command stack
        self.mock_execute.assert_called_once_with("arg1", kwarg1="value1")
        self.assertEqual(len(history.command_stack), 1)
        self.assertEqual(len(history.undo_stack), 0)


if __name__ == '__main__':
    unittest.main() 