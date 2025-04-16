# psi_c_ai_sdk/core/trace_context.py

"""
Trace Context Module

This module provides a global context for tracing operations across the system,
allowing different components to access the current trace ID and propagate it
to related operations for introspection and visualization purposes.
"""

import uuid
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, Set, List


class TraceContextManager:
    """
    Thread-local storage for trace context information.
    
    This class allows trace information to be propagated through
    the call stack without explicitly passing it as parameters.
    """
    
    def __init__(self):
        """Initialize a new trace context manager."""
        self._local_storage = threading.local()
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize the thread-local storage if needed."""
        if not hasattr(self._local_storage, "trace_id"):
            self._local_storage.trace_id = None
            self._local_storage.trace_data = {}
            self._local_storage.active_contexts = set()
    
    @property
    def current_trace_id(self) -> Optional[str]:
        """Get the current trace ID, if any."""
        self._initialize_storage()
        return self._local_storage.trace_id
    
    @current_trace_id.setter
    def current_trace_id(self, trace_id: Optional[str]) -> None:
        """Set the current trace ID."""
        self._initialize_storage()
        self._local_storage.trace_id = trace_id
    
    @property
    def trace_data(self) -> Dict[str, Any]:
        """Get the current trace data dictionary."""
        self._initialize_storage()
        return self._local_storage.trace_data
    
    def get_trace_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the current trace data dictionary.
        
        Args:
            key: Key to look up
            default: Default value if key not found
            
        Returns:
            Value associated with the key or default
        """
        self._initialize_storage()
        return self._local_storage.trace_data.get(key, default)
    
    def set_trace_value(self, key: str, value: Any) -> None:
        """
        Set a value in the current trace data dictionary.
        
        Args:
            key: Key to set
            value: Value to associate with the key
        """
        self._initialize_storage()
        self._local_storage.trace_data[key] = value
    
    def register_context(self, context_name: str) -> None:
        """
        Register an active trace context.
        
        Args:
            context_name: Name of the context
        """
        self._initialize_storage()
        self._local_storage.active_contexts.add(context_name)
    
    def unregister_context(self, context_name: str) -> None:
        """
        Unregister an active trace context.
        
        Args:
            context_name: Name of the context
        """
        self._initialize_storage()
        if hasattr(self._local_storage, "active_contexts"):
            self._local_storage.active_contexts.discard(context_name)
    
    @property
    def active_contexts(self) -> Set[str]:
        """Get the set of active context names."""
        self._initialize_storage()
        return self._local_storage.active_contexts
    
    def is_context_active(self, context_name: str) -> bool:
        """
        Check if a context is active.
        
        Args:
            context_name: Name of the context to check
            
        Returns:
            True if the context is active, False otherwise
        """
        self._initialize_storage()
        return context_name in self._local_storage.active_contexts
    
    def clear(self) -> None:
        """Clear all trace context information."""
        self._initialize_storage()
        self._local_storage.trace_id = None
        self._local_storage.trace_data = {}
        self._local_storage.active_contexts = set()


# Global trace context manager instance
_trace_context_manager = TraceContextManager()


@contextmanager
def trace_context(context_name: str, trace_id: Optional[str] = None, **kwargs) -> None:
    """
    Context manager for a trace operation.
    
    This context manager sets up a trace context with the specified name and ID,
    and optionally adds additional data to the trace context.
    
    Args:
        context_name: Name of the context
        trace_id: Optional trace ID (generated if not provided)
        **kwargs: Additional data to add to the trace context
    """
    # Save previous trace ID
    previous_trace_id = _trace_context_manager.current_trace_id
    previous_trace_data = _trace_context_manager.trace_data.copy()
    
    # Set new trace context
    new_trace_id = trace_id or str(uuid.uuid4())
    _trace_context_manager.current_trace_id = new_trace_id
    
    # Update trace data with kwargs
    for key, value in kwargs.items():
        _trace_context_manager.set_trace_value(key, value)
    
    # Register context
    _trace_context_manager.register_context(context_name)
    
    try:
        yield
    finally:
        # Restore previous trace context
        _trace_context_manager.unregister_context(context_name)
        _trace_context_manager.current_trace_id = previous_trace_id
        
        # Restore previous trace data
        _trace_context_manager.clear()
        for key, value in previous_trace_data.items():
            _trace_context_manager.set_trace_value(key, value)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID.
    
    Returns:
        Current trace ID or None if no trace context is active
    """
    return _trace_context_manager.current_trace_id


def set_current_trace_id(trace_id: Optional[str]) -> None:
    """
    Set the current trace ID.
    
    Args:
        trace_id: Trace ID to set
    """
    _trace_context_manager.current_trace_id = trace_id


def get_trace_value(key: str, default: Any = None) -> Any:
    """
    Get a value from the current trace context.
    
    Args:
        key: Key to look up
        default: Default value if key not found
        
    Returns:
        Value associated with the key or default
    """
    return _trace_context_manager.get_trace_value(key, default)


def set_trace_value(key: str, value: Any) -> None:
    """
    Set a value in the current trace context.
    
    Args:
        key: Key to set
        value: Value to associate with the key
    """
    _trace_context_manager.set_trace_value(key, value)


def is_context_active(context_name: str) -> bool:
    """
    Check if a trace context is active.
    
    Args:
        context_name: Name of the context to check
        
    Returns:
        True if the context is active, False otherwise
    """
    return _trace_context_manager.is_context_active(context_name)


def get_active_contexts() -> List[str]:
    """
    Get the list of active trace contexts.
    
    Returns:
        List of active context names
    """
    return list(_trace_context_manager.active_contexts)


def clear_trace_context() -> None:
    """Clear all trace context information."""
    _trace_context_manager.clear()
