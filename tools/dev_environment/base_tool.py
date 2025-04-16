"""
Base Tool - Foundation class for all development environment tools

This module provides the BaseTool class which defines common functionality
and interfaces that all development tools should implement.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Setup logging
logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """
    Base class for all development environment tools.
    
    This abstract class defines the common interface and functionality
    that all development tools should implement.
    """
    
    def __init__(self, agent=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool.
        
        Args:
            agent: ΨC agent instance to interact with
            config: Tool configuration options
        """
        self.agent = agent
        self.config = config or {}
        self.running = False
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def launch(self):
        """Launch the tool interface."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the tool interface and clean up resources."""
        pass
    
    def is_running(self) -> bool:
        """Check if the tool is currently running."""
        return self.running
    
    def get_agent(self):
        """Get the associated agent instance."""
        return self.agent
    
    def set_agent(self, agent):
        """Set a new agent instance to interact with."""
        self.agent = agent
        logger.info(f"{self.__class__.__name__} now using new agent instance")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the tool's configuration."""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update the tool's configuration."""
        self.config.update(new_config)
        logger.info(f"Updated {self.__class__.__name__} configuration")
    
    def save_state(self, path: str):
        """
        Save the tool's current state to a file.
        
        Args:
            path: File path to save the state to
        """
        logger.info(f"Saving {self.__class__.__name__} state to {path}")
    
    def load_state(self, path: str):
        """
        Load the tool's state from a file.
        
        Args:
            path: File path to load the state from
        """
        logger.info(f"Loading {self.__class__.__name__} state from {path}")
        
    def refresh(self):
        """Refresh/update the tool's display or state."""
        logger.debug(f"Refreshing {self.__class__.__name__}")
        
    def __enter__(self):
        """Context manager enter method."""
        self.launch()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.close() 