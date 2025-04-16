"""
Plugin Loader for ΨC-AI SDK

This module provides utilities for loading and instantiating plugins for the ΨC-AI SDK.
It handles dynamic import of plugin modules and instantiation of plugin classes.
"""

import importlib
import logging
import sys
from typing import Dict, List, Optional, Type, Any, Union

from psi_c_ai_sdk.plugin.base import PluginBase, PluginInfo

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Utility class for loading ΨC-AI SDK plugins.
    
    This class provides methods for dynamically loading plugin modules and
    instantiating plugin classes.
    """
    
    @staticmethod
    def load_plugin_class(entry_point: str) -> Type[PluginBase]:
        """
        Load a plugin class from its entry point.
        
        Args:
            entry_point: Module path to the plugin class (e.g., "my_package.my_module.MyPlugin")
            
        Returns:
            Plugin class
            
        Raises:
            ImportError: If the plugin module cannot be imported
            AttributeError: If the plugin class cannot be found in the module
            TypeError: If the plugin class does not inherit from PluginBase
        """
        if not entry_point:
            raise ValueError("Entry point cannot be empty")
        
        # Split the entry point into module path and class name
        try:
            module_path, class_name = entry_point.rsplit(".", 1)
        except ValueError:
            raise ValueError(f"Invalid entry point format: {entry_point}")
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the plugin class
            plugin_class = getattr(module, class_name)
            
            # Verify that the class inherits from PluginBase
            if not issubclass(plugin_class, PluginBase):
                raise TypeError(
                    f"Plugin class {class_name} does not inherit from PluginBase"
                )
            
            return plugin_class
            
        except ImportError as e:
            logger.error(f"Failed to import plugin module {module_path}: {str(e)}")
            raise
        
        except AttributeError as e:
            logger.error(f"Failed to find plugin class {class_name} in module {module_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_plugin(
        entry_point: str, config: Optional[Dict[str, Any]] = None
    ) -> PluginBase:
        """
        Load and instantiate a plugin from its entry point.
        
        Args:
            entry_point: Module path to the plugin class
            config: Optional configuration for the plugin
            
        Returns:
            Plugin instance
            
        Raises:
            ImportError: If the plugin module cannot be imported
            AttributeError: If the plugin class cannot be found in the module
            TypeError: If the plugin class does not inherit from PluginBase
            RuntimeError: If the plugin fails to initialize
        """
        # Load the plugin class
        plugin_class = PluginLoader.load_plugin_class(entry_point)
        
        # Instantiate the plugin
        plugin = plugin_class(config)
        
        # Initialize the plugin
        if not plugin.initialize():
            raise RuntimeError(f"Failed to initialize plugin: {entry_point}")
        
        return plugin
    
    @staticmethod
    def add_plugin_path(path: str) -> None:
        """
        Add a path to the Python module search path.
        
        Args:
            path: Directory path to add to sys.path
        """
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.info(f"Added plugin path to sys.path: {path}")


def load_plugin_class(entry_point: str) -> Type[PluginBase]:
    """
    Convenience function to load a plugin class from its entry point.
    
    Args:
        entry_point: Module path to the plugin class
        
    Returns:
        Plugin class
    """
    return PluginLoader.load_plugin_class(entry_point)


def load_plugin(
    entry_point: str, config: Optional[Dict[str, Any]] = None
) -> PluginBase:
    """
    Convenience function to load and instantiate a plugin from its entry point.
    
    Args:
        entry_point: Module path to the plugin class
        config: Optional configuration for the plugin
        
    Returns:
        Plugin instance
    """
    return PluginLoader.load_plugin(entry_point, config) 