"""
Base Plugin classes and interfaces for ΨC-AI SDK

This module defines the base classes and interfaces for creating plugins in the ΨC-AI SDK.
All plugins must inherit from the PluginBase class and implement its required methods.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union, Callable

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins that can be created for the ΨC-AI SDK."""
    REFLECTION = auto()         # Custom reflection strategies
    MEMORY = auto()             # Memory management extensions
    BELIEF = auto()             # Belief system extensions
    SCHEMA = auto()             # Schema manipulation plugins
    COHERENCE = auto()          # Custom coherence scoring
    VISUALIZATION = auto()      # Visualization and UI extensions
    INTEGRATION = auto()        # External system integrations
    ANALYSIS = auto()           # Analysis and monitoring tools
    SECURITY = auto()           # Safety and security extensions
    CUSTOM = auto()             # Custom plugin types


class PluginHook(Enum):
    """Extension points where plugins can hook into the ΨC-AI SDK."""
    PRE_REFLECTION = auto()     # Before reflection cycle
    POST_REFLECTION = auto()    # After reflection cycle
    PRE_MEMORY_ADD = auto()     # Before adding a memory
    POST_MEMORY_ADD = auto()    # After adding a memory
    PRE_SCHEMA_UPDATE = auto()  # Before updating the schema
    POST_SCHEMA_UPDATE = auto() # After updating the schema
    COHERENCE_SCORING = auto()  # During coherence calculation
    BELIEF_REVISION = auto()    # During belief revision
    RUNTIME_MONITORING = auto() # During runtime
    CUSTOM_HOOK = auto()        # Custom extension point


class PluginStatus(Enum):
    """Status of a plugin in the registry."""
    REGISTERED = auto()     # Plugin is registered but not loaded
    ACTIVE = auto()         # Plugin is loaded and active
    DISABLED = auto()       # Plugin is loaded but disabled
    ERROR = auto()          # Plugin encountered an error
    UNINSTALLED = auto()    # Plugin is uninstalled


@dataclass
class PluginInfo:
    """Metadata about a plugin."""
    id: str                           # Unique identifier for the plugin
    name: str                         # Human-readable name
    version: str                      # Version string (semver)
    description: str                  # Description of the plugin
    author: str                       # Author name or organization
    plugin_type: PluginType           # Type of plugin
    hooks: Set[PluginHook]            # Hooks implemented by plugin
    dependencies: Dict[str, str] = field(default_factory=dict)  # Plugin ID to version requirement
    created_at: datetime = field(default_factory=datetime.now)  # Creation time
    updated_at: datetime = field(default_factory=datetime.now)  # Last update time
    config_schema: Optional[Dict] = None  # JSON schema for config validation
    tags: List[str] = field(default_factory=list)  # Searchable tags
    homepage: Optional[str] = None    # Plugin homepage URL
    license: str = "MIT"              # License information
    status: PluginStatus = PluginStatus.REGISTERED  # Current status
    entry_point: Optional[str] = None  # Module path to plugin class
    error_message: Optional[str] = None  # Error message if status is ERROR


T = TypeVar('T', bound='PluginBase')


class PluginBase(ABC):
    """
    Base class for all ΨC-AI SDK plugins.
    
    All plugins must inherit from this class and implement its required methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.info = self._get_plugin_info()
        self.hooks = self._register_hooks()
        self.logger = logging.getLogger(f"plugin.{self.info.id}")
    
    @classmethod
    @abstractmethod
    def _get_plugin_info(cls) -> PluginInfo:
        """
        Get metadata about the plugin.
        
        This method must be implemented by all plugin classes.
        
        Returns:
            PluginInfo object describing the plugin
        """
        pass
    
    @abstractmethod
    def _register_hooks(self) -> Dict[PluginHook, Callable]:
        """
        Register the hooks that this plugin implements.
        
        This method must be implemented by all plugin classes.
        
        Returns:
            Dictionary mapping from hook type to handler function
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the plugin. Called when the plugin is loaded.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        return True
    
    def shutdown(self) -> None:
        """
        Shutdown the plugin. Called when the plugin is unloaded.
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and process the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        return config
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Update plugin configuration.
        
        Args:
            config: New configuration dictionary
            
        Returns:
            True if configuration was updated successfully, False otherwise
        """
        try:
            validated_config = self.validate_config(config)
            self.config = validated_config
            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_hook_handler(self, hook: PluginHook) -> Optional[Callable]:
        """
        Get the handler function for a specific hook.
        
        Args:
            hook: The plugin hook to get the handler for
            
        Returns:
            Handler function if the hook is implemented, None otherwise
        """
        return self.hooks.get(hook)
    
    def has_hook(self, hook: PluginHook) -> bool:
        """
        Check if the plugin implements a specific hook.
        
        Args:
            hook: The plugin hook to check for
            
        Returns:
            True if the hook is implemented, False otherwise
        """
        return hook in self.hooks
    
    def __str__(self) -> str:
        """String representation of the plugin."""
        return f"{self.info.name} v{self.info.version} ({self.info.id})"


def create_plugin_id(name: str, author: str) -> str:
    """
    Create a unique plugin ID based on name and author.
    
    Args:
        name: Plugin name
        author: Plugin author
        
    Returns:
        Unique plugin ID
    """
    # Create a deterministic ID based on name and author
    base = f"{author}.{name}".lower().replace(" ", "-")
    # Add a short hash to ensure uniqueness
    unique_part = uuid.uuid5(uuid.NAMESPACE_DNS, base).hex[:8]
    return f"{base}-{unique_part}" 