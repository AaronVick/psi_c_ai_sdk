"""
Plugin Registry for ΨC-AI SDK

This module provides a central registry for managing plugins in the ΨC-AI SDK.
The registry keeps track of all registered plugins, their status, and provides
methods for loading, unloading, and accessing plugins.
"""

import json
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Type, Any, Set, Union, Callable

from psi_c_ai_sdk.plugin.base import (
    PluginBase, 
    PluginType, 
    PluginInfo, 
    PluginHook,
    PluginStatus
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for ΨC-AI SDK plugins.
    
    This class keeps track of all registered plugins, their status, and provides
    methods for loading, unloading, and accessing plugins.
    """
    
    _instance = None
    _lock = RLock()
    
    def __init__(self):
        """Initialize the plugin registry."""
        if PluginRegistry._instance is not None:
            raise RuntimeError("PluginRegistry is a singleton, use get_plugin_registry() instead")
        
        self.plugins: Dict[str, PluginInfo] = {}  # Plugin ID -> PluginInfo
        self.instances: Dict[str, PluginBase] = {}  # Plugin ID -> Plugin instance
        self.hooks: Dict[PluginHook, List[str]] = {hook: [] for hook in PluginHook}  # Hook -> List of plugin IDs
        
        # Paths for plugin discovery
        self.plugin_paths: List[Path] = []
        
        # Default plugin directory
        default_path = Path.home() / ".psi_c_ai_sdk" / "plugins"
        self.add_plugin_path(default_path)
        
        # Add environment variable path if present
        if env_path := os.environ.get("PSI_C_PLUGIN_PATH"):
            self.add_plugin_path(Path(env_path))
    
    def add_plugin_path(self, path: Union[str, Path]) -> None:
        """
        Add a path to search for plugins.
        
        Args:
            path: Directory path to search for plugins
        """
        if isinstance(path, str):
            path = Path(path)
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        if path not in self.plugin_paths:
            self.plugin_paths.append(path)
            logger.info(f"Added plugin path: {path}")
    
    def register_plugin(self, plugin_info: PluginInfo) -> bool:
        """
        Register a plugin with the registry.
        
        Args:
            plugin_info: Plugin information
            
        Returns:
            True if registration was successful, False otherwise
        """
        with self._lock:
            if plugin_info.id in self.plugins:
                logger.warning(f"Plugin {plugin_info.id} is already registered")
                return False
            
            self.plugins[plugin_info.id] = plugin_info
            
            # Register plugin hooks
            for hook in plugin_info.hooks:
                if plugin_info.id not in self.hooks[hook]:
                    self.hooks[hook].append(plugin_info.id)
            
            logger.info(f"Registered plugin: {plugin_info.name} ({plugin_info.id})")
            return True
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """
        Unregister a plugin from the registry.
        
        Args:
            plugin_id: ID of the plugin to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        with self._lock:
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin {plugin_id} is not registered")
                return False
            
            # Unload plugin if it's loaded
            if plugin_id in self.instances:
                self.unload_plugin(plugin_id)
            
            # Remove plugin hooks
            for hook_type, plugin_ids in self.hooks.items():
                if plugin_id in plugin_ids:
                    plugin_ids.remove(plugin_id)
            
            # Remove plugin info
            plugin_info = self.plugins.pop(plugin_id)
            logger.info(f"Unregistered plugin: {plugin_info.name} ({plugin_id})")
            
            return True
    
    def load_plugin(self, plugin_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[PluginBase]:
        """
        Load a plugin by ID.
        
        Args:
            plugin_id: ID of the plugin to load
            config: Optional configuration for the plugin
            
        Returns:
            Plugin instance if loading was successful, None otherwise
        """
        with self._lock:
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin {plugin_id} is not registered")
                return None
            
            if plugin_id in self.instances:
                logger.warning(f"Plugin {plugin_id} is already loaded")
                return self.instances[plugin_id]
            
            plugin_info = self.plugins[plugin_id]
            
            try:
                # Use plugin loader to instantiate the plugin
                # This avoids a circular import with the loader module
                from psi_c_ai_sdk.plugin.loader import load_plugin_class
                plugin_class = load_plugin_class(plugin_info.entry_point)
                
                # Create plugin instance
                plugin = plugin_class(config)
                
                # Initialize the plugin
                if not plugin.initialize():
                    raise RuntimeError(f"Failed to initialize plugin {plugin_id}")
                
                # Store plugin instance
                self.instances[plugin_id] = plugin
                
                # Update plugin status
                plugin_info.status = PluginStatus.ACTIVE
                
                logger.info(f"Loaded plugin: {plugin_info.name} ({plugin_id})")
                return plugin
                
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_id}: {str(e)}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)
                return None
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin by ID.
        
        Args:
            plugin_id: ID of the plugin to unload
            
        Returns:
            True if unloading was successful, False otherwise
        """
        with self._lock:
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin {plugin_id} is not registered")
                return False
            
            if plugin_id not in self.instances:
                logger.warning(f"Plugin {plugin_id} is not loaded")
                return False
            
            plugin = self.instances[plugin_id]
            plugin_info = self.plugins[plugin_id]
            
            try:
                # Shutdown the plugin
                plugin.shutdown()
                
                # Remove plugin instance
                del self.instances[plugin_id]
                
                # Update plugin status
                plugin_info.status = PluginStatus.REGISTERED
                
                logger.info(f"Unloaded plugin: {plugin_info.name} ({plugin_id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload plugin {plugin_id}: {str(e)}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)
                return False
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """
        Get a plugin instance by ID.
        
        Args:
            plugin_id: ID of the plugin to get
            
        Returns:
            Plugin instance if loaded, None otherwise
        """
        return self.instances.get(plugin_id)
    
    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """
        Get plugin information by ID.
        
        Args:
            plugin_id: ID of the plugin to get info for
            
        Returns:
            PluginInfo if registered, None otherwise
        """
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to get
            
        Returns:
            List of PluginInfo objects for plugins of the specified type
        """
        return [
            info for info in self.plugins.values()
            if info.plugin_type == plugin_type
        ]
    
    def get_plugins_for_hook(self, hook: PluginHook) -> List[PluginBase]:
        """
        Get all active plugins that implement a specific hook.
        
        Args:
            hook: Hook to get plugins for
            
        Returns:
            List of plugin instances that implement the hook
        """
        plugin_ids = self.hooks.get(hook, [])
        return [
            self.instances[plugin_id] for plugin_id in plugin_ids
            if plugin_id in self.instances
        ]
    
    def run_hook(self, hook: PluginHook, *args, **kwargs) -> Dict[str, Any]:
        """
        Run a hook on all plugins that implement it.
        
        Args:
            hook: Hook to run
            *args: Positional arguments to pass to the hook handler
            **kwargs: Keyword arguments to pass to the hook handler
            
        Returns:
            Dictionary mapping from plugin ID to hook result
        """
        results = {}
        for plugin in self.get_plugins_for_hook(hook):
            try:
                handler = plugin.get_hook_handler(hook)
                if handler:
                    results[plugin.info.id] = handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error running hook {hook} on plugin {plugin.info.id}: {str(e)}")
                results[plugin.info.id] = {"error": str(e)}
        
        return results
    
    def discover_plugins(self) -> List[PluginInfo]:
        """
        Discover plugins in the registered plugin paths.
        
        Returns:
            List of discovered plugin infos
        """
        discovered = []
        
        for path in self.plugin_paths:
            if not path.exists():
                continue
            
            # Look for plugin manifest files
            for manifest_file in path.glob("*/plugin.json"):
                try:
                    with open(manifest_file, "r") as f:
                        manifest = json.load(f)
                    
                    # Create plugin info from manifest
                    plugin_info = self._create_plugin_info_from_manifest(manifest, manifest_file.parent)
                    
                    # Register plugin
                    if plugin_info and self.register_plugin(plugin_info):
                        discovered.append(plugin_info)
                        
                except Exception as e:
                    logger.error(f"Error loading plugin manifest {manifest_file}: {str(e)}")
        
        return discovered
    
    def _create_plugin_info_from_manifest(
        self, manifest: Dict[str, Any], plugin_dir: Path
    ) -> Optional[PluginInfo]:
        """
        Create PluginInfo from a plugin manifest.
        
        Args:
            manifest: Plugin manifest dictionary
            plugin_dir: Directory containing the plugin
            
        Returns:
            PluginInfo if manifest is valid, None otherwise
        """
        try:
            # Required fields
            required_fields = ["id", "name", "version", "description", "author", "type", "entry_point"]
            for field in required_fields:
                if field not in manifest:
                    logger.error(f"Plugin manifest missing required field: {field}")
                    return None
            
            # Parse plugin type
            try:
                plugin_type = PluginType[manifest["type"].upper()]
            except KeyError:
                logger.error(f"Invalid plugin type: {manifest['type']}")
                return None
            
            # Parse hooks
            hooks = set()
            for hook_name in manifest.get("hooks", []):
                try:
                    hook = PluginHook[hook_name.upper()]
                    hooks.add(hook)
                except KeyError:
                    logger.warning(f"Invalid hook: {hook_name}")
            
            # Create plugin info
            plugin_info = PluginInfo(
                id=manifest["id"],
                name=manifest["name"],
                version=manifest["version"],
                description=manifest["description"],
                author=manifest["author"],
                plugin_type=plugin_type,
                hooks=hooks,
                dependencies=manifest.get("dependencies", {}),
                config_schema=manifest.get("config_schema"),
                tags=manifest.get("tags", []),
                homepage=manifest.get("homepage"),
                license=manifest.get("license", "MIT"),
                entry_point=manifest["entry_point"]
            )
            
            return plugin_info
            
        except Exception as e:
            logger.error(f"Error creating plugin info from manifest: {str(e)}")
            return None
    
    def get_all_plugins(self) -> List[PluginInfo]:
        """
        Get all registered plugins.
        
        Returns:
            List of all registered plugin infos
        """
        return list(self.plugins.values())
    
    def get_active_plugins(self) -> List[PluginInfo]:
        """
        Get all active plugins.
        
        Returns:
            List of active plugin infos
        """
        return [
            info for info in self.plugins.values()
            if info.status == PluginStatus.ACTIVE
        ]
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """
        Enable a plugin.
        
        Args:
            plugin_id: ID of the plugin to enable
            
        Returns:
            True if the plugin was enabled, False otherwise
        """
        with self._lock:
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin {plugin_id} is not registered")
                return False
            
            plugin_info = self.plugins[plugin_id]
            
            if plugin_info.status == PluginStatus.ACTIVE:
                logger.info(f"Plugin {plugin_id} is already active")
                return True
            
            # Load the plugin
            if self.load_plugin(plugin_id):
                return True
            
            return False
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """
        Disable a plugin.
        
        Args:
            plugin_id: ID of the plugin to disable
            
        Returns:
            True if the plugin was disabled, False otherwise
        """
        with self._lock:
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin {plugin_id} is not registered")
                return False
            
            plugin_info = self.plugins[plugin_id]
            
            if plugin_info.status != PluginStatus.ACTIVE:
                logger.info(f"Plugin {plugin_id} is not active")
                return True
            
            # Unload the plugin
            if self.unload_plugin(plugin_id):
                plugin_info.status = PluginStatus.DISABLED
                return True
            
            return False


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    Returns:
        The global plugin registry
    """
    if PluginRegistry._instance is None:
        with PluginRegistry._lock:
            if PluginRegistry._instance is None:
                PluginRegistry._instance = PluginRegistry()
    
    return PluginRegistry._instance 