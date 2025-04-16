"""
Plugin System for ΨC-AI SDK

This module provides extension points and plugin management for the ΨC-AI SDK.
It allows developers to extend the functionality of the system through custom
plugins for reflection logic, memory management, belief systems, and more.

The plugin system offers a standardized way to register, discover, and load
plugins, ensuring they integrate properly with the core SDK components.
"""

from psi_c_ai_sdk.plugin.registry import (
    PluginRegistry,
    get_plugin_registry
)

from psi_c_ai_sdk.plugin.base import (
    PluginBase,
    PluginInfo,
    PluginType,
    PluginHook,
    PluginStatus
)

from psi_c_ai_sdk.plugin.loader import (
    PluginLoader,
    load_plugin
)

__all__ = [
    'PluginRegistry',
    'get_plugin_registry',
    'PluginBase',
    'PluginInfo',
    'PluginType',
    'PluginHook',
    'PluginStatus',
    'PluginLoader',
    'load_plugin'
] 