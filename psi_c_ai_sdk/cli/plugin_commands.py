"""
CLI commands for managing ΨC-AI SDK plugins.

This module provides CLI commands for listing, installing, enabling, disabling,
and managing plugins for the ΨC-AI SDK.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from psi_c_ai_sdk.plugin.registry import get_plugin_registry, PluginRegistry
from psi_c_ai_sdk.plugin.base import PluginStatus, PluginType


def register_plugin_commands(subparsers: argparse._SubParsersAction) -> None:
    """
    Register plugin management commands with the CLI.
    
    Args:
        subparsers: Subparsers object to add commands to
    """
    # Create plugin parser
    plugin_parser = subparsers.add_parser(
        "plugin",
        help="Plugin management commands"
    )
    
    # Add plugin subcommands
    plugin_subparsers = plugin_parser.add_subparsers(
        dest="subcommand",
        help="Plugin management subcommands"
    )
    
    # List plugins command
    list_parser = plugin_subparsers.add_parser(
        "list",
        help="List available plugins"
    )
    list_parser.add_argument(
        "--type",
        choices=[t.name.lower() for t in PluginType],
        help="Filter by plugin type"
    )
    list_parser.add_argument(
        "--status",
        choices=[s.name.lower() for s in PluginStatus],
        help="Filter by plugin status"
    )
    list_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed plugin information"
    )
    list_parser.set_defaults(func=list_plugins)
    
    # Info command
    info_parser = plugin_subparsers.add_parser(
        "info",
        help="Show detailed information about a plugin"
    )
    info_parser.add_argument(
        "plugin_id",
        help="ID of the plugin to show information for"
    )
    info_parser.set_defaults(func=show_plugin_info)
    
    # Enable command
    enable_parser = plugin_subparsers.add_parser(
        "enable",
        help="Enable a plugin"
    )
    enable_parser.add_argument(
        "plugin_id",
        help="ID of the plugin to enable"
    )
    enable_parser.set_defaults(func=enable_plugin)
    
    # Disable command
    disable_parser = plugin_subparsers.add_parser(
        "disable",
        help="Disable a plugin"
    )
    disable_parser.add_argument(
        "plugin_id",
        help="ID of the plugin to disable"
    )
    disable_parser.set_defaults(func=disable_plugin)
    
    # Discover command
    discover_parser = plugin_subparsers.add_parser(
        "discover",
        help="Discover available plugins"
    )
    discover_parser.add_argument(
        "--path",
        help="Path to search for plugins"
    )
    discover_parser.set_defaults(func=discover_plugins)
    
    # Add path command
    add_path_parser = plugin_subparsers.add_parser(
        "add-path",
        help="Add a path to search for plugins"
    )
    add_path_parser.add_argument(
        "path",
        help="Path to add"
    )
    add_path_parser.set_defaults(func=add_plugin_path)


def list_plugins(args: argparse.Namespace) -> None:
    """
    List available plugins.
    
    Args:
        args: Command-line arguments
    """
    registry = get_plugin_registry()
    
    # Get all plugins
    plugins = registry.get_all_plugins()
    
    # Filter by type if specified
    if args.type:
        plugin_type = PluginType[args.type.upper()]
        plugins = [p for p in plugins if p.plugin_type == plugin_type]
    
    # Filter by status if specified
    if args.status:
        plugin_status = PluginStatus[args.status.upper()]
        plugins = [p for p in plugins if p.status == plugin_status]
    
    # Display plugin information
    if not plugins:
        print("No plugins found.")
        return
    
    print(f"Found {len(plugins)} plugins:")
    
    for plugin in plugins:
        # Basic info
        status_str = f"[{plugin.status.name}]"
        print(f"{plugin.id} - {plugin.name} v{plugin.version} {status_str}")
        
        # Verbose mode
        if args.verbose:
            print(f"  Description: {plugin.description}")
            print(f"  Author: {plugin.author}")
            print(f"  Type: {plugin.plugin_type.name}")
            print(f"  Hooks: {', '.join(h.name for h in plugin.hooks)}")
            print(f"  Tags: {', '.join(plugin.tags)}")
            
            if plugin.status == PluginStatus.ERROR and plugin.error_message:
                print(f"  Error: {plugin.error_message}")
            
            print()


def show_plugin_info(args: argparse.Namespace) -> None:
    """
    Show detailed information about a plugin.
    
    Args:
        args: Command-line arguments
    """
    registry = get_plugin_registry()
    
    # Get plugin info
    plugin_info = registry.get_plugin_info(args.plugin_id)
    
    if not plugin_info:
        print(f"Plugin not found: {args.plugin_id}")
        return
    
    # Display detailed plugin information
    print(f"Plugin: {plugin_info.name} ({plugin_info.id})")
    print(f"Version: {plugin_info.version}")
    print(f"Status: {plugin_info.status.name}")
    print(f"Description: {plugin_info.description}")
    print(f"Author: {plugin_info.author}")
    print(f"Type: {plugin_info.plugin_type.name}")
    print(f"License: {plugin_info.license}")
    
    if plugin_info.homepage:
        print(f"Homepage: {plugin_info.homepage}")
    
    print(f"Hooks: {', '.join(h.name for h in plugin_info.hooks)}")
    
    if plugin_info.tags:
        print(f"Tags: {', '.join(plugin_info.tags)}")
    
    if plugin_info.dependencies:
        print("Dependencies:")
        for dep_id, version in plugin_info.dependencies.items():
            print(f"  {dep_id}: {version}")
    
    if plugin_info.status == PluginStatus.ERROR and plugin_info.error_message:
        print(f"Error: {plugin_info.error_message}")
    
    # Show if plugin is loaded
    plugin = registry.get_plugin(args.plugin_id)
    if plugin:
        print("Plugin is currently loaded.")


def enable_plugin(args: argparse.Namespace) -> None:
    """
    Enable a plugin.
    
    Args:
        args: Command-line arguments
    """
    registry = get_plugin_registry()
    
    # Check if plugin exists
    if not registry.get_plugin_info(args.plugin_id):
        print(f"Plugin not found: {args.plugin_id}")
        return
    
    # Enable plugin
    success = registry.enable_plugin(args.plugin_id)
    
    if success:
        print(f"Plugin enabled: {args.plugin_id}")
    else:
        print(f"Failed to enable plugin: {args.plugin_id}")
        
        # Show error message if available
        plugin_info = registry.get_plugin_info(args.plugin_id)
        if plugin_info and plugin_info.error_message:
            print(f"Error: {plugin_info.error_message}")


def disable_plugin(args: argparse.Namespace) -> None:
    """
    Disable a plugin.
    
    Args:
        args: Command-line arguments
    """
    registry = get_plugin_registry()
    
    # Check if plugin exists
    if not registry.get_plugin_info(args.plugin_id):
        print(f"Plugin not found: {args.plugin_id}")
        return
    
    # Disable plugin
    success = registry.disable_plugin(args.plugin_id)
    
    if success:
        print(f"Plugin disabled: {args.plugin_id}")
    else:
        print(f"Failed to disable plugin: {args.plugin_id}")
        
        # Show error message if available
        plugin_info = registry.get_plugin_info(args.plugin_id)
        if plugin_info and plugin_info.error_message:
            print(f"Error: {plugin_info.error_message}")


def discover_plugins(args: argparse.Namespace) -> None:
    """
    Discover available plugins.
    
    Args:
        args: Command-line arguments
    """
    registry = get_plugin_registry()
    
    # Add path if specified
    if args.path:
        registry.add_plugin_path(args.path)
    
    # Discover plugins
    discovered = registry.discover_plugins()
    
    if not discovered:
        print("No new plugins discovered.")
        return
    
    print(f"Discovered {len(discovered)} new plugins:")
    
    for plugin in discovered:
        print(f"{plugin.id} - {plugin.name} v{plugin.version}")


def add_plugin_path(args: argparse.Namespace) -> None:
    """
    Add a path to search for plugins.
    
    Args:
        args: Command-line arguments
    """
    registry = get_plugin_registry()
    
    # Add path
    registry.add_plugin_path(args.path)
    
    print(f"Added plugin path: {args.path}")
    print("Use 'psi plugin discover' to find plugins in this path.") 