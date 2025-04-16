#!/usr/bin/env python3
"""
ΨC-AI SDK Command Line Interface

This module serves as the main entry point for the ΨC-AI SDK CLI,
providing tools for schema management, memory operations, and more.
"""

import sys
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union

from psi_c_ai_sdk.cli.schema_diff import (
    create_parser as create_schema_diff_parser,
    handle_diff_command,
    handle_list_command,
    handle_snapshot_command,
    handle_drift_command
)
from psi_c_ai_sdk.cli.memory_commands import (
    create_memory_command_parser,
    memory_cli_main,
    handle_list_command as handle_memory_list_command,
    handle_add_command,
    handle_get_command,
    handle_delete_command,
    handle_archive_command,
    handle_restore_command,
    handle_pin_command,
    handle_unpin_command,
    handle_export_command,
    handle_import_command,
    handle_decay_command,
    handle_cull_command
)
from psi_c_ai_sdk.cli.psic_commands import (
    create_psic_command_parser,
    psic_cli_main,
    handle_status_command,
    handle_log_command,
    handle_health_command,
    handle_collapse_command,
    handle_configure_command
)
from psi_c_ai_sdk.cli.plugin_commands import register_plugin_commands


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ΨC-AI SDK Command Line Interface",
        prog="psi",
        epilog="Use 'psi <command> --help' for more information on a specific command."
    )
    
    # Add version info
    parser.add_argument(
        "--version", 
        action="store_true", 
        help="Show version information and exit"
    )
    
    # Create subparsers for each command category
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Schema command group
    schema_parser = subparsers.add_parser(
        "schema", 
        help="Schema graph management commands"
    )
    schema_subparsers = schema_parser.add_subparsers(dest="subcommand")
    
    # Add schema diff subcommands
    schema_diff_parser = create_schema_diff_parser()
    diff_parser = schema_subparsers.add_parser(
        "diff", 
        parents=[schema_diff_parser], 
        help="Compare schema versions",
        conflict_handler='resolve'
    )
    diff_parser.set_defaults(func=handle_diff_command)
    
    list_parser = schema_subparsers.add_parser(
        "list", 
        help="List available schema snapshots"
    )
    list_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    list_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    list_parser.set_defaults(func=handle_list_command)
    
    snapshot_parser = schema_subparsers.add_parser(
        "snapshot", 
        help="Save current schema snapshot"
    )
    snapshot_parser.add_argument(
        "--label", 
        help="Label for the snapshot"
    )
    snapshot_parser.set_defaults(func=handle_snapshot_command)
    
    drift_parser = schema_subparsers.add_parser(
        "drift", 
        help="Analyze schema drift over time"
    )
    drift_parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.1,
        help="Drift significance threshold (default: 0.1)"
    )
    drift_parser.add_argument(
        "--history", 
        type=int, 
        default=5,
        help="Number of historical snapshots to analyze (default: 5)"
    )
    drift_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    drift_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    drift_parser.set_defaults(func=handle_drift_command)
    
    # Memory command group
    memory_parser = subparsers.add_parser(
        "memory", 
        help="Memory management commands"
    )
    memory_subparsers = memory_parser.add_subparsers(dest="subcommand")
    
    # List command
    memory_list_parser = memory_subparsers.add_parser(
        "list", 
        help="List memories in the store"
    )
    memory_list_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    memory_list_parser.add_argument(
        "--limit", 
        type=int, 
        default=20,
        help="Maximum number of memories to list (default: 20)"
    )
    memory_list_parser.add_argument(
        "--sort-by", 
        choices=["importance", "created", "accessed"], 
        default="importance",
        help="Sort criteria (default: importance)"
    )
    memory_list_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    memory_list_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_list_parser.set_defaults(func=handle_memory_list_command)
    
    # Add command
    memory_add_parser = memory_subparsers.add_parser(
        "add", 
        help="Add a new memory"
    )
    memory_add_parser.add_argument(
        "content", 
        help="Memory content text"
    )
    memory_add_parser.add_argument(
        "--importance", 
        type=float, 
        default=1.0,
        help="Initial importance score (default: 1.0)"
    )
    memory_add_parser.add_argument(
        "--tags", 
        nargs="+", 
        help="Tags for the memory"
    )
    memory_add_parser.add_argument(
        "--pin", 
        action="store_true",
        help="Pin the memory"
    )
    memory_add_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_add_parser.set_defaults(func=handle_add_command)
    
    # Get command
    memory_get_parser = memory_subparsers.add_parser(
        "get", 
        help="Retrieve a specific memory"
    )
    memory_get_parser.add_argument(
        "uuid", 
        help="UUID of the memory to retrieve"
    )
    memory_get_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    memory_get_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    memory_get_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_get_parser.set_defaults(func=handle_get_command)
    
    # Delete command
    memory_delete_parser = memory_subparsers.add_parser(
        "delete", 
        help="Delete a memory"
    )
    memory_delete_parser.add_argument(
        "uuid", 
        help="UUID of the memory to delete"
    )
    memory_delete_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_delete_parser.set_defaults(func=handle_delete_command)
    
    # Archive command
    memory_archive_parser = memory_subparsers.add_parser(
        "archive", 
        help="Archive a memory"
    )
    memory_archive_parser.add_argument(
        "uuid", 
        nargs="+",
        help="UUID(s) of the memory/memories to archive"
    )
    memory_archive_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_archive_parser.set_defaults(func=handle_archive_command)
    
    # Restore command
    memory_restore_parser = memory_subparsers.add_parser(
        "restore", 
        help="Restore a memory from archive"
    )
    memory_restore_parser.add_argument(
        "uuid", 
        help="UUID of the memory to restore"
    )
    memory_restore_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_restore_parser.set_defaults(func=handle_restore_command)
    
    # Pin/unpin commands
    memory_pin_parser = memory_subparsers.add_parser(
        "pin", 
        help="Pin a memory"
    )
    memory_pin_parser.add_argument(
        "uuid", 
        help="UUID of the memory to pin"
    )
    memory_pin_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_pin_parser.set_defaults(func=handle_pin_command)
    
    memory_unpin_parser = memory_subparsers.add_parser(
        "unpin", 
        help="Unpin a memory"
    )
    memory_unpin_parser.add_argument(
        "uuid", 
        help="UUID of the memory to unpin"
    )
    memory_unpin_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_unpin_parser.set_defaults(func=handle_unpin_command)
    
    # Export command
    memory_export_parser = memory_subparsers.add_parser(
        "export", 
        help="Export memories to file"
    )
    memory_export_parser.add_argument(
        "output_path", 
        help="Path to export memories to"
    )
    memory_export_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_export_parser.set_defaults(func=handle_export_command)
    
    # Import command
    memory_import_parser = memory_subparsers.add_parser(
        "import", 
        help="Import memories from file"
    )
    memory_import_parser.add_argument(
        "input_path", 
        help="Path to import memories from"
    )
    memory_import_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_import_parser.set_defaults(func=handle_import_command)
    
    # Decay command
    memory_decay_parser = memory_subparsers.add_parser(
        "decay", 
        help="Apply importance decay to memories"
    )
    memory_decay_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_decay_parser.set_defaults(func=handle_decay_command)
    
    # Cull command
    memory_cull_parser = memory_subparsers.add_parser(
        "cull", 
        help="Remove low-importance memories"
    )
    memory_cull_parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.2,
        help="Importance threshold (default: 0.2)"
    )
    memory_cull_parser.add_argument(
        "--max", 
        type=int, 
        help="Maximum number of memories to retain"
    )
    memory_cull_parser.add_argument(
        "--no-archive", 
        action="store_true",
        help="Don't archive culled memories"
    )
    memory_cull_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    memory_cull_parser.set_defaults(func=handle_cull_command)
    
    # PsiC command group
    psic_parser = subparsers.add_parser(
        "psic", 
        help="ΨC operator commands"
    )
    psic_subparsers = psic_parser.add_subparsers(dest="subcommand")
    
    # Status command
    psic_status_parser = psic_subparsers.add_parser(
        "status", 
        help="Get current ΨC consciousness status"
    )
    psic_status_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    psic_status_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    psic_status_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    psic_status_parser.set_defaults(func=handle_status_command)
    
    # Log command
    psic_log_parser = psic_subparsers.add_parser(
        "log", 
        help="View ΨC activation log"
    )
    psic_log_parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Maximum number of log entries to show (default: 10)"
    )
    psic_log_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    psic_log_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    psic_log_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    psic_log_parser.set_defaults(func=handle_log_command)
    
    # Health command
    psic_health_parser = psic_subparsers.add_parser(
        "health", 
        help="Get ΨC coherence health metrics"
    )
    psic_health_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    psic_health_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    psic_health_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    psic_health_parser.set_defaults(func=handle_health_command)
    
    # Collapse command
    psic_collapse_parser = psic_subparsers.add_parser(
        "collapse", 
        help="Simulate a quantum collapse event"
    )
    psic_collapse_parser.add_argument(
        "--outcomes", 
        type=int, 
        default=2,
        help="Number of possible outcomes (default: 2)"
    )
    psic_collapse_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    psic_collapse_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    psic_collapse_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    psic_collapse_parser.set_defaults(func=handle_collapse_command)
    
    # Configure command
    psic_configure_parser = psic_subparsers.add_parser(
        "configure", 
        help="Configure ΨC operator parameters"
    )
    psic_configure_parser.add_argument(
        "--threshold", 
        type=float,
        help="Consciousness threshold (θ) value"
    )
    psic_configure_parser.add_argument(
        "--hard-mode", 
        action="store_true",
        help="Use binary consciousness (0 or 1)"
    )
    psic_configure_parser.add_argument(
        "--window-size", 
        type=int,
        help="Window size for temporal integration"
    )
    psic_configure_parser.add_argument(
        "--reflection-weight", 
        type=float,
        help="Weight of reflection readiness vs. coherence"
    )
    psic_configure_parser.add_argument(
        "--integration-step", 
        type=float,
        help="Step size for temporal integration"
    )
    psic_configure_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    psic_configure_parser.set_defaults(func=handle_configure_command)
    
    # Add plugin commands
    register_plugin_commands(subparsers)
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments, defaults to sys.argv if None
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_main_parser()
    
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parser.parse_args(args)
    
    # Handle version request
    if parsed_args.version:
        from psi_c_ai_sdk import __version__
        print(f"ΨC-AI SDK version {__version__}")
        return 0
    
    # Handle no command
    if not parsed_args.command:
        parser.print_help()
        return 0
    
    # Handle schema commands
    if parsed_args.command == "schema":
        if not parsed_args.subcommand:
            # If no subcommand specified, show schema command help
            schema_parser = [p for p in parser._subparsers._actions 
                             if isinstance(p, argparse._SubParsersAction)][0]
            schema_parser_obj = schema_parser.choices["schema"]
            schema_parser_obj.print_help()
            return 0
        
        # Execute subcommand function if available
        if hasattr(parsed_args, 'func'):
            parsed_args.func(parsed_args)
            return 0
    
    # Handle memory commands
    elif parsed_args.command == "memory":
        if not parsed_args.subcommand:
            # If no subcommand specified, show memory command help
            memory_parser = [p for p in parser._subparsers._actions 
                            if isinstance(p, argparse._SubParsersAction)][0]
            memory_parser_obj = memory_parser.choices["memory"]
            memory_parser_obj.print_help()
            return 0
        
        # Execute subcommand function if available
        if hasattr(parsed_args, 'func'):
            parsed_args.func(parsed_args)
            return 0
    
    # Handle PsiC commands
    elif parsed_args.command == "psic":
        if not parsed_args.subcommand:
            # If no subcommand specified, show PsiC command help
            psic_parser = [p for p in parser._subparsers._actions 
                          if isinstance(p, argparse._SubParsersAction)][0]
            psic_parser_obj = psic_parser.choices["psic"]
            psic_parser_obj.print_help()
            return 0
        
        # Execute subcommand function if available
        if hasattr(parsed_args, 'func'):
            parsed_args.func(parsed_args)
            return 0
    
    # Handle plugin commands
    elif parsed_args.command == "plugin":
        if not parsed_args.subcommand:
            # If no subcommand specified, show plugin command help
            plugin_parser = [p for p in parser._subparsers._actions 
                          if isinstance(p, argparse._SubParsersAction)][0]
            plugin_parser_obj = plugin_parser.choices["plugin"]
            plugin_parser_obj.print_help()
            return 0
        
        # Execute subcommand function if available
        if hasattr(parsed_args, 'func'):
            parsed_args.func(parsed_args)
            return 0
    
    # Print help for other commands (placeholders)
    else:
        print(f"The '{parsed_args.command}' command will be implemented in a future update.")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 