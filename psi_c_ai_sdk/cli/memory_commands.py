"""
Memory Commands CLI Module

This module implements CLI commands for managing memories in the ΨC-AI SDK,
including adding, listing, exporting, and manipulating memory objects.
"""

import os
import sys
import argparse
import json
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from psi_c_ai_sdk.memory import Memory, MemoryStore


def create_memory_command_parser() -> argparse.ArgumentParser:
    """Create the argument parser for memory commands."""
    parser = argparse.ArgumentParser(
        description="ΨC-AI SDK Memory Management Tools",
        epilog="Use 'psi memory <command> --help' for more information on a specific command."
    )
    
    # Create subcommand parsers
    subparsers = parser.add_subparsers(dest="subcommand", help="Memory command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List memories in the store")
    list_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    list_parser.add_argument(
        "--limit", 
        type=int, 
        default=20,
        help="Maximum number of memories to list (default: 20)"
    )
    list_parser.add_argument(
        "--sort-by", 
        choices=["importance", "created", "accessed"], 
        default="importance",
        help="Sort criteria (default: importance)"
    )
    list_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    list_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new memory")
    add_parser.add_argument(
        "content", 
        help="Memory content text"
    )
    add_parser.add_argument(
        "--importance", 
        type=float, 
        default=1.0,
        help="Initial importance score (default: 1.0)"
    )
    add_parser.add_argument(
        "--tags", 
        nargs="+", 
        help="Tags for the memory"
    )
    add_parser.add_argument(
        "--pin", 
        action="store_true",
        help="Pin the memory"
    )
    add_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Retrieve a specific memory")
    get_parser.add_argument(
        "uuid", 
        help="UUID of the memory to retrieve"
    )
    get_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    get_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    get_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory")
    delete_parser.add_argument(
        "uuid", 
        help="UUID of the memory to delete"
    )
    delete_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a memory")
    archive_parser.add_argument(
        "uuid", 
        nargs="+",
        help="UUID(s) of the memory/memories to archive"
    )
    archive_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a memory from archive")
    restore_parser.add_argument(
        "uuid", 
        help="UUID of the memory to restore"
    )
    restore_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Pin/unpin commands
    pin_parser = subparsers.add_parser("pin", help="Pin a memory")
    pin_parser.add_argument(
        "uuid", 
        help="UUID of the memory to pin"
    )
    pin_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    unpin_parser = subparsers.add_parser("unpin", help="Unpin a memory")
    unpin_parser.add_argument(
        "uuid", 
        help="UUID of the memory to unpin"
    )
    unpin_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export memories to file")
    export_parser.add_argument(
        "output_path", 
        help="Path to export memories to"
    )
    export_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import memories from file")
    import_parser.add_argument(
        "input_path", 
        help="Path to import memories from"
    )
    import_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Decay command
    decay_parser = subparsers.add_parser("decay", help="Apply importance decay to memories")
    decay_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Cull command
    cull_parser = subparsers.add_parser("cull", help="Remove low-importance memories")
    cull_parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.2,
        help="Importance threshold (default: 0.2)"
    )
    cull_parser.add_argument(
        "--max", 
        type=int, 
        help="Maximum number of memories to retain"
    )
    cull_parser.add_argument(
        "--no-archive", 
        action="store_true",
        help="Don't archive culled memories"
    )
    cull_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    return parser


def get_store_path() -> str:
    """Get the default memory store path."""
    home_dir = Path.home()
    store_dir = home_dir / ".psi_c_ai_sdk" / "data"
    store_dir.mkdir(parents=True, exist_ok=True)
    return str(store_dir / "memory_store.json")


def load_memory_store(path: Optional[str] = None) -> MemoryStore:
    """
    Load a memory store from file or create a new one.
    
    Args:
        path: Path to the memory store file (optional)
        
    Returns:
        Loaded MemoryStore instance
    """
    memory_store = MemoryStore()
    
    # Use default path if none provided
    store_path = path or get_store_path()
    
    # Try to load if file exists
    if os.path.exists(store_path):
        try:
            memory_store.load(store_path)
            print(f"Loaded memory store from {store_path}")
        except Exception as e:
            print(f"Error loading memory store: {e}")
            print("Creating a new memory store.")
    else:
        print(f"No existing memory store found. Creating a new one.")
    
    return memory_store


def save_memory_store(memory_store: MemoryStore, path: Optional[str] = None) -> None:
    """
    Save a memory store to file.
    
    Args:
        memory_store: MemoryStore to save
        path: Path to save to (optional)
    """
    store_path = path or get_store_path()
    
    try:
        memory_store.export(store_path)
        print(f"Saved memory store to {store_path}")
    except Exception as e:
        print(f"Error saving memory store: {e}")


def format_memory(memory: Memory, format_type: str = "text") -> str:
    """
    Format a memory for display.
    
    Args:
        memory: Memory to format
        format_type: Output format (text or json)
        
    Returns:
        Formatted memory string
    """
    if format_type == "json":
        return json.dumps(memory.to_dict(), indent=2)
    
    # Text format
    output = []
    output.append(f"UUID: {memory.uuid}")
    output.append(f"Content: {memory.content}")
    output.append(f"Importance: {memory.importance:.4f}")
    output.append(f"Created: {datetime.fromtimestamp(memory.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Last Accessed: {datetime.fromtimestamp(memory.last_accessed).strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Tags: {', '.join(memory.tags) if memory.tags else 'None'}")
    output.append(f"Pinned: {'Yes' if memory.is_pinned else 'No'}")
    
    return "\n".join(output)


def handle_list_command(args: argparse.Namespace) -> None:
    """
    Handle the list command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Get all memories
    memories = memory_store.get_all_memories()
    
    # Sort memories
    if args.sort_by == "importance":
        memories.sort(key=lambda x: x.importance, reverse=True)
    elif args.sort_by == "created":
        memories.sort(key=lambda x: x.created_at, reverse=True)
    elif args.sort_by == "accessed":
        memories.sort(key=lambda x: x.last_accessed, reverse=True)
    
    # Apply limit
    if args.limit > 0:
        memories = memories[:args.limit]
    
    # Format output
    if args.format == "json":
        output = json.dumps([memory.to_dict() for memory in memories], indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Memory list saved to {args.output}")
        else:
            print(output)
    else:
        # Text format
        output = [
            "\nMemories:",
            "=" * 60
        ]
        
        for memory in memories:
            output.append(format_memory(memory))
            output.append("-" * 60)
        
        output.append(f"Total: {len(memories)} memories")
        
        if args.output:
            with open(args.output, "w") as f:
                f.write("\n".join(output))
            print(f"Memory list saved to {args.output}")
        else:
            print("\n".join(output))


def handle_add_command(args: argparse.Namespace) -> None:
    """
    Handle the add command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Add memory
    uuid = memory_store.add(
        content=args.content,
        importance=args.importance,
        tags=args.tags,
        is_pinned=args.pin
    )
    
    # Save store
    save_memory_store(memory_store, args.store_path)
    
    print(f"Added memory with UUID: {uuid}")
    if args.pin:
        print("Memory is pinned.")


def handle_get_command(args: argparse.Namespace) -> None:
    """
    Handle the get command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Get memory
    memory = memory_store.get_memory(args.uuid)
    
    if not memory:
        print(f"Memory with UUID {args.uuid} not found.")
        return
    
    # Format and output
    formatted_memory = format_memory(memory, args.format)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(formatted_memory)
        print(f"Memory details saved to {args.output}")
    else:
        print(formatted_memory)


def handle_delete_command(args: argparse.Namespace) -> None:
    """
    Handle the delete command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Delete memory
    success = memory_store.delete_memory(args.uuid)
    
    if success:
        # Save store
        save_memory_store(memory_store, args.store_path)
        print(f"Memory {args.uuid} deleted.")
    else:
        print(f"Memory with UUID {args.uuid} not found.")


def handle_archive_command(args: argparse.Namespace) -> None:
    """
    Handle the archive command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Check if single UUID or multiple
    if len(args.uuid) == 1:
        # Archive single memory
        success = memory_store.archive_memory(args.uuid[0])
        
        if success:
            # Save store
            save_memory_store(memory_store, args.store_path)
            print(f"Memory {args.uuid[0]} archived.")
        else:
            print(f"Memory with UUID {args.uuid[0]} not found.")
    else:
        # Archive multiple memories
        num_archived = memory_store.archive_memories(args.uuid)
        
        # Save store
        save_memory_store(memory_store, args.store_path)
        print(f"{num_archived} of {len(args.uuid)} memories archived.")


def handle_restore_command(args: argparse.Namespace) -> None:
    """
    Handle the restore command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Restore memory
    success = memory_store.restore_from_archive(args.uuid)
    
    if success:
        # Save store
        save_memory_store(memory_store, args.store_path)
        print(f"Memory {args.uuid} restored from archive.")
    else:
        print(f"Memory with UUID {args.uuid} not found in archive.")


def handle_pin_command(args: argparse.Namespace) -> None:
    """
    Handle the pin command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Pin memory
    success = memory_store.pin_memory(args.uuid)
    
    if success:
        # Save store
        save_memory_store(memory_store, args.store_path)
        print(f"Memory {args.uuid} pinned.")
    else:
        print(f"Memory with UUID {args.uuid} not found.")


def handle_unpin_command(args: argparse.Namespace) -> None:
    """
    Handle the unpin command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Unpin memory
    success = memory_store.unpin_memory(args.uuid)
    
    if success:
        # Save store
        save_memory_store(memory_store, args.store_path)
        print(f"Memory {args.uuid} unpinned.")
    else:
        print(f"Memory with UUID {args.uuid} not found.")


def handle_export_command(args: argparse.Namespace) -> None:
    """
    Handle the export command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Export memories
    try:
        memory_store.export(args.output_path)
        print(f"Exported memories to {args.output_path}")
    except Exception as e:
        print(f"Error exporting memories: {e}")


def handle_import_command(args: argparse.Namespace) -> None:
    """
    Handle the import command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Import memories
    try:
        memory_store.load(args.input_path)
        
        # Save to default location if different from input
        if args.input_path != args.store_path:
            save_memory_store(memory_store, args.store_path)
            
        print(f"Imported memories from {args.input_path}")
        print(f"Total memories: {len(memory_store.get_all_memories())}")
    except Exception as e:
        print(f"Error importing memories: {e}")


def handle_decay_command(args: argparse.Namespace) -> None:
    """
    Handle the decay command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Apply decay
    memory_store.apply_importance_decay()
    
    # Save store
    save_memory_store(memory_store, args.store_path)
    
    print("Applied importance decay to all memories.")


def handle_cull_command(args: argparse.Namespace) -> None:
    """
    Handle the cull command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store
    memory_store = load_memory_store(args.store_path)
    
    # Cull memories
    removed, archived = memory_store.cull_memories(
        importance_threshold=args.threshold,
        max_memories=args.max,
        archive=not args.no_archive
    )
    
    # Save store
    save_memory_store(memory_store, args.store_path)
    
    print(f"Culled {removed} memories.")
    if not args.no_archive:
        print(f"Archived {archived} memories.")
    
    print(f"Remaining memories: {len(memory_store.get_all_memories())}")


def memory_cli_main(args: argparse.Namespace) -> int:
    """
    Main entry point for memory CLI.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Create parser
    parser = create_memory_command_parser()
    
    # Process subcommand
    if not args.subcommand:
        # If no subcommand specified, show help
        parser.print_help()
        return 0
    
    # Handle commands
    if args.subcommand == "list":
        handle_list_command(args)
    elif args.subcommand == "add":
        handle_add_command(args)
    elif args.subcommand == "get":
        handle_get_command(args)
    elif args.subcommand == "delete":
        handle_delete_command(args)
    elif args.subcommand == "archive":
        handle_archive_command(args)
    elif args.subcommand == "restore":
        handle_restore_command(args)
    elif args.subcommand == "pin":
        handle_pin_command(args)
    elif args.subcommand == "unpin":
        handle_unpin_command(args)
    elif args.subcommand == "export":
        handle_export_command(args)
    elif args.subcommand == "import":
        handle_import_command(args)
    elif args.subcommand == "decay":
        handle_decay_command(args)
    elif args.subcommand == "cull":
        handle_cull_command(args)
    else:
        parser.print_help()
        return 1
    
    return 0 