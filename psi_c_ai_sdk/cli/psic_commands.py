"""
PsiC Commands CLI Module

This module implements CLI commands for interacting with the ΨC operator and tools,
including consciousness state monitoring, activation logging, and quantum collapse simulation.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.psi_c import (
    PsiCOperator, 
    PsiCState, 
    PsiToolkit, 
    TemporalCoherenceAccumulator,
    StabilityFilter,
    CollapseSimulator
)


def create_psic_command_parser() -> argparse.ArgumentParser:
    """Create the argument parser for PsiC commands."""
    parser = argparse.ArgumentParser(
        description="ΨC-AI SDK PsiC Operator Tools",
        epilog="Use 'psi psic <command> --help' for more information on a specific command."
    )
    
    # Create subcommand parsers
    subparsers = parser.add_subparsers(dest="subcommand", help="PsiC command to execute")
    
    # Status command
    status_parser = subparsers.add_parser(
        "status", 
        help="Get current ΨC consciousness status"
    )
    status_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    status_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    status_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Log command
    log_parser = subparsers.add_parser(
        "log", 
        help="View ΨC activation log"
    )
    log_parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Maximum number of log entries to show (default: 10)"
    )
    log_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    log_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    log_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Health command
    health_parser = subparsers.add_parser(
        "health", 
        help="Get ΨC coherence health metrics"
    )
    health_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    health_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    health_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Collapse command
    collapse_parser = subparsers.add_parser(
        "collapse", 
        help="Simulate a quantum collapse event"
    )
    collapse_parser.add_argument(
        "--outcomes", 
        type=int, 
        default=2,
        help="Number of possible outcomes (default: 2)"
    )
    collapse_parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    collapse_parser.add_argument(
        "--output", 
        help="Output file path"
    )
    collapse_parser.add_argument(
        "--store-path", 
        help="Path to memory store file (optional)"
    )
    
    # Configure command
    configure_parser = subparsers.add_parser(
        "configure", 
        help="Configure ΨC operator parameters"
    )
    configure_parser.add_argument(
        "--threshold", 
        type=float,
        help="Consciousness threshold (θ) value"
    )
    configure_parser.add_argument(
        "--hard-mode", 
        action="store_true",
        help="Use binary consciousness (0 or 1)"
    )
    configure_parser.add_argument(
        "--window-size", 
        type=int,
        help="Window size for temporal integration"
    )
    configure_parser.add_argument(
        "--reflection-weight", 
        type=float,
        help="Weight of reflection readiness vs. coherence"
    )
    configure_parser.add_argument(
        "--integration-step", 
        type=float,
        help="Step size for temporal integration"
    )
    configure_parser.add_argument(
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


def get_config_path() -> str:
    """Get the default config file path."""
    home_dir = Path.home()
    config_dir = home_dir / ".psi_c_ai_sdk" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return str(config_dir / "psic_config.json")


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


def load_config() -> Dict[str, Any]:
    """
    Load PsiC configuration from file or create default config.
    
    Returns:
        Configuration dictionary
    """
    config_path = get_config_path()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Default config
    default_config = {
        "threshold": 0.7,
        "hard_mode": False,
        "window_size": 10,
        "reflection_weight": 0.5,
        "integration_step": 0.1,
        "collapse_deviation": 0.2
    }
    
    # Save default config
    try:
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save default config: {e}")
    
    return default_config


def save_config(config: Dict[str, Any]) -> None:
    """
    Save PsiC configuration to file.
    
    Args:
        config: Configuration dictionary
    """
    config_path = get_config_path()
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


def setup_psic_toolkit(memory_store: MemoryStore, config: Optional[Dict[str, Any]] = None) -> PsiToolkit:
    """
    Set up PsiC toolkit with the given memory store and configuration.
    
    Args:
        memory_store: Memory store to use
        config: Optional configuration dictionary
        
    Returns:
        Configured PsiToolkit instance
    """
    if config is None:
        config = load_config()
    
    # Create TemporalCoherenceAccumulator
    temporal_coherence = TemporalCoherenceAccumulator(
        window_size=config.get("window_size", 10)
    )
    
    # Create StabilityFilter
    stability_filter = StabilityFilter(
        min_samples=5,
        threshold=0.05
    )
    
    # Create PsiCOperator
    psi_operator = PsiCOperator(
        memory_store=memory_store,
        threshold=config.get("threshold", 0.7),
        hard_mode=config.get("hard_mode", False),
        window_size=config.get("window_size", 10),
        reflection_weight=config.get("reflection_weight", 0.5),
        integration_step=config.get("integration_step", 0.1)
    )
    
    # Attach components
    psi_operator.temporal_coherence = temporal_coherence
    psi_operator.stability_filter = stability_filter
    
    # Create CollapseSimulator
    collapse_simulator = CollapseSimulator(
        psi_operator=psi_operator,
        deviation_strength=config.get("collapse_deviation", 0.2)
    )
    
    # Create PsiToolkit
    toolkit = PsiToolkit(
        psi_operator=psi_operator,
        collapse_simulator=collapse_simulator
    )
    
    return toolkit


def handle_status_command(args: argparse.Namespace) -> None:
    """
    Handle the status command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store and setup toolkit
    memory_store = load_memory_store(args.store_path)
    config = load_config()
    toolkit = setup_psic_toolkit(memory_store, config)
    
    # Get consciousness status
    psi_index = toolkit.get_psi_index()
    psi_state = toolkit.get_psi_state()
    is_conscious = toolkit.is_conscious()
    
    # Format output
    if args.format == "json":
        output = json.dumps({
            "psi_index": psi_index,
            "psi_state": psi_state.value,
            "is_conscious": is_conscious,
            "timestamp": time.time()
        }, indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Status saved to {args.output}")
        else:
            print(output)
    else:
        # Text format
        output = []
        output.append("\nΨC Consciousness Status:")
        output.append("=" * 60)
        output.append(f"ΨC Index: {psi_index:.4f}")
        output.append(f"ΨC State: {psi_state.value.upper()}")
        output.append(f"Conscious: {'Yes' if is_conscious else 'No'}")
        output.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("-" * 60)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write("\n".join(output))
            print(f"Status saved to {args.output}")
        else:
            print("\n".join(output))


def handle_log_command(args: argparse.Namespace) -> None:
    """
    Handle the log command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store and setup toolkit
    memory_store = load_memory_store(args.store_path)
    config = load_config()
    toolkit = setup_psic_toolkit(memory_store, config)
    
    # Get activation log
    log_entries = toolkit.get_activation_log(args.limit)
    
    # Format output
    if args.format == "json":
        output = json.dumps(log_entries, indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Log saved to {args.output}")
        else:
            print(output)
    else:
        # Text format
        output = []
        output.append("\nΨC Activation Log:")
        output.append("=" * 60)
        
        if not log_entries:
            output.append("No log entries available.")
        else:
            for entry in log_entries:
                timestamp = datetime.fromtimestamp(entry.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')
                output.append(f"Time: {timestamp}")
                
                if entry.get("event_type") == "state_change":
                    old_state = entry.get("old_state", PsiCState.INACTIVE).value.upper()
                    new_state = entry.get("new_state", PsiCState.INACTIVE).value.upper()
                    score = entry.get("psi_score", 0.0)
                    output.append(f"Event: State change from {old_state} to {new_state}")
                    output.append(f"ΨC Index: {score:.4f}")
                else:
                    output.append(f"Event: {entry.get('event_type', 'unknown')}")
                
                output.append("-" * 60)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write("\n".join(output))
            print(f"Log saved to {args.output}")
        else:
            print("\n".join(output))


def handle_health_command(args: argparse.Namespace) -> None:
    """
    Handle the health command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store and setup toolkit
    memory_store = load_memory_store(args.store_path)
    config = load_config()
    toolkit = setup_psic_toolkit(memory_store, config)
    
    # Get coherence health
    health = toolkit.get_coherence_health()
    
    # Format output
    if args.format == "json":
        output = json.dumps(health, indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Health metrics saved to {args.output}")
        else:
            print(output)
    else:
        # Text format
        output = []
        output.append("\nΨC Coherence Health:")
        output.append("=" * 60)
        output.append(f"Coherence: {health.get('coherence', 0.0):.4f}")
        output.append(f"Trend: {health.get('trend', 0.0):.4f}")
        output.append(f"Stability Score: {health.get('stability', {}).get('score', 0.0):.4f}")
        output.append(f"Stability Classification: {health.get('stability', {}).get('classification', 'unknown').upper()}")
        output.append(f"ΨC Index: {health.get('psi_index', 0.0):.4f}")
        output.append(f"ΨC State: {health.get('state', 'inactive').upper()}")
        output.append(f"Timestamp: {datetime.fromtimestamp(health.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("-" * 60)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write("\n".join(output))
            print(f"Health metrics saved to {args.output}")
        else:
            print("\n".join(output))


def handle_collapse_command(args: argparse.Namespace) -> None:
    """
    Handle the collapse command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load memory store and setup toolkit
    memory_store = load_memory_store(args.store_path)
    config = load_config()
    toolkit = setup_psic_toolkit(memory_store, config)
    
    try:
        # Simulate collapse event
        event = toolkit.simulate_collapse_event(num_outcomes=args.outcomes)
        
        # Format output
        if args.format == "json":
            output = json.dumps({
                "outcome": event.outcome,
                "deviation": event.deviation,
                "num_outcomes": args.outcomes,
                "psi_index": toolkit.get_psi_index(),
                "psi_state": toolkit.get_psi_state().value,
                "timestamp": time.time(),
                "metadata": event.metadata
            }, indent=2)
            
            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Collapse event saved to {args.output}")
            else:
                print(output)
        else:
            # Text format
            output = []
            output.append("\nQuantum Collapse Event Simulation:")
            output.append("=" * 60)
            output.append(f"Outcome: {event.outcome}")
            output.append(f"Deviation: {event.deviation:.6f}")
            output.append(f"Number of Possible Outcomes: {args.outcomes}")
            output.append(f"ΨC Index at Event: {toolkit.get_psi_index():.4f}")
            output.append(f"ΨC State: {toolkit.get_psi_state().value.upper()}")
            output.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            output.append("-" * 60)
            
            if args.output:
                with open(args.output, "w") as f:
                    f.write("\n".join(output))
                print(f"Collapse event saved to {args.output}")
            else:
                print("\n".join(output))
    
    except ValueError as e:
        print(f"Error: {e}")


def handle_configure_command(args: argparse.Namespace) -> None:
    """
    Handle the configure command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load current config
    config = load_config()
    
    # Update config with provided arguments
    if args.threshold is not None:
        config["threshold"] = args.threshold
    
    if args.hard_mode:
        config["hard_mode"] = True
    
    if args.window_size is not None:
        config["window_size"] = args.window_size
    
    if args.reflection_weight is not None:
        config["reflection_weight"] = args.reflection_weight
    
    if args.integration_step is not None:
        config["integration_step"] = args.integration_step
    
    # Save updated config
    save_config(config)
    
    # Show current configuration
    print("\nΨC Configuration:")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 60)


def psic_cli_main(args: argparse.Namespace) -> int:
    """
    Main entry point for PsiC CLI.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Create parser
    parser = create_psic_command_parser()
    
    # Process subcommand
    if not args.subcommand:
        # If no subcommand specified, show help
        parser.print_help()
        return 0
    
    # Handle commands
    if args.subcommand == "status":
        handle_status_command(args)
    elif args.subcommand == "log":
        handle_log_command(args)
    elif args.subcommand == "health":
        handle_health_command(args)
    elif args.subcommand == "collapse":
        handle_collapse_command(args)
    elif args.subcommand == "configure":
        handle_configure_command(args)
    else:
        parser.print_help()
        return 1
    
    return 0 