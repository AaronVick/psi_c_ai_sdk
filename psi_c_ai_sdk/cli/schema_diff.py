"""
Schema Diff CLI Command

This module implements the CLI command for visualizing schema differences
and tracking drift over time using Merkle hashes.
"""

import os
import sys
import argparse
import json
import time
import tempfile
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import (
    SchemaGraph,
    SchemaFingerprint,
    SchemaDiffCalculator,
    SchemaDriftMonitor
)


def get_cache_dir() -> Path:
    """Get or create the ΨC-AI SDK cache directory."""
    home_dir = Path.home()
    cache_dir = home_dir / ".psi_c_ai_sdk" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_schema_history_file() -> Path:
    """Get the schema history file path."""
    cache_dir = get_cache_dir()
    return cache_dir / "schema_history.pkl"


def save_schema_snapshot(schema_graph: SchemaGraph, label: str = "auto") -> str:
    """
    Save a snapshot of the current schema graph.
    
    Args:
        schema_graph: The schema graph to snapshot
        label: Optional label for the snapshot
        
    Returns:
        ID of the snapshot
    """
    # Generate a unique ID for this snapshot
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    snapshot_id = f"{timestamp}_{label}"
    
    # Create a fingerprint for this schema
    fingerprinter = SchemaFingerprint(schema_graph)
    merkle_root = fingerprinter.compute_fingerprint()
    
    # Get schema statistics
    stats = schema_graph.get_stats()
    
    # Create snapshot data
    snapshot_data = {
        "id": snapshot_id,
        "label": label,
        "timestamp": time.time(),
        "merkle_root": merkle_root,
        "node_count": stats.get("node_count", 0),
        "edge_count": stats.get("edge_count", 0),
        "schema_graph": schema_graph
    }
    
    # Get history file path
    history_file = get_schema_history_file()
    
    # Load existing history
    snapshots = {}
    if history_file.exists():
        try:
            with open(history_file, "rb") as f:
                snapshots = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load schema history: {e}")
            snapshots = {}
    
    # Add new snapshot
    snapshots[snapshot_id] = snapshot_data
    
    # Save updated history
    try:
        with open(history_file, "wb") as f:
            pickle.dump(snapshots, f)
    except Exception as e:
        print(f"Error: Failed to save schema snapshot: {e}")
        return ""
    
    return snapshot_id


def list_schema_snapshots() -> List[Dict[str, Any]]:
    """
    List all available schema snapshots.
    
    Returns:
        List of snapshot metadata
    """
    # Get history file path
    history_file = get_schema_history_file()
    
    # Check if history file exists
    if not history_file.exists():
        return []
    
    # Load snapshots
    try:
        with open(history_file, "rb") as f:
            snapshots = pickle.load(f)
    except Exception as e:
        print(f"Error: Failed to load schema history: {e}")
        return []
    
    # Create list of snapshot metadata
    snapshot_list = []
    for snapshot_id, snapshot_data in snapshots.items():
        snapshot_list.append({
            "id": snapshot_id,
            "label": snapshot_data.get("label", ""),
            "timestamp": snapshot_data.get("timestamp", 0),
            "formatted_time": datetime.fromtimestamp(
                snapshot_data.get("timestamp", 0)
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "merkle_root": snapshot_data.get("merkle_root", ""),
            "node_count": snapshot_data.get("node_count", 0),
            "edge_count": snapshot_data.get("edge_count", 0)
        })
    
    # Sort by timestamp (newest first)
    snapshot_list.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    
    return snapshot_list


def load_schema_snapshot(snapshot_id: str) -> Optional[SchemaGraph]:
    """
    Load a schema snapshot by ID.
    
    Args:
        snapshot_id: ID of the snapshot to load
        
    Returns:
        The loaded schema graph, or None if not found
    """
    # Get history file path
    history_file = get_schema_history_file()
    
    # Check if history file exists
    if not history_file.exists():
        print(f"Error: Schema history file not found")
        return None
    
    # Load snapshots
    try:
        with open(history_file, "rb") as f:
            snapshots = pickle.load(f)
    except Exception as e:
        print(f"Error: Failed to load schema history: {e}")
        return None
    
    # Find snapshot by ID
    if snapshot_id in snapshots:
        return snapshots[snapshot_id].get("schema_graph")
    
    # Handle special snapshot IDs
    if snapshot_id == "latest":
        # Get the most recent snapshot
        timestamps = [(id, data.get("timestamp", 0)) for id, data in snapshots.items()]
        if not timestamps:
            print("Error: No snapshots available")
            return None
        latest_id = max(timestamps, key=lambda x: x[1])[0]
        return snapshots[latest_id].get("schema_graph")
    
    if snapshot_id == "previous":
        # Get the second most recent snapshot
        timestamps = [(id, data.get("timestamp", 0)) for id, data in snapshots.items()]
        if len(timestamps) < 2:
            print("Error: Not enough snapshots available")
            return None
        sorted_ids = [id for id, _ in sorted(timestamps, key=lambda x: x[1], reverse=True)]
        return snapshots[sorted_ids[1]].get("schema_graph")
    
    print(f"Error: Snapshot {snapshot_id} not found")
    return None


def load_current_schema() -> SchemaGraph:
    """
    Load the current schema from the active memory store.
    
    Returns:
        A schema graph of the current system state
    """
    # Create memory store and coherence scorer
    memory_store = MemoryStore()
    coherence_scorer = CoherenceScorer()
    
    # Create and update schema graph
    schema = SchemaGraph(
        memory_store=memory_store,
        coherence_scorer=coherence_scorer
    )
    schema.update_schema()
    
    return schema


def compare_schemas(
    current_schema: SchemaGraph,
    previous_schema: SchemaGraph,
    output_format: str = "text",
    output_file: Optional[str] = None
) -> None:
    """
    Compare two schema graphs and visualize the differences.
    
    Args:
        current_schema: Current schema graph
        previous_schema: Previous schema graph
        output_format: Output format (text, json, or png)
        output_file: Optional output file path
    """
    # Create diff calculator
    diff_calculator = SchemaDiffCalculator(current_schema, previous_schema)
    
    # Calculate diff
    diff_results = diff_calculator.calculate_diff()
    
    # Generate output based on format
    if output_format == "json":
        # Convert any non-serializable values
        serializable_diff = {}
        for key, value in diff_results.items():
            if key not in ["nodes", "edges", "clusters", "metrics"]:
                serializable_diff[key] = value
        
        # Include node and edge counts
        serializable_diff["nodes"] = {
            "added_count": diff_results["nodes"]["added_count"],
            "removed_count": diff_results["nodes"]["removed_count"],
            "modified_count": diff_results["nodes"]["modified_count"]
        }
        serializable_diff["edges"] = {
            "added_count": diff_results["edges"]["added_count"],
            "removed_count": diff_results["edges"]["removed_count"],
            "modified_count": diff_results["edges"]["modified_count"]
        }
        
        # Include metrics
        if "metrics" in diff_results:
            serializable_diff["metrics"] = diff_results["metrics"]
        
        # Output JSON
        if output_file:
            with open(output_file, "w") as f:
                json.dump(serializable_diff, f, indent=2)
            print(f"Diff results saved to {output_file}")
        else:
            print(json.dumps(serializable_diff, indent=2))
    
    elif output_format == "png":
        # Visualize diff as PNG
        if not output_file:
            output_file = "schema_diff.png"
        
        try:
            diff_calculator.visualize_diff(filename=output_file, show=False)
            print(f"Diff visualization saved to {output_file}")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    else:  # text format
        # Print text summary
        print("\nSchema Comparison Results:")
        print("=" * 60)
        print(f"Schemas Match: {diff_results['schemas_match']}")
        print(f"Current Schema Hash: {diff_results['current_schema_hash'][:10]}...")
        print(f"Previous Schema Hash: {diff_results['previous_schema_hash'][:10]}...")
        print("\nChange Summary:")
        print(f"- Added Nodes: {diff_results['nodes']['added_count']}")
        print(f"- Removed Nodes: {diff_results['nodes']['removed_count']}")
        print(f"- Modified Nodes: {diff_results['nodes']['modified_count']}")
        print(f"- Added Edges: {diff_results['edges']['added_count']}")
        print(f"- Removed Edges: {diff_results['edges']['removed_count']}")
        print(f"- Modified Edges: {diff_results['edges']['modified_count']}")
        
        # Output to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write("Schema Comparison Results:\n")
                f.write("=" * 60 + "\n")
                f.write(f"Schemas Match: {diff_results['schemas_match']}\n")
                f.write(f"Current Schema Hash: {diff_results['current_schema_hash'][:10]}...\n")
                f.write(f"Previous Schema Hash: {diff_results['previous_schema_hash'][:10]}...\n")
                f.write("\nChange Summary:\n")
                f.write(f"- Added Nodes: {diff_results['nodes']['added_count']}\n")
                f.write(f"- Removed Nodes: {diff_results['nodes']['removed_count']}\n")
                f.write(f"- Modified Nodes: {diff_results['nodes']['modified_count']}\n")
                f.write(f"- Added Edges: {diff_results['edges']['added_count']}\n")
                f.write(f"- Removed Edges: {diff_results['edges']['removed_count']}\n")
                f.write(f"- Modified Edges: {diff_results['edges']['modified_count']}\n")
            print(f"Diff results saved to {output_file}")


def handle_diff_command(args: argparse.Namespace) -> None:
    """
    Handle the diff command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load the schema graphs to compare
    current_schema = None
    previous_schema = None
    
    # Load current schema
    if args.now:
        # Load current schema from active memory store
        current_schema = load_current_schema()
        
        # Take snapshot if enabled
        if args.snapshot:
            snapshot_id = save_schema_snapshot(current_schema, args.snapshot_label or "auto")
            print(f"Saved current schema snapshot with ID: {snapshot_id}")
    else:
        # Load from snapshot ID
        current_schema = load_schema_snapshot(args.current or "latest")
        if not current_schema:
            print("Error: Failed to load current schema")
            return
    
    # Load previous schema
    if args.previous:
        # Load previous schema snapshot
        previous_schema = load_schema_snapshot("previous")
        if not previous_schema:
            print("Error: Failed to load previous schema")
            return
    else:
        # Load from snapshot ID
        previous_schema = load_schema_snapshot(args.compare)
        if not previous_schema:
            print("Error: Failed to load comparison schema")
            return
    
    # Compare schemas
    compare_schemas(
        current_schema,
        previous_schema,
        args.format,
        args.output
    )


def handle_list_command(args: argparse.Namespace) -> None:
    """
    Handle the list command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    snapshots = list_schema_snapshots()
    
    if not snapshots:
        print("No schema snapshots available.")
        return
    
    if args.format == "json":
        if args.output:
            with open(args.output, "w") as f:
                json.dump(snapshots, f, indent=2)
            print(f"Snapshot list saved to {args.output}")
        else:
            print(json.dumps(snapshots, indent=2))
    else:
        # Print table of snapshots
        print("\nAvailable Schema Snapshots:")
        print("=" * 80)
        print(f"{'ID':<20} {'Label':<15} {'Time':<20} {'Nodes':<8} {'Edges':<8} {'Hash':<12}")
        print("-" * 80)
        
        for snapshot in snapshots:
            print(f"{snapshot['id'][:18]:<20} {snapshot['label'][:15]:<15} "
                 f"{snapshot['formatted_time']:<20} {snapshot['node_count']:<8} "
                 f"{snapshot['edge_count']:<8} {snapshot['merkle_root'][:10]}...")
        
        print("-" * 80)
        print(f"Total snapshots: {len(snapshots)}")
        
        # Output to file if specified
        if args.output:
            with open(args.output, "w") as f:
                f.write("\nAvailable Schema Snapshots:\n")
                f.write("=" * 80 + "\n")
                f.write(f"{'ID':<20} {'Label':<15} {'Time':<20} {'Nodes':<8} {'Edges':<8} {'Hash':<12}\n")
                f.write("-" * 80 + "\n")
                
                for snapshot in snapshots:
                    f.write(f"{snapshot['id'][:18]:<20} {snapshot['label'][:15]:<15} "
                           f"{snapshot['formatted_time']:<20} {snapshot['node_count']:<8} "
                           f"{snapshot['edge_count']:<8} {snapshot['merkle_root'][:10]}...\n")
                
                f.write("-" * 80 + "\n")
                f.write(f"Total snapshots: {len(snapshots)}\n")
            
            print(f"Snapshot list saved to {args.output}")


def handle_snapshot_command(args: argparse.Namespace) -> None:
    """
    Handle the snapshot command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load current schema
    current_schema = load_current_schema()
    
    # Save snapshot
    snapshot_id = save_schema_snapshot(current_schema, args.label or "manual")
    
    if snapshot_id:
        print(f"Saved schema snapshot with ID: {snapshot_id}")
    else:
        print("Error: Failed to save schema snapshot")


def handle_drift_command(args: argparse.Namespace) -> None:
    """
    Handle the drift command with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Load current schema
    current_schema = load_current_schema()
    
    # Create drift monitor
    drift_monitor = SchemaDriftMonitor(
        schema_graph=current_schema,
        drift_threshold=args.threshold
    )
    
    # Load historical snapshots to analyze drift
    snapshots = list_schema_snapshots()
    
    if not snapshots or len(snapshots) < 2:
        print("Error: Not enough schema snapshots available for drift analysis")
        return
    
    # Sort snapshots by timestamp
    snapshots.sort(key=lambda x: x.get("timestamp", 0))
    
    # Take current snapshot
    current_snapshot = drift_monitor.take_snapshot()
    
    # Print drift information
    print("\nSchema Drift Analysis:")
    print("=" * 60)
    print(f"Current schema nodes: {current_snapshot['node_count']}")
    print(f"Current schema edges: {current_snapshot['edge_count']}")
    print(f"Current schema hash: {current_snapshot['merkle_root'][:10]}...")
    
    if len(snapshots) >= args.history:
        history_snapshots = snapshots[-args.history:]
        print(f"\nAnalyzing drift over {len(history_snapshots)} historical snapshots:")
        
        # Load each schema and calculate drift
        drift_scores = []
        
        for i, snapshot_meta in enumerate(history_snapshots):
            # Load schema
            schema = load_schema_snapshot(snapshot_meta["id"])
            if not schema:
                continue
                
            # Calculate fingerprint
            fingerprinter = SchemaFingerprint(schema)
            merkle_root = fingerprinter.compute_fingerprint()
            
            # Calculate drift if not the first snapshot
            if i > 0:
                # Create diff calculator
                prev_schema = load_schema_snapshot(history_snapshots[i-1]["id"])
                if prev_schema:
                    diff_calculator = SchemaDiffCalculator(schema, prev_schema)
                    diff = diff_calculator.calculate_diff()
                    
                    # Get drift components
                    node_change = diff["metrics"]["node_count"]["percent_change"] / 100
                    edge_change = diff["metrics"]["edge_count"]["percent_change"] / 100
                    
                    # Calculate drift score
                    drift_score = (node_change * 0.5) + (edge_change * 0.5)
                    drift_score = min(1.0, max(0.0, drift_score))
                    
                    drift_scores.append({
                        "from": history_snapshots[i-1]["formatted_time"],
                        "to": snapshot_meta["formatted_time"],
                        "score": drift_score,
                        "significant": drift_score > args.threshold
                    })
        
        # Output drift scores
        if drift_scores:
            if args.format == "json":
                result = {
                    "current_schema": {
                        "node_count": current_snapshot["node_count"],
                        "edge_count": current_snapshot["edge_count"],
                        "merkle_root": current_snapshot["merkle_root"]
                    },
                    "drift_scores": drift_scores,
                    "threshold": args.threshold
                }
                
                if args.output:
                    with open(args.output, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"Drift analysis saved to {args.output}")
                else:
                    print(json.dumps(result, indent=2))
            else:
                print("\nDrift Score Timeline:")
                print("-" * 60)
                print(f"{'From':<20} {'To':<20} {'Score':<10} {'Significant':<12}")
                print("-" * 60)
                
                for drift in drift_scores:
                    print(f"{drift['from']:<20} {drift['to']:<20} "
                         f"{drift['score']:.4f}    {'Yes' if drift['significant'] else 'No'}")
                
                # Calculate average drift
                avg_drift = sum(d["score"] for d in drift_scores) / len(drift_scores)
                print("-" * 60)
                print(f"Average drift: {avg_drift:.4f}")
                print(f"Drift threshold: {args.threshold}")
                
                # Output to file if specified
                if args.output:
                    with open(args.output, "w") as f:
                        f.write("\nDrift Score Timeline:\n")
                        f.write("-" * 60 + "\n")
                        f.write(f"{'From':<20} {'To':<20} {'Score':<10} {'Significant':<12}\n")
                        f.write("-" * 60 + "\n")
                        
                        for drift in drift_scores:
                            f.write(f"{drift['from']:<20} {drift['to']:<20} "
                                  f"{drift['score']:.4f}    {'Yes' if drift['significant'] else 'No'}\n")
                        
                        f.write("-" * 60 + "\n")
                        f.write(f"Average drift: {avg_drift:.4f}\n")
                        f.write(f"Drift threshold: {args.threshold}\n")
                    
                    print(f"Drift analysis saved to {args.output}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    # Create main parser
    parser = argparse.ArgumentParser(
        description="ΨC-AI Schema Diff Tool",
        epilog="Use 'psi schema diff --help' for more information on a specific command."
    )
    
    # Create subcommand parsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare schema versions")
    diff_parser.add_argument("--now", action="store_true", help="Use current live schema")
    diff_parser.add_argument("--previous", action="store_true", help="Compare with previous snapshot")
    diff_parser.add_argument("--current", help="Specify current schema snapshot ID")
    diff_parser.add_argument("--compare", help="Specify comparison schema snapshot ID")
    diff_parser.add_argument("--format", choices=["text", "json", "png"], default="text", 
                           help="Output format (default: text)")
    diff_parser.add_argument("--output", help="Output file path")
    diff_parser.add_argument("--snapshot", action="store_true", help="Save current schema as a snapshot")
    diff_parser.add_argument("--snapshot-label", help="Label for the snapshot")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available schema snapshots")
    list_parser.add_argument("--format", choices=["text", "json"], default="text", 
                           help="Output format (default: text)")
    list_parser.add_argument("--output", help="Output file path")
    
    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Save current schema snapshot")
    snapshot_parser.add_argument("--label", help="Label for the snapshot")
    
    # Drift command
    drift_parser = subparsers.add_parser("drift", help="Analyze schema drift over time")
    drift_parser.add_argument("--threshold", type=float, default=0.1, 
                            help="Drift significance threshold (default: 0.1)")
    drift_parser.add_argument("--history", type=int, default=5, 
                            help="Number of historical snapshots to analyze (default: 5)")
    drift_parser.add_argument("--format", choices=["text", "json"], default="text", 
                            help="Output format (default: text)")
    drift_parser.add_argument("--output", help="Output file path")
    
    return parser


def main():
    """Main entry point for the CLI tool."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Handle commands
    if args.command == "diff":
        handle_diff_command(args)
    elif args.command == "list":
        handle_list_command(args)
    elif args.command == "snapshot":
        handle_snapshot_command(args)
    elif args.command == "drift":
        handle_drift_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 