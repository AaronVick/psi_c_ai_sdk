#!/usr/bin/env python3
"""
Memory Sandbox Command Line Tool

A standalone CLI for the MemorySandbox from the ΨC-AI SDK Development Environment.
This tool allows for testing and experimentation with agent memory structures.

Usage:
  python memory_sandbox_cli.py [command] [options]

Commands:
  simulate    - Run a memory simulation
  create      - Create synthetic memories
  visualize   - Visualize memory patterns
  stats       - Show memory statistics

Examples:
  python memory_sandbox_cli.py simulate --days 30 --rate 5
  python memory_sandbox_cli.py create --count 10 --type EPISODIC
  python memory_sandbox_cli.py visualize --recent 20
  python memory_sandbox_cli.py stats
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import the MemorySandbox
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tools.dev_environment.memory_sandbox import MemorySandbox
    from psi_c_ai_sdk.memory.memory_store import MemoryStore
    from psi_c_ai_sdk.memory.memory_types import Memory, MemoryType
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure the ΨC-AI SDK is properly installed")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Memory Sandbox CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate memory dynamics")
    simulate_parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    simulate_parser.add_argument("--rate", type=float, default=5, help="Memory creation rate per day")
    simulate_parser.add_argument("--decay", type=float, default=0.05, help="Memory decay rate")
    simulate_parser.add_argument("--drift", type=float, default=0.02, help="Importance drift per day")
    simulate_parser.add_argument("--output", type=str, help="Output file for simulation statistics (JSON)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create synthetic memories")
    create_parser.add_argument("--count", type=int, default=10, help="Number of memories to create")
    create_parser.add_argument("--type", choices=["EPISODIC", "SEMANTIC", "PROCEDURAL", "EMOTIONAL"], 
                               default="EPISODIC", help="Memory type")
    create_parser.add_argument("--template", type=str, default="Synthetic memory {index}",
                               help="Content template")
    create_parser.add_argument("--min-importance", type=float, default=0.1, help="Minimum importance")
    create_parser.add_argument("--max-importance", type=float, default=0.9, help="Maximum importance")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize memory patterns")
    visualize_parser.add_argument("--recent", type=int, default=50, help="Number of recent memories to visualize")
    visualize_parser.add_argument("--output", type=str, help="Output file for visualization (PNG)")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--output", type=str, help="Output file for statistics (JSON)")
    
    # Common arguments
    parser.add_argument("--snapshot", type=str, help="Load a specific memory snapshot")
    parser.add_argument("--store", type=str, help="Path to memory store file (JSON)")
    parser.add_argument("--save-snapshot", type=str, help="Save a snapshot after operation")
    
    return parser.parse_args()

def load_memory_store(file_path: Optional[str] = None) -> MemoryStore:
    """Load a memory store from file or create a new one."""
    memory_store = MemoryStore()
    
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Process the loaded data
            if "memories" in data and isinstance(data["memories"], list):
                for memory_data in data["memories"]:
                    memory = Memory.from_dict(memory_data)
                    memory_store.add_memory(memory)
                
                logger.info(f"Loaded {len(data['memories'])} memories from {file_path}")
            else:
                logger.warning(f"No valid memories found in {file_path}")
        except Exception as e:
            logger.error(f"Failed to load memory store: {e}")
    
    return memory_store

def save_memory_store(memory_store: MemoryStore, file_path: str) -> bool:
    """Save a memory store to a file."""
    try:
        memories = memory_store.get_all_memories()
        data = {
            "timestamp": datetime.now().isoformat(),
            "memory_count": len(memories),
            "memories": [memory.to_dict() for memory in memories]
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(memories)} memories to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save memory store: {e}")
        return False

def main():
    """Main function to execute commands."""
    args = parse_args()
    
    if not args.command:
        print("Please specify a command. Use --help for more information.")
        return
    
    # Initialize the sandbox
    memory_store = load_memory_store(args.store)
    sandbox = MemorySandbox(memory_store=memory_store)
    
    # Load snapshot if specified
    if args.snapshot:
        try:
            sandbox.load_snapshot(args.snapshot)
            print(f"Loaded snapshot: {args.snapshot}")
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return
    
    # Execute the appropriate command
    if args.command == "simulate":
        print(f"Simulating memory dynamics over {args.days} days...")
        sandbox.simulate_memory_dynamics(
            duration_days=args.days,
            memory_creation_rate=args.rate,
            memory_decay_rate=args.decay,
            importance_drift=args.drift
        )
        
        # Output statistics if requested
        if args.output:
            try:
                stats = sandbox.calculate_memory_statistics()
                with open(args.output, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Statistics saved to {args.output}")
            except Exception as e:
                logger.error(f"Failed to save statistics: {e}")
    
    elif args.command == "create":
        print(f"Creating {args.count} {args.type} memories...")
        memory_type = getattr(MemoryType, args.type)
        sandbox.batch_create_memories(
            count=args.count,
            memory_type=memory_type,
            template=args.template,
            importance_range=(args.min_importance, args.max_importance)
        )
        print(f"Created {args.count} memories")
    
    elif args.command == "visualize":
        print(f"Visualizing {args.recent} recent memories...")
        
        # If output is specified, configure matplotlib to save to file
        if args.output:
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')  # Non-interactive backend
        
        sandbox.visualize_memory_activation(n_recent=args.recent)
        
        if args.output:
            import matplotlib.pyplot as plt
            plt.savefig(args.output)
            print(f"Visualization saved to {args.output}")
    
    elif args.command == "stats":
        try:
            stats = sandbox.calculate_memory_statistics()
            
            # Print basic stats to console
            print("\nMemory Statistics:")
            print(f"Total memories: {stats['basic']['memory_count']}")
            print(f"Average importance: {stats['basic']['avg_importance']:.4f}")
            print(f"Time span: {stats['basic']['time_span_days']:.1f} days")
            
            print("\nMemory Types:")
            for memory_type, count in stats['basic']['memory_types'].items():
                percentage = (count / stats['basic']['memory_count']) * 100
                print(f"  {memory_type}: {count} ({percentage:.1f}%)")
            
            print("\nTop Retrievable Memories:")
            for i, memory in enumerate(stats["retrieval"]["top_memories"]):
                print(f"  {i+1}. [{memory['type']}] {memory['content']} " +
                     f"(Retrieval Probability: {memory['probability']:.4f})")
            
            # Save full stats to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"\nDetailed statistics saved to {args.output}")
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
    
    # Save snapshot if requested
    if args.save_snapshot:
        try:
            sandbox.take_snapshot(args.save_snapshot)
            print(f"Saved snapshot: {args.save_snapshot}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    # Save memory store if store path was provided
    if args.store:
        save_memory_store(sandbox.memory_store, args.store)

if __name__ == "__main__":
    main() 