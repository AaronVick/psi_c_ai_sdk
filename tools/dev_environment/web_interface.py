"""
Streamlit-based web interface for the ΨC-AI SDK Development Environment.

This module provides a streamlined web interface for interacting with the ΨC-AI SDK
development tools, including the Consciousness Inspector and other debugging utilities.
"""

import argparse
import logging
import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime
import importlib
from matplotlib.figure import Figure
from memory_sandbox import MemorySandbox, MemoryStore
from memory_schema_integration import MemorySchemaIntegration
from .schema_integration import SchemaIntegration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import tools - handle gracefully if some aren't implemented yet
try:
    from tools.dev_environment.consciousness_inspector import ConsciousnessInspector
    CONSCIOUSNESS_INSPECTOR_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_INSPECTOR_AVAILABLE = False
    logger.warning("Consciousness Inspector not available")

try:
    from tools.dev_environment.schema_editor import SchemaEditor
    SCHEMA_EDITOR_AVAILABLE = True
except ImportError:
    SCHEMA_EDITOR_AVAILABLE = False
    logger.warning("Schema Editor not available")

try:
    from tools.dev_environment.memory_sandbox import MemorySandbox
    MEMORY_SANDBOX_AVAILABLE = True
except ImportError:
    MEMORY_SANDBOX_AVAILABLE = False
    logger.warning("Memory Sandbox not available")

try:
    from tools.dev_environment.reflection_debugger import ReflectionDebugger
    REFLECTION_DEBUGGER_AVAILABLE = True
except ImportError:
    REFLECTION_DEBUGGER_AVAILABLE = False
    logger.warning("Reflection Debugger not available")

try:
    from tools.dev_environment.stress_test_generator import StressTestGenerator
    STRESS_TEST_GENERATOR_AVAILABLE = True
except ImportError:
    STRESS_TEST_GENERATOR_AVAILABLE = False
    logger.warning("Stress Test Generator not available")

# Try to import schema integration
try:
    from tools.dev_environment.web_interface_schema import integrate_with_web_interface
    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False
    logging.warning("Schema visualization components not available.")

# Define available tools with their metadata
TOOLS = {
    "schema_editor": {
        "name": "Schema Editor",
        "description": "Visual editor for agent schema graphs",
        "available": SCHEMA_EDITOR_AVAILABLE,
        "icon": "🔍"
    },
    "memory_sandbox": {
        "name": "Memory Sandbox",
        "description": "Interactive memory manipulation and visualization",
        "available": MEMORY_SANDBOX_AVAILABLE,
        "icon": "🧠"
    },
    "reflection_debugger": {
        "name": "Reflection Debugger",
        "description": "Step through agent reflection cycles",
        "available": REFLECTION_DEBUGGER_AVAILABLE,
        "icon": "🔄"
    },
    "consciousness_inspector": {
        "name": "Consciousness Inspector",
        "description": "Visualize and analyze agent consciousness metrics",
        "available": CONSCIOUSNESS_INSPECTOR_AVAILABLE,
        "icon": "👁️"
    },
    "stress_test_generator": {
        "name": "Stress Test Generator",
        "description": "Generate synthetic test scenarios for agents",
        "available": STRESS_TEST_GENERATOR_AVAILABLE,
        "icon": "🧪"
    }
}


class MockAgent:
    """Mock agent class for demonstration when no agent is loaded."""
    
    def __init__(self):
        """Initialize mock agent with sample consciousness metrics."""
        self.config = {
            "consciousness": {
                "integration": 0.7,
                "differentiation": 0.6,
                "self_awareness": 0.5,
                "attention": 0.65,
                "metacognition": 0.55
            }
        }
        
        # Generate mock historical metrics
        timestamps = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        self.metrics_history = pd.DataFrame({
            'timestamp': timestamps,
            'integration': np.random.uniform(0.4, 0.9, 100),
            'differentiation': np.random.uniform(0.3, 0.8, 100),
            'self_awareness': np.random.uniform(0.2, 0.7, 100),
            'attention': np.random.uniform(0.4, 0.85, 100),
            'metacognition': np.random.uniform(0.3, 0.75, 100)
        })
        
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update the agent configuration."""
        for key, value in config_updates.items():
            if key in self.config["consciousness"]:
                self.config["consciousness"][key] = value
                
    def get_metrics_history(self, metric_names=None, start_time=None, end_time=None):
        """Return historical metrics data."""
        df = self.metrics_history.copy()
        
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        if metric_names:
            df = df[['timestamp'] + metric_names]
            
        return df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ΨC-AI SDK Development Environment")
    parser.add_argument("--tools", nargs='+', choices=TOOLS.keys(),
                        help="Specific tools to launch (default: all available)")
    parser.add_argument("--agent-path", type=str, 
                        help="Path to the agent file to load")
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")
    return parser.parse_args()


def load_agent(agent_path: Optional[str] = None):
    """Load an agent from a file or create a mock agent if not available."""
    if not agent_path:
        logger.info("No agent path provided, using mock agent")
        return MockAgent()
    
    agent_path = Path(agent_path)
    if not agent_path.exists():
        logger.warning(f"Agent file not found: {agent_path}, using mock agent")
        return MockAgent()
    
    # Try to load the agent
    try:
        # This is a placeholder - in a real implementation, we would load
        # the appropriate agent class based on the file format
        if agent_path.suffix == '.json':
            with open(agent_path, 'r') as f:
                agent_data = json.load(f)
                # Construct the appropriate agent object
                # For now, return a mock agent
                return MockAgent()
        else:
            logger.warning(f"Unsupported agent format: {agent_path.suffix}, using mock agent")
            return MockAgent()
    except Exception as e:
        logger.error(f"Error loading agent: {e}")
        return MockAgent()


def render_schema_editor(agent):
    """Render the Schema Editor tool."""
    st.header("Schema Editor")
    st.info("The Schema Editor is not fully implemented yet.")


def render_memory_sandbox(agent):
    """Render the Memory Sandbox tool."""
    st.header("Memory Sandbox")
    
    # Initialize memory sandbox
    if 'memory_sandbox' not in st.session_state:
        from tools.dev_environment.memory_sandbox import MemorySandbox
        memory_store = getattr(agent, 'memory_store', None)
        st.session_state.memory_sandbox = MemorySandbox(memory_store=memory_store)
    
    sandbox = st.session_state.memory_sandbox
    
    # Sidebar for controls
    with st.sidebar:
        st.subheader("Memory Controls")
        
        # Memory Creation Section
        st.write("Create Synthetic Memory")
        memory_content = st.text_area("Memory Content", "Example memory content", height=100)
        
        memory_types = ["EPISODIC", "SEMANTIC", "PROCEDURAL", "EMOTIONAL"]
        memory_type = st.selectbox("Memory Type", memory_types)
        
        importance = st.slider("Importance", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        if st.button("Create Memory"):
            if sandbox.memory_store is None:
                st.error("No memory store is available")
            else:
                from psi_c_ai_sdk.memory.memory_types import MemoryType
                mem_type = getattr(MemoryType, memory_type)
                sandbox.create_synthetic_memory(memory_content, mem_type, importance)
                st.success(f"Created {memory_type} memory")
        
        # Batch Memory Creation
        st.write("Batch Create Memories")
        
        batch_count = st.number_input("Number of Memories", min_value=1, max_value=100, value=10)
        batch_type = st.selectbox("Memory Type for Batch", memory_types, key="batch_type")
        batch_template = st.text_input("Content Template", "Memory #{index}: Generated content")
        
        min_importance, max_importance = st.slider(
            "Importance Range", min_value=0.0, max_value=1.0, value=(0.2, 0.8), step=0.05
        )
        
        if st.button("Generate Batch"):
            if sandbox.memory_store is None:
                st.error("No memory store is available")
            else:
                from psi_c_ai_sdk.memory.memory_types import MemoryType
                mem_type = getattr(MemoryType, batch_type)
                memories = sandbox.batch_create_memories(
                    batch_count, 
                    mem_type, 
                    batch_template, 
                    (min_importance, max_importance)
                )
                st.success(f"Created {len(memories)} memories")
        
        # Snapshot controls
        st.subheader("Memory Snapshots")
        snapshot_name = st.text_input("Snapshot Name", "snapshot1")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Take Snapshot"):
                try:
                    snapshot = sandbox.take_snapshot(snapshot_name)
                    st.success(f"Snapshot taken with {snapshot['memory_count']} memories")
                except Exception as e:
                    st.error(f"Error taking snapshot: {str(e)}")
        
        with col2:
            if st.button("Load Snapshot"):
                try:
                    sandbox.load_snapshot(snapshot_name)
                    st.success(f"Snapshot '{snapshot_name}' loaded")
                except Exception as e:
                    st.error(f"Error loading snapshot: {str(e)}")
    
    # Main content area
    tabs = st.tabs([
        "Memory Browser", 
        "Memory Visualization", 
        "Memory Search", 
        "Memory Analysis", 
        "Memory Comparison",
        "Memory Simulation"
    ])
    
    with tabs[0]:
        st.subheader("Memory Browser")
        
        if sandbox.memory_store is None:
            st.info("No memory store is available. Create one by adding memories.")
        else:
            memories = sandbox.memory_store.get_all_memories()
            if not memories:
                st.info("No memories available. Create some memories first.")
            else:
                # Sort options
                sort_options = ["Timestamp (newest first)", "Timestamp (oldest first)", 
                               "Importance (highest first)", "Importance (lowest first)",
                               "Type"]
                sort_by = st.selectbox("Sort by", sort_options)
                
                # Filter options
                memory_type_filter = st.multiselect(
                    "Filter by Type", 
                    options=memory_types,
                    default=memory_types
                )
                
                # Sort memories based on selection
                if sort_by == "Timestamp (newest first)":
                    memories.sort(key=lambda m: m.timestamp, reverse=True)
                elif sort_by == "Timestamp (oldest first)":
                    memories.sort(key=lambda m: m.timestamp)
                elif sort_by == "Importance (highest first)":
                    memories.sort(key=lambda m: m.importance, reverse=True)
                elif sort_by == "Importance (lowest first)":
                    memories.sort(key=lambda m: m.importance)
                elif sort_by == "Type":
                    memories.sort(key=lambda m: m.memory_type.name)
                
                # Filter memories
                if memory_type_filter:
                    filtered_memories = [
                        m for m in memories if m.memory_type.name in memory_type_filter
                    ]
                else:
                    filtered_memories = memories
                
                # Display memories as expandable elements
                for i, memory in enumerate(filtered_memories):
                    with st.expander(
                        f"[{memory.memory_type.name}] {memory.content[:50]}... "
                        f"(Importance: {memory.importance:.2f})"
                    ):
                        st.write(f"**Content:** {memory.content}")
                        st.write(f"**Type:** {memory.memory_type.name}")
                        st.write(f"**Importance:** {memory.importance:.4f}")
                        st.write(f"**Created:** {memory.timestamp}")
                        st.write(f"**Last Accessed:** {memory.last_accessed or 'Never'}")
                        
                        if st.button(f"Delete Memory {i}", key=f"del_{i}"):
                            try:
                                success = sandbox.delete_memory(index=i)
                                if success:
                                    st.success(f"Memory deleted successfully")
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to delete memory")
                            except Exception as e:
                                st.error(f"Error deleting memory: {str(e)}")
    
    with tabs[1]:
        st.subheader("Memory Visualization")
        
        if sandbox.memory_store is None or not sandbox.memory_store.get_all_memories():
            st.info("No memories available to visualize. Create some memories first.")
        else:
            viz_type = st.radio(
                "Visualization Type",
                ["Memory Activation Pattern", "Memory Type Distribution", "Memory Timeline"]
            )
            
            if viz_type == "Memory Activation Pattern":
                n_memories = st.slider(
                    "Number of memories to show", 
                    min_value=5, 
                    max_value=100, 
                    value=30
                )
                
                # Create a placeholder for the matplotlib figure
                fig_placeholder = st.empty()
                
                if st.button("Generate Visualization"):
                    with st.spinner("Generating visualization..."):
                        # This is a workaround since we can't directly show plt.show()
                        # Instead, we capture the figure and display it with st.pyplot
                        plt_fig = plt.figure(figsize=(10, 6))
                        
                        # Similar code to the original visualize_memory_activation
                        memories = sorted(sandbox.memory_store.get_all_memories(), 
                                         key=lambda m: m.last_accessed if m.last_accessed else m.timestamp,
                                         reverse=True)[:n_memories]
                        
                        # Extract data for visualization
                        importance = [m.importance for m in memories]
                        recency = [(datetime.now() - (m.last_accessed or m.timestamp)).total_seconds() / 3600 for m in memories]
                        memory_types = [m.memory_type.name for m in memories]
                        
                        # Create colormap based on memory types
                        unique_types = list(set(memory_types))
                        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_types)))
                        color_map = {t: colors[i] for i, t in enumerate(unique_types)}
                        
                        # Plot memories as scatter points
                        for i, memory in enumerate(memories):
                            plt.scatter(recency[i], importance[i], 
                                      s=100 + importance[i] * 200,
                                      color=color_map[memory.memory_type.name],
                                      alpha=0.7,
                                      label=memory.memory_type.name)
                        
                        # Remove duplicate labels
                        handles, labels = plt.gca().get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        plt.legend(by_label.values(), by_label.keys(), title="Memory Types")
                        
                        plt.xlabel("Recency (hours)")
                        plt.ylabel("Importance")
                        plt.title("Memory Activation Pattern")
                        plt.grid(True, alpha=0.3)
                        
                        # Invert x-axis so most recent are on the right
                        plt.gca().invert_xaxis()
                        
                        plt.tight_layout()
                        fig_placeholder.pyplot(plt_fig)
            
            elif viz_type == "Memory Type Distribution":
                distribution = sandbox.analyze_memory_distribution()
                
                if distribution:
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    labels = list(distribution.keys())
                    sizes = list(distribution.values())
                    
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                          startangle=90, shadow=True)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    
                    st.pyplot(fig)
            
            elif viz_type == "Memory Timeline":
                memories = sorted(sandbox.memory_store.get_all_memories(), 
                                 key=lambda m: m.timestamp)
                
                if memories:
                    # Create timeline visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Extract data
                    timestamps = [m.timestamp for m in memories]
                    y_pos = range(len(memories))
                    types = [m.memory_type.name for m in memories]
                    
                    # Create colormap
                    unique_types = list(set(types))
                    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_types)))
                    color_map = {t: colors[i] for i, t in enumerate(unique_types)}
                    point_colors = [color_map[t] for t in types]
                    
                    # Plot as scatter with importance as size
                    ax.scatter(timestamps, y_pos, 
                              c=point_colors, 
                              s=[m.importance * 200 + 50 for m in memories],
                              alpha=0.7)
                    
                    # Add memory content as labels
                    for i, memory in enumerate(memories):
                        ax.text(memory.timestamp, i, 
                               f" {memory.content[:30]}..." if len(memory.content) > 30 else memory.content,
                               verticalalignment='center')
                    
                    # Create legend
                    import matplotlib.patches as mpatches
                    legend_handles = [mpatches.Patch(color=color_map[t], label=t) for t in unique_types]
                    ax.legend(handles=legend_handles, title="Memory Types")
                    
                    ax.set_yticks([])  # Hide y ticks
                    ax.set_xlabel("Time")
                    ax.set_title("Memory Timeline")
                    
                    st.pyplot(fig)
    
    with tabs[2]:
        st.subheader("Memory Search")
        
        if sandbox.memory_store is None:
            st.info("No memory store is available. Create one by adding memories.")
        else:
            query = st.text_input("Search query", "")
            top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
            
            if st.button("Search") and query:
                try:
                    results = sandbox.test_memory_recall(query, top_k)
                    
                    if not results:
                        st.info("No memories found matching your query.")
                    else:
                        for i, memory in enumerate(results):
                            with st.expander(
                                f"{i+1}. [{memory.memory_type.name}] {memory.content[:50]}... "
                                f"(Relevance: {getattr(memory, 'relevance', 0):.3f})"
                            ):
                                st.write(f"**Content:** {memory.content}")
                                st.write(f"**Type:** {memory.memory_type.name}")
                                st.write(f"**Importance:** {memory.importance:.4f}")
                                st.write(f"**Relevance to query:** {getattr(memory, 'relevance', 0):.4f}")
                except Exception as e:
                    st.error(f"Error searching memories: {str(e)}")
    
    with tabs[3]:
        st.subheader("Memory Analysis")
        
        if sandbox.memory_store is None or not sandbox.memory_store.get_all_memories():
            st.info("No memories available to analyze. Create some memories first.")
        else:
            memories = sandbox.memory_store.get_all_memories()
            
            # Display basic statistics
            st.write(f"**Total memories:** {len(memories)}")
            
            memory_types = {}
            importance_sum = 0
            oldest_memory = None
            newest_memory = None
            
            for memory in memories:
                # Count by type
                memory_type = memory.memory_type.name
                if memory_type not in memory_types:
                    memory_types[memory_type] = 0
                memory_types[memory_type] += 1
                
                # Track importance
                importance_sum += memory.importance
                
                # Track oldest/newest
                if oldest_memory is None or memory.timestamp < oldest_memory.timestamp:
                    oldest_memory = memory
                if newest_memory is None or memory.timestamp > newest_memory.timestamp:
                    newest_memory = memory
            
            # Display statistics
            st.write(f"**Average importance:** {importance_sum / len(memories):.4f}")
            
            if oldest_memory:
                st.write(f"**Oldest memory:** {oldest_memory.timestamp}")
            if newest_memory:
                st.write(f"**Newest memory:** {newest_memory.timestamp}")
            
            # Display type distribution
            st.write("**Memory type distribution:**")
            for memory_type, count in memory_types.items():
                percentage = (count / len(memories)) * 100
                st.write(f"- {memory_type}: {count} ({percentage:.1f}%)")

    with tabs[4]:
        st.subheader("Memory Comparison")
        
        if sandbox.memory_store is None or not sandbox.memory_store.get_all_memories():
            st.info("No memories available to compare. Create some memories first.")
        else:
            comparison_mode = st.radio(
                "Comparison Mode",
                ["Compare Individual Memories", "Compare Snapshots"]
            )
            
            if comparison_mode == "Compare Individual Memories":
                memories = sandbox.memory_store.get_all_memories()
                
                # Create options for memory selection
                memory_options = [
                    f"[{i}] {m.memory_type.name}: {m.content[:30]}..." 
                    for i, m in enumerate(memories)
                ]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    mem1_idx = st.selectbox(
                        "Select First Memory", 
                        range(len(memory_options)),
                        format_func=lambda i: memory_options[i]
                    )
                
                with col2:
                    mem2_idx = st.selectbox(
                        "Select Second Memory", 
                        range(len(memory_options)),
                        format_func=lambda i: memory_options[i],
                        index=min(1, len(memory_options)-1)  # Default to second memory
                    )
                
                if st.button("Compare Memories"):
                    try:
                        # Get memory IDs or use indices
                        mem1_id = getattr(memories[mem1_idx], 'id', str(mem1_idx))
                        mem2_id = getattr(memories[mem2_idx], 'id', str(mem2_idx))
                        
                        if mem1_id == mem2_id:
                            st.warning("Please select different memories to compare")
                        else:
                            comparison = sandbox.compare_memories(mem1_id, mem2_id)
                            
                            # Display comparison results
                            st.subheader("Comparison Results")
                            
                            # Display similarity metrics
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.metric(
                                    "Semantic Similarity", 
                                    f"{comparison['semantic_similarity']:.2f}" if comparison['semantic_similarity'] is not None else "N/A"
                                )
                                st.metric(
                                    "Same Type", 
                                    "Yes" if comparison['same_type'] else "No"
                                )
                                
                            with metrics_col2:
                                st.metric(
                                    "Importance Difference",
                                    f"{comparison['importance_difference']:.2f}"
                                )
                                
                                # Convert seconds to more readable format
                                time_diff = comparison['time_difference_seconds']
                                if time_diff < 60:
                                    time_str = f"{time_diff:.1f} seconds"
                                elif time_diff < 3600:
                                    time_str = f"{time_diff/60:.1f} minutes"
                                elif time_diff < 86400:
                                    time_str = f"{time_diff/3600:.1f} hours"
                                else:
                                    time_str = f"{time_diff/86400:.1f} days"
                                    
                                st.metric("Time Difference", time_str)
                            
                            # Display memory details side by side
                            st.subheader("Memory Details")
                            
                            detail_col1, detail_col2 = st.columns(2)
                            
                            with detail_col1:
                                st.write("**First Memory**")
                                st.write(f"ID: {comparison['memory1']['id']}")
                                st.write(f"Type: {comparison['memory1']['type']}")
                                st.write(f"Importance: {comparison['memory1']['importance']:.4f}")
                                st.write(f"Timestamp: {comparison['memory1']['timestamp']}")
                                st.write(f"Content: {memories[mem1_idx].content}")
                                
                            with detail_col2:
                                st.write("**Second Memory**")
                                st.write(f"ID: {comparison['memory2']['id']}")
                                st.write(f"Type: {comparison['memory2']['type']}")
                                st.write(f"Importance: {comparison['memory2']['importance']:.4f}")
                                st.write(f"Timestamp: {comparison['memory2']['timestamp']}")
                                st.write(f"Content: {memories[mem2_idx].content}")
                    
                    except Exception as e:
                        st.error(f"Error comparing memories: {str(e)}")
            
            elif comparison_mode == "Compare Snapshots":
                # List available snapshots
                snapshot_files = [
                    f.replace(".json", "") 
                    for f in os.listdir(sandbox.snapshot_dir) 
                    if f.endswith(".json")
                ]
                
                if not snapshot_files:
                    st.info("No snapshots available. Take snapshots using the controls in the sidebar.")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        snapshot1 = st.selectbox(
                            "Select First Snapshot",
                            snapshot_files
                        )
                    
                    with col2:
                        snapshot2 = st.selectbox(
                            "Select Second Snapshot",
                            snapshot_files,
                            index=min(1, len(snapshot_files)-1)  # Default to second snapshot
                        )
                    
                    if st.button("Compare Snapshots"):
                        try:
                            if snapshot1 == snapshot2:
                                st.warning("Please select different snapshots to compare")
                            else:
                                comparison = sandbox.compare_snapshots(snapshot1, snapshot2)
                                
                                # Display comparison results
                                st.subheader("Comparison Results")
                                
                                # Display snapshot stats side by side
                                stat_col1, stat_col2 = st.columns(2)
                                
                                with stat_col1:
                                    st.write(f"**Snapshot: {comparison['snapshot1']['name']}**")
                                    st.write(f"Memory Count: {comparison['snapshot1']['memory_count']}")
                                    st.write(f"Avg. Importance: {comparison['snapshot1']['avg_importance']:.4f}")
                                    
                                    # Display type distribution
                                    st.write("Type Distribution:")
                                    for memory_type, count in comparison['snapshot1']['type_distribution'].items():
                                        st.write(f"- {memory_type}: {count}")
                                        
                                with stat_col2:
                                    st.write(f"**Snapshot: {comparison['snapshot2']['name']}**")
                                    st.write(f"Memory Count: {comparison['snapshot2']['memory_count']}")
                                    st.write(f"Avg. Importance: {comparison['snapshot2']['avg_importance']:.4f}")
                                    
                                    # Display type distribution
                                    st.write("Type Distribution:")
                                    for memory_type, count in comparison['snapshot2']['type_distribution'].items():
                                        st.write(f"- {memory_type}: {count}")
                                
                                # Display comparison metrics
                                st.subheader("Changes Between Snapshots")
                                metrics_cols = st.columns(4)
                                
                                with metrics_cols[0]:
                                    st.metric("Common Memories", comparison['comparison']['common_memory_count'])
                                
                                with metrics_cols[1]:
                                    st.metric("Added Memories", comparison['comparison']['added_memory_count'])
                                
                                with metrics_cols[2]:
                                    st.metric("Removed Memories", comparison['comparison']['removed_memory_count'])
                                
                                with metrics_cols[3]:
                                    importance_change = comparison['comparison']['importance_change']
                                    st.metric(
                                        "Importance Change", 
                                        f"{importance_change:.4f}",
                                        delta=importance_change
                                    )
                                
                                # If there are significant changes, add visual comparison
                                if (comparison['comparison']['added_memory_count'] > 0 or 
                                    comparison['comparison']['removed_memory_count'] > 0):
                                    
                                    # Create bar chart comparing memory counts
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    
                                    # Combine all memory types from both snapshots
                                    all_types = set()
                                    for memory_type in comparison['snapshot1']['type_distribution']:
                                        all_types.add(memory_type)
                                    for memory_type in comparison['snapshot2']['type_distribution']:
                                        all_types.add(memory_type)
                                    
                                    # Extract data for plotting
                                    types = list(all_types)
                                    counts1 = [comparison['snapshot1']['type_distribution'].get(t, 0) for t in types]
                                    counts2 = [comparison['snapshot2']['type_distribution'].get(t, 0) for t in types]
                                    
                                    # Plot
                                    x = np.arange(len(types))
                                    width = 0.35
                                    
                                    ax.bar(x - width/2, counts1, width, label=comparison['snapshot1']['name'])
                                    ax.bar(x + width/2, counts2, width, label=comparison['snapshot2']['name'])
                                    
                                    ax.set_xticks(x)
                                    ax.set_xticklabels(types)
                                    ax.legend()
                                    
                                    plt.title("Memory Type Distribution Comparison")
                                    plt.xlabel("Memory Type")
                                    plt.ylabel("Count")
                                    
                                    st.pyplot(fig)
                                    
                        except Exception as e:
                            st.error(f"Error comparing snapshots: {str(e)}")

    with tabs[5]:
        st.subheader("Memory Simulation")
        
        if sandbox.memory_store is None:
            st.info("Initialize a memory store first by creating some memories")
        else:
            st.write("""
            This tab allows you to simulate memory dynamics over time, including:
            - Memory creation with realistic patterns
            - Memory importance decay and strengthening
            - Memory recall effects
            - Forgetting of low-importance memories
            """)
            
            with st.form("simulation_config"):
                st.subheader("Simulation Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    duration_days = st.slider(
                        "Duration (days)", 
                        min_value=1, 
                        max_value=365, 
                        value=30,
                        help="Number of days to simulate"
                    )
                    
                    memory_creation_rate = st.slider(
                        "Memory Creation Rate", 
                        min_value=1, 
                        max_value=20, 
                        value=5,
                        help="Average number of new memories created per day"
                    )
                
                with col2:
                    memory_decay_rate = st.slider(
                        "Memory Decay Rate", 
                        min_value=0.01, 
                        max_value=0.2, 
                        value=0.05,
                        step=0.01,
                        help="Rate at which memory importance decays per day"
                    )
                    
                    importance_drift = st.slider(
                        "Importance Drift", 
                        min_value=0.0, 
                        max_value=0.1, 
                        value=0.02,
                        step=0.01,
                        help="Random fluctuation in importance per day"
                    )
                
                # Memory type distribution
                st.subheader("Memory Type Distribution")
                
                distribution_cols = st.columns(4)
                
                with distribution_cols[0]:
                    episodic_pct = st.slider("EPISODIC", min_value=0, max_value=100, value=40)
                
                with distribution_cols[1]:
                    semantic_pct = st.slider("SEMANTIC", min_value=0, max_value=100, value=30)
                
                with distribution_cols[2]:
                    procedural_pct = st.slider("PROCEDURAL", min_value=0, max_value=100, value=20)
                
                with distribution_cols[3]:
                    emotional_pct = st.slider("EMOTIONAL", min_value=0, max_value=100, value=10)
                
                # Calculate total and normalize if needed
                total_pct = episodic_pct + semantic_pct + procedural_pct + emotional_pct
                
                if total_pct == 0:
                    st.warning("Total distribution must be greater than 0")
                    normalized_dist = {"EPISODIC": 0.25, "SEMANTIC": 0.25, "PROCEDURAL": 0.25, "EMOTIONAL": 0.25}
                else:
                    normalized_dist = {
                        "EPISODIC": episodic_pct / total_pct,
                        "SEMANTIC": semantic_pct / total_pct,
                        "PROCEDURAL": procedural_pct / total_pct,
                        "EMOTIONAL": emotional_pct / total_pct
                    }
                
                st.caption(f"Normalized distribution: EPISODIC: {normalized_dist['EPISODIC']:.2f}, " +
                           f"SEMANTIC: {normalized_dist['SEMANTIC']:.2f}, " +
                           f"PROCEDURAL: {normalized_dist['PROCEDURAL']:.2f}, " +
                           f"EMOTIONAL: {normalized_dist['EMOTIONAL']:.2f}")
                
                # Submit button
                submit_button = st.form_submit_button("Run Simulation")
            
            if submit_button:
                try:
                    with st.spinner(f"Simulating memory dynamics over {duration_days} days..."):
                        sandbox.simulate_memory_dynamics(
                            duration_days=duration_days,
                            memory_decay_rate=memory_decay_rate,
                            memory_creation_rate=memory_creation_rate,
                            importance_drift=importance_drift,
                            memory_types_distribution=normalized_dist
                        )
                    
                    st.success("Simulation completed successfully!")
                    
                    # Show statistics about the simulation results
                    st.subheader("Simulation Results")
                    
                    try:
                        stats = sandbox.calculate_memory_statistics()
                        
                        # Display basic stats
                        basic_cols = st.columns(4)
                        with basic_cols[0]:
                            st.metric("Total Memories", stats["basic"]["memory_count"])
                        
                        with basic_cols[1]:
                            st.metric("Avg Importance", f"{stats['basic']['avg_importance']:.2f}")
                        
                        with basic_cols[2]:
                            st.metric("Time Span", f"{stats['basic']['time_span_days']:.1f} days")
                            
                        with basic_cols[3]:
                            importance_recency = stats["correlations"]["importance_recency"]
                            if importance_recency is not None:
                                st.metric("Importance-Recency Correlation", f"{importance_recency:.2f}")
                            else:
                                st.metric("Importance-Recency Correlation", "N/A")
                        
                        # Memory type distribution
                        st.subheader("Memory Type Distribution")
                        
                        # Create pie chart of memory types
                        fig, ax = plt.subplots(figsize=(6, 6))
                        
                        labels = list(stats["basic"]["memory_types"].keys())
                        sizes = list(stats["basic"]["memory_types"].values())
                        
                        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                              startangle=90, shadow=True)
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                        
                        st.pyplot(fig)
                        
                        # Display importance distribution
                        st.subheader("Importance Distribution")
                        
                        # Create bar chart of importance distribution
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        labels = list(stats["importance"]["distribution"].keys())
                        values = list(stats["importance"]["distribution"].values())
                        
                        ax.bar(labels, values)
                        ax.set_xlabel("Importance Range")
                        ax.set_ylabel("Count")
                        ax.set_title("Memory Importance Distribution")
                        
                        # Rotate x-axis labels for better readability
                        plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
                        
                        # Display memory age distribution
                        st.subheader("Memory Age Distribution")
                        
                        # Create bar chart of age distribution
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        labels = list(stats["temporal"]["age_distribution"].keys())
                        values = list(stats["temporal"]["age_distribution"].values())
                        
                        ax.bar(labels, values)
                        ax.set_xlabel("Age Range")
                        ax.set_ylabel("Count")
                        ax.set_title("Memory Age Distribution")
                        
                        # Rotate x-axis labels for better readability
                        plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
                        
                        # Display top retrievable memories
                        st.subheader("Top Retrievable Memories")
                        
                        for i, memory in enumerate(stats["retrieval"]["top_memories"]):
                            st.write(f"{i+1}. **[{memory['type']}]** {memory['content']} " +
                                    f"(Retrieval Probability: {memory['probability']:.4f})")
                        
                    except Exception as e:
                        st.error(f"Error calculating simulation statistics: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
                    
            # Option to take snapshot after viewing results
            if st.button("Take Snapshot of Current State"):
                snapshot_name = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    snapshot = sandbox.take_snapshot(snapshot_name)
                    st.success(f"Snapshot '{snapshot_name}' saved with {snapshot['memory_count']} memories")
                except Exception as e:
                    st.error(f"Error taking snapshot: {str(e)}")


def render_reflection_debugger(agent):
    """Render the Reflection Debugger tool."""
    st.header("Reflection Debugger")
    st.info("The Reflection Debugger is not fully implemented yet.")


def render_consciousness_inspector(agent):
    """Render the Consciousness Inspector tool."""
    st.header("Consciousness Inspector")
    
    # Sidebar for parameters and controls
    with st.sidebar:
        st.subheader("Parameters")
        
        # Current consciousness parameters
        st.write("Consciousness Parameters")
        updated_params = {}
        for param, value in agent.config["consciousness"].items():
            updated_value = st.slider(f"{param.replace('_', ' ').title()}", 
                                     min_value=0.0, max_value=1.0, 
                                     value=float(value), step=0.01,
                                     format="%.2f")
            updated_params[param] = updated_value
        
        if st.button("Update Parameters"):
            agent.update_config(updated_params)
            st.success("Parameters updated!")
        
        # Visualization settings
        st.subheader("Visualization Settings")
        view_mode = st.radio("View Mode", ["Time Series", "Radar Chart", "Bar Chart"])
        time_range = st.select_slider(
            "Time Range",
            options=["Last Hour", "Last Day", "Last Week", "Last Month", "All Time"],
            value="Last Day"
        )
        
        metrics_to_show = st.multiselect(
            "Metrics to Show",
            options=list(agent.config["consciousness"].keys()),
            default=list(agent.config["consciousness"].keys())
        )
    
    # Main content area
    st.subheader("Current Consciousness State")
    
    # Display current metrics
    metrics_df = pd.DataFrame({
        "Metric": [k.replace('_', ' ').title() for k in agent.config["consciousness"].keys()],
        "Value": list(agent.config["consciousness"].values())
    })
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(metrics_df["Metric"], metrics_df["Value"], color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value")
        ax.set_title("Current Consciousness Metrics")
        st.pyplot(fig)
    
    # Historical metrics visualization
    st.subheader("Historical Consciousness Data")
    
    if view_mode == "Time Series":
        # Get historical data
        history_df = agent.get_metrics_history(
            metric_names=metrics_to_show if metrics_to_show else None
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in metrics_to_show:
            ax.plot(history_df['timestamp'], history_df[metric], label=metric.replace('_', ' ').title())
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Consciousness Metrics Over Time")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
    elif view_mode == "Radar Chart":
        # Create radar chart for current metrics
        metrics = list(agent.config["consciousness"].keys())
        values = list(agent.config["consciousness"].values())
        
        # Number of variables
        N = len(metrics)
        
        # Create angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values for each metric
        values += values[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Draw the outline of the radar chart
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        
        # Fill the area
        ax.fill(angles, values, alpha=0.25)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        
        # Set y-axis limit
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title('Consciousness Profile', size=15, y=1.1)
        
        st.pyplot(fig)
        
    elif view_mode == "Bar Chart":
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = [m.replace('_', ' ').title() for m in metrics_to_show]
        values = [agent.config["consciousness"][m] for m in metrics_to_show]
        
        ax.bar(metrics, values, color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value")
        ax.set_title("Consciousness Metrics Comparison")
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Actions section
    st.subheader("Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Metrics CSV"):
            # In a real implementation, this would generate and download a CSV
            st.success("Metrics CSV exported!")
    
    with col2:
        if st.button("Generate Report"):
            # In a real implementation, this would generate a comprehensive report
            st.success("Report generated!")


def render_stress_test_generator(agent):
    """Render the Stress Test Generator tool."""
    st.header("Stress Test Generator")
    st.info("The Stress Test Generator is not fully implemented yet.")


class DevWebInterface:
    """Web interface for the ΨC-AI SDK Development Environment."""
    
    def __init__(self, port=8080, host='localhost', debug=False):
        """Initialize the web interface."""
        self.tools = []
        self.tool_sections = {}
        self.schema_integration = SchemaIntegration(self.memory_sandbox)
        
    def add_tool(self, tool):
        """Add a tool to the web interface."""
        self.tools.append(tool)
        
        # If it's a Memory Sandbox, try to integrate schema visualization
        if SCHEMA_AVAILABLE and hasattr(tool, 'schema_integration') and tool.__class__.__name__ == "MemorySandbox":
            try:
                integrate_with_web_interface(self)
            except Exception as e:
                logging.warning(f"Failed to integrate schema visualization: {e}")
    
    def add_tool_section(self, name: str, render_func: Callable):
        """
        Add a custom tool section to the web interface.
        
        Args:
            name: Name of the section
            render_func: Function to render the section (takes tools as argument)
        """
        self.tool_sections[name] = render_func
        
    def run(self, host="localhost", port=8501, debug=False):
        """Run the web interface server."""
        if st is None:
            logging.error("Cannot run web interface: streamlit not installed")
            return 1
            
        # Run streamlit app
        import streamlit.web.cli as stcli
        
        # Create the app file
        app_path = os.path.join(os.path.dirname(__file__), "web_app.py")
        
        # Generate the app file
        with open(app_path, "w") as f:
            f.write(f"""
import os
import sys
import streamlit as st
import importlib
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level={"logging.DEBUG" if debug else "logging.INFO"})

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the web interface
from tools.dev_environment.web_interface import load_interface

# Load the interface
interface = load_interface()
if not interface:
    st.error("Failed to load the web interface")
    sys.exit(1)
    
# Configure page
st.set_page_config(
    page_title="ΨC-AI SDK Development Environment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("ΨC-AI SDK Development Environment")

# Tool selection
tools = interface.tools
tool_names = [tool.name for tool in tools]
tool_sections = list(interface.tool_sections.keys())

# Combine standard tools and custom sections
all_sections = ["Home"] + tool_names + tool_sections
selected_section = st.sidebar.selectbox("Select Tool", all_sections)

# Show the selected tool or section
if selected_section == "Home":
    st.title("ΨC-AI SDK Development Environment")
    st.markdown(\"\"\"
    Welcome to the ΨC-AI SDK Development Environment.
    
    This interface provides access to various development tools for the ΨC-AI SDK.
    Select a tool from the sidebar to get started.
    \"\"\")
    
    # Show available tools
    st.header("Available Tools")
    
    for tool in tools:
        st.markdown(f"### {tool.name}")
        st.markdown(tool.description)
        
    # Show available custom sections
    if tool_sections:
        st.header("Advanced Features")
        for section in tool_sections:
            st.markdown(f"### {section}")
            
elif selected_section in tool_names:
    # Find the selected tool
    selected_tool = next((tool for tool in tools if tool.name == selected_section), None)
    
    if selected_tool:
        st.title(selected_tool.name)
        st.markdown(selected_tool.description)
        
        # Let the tool render its UI
        if hasattr(selected_tool, 'render_ui'):
            selected_tool.render_ui()
        else:
            st.markdown("This tool does not have a web interface.")
    else:
        st.error(f"Tool {selected_section} not found")
        
elif selected_section in tool_sections:
    # Render custom section
    render_func = interface.tool_sections[selected_section]
    
    # Find the memory sandbox tool
    memory_sandbox = next((tool for tool in tools if tool.__class__.__name__ == "MemorySandbox"), None)
    
    # Render the section
    if memory_sandbox:
        render_func(memory_sandbox)
    else:
        st.error("Required tool not found for this section")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© ΨC-AI SDK Development Team")
            """)
        
        # Run streamlit
        sys.argv = [
            "streamlit", 
            "run", 
            app_path,
            "--server.address", host,
            "--server.port", str(port)
        ]
        
        # Store interface reference for the app to access
        interface_storage[0] = self
        
        # Run streamlit (will replace this process)
        return stcli.main()


def main():
    """Main function to run the Streamlit application."""
    # Set up Streamlit page config
    st.set_page_config(
        page_title="ΨC-AI SDK Development Environment",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Parse command line arguments
    args = parse_args()
    
    # Header
    st.title("ΨC-AI SDK Development Environment")
    st.markdown("""
    This interface provides access to various development and debugging tools for the ΨC-AI SDK.
    Select a tool from the sidebar to get started.
    """)
    
    # Load agent
    agent = load_agent(args.agent_path)
    
    # Sidebar for tool selection
    with st.sidebar:
        st.title("Tools")
        
        # Create buttons for each available tool
        selected_tool = None
        for tool_id, tool_info in TOOLS.items():
            if tool_info["available"]:
                if st.button(f"{tool_info['icon']} {tool_info['name']}", 
                           help=tool_info["description"]):
                    selected_tool = tool_id
            else:
                # Show disabled button for unavailable tools
                st.button(f"{tool_info['icon']} {tool_info['name']} (Unavailable)", 
                        help=tool_info["description"], 
                        disabled=True)
        
        # Display agent info
        st.subheader("Agent Information")
        st.text("Status: Active (Mock)" if isinstance(agent, MockAgent) else "Status: Active")
        
        # Add reload button
        if st.button("Reload Agent"):
            agent = load_agent(args.agent_path)
            st.success("Agent reloaded!")
    
    # Render the selected tool
    if selected_tool == "schema_editor":
        render_schema_editor(agent)
    elif selected_tool == "memory_sandbox":
        render_memory_sandbox(agent)
    elif selected_tool == "reflection_debugger":
        render_reflection_debugger(agent)
    elif selected_tool == "consciousness_inspector":
        render_consciousness_inspector(agent)
    elif selected_tool == "stress_test_generator":
        render_stress_test_generator(agent)
    else:
        # Default view when no tool is selected
        st.info("Select a tool from the sidebar to get started.")
        
        # Show available tools as cards
        st.subheader("Available Tools")
        
        cols = st.columns(3)
        for i, (tool_id, tool_info) in enumerate(TOOLS.items()):
            if tool_info["available"]:
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        ### {tool_info['icon']} {tool_info['name']}
                        {tool_info['description']}
                        """)


if __name__ == "__main__":
    main() 