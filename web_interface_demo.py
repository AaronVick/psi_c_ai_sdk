#!/usr/bin/env python3
"""
ΨC Schema Integration Demo - Web Interface
------------------------------------------
Interactive web interface for the ΨC-AI SDK demonstration system.
This module creates a Streamlit-based UI that visualizes schema evolution,
coherence metrics, and memory integration in real-time.
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import shutil
import uuid

# Add the current directory and parent directory to the Python path for imports
script_dir = Path(__file__).parent.absolute()
sys.path.append(str(script_dir))
parent_dir = script_dir.parent.absolute()
sys.path.append(str(parent_dir))

# Try different import approaches to handle various directory structures
try:
    from demo_runner import DemoRunner
except ImportError:
    # If demo_runner.py is not in the current path, try alternative paths
    if (script_dir / "demo_runner.py").exists():
        sys.path.insert(0, str(script_dir))
    elif (parent_dir / "demo_runner.py").exists():
        sys.path.insert(0, str(parent_dir))
    elif (script_dir / "demo" / "demo_runner.py").exists():
        sys.path.insert(0, str(script_dir / "demo"))
    
    # Try import again
    from demo_runner import DemoRunner

# Configure page settings
st.set_page_config(
    page_title="ΨC Schema Integration Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define constants
PROFILES = ["default", "healthcare", "legal", "narrative"]
GRAPH_HEIGHT = 400
PLOT_HEIGHT = 250
UPDATE_INTERVAL = 1  # seconds
STORAGE_DIR = Path("demo_data/history")

# Define custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .reflection-container {
        background-color: rgba(240, 242, 246, 0.1);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .memory-entry {
        padding: 0.5rem;
        border-left: 3px solid #4CAF50;
        margin-bottom: 0.5rem;
    }
    .contradiction {
        color: #FF5252;
    }
    .coherence-increase {
        color: #4CAF50;
    }
    .coherence-decrease {
        color: #FF5252;
    }
    .phase-transition {
        font-weight: bold;
        color: #FF9800;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.5; }
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
    }
    .badge-success {
        background-color: #4CAF50;
        color: white;
    }
    .badge-warning {
        background-color: #FF9800;
        color: white;
    }
    .badge-danger {
        background-color: #FF5252;
        color: white;
    }
    .info-box {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 3px solid #2196F3;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stExpander {
        border: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialization functions
def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Set default profile if not already set
        if 'profile' not in st.session_state:
            st.session_state.profile = "default"
        
        # Initialize demo runner with current profile
        st.session_state.demo_runner = create_demo_runner(st.session_state.profile)
        
        # Create a new session ID for tracking
        st.session_state.current_session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        st.session_state.session_created_at = datetime.datetime.now().isoformat()
        
        # Initialize UI state variables
        st.session_state.message = ""
        st.session_state.response = ""
        st.session_state.reflection_log = []
        st.session_state.show_help = True
        st.session_state.show_graph = True
        st.session_state.show_metrics = True
        st.session_state.dark_mode = True
        
        # Initialize metrics tracking
        st.session_state.coherence_history = [0.75]
        st.session_state.schema_complexity = [1]
        st.session_state.performance_history = []
        st.session_state.contradiction_rate = 0.0
        st.session_state.avg_memory_time = 150.0
        st.session_state.avg_query_time = 200.0
        st.session_state.last_action = None
        st.session_state.previous_schema_state = None
        st.session_state.delta_concepts = None
        st.session_state.delta_connections = None
        
        # Initialize session history management
        st.session_state.available_sessions = {}
        st.session_state.auto_save_enabled = True
        
        # Apply dark mode by default
        toggle_dark_mode()
        
        # Save initial state
        save_agent_state(st.session_state.profile, st.session_state.current_session_id)

def change_profile():
    """Change the demo profile and reinitialize the demo runner."""
    # Get the selected profile
    new_profile = st.session_state.profile_selector
    
    # Only take action if profile is actually changing
    if new_profile != st.session_state.profile:
        # Archive the current session before changing
        archive_current_session(f"Profile switch from {st.session_state.profile} to {new_profile}")
        
        # Update profile
        st.session_state.profile = new_profile
        
        # Create a new session ID
        st.session_state.current_session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        st.session_state.session_created_at = datetime.datetime.now().isoformat()
        
        # Reinitialize with new profile
        st.session_state.demo_runner = create_demo_runner(new_profile)
        st.session_state.reflection_log = []
        st.session_state.response = ""
        
        # Reset metrics
        st.session_state.coherence_history = [0.75]
        st.session_state.schema_complexity = [1]
        st.session_state.performance_history = []
        
        # Reset delta tracking
        st.session_state.previous_schema_state = None
        st.session_state.delta_concepts = None
        st.session_state.delta_connections = None
        
        # Save the new initial state
        save_agent_state(new_profile, st.session_state.current_session_id)
        
        # Show success message
        st.session_state.message = f"Profile changed to {new_profile}"

def reset_demo():
    """Reset the demo to initial state while preserving history."""
    # Archive the current session
    archive_current_session("Manual reset")
    
    # Keep some preferences
    dark_mode = st.session_state.get("dark_mode", True)
    profile = st.session_state.get("profile", "default")
    
    # Remove session state variables
    for key in list(st.session_state.keys()):
        if key not in ["dark_mode", "profile"]:
            del st.session_state[key]
    
    # Re-initialize session state
    st.session_state.initialized = False
    init_session_state()
    
    # Show success message
    st.session_state.message = "Demo has been reset to initial state."

def toggle_graph():
    """Toggle schema graph visibility."""
    st.session_state.show_graph = not st.session_state.show_graph


def toggle_metrics():
    """Toggle metrics visibility."""
    st.session_state.show_metrics = not st.session_state.show_metrics


def toggle_dark_mode():
    """Toggle between dark and light mode."""
    if st.session_state.dark_mode:
        # Dark mode styles
        dark_mode_css = """
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        .main-header {
            color: #90CAF9;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            border-bottom: 1px solid #555555;
            padding-bottom: 8px;
        }
        .sub-header {
            color: #81C784;
            font-size: 18px;
            margin-top: 15px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #424242;
            color: #E0E0E0;
            border: 1px solid #666666;
        }
        .stButton>button:hover {
            background-color: #555555;
            border: 1px solid #999999;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: #E0E0E0;
        }
        .stTextArea>div>div>textarea {
            background-color: #333333;
            color: #E0E0E0;
        }
        </style>
        """
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        # Light mode styles
        light_mode_css = """
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #212121;
        }
        .main-header {
            color: #1976D2;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            border-bottom: 1px solid #DDDDDD;
            padding-bottom: 8px;
        }
        .sub-header {
            color: #388E3C;
            font-size: 18px;
            margin-top: 15px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #E0E0E0;
            color: #212121;
            border: 1px solid #CCCCCC;
        }
        .stButton>button:hover {
            background-color: #EEEEEE;
            border: 1px solid #999999;
        }
        </style>
        """
        st.markdown(light_mode_css, unsafe_allow_html=True)


def toggle_llm():
    """Toggle LLM integration."""
    # Only allow toggling if LLM bridge is available
    if hasattr(st.session_state.demo_runner, "llm_bridge") and st.session_state.demo_runner.llm_bridge is not None:
        # Update config to enable/disable LLM
        config = st.session_state.demo_runner._load_config()
        config["use_llm"] = not config.get("use_llm", False)
        
        # Save config and reinitialize demo runner
        script_dir = Path(__file__).parent.absolute()
        config_dirs = [
            script_dir / "demo_config",                 # If running from same directory
            script_dir.parent / "demo_config",          # If running from parent
            script_dir / "demo" / "demo_config"         # If running from sibling
        ]
        
        # Find the first valid config directory
        config_dir = None
        for potential_dir in config_dirs:
            if potential_dir.exists():
                config_dir = potential_dir
                break
        
        # Create the directory if it doesn't exist
        if config_dir is None:
            config_dir = script_dir / "demo_config"
            config_dir.mkdir(exist_ok=True)
        
        # Save the config file
        try:
            with open(config_dir / f"{st.session_state.profile}_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            st.error(f"Error saving config: {e}")
        
        # Reinitialize the demo runner
        st.session_state.demo_runner = DemoRunner(profile=st.session_state.profile)
        
        # Update LLM enabled flag
        has_llm = hasattr(st.session_state.demo_runner, "llm_bridge") and st.session_state.demo_runner.llm_bridge is not None
        st.session_state.llm_enabled = has_llm and st.session_state.demo_runner.llm_bridge.is_enabled()


# Memory interaction functions
def add_memory(memory_content=None):
    """Add a new memory to the system and log empirical data.
    
    Args:
        memory_content: The memory content to add
        
    Returns:
        Boolean indicating success
    """
    if memory_content is None:
        memory_content = st.session_state.get("memory_input", "")
    
    if not memory_content:
        return False
    
    # Record start time for performance tracking
    start_time = time.time()
    
    try:
        # Capture schema state before adding memory for comparison
        before_schema = st.session_state.demo_runner.get_schema_graph_data()
        before_coherence = st.session_state.demo_runner.get_coherence_score()
        before_node_count = len(before_schema["nodes"])
        before_edge_count = len(before_schema["edges"])
        
        # Add the memory using the demo runner
        result = st.session_state.demo_runner.add_memory(memory_content)
        
        # Compute processing time
        memory_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Capture schema state after adding memory
        after_schema = st.session_state.demo_runner.get_schema_graph_data()
        after_coherence = st.session_state.demo_runner.get_coherence_score()
        after_node_count = len(after_schema["nodes"])
        after_edge_count = len(after_schema["edges"])
        
        # Calculate changes
        delta_coherence = after_coherence - before_coherence
        delta_nodes = after_node_count - before_node_count
        delta_edges = after_edge_count - before_edge_count
        
        # Update metrics
        if 'avg_memory_time' in st.session_state:
            # Calculate moving average
            st.session_state.avg_memory_time = 0.7 * st.session_state.avg_memory_time + 0.3 * memory_time
        else:
            st.session_state.avg_memory_time = memory_time
        
        # Add to performance history
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
        
        st.session_state.performance_history.append({
            'memory_time': memory_time,
            'query_time': st.session_state.get('avg_query_time', 0)
        })
        
        # Update coherence score
        coherence = st.session_state.demo_runner.get_coherence_score()
        if 'coherence_history' not in st.session_state:
            st.session_state.coherence_history = []
        st.session_state.coherence_history.append(coherence)
        
        # Update schema complexity
        node_count = len(st.session_state.demo_runner.schema.get_all_nodes())
        edge_count = len(st.session_state.demo_runner.schema.get_all_edges())
        complexity = node_count + edge_count
        
        if 'schema_complexity' not in st.session_state:
            st.session_state.schema_complexity = []
        st.session_state.schema_complexity.append(complexity)
        
        # Update contradiction rate
        contradiction_count = st.session_state.demo_runner.get_contradiction_count()
        if node_count > 0:
            st.session_state.contradiction_rate = (contradiction_count / node_count) * 100
        else:
            st.session_state.contradiction_rate = 0
        
        # Log detailed empirical data about this memory operation
        memory_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": "add_memory",
            "content": memory_content,
            "processing_time_ms": memory_time,
            "before": {
                "coherence": before_coherence,
                "node_count": before_node_count,
                "edge_count": before_edge_count
            },
            "after": {
                "coherence": after_coherence,
                "node_count": after_node_count,
                "edge_count": after_edge_count
            },
            "delta": {
                "coherence": delta_coherence,
                "nodes": delta_nodes,
                "edges": delta_edges
            },
            "contradictions": contradiction_count,
            "contradiction_rate": st.session_state.contradiction_rate
        }
        
        # Add to reflection log for UI display
        if 'detailed_log' not in st.session_state:
            st.session_state.detailed_log = []
        st.session_state.detailed_log.append(memory_log)
        
        # Update previous schema state for diff visualization
        st.session_state.previous_schema_state = before_schema
        st.session_state.delta_concepts = delta_nodes
        st.session_state.delta_connections = delta_edges
        
        return True
    except Exception as e:
        st.error(f"Error adding memory: {str(e)}")
        # Log the error for empirical analysis
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": "add_memory",
            "content": memory_content,
            "error": str(e)
        })
        return False

def process_query(query_content=None):
    """Process a query, update visualizations, and log empirical data.
    
    Args:
        query_content: The query to process
        
    Returns:
        Boolean indicating success
    """
    if query_content is None:
        query_content = st.session_state.get("query_input", "")
    
    if not query_content:
        return False
    
    # Record start time for performance tracking
    start_time = time.time()
    
    try:
        # Capture schema state before query for comparison
        before_schema = st.session_state.demo_runner.get_schema_graph_data()
        before_coherence = st.session_state.demo_runner.get_coherence_score()
        before_node_count = len(before_schema["nodes"])
        before_edge_count = len(before_schema["edges"])
        
        # Process the query
        response = st.session_state.demo_runner.process_query(query_content)
        
        # Store the response in the session state
        st.session_state.response = response
        
        # Compute processing time
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Capture schema state after query
        after_schema = st.session_state.demo_runner.get_schema_graph_data()
        after_coherence = st.session_state.demo_runner.get_coherence_score()
        after_node_count = len(after_schema["nodes"])
        after_edge_count = len(after_schema["edges"])
        
        # Calculate changes
        delta_coherence = after_coherence - before_coherence
        delta_nodes = after_node_count - before_node_count
        delta_edges = after_edge_count - before_edge_count
        
        # Update metrics
        if 'avg_query_time' in st.session_state:
            # Calculate moving average
            st.session_state.avg_query_time = 0.7 * st.session_state.avg_query_time + 0.3 * query_time
        else:
            st.session_state.avg_query_time = query_time
        
        # Add to performance history
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
        
        st.session_state.performance_history.append({
            'memory_time': st.session_state.get('avg_memory_time', 0),
            'query_time': query_time
        })
        
        # Update coherence score
        coherence = st.session_state.demo_runner.get_coherence_score()
        if 'coherence_history' not in st.session_state:
            st.session_state.coherence_history = []
        st.session_state.coherence_history.append(coherence)
        
        # Update schema complexity
        node_count = len(st.session_state.demo_runner.schema.get_all_nodes())
        edge_count = len(st.session_state.demo_runner.schema.get_all_edges())
        complexity = node_count + edge_count
        
        if 'schema_complexity' not in st.session_state:
            st.session_state.schema_complexity = []
        st.session_state.schema_complexity.append(complexity)
        
        # Calculate contradiction rate
        contradiction_count = st.session_state.demo_runner.get_contradiction_count()
        if node_count > 0:
            st.session_state.contradiction_rate = (contradiction_count / node_count) * 100
        else:
            st.session_state.contradiction_rate = 0
        
        # Log detailed empirical data about this query operation
        query_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": "process_query",
            "query": query_content,
            "response": response,
            "processing_time_ms": query_time,
            "before": {
                "coherence": before_coherence,
                "node_count": before_node_count,
                "edge_count": before_edge_count
            },
            "after": {
                "coherence": after_coherence,
                "node_count": after_node_count,
                "edge_count": after_edge_count
            },
            "delta": {
                "coherence": delta_coherence,
                "nodes": delta_nodes,
                "edges": delta_edges
            },
            "contradictions": contradiction_count,
            "contradiction_rate": st.session_state.contradiction_rate
        }
        
        # Add to detailed log for empirical analysis
        if 'detailed_log' not in st.session_state:
            st.session_state.detailed_log = []
        st.session_state.detailed_log.append(query_log)
        
        # Update previous schema state for diff visualization
        st.session_state.previous_schema_state = before_schema
        st.session_state.delta_concepts = delta_nodes
        st.session_state.delta_connections = delta_edges
        
        return True
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        # Log the error for empirical analysis
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": "process_query",
            "query": query_content,
            "error": str(e)
        })
        return False

def export_summary():
    """Export session summary with empirical data."""
    format_type = st.session_state.export_format
    
    # Collect all data for the export
    export_data = {
        "session_id": st.session_state.current_session_id,
        "profile": st.session_state.profile,
        "created_at": st.session_state.session_created_at,
        "exported_at": datetime.datetime.now().isoformat(),
        "schema": st.session_state.demo_runner.get_schema_graph_data(),
        "metrics": {
            "coherence_history": st.session_state.coherence_history,
            "schema_complexity": st.session_state.schema_complexity,
            "performance_history": st.session_state.performance_history,
            "contradiction_rate": st.session_state.contradiction_rate
        },
        "memory_count": len(st.session_state.demo_runner.memory_store.get_all_memories()) if hasattr(st.session_state.demo_runner, 'memory_store') else 0,
        "reflection_log": st.session_state.reflection_log,
        "detailed_log": st.session_state.get("detailed_log", []),
        "error_log": st.session_state.get("error_log", [])
    }
    
    # Generate summary based on format
    if format_type == "json":
        # For JSON, simply dump the data
        summary = json.dumps(export_data, indent=2)
        filename = f"psi_c_demo_{st.session_state.profile}_{st.session_state.current_session_id}.json"
        st.download_button(
            label="Download JSON",
            data=summary,
            file_name=filename,
            mime="application/json"
        )
    elif format_type == "markdown":
        # For Markdown, format the data more readably
        memories = st.session_state.demo_runner.memory_store.get_all_memories() if hasattr(st.session_state.demo_runner, 'memory_store') else []
        
        md_content = [
            f"# ΨC Demo Session Summary",
            f"## Session Information",
            f"- **Profile:** {st.session_state.profile}",
            f"- **Session ID:** {st.session_state.current_session_id}",
            f"- **Created:** {st.session_state.session_created_at}",
            f"- **Exported:** {datetime.datetime.now().isoformat()}",
            f"",
            f"## Memories ({len(memories)})",
        ]
        
        # Add memories
        for i, memory in enumerate(memories):
            md_content.append(f"{i+1}. {memory.content}")
        
        md_content.extend([
            f"",
            f"## Schema Statistics",
            f"- **Nodes:** {len(export_data['schema']['nodes'])}",
            f"- **Edges:** {len(export_data['schema']['edges'])}",
            f"- **Coherence:** {export_data['metrics']['coherence_history'][-1] if export_data['metrics']['coherence_history'] else 'N/A'}",
            f"- **Complexity:** {export_data['metrics']['schema_complexity'][-1] if export_data['metrics']['schema_complexity'] else 'N/A'}",
            f"- **Contradiction Rate:** {export_data['contradiction_rate']}%",
            f"",
            f"## Interaction Log",
        ])
        
        # Add detailed log entries
        for entry in export_data.get("detailed_log", []):
            if entry["operation"] == "add_memory":
                md_content.append(f"### Memory Added: {entry['timestamp']}")
                md_content.append(f"Content: \"{entry['content']}\"")
                md_content.append(f"Processing Time: {entry['processing_time_ms']:.2f}ms")
                md_content.append(f"Coherence Change: {entry['before']['coherence']:.4f} → {entry['after']['coherence']:.4f} ({'+' if entry['delta']['coherence'] >= 0 else ''}{entry['delta']['coherence']:.4f})")
                md_content.append(f"Schema Changes: +{entry['delta']['nodes']} nodes, +{entry['delta']['edges']} edges")
                md_content.append("")
            elif entry["operation"] == "process_query":
                md_content.append(f"### Query Processed: {entry['timestamp']}")
                md_content.append(f"Query: \"{entry['query']}\"")
                md_content.append(f"Response: \"{entry['response']}\"")
                md_content.append(f"Processing Time: {entry['processing_time_ms']:.2f}ms")
                md_content.append(f"Coherence Change: {entry['before']['coherence']:.4f} → {entry['after']['coherence']:.4f} ({'+' if entry['delta']['coherence'] >= 0 else ''}{entry['delta']['coherence']:.4f})")
                md_content.append(f"Schema Changes: +{entry['delta']['nodes']} nodes, +{entry['delta']['edges']} edges")
                md_content.append("")
        
        # Join all lines into a single string
        summary = "\n".join(md_content)
        filename = f"psi_c_demo_{st.session_state.profile}_{st.session_state.current_session_id}.md"
        st.download_button(
            label="Download Markdown",
            data=summary,
            file_name=filename,
            mime="text/markdown"
        )


# Visualization functions
def render_graph():
    """Render the schema graph visualization with improved explanations."""
    if st.session_state.show_graph:
        st.subheader("Cognitive Schema Visualization")
        
        if st.session_state.show_help:
            st.markdown("""
            **Schema Graph Explanation:**
            This visualization represents the agent's current knowledge structure.
            - **Nodes** represent concepts or entities
            - **Edges** show relationships between concepts
            - **Colors** indicate different concept categories
            - **Size** reflects concept importance
            
            Watch how this structure evolves as new memories are added!
            """)
        
        # Create tabs for different graph views
        graph_tabs = st.tabs(["Full Schema", "Core Concepts", "Recent Changes"])
        
        with graph_tabs[0]:
            st.write("Complete cognitive schema representation")
            fig = create_schema_graph()
            st.pyplot(fig)
        
        with graph_tabs[1]:
            st.write("Most important concepts and their relationships")
            fig = create_schema_graph(core_only=True)
            st.pyplot(fig)
            
        with graph_tabs[2]:
            st.write("Recently modified parts of the schema")
            if st.session_state.last_action in ["add_memory", "process_query"]:
                fig = create_schema_diff()
                st.pyplot(fig)
            else:
                st.info("Add a memory or process a query to see changes")
        
        # Add schema metrics below the graph
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Concepts", 
                value=len(st.session_state.demo_runner.schema.get_all_nodes()),
                delta=st.session_state.delta_concepts if hasattr(st.session_state, 'delta_concepts') else None,
                help="Total number of distinct concepts in the schema"
            )
        with col2:
            st.metric(
                label="Connections", 
                value=len(st.session_state.demo_runner.schema.get_all_edges()),
                delta=st.session_state.delta_connections if hasattr(st.session_state, 'delta_connections') else None,
                help="Total number of relationships between concepts"
            )


def render_coherence_plot():
    """Render the coherence history plot."""
    coherence_history = st.session_state.demo_runner.get_coherence_history()
    
    if not coherence_history:
        st.info("No coherence data available yet.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Plot coherence
    x = list(range(len(coherence_history)))
    ax.plot(x, coherence_history, marker="o", linestyle="-", color="#2196F3", label="Coherence (ΨC)")
    
    # Add phase transition markers if available
    session_log = st.session_state.demo_runner.session_log
    for i, entry in enumerate(session_log):
        if entry.get("phase_transition", False) and i < len(coherence_history):
            ax.axvline(x=i, color="#FF9800", linestyle="--", alpha=0.5)
            ax.plot(i, coherence_history[i], marker="*", color="#FF9800", markersize=12)
    
    ax.set_xlabel("Memory Updates")
    ax.set_ylabel("Coherence (ΨC)")
    ax.set_title("Coherence Evolution")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Only show integer ticks on x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    st.pyplot(fig)


def render_entropy_plot():
    """Render the entropy history plot."""
    entropy_history = st.session_state.demo_runner.get_entropy_history()
    
    if not entropy_history:
        st.info("No entropy data available yet.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Plot entropy
    x = list(range(len(entropy_history)))
    ax.plot(x, entropy_history, marker="o", linestyle="-", color="#FF5252", label="Entropy (H)")
    
    # Add phase transition markers if available
    session_log = st.session_state.demo_runner.session_log
    for i, entry in enumerate(session_log):
        if entry.get("phase_transition", False) and i < len(entropy_history):
            ax.axvline(x=i, color="#FF9800", linestyle="--", alpha=0.5)
            ax.plot(i, entropy_history[i], marker="*", color="#FF9800", markersize=12)
    
    ax.set_xlabel("Memory Updates")
    ax.set_ylabel("Entropy (H)")
    ax.set_title("Entropy Evolution")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Only show integer ticks on x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    st.pyplot(fig)


def render_reflection_log():
    """Render the reflection log with formatted entries."""
    if not st.session_state.reflection_log:
        st.info("No memories processed yet. Add a memory to see reflection results.")
        return
    
    for entry in reversed(st.session_state.reflection_log):
        # Handle query-response pairs
        if entry.get("is_query", False):
            with st.container():
                st.markdown(f"**{entry['timestamp']}** - 🔍 **Query:** {entry['query']}")
                st.markdown(f"🤖 **Response:** {entry['response']}")
                st.markdown("---")
            continue
        
        # Handle regular memory entries
        with st.container():
            result = entry["result"]
            
            # Memory content
            st.markdown(f"**{entry['timestamp']}** - 🧠 **Memory Added:** \"{entry['content']}\"")
            
            # Reflection result
            if "enhanced_reflection" in result and st.session_state.llm_enabled:
                st.markdown(f"🔄 **Reflection:** {result['enhanced_reflection']}")
            else:
                st.markdown(f"🔄 **Reflection:** Memory processed and analyzed.")
            
            # Contradictions
            if result["contradictions"] > 0:
                st.markdown(f"⚠️ **Contradictions Detected:** {result['contradictions']}")
            
            # Schema update
            if result["schema_updated"]:
                st.markdown("📐 **Schema Updated** to resolve contradictions")
            
            # Coherence change
            coherence_change = result["coherence_change"]
            if coherence_change > 0:
                st.markdown(f"📈 **Coherence:** {entry['coherence']-coherence_change:.4f} → {entry['coherence']:.4f} <span class='coherence-increase'>(+{coherence_change:.4f})</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"📉 **Coherence:** {entry['coherence']-coherence_change:.4f} → {entry['coherence']:.4f} <span class='coherence-decrease'>({coherence_change:.4f})</span>", unsafe_allow_html=True)
            
            # Entropy change
            entropy_change = result["entropy_change"]
            st.markdown(f"🔍 **Entropy:** {entry['entropy']-entropy_change:.4f} → {entry['entropy']:.4f} ({'+' if entropy_change > 0 else ''}{entropy_change:.4f})")
            
            # Phase transition
            if result["phase_transition"]:
                st.markdown("<span class='phase-transition'>🌪 PHASE TRANSITION DETECTED: Significant cognitive restructuring occurred</span>", unsafe_allow_html=True)
            
            # Change summary if available
            if "change_summary" in result and st.session_state.llm_enabled:
                st.markdown(f"📋 **Summary:** {result['change_summary']}")
            
            st.markdown("---")


def render_system_metrics():
    """Render the system metrics dashboard."""
    demo = st.session_state.demo_runner
    
    # Get current metrics
    coherence = demo.get_current_coherence()
    entropy = demo.get_current_entropy()
    memory_count = len(demo.memory_store.get_all_memories())
    node_count = len(demo.schema_graph.nodes)
    edge_count = len(demo.schema_graph.edges)
    schema_version = demo.schema_version
    
    # Create metrics container
    cols = st.columns(3)
    
    # Coherence
    with cols[0]:
        coherence_color = "#4CAF50" if coherence > 0.6 else "#FF9800" if coherence > 0.3 else "#FF5252"
        st.metric(
            label="Coherence (ΨC)",
            value=f"{coherence:.4f}",
            delta=None,
            delta_color="normal"
        )
        st.progress(min(coherence, 1.0))
    
    # Entropy
    with cols[1]:
        entropy_color = "#4CAF50" if entropy < 0.3 else "#FF9800" if entropy < 0.6 else "#FF5252"
        st.metric(
            label="Entropy (H)",
            value=f"{entropy:.4f}",
            delta=None,
            delta_color="normal"
        )
        st.progress(min(entropy, 1.0))
    
    # Schema version
    with cols[2]:
        st.metric(
            label="Schema Version",
            value=schema_version,
            delta=None,
            delta_color="normal"
        )
    
    # Create second row
    cols = st.columns(3)
    
    # Memory count
    with cols[0]:
        st.metric(
            label="Memories",
            value=memory_count,
            delta=None,
            delta_color="normal"
        )
    
    # Node count
    with cols[1]:
        st.metric(
            label="Schema Nodes",
            value=node_count,
            delta=None,
            delta_color="normal"
        )
    
    # Edge count
    with cols[2]:
        st.metric(
            label="Schema Edges",
            value=edge_count,
            delta=None,
            delta_color="normal"
        )

def render_sidebar():
    """Render the sidebar with controls and settings."""
    st.sidebar.title("ΨC-AI Demo Controls")
    
    # Add helpful description
    st.sidebar.markdown("""
    This interface demonstrates the ΨC-AI cognitive architecture in action.
    Add memories, ask questions, and observe how the schema evolves in real-time.
    """)
    
    # Profile selector with proper isolation
    st.sidebar.subheader("Profile Selection")
    
    # Get list of available profiles
    profiles = PROFILES
    
    # Current profile
    current_profile = st.session_state.get("profile", "default")
    
    # Profile selector
    selected_profile = st.sidebar.selectbox(
        "Active Profile",
        options=profiles,
        index=profiles.index(current_profile),
        key="profile_selector",
        help="Select a domain-specific profile for the agent"
    )
    
    # Change profile button
    if selected_profile != current_profile:
        if st.sidebar.button("Switch Profile", help="Save current state and switch to the selected profile"):
            change_profile()
            st.experimental_rerun()
    
    # Session Management
    st.sidebar.subheader("Session Management")
    
    # Update available session list
    available_sessions = get_profile_sessions(current_profile)
    session_labels = [f"{s.get('archive_name', s.get('created_at', 'Unknown'))} ({s.get('memory_count', 0)} memories)" 
                      for s in available_sessions]
    
    if session_labels:
        # Session selector
        selected_session_idx = st.sidebar.selectbox(
            "Saved Sessions",
            options=range(len(session_labels)),
            format_func=lambda i: session_labels[i],
            help="Select a previously saved session to load"
        )
        
        # Load selected session
        if st.sidebar.button("Load Session", help="Load the selected session"):
            session_id = available_sessions[selected_session_idx]["session_id"]
            if load_agent_state(current_profile, session_id):
                st.session_state.message = f"Session loaded successfully"
                st.experimental_rerun()
    else:
        st.sidebar.info("No saved sessions found for this profile")
    
    # Auto-save toggle
    auto_save_enabled = st.sidebar.checkbox(
        "Auto-save Sessions", 
        value=st.session_state.get("auto_save_enabled", True),
        help="Automatically save sessions after each interaction"
    )
    if auto_save_enabled != st.session_state.get("auto_save_enabled", True):
        st.session_state.auto_save_enabled = auto_save_enabled
    
    # Memory Controls Section with tooltips
    st.sidebar.subheader("Memory Management")
    
    memory_input = st.sidebar.text_area(
        "Add New Memory", 
        help="Enter a statement that will be integrated into the agent's memory system"
    )
    
    if st.sidebar.button("Submit Memory", help="Click to add this memory to the cognitive system"):
        if memory_input:
            add_memory(memory_input)
            st.session_state.reflection_log.append(f"Memory added: {memory_input}")
            st.session_state.message = "Memory successfully integrated."
            st.session_state.last_action = "add_memory"
            
            # Auto-save after adding memory
            if st.session_state.auto_save_enabled:
                auto_save()
                
            st.experimental_rerun()
        else:
            st.sidebar.warning("Please enter a memory first.")
    
    # Query Controls with better labeling
    st.sidebar.subheader("Query System")
    
    query_input = st.sidebar.text_area(
        "Ask a Question", 
        help="Enter a question for the agent to answer based on its current knowledge"
    )
    
    if st.sidebar.button("Submit Question", help="Process this question through the cognitive system"):
        if query_input:
            process_query(query_input)
            st.session_state.reflection_log.append(f"Query processed: {query_input}")
            st.session_state.message = "Query successfully processed."
            st.session_state.last_action = "process_query"
            
            # Auto-save after processing query
            if st.session_state.auto_save_enabled:
                auto_save()
                
            st.experimental_rerun()
        else:
            st.sidebar.warning("Please enter a query first.")
    
    # Visualization Controls with clearer labels
    st.sidebar.subheader("Visualization Options")
    
    show_help = st.sidebar.checkbox(
        "Show Help Text", 
        value=st.session_state.show_help,
        help="Toggle explanatory text for each visualization"
    )
    if show_help != st.session_state.show_help:
        st.session_state.show_help = show_help
    
    show_graph = st.sidebar.checkbox(
        "Show Schema Graph", 
        value=st.session_state.show_graph,
        help="Display the current cognitive schema as a network graph"
    )
    if show_graph != st.session_state.show_graph:
        st.session_state.show_graph = show_graph
    
    show_metrics = st.sidebar.checkbox(
        "Show Metrics", 
        value=st.session_state.show_metrics,
        help="Display coherence and stability metrics over time"
    )
    if show_metrics != st.session_state.show_metrics:
        st.session_state.show_metrics = show_metrics
    
    # Display Mode with better labeling
    st.sidebar.subheader("Display Settings")
    
    dark_mode = st.sidebar.checkbox(
        "Dark Mode", 
        value=st.session_state.dark_mode,
        help="Toggle between light and dark theme"
    )
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        toggle_dark_mode()
    
    # System Controls with clearer actions
    st.sidebar.subheader("System Controls")
    
    # Manual save button
    if st.sidebar.button("Save Current State", help="Manually save the current state"):
        session_id = save_agent_state(st.session_state.profile, st.session_state.current_session_id)
        if session_id:
            st.session_state.message = f"Session saved successfully"
            st.experimental_rerun()
    
    # Reset demo
    if st.sidebar.button("Reset Demo", help="Archive current state and reset the system"):
        reset_demo()
        st.experimental_rerun()
    
    # Export system data
    export_format = st.sidebar.radio(
        "Export Format",
        options=["json", "markdown"],
        horizontal=True,
        key="export_format"
    )
    
    if st.sidebar.button("Export Summary", help="Export the current session data"):
        export_summary()
    
    # Add version information
    st.sidebar.markdown("---")
    st.sidebar.info(f"ΨC-AI SDK Demo v1.0 | Profile: {current_profile}")

# Main application
def main():
    """Main application function."""
    # Set page config
    st.set_page_config(
        page_title="ΨC-AI Demo Interface",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Header
    with st.container():
        st.markdown("<div class='main-header'>ΨC-AI Cognitive Architecture Demo</div>", unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Main content area with tabs
    with st.container():
        # Display message if present
        if st.session_state.message:
            st.success(st.session_state.message)
            st.session_state.message = ""
        
        # Create tabs for different views
        main_tabs = st.tabs(["Interaction", "Memory & Schema", "System Metrics", "Reflection Log"])
        
        with main_tabs[0]:
            st.subheader("Interact with the Cognitive System")
            
            if st.session_state.show_help:
                st.info("""
                This is where you can directly interact with the cognitive system.
                Add new memories or ask questions to see how the system processes information.
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                memory_input = st.text_area(
                    "Add New Memory", 
                    height=100,
                    help="Enter information that the agent should remember"
                )
                
                if st.button("Submit Memory", help="Add this memory to the cognitive system"):
                    if memory_input:
                        add_memory(memory_input)
                        st.session_state.reflection_log.append(f"Memory added: {memory_input}")
                        st.session_state.message = "Memory successfully integrated."
                        st.session_state.last_action = "add_memory"
                        st.experimental_rerun()
                    else:
                        st.warning("Please enter a memory first.")
            
            with col2:
                query_input = st.text_area(
                    "Ask a Question", 
                    height=100,
                    help="Ask a question based on the agent's existing knowledge"
                )
                
                if st.button("Submit Question", help="Process this question through the cognitive system"):
                    if query_input:
                        process_query(query_input)
                        st.session_state.reflection_log.append(f"Query processed: {query_input}")
                        st.session_state.message = "Query successfully processed."
                        st.session_state.last_action = "process_query"
                        st.experimental_rerun()
                    else:
                        st.warning("Please enter a query first.")
            
            # Display system response
            if st.session_state.response:
                st.markdown("### System Response")
                st.markdown(st.session_state.response)
        
        with main_tabs[1]:
            st.subheader("Memory & Schema")
            
            if st.session_state.show_help:
                st.info("""
                This tab shows the agent's memories and knowledge structure.
                The schema graph visualizes how concepts are connected.
                """)
            
            # Display memories
            with st.expander("Stored Memories", expanded=True):
                if hasattr(st.session_state.demo_runner, 'memory_store') and st.session_state.demo_runner.memory_store:
                    memories = st.session_state.demo_runner.memory_store.get_all_memories()
                    if memories:
                        for i, memory in enumerate(memories):
                            st.markdown(f"**Memory {i+1}:** {memory.content}")
                    else:
                        st.info("No memories stored yet. Add some using the input above.")
                else:
                    st.info("Memory store not initialized.")
            
            # Display schema visualization
            render_graph()
        
        with main_tabs[2]:
            st.subheader("System Metrics")
            
            if st.session_state.show_help:
                st.info("""
                This tab displays metrics about the cognitive system's performance.
                Monitor coherence, stability, and other key indicators over time.
                """)
            
            render_metrics()
        
        with main_tabs[3]:
            st.subheader("Reflection Log")
            
            if st.session_state.show_help:
                st.info("""
                This tab shows the agent's internal reflection process.
                See how it reasons about new information and resolves contradictions.
                """)
            
            if st.session_state.reflection_log:
                for entry in st.session_state.reflection_log:
                    st.markdown(f"- {entry}")
            else:
                st.info("No reflection entries yet. Add memories or process queries to see reflections.")

def create_schema_graph(core_only=False):
    """Create and return a schema graph visualization figure.
    
    Args:
        core_only (bool): If True, only show core concepts with high importance.
    
    Returns:
        matplotlib.figure.Figure: The generated graph figure
    """
    graph_data = st.session_state.demo_runner.get_schema_graph_data()
    
    if not graph_data["nodes"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Schema graph is empty. Add memories to build the schema.", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in graph_data["nodes"]:
        # Skip non-core nodes if core_only is True
        if core_only and node.get("importance", 0.5) < 0.7:
            continue
            
        G.add_node(
            node["id"], 
            label=node["label"], 
            type=node["type"],
            importance=node["importance"]
        )
    
    # Add edges with attributes
    for edge in graph_data["edges"]:
        # Only add edges between nodes that exist in the graph
        if edge["source"] in G and edge["target"] in G:
            G.add_edge(
                edge["source"], 
                edge["target"], 
                weight=edge["weight"],
                type=edge["type"]
            )
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get node colors based on type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "concept")
        if node_type == "memory":
            node_colors.append("#4CAF50")  # Green for memories
        elif node_type == "contradiction":
            node_colors.append("#FF5252")  # Red for contradictions
        elif node_type == "belief":
            node_colors.append("#FFC107")  # Yellow for beliefs
        else:
            node_colors.append("#2196F3")  # Blue for concepts
    
    # Get node sizes based on importance
    node_sizes = []
    for node in G.nodes():
        importance = G.nodes[node].get("importance", 0.5)
        node_sizes.append(100 + importance * 300)
    
    # Get edge widths based on weight
    edge_widths = []
    for u, v in G.edges():
        weight = G.edges[u, v].get("weight", 0.5)
        edge_widths.append(weight * 2)
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="#9E9E9E")
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    plt.axis("off")
    return fig

def create_schema_diff():
    """Create and return a visualization of recent schema changes.
    
    Returns:
        matplotlib.figure.Figure: The generated graph figure showing changes
    """
    if not hasattr(st.session_state, 'previous_schema_state'):
        st.session_state.previous_schema_state = None
    
    current_graph_data = st.session_state.demo_runner.get_schema_graph_data()
    
    # If we don't have a previous state, save current and return empty
    if st.session_state.previous_schema_state is None:
        st.session_state.previous_schema_state = current_graph_data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No previous state to compare with. Run another operation to see changes.", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig
    
    # Identify new nodes and changed nodes
    previous_nodes = {node["id"]: node for node in st.session_state.previous_schema_state["nodes"]}
    current_nodes = {node["id"]: node for node in current_graph_data["nodes"]}
    
    new_node_ids = set(current_nodes.keys()) - set(previous_nodes.keys())
    
    # Identify changed edges
    previous_edges = {(edge["source"], edge["target"]): edge 
                      for edge in st.session_state.previous_schema_state["edges"]}
    current_edges = {(edge["source"], edge["target"]): edge 
                     for edge in current_graph_data["edges"]}
    
    new_edge_pairs = set(current_edges.keys()) - set(previous_edges.keys())
    
    # Create networkx graph for visualizing changes
    G = nx.Graph()
    
    # Add new nodes
    for node_id in new_node_ids:
        node = current_nodes[node_id]
        G.add_node(
            node["id"], 
            label=node["label"], 
            type=node["type"],
            importance=node["importance"],
            is_new=True
        )
    
    # Add nodes connected to new edges
    for source, target in new_edge_pairs:
        if source not in G:
            if source in current_nodes:
                node = current_nodes[source]
                G.add_node(
                    source, 
                    label=node["label"], 
                    type=node["type"],
                    importance=node["importance"],
                    is_new=False
                )
        
        if target not in G:
            if target in current_nodes:
                node = current_nodes[target]
                G.add_node(
                    target, 
                    label=node["label"], 
                    type=node["type"],
                    importance=node["importance"],
                    is_new=False
                )
    
    # Add new edges
    for source, target in new_edge_pairs:
        if source in G and target in G:
            edge = current_edges[(source, target)]
            G.add_edge(
                source, 
                target, 
                weight=edge["weight"],
                type=edge["type"],
                is_new=True
            )
    
    # If graph is empty, return message
    if not G.nodes():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No changes detected in the schema.", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get node colors
    node_colors = []
    for node in G.nodes():
        is_new = G.nodes[node].get("is_new", False)
        node_type = G.nodes[node].get("type", "concept")
        
        if is_new:
            node_colors.append("#FF5722")  # Orange for new nodes
        elif node_type == "memory":
            node_colors.append("#4CAF50")  # Green for memories
        elif node_type == "contradiction":
            node_colors.append("#FF5252")  # Red for contradictions
        else:
            node_colors.append("#2196F3")  # Blue for concepts
    
    # Get node sizes
    node_sizes = []
    for node in G.nodes():
        is_new = G.nodes[node].get("is_new", False)
        importance = G.nodes[node].get("importance", 0.5)
        
        size = 100 + importance * 200
        if is_new:
            size *= 1.5  # Make new nodes larger
        
        node_sizes.append(size)
    
    # Get edge colors and widths
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        is_new = G.edges[u, v].get("is_new", False)
        weight = G.edges[u, v].get("weight", 0.5)
        
        if is_new:
            edge_colors.append("#FF5722")  # Orange for new edges
            edge_widths.append(weight * 3)  # Make new edges wider
        else:
            edge_colors.append("#9E9E9E")  # Grey for existing edges
            edge_widths.append(weight * 1.5)
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    # Add legend
    orange_patch = mpatches.Patch(color="#FF5722", label="Newly Added")
    blue_patch = mpatches.Patch(color="#2196F3", label="Existing Concept")
    green_patch = mpatches.Patch(color="#4CAF50", label="Memory")
    red_patch = mpatches.Patch(color="#FF5252", label="Contradiction")
    
    plt.legend(handles=[orange_patch, blue_patch, green_patch, red_patch], 
               loc="upper right", fontsize=8)
    
    plt.axis("off")
    
    # Update previous state
    st.session_state.previous_schema_state = current_graph_data
    
    # Update delta metrics for changes
    st.session_state.delta_concepts = len(new_node_ids)
    st.session_state.delta_connections = len(new_edge_pairs)
    
    return fig

def render_metrics():
    """Render the metrics visualization with improved explanations."""
    if st.session_state.show_metrics:
        if st.session_state.show_help:
            st.markdown("""
            **Cognitive Metrics Explanation:**
            These metrics show how the cognitive system's internal state changes over time:
            
            - **Coherence Score**: Measures how well the beliefs fit together (higher is better)
            - **Schema Complexity**: Tracks how sophisticated the knowledge structure is becoming
            - **Contradiction Rate**: Shows how many contradictions are detected (lower is better)
            
            These metrics update when you add memories or process queries.
            """)
        
        # Create tabs for different metric views
        metric_tabs = st.tabs(["Coherence Trends", "Schema Evolution", "System Performance"])
        
        with metric_tabs[0]:
            st.write("Coherence score over time")
            
            # Get coherence data
            if 'coherence_history' in st.session_state:
                coherence_data = st.session_state.coherence_history
                x = list(range(len(coherence_data)))
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x, coherence_data, marker='o', linestyle='-', color='#2196F3')
                ax.set_xlabel('Events')
                ax.set_ylabel('Coherence Score')
                ax.set_ylim(0, 1.0)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add annotations for significant changes
                if len(coherence_data) > 1:
                    for i in range(1, len(coherence_data)):
                        if abs(coherence_data[i] - coherence_data[i-1]) > 0.1:
                            ax.annotate(
                                f'Δ {coherence_data[i] - coherence_data[i-1]:.2f}', 
                                xy=(i, coherence_data[i]),
                                xytext=(0, 10),
                                textcoords='offset points',
                                ha='center',
                                fontsize=8,
                                arrowprops=dict(arrowstyle='->', color='gray')
                            )
                
                st.pyplot(fig)
                
                # Current coherence metric
                if coherence_data:
                    current = coherence_data[-1]
                    delta = None
                    if len(coherence_data) > 1:
                        delta = current - coherence_data[-2]
                    
                    st.metric(
                        label="Current Coherence", 
                        value=f"{current:.2f}",
                        delta=f"{delta:.2f}" if delta is not None else None,
                        delta_color="normal",
                        help="Score between 0-1 indicating how well current beliefs fit together"
                    )
            else:
                st.info("No coherence data available yet. Add memories or process queries to generate data.")
        
        with metric_tabs[1]:
            st.write("Schema evolution and complexity")
            
            # Get schema complexity data
            if 'schema_complexity' in st.session_state:
                complexity_data = st.session_state.schema_complexity
                x = list(range(len(complexity_data)))
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x, complexity_data, marker='s', linestyle='-', color='#FF9800')
                ax.set_xlabel('Events')
                ax.set_ylabel('Schema Complexity')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                
                # Current complexity metric
                if complexity_data:
                    current = complexity_data[-1]
                    delta = None
                    if len(complexity_data) > 1:
                        delta = current - complexity_data[-2]
                    
                    st.metric(
                        label="Schema Complexity", 
                        value=current,
                        delta=delta,
                        help="Measure of how sophisticated the schema structure has become"
                    )
            else:
                st.info("No schema complexity data available yet. Add memories or process queries to generate data.")
        
        with metric_tabs[2]:
            st.write("System processing performance metrics")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Contradiction resolution metric
                contradiction_rate = st.session_state.get('contradiction_rate', 0)
                st.metric(
                    label="Contradiction Rate", 
                    value=f"{contradiction_rate:.2f}",
                    delta=st.session_state.get('delta_contradiction', None),
                    delta_color="inverse",
                    help="Percentage of beliefs that contain contradictions (lower is better)"
                )
            
            with col2:
                # Memory integration speed
                memory_time = st.session_state.get('avg_memory_time', 0)
                st.metric(
                    label="Memory Integration (ms)", 
                    value=f"{memory_time:.1f}",
                    delta=st.session_state.get('delta_memory_time', None),
                    delta_color="inverse",
                    help="Average time to integrate a new memory (lower is better)"
                )
            
            with col3:
                # Query response speed
                query_time = st.session_state.get('avg_query_time', 0)
                st.metric(
                    label="Query Response (ms)", 
                    value=f"{query_time:.1f}",
                    delta=st.session_state.get('delta_query_time', None),
                    delta_color="inverse",
                    help="Average time to process a query (lower is better)"
                )
            
            # Performance over time chart
            if 'performance_history' in st.session_state and st.session_state.performance_history:
                performance_data = st.session_state.performance_history
                x = list(range(len(performance_data)))
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x, [p.get('memory_time', 0) for p in performance_data], 
                        marker='o', linestyle='-', color='#2196F3', label='Memory Integration')
                ax.plot(x, [p.get('query_time', 0) for p in performance_data], 
                        marker='s', linestyle='-', color='#FF5722', label='Query Processing')
                ax.set_xlabel('Events')
                ax.set_ylabel('Processing Time (ms)')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            else:
                st.info("No performance data available yet. Add memories or process queries to generate data.")

def create_demo_runner(profile_name="default"):
    """Create and initialize a new demo runner with the specified profile.
    
    Args:
        profile_name: Name of the profile to use
        
    Returns:
        Initialized DemoRunner instance
    """
    try:
        from demo_runner import DemoRunner
        runner = DemoRunner(profile=profile_name)
        
        # Initialize with some default memories if needed
        # runner.add_memory("The ΨC-AI system helps manage cognitive coherence and contradiction resolution.")
        
        return runner
    except Exception as e:
        st.error(f"Error creating demo runner: {str(e)}")
        return None

def load_css():
    """Load custom CSS styles."""
    # Common styles
    css = """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 500;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 95%;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .memory-card {
        background-color: rgba(76, 175, 80, 0.1);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# History and Persistence Functions
def ensure_storage_directory(profile_name=None):
    """Ensure the storage directory exists for the given profile.
    
    Args:
        profile_name: Optional profile name for profile-specific directories
        
    Returns:
        Path object for the directory
    """
    # Create base storage directory
    base_dir = Path(STORAGE_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # If profile specified, create and return profile directory
    if profile_name:
        profile_dir = base_dir / profile_name
        profile_dir.mkdir(exist_ok=True)
        return profile_dir
    
    return base_dir

def get_profile_sessions(profile_name):
    """Get a list of all stored sessions for a given profile.
    
    Args:
        profile_name: Name of the profile to check
        
    Returns:
        List of session information dictionaries
    """
    profile_dir = ensure_storage_directory(profile_name)
    sessions = []
    
    # Scan for session directories
    for session_dir in profile_dir.glob("session_*"):
        if session_dir.is_dir():
            # Load session metadata if it exists
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        # Add path information
                        metadata["session_path"] = str(session_dir)
                        sessions.append(metadata)
                except Exception as e:
                    # If metadata is corrupted, create minimal info
                    sessions.append({
                        "session_id": session_dir.name.replace("session_", ""),
                        "session_path": str(session_dir),
                        "created_at": "Unknown",
                        "last_updated": "Unknown",
                        "memory_count": 0,
                        "status": "Metadata corrupted"
                    })
    
    # Sort by last updated time (most recent first)
    sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
    return sessions

def save_agent_state(profile_name, session_id=None, metadata=None):
    """Save the current agent state to persistent storage.
    
    Args:
        profile_name: Profile name the agent is using
        session_id: Optional session ID (will generate if not provided)
        metadata: Optional metadata to store with the session
        
    Returns:
        The session ID used
    """
    if not session_id:
        # Generate a new session ID if not provided
        session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Ensure directory exists
    profile_dir = ensure_storage_directory(profile_name)
    session_dir = profile_dir / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)
    
    # Create metadata if not provided
    if not metadata:
        metadata = {}
    
    # Add standard metadata
    metadata.update({
        "session_id": session_id,
        "profile": profile_name,
        "created_at": metadata.get("created_at", datetime.datetime.now().isoformat()),
        "last_updated": datetime.datetime.now().isoformat(),
        "memory_count": len(st.session_state.demo_runner.memory_store.get_all_memories()) if hasattr(st.session_state.demo_runner, 'memory_store') else 0,
        "status": "active"
    })
    
    try:
        # Save metadata
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save schema state
        schema_data = st.session_state.demo_runner.get_schema_graph_data()
        with open(session_dir / "schema.json", "w") as f:
            json.dump(schema_data, f, indent=2)
        
        # Save memory data
        memories = []
        if hasattr(st.session_state.demo_runner, 'memory_store'):
            for memory in st.session_state.demo_runner.memory_store.get_all_memories():
                # Convert memory object to serializable dict
                memory_dict = {
                    "content": memory.content,
                    "created_at": memory.created_at.isoformat() if hasattr(memory, 'created_at') else datetime.datetime.now().isoformat(),
                    "id": str(memory.id) if hasattr(memory, 'id') else str(uuid.uuid4())
                }
                memories.append(memory_dict)
        
        with open(session_dir / "memories.json", "w") as f:
            json.dump(memories, f, indent=2)
        
        # Save metrics history
        metrics_data = {
            "coherence_history": st.session_state.coherence_history,
            "schema_complexity": st.session_state.schema_complexity,
            "performance_history": st.session_state.performance_history
        }
        
        with open(session_dir / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save reflection log
        with open(session_dir / "reflection_log.json", "w") as f:
            json.dump(st.session_state.reflection_log, f, indent=2)
        
        return session_id
    except Exception as e:
        st.error(f"Error saving agent state: {str(e)}")
        return None

def load_agent_state(profile_name, session_id):
    """Load a previously saved agent state.
    
    Args:
        profile_name: Profile name the agent was using
        session_id: Session ID to load
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find the session directory
        profile_dir = ensure_storage_directory(profile_name)
        session_dir = profile_dir / f"session_{session_id}"
        
        if not session_dir.exists():
            st.error(f"Session {session_id} for profile {profile_name} not found.")
            return False
        
        # Create a new demo runner with the specified profile
        st.session_state.demo_runner = create_demo_runner(profile_name)
        
        # Load memories
        if (session_dir / "memories.json").exists():
            with open(session_dir / "memories.json", "r") as f:
                memories = json.load(f)
                
            # Add each memory to the runner
            for memory in memories:
                st.session_state.demo_runner.add_memory(memory["content"])
        
        # Load metrics history
        if (session_dir / "metrics.json").exists():
            with open(session_dir / "metrics.json", "r") as f:
                metrics_data = json.load(f)
                
            # Update session state with metrics
            st.session_state.coherence_history = metrics_data.get("coherence_history", [0.75])
            st.session_state.schema_complexity = metrics_data.get("schema_complexity", [1])
            st.session_state.performance_history = metrics_data.get("performance_history", [])
        
        # Load reflection log
        if (session_dir / "reflection_log.json").exists():
            with open(session_dir / "reflection_log.json", "r") as f:
                st.session_state.reflection_log = json.load(f)
        
        # Read metadata and update session state
        with open(session_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        # Set current session ID and profile
        st.session_state.current_session_id = session_id
        st.session_state.profile = profile_name
        
        # Update metadata to indicate last accessed time
        metadata["last_accessed"] = datetime.datetime.now().isoformat()
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error loading agent state: {str(e)}")
        return False

def archive_current_session(archive_name=None):
    """Archive the current session before reset or profile change.
    
    Args:
        archive_name: Optional custom name for the archive
        
    Returns:
        The session ID of the archived session
    """
    if not hasattr(st.session_state, 'profile') or not hasattr(st.session_state, 'demo_runner'):
        return None
    
    profile = st.session_state.profile
    session_id = st.session_state.get('current_session_id')
    
    # If no active session ID, generate one
    if not session_id:
        session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        st.session_state.current_session_id = session_id
    
    # Create archive metadata
    metadata = {
        "archive_name": archive_name or f"Archive {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": st.session_state.get('session_created_at', datetime.datetime.now().isoformat()),
        "archived_at": datetime.datetime.now().isoformat(),
        "status": "archived"
    }
    
    # Save the current state
    return save_agent_state(profile, session_id, metadata)

def auto_save():
    """Automatically save the current agent state."""
    # Only save if auto-save is enabled and we have an active session
    if st.session_state.get('auto_save_enabled', True) and hasattr(st.session_state, 'current_session_id'):
        # Check if any changes have occurred since last save
        if st.session_state.get('last_action'):
            save_agent_state(
                st.session_state.profile,
                st.session_state.current_session_id
            )
            # Reset last_action after saving
            st.session_state.last_action = None

if __name__ == "__main__":
    main() 