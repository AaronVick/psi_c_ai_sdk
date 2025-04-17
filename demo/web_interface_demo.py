#!/usr/bin/env python3
"""
ΨC Schema Integration Demo - Web Interface
-----------------------------------------
Streamlit-based web interface for the ΨC-AI SDK demo.
This provides an interactive, visual demonstration of
the ΨC cognitive architecture.
"""

import os
import json
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Import demo components
from demo_runner import DemoRunner
from llm_bridge import LLMBridge, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_interface")

# Constants
AVAILABLE_PROFILES = ["default", "healthcare", "legal", "story"]
DEMO_DIR = Path(__file__).parent
CONFIG_DIR = DEMO_DIR / "demo_config"
STATE_DIR = DEMO_DIR / "state"
EXPORTS_DIR = DEMO_DIR / "exports"

# Ensure directories exist
EXPORTS_DIR.mkdir(exist_ok=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.profile = "default"
        st.session_state.runner = None
        st.session_state.llm_bridge = None
        st.session_state.messages = []
        st.session_state.llm_enabled = False
        st.session_state.dark_mode = True
        st.session_state.show_graph = True
        st.session_state.show_metrics = True
        st.session_state.show_memories = True
        st.session_state.last_update_time = time.time()


# Streamlit page configuration
def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="ΨC Schema Integration Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .metric-box {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .memory-box {
        background-color: #2b313e;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .phase-transition {
        color: #FF4B4B;
        font-weight: bold;
    }
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
    
    /* New UI elements */
    .metric-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9E9E9E;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-description {
        font-size: 0.8rem;
        color: #BDBDBD;
        margin-top: 0.2rem;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .learning-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #4CAF50;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
        }
    }
    .event-box {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 3px solid #2196F3;
    }
    .event-learning {
        border-left-color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
    }
    .event-contradiction {
        border-left-color: #FF9800;
        background-color: rgba(255, 152, 0, 0.1);
    }
    .event-reorganization {
        border-left-color: #2196F3;
        background-color: rgba(33, 150, 243, 0.1);
    }
    .event-breakthrough {
        border-left-color: #9C27B0;
        background-color: rgba(156, 39, 176, 0.1);
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        from {
            box-shadow: 0 0 5px -5px #9C27B0;
        }
        to {
            box-shadow: 0 0 10px 5px rgba(156, 39, 176, 0.3);
        }
    }
    /* Status Gauges */
    .gauge-container {
        position: relative;
        width: 100%;
        height: 8px;
        background-color: #424242;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        overflow: hidden;
    }
    .gauge-fill {
        position: absolute;
        height: 100%;
        background-color: #4CAF50;
        border-radius: 4px;
        transition: width 1s ease-in-out;
    }
    .gauge-poor {
        background-color: #FF5252;
    }
    .gauge-fair {
        background-color: #FF9800;
    }
    .gauge-good {
        background-color: #4CAF50;
    }
    .gauge-excellent {
        background-color: #2196F3;
    }
    .health-status {
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .health-poor {
        color: #FF5252;
    }
    .health-fair {
        color: #FF9800;
    }
    .health-good {
        color: #4CAF50;
    }
    .health-excellent {
        color: #2196F3;
    }
    /* System Status Dashboard */
    .dashboard-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .dashboard-item {
        flex: 1;
        min-width: 150px;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .dashboard-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .dashboard-label {
        font-size: 0.8rem;
        color: #9E9E9E;
    }
    /* Help Panel */
    .help-panel {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
    }
    .help-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .help-text {
        font-size: 0.9rem;
        color: #BDBDBD;
    }
    </style>
    """, unsafe_allow_html=True)


# Sidebar components
def render_sidebar():
    """Render the sidebar with controls and settings."""
    st.sidebar.title("ΨC Demo Controls")
    
    # Profile selection
    selected_profile = st.sidebar.selectbox(
        "Domain Profile",
        AVAILABLE_PROFILES,
        index=AVAILABLE_PROFILES.index(st.session_state.profile) if st.session_state.profile in AVAILABLE_PROFILES else 0,
        help="Select a domain-specific profile for the demo"
    )
    
    # Only reload if profile changed
    if selected_profile != st.session_state.profile:
        st.session_state.profile = selected_profile
        load_demo_runner()
    
    # LLM toggle
    llm_enabled = st.sidebar.checkbox(
        "Enable LLM Integration",
        value=st.session_state.llm_enabled,
        help="Enable LLM integration for enhanced text generation (requires API key)"
    )
    
    if llm_enabled != st.session_state.llm_enabled:
        st.session_state.llm_enabled = llm_enabled
        load_llm_bridge()
    
    # UI toggles in an expander
    with st.sidebar.expander("Interface Settings", expanded=False):
        st.session_state.dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
        st.session_state.show_graph = st.checkbox("Show Schema Graph", value=st.session_state.show_graph)
        st.session_state.show_metrics = st.checkbox("Show Metrics", value=st.session_state.show_metrics)
        st.session_state.show_memories = st.checkbox("Show Recent Memories", value=st.session_state.show_memories)
    
    # Action buttons
    st.sidebar.button("Reset System", on_click=reset_system)
    
    # Export buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("Export JSON", on_click=export_summary, args=("json",))
    with col2:
        st.button("Export Markdown", on_click=export_summary, args=("markdown",))
    
    # Display system health
    if st.session_state.runner:
        with st.sidebar.expander("System Health", expanded=True):
            coherence = st.session_state.runner.get_current_coherence()
            entropy = st.session_state.runner.get_current_entropy()
            
            st.markdown(f"**Coherence Score (ΨC):** {coherence:.4f}")
            st.progress(min(coherence, 1.0))
            
            st.markdown(f"**Entropy:** {entropy:.4f}")
            st.progress(min(entropy, 1.0))
            
            st.markdown(f"**Schema Version:** {st.session_state.runner.schema_version}")
            
            memory_count = len(st.session_state.runner.memory_store.get_all_memories())
            st.markdown(f"**Memory Count:** {memory_count}")
            
            st.markdown(f"**Node Count:** {len(st.session_state.runner.schema_graph.nodes)}")
            st.markdown(f"**Edge Count:** {len(st.session_state.runner.schema_graph.edges)}")


# Main content
def render_main_content():
    """Render the main content of the demo."""
    st.title("ΨC Schema Integration Demo")
    
    # Create two columns for the layout
    col1, col2 = st.columns([4, 6])
    
    with col1:
        render_input_section()
        render_chat_section()
    
    with col2:
        if st.session_state.show_graph:
            render_graph_section()
        
        if st.session_state.show_metrics:
            render_metrics_section()
        
        if st.session_state.show_memories:
            render_memories_section()


# Input section
def render_input_section():
    """Render the memory input section."""
    st.subheader("Add New Memory")
    
    memory_input = st.text_area(
        "Enter a new memory or observation:",
        height=100,
        key="memory_input",
        help="Type a statement, observation, or belief to add to the system"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        confidence = st.slider(
            "Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="How confident are you in this memory?"
        )
    
    with col2:
        source = st.text_input(
            "Source (optional)",
            value="user",
            help="Where did this memory come from?"
        )
    
    if st.button("Add Memory", key="add_memory_button"):
        if memory_input.strip():
            add_memory(memory_input, confidence, source)
            # Clear the input
            st.session_state.memory_input = ""
        else:
            st.warning("Please enter a memory.")


# Chat/log section
def render_chat_section():
    """Render the chat log section."""
    st.subheader("Reflection Log")
    
    # Container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(msg)
    
    # Optional LLM query section if LLM is enabled
    if st.session_state.llm_enabled and st.session_state.llm_bridge and st.session_state.llm_bridge.is_enabled():
        st.subheader("Ask a Question")
        query = st.text_input(
            "Ask about the system's current state or reasoning:",
            key="llm_query",
            help="Ask a question about the system's current state, reasoning, or beliefs"
        )
        
        if st.button("Submit Question", key="submit_query_button"):
            if query.strip():
                process_llm_query(query)
                # Clear the input
                st.session_state.llm_query = ""
            else:
                st.warning("Please enter a question.")


# Render a single chat message
def render_chat_message(message: Dict[str, Any]):
    """Render a chat message."""
    role = message.get("role", "system")
    content = message.get("content", "")
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">🧠</div>
            <div class="content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="avatar">🤖</div>
            <div class="content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        if "phase_transition" in message and message["phase_transition"]:
            st.markdown(f"""
            <div class="chat-message system">
                <div class="avatar">⚠️</div>
                <div class="content phase-transition">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message system">
                <div class="avatar">ℹ️</div>
                <div class="content">{content}</div>
            </div>
            """, unsafe_allow_html=True)


# Graph visualization section
def render_graph_section():
    """Render the schema graph visualization."""
    st.subheader("Schema Graph")
    
    if st.session_state.runner:
        graph_data = st.session_state.runner.get_schema_graph_data()
        
        if not graph_data["nodes"]:
            st.info("No schema graph data available yet. Add some memories to see the graph.")
            return
        
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data["nodes"]:
            G.add_node(
                node["id"],
                label=node["label"],
                type=node["type"],
                importance=node["importance"]
            )
        
        # Add edges
        for edge in graph_data["edges"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"],
                type=edge["type"]
            )
        
        # Create a matplotlib figure
        plt.figure(figsize=(8, 6))
        
        # Node colors based on type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "concept")
            if node_type == "memory":
                node_colors.append("#4287f5")  # Blue for memories
            elif node_type == "concept":
                node_colors.append("#f54242")  # Red for concepts
            else:
                node_colors.append("#42f54e")  # Green for others
        
        # Node sizes based on importance
        node_sizes = [300 * G.nodes[node].get("importance", 0.5) + 100 for node in G.nodes()]
        
        # Edge weights
        edge_weights = [G[u][v].get("weight", 0.5) * 2 for u, v in G.edges()]
        
        # Layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            width=edge_weights,
            edge_color="gray",
            font_size=10,
            font_color="white",
            font_weight="bold",
            alpha=0.9
        )
        
        plt.axis("off")
        st.pyplot(plt)
        
        # Graph stats
        st.markdown(f"""
        **Graph Statistics:**
        - Nodes: {len(G.nodes())}
        - Edges: {len(G.edges())}
        - Density: {nx.density(G):.4f}
        - Connected Components: {nx.number_connected_components(G)}
        """)


# Metrics visualization section
def render_metrics_section():
    """Render the metrics visualization section."""
    st.subheader("Cognitive Metrics")
    
    if not st.session_state.runner:
        st.info("Runner not initialized.")
        return
    
    coherence_history = st.session_state.runner.get_coherence_history()
    entropy_history = st.session_state.runner.get_entropy_history()
    
    if not coherence_history or not entropy_history:
        st.info("No metrics data available yet.")
        return
    
    # Create a DataFrame for the metrics
    data = {
        "Index": list(range(len(coherence_history))),
        "Coherence": coherence_history,
        "Entropy": entropy_history
    }
    df = pd.DataFrame(data)
    
    # Plot coherence over time
    st.markdown("#### Coherence (ΨC) Evolution")
    
    coherence_chart = plt.figure(figsize=(8, 4))
    plt.plot(df["Index"], df["Coherence"], color="#4287f5", linewidth=2)
    plt.xlabel("Memory Updates")
    plt.ylabel("Coherence (ΨC)")
    plt.title("Coherence Evolution Over Time")
    plt.grid(True, alpha=0.3)
    st.pyplot(coherence_chart)
    
    # Plot entropy over time
    st.markdown("#### Entropy Evolution")
    
    entropy_chart = plt.figure(figsize=(8, 4))
    plt.plot(df["Index"], df["Entropy"], color="#f54242", linewidth=2)
    plt.xlabel("Memory Updates")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution Over Time")
    plt.grid(True, alpha=0.3)
    st.pyplot(entropy_chart)
    
    # Combined view
    st.markdown("#### Combined View")
    
    combined_chart = plt.figure(figsize=(8, 4))
    plt.plot(df["Index"], df["Coherence"], color="#4287f5", linewidth=2, label="Coherence")
    plt.plot(df["Index"], df["Entropy"], color="#f54242", linewidth=2, label="Entropy")
    plt.xlabel("Memory Updates")
    plt.ylabel("Value")
    plt.title("Coherence and Entropy Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(combined_chart)


# Recent memories section
def render_memories_section():
    """Render the recent memories section."""
    st.subheader("Recent Memories")
    
    if not st.session_state.runner:
        st.info("Runner not initialized.")
        return
    
    memories = st.session_state.runner.get_latest_memories(10)
    
    if not memories:
        st.info("No memories available yet.")
        return
    
    for mem in memories:
        content = mem.get("content", "")
        confidence = mem.get("metadata", {}).get("confidence", 0.0)
        source = mem.get("metadata", {}).get("source", "unknown")
        timestamp = mem.get("metadata", {}).get("timestamp", 0)
        
        # Format timestamp
        if timestamp:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        else:
            timestamp_str = "Unknown time"
        
        st.markdown(f"""
        <div class="memory-box">
            <strong>{content}</strong><br>
            <small>Confidence: {confidence:.2f} | Source: {source} | Added: {timestamp_str}</small>
        </div>
        """, unsafe_allow_html=True)


# Helper functions
def load_demo_runner():
    """Initialize or reload the demo runner."""
    try:
        st.session_state.runner = DemoRunner(profile=st.session_state.profile)
        logger.info(f"Demo runner initialized with profile: {st.session_state.profile}")
    except Exception as e:
        st.error(f"Error initializing demo runner: {e}")
        logger.error(f"Error initializing demo runner: {e}")


def load_llm_bridge():
    """Initialize or reload the LLM bridge."""
    try:
        llm_config = LLMConfig(is_enabled=st.session_state.llm_enabled)
        st.session_state.llm_bridge = LLMBridge(config=llm_config)
        logger.info(f"LLM bridge initialized. Enabled: {st.session_state.llm_enabled}")
    except Exception as e:
        st.error(f"Error initializing LLM bridge: {e}")
        logger.error(f"Error initializing LLM bridge: {e}")


def add_memory(content: str, confidence: float, source: str):
    """Add a new memory to the system."""
    if not st.session_state.runner:
        st.error("Demo runner not initialized.")
        return
    
    try:
        # Prepare metadata
        metadata = {
            "confidence": confidence,
            "source": source,
            "timestamp": time.time()
        }
        
        # Add the memory
        result = st.session_state.runner.add_memory(content, metadata)
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": f"🧠 Memory Added: \"{content}\""
        })
        
        # Process reflection
        reflection_text = ""
        if st.session_state.llm_enabled and st.session_state.llm_bridge and st.session_state.llm_bridge.is_enabled():
            # Get relevant memories for context
            memories = st.session_state.runner.get_latest_memories(5)
            relevant_memory_texts = [m["content"] for m in memories if m["content"] != content][:3]
            
            # Generate enhanced reflection
            reflection_text = st.session_state.llm_bridge.enhance_reflection(
                content, relevant_memory_texts
            )
        else:
            if result["contradictions"] > 0:
                reflection_text = f"🔄 Reflection detected {result['contradictions']} contradiction(s)."
            else:
                reflection_text = "🔄 Reflection complete. No contradictions found."
        
        # Add reflection message
        st.session_state.messages.append({
            "role": "assistant",
            "content": reflection_text
        })
        
        # Add system update message
        update_text = ""
        if st.session_state.llm_enabled and st.session_state.llm_bridge and st.session_state.llm_bridge.is_enabled():
            # Generate enhanced summary
            update_text = st.session_state.llm_bridge.generate_change_summary(
                content,
                result["coherence_change"] + st.session_state.runner.get_current_coherence(),
                st.session_state.runner.get_current_coherence(),
                result["entropy_change"] + st.session_state.runner.get_current_entropy(),
                st.session_state.runner.get_current_entropy(),
                result["contradictions"],
                result["schema_updated"],
                result["phase_transition"]
            )
        else:
            if result["schema_updated"]:
                update_text += f"📐 Schema updated (Version {st.session_state.runner.schema_version}).\n"
            
            coherence_change = result["coherence_change"]
            update_text += f"📈 ΨC {'increased' if coherence_change > 0 else 'decreased'} by {abs(coherence_change):.4f} to {st.session_state.runner.get_current_coherence():.4f}\n"
            
            entropy_change = result["entropy_change"]
            update_text += f"🔍 Entropy {'increased' if entropy_change > 0 else 'decreased'} by {abs(entropy_change):.4f} to {st.session_state.runner.get_current_entropy():.4f}"
        
        st.session_state.messages.append({
            "role": "system",
            "content": update_text,
            "phase_transition": result["phase_transition"]
        })
        
        # Special message for phase transitions
        if result["phase_transition"]:
            st.session_state.messages.append({
                "role": "system",
                "content": "⚠️ **Phase transition detected!** The system has undergone a significant shift in its cognitive structure.",
                "phase_transition": True
            })
        
        # Update timestamp
        st.session_state.last_update_time = time.time()
        
        # Rerun the app to update the UI
        st.experimental_rerun()
    
    except Exception as e:
        st.error(f"Error adding memory: {e}")
        logger.error(f"Error adding memory: {e}")


def process_llm_query(query: str):
    """Process a user query using the LLM bridge."""
    if not st.session_state.llm_bridge or not st.session_state.llm_bridge.is_enabled():
        st.error("LLM bridge not available.")
        return
    
    try:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": f"❓ Question: \"{query}\""
        })
        
        # Prepare context for the LLM
        context = {
            "coherence": st.session_state.runner.get_current_coherence(),
            "entropy": st.session_state.runner.get_current_entropy(),
            "memory_count": len(st.session_state.runner.memory_store.get_all_memories()),
            "node_count": len(st.session_state.runner.schema_graph.nodes),
            "recent_memories": [m["content"] for m in st.session_state.runner.get_latest_memories(3)]
        }
        
        # Get response from LLM
        response = st.session_state.llm_bridge.process_user_query(query, context)
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Update timestamp
        st.session_state.last_update_time = time.time()
        
        # Rerun the app to update the UI
        st.experimental_rerun()
    
    except Exception as e:
        st.error(f"Error processing query: {e}")
        logger.error(f"Error processing query: {e}")


def reset_system():
    """Reset the demo system."""
    if not st.session_state.runner:
        st.error("Demo runner not initialized.")
        return
    
    try:
        # Reset the runner
        st.session_state.runner.reset_system()
        
        # Clear messages
        st.session_state.messages = []
        
        # Add system message
        st.session_state.messages.append({
            "role": "system",
            "content": "System reset complete. All memories and schema data have been cleared.",
            "phase_transition": False
        })
        
        # Update timestamp
        st.session_state.last_update_time = time.time()
        
        # Rerun the app to update the UI
        st.experimental_rerun()
    
    except Exception as e:
        st.error(f"Error resetting system: {e}")
        logger.error(f"Error resetting system: {e}")


def export_summary(format: str):
    """Export a summary of the current session."""
    if not st.session_state.runner:
        st.error("Demo runner not initialized.")
        return
    
    try:
        # Generate summary
        summary = st.session_state.runner.export_session_summary(format)
        
        # Create filename
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        profile = st.session_state.profile
        
        if format == "json":
            filename = f"{profile}_summary_{timestamp}.json"
            mime_type = "application/json"
        else:  # markdown
            filename = f"{profile}_summary_{timestamp}.md"
            mime_type = "text/markdown"
        
        # Save summary to file
        filepath = EXPORTS_DIR / filename
        with open(filepath, 'w') as f:
            f.write(summary)
        
        # Provide download link
        with open(filepath, 'r') as f:
            content = f.read()
        
        st.sidebar.download_button(
            label=f"Download {format.upper()}",
            data=content,
            file_name=filename,
            mime=mime_type
        )
        
        st.sidebar.success(f"Summary exported as {filename}")
        logger.info(f"Summary exported as {filename}")
    
    except Exception as e:
        st.error(f"Error exporting summary: {e}")
        logger.error(f"Error exporting summary: {e}")


# Helper functions for user-friendly displays
def get_health_status(value, is_entropy=False):
    """Get a health status label based on value."""
    if is_entropy:
        value = 1.0 - value  # Invert entropy for status
    
    if value < 0.3:
        return "Poor", "poor"
    elif value < 0.6:
        return "Fair", "fair"
    elif value < 0.8:
        return "Good", "good"
    else:
        return "Excellent", "excellent"

def format_tooltip(text, tooltip_text):
    """Format text with a tooltip."""
    return f'<span class="tooltip">{text}<span class="tooltiptext">{tooltip_text}</span></span>'

def render_gauge(value, is_entropy=False):
    """Render a gauge visualization for a value between 0 and 1."""
    if is_entropy:
        value = 1.0 - value  # Invert entropy for gauge
    
    percentage = max(min(value * 100, 100), 0)  # Keep between 0-100
    status_text, status_class = get_health_status(value, is_entropy=False)
    
    html = f"""
    <div class="gauge-container">
        <div class="gauge-fill gauge-{status_class}" style="width: {percentage}%;"></div>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>0%</span>
        <span class="health-status health-{status_class}">{status_text} ({percentage:.0f}%)</span>
        <span>100%</span>
    </div>
    """
    return html

def get_status_label(value):
    """Convert a numeric value (0-1) to a user-friendly status label."""
    if value >= 0.9:
        return "Excellent", "🟢"
    elif value >= 0.75:
        return "Good", "🟡"
    elif value >= 0.6:
        return "Moderate", "🟠"
    elif value >= 0.4:
        return "Fair", "🔵"
    else:
        return "Poor", "🔴"

def render_system_health_dashboard(demo):
    """Render a user-friendly system health dashboard."""
    coherence = demo.get_current_coherence()
    entropy = demo.get_current_entropy()
    
    # Convert to user-friendly metrics (percentages)
    consistency = coherence * 100
    organization = (1 - entropy) * 100
    overall_health = (consistency + organization) / 2
    
    # Status labels
    consistency_label, consistency_icon = get_status_label(coherence)
    organization_label, organization_icon = get_status_label(1 - entropy)
    overall_label, overall_icon = get_status_label(overall_health / 100)
    
    # Stats for additional context
    memory_count = len(demo.memories)
    concept_count = len(demo.schema.get_all_concepts()) if hasattr(demo.schema, "get_all_concepts") else 0
    relationship_count = len(demo.schema.get_all_relationships()) if hasattr(demo.schema, "get_all_relationships") else 0
    
    # Render learning indicator
    is_active = memory_count > 0
    
    html = f"""
    <div class="system-health-dashboard">
        <div class="dashboard-header">
            <div class="system-status">
                <div class="status-label">System Health</div>
                <div class="status-value">
                    <span class="status-badge status-{overall_label.lower()}">{overall_label}</span>
                    <span class="status-percentage">{overall_health:.0f}%</span>
                </div>
            </div>
            <div class="learning-indicator">
                <div class="indicator-dot {'active' if is_active else ''}"></div>
                <div class="indicator-label">{'Learning Active' if is_active else 'Awaiting Input'}</div>
                <div class="tooltip">This indicator shows if the system is actively processing and learning.
                    <span class="tooltiptext">
                        The pulsing green dot indicates the system is actively learning from recently added information.
                    </span>
                </div>
            </div>
        </div>
        
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-header">
                    <div>Knowledge Consistency</div>
                    <div class="tooltip">?
                        <span class="tooltiptext">
                            How well all the information fits together without contradictions.
                            Higher is better.
                        </span>
                    </div>
                </div>
                <div class="metric-value status-{consistency_label.lower()}">{consistency:.0f}%</div>
                <div class="metric-status">{consistency_label}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-header">
                    <div>Information Organization</div>
                    <div class="tooltip">?
                        <span class="tooltiptext">
                            How well-structured vs. chaotic the knowledge is.
                            Higher is better.
                        </span>
                    </div>
                </div>
                <div class="metric-value status-{organization_label.lower()}">{organization:.0f}%</div>
                <div class="metric-status">{organization_label}</div>
            </div>
        </div>
        
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-label">Information Pieces:</div>
                <div class="stat-value">{memory_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Concepts Learned:</div>
                <div class="stat-value">{concept_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Connections:</div>
                <div class="stat-value">{relationship_count}</div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
    return html


def render_help_panel():
    """Render an expandable help panel with user-friendly explanations."""
    with st.expander("📘 Understanding How This System Works", expanded=False):
        st.markdown("""
        ### How the ΨC System Learns
        
        This system learns by processing information and building connections between concepts. Here's what you're seeing:
        
        #### Dashboard Indicators
        
        - **Knowledge Consistency**: Shows how well all the information fits together without contradictions.
          - Higher percentages mean fewer contradictions and more stable knowledge.
          - When new information contradicts existing knowledge, this number may temporarily drop.
        
        - **Information Organization**: Indicates how well-structured vs. chaotic the knowledge is.
          - Higher percentages mean better organization and more meaningful connections.
          - This improves as the system learns more about related concepts.
        
        #### Graph Visualizations
        
        - **Knowledge Graph**: The interconnected web of concepts and their relationships.
          - Each node is a concept the system has learned.
          - Lines show how concepts are connected to each other.
          - Thicker lines represent stronger connections.
        
        - **Before/After Views**: When you add new information, you'll see how it changed the system's understanding.
          - Added nodes/edges are highlighted in green.
          - Modified connections may appear in yellow.
          - Removed connections (resolved contradictions) appear in red.
        
        #### How It Processes New Information
        
        1. **Takes in information** through text or structured data
        2. **Identifies key concepts** and their relationships
        3. **Integrates with existing knowledge** by finding connections
        4. **Resolves contradictions** when new information conflicts with old
        5. **Updates its understanding** based on the most reliable sources
        
        The system aims to maintain consistent knowledge while continuously learning and improving its understanding.
        """)

def render_event_explanation(event_type, details):
    """Render a clear explanation of events when new information is processed."""
    explanation = ""
    icon = "ℹ️"
    
    if event_type == "contradiction_detected":
        icon = "⚠️"
        explanation = f"Found conflicting information about '{details.get('concept', 'a concept')}'. "
        resolution = details.get('resolution', 'Evaluating which information is more reliable.')
        explanation += resolution
    
    elif event_type == "schema_updated":
        icon = "🔄"
        concept_count = details.get('new_concepts', 0)
        relation_count = details.get('new_relations', 0)
        
        if concept_count > 0 and relation_count > 0:
            explanation = f"Learned {concept_count} new concept(s) and {relation_count} new connection(s)."
        elif concept_count > 0:
            explanation = f"Learned {concept_count} new concept(s)."
        elif relation_count > 0:
            explanation = f"Discovered {relation_count} new connection(s) between concepts."
        else:
            explanation = "Updated understanding of existing concepts."
    
    elif event_type == "confidence_updated":
        icon = "📈"
        concept = details.get('concept', 'a concept')
        old_conf = details.get('old_confidence', 0) * 100
        new_conf = details.get('new_confidence', 0) * 100
        change = new_conf - old_conf
        
        if change > 0:
            explanation = f"Confidence in '{concept}' increased from {old_conf:.0f}% to {new_conf:.0f}%."
        else:
            explanation = f"Confidence in '{concept}' decreased from {old_conf:.0f}% to {new_conf:.0f}%."
    
    elif event_type == "memory_processed":
        icon = "📝"
        explanation = "Processed new information and updated knowledge."
    
    html = f"""
    <div class="event-explanation">
        <div class="event-icon">{icon}</div>
        <div class="event-text">{explanation}</div>
    </div>
    """
    
    return html

def render_before_after(before_metrics, after_metrics):
    """Show before-and-after comparison of memory processing."""
    # Convert raw metrics to user-friendly percentages
    before_coherence = before_metrics.get('coherence', 0) * 100
    after_coherence = after_metrics.get('coherence', 0) * 100
    coherence_change = after_coherence - before_coherence
    
    before_entropy = before_metrics.get('entropy', 0) 
    after_entropy = after_metrics.get('entropy', 0)
    # For entropy, lower is better, so we invert the percentage
    before_organization = (1 - before_entropy) * 100
    after_organization = (1 - after_entropy) * 100
    organization_change = after_organization - before_organization
    
    html = f"""
    <div class="metrics-comparison">
        <div class="comparison-header">Changes After Processing</div>
        <div class="comparison-metrics">
            <div class="comparison-metric">
                <div class="metric-name">Knowledge Consistency</div>
                <div class="metric-values">
                    <div class="before-value">{before_coherence:.1f}%</div>
                    <div class="arrow">→</div>
                    <div class="after-value">{after_coherence:.1f}%</div>
                    <div class="change-value {
                        'positive' if coherence_change > 0 else 'negative' if coherence_change < 0 else ''
                    }">{
                        f"+{coherence_change:.1f}%" if coherence_change > 0
                        else f"{coherence_change:.1f}%" if coherence_change < 0
                        else "0%"
                    }</div>
                </div>
            </div>
            
            <div class="comparison-metric">
                <div class="metric-name">Information Organization</div>
                <div class="metric-values">
                    <div class="before-value">{before_organization:.1f}%</div>
                    <div class="arrow">→</div>
                    <div class="after-value">{after_organization:.1f}%</div>
                    <div class="change-value {
                        'positive' if organization_change > 0 else 'negative' if organization_change < 0 else ''
                    }">{
                        f"+{organization_change:.1f}%" if organization_change > 0
                        else f"{organization_change:.1f}%" if organization_change < 0
                        else "0%"
                    }</div>
                </div>
            </div>
        </div>
    </div>
    """
    
    return html

def render_reflection_log(memories, max_entries=5):
    """Display a log of processed memories with explanations."""
    if not memories:
        return "<div class='reflection-log-empty'>No memories processed yet.</div>"
    
    entries = []
    for i, memory in enumerate(memories[-max_entries:]):
        # Get memory details
        query = memory.get('query', 'N/A')
        response = memory.get('response', 'N/A')
        timestamp = memory.get('timestamp', 'N/A')
        event_type = memory.get('event_type', 'memory_processed')
        details = memory.get('details', {})
        
        # Get event explanation
        explanation = render_event_explanation(event_type, details)
        
        entry = f"""
        <div class="reflection-entry">
            <div class="reflection-header">
                <div class="reflection-number">#{len(memories) - i}</div>
                <div class="reflection-time">{timestamp}</div>
            </div>
            <div class="reflection-content">
                <div class="reflection-query">{query}</div>
                <div class="reflection-response">{response}</div>
            </div>
            <div class="reflection-explanation">
                {explanation}
            </div>
        </div>
        """
        entries.append(entry)
    
    html = f"""
    <div class="reflection-log">
        <div class="reflection-log-header">Memory Processing Log</div>
        {"".join(entries)}
    </div>
    """
    
    return html

def render_system_metrics():
    """Render system metrics in a user-friendly dashboard format."""
    st.header("System Dashboard")
    
    # Get metrics from demo if available
    coherence = getattr(st.session_state.demo, "coherence", 0.75)
    entropy = getattr(st.session_state.demo, "entropy", 0.25)
    memory_count = getattr(st.session_state.demo, "memory_count", 0)
    concept_count = getattr(st.session_state.demo, "concept_count", 0)
    relation_count = getattr(st.session_state.demo, "relation_count", 0)
    
    # Calculate user-friendly metrics
    consistency = coherence * 100
    organization = (1 - entropy) * 100
    overall_health = (consistency + organization) / 2
    
    # Get status labels
    consistency_label, consistency_icon = get_status_label(coherence)
    organization_label, organization_icon = get_status_label(1 - entropy)
    overall_label, overall_icon = get_status_label(overall_health / 100)
    
    # Create three columns for the metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"{consistency_icon} Knowledge Consistency",
            value=f"{consistency:.1f}%",
            delta=None,
            help="How well all information fits together without contradictions"
        )
        st.caption(f"Status: {consistency_label}")
    
    with col2:
        st.metric(
            label=f"{organization_icon} Information Organization",
            value=f"{organization:.1f}%",
            delta=None,
            help="How structured vs. chaotic the knowledge is"
        )
        st.caption(f"Status: {organization_label}")
    
    with col3:
        st.metric(
            label=f"{overall_icon} Overall System Health",
            value=f"{overall_health:.1f}%",
            delta=None,
            help="Combined health score of the system"
        )
        st.caption(f"Status: {overall_label}")
    
    # Add a learning indicator
    is_learning = getattr(st.session_state.demo, "is_learning", False)
    learning_status = "Active" if is_learning else "Idle"
    learning_color = "green" if is_learning else "gray"
    
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-top: 10px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {learning_color}; 
                  animation: {
                      'pulse 1.5s infinite' if is_learning else 'none'
                  }"></div>
            <div style="margin-left: 8px; font-size: 0.9rem;">Learning Status: <b>{learning_status}</b></div>
        </div>
        <style>
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create horizontal rule
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # Add statistics in a more compact format
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div>
                <span style="font-size: 0.8rem; color: #888;">Information Pieces</span><br>
                <span style="font-size: 1.2rem; font-weight: bold;">{memory_count}</span>
            </div>
            <div>
                <span style="font-size: 0.8rem; color: #888;">Concepts Learned</span><br>
                <span style="font-size: 1.2rem; font-weight: bold;">{concept_count}</span>
            </div>
            <div>
                <span style="font-size: 0.8rem; color: #888;">Connections</span><br>
                <span style="font-size: 1.2rem; font-weight: bold;">{relation_count}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Main application logic
def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Setup page
    setup_page()
    
    # Initialize components if needed
    if not st.session_state.runner:
        load_demo_runner()
    
    if st.session_state.llm_bridge is None:
        load_llm_bridge()
    
    # Add help panel first
    render_help_panel()
    
    # Render layout
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main() 