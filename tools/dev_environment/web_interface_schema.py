#!/usr/bin/env python3
"""
Schema Graph Visualization Web Interface

This module provides a Streamlit-based web interface for visualizing
memory schema graphs in the ΨC-AI SDK Development Environment.
"""

import os
import sys
import json
import base64
import tempfile
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from tools.dev_environment.memory_schema_integration import is_schema_integration_available

logger = logging.getLogger(__name__)

def render_schema_section(memory_sandbox):
    """
    Render the schema visualization section in the Streamlit interface
    
    Args:
        memory_sandbox: The MemorySandbox instance
    """
    st.title("📊 Schema Graph Visualization")
    
    # Check if schema integration is available
    if not hasattr(memory_sandbox, 'schema_integration') or not is_schema_integration_available():
        st.error("Schema integration is not available. Please install scikit-learn with 'pip install scikit-learn'")
        return
    
    # Sidebar parameters
    with st.sidebar:
        st.subheader("Schema Graph Parameters")
        min_edge_weight = st.slider(
            "Minimum Edge Weight", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.3, 
            step=0.05,
            help="Minimum similarity threshold for showing edges between memories"
        )
        
        cluster_threshold = st.slider(
            "Cluster Detection Threshold", 
            min_value=0.3, 
            max_value=0.9, 
            value=0.6, 
            step=0.05,
            help="Threshold for memory cluster detection (higher = more distinct clusters)"
        )
    
    # Create tabs for different views
    tabs = st.tabs(["Graph Visualization", "Memory Clusters", "Concept Suggestions", "Export"])
    
    # Schema Graph Visualization Tab
    with tabs[0]:
        # Check if schema graph exists
        schema_integration = memory_sandbox.schema_integration
        has_graph = schema_integration.has_schema_graph()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if not has_graph:
                st.info("No schema graph created yet. Click the button to create one.")
                if st.button("Create Schema Graph", type="primary"):
                    with st.spinner("Creating schema graph..."):
                        schema_integration.create_schema_graph()
                        st.success("Schema graph created successfully!")
                        st.experimental_rerun()
            else:
                # Graph created - show visualize/update buttons
                if st.button("Update Schema Graph"):
                    with st.spinner("Updating schema graph..."):
                        schema_integration.update_schema_graph()
                        st.success("Schema graph updated successfully!")
                        st.experimental_rerun()
                
                # Memory highlight selector
                highlight_memory = None
                if memory_sandbox.memory_store.memories:
                    memory_options = {f"{m.id} - {m.content[:30]}...": m.id for m in memory_sandbox.memory_store.memories}
                    highlight_key = st.selectbox(
                        "Highlight Memory (Optional)", 
                        ["None"] + list(memory_options.keys()),
                        index=0
                    )
                    if highlight_key != "None":
                        highlight_memory = memory_options[highlight_key]
                
                # Visualize the graph
                if schema_integration.has_schema_graph():
                    fig = visualize_schema_graph(
                        schema_integration, 
                        min_edge_weight=min_edge_weight,
                        highlight_memory=highlight_memory
                    )
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("No visible nodes in graph with current parameters.")
        
        with col2:
            if has_graph:
                # Graph statistics
                G = schema_integration.get_schema_graph()
                if G:
                    st.subheader("Graph Statistics")
                    memory_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'memory']
                    concept_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'concept']
                    
                    st.markdown(f"**Total Nodes:** {len(G.nodes)}")
                    st.markdown(f"**Memory Nodes:** {len(memory_nodes)}")
                    st.markdown(f"**Concept Nodes:** {len(concept_nodes)}")
                    st.markdown(f"**Total Edges:** {len(G.edges)}")
                    
                    # Display highest degree nodes
                    st.subheader("Most Connected Nodes")
                    degrees = dict(G.degree())
                    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    for node_id, degree in top_nodes:
                        node_type = G.nodes[node_id].get('type', 'unknown')
                        label = G.nodes[node_id].get('label', node_id)
                        st.markdown(f"**{label}** ({node_type}) - {degree} connections")
    
    # Memory Clusters Tab
    with tabs[1]:
        st.header("Memory Clusters")
        
        if not has_graph:
            st.info("No schema graph created yet. Please create a schema graph first.")
        else:
            if st.button("Detect Memory Clusters", key="detect_clusters"):
                with st.spinner("Detecting memory clusters..."):
                    clusters = schema_integration.detect_memory_clusters(threshold=cluster_threshold)
                    if not clusters:
                        st.warning("No memory clusters detected with current threshold.")
                    else:
                        st.success(f"Detected {len(clusters)} memory clusters!")
                        
                        # Display clusters
                        for i, cluster in enumerate(clusters):
                            with st.expander(f"Cluster {i+1}: {cluster['size']} memories (Coherence: {cluster['coherence']:.2f})"):
                                # Common themes
                                if cluster['themes']:
                                    st.subheader("Common Themes")
                                    st.write(", ".join(cluster['themes']))
                                
                                # Memory list
                                st.subheader("Memories in Cluster")
                                for memory_id in cluster['memory_ids']:
                                    memory = memory_sandbox.memory_store.get_memory(memory_id)
                                    if memory:
                                        st.markdown(f"**{memory.type}**: {memory.content}")
    
    # Concept Suggestions Tab
    with tabs[2]:
        st.header("Concept Suggestions")
        
        if not has_graph:
            st.info("No schema graph created yet. Please create a schema graph first.")
        else:
            max_concepts = st.slider("Max Concepts", min_value=1, max_value=10, value=5)
            
            if st.button("Generate Concept Suggestions"):
                with st.spinner("Generating concept suggestions..."):
                    concepts = schema_integration.suggest_concepts(max_concepts=max_concepts)
                    
                    if not concepts:
                        st.warning("No concept suggestions generated. Try reducing the cluster threshold.")
                    else:
                        st.success(f"Generated {len(concepts)} concept suggestions!")
                        
                        # Display concepts
                        for i, concept in enumerate(concepts):
                            with st.expander(f"Concept: {concept['name']} (Importance: {concept['importance']:.2f})"):
                                st.markdown(f"**Description:** {concept['description']}")
                                
                                # Themes
                                if concept['themes']:
                                    st.subheader("Themes")
                                    st.write(", ".join(concept['themes']))
                                
                                # Add concept button
                                if st.button(f"Add Concept to Graph", key=f"add_concept_{i}"):
                                    schema_integration.add_concept_node(
                                        concept_name=concept['name'],
                                        memory_ids=concept['memory_ids'],
                                        properties={
                                            'description': concept['description'],
                                            'importance': concept['importance'],
                                            'themes': concept['themes']
                                        }
                                    )
                                    st.success(f"Added concept '{concept['name']}' to graph!")
    
    # Export Tab
    with tabs[3]:
        st.header("Export Schema Graph")
        
        if not has_graph:
            st.info("No schema graph created yet. Please create a schema graph first.")
        else:
            if st.button("Export as JSON"):
                with st.spinner("Exporting schema graph..."):
                    json_data = schema_integration.export_schema_graph(format='json')
                    if json_data:
                        st.success("Schema graph exported successfully!")
                        st.json(json.loads(json_data))
                        
                        # Download button
                        st.download_button(
                            "Download JSON",
                            data=json_data,
                            file_name="memory_schema_graph.json",
                            mime="application/json"
                        )

def visualize_schema_graph(schema_integration, min_edge_weight=0.3, highlight_memory=None):
    """
    Visualize the schema graph with matplotlib
    
    Args:
        schema_integration: MemorySchemaIntegration instance
        min_edge_weight: Minimum edge weight to display
        highlight_memory: Memory ID to highlight
        
    Returns:
        Matplotlib figure or None if no visible nodes
    """
    # Get the graph
    G = schema_integration.get_schema_graph()
    if not G:
        return None
    
    # Filter edges by weight
    filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) >= min_edge_weight]
    if not filtered_edges:
        return None
    
    # Create a subgraph with the filtered edges
    visible_nodes = set()
    for u, v in filtered_edges:
        visible_nodes.add(u)
        visible_nodes.add(v)
    
    if not visible_nodes:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a subgraph
    subgraph = G.subgraph(visible_nodes)
    
    # Node positions using spring layout
    pos = nx.spring_layout(subgraph, k=0.3, seed=42)
    
    # Node colors and sizes
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        node_type = node_data.get('type', 'unknown')
        
        # Set node color based on type
        if node_type == 'memory':
            memory_type = node_data.get('memory_type', 'unknown')
            if memory_type == 'episodic':
                node_colors.append('skyblue')
            elif memory_type == 'semantic':
                node_colors.append('lightgreen')
            elif memory_type == 'procedural':
                node_colors.append('orange')
            elif memory_type == 'emotional':
                node_colors.append('pink')
            else:
                node_colors.append('gray')
        elif node_type == 'concept':
            node_colors.append('yellow')
        else:
            node_colors.append('gray')
        
        # Set node size based on importance
        importance = node_data.get('importance', 0.5)
        node_sizes.append(300 * importance + 100)
        
        # Node labels
        label = node_data.get('label', str(node))
        if len(label) > 20:
            label = label[:18] + '...'
        node_labels[node] = label
    
    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax
    )
    
    # Highlight the selected memory if any
    if highlight_memory and highlight_memory in subgraph:
        nx.draw_networkx_nodes(
            subgraph, pos,
            nodelist=[highlight_memory],
            node_color='red',
            node_size=node_sizes[list(subgraph.nodes()).index(highlight_memory)] * 1.2,
            alpha=1.0,
            ax=ax
        )
    
    # Edge widths and colors
    edge_widths = []
    edge_colors = []
    
    for u, v in subgraph.edges():
        edge_data = subgraph.edges[u, v]
        weight = edge_data.get('weight', 0.5)
        edge_type = edge_data.get('type', 'unknown')
        
        edge_widths.append(weight * 3)
        
        if edge_type == 'concept_memory':
            edge_colors.append('purple')
        else:
            edge_colors.append('gray')
    
    # Draw edges
    nx.draw_networkx_edges(
        subgraph, pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=node_labels,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='k', label='Episodic Memory'),
        Patch(facecolor='lightgreen', edgecolor='k', label='Semantic Memory'),
        Patch(facecolor='orange', edgecolor='k', label='Procedural Memory'),
        Patch(facecolor='pink', edgecolor='k', label='Emotional Memory'),
        Patch(facecolor='yellow', edgecolor='k', label='Concept'),
        Line2D([0], [0], color='gray', lw=2, label='Memory Similarity'),
        Line2D([0], [0], color='purple', lw=2, label='Concept Relation')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Memory Schema Graph")
    plt.axis("off")
    plt.tight_layout()
    
    return fig

def download_button(data, file_name, button_text):
    """
    Generate a download link for data
    
    Args:
        data: Data to download
        file_name: Name of the file
        button_text: Text of the button
    
    Returns:
        HTML for a download button
    """
    b64 = base64.b64encode(data.encode()).decode()
    dl_link = f'<a href="data:application/json;base64,{b64}" download="{file_name}">{button_text}</a>'
    return dl_link

def integrate_with_web_interface(web_interface):
    """
    Integrate the schema visualization with the web interface
    
    Args:
        web_interface: The WebInterface instance
    """
    # Add schema section to the web interface
    web_interface.add_section("schema", render_schema_section)
    logger.info("Integrated schema visualization with web interface")

if __name__ == "__main__":
    # This module should not be run directly
    print("This module is part of the ΨC-AI SDK Development Environment and should not be run directly.")
    print("Please use the launcher.py script to start the environment.")
    sys.exit(1) 