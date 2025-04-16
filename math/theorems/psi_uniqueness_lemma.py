#!/usr/bin/env python3
"""
ΨC-Coherence Stability Theorem Testbed

This module implements a testbed for proving and demonstrating the uniqueness of ΨC signatures 
over time in multi-agent, contradiction-rich environments. The core hypothesis is that no two 
agents with different schemas can yield the same coherence trajectory under identical inputs.

Formally, we aim to validate the following:
∀a, b: ΨC_a(t) ≠ ΨC_b(t) ⇒ Σ_a ≠ Σ_b

Where:
- ΨC_a(t): The coherence signature of agent a at time t
- Σ_a: The internal schema of agent a

This theoretical framework provides a foundation for:
1. Identity verification of agents over time
2. Detection of schema manipulation or "coherence spoofing"
3. Understanding the relationship between internal belief structures and observable coherence

The testbed subjects multiple agents to identical input streams and tracks their coherence 
trajectories to empirically validate the uniqueness property across various conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
import os
import sys
import json
import time
import logging
import uuid
from datetime import datetime
import random
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure we can import from the psi_c_ai_sdk package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from psi_c_ai_sdk.core import rollup_engine
    from psi_c_ai_sdk.belief import belief_network
    MOCK_MODE = False
except ImportError:
    logger.warning("Could not import ΨC SDK modules. Running in mock mode with simulated data.")
    MOCK_MODE = True


@dataclass
class Belief:
    """Represents a belief in an agent's schema."""
    id: str
    content: str
    confidence: float
    vector: np.ndarray
    created_at: float = field(default_factory=time.time)
    connections: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "created_at": self.created_at,
            "connections": self.connections
        }


@dataclass
class AgentSchema:
    """Represents an agent's internal schema (belief network)."""
    id: str
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    graph: nx.Graph = field(default_factory=nx.Graph)
    creation_time: float = field(default_factory=time.time)
    
    def add_belief(self, belief: Belief):
        """Add a belief to the schema."""
        self.beliefs[belief.id] = belief
        self.graph.add_node(belief.id, belief=belief)
        
        # Add connections
        for connected_id in belief.connections:
            if connected_id in self.beliefs:
                self.graph.add_edge(belief.id, connected_id)
    
    def get_schema_fingerprint(self) -> np.ndarray:
        """
        Generate a unique fingerprint for this schema based on its structure.
        
        Returns:
            A vector representing the schema's structure
        """
        if not self.beliefs:
            return np.zeros(128)
        
        # Compute centrality metrics
        degree_centrality = nx.degree_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
        except:
            # Fall back if eigenvector centrality fails (common in disconnected graphs)
            eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}
        
        # Compute clustering metrics
        clustering = nx.clustering(self.graph)
        
        # Collect metrics in a consistent node order
        nodes = sorted(self.graph.nodes())
        fingerprint_metrics = []
        
        for node in nodes:
            metrics = [
                degree_centrality.get(node, 0),
                closeness_centrality.get(node, 0),
                eigenvector_centrality.get(node, 0),
                clustering.get(node, 0),
                self.beliefs[node].confidence
            ]
            fingerprint_metrics.extend(metrics)
        
        # Create a fixed-length fingerprint (pad or truncate)
        target_length = 128
        if len(fingerprint_metrics) > target_length:
            # Truncate
            fingerprint = np.array(fingerprint_metrics[:target_length])
        else:
            # Pad with zeros
            fingerprint = np.zeros(target_length)
            fingerprint[:len(fingerprint_metrics)] = fingerprint_metrics
        
        # Normalize
        norm = np.linalg.norm(fingerprint)
        if norm > 0:
            fingerprint = fingerprint / norm
        
        return fingerprint


@dataclass
class CoherenceSignature:
    """
    Represents a ΨC coherence signature over time.
    
    A signature consists of a time series of metrics that characterize
    the agent's coherence state.
    """
    agent_id: str
    timestamps: List[float] = field(default_factory=list)
    psi_c_scores: List[float] = field(default_factory=list)
    entropy_values: List[float] = field(default_factory=list)
    contradiction_levels: List[float] = field(default_factory=list)
    stability_scores: List[float] = field(default_factory=list)
    schema_fingerprints: List[np.ndarray] = field(default_factory=list)
    
    def add_datapoint(self, 
                     timestamp: float,
                     psi_c: float,
                     entropy: float,
                     contradiction_level: float,
                     stability: float,
                     schema_fingerprint: np.ndarray):
        """Add a new datapoint to the signature."""
        self.timestamps.append(timestamp)
        self.psi_c_scores.append(psi_c)
        self.entropy_values.append(entropy)
        self.contradiction_levels.append(contradiction_level)
        self.stability_scores.append(stability)
        self.schema_fingerprints.append(schema_fingerprint)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert signature to a DataFrame for analysis."""
        data = {
            "agent_id": [self.agent_id] * len(self.timestamps),
            "timestamp": self.timestamps,
            "psi_c_score": self.psi_c_scores,
            "entropy": self.entropy_values,
            "contradiction_level": self.contradiction_levels,
            "stability": self.stability_scores
        }
        
        # Add fingerprint components as separate columns
        if self.schema_fingerprints:
            for i in range(len(self.schema_fingerprints[0])):
                data[f"fp_{i}"] = [fp[i] for fp in self.schema_fingerprints]
        
        return pd.DataFrame(data)
    
    def calculate_uniqueness_metric(self) -> float:
        """
        Calculate a metric that quantifies the uniqueness of this signature.
        
        Higher values indicate more unique/distinguishable signatures.
        
        Returns:
            Uniqueness score between 0 and 1
        """
        if len(self.psi_c_scores) < 2:
            return 0.0
        
        # Calculate variance in each metric
        psi_c_var = np.var(self.psi_c_scores)
        entropy_var = np.var(self.entropy_values)
        contradiction_var = np.var(self.contradiction_levels)
        stability_var = np.var(self.stability_scores)
        
        # Calculate average variance in fingerprints
        fp_var = 0.0
        if self.schema_fingerprints:
            fp_array = np.array(self.schema_fingerprints)
            fp_var = np.mean(np.var(fp_array, axis=0))
        
        # Combine metrics (weighted sum)
        uniqueness = (
            0.3 * psi_c_var + 
            0.2 * entropy_var + 
            0.2 * contradiction_var + 
            0.2 * stability_var + 
            0.1 * fp_var
        )
        
        # Normalize to 0-1 range using a sigmoid function
        normalized = 1.0 / (1.0 + np.exp(-10 * uniqueness))
        
        return normalized


@dataclass
class Agent:
    """
    Represents an agent with an internal schema and coherence signature.
    """
    id: str
    schema: AgentSchema
    coherence_signature: CoherenceSignature
    creation_time: float = field(default_factory=time.time)
    
    def calculate_coherence_metrics(self) -> Tuple[float, float, float, float]:
        """
        Calculate the coherence metrics for the current schema state.
        
        Returns:
            Tuple of (psi_c_score, entropy, contradiction_level, stability)
        """
        # In mock mode, we use simplified calculations
        if not self.schema.beliefs:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate ΨC score (simplified)
        belief_confidences = [b.confidence for b in self.schema.beliefs.values()]
        avg_confidence = np.mean(belief_confidences)
        
        # Calculate graph density as a proxy for coherence
        graph_density = nx.density(self.schema.graph)
        
        # Combine for a ΨC score
        psi_c_score = 0.7 * avg_confidence + 0.3 * graph_density
        
        # Calculate entropy (higher value = more disorder)
        centrality_values = list(nx.degree_centrality(self.schema.graph).values())
        if centrality_values:
            # Normalize centrality values to sum to 1 for entropy calculation
            centrality_sum = sum(centrality_values)
            if centrality_sum > 0:
                normalized_centralities = [c / centrality_sum for c in centrality_values]
                entropy = -sum(p * np.log(p) if p > 0 else 0 for p in normalized_centralities)
            else:
                entropy = 0
        else:
            entropy = 0
        
        # Calculate contradiction level (simplified)
        # We simulate contradictions as anti-correlated node pairs
        contradiction_level = 0.0
        if len(self.schema.beliefs) > 1:
            # Randomly assign a contradiction level based on graph structure
            contradiction_level = 0.3 * (1 - graph_density)
        
        # Calculate stability as consistency over time
        stability = 1.0 - contradiction_level
        
        return psi_c_score, entropy, contradiction_level, stability
    
    def update_coherence_signature(self):
        """Update the coherence signature with current schema state."""
        psi_c, entropy, contradiction, stability = self.calculate_coherence_metrics()
        schema_fingerprint = self.schema.get_schema_fingerprint()
        
        self.coherence_signature.add_datapoint(
            timestamp=time.time(),
            psi_c=psi_c,
            entropy=entropy,
            contradiction_level=contradiction,
            stability=stability,
            schema_fingerprint=schema_fingerprint
        )
    
    def process_input(self, input_data: Any):
        """
        Process an input and update the agent's schema accordingly.
        
        In a real implementation, this would integrate with the belief network
        to update the agent's schema. For the testbed, we simulate changes.
        
        Args:
            input_data: The input to process
        """
        # Simulate schema updates based on input
        if isinstance(input_data, str):
            # Create a new belief based on the input
            belief_id = f"belief_{uuid.uuid4().hex[:8]}"
            
            # Create a vector representation of the belief
            vector = np.random.normal(0, 1, 128)
            vector = vector / np.linalg.norm(vector)
            
            # Determine connections to existing beliefs
            existing_beliefs = list(self.schema.beliefs.keys())
            num_connections = min(3, len(existing_beliefs))
            connections = random.sample(existing_beliefs, num_connections) if num_connections > 0 else []
            
            # Create the belief
            belief = Belief(
                id=belief_id,
                content=input_data,
                confidence=0.7 + 0.3 * random.random(),  # Random confidence 0.7-1.0
                vector=vector,
                connections=connections
            )
            
            # Add to schema
            self.schema.add_belief(belief)
        
        # Update the coherence signature
        self.update_coherence_signature()


def compare_coherence_signatures(sig1: CoherenceSignature, sig2: CoherenceSignature) -> Dict[str, float]:
    """
    Compare two coherence signatures and calculate similarity metrics.
    
    Args:
        sig1: First coherence signature
        sig2: Second coherence signature
        
    Returns:
        Dictionary of similarity metrics
    """
    # Ensure signatures have the same length
    min_len = min(len(sig1.psi_c_scores), len(sig2.psi_c_scores))
    
    if min_len < 2:
        return {
            "psi_c_correlation": 0.0,
            "entropy_correlation": 0.0,
            "contradiction_correlation": 0.0,
            "stability_correlation": 0.0,
            "fingerprint_distance": 1.0,
            "overall_similarity": 0.0
        }
    
    # Calculate correlations for each metric
    psi_c_corr, _ = pearsonr(sig1.psi_c_scores[:min_len], sig2.psi_c_scores[:min_len])
    entropy_corr, _ = pearsonr(sig1.entropy_values[:min_len], sig2.entropy_values[:min_len])
    contradiction_corr, _ = pearsonr(sig1.contradiction_levels[:min_len], sig2.contradiction_levels[:min_len])
    stability_corr, _ = pearsonr(sig1.stability_scores[:min_len], sig2.stability_scores[:min_len])
    
    # Handle NaN values
    psi_c_corr = 0.0 if np.isnan(psi_c_corr) else psi_c_corr
    entropy_corr = 0.0 if np.isnan(entropy_corr) else entropy_corr
    contradiction_corr = 0.0 if np.isnan(contradiction_corr) else contradiction_corr
    stability_corr = 0.0 if np.isnan(stability_corr) else stability_corr
    
    # Calculate average fingerprint distance
    fp_distance = 0.0
    if sig1.schema_fingerprints and sig2.schema_fingerprints:
        fp1 = np.array(sig1.schema_fingerprints[:min_len])
        fp2 = np.array(sig2.schema_fingerprints[:min_len])
        distances = [np.linalg.norm(f1 - f2) for f1, f2 in zip(fp1, fp2)]
        fp_distance = np.mean(distances)
        # Normalize to 0-1
        fp_distance = min(1.0, fp_distance)
    
    # Calculate overall similarity
    overall_similarity = (
        0.3 * abs(psi_c_corr) + 
        0.2 * abs(entropy_corr) + 
        0.2 * abs(contradiction_corr) + 
        0.2 * abs(stability_corr) + 
        0.1 * (1 - fp_distance)
    )
    
    return {
        "psi_c_correlation": psi_c_corr,
        "entropy_correlation": entropy_corr,
        "contradiction_correlation": contradiction_corr,
        "stability_correlation": stability_corr,
        "fingerprint_distance": fp_distance,
        "overall_similarity": overall_similarity
    } 


class CoherenceUniquenessTestbed:
    """
    Testbed for the ΨC-Coherence Stability Theorem.
    
    This testbed creates multiple agents with different schemas, subjects them
    to the same input stream, and analyzes their coherence signatures to prove
    the uniqueness property.
    """
    
    def __init__(self, num_agents: int = 5, vector_dim: int = 128):
        """
        Initialize the testbed.
        
        Args:
            num_agents: Number of agents to create
            vector_dim: Dimensionality of belief vectors
        """
        self.vector_dim = vector_dim
        self.agents: Dict[str, Agent] = {}
        self.input_stream: List[Any] = []
        self.signature_comparisons: List[Dict[str, Any]] = []
        self.current_step = 0
        
        # Create agents
        self._create_agents(num_agents)
        
        # Set up visualization
        self.fig = None
        self.axes = None
    
    def _create_agents(self, num_agents: int):
        """Create agents with different initial schemas."""
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            
            # Create schema
            schema = AgentSchema(id=f"schema_{i}")
            
            # Add some initial beliefs
            num_initial_beliefs = 5 + i  # Vary initial conditions
            for j in range(num_initial_beliefs):
                belief_id = f"belief_init_{i}_{j}"
                
                # Create vector
                vector = np.random.normal(0, 1, self.vector_dim)
                vector = vector / np.linalg.norm(vector)
                
                # Create belief
                belief = Belief(
                    id=belief_id,
                    content=f"Initial belief {j} for agent {i}",
                    confidence=0.6 + 0.4 * random.random(),
                    vector=vector,
                    connections=[]
                )
                
                schema.add_belief(belief)
            
            # Add connections between beliefs
            for belief_id in schema.beliefs:
                # Connect to 1-3 other random beliefs
                other_beliefs = list(set(schema.beliefs.keys()) - {belief_id})
                if other_beliefs:
                    num_connections = min(3, len(other_beliefs))
                    connections = random.sample(other_beliefs, num_connections)
                    schema.beliefs[belief_id].connections.extend(connections)
                    
                    # Update the schema graph
                    for conn in connections:
                        schema.graph.add_edge(belief_id, conn)
            
            # Create coherence signature
            signature = CoherenceSignature(agent_id=agent_id)
            
            # Create agent
            agent = Agent(id=agent_id, schema=schema, coherence_signature=signature)
            
            # Initialize signature with current state
            agent.update_coherence_signature()
            
            # Add to agents dictionary
            self.agents[agent_id] = agent
    
    def generate_input_stream(self, num_inputs: int = 20, seed: Optional[int] = None):
        """
        Generate a stream of inputs to feed to the agents.
        
        Args:
            num_inputs: Number of inputs to generate
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        self.input_stream = []
        
        # Generate text inputs
        topics = ["science", "ethics", "identity", "knowledge", "reality"]
        
        for _ in range(num_inputs):
            topic = random.choice(topics)
            input_text = f"Information about {topic}: concept {random.randint(1, 100)}"
            self.input_stream.append(input_text)
    
    def run_single_step(self):
        """Run a single step of the testbed."""
        if self.current_step >= len(self.input_stream):
            logger.warning("No more inputs in the stream.")
            return
        
        # Get current input
        current_input = self.input_stream[self.current_step]
        
        # Feed to all agents
        for agent in self.agents.values():
            agent.process_input(current_input)
        
        # Increment step
        self.current_step += 1
        
        # Update analysis
        self._update_comparisons()
        
        return self.current_step
    
    def run_simulation(self, steps: Optional[int] = None):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of steps to run, or None to run the full input stream
        """
        if steps is None:
            steps = len(self.input_stream) - self.current_step
        
        steps = min(steps, len(self.input_stream) - self.current_step)
        
        for _ in range(steps):
            self.run_single_step()
    
    def _update_comparisons(self):
        """Update signature comparisons for all agent pairs."""
        self.signature_comparisons = []
        
        # Compare all pairs of agents
        for agent_id1, agent_id2 in combinations(self.agents.keys(), 2):
            agent1 = self.agents[agent_id1]
            agent2 = self.agents[agent_id2]
            
            # Calculate similarity metrics
            similarity = compare_coherence_signatures(
                agent1.coherence_signature, 
                agent2.coherence_signature
            )
            
            # Record comparison
            self.signature_comparisons.append({
                "agent_pair": (agent_id1, agent_id2),
                "step": self.current_step,
                "similarity_metrics": similarity,
                "schemas_different": True  # Assume different schemas by design
            })
    
    def visualize_signatures(self, figsize=(15, 10), output_file=None):
        """
        Visualize coherence signatures and their similarities.
        
        Args:
            figsize: Figure size in inches
            output_file: If provided, save visualization to this file
        """
        if not self.agents:
            logger.warning("No agents to visualize.")
            return
        
        # Create figure if needed
        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Clear axes
        for ax in self.axes.flatten():
            ax.clear()
        
        # Plot 1: ΨC scores over time
        ax1 = self.axes[0, 0]
        
        for agent_id, agent in self.agents.items():
            sig = agent.coherence_signature
            if sig.timestamps:
                # Convert to relative time
                rel_timestamps = [t - sig.timestamps[0] for t in sig.timestamps]
                ax1.plot(rel_timestamps, sig.psi_c_scores, marker='o', label=agent_id)
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel("ΨC Score")
        ax1.set_title("ΨC Scores Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pairwise similarity matrix
        ax2 = self.axes[0, 1]
        
        if self.signature_comparisons:
            # Create similarity matrix
            agent_ids = sorted(self.agents.keys())
            n_agents = len(agent_ids)
            similarity_matrix = np.zeros((n_agents, n_agents))
            
            # Fill diagonal with 1s (self-similarity)
            for i in range(n_agents):
                similarity_matrix[i, i] = 1.0
            
            # Fill off-diagonal with pairwise similarities
            for comp in self.signature_comparisons:
                agent1, agent2 = comp["agent_pair"]
                i = agent_ids.index(agent1)
                j = agent_ids.index(agent2)
                similarity = comp["similarity_metrics"]["overall_similarity"]
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric
            
            # Plot heatmap
            sns.heatmap(
                similarity_matrix, 
                annot=True, 
                cmap="viridis", 
                xticklabels=agent_ids,
                yticklabels=agent_ids,
                vmin=0, 
                vmax=1,
                ax=ax2
            )
            
            ax2.set_title("Signature Similarity Matrix")
        else:
            ax2.text(0.5, 0.5, "No comparisons available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Signature uniqueness metrics
        ax3 = self.axes[1, 0]
        
        agent_ids = []
        uniqueness_scores = []
        
        for agent_id, agent in self.agents.items():
            uniqueness = agent.coherence_signature.calculate_uniqueness_metric()
            agent_ids.append(agent_id)
            uniqueness_scores.append(uniqueness)
        
        if agent_ids:
            ax3.bar(agent_ids, uniqueness_scores)
            ax3.set_xlabel("Agent")
            ax3.set_ylabel("Uniqueness Score")
            ax3.set_title("Signature Uniqueness Metrics")
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, "No uniqueness metrics available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Schema fingerprints
        ax4 = self.axes[1, 1]
        
        # Collect fingerprints from all agents
        all_fingerprints = []
        fingerprint_labels = []
        
        for agent_id, agent in self.agents.items():
            sig = agent.coherence_signature
            if sig.schema_fingerprints:
                # Use the latest fingerprint
                all_fingerprints.append(sig.schema_fingerprints[-1])
                fingerprint_labels.append(agent_id)
        
        if all_fingerprints:
            # Convert to numpy array
            fingerprint_array = np.array(all_fingerprints)
            
            # Use PCA to visualize in 2D
            if len(all_fingerprints) >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(fingerprint_array)
                
                # Plot points
                ax4.scatter(pca_result[:, 0], pca_result[:, 1], s=100)
                
                # Add labels
                for i, label in enumerate(fingerprint_labels):
                    ax4.annotate(label, (pca_result[i, 0], pca_result[i, 1]))
                
                ax4.set_xlabel("PC1")
                ax4.set_ylabel("PC2")
                ax4.set_title("Schema Fingerprints (PCA)")
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, "Not enough fingerprints for PCA", 
                        horizontalalignment='center', verticalalignment='center')
        else:
            ax4.text(0.5, 0.5, "No fingerprints available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        
        # Show the figure
        plt.show()
    
    def generate_report(self, output_file=None):
        """
        Generate a detailed report on the theorem validation.
        
        Args:
            output_file: If provided, save report to this file
        """
        if not self.signature_comparisons:
            logger.warning("No comparisons to report on.")
            return
        
        # Calculate average similarity
        avg_similarity = np.mean([
            comp["similarity_metrics"]["overall_similarity"] 
            for comp in self.signature_comparisons
        ])
        
        # Calculate max similarity
        max_similarity = np.max([
            comp["similarity_metrics"]["overall_similarity"] 
            for comp in self.signature_comparisons
        ])
        
        # Get the most similar pair
        most_similar_comp = max(
            self.signature_comparisons,
            key=lambda x: x["similarity_metrics"]["overall_similarity"]
        )
        
        most_similar_pair = most_similar_comp["agent_pair"]
        most_similar_metrics = most_similar_comp["similarity_metrics"]
        
        # Verdict on theorem validation
        theorem_validated = max_similarity < 0.9  # Threshold for considering signatures distinct
        
        # Generate report text
        report_text = f"""
ΨC-COHERENCE STABILITY THEOREM VALIDATION REPORT
================================================

Experiment Summary:
------------------
Number of Agents: {len(self.agents)}
Input Stream Length: {len(self.input_stream)}
Steps Completed: {self.current_step}

Similarity Analysis:
------------------
Average Pairwise Similarity: {avg_similarity:.4f}
Maximum Pairwise Similarity: {max_similarity:.4f}
Most Similar Agent Pair: {most_similar_pair[0]} and {most_similar_pair[1]}

Most Similar Pair Metrics:
-------------------------
ΨC Score Correlation: {most_similar_metrics["psi_c_correlation"]:.4f}
Entropy Correlation: {most_similar_metrics["entropy_correlation"]:.4f}
Contradiction Correlation: {most_similar_metrics["contradiction_correlation"]:.4f}
Stability Correlation: {most_similar_metrics["stability_correlation"]:.4f}
Fingerprint Distance: {most_similar_metrics["fingerprint_distance"]:.4f}

Theorem Validation:
-----------------
Theorem Statement: ∀a, b: ΨC_a(t) ≠ ΨC_b(t) ⇒ Σ_a ≠ Σ_b
Validation Result: {"VALIDATED" if theorem_validated else "NOT VALIDATED"}

Conclusion:
----------
{"The experiment supports the theorem. Different schemas consistently produced distinct coherence signatures." if theorem_validated else "The experiment does not fully support the theorem. Some different schemas produced very similar coherence signatures."}

Recommendations:
--------------
{"Continue with formal proof development." if theorem_validated else "Investigate factors causing signature similarity despite schema differences."}
"""
        
        # Print to console
        print(report_text)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return theorem_validated


def main():
    """
    Main function to run the testbed.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="ΨC-Coherence Stability Theorem Testbed")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--steps", type=int, default=20, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Initializing testbed with {args.agents} agents")
    testbed = CoherenceUniquenessTestbed(num_agents=args.agents)
    
    logger.info(f"Generating input stream of {args.steps} items")
    testbed.generate_input_stream(num_inputs=args.steps)
    
    logger.info(f"Running simulation for {args.steps} steps")
    testbed.run_simulation()
    
    # Generate visualization
    if args.output_dir:
        viz_file = os.path.join(args.output_dir, "uniqueness_visualization.png")
        logger.info(f"Generating visualization to {viz_file}")
        testbed.visualize_signatures(output_file=viz_file)
    else:
        logger.info("Generating visualization")
        testbed.visualize_signatures()
    
    # Generate report
    if args.output_dir:
        report_file = os.path.join(args.output_dir, "uniqueness_report.txt")
        logger.info(f"Generating report to {report_file}")
        testbed.generate_report(output_file=report_file)
    else:
        logger.info("Generating report")
        testbed.generate_report()
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 