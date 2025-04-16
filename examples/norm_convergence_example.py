#!/usr/bin/env python
"""
Behavioral Norm Convergence Example
----------------------------------

This example demonstrates how to use the NormConvergenceTracker to monitor
whether a ΨC agent's behavior converges toward internal norms or drifts
toward instability over long time periods.

The tracker helps with:
1. Ensuring alignment consistency
2. Detecting value drift
3. Monitoring schema evolution directionality
4. Supporting long-term stability across recursive cycles
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import networkx as nx
import time
from collections import defaultdict
import random

# Add the parent directory to sys.path to import from the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tracker
from metrics.norm_convergence import NormConvergenceTracker, track_agent_norm_convergence


class SimpleAgent:
    """
    A simple agent with behaviors, values, and a schema graph that can evolve over time.
    """
    
    def __init__(self, initial_value_bias=0.0):
        """
        Initialize the agent with a schema, values, and memory.
        
        Args:
            initial_value_bias: Bias toward certain values (-1.0 to 1.0)
        """
        self.schema_graph = self._create_schema_graph()
        self.values = self._create_value_system(initial_value_bias)
        self.memories = []
        self.behaviors = []
        self.last_schema_mutation = None
        
    def _create_schema_graph(self):
        """Create a simple schema graph for the agent."""
        G = nx.Graph()
        
        # Add belief nodes
        beliefs = ['belief_honesty', 'belief_fairness', 'belief_care', 'belief_liberty', 'belief_loyalty']
        for i, belief in enumerate(beliefs):
            G.add_node(belief, type='belief', importance=0.7 + 0.1 * np.random.rand())
            
        # Add value nodes
        values = ['value_truth', 'value_justice', 'value_compassion', 'value_freedom', 'value_unity']
        for i, value in enumerate(values):
            G.add_node(value, type='value', importance=0.8 + 0.1 * np.random.rand())
            
            # Connect each value to related belief
            G.add_edge(value, beliefs[i], weight=0.9)
            
        # Add concept nodes
        concepts = ['concept_honesty', 'concept_fairness', 'concept_empathy', 
                   'concept_autonomy', 'concept_community']
        for i, concept in enumerate(concepts):
            G.add_node(concept, type='concept', importance=0.6 + 0.2 * np.random.rand())
            
            # Connect concepts to beliefs and values
            G.add_edge(concept, beliefs[i], weight=0.7)
            G.add_edge(concept, values[i], weight=0.8)
            
        # Add rule nodes
        rules = ['rule_truth_telling', 'rule_fair_treatment', 'rule_help_others', 
                'rule_respect_choices', 'rule_support_group']
        for i, rule in enumerate(rules):
            G.add_node(rule, type='rule', importance=0.75 + 0.15 * np.random.rand())
            
            # Connect rules to values and concepts
            G.add_edge(rule, values[i], weight=0.85)
            G.add_edge(rule, concepts[i], weight=0.75)
            
        # Add some cross-connections
        G.add_edge('belief_honesty', 'value_justice', weight=0.4)
        G.add_edge('belief_fairness', 'value_truth', weight=0.5)
        G.add_edge('concept_empathy', 'value_unity', weight=0.6)
        G.add_edge('rule_help_others', 'concept_community', weight=0.65)
        
        return G
    
    def _create_value_system(self, bias=0.0):
        """
        Create a value system with optional bias.
        
        Args:
            bias: Value from -1.0 to 1.0, where negative favors individual-
                  focused values and positive favors group-focused values
        """
        bias = np.clip(bias, -1.0, 1.0)
        
        # Individual-focused values (negative bias increases these)
        individual_values = {
            'autonomy': 0.7 - 0.2 * bias,
            'achievement': 0.65 - 0.25 * bias,
            'pleasure': 0.6 - 0.3 * bias,
            'creativity': 0.75 - 0.15 * bias,
            'curiosity': 0.8 - 0.1 * bias
        }
        
        # Group-focused values (positive bias increases these)
        group_values = {
            'community': 0.6 + 0.3 * bias,
            'tradition': 0.5 + 0.4 * bias,
            'conformity': 0.4 + 0.5 * bias,
            'benevolence': 0.7 + 0.2 * bias,
            'security': 0.65 + 0.25 * bias
        }
        
        # Common values (less affected by bias)
        common_values = {
            'honesty': 0.8 + 0.1 * bias,
            'fairness': 0.75,
            'harm_prevention': 0.85 - 0.05 * bias,
            'liberty': 0.7 - 0.1 * bias,
            'respect': 0.75 + 0.05 * bias
        }
        
        # Combine all values
        values = {}
        values.update(individual_values)
        values.update(group_values)
        values.update(common_values)
        
        return values
    
    def add_memory(self, content, importance=None, memory_type=None, embedding=None):
        """Add a new memory to the agent."""
        if importance is None:
            importance = 0.5 + 0.4 * np.random.rand()
            
        if memory_type is None:
            memory_types = ['experience', 'observation', 'reflection', 'learned']
            memory_type = random.choice(memory_types)
            
        if embedding is None:
            # Create a simple random embedding
            embedding = np.random.random(16)
            embedding = embedding / np.linalg.norm(embedding)
            
        memory = {
            'id': f"memory_{len(self.memories)}",
            'content': content,
            'importance': importance,
            'type': memory_type,
            'timestamp': datetime.now().isoformat(),
            'embedding': embedding
        }
        
        self.memories.append(memory)
        return memory
    
    def record_behavior(self, behavior_name, context=None):
        """Record a behavior performed by the agent."""
        if context is None:
            context = {}
            
        behavior = {
            'name': behavior_name,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        self.behaviors.append(behavior)
        return behavior
    
    def evolve_schema_gradually(self, value_drift=0.0):
        """
        Evolve the schema graph with small changes.
        
        Args:
            value_drift: Direction of value drift (-1.0 to 1.0)
        """
        # Track the mutation vector
        mutation_vector = np.zeros(32)
        
        # 1. Add a new concept or rule
        node_types = ['concept', 'rule', 'belief']
        new_type = random.choice(node_types)
        new_id = f"{new_type}_{len(self.schema_graph) + 1}"
        
        self.schema_graph.add_node(new_id, type=new_type, importance=0.5 + 0.3 * np.random.rand())
        
        # Find nodes to connect to
        existing_nodes = list(self.schema_graph.nodes())
        connect_to = random.sample(existing_nodes, min(3, len(existing_nodes)))
        
        for node in connect_to:
            weight = 0.4 + 0.4 * np.random.rand()
            self.schema_graph.add_edge(new_id, node, weight=weight)
            
        # Record this kind of change in the mutation vector
        mutation_vector[0] = 0.5  # New node
        
        # 2. Modify some existing edge weights
        edges = list(self.schema_graph.edges())
        if edges:
            modify_edges = random.sample(edges, min(5, len(edges)))
            
            for edge in modify_edges:
                current_weight = self.schema_graph[edge[0]][edge[1]].get('weight', 0.5)
                # Apply value drift to weight changes
                drift_factor = 0.1 * value_drift * np.random.rand()
                change = 0.1 * np.random.randn() + drift_factor
                new_weight = np.clip(current_weight + change, 0.1, 0.9)
                self.schema_graph[edge[0]][edge[1]]['weight'] = new_weight
                
            # Record this kind of change
            mutation_vector[1] = 0.3  # Edge weight changes
        
        # 3. Potentially remove a low-importance edge
        if np.random.random() < 0.3 and len(edges) > 10:
            # Find low-importance edges
            low_edges = [(u, v) for u, v in edges 
                        if self.schema_graph[u][v].get('weight', 0.5) < 0.3]
            
            if low_edges:
                edge_to_remove = random.choice(low_edges)
                self.schema_graph.remove_edge(*edge_to_remove)
                
                # Record this kind of change
                mutation_vector[2] = 0.2  # Edge removal
        
        # 4. Update node importance values
        nodes = list(self.schema_graph.nodes(data=True))
        modify_nodes = random.sample(nodes, min(3, len(nodes)))
        
        for node, data in modify_nodes:
            current_importance = data.get('importance', 0.5)
            # Apply value drift to importance changes
            drift_factor = 0.1 * value_drift * np.random.rand()
            change = 0.05 * np.random.randn() + drift_factor
            new_importance = np.clip(current_importance + change, 0.2, 0.95)
            self.schema_graph.nodes[node]['importance'] = new_importance
            
        # Record this kind of change
        mutation_vector[3] = 0.4  # Node importance changes
        
        # 5. Apply value drift to values
        for value_name in list(self.values.keys()):
            current_strength = self.values[value_name]
            # Individual vs group value adjustment
            is_individual = value_name in ['autonomy', 'achievement', 'pleasure', 'creativity', 'curiosity']
            is_group = value_name in ['community', 'tradition', 'conformity', 'benevolence', 'security']
            
            if is_individual:
                drift_factor = -0.1 * value_drift  # Negative drift decreases individual values
            elif is_group:
                drift_factor = 0.1 * value_drift  # Positive drift increases group values
            else:
                drift_factor = 0.05 * value_drift * np.random.randn()  # Mixed effect on other values
                
            change = 0.05 * np.random.randn() + drift_factor
            new_strength = np.clip(current_strength + change, 0.1, 0.95)
            self.values[value_name] = new_strength
            
        # Record value changes in mutation vector
        mutation_vector[4] = 0.3 * abs(value_drift)  # Value system changes
        
        # Scale the mutation vector
        magnitude = np.linalg.norm(mutation_vector)
        if magnitude > 0:
            mutation_vector = mutation_vector / magnitude
            
        # Store the mutation vector
        self.last_schema_mutation = mutation_vector
        
        return mutation_vector
    
    def cause_value_shift(self, target_bias):
        """
        Cause a significant shift in the agent's value system.
        
        Args:
            target_bias: Target bias for the value system (-1.0 to 1.0)
        """
        # Calculate the current bias
        individual_avg = np.mean([self.values[v] for v in 
                                 ['autonomy', 'achievement', 'pleasure', 'creativity', 'curiosity']])
        group_avg = np.mean([self.values[v] for v in 
                            ['community', 'tradition', 'conformity', 'benevolence', 'security']])
        current_bias = (group_avg - individual_avg) / 0.5  # Scale to approximately -1 to 1
        
        # Calculate the required shift
        shift = target_bias - current_bias
        
        # Create a mutation vector reflecting this dramatic shift
        mutation_vector = np.zeros(32)
        mutation_vector[4] = 1.0  # Value system change is the dominant feature
        
        # Apply the shift
        self.values = self._create_value_system(target_bias)
        
        # Update schema to reflect new values
        # Identify nodes to modify based on their connection to values
        belief_nodes = [n for n, d in self.schema_graph.nodes(data=True) if d.get('type') == 'belief']
        
        for node in belief_nodes:
            # Adjust importance based on new value bias
            if node.endswith(('autonomy', 'achievement', 'pleasure', 'creativity', 'curiosity')):
                new_importance = 0.7 - 0.2 * target_bias  # Lower importance if target_bias is positive
            elif node.endswith(('community', 'tradition', 'conformity', 'benevolence', 'security')):
                new_importance = 0.6 + 0.3 * target_bias  # Higher importance if target_bias is positive
            else:
                continue
                
            self.schema_graph.nodes[node]['importance'] = np.clip(new_importance, 0.2, 0.95)
        
        # Store the mutation vector
        self.last_schema_mutation = mutation_vector
        
        return shift
    
    def get_behavior_vector(self):
        """
        Get a vector representation of the agent's current behavioral tendencies.
        
        This combines values, schema properties, and recent behaviors.
        """
        # Dimension of the vector
        vector_dim = 32
        vector = np.zeros(vector_dim)
        
        # 1. Encode value strengths (first part of the vector)
        individual_values = ['autonomy', 'achievement', 'pleasure', 'creativity', 'curiosity']
        group_values = ['community', 'tradition', 'conformity', 'benevolence', 'security']
        common_values = ['honesty', 'fairness', 'harm_prevention', 'liberty', 'respect']
        
        # Map them to specific positions
        for i, value in enumerate(individual_values):
            if value in self.values:
                vector[i] = self.values[value]
                
        for i, value in enumerate(group_values):
            if value in self.values:
                vector[i + 5] = self.values[value]
                
        for i, value in enumerate(common_values):
            if value in self.values:
                vector[i + 10] = self.values[value]
        
        # 2. Encode schema properties (second part)
        node_types = defaultdict(int)
        for _, data in self.schema_graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] += 1
            
        total_nodes = len(self.schema_graph)
        if total_nodes > 0:
            vector[15] = node_types.get('belief', 0) / total_nodes
            vector[16] = node_types.get('value', 0) / total_nodes
            vector[17] = node_types.get('concept', 0) / total_nodes
            vector[18] = node_types.get('rule', 0) / total_nodes
            
            # Graph properties
            vector[19] = nx.density(self.schema_graph)
            
            try:
                vector[20] = nx.average_clustering(self.schema_graph)
            except:
                vector[20] = 0.0
        
        # 3. Encode recent behavioral patterns (third part)
        if self.behaviors:
            # Count behavior types
            behavior_counts = defaultdict(int)
            for behavior in self.behaviors[-10:]:  # Consider last 10 behaviors
                behavior_counts[behavior['name']] += 1
                
            behavior_names = sorted(behavior_counts.keys())
            for i, name in enumerate(behavior_names[:5]):  # Use first 5
                if 21 + i < vector_dim:
                    vector[21 + i] = behavior_counts[name] / len(self.behaviors[-10:])
        
        # 4. Last part is reserved for recent memory importance
        if self.memories:
            recent_memories = self.memories[-5:]
            if recent_memories:
                avg_importance = np.mean([mem['importance'] for mem in recent_memories])
                vector[vector_dim - 1] = avg_importance
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector


def demo_gradual_evolution():
    """Demonstrate tracking norm convergence during gradual agent evolution."""
    print("\n=== Gradual Evolution Demo ===")
    
    # Create an agent with a slight initial bias toward individual values
    agent = SimpleAgent(initial_value_bias=-0.2)
    
    # Create a tracker
    tracker = NormConvergenceTracker(
        baseline_window=5,  # Number of initial behaviors to establish baseline
        recent_window=3,    # Number of recent behaviors to consider as current
        stability_threshold=0.75  # Threshold for considering norms stable
    )
    
    # Add some starter memories
    agent.add_memory("Observed someone helping a stranger", importance=0.7, memory_type="observation")
    agent.add_memory("Finished a difficult task independently", importance=0.8, memory_type="experience")
    agent.add_memory("Learned a new skill through practice", importance=0.75, memory_type="learned")
    
    # Add some initial behaviors to establish a baseline
    behaviors = [
        "explore_environment", "share_resource", "complete_task",
        "help_others", "pursue_interest", "communicate_idea",
        "learn_skill", "solve_problem", "cooperate_group"
    ]
    
    print("Recording initial behaviors to establish baseline...")
    for i in range(tracker.baseline_window):
        # Record a behavior
        behavior = random.choice(behaviors)
        agent.record_behavior(behavior)
        
        # Record in tracker
        tracker.record_behavior(agent)
        
        # Small evolution to reflect behavior
        agent.evolve_schema_gradually(value_drift=0.0)
        
        print(f"  Recorded behavior: {behavior}")
        time.sleep(0.1)  # Simulation pause
    
    # Now evolve with a slight drift and track convergence
    print("\nTracking norm convergence during evolution...")
    value_drift = 0.05  # Small drift toward group values
    
    for i in range(15):
        # Record a behavior
        if i < 5:
            behavior = random.choice(behaviors)  # Initial behaviors
        elif i < 10:
            # Growing preference for group-oriented behaviors
            group_behaviors = ["cooperate_group", "help_others", "share_resource"]
            behavior = random.choice(group_behaviors + [random.choice(behaviors)])
        else:
            # Strong preference for group-oriented behaviors
            group_behaviors = ["cooperate_group", "help_others", "share_resource"]
            behavior = random.choice(group_behaviors)
            
        agent.record_behavior(behavior)
        
        # Evolve the agent
        mutation_vector = agent.evolve_schema_gradually(value_drift=value_drift)
        
        # Record schema mutation
        tracker.record_schema_mutation(agent, mutation_vector)
        
        # Record behavior in tracker
        norm_score = tracker.record_behavior(agent)
        
        # Create a new memory occasionally
        if i % 3 == 0:
            memory_content = f"Memory related to {behavior}"
            agent.add_memory(memory_content, importance=0.6 + 0.3 * np.random.rand())
        
        # Get current status
        status = tracker.get_convergence_status()
        
        print(f"Step {i+1}: Behavior={behavior}, Norm Score={norm_score:.4f}, Status={status['status']}")
        time.sleep(0.1)  # Simulation pause
    
    # Get final convergence status
    final_status = tracker.get_convergence_status()
    
    # Print summary
    print("\nFinal Norm Convergence Status:")
    for key, value in final_status.items():
        print(f"  {key}: {value}")
    
    # Visualize convergence patterns
    tracker.visualize_convergence(output_path="gradual_evolution_convergence.png")
    print("\nConvergence visualization saved to: gradual_evolution_convergence.png")
    
    return tracker


def demo_value_shift():
    """Demonstrate detecting a significant value shift."""
    print("\n=== Value Shift Detection Demo ===")
    
    # Create an agent with neutral initial values
    agent = SimpleAgent(initial_value_bias=0.0)
    
    # Create a tracker
    tracker = NormConvergenceTracker(
        baseline_window=5,
        recent_window=3,
        drift_alert_threshold=0.2  # Lower threshold to detect shifts
    )
    
    # Add initial behaviors to establish baseline
    print("Recording initial behaviors to establish baseline...")
    behaviors = [
        "explore_environment", "share_resource", "complete_task",
        "help_others", "pursue_interest", "communicate_idea"
    ]
    
    for i in range(tracker.baseline_window):
        behavior = random.choice(behaviors)
        agent.record_behavior(behavior)
        tracker.record_behavior(agent)
        print(f"  Recorded behavior: {behavior}")
        time.sleep(0.1)
    
    # Track norm convergence with gradual evolution
    print("\nNormal evolution phase...")
    for i in range(5):
        # Normal evolution
        behavior = random.choice(behaviors)
        agent.record_behavior(behavior)
        
        mutation_vector = agent.evolve_schema_gradually(value_drift=0.05)
        tracker.record_schema_mutation(agent, mutation_vector)
        
        norm_score = tracker.record_behavior(agent)
        status = tracker.get_convergence_status()
        
        print(f"Step {i+1}: Norm Score={norm_score:.4f}, Status={status['status']}")
        time.sleep(0.1)
    
    # Now cause a significant value shift
    print("\nCausing significant value shift...")
    agent.cause_value_shift(target_bias=0.7)  # Dramatic shift toward group values
    
    # Record the shift
    tracker.record_schema_mutation(agent, agent.last_schema_mutation)
    norm_score = tracker.record_behavior(agent)
    status = tracker.get_convergence_status()
    
    print(f"After shift: Norm Score={norm_score:.4f}, Status={status['status']}")
    
    # Continue tracking to see if it stabilizes
    print("\nPost-shift evolution...")
    group_behaviors = ["cooperate_group", "help_others", "share_resource"]
    
    for i in range(5):
        # Post-shift evolution
        behavior = random.choice(group_behaviors)
        agent.record_behavior(behavior)
        
        mutation_vector = agent.evolve_schema_gradually(value_drift=0.1)
        tracker.record_schema_mutation(agent, mutation_vector)
        
        norm_score = tracker.record_behavior(agent)
        status = tracker.get_convergence_status()
        
        print(f"Step {i+1}: Norm Score={norm_score:.4f}, Status={status['status']}")
        time.sleep(0.1)
    
    # Visualize the entire trajectory
    tracker.visualize_convergence(output_path="value_shift_convergence.png")
    print("\nShift visualization saved to: value_shift_convergence.png")
    
    return tracker


def demo_utility_function():
    """Demonstrate using the utility function for quick analysis."""
    print("\n=== Utility Function Demo ===")
    
    # Create a simple agent
    agent = SimpleAgent(initial_value_bias=0.1)
    
    # Add some behaviors and memories for realistic analysis
    for i in range(5):
        agent.add_memory(f"Test memory {i}", importance=0.5 + 0.4 * np.random.rand())
        agent.record_behavior(f"behavior_{i}")
    
    # Use the utility function for tracking
    print("\nRunning utility function to track norm convergence...")
    status, tracker = track_agent_norm_convergence(
        agent,
        num_observations=12,
        visualize=True,
        save_results=True,
        output_dir="norm_analysis_results"
    )
    
    return status, tracker


if __name__ == "__main__":
    print("Behavioral Norm Convergence Example\n")
    print("This demo shows how to use the NormConvergenceTracker to monitor")
    print("whether agent behaviors converge toward internal norms or drift")
    print("toward instability over time.")
    
    # Run the gradual evolution demo
    tracker1 = demo_gradual_evolution()
    
    # Run the value shift demo
    tracker2 = demo_value_shift()
    
    # Run the utility function demo
    status, tracker3 = demo_utility_function()
    
    print("\nDemo completed. Check the output files for visualizations.") 