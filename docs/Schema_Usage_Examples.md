# ΨC Schema Integration: Usage Examples

This document provides practical examples of how to use the ΨC Schema Integration system's key features, focusing on real-world applications of the mathematical models and algorithms.

## Basic Usage

### Initializing the Integration Module

```python
from tools.dev_environment.memory_sandbox import MemorySandbox
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration
import numpy as np

# Initialize memory sandbox
memory_sandbox = MemorySandbox()

# Initialize schema integration with memory sandbox
schema_integration = memory_sandbox.schema_integration

# Alternatively, initialize directly
schema_integration = MemorySchemaIntegration(memory_store=memory_sandbox.memory_store)
```

### Creating Synthetic Memories for Testing

```python
# Create some synthetic memories to work with
memories = []

# Create episodic memory
episodic_memory = memory_sandbox.create_synthetic_memory(
    content="Yesterday I went to the park and saw a red bird.",
    memory_type="episodic",
    importance=0.8,
    embedding=np.random.random(128)  # For real use, replace with actual embedding
)
memories.append(episodic_memory)

# Create semantic memory
semantic_memory = memory_sandbox.create_synthetic_memory(
    content="Birds belong to the class Aves and are characterized by feathers and beaks.",
    memory_type="semantic",
    importance=0.7,
    embedding=np.random.random(128)
)
memories.append(semantic_memory)

# Create procedural memory
procedural_memory = memory_sandbox.create_synthetic_memory(
    content="To identify birds, observe their size, coloration, beak shape, and behavior.",
    memory_type="procedural",
    importance=0.6,
    embedding=np.random.random(128)
)
memories.append(procedural_memory)
```

## Schema Graph Construction

### Building a Schema Graph

```python
# Build the schema graph from memories
schema_integration.create_schema_graph(min_similarity=0.3)

# Access the schema graph
graph = schema_integration.get_schema_graph()

# Print basic graph statistics
print(f"Number of nodes: {len(graph.nodes())}")
print(f"Number of edges: {len(graph.edges())}")

# Display node types
memory_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'memory']
concept_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'concept']
print(f"Memory nodes: {len(memory_nodes)}")
print(f"Concept nodes: {len(concept_nodes)}")
```

### Adding a New Memory and Updating the Graph

```python
# Add a new memory
new_memory = memory_sandbox.create_synthetic_memory(
    content="Red birds like cardinals have bright plumage to attract mates.",
    memory_type="semantic",
    importance=0.75,
    embedding=np.random.random(128)
)

# Update the schema graph to include the new memory
schema_integration.update_schema_graph()

# Check updated graph statistics
updated_graph = schema_integration.get_schema_graph()
print(f"Updated number of nodes: {len(updated_graph.nodes())}")
```

## Memory Cluster Analysis

### Detecting Memory Clusters

```python
# Detect memory clusters with specified threshold
clusters = schema_integration.detect_memory_clusters(threshold=0.6)

# Print cluster information
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    print(f"  Size: {cluster['size']} memories")
    print(f"  Coherence: {cluster['coherence']:.2f}")
    print(f"  Themes: {', '.join(cluster['themes'])}")
    print(f"  Memory IDs: {cluster['memory_ids']}")
    print()
```

### Generating Concept Suggestions

```python
# Generate concept suggestions based on memory clusters
concepts = schema_integration.suggest_concepts(max_concepts=3)

# Print concept suggestions
for i, concept in enumerate(concepts):
    print(f"Concept {i+1}: {concept['name']}")
    print(f"  Description: {concept['description']}")
    print(f"  Importance: {concept['importance']:.2f}")
    print(f"  Themes: {', '.join(concept['themes'])}")
    print(f"  Related Memory IDs: {concept['memory_ids']}")
    print()

# Add a suggested concept to the schema graph
if concepts:
    concept = concepts[0]
    schema_integration.add_concept_node(
        concept_name=concept['name'],
        memory_ids=concept['memory_ids'],
        properties={
            'description': concept['description'],
            'importance': concept['importance']
        }
    )
```

## Mathematical Operations

### Calculating Coherence Accumulation

```python
from tools.dev_environment.schema_math import calculate_coherence_accumulation

# Initial coherence and entropy values
current_coherence = 0.5
entropy = 0.3

# Parameters
alpha = 0.2  # Growth rate
max_coherence = 1.0  # Carrying capacity
beta = 0.1  # Entropy influence factor
dt = 1.0  # Time step

# Calculate updated coherence
new_coherence = calculate_coherence_accumulation(
    current_coherence=current_coherence,
    entropy=entropy,
    alpha=alpha,
    max_coherence=max_coherence,
    beta=beta,
    dt=dt
)

print(f"Initial coherence: {current_coherence}")
print(f"Updated coherence: {new_coherence}")

# Simulate coherence evolution over time
coherence_values = [current_coherence]
for _ in range(10):
    current_coherence = calculate_coherence_accumulation(
        current_coherence=current_coherence,
        entropy=entropy,
        alpha=alpha,
        max_coherence=max_coherence,
        beta=beta,
        dt=dt
    )
    coherence_values.append(current_coherence)
    # Gradually decrease entropy to simulate learning
    entropy = max(0.1, entropy * 0.95)

print(f"Coherence evolution: {coherence_values}")
```

### Shannon Entropy Calculation

```python
from tools.dev_environment.schema_math import calculate_shannon_entropy

# Example memory probability distributions
uniform_distribution = [0.25, 0.25, 0.25, 0.25]
skewed_distribution = [0.7, 0.1, 0.1, 0.1]
deterministic_distribution = [1.0, 0.0, 0.0, 0.0]

# Calculate entropy for each distribution
uniform_entropy = calculate_shannon_entropy(uniform_distribution)
skewed_entropy = calculate_shannon_entropy(skewed_distribution)
deterministic_entropy = calculate_shannon_entropy(deterministic_distribution)

print(f"Uniform distribution entropy: {uniform_entropy:.2f}")
print(f"Skewed distribution entropy: {skewed_entropy:.2f}")
print(f"Deterministic distribution entropy: {deterministic_entropy:.2f}")

# Extract memory importances for entropy calculation
memory_importances = [memory.importance for memory in memories]
total_importance = sum(memory_importances)
memory_probabilities = [imp/total_importance for imp in memory_importances]

# Calculate entropy of the memory system
memory_entropy = calculate_shannon_entropy(memory_probabilities)
print(f"Memory system entropy: {memory_entropy:.2f}")
```

### Phase Transition Detection

```python
from tools.dev_environment.schema_math import calculate_phase_transition_threshold

# Simulate entropy history
entropy_history = [0.8, 0.75, 0.7, 0.65, 0.6, 0.58, 0.55, 0.54, 0.53, 0.52]

# Calculate phase transition threshold
lambda_theta = 1.5  # Scaling factor for variance influence
threshold = calculate_phase_transition_threshold(
    entropy_history=entropy_history,
    lambda_theta=lambda_theta
)

print(f"Entropy history: {entropy_history}")
print(f"Phase transition threshold: {threshold:.2f}")

# Check if the current entropy crosses the threshold
current_entropy = 0.51
is_transition = current_entropy <= threshold
print(f"Current entropy: {current_entropy}")
print(f"Phase transition detected: {is_transition}")
```

## Multi-Agent Coherence

### Calculating Multi-Agent Coherence

```python
from tools.dev_environment.schema_math import calculate_multi_agent_coherence

# Agent coherence values
agent_coherences = {
    "agent1": 0.7,  # First agent has coherence of 0.7
    "agent2": 0.5,  # Second agent has coherence of 0.5
    "agent3": 0.8   # Third agent has coherence of 0.8
}

# Equal agent weights (default)
equal_weights = {
    "agent1": 1.0,
    "agent2": 1.0,
    "agent3": 1.0
}

# Different agent weights (based on expertise or reliability)
different_weights = {
    "agent1": 2.0,  # Expert agent with higher weight
    "agent2": 0.5,  # Less reliable agent with lower weight
    "agent3": 1.0   # Average agent with normal weight
}

# Calculate multi-agent coherence with equal weights
equal_coherence = calculate_multi_agent_coherence(
    agent_coherences=agent_coherences,
    agent_weights=equal_weights
)

# Calculate multi-agent coherence with different weights
weighted_coherence = calculate_multi_agent_coherence(
    agent_coherences=agent_coherences,
    agent_weights=different_weights
)

print(f"Multi-agent coherence (equal weights): {equal_coherence:.4f}")
print(f"Multi-agent coherence (different weights): {weighted_coherence:.4f}")
```

### Implementing Gradient Descent with Noise

```python
from tools.dev_environment.schema_math import gradient_descent_step, add_gaussian_noise
import numpy as np

# Initial coherence values
coherence_values = [0.5, 0.6, 0.7, 0.8]

# Self-loss gradients (would normally be calculated from coherence values)
# Negative gradients increase coherence, positive gradients decrease coherence
gradients = [-0.1, -0.2, 0.1, 0.05]

# Convert to numpy array for noise addition
np_gradients = np.array(gradients)

# Add Gaussian noise for robustness
noisy_gradients = add_gaussian_noise(
    gradient=np_gradients,
    mean=0.0,
    std_dev=0.05
)

print(f"Original gradients: {gradients}")
print(f"Noisy gradients: {noisy_gradients}")

# Perform gradient descent step
learning_rate = 0.1
updated_values = gradient_descent_step(
    coherence_values=coherence_values,
    self_loss_gradient=noisy_gradients,
    learning_rate=learning_rate
)

print(f"Initial coherence values: {coherence_values}")
print(f"Updated coherence values: {updated_values}")
```

## Schema Health Monitoring

### Calculating Schema Health Index

```python
# Simplified schema health calculation
def calculate_schema_health(coherence, entropy, alignment, drift_rate):
    # Weights for different components
    w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
    
    # Calculate health index
    health_index = w1 * coherence - w2 * entropy + w3 * alignment - w4 * drift_rate
    
    # Ensure result is in [0, 1] range
    return max(0.0, min(1.0, health_index))

# Example schema metrics
coherence = 0.65  # Average coherence
entropy = 0.4     # System entropy
alignment = 0.7   # Alignment score
drift_rate = 0.2  # Schema drift rate

# Calculate health index
health_index = calculate_schema_health(coherence, entropy, alignment, drift_rate)
print(f"Schema Health Index: {health_index:.4f}")

# Track health over time
health_history = []
for i in range(10):
    # Simulate changing conditions
    coherence += 0.02  # Improving coherence
    entropy -= 0.01    # Decreasing entropy
    alignment += 0.01  # Improving alignment
    drift_rate -= 0.01 # Decreasing drift
    
    # Keep values in bounds
    coherence = min(1.0, coherence)
    entropy = max(0.1, entropy)
    alignment = min(1.0, alignment)
    drift_rate = max(0.0, drift_rate)
    
    # Calculate and store health
    health = calculate_schema_health(coherence, entropy, alignment, drift_rate)
    health_history.append(health)

print(f"Health evolution: {health_history}")
```

## Visualization

### Visualizing Schema Graph

```python
import matplotlib.pyplot as plt
import networkx as nx

# Get schema graph from integration
graph = schema_integration.get_schema_graph()

# Create a simple visualization function
def visualize_schema_graph(graph, min_edge_weight=0.3):
    # Filter edges by weight
    filtered_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                      if d.get('weight', 0) >= min_edge_weight]
    
    # Create a subgraph with the filtered edges
    visible_nodes = set()
    for u, v in filtered_edges:
        visible_nodes.add(u)
        visible_nodes.add(v)
    
    if not visible_nodes:
        print("No visible nodes with current parameters.")
        return None
    
    # Create subgraph
    subgraph = graph.subgraph(visible_nodes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Node positions
    pos = nx.spring_layout(subgraph, k=0.3, seed=42)
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    
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
            else:
                node_colors.append('gray')
        elif node_type == 'concept':
            node_colors.append('yellow')
        else:
            node_colors.append('gray')
        
        # Set node size based on importance
        importance = node_data.get('importance', 0.5)
        node_sizes.append(300 * importance + 100)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax
    )
    
    # Draw edges
    edge_widths = [subgraph.edges[u, v].get('weight', 0.5) * 3 for u, v in subgraph.edges()]
    nx.draw_networkx_edges(
        subgraph, pos,
        width=edge_widths,
        alpha=0.6,
        ax=ax
    )
    
    # Draw labels
    labels = {node: subgraph.nodes[node].get('label', str(node)) for node in subgraph.nodes()}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=labels,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    plt.title("Memory Schema Graph")
    plt.axis("off")
    
    return fig

# Visualize the graph
fig = visualize_schema_graph(graph, min_edge_weight=0.3)
if fig:
    plt.show()
```

### Plotting Coherence and Entropy Over Time

```python
import matplotlib.pyplot as plt

# Simulate coherence and entropy values over time
coherence_history = [0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.65, 0.68, 0.7, 0.72]
entropy_history = [0.8, 0.75, 0.7, 0.65, 0.6, 0.58, 0.55, 0.54, 0.53, 0.52]
time_steps = list(range(len(coherence_history)))

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot coherence
ax1.plot(time_steps, coherence_history, 'b-o', label='Coherence')
ax1.set_ylabel('Coherence')
ax1.set_title('Schema Coherence Over Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot entropy
ax2.plot(time_steps, entropy_history, 'r-o', label='Entropy')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Entropy')
ax2.set_title('Schema Entropy Over Time')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add phase transition threshold line
threshold = calculate_phase_transition_threshold(entropy_history)
ax2.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Conclusion

These examples demonstrate how to use the ΨC Schema Integration system for various tasks, from basic memory organization to advanced multi-agent coherence modeling. By combining these examples, you can build sophisticated applications that leverage the system's mathematical foundations for knowledge representation, concept generation, and coherence optimization. 