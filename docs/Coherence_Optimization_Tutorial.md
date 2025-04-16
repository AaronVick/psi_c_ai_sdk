# Coherence Optimization Tutorial

This tutorial provides a comprehensive guide to optimizing coherence in the ΨC Schema Integration system, with practical examples and step-by-step instructions.

## Introduction to Coherence Optimization

Coherence is a fundamental metric in the ΨC system that measures how well-integrated and consistent the knowledge structure is. Higher coherence values indicate a more structured, efficient, and usable memory organization.

The optimization process involves:
1. Measuring current coherence
2. Identifying areas for improvement
3. Applying gradient-based optimization techniques
4. Monitoring and adjusting parameters

## Prerequisites

Before starting this tutorial, ensure you have:

- ΨC-AI SDK installed
- Basic understanding of the Schema Integration system
- Python environment with required dependencies:
  - numpy
  - matplotlib
  - networkx
  - scikit-learn (optional, for advanced clustering)

## Setup

First, let's import the necessary modules:

```python
import numpy as np
import matplotlib.pyplot as plt
from tools.dev_environment.memory_sandbox import MemorySandbox
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration
from tools.dev_environment.schema_math import (
    calculate_coherence_accumulation,
    calculate_shannon_entropy,
    calculate_self_loss,
    gradient_descent_step,
    calculate_coherence_kernel,
    add_gaussian_noise
)
```

Create a memory sandbox and initialize the schema integration:

```python
# Initialize the memory sandbox
memory_sandbox = MemorySandbox()

# Access the schema integration module
schema_integration = memory_sandbox.schema_integration
```

## Step 1: Creating a Test Memory Set

Let's create a set of related memories to work with:

```python
# Create a set of related memories about a specific topic
memory_contents = [
    "Machine learning algorithms can be classified as supervised, unsupervised, and reinforcement learning.",
    "Neural networks are a class of machine learning models inspired by the human brain.",
    "Deep learning is a subset of machine learning that uses multi-layered neural networks.",
    "Supervised learning requires labeled training data with inputs and expected outputs.",
    "Decision trees are interpretable machine learning models that make decisions based on feature values.",
    "Clustering algorithms group similar data points together without labeled training data.",
    "Reinforcement learning involves an agent learning to make decisions by receiving rewards or penalties."
]

# Add memories to the sandbox
memories = []
for i, content in enumerate(memory_contents):
    memory = memory_sandbox.create_synthetic_memory(
        content=content,
        memory_type="semantic",
        importance=0.7 + 0.05 * (i % 3),  # Varying importance
        embedding=np.random.random(128)  # In real applications, use proper embeddings
    )
    memories.append(memory)

print(f"Created {len(memories)} memories")
```

## Step 2: Measuring Initial Coherence

Before optimization, let's measure the initial coherence of our memory system:

```python
# Build the schema graph
schema_integration.create_schema_graph(min_similarity=0.3)
graph = schema_integration.get_schema_graph()

# Extract edge weights for coherence calculation
edge_weights = {}
frequencies = {}

for u, v, data in graph.edges(data=True):
    edge_weights[(u, v)] = data.get('weight', 0.5)
    # For this example, we'll use a default frequency of 1.0
    frequencies[(u, v)] = 1.0

# Calculate initial coherence using the coherence kernel
initial_coherence = calculate_coherence_kernel(edge_weights, frequencies)
print(f"Initial coherence: {initial_coherence:.4f}")

# Visualize the initial graph
def visualize_graph(graph, title="Schema Graph"):
    plt.figure(figsize=(10, 8))
    
    # Node positions
    pos = nx.spring_layout(graph, seed=42)
    
    # Edge weights for line thickness
    edge_weights = [graph[u][v].get('weight', 0.5) * 3 for u, v in graph.edges()]
    
    # Draw the graph
    nx.draw(
        graph, pos,
        with_labels=True,
        node_color='lightblue',
        node_size=500,
        font_size=8,
        width=edge_weights,
        edge_color='gray',
        alpha=0.8
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Visualize initial graph
visualize_graph(graph, title=f"Initial Schema Graph (Coherence = {initial_coherence:.4f})")
```

## Step 3: Analyzing the Self-Loss Function

The self-loss function measures how far our system is from the optimal coherence:

```python
# Extract current coherence values from edge weights
coherence_values = list(edge_weights.values())

# Define optimal coherence (target)
optimal_coherence = 0.9

# Calculate self-loss
self_loss = calculate_self_loss(coherence_values, optimal_coherence)
print(f"Self-loss: {self_loss:.4f}")

# Calculate gradients for optimization
# In a simple case, the gradient is proportional to the difference from optimal
self_loss_gradient = [2 * (c - optimal_coherence) for c in coherence_values]

print("Sample of gradients:")
for i, grad in enumerate(self_loss_gradient[:5]):
    print(f"Edge {i}: Coherence = {coherence_values[i]:.4f}, Gradient = {grad:.4f}")
```

## Step 4: Applying Gradient Descent

Now, let's apply gradient descent to optimize the coherence values:

```python
# Define learning rate
learning_rate = 0.1

# Add noise to gradients for robustness (optional)
np_gradients = np.array(self_loss_gradient)
noisy_gradients = add_gaussian_noise(
    gradient=np_gradients,
    mean=0.0,
    std_dev=0.05
)

# Perform gradient descent step
updated_coherence_values = gradient_descent_step(
    coherence_values=coherence_values,
    self_loss_gradient=noisy_gradients,
    learning_rate=learning_rate
)

print("\nCoherence Optimization Results:")
print("Before | After | Change")
for i in range(min(5, len(coherence_values))):
    change = updated_coherence_values[i] - coherence_values[i]
    print(f"{coherence_values[i]:.4f} | {updated_coherence_values[i]:.4f} | {change:+.4f}")

# Calculate new self-loss
new_self_loss = calculate_self_loss(updated_coherence_values, optimal_coherence)
print(f"\nSelf-loss before optimization: {self_loss:.4f}")
print(f"Self-loss after optimization: {new_self_loss:.4f}")
print(f"Improvement: {self_loss - new_self_loss:.4f}")
```

## Step 5: Multi-Step Optimization

A single step of gradient descent might not be enough. Let's implement a multi-step optimization process:

```python
def optimize_coherence(
    coherence_values, 
    optimal_coherence=0.9, 
    learning_rate=0.1,
    noise_std_dev=0.05,
    max_iterations=100,
    convergence_threshold=0.001
):
    """
    Perform multi-step coherence optimization using gradient descent.
    
    Args:
        coherence_values: Initial coherence values
        optimal_coherence: Target coherence value
        learning_rate: Step size for gradient descent
        noise_std_dev: Standard deviation for gradient noise
        max_iterations: Maximum number of iterations
        convergence_threshold: Stop when improvement is below this threshold
        
    Returns:
        Tuple of (optimized_values, loss_history, iterations)
    """
    current_values = coherence_values.copy()
    loss_history = []
    
    # Initial loss
    current_loss = calculate_self_loss(current_values, optimal_coherence)
    loss_history.append(current_loss)
    
    for iteration in range(max_iterations):
        # Calculate gradient
        gradient = [2 * (c - optimal_coherence) for c in current_values]
        
        # Add noise for robustness
        np_gradient = np.array(gradient)
        noisy_gradient = add_gaussian_noise(
            gradient=np_gradient,
            mean=0.0,
            std_dev=noise_std_dev
        )
        
        # Perform gradient descent step
        updated_values = gradient_descent_step(
            coherence_values=current_values,
            self_loss_gradient=noisy_gradient,
            learning_rate=learning_rate
        )
        
        # Calculate new loss
        new_loss = calculate_self_loss(updated_values, optimal_coherence)
        loss_history.append(new_loss)
        
        # Check for convergence
        improvement = current_loss - new_loss
        if improvement < convergence_threshold:
            break
            
        # Update for next iteration
        current_values = updated_values
        current_loss = new_loss
    
    return current_values, loss_history, iteration + 1

# Run multi-step optimization
optimized_values, loss_history, iterations = optimize_coherence(
    coherence_values=coherence_values,
    optimal_coherence=0.9,
    learning_rate=0.1,
    noise_std_dev=0.05
)

print(f"\nMulti-step optimization completed in {iterations} iterations")
print(f"Initial loss: {loss_history[0]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Improvement: {loss_history[0] - loss_history[-1]:.4f}")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Self-Loss')
plt.title('Coherence Optimization Progress')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Step 6: Updating the Schema Graph

Now that we have optimized coherence values, let's update the schema graph:

```python
# Create a copy of the original graph
optimized_graph = graph.copy()

# Update edge weights with optimized values
for i, ((u, v), _) in enumerate(edge_weights.items()):
    if i < len(optimized_values):
        optimized_graph[u][v]['weight'] = optimized_values[i]

# Visualize the optimized graph
visualize_graph(
    optimized_graph, 
    title=f"Optimized Schema Graph (After {iterations} iterations)"
)

# Calculate new overall coherence
new_edge_weights = {}
for i, ((u, v), _) in enumerate(edge_weights.items()):
    if i < len(optimized_values):
        new_edge_weights[(u, v)] = optimized_values[i]

final_coherence = calculate_coherence_kernel(new_edge_weights, frequencies)
print(f"Initial coherence: {initial_coherence:.4f}")
print(f"Final coherence: {final_coherence:.4f}")
print(f"Improvement: {final_coherence - initial_coherence:+.4f}")
```

## Step 7: Integrating with Dynamic Coherence Accumulation

In a dynamic system, coherence evolves over time based on the entropy of the system. Let's simulate this process:

```python
# Calculate memory probability distribution
memory_importances = [memory.importance for memory in memories]
total_importance = sum(memory_importances)
memory_probabilities = [imp/total_importance for imp in memory_importances]

# Calculate entropy
entropy = calculate_shannon_entropy(memory_probabilities)
print(f"System entropy: {entropy:.4f}")

# Simulate coherence accumulation over time
simulation_steps = 20
coherence_trajectory = [initial_coherence]
entropy_trajectory = [entropy]

current_coherence = initial_coherence
current_entropy = entropy

for step in range(simulation_steps):
    # Calculate new coherence using the accumulation model
    new_coherence = calculate_coherence_accumulation(
        current_coherence=current_coherence,
        entropy=current_entropy,
        alpha=0.2,  # Growth rate
        max_coherence=1.0,  # Carrying capacity
        beta=0.1,  # Entropy influence factor
        dt=1.0  # Time step
    )
    
    # Reduce entropy slightly to simulate learning
    new_entropy = max(0.1, current_entropy * 0.95)
    
    # Store values
    coherence_trajectory.append(new_coherence)
    entropy_trajectory.append(new_entropy)
    
    # Update for next step
    current_coherence = new_coherence
    current_entropy = new_entropy

# Plot trajectories
plt.figure(figsize=(12, 6))
time_steps = list(range(simulation_steps + 1))

plt.subplot(1, 2, 1)
plt.plot(time_steps, coherence_trajectory, 'b-o')
plt.xlabel('Time Step')
plt.ylabel('Coherence')
plt.title('Coherence Accumulation')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(time_steps, entropy_trajectory, 'r-o')
plt.xlabel('Time Step')
plt.ylabel('Entropy')
plt.title('Entropy Evolution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Initial coherence: {coherence_trajectory[0]:.4f}")
print(f"Final coherence: {coherence_trajectory[-1]:.4f}")
print(f"Initial entropy: {entropy_trajectory[0]:.4f}")
print(f"Final entropy: {entropy_trajectory[-1]:.4f}")
```

## Step 8: Implementing Adaptive Learning Rate

For better convergence, we can implement an adaptive learning rate:

```python
def adaptive_optimization(
    coherence_values, 
    optimal_coherence=0.9, 
    initial_learning_rate=0.1,
    min_learning_rate=0.01,
    max_learning_rate=0.5,
    adaptation_factor=0.8,
    noise_std_dev=0.05,
    max_iterations=100,
    convergence_threshold=0.001
):
    """
    Perform coherence optimization with adaptive learning rate.
    """
    current_values = coherence_values.copy()
    loss_history = []
    learning_rates = []
    
    # Initial loss
    current_loss = calculate_self_loss(current_values, optimal_coherence)
    loss_history.append(current_loss)
    
    # Current learning rate
    learning_rate = initial_learning_rate
    learning_rates.append(learning_rate)
    
    for iteration in range(max_iterations):
        # Calculate gradient
        gradient = [2 * (c - optimal_coherence) for c in current_values]
        
        # Add noise for robustness
        np_gradient = np.array(gradient)
        noisy_gradient = add_gaussian_noise(
            gradient=np_gradient,
            mean=0.0,
            std_dev=noise_std_dev
        )
        
        # Perform gradient descent step
        updated_values = gradient_descent_step(
            coherence_values=current_values,
            self_loss_gradient=noisy_gradient,
            learning_rate=learning_rate
        )
        
        # Calculate new loss
        new_loss = calculate_self_loss(updated_values, optimal_coherence)
        
        # Adjust learning rate based on performance
        if new_loss < current_loss:
            # Loss improved, increase learning rate slightly
            learning_rate = min(max_learning_rate, learning_rate * 1.1)
            
            # Update for next iteration
            current_values = updated_values
            current_loss = new_loss
        else:
            # Loss got worse, reduce learning rate and try again
            learning_rate = max(min_learning_rate, learning_rate * adaptation_factor)
            
            # Don't update values, try again with lower learning rate
            new_loss = current_loss
        
        # Store history
        loss_history.append(new_loss)
        learning_rates.append(learning_rate)
        
        # Check for convergence
        if iteration > 0:
            improvement = loss_history[iteration-1] - loss_history[iteration]
            if improvement < convergence_threshold and learning_rate <= min_learning_rate * 1.1:
                break
    
    return current_values, loss_history, learning_rates, iteration + 1

# Run adaptive optimization
optimized_values, loss_history, learning_rates, iterations = adaptive_optimization(
    coherence_values=coherence_values,
    optimal_coherence=0.9,
    initial_learning_rate=0.2,
    min_learning_rate=0.01,
    adaptation_factor=0.5
)

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(loss_history, 'b-', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Self-Loss')
plt.title('Loss Evolution with Adaptive Learning Rate')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(learning_rates, 'g-', marker='s')
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Adaptive Learning Rate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Adaptive optimization completed in {iterations} iterations")
print(f"Initial loss: {loss_history[0]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Final learning rate: {learning_rates[-1]:.4f}")
```

## Step 9: Implementing a Complete Optimization Pipeline

Finally, let's integrate everything into a complete coherence optimization pipeline:

```python
def optimize_schema_coherence(schema_integration, optimal_coherence=0.9, verbose=True):
    """
    Complete pipeline for schema coherence optimization.
    
    Args:
        schema_integration: MemorySchemaIntegration instance
        optimal_coherence: Target coherence value
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with optimization results
    """
    # Get the schema graph
    graph = schema_integration.get_schema_graph()
    if not graph:
        if verbose:
            print("No schema graph available. Creating one...")
        schema_integration.create_schema_graph()
        graph = schema_integration.get_schema_graph()
    
    # Extract edge weights and frequencies
    edge_weights = {}
    frequencies = {}
    edge_list = []
    
    for u, v, data in graph.edges(data=True):
        edge_weights[(u, v)] = data.get('weight', 0.5)
        frequencies[(u, v)] = 1.0  # Default frequency
        edge_list.append((u, v))
    
    # Calculate initial coherence
    initial_coherence = calculate_coherence_kernel(edge_weights, frequencies)
    coherence_values = list(edge_weights.values())
    
    if verbose:
        print(f"Initial schema coherence: {initial_coherence:.4f}")
        print(f"Number of edges to optimize: {len(coherence_values)}")
    
    # Optimize coherence values
    optimized_values, loss_history, learning_rates, iterations = adaptive_optimization(
        coherence_values=coherence_values,
        optimal_coherence=optimal_coherence
    )
    
    if verbose:
        print(f"Optimization completed in {iterations} iterations")
        print(f"Initial loss: {loss_history[0]:.4f}")
        print(f"Final loss: {loss_history[-1]:.4f}")
    
    # Update the graph with optimized values
    optimized_graph = graph.copy()
    for i, (u, v) in enumerate(edge_list):
        optimized_graph[u][v]['weight'] = optimized_values[i]
    
    # Calculate final coherence
    optimized_edge_weights = {}
    for i, (u, v) in enumerate(edge_list):
        optimized_edge_weights[(u, v)] = optimized_values[i]
    
    final_coherence = calculate_coherence_kernel(optimized_edge_weights, frequencies)
    
    if verbose:
        print(f"Final schema coherence: {final_coherence:.4f}")
        print(f"Improvement: {final_coherence - initial_coherence:+.4f}")
    
    # Return results
    return {
        "initial_coherence": initial_coherence,
        "final_coherence": final_coherence,
        "improvement": final_coherence - initial_coherence,
        "loss_history": loss_history,
        "iterations": iterations,
        "optimized_graph": optimized_graph
    }

# Execute the complete optimization pipeline
results = optimize_schema_coherence(schema_integration)

# Visualize before and after
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
visualize_graph(graph, title=f"Original Graph\nCoherence = {results['initial_coherence']:.4f}")

plt.subplot(1, 2, 2)
visualize_graph(results['optimized_graph'], title=f"Optimized Graph\nCoherence = {results['final_coherence']:.4f}")

plt.tight_layout()
plt.show()
```

## Conclusion

You've now learned how to:

1. Measure and analyze coherence in a schema graph
2. Apply gradient descent optimization to improve coherence
3. Implement adaptive learning rate strategies
4. Create a complete coherence optimization pipeline
5. Visualize and monitor the optimization process

This approach can be extended to more complex scenarios by:
- Integrating with real-time memory systems
- Applying more sophisticated regularization techniques
- Implementing distributed optimization for large-scale applications
- Combining with other ΨC system components for comprehensive cognitive architectures

## Next Steps

To further enhance your understanding and implementation:

1. Experiment with different memory distributions and relationships
2. Try varying the optimization parameters to see their effects
3. Implement more advanced noise models for robustness
4. Integrate with the full ΨC agent architecture
5. Develop custom coherence metrics for specific application domains 