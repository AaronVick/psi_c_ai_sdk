# ΨC-AI SDK Memory Sandbox

The Memory Sandbox is a tool for experimentation with agent memory structures in the ΨC-AI SDK Development Environment. It provides features for creating, manipulating, and analyzing memory states.

## Features

- **Memory Creation**: Create synthetic memories of different types with controlled properties
- **Memory Snapshots**: Save and restore memory states for comparison
- **Memory Visualization**: View memory activation patterns and distributions
- **Memory Search**: Find specific memories based on content or metadata
- **Memory Analysis**: Calculate and display detailed memory statistics
- **Memory Simulation**: Test memory dynamics including decay and reinforcement
- **Memory Comparison**: Compare memory states before and after operations
- **Schema Integration**: Analyze memory structures and build knowledge graphs

## Usage

### Web Interface

Launch the Memory Sandbox with the web interface:

```bash
python -m tools.dev_environment.launcher --tool memory_sandbox
```

The web interface provides a graphical way to interact with the Memory Sandbox.

### Python API

Use the Memory Sandbox programmatically:

```python
from tools.dev_environment.memory_sandbox import MemorySandbox, MemoryStore
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration

# Initialize
memory_store = MemoryStore()
sandbox = MemorySandbox(memory_store=memory_store)

# Create memories
sandbox.create_synthetic_memory(
    content="The sky was particularly blue today",
    memory_type="episodic",
    importance=0.7,
    tags=["weather", "observation"]
)

# Take a snapshot
sandbox.take_snapshot("initial_state")

# Simulate memory dynamics
sandbox.simulate_memory_dynamics(duration=5, decay_rate=0.1)

# Compare with initial state
diff = sandbox.compare_with_snapshot("initial_state")
print(f"Changes: {diff}")

# Create schema integration
schema = MemorySchemaIntegration(sandbox)

# Build knowledge graph
schema.build_schema_graph()

# Detect memory clusters
clusters = schema.detect_memory_clusters()

# Generate concept suggestions
concepts = schema.generate_concept_suggestions()

# Visualize schema graph
schema.visualize_schema_graph()
```

### Command-Line Interface

Access the Memory Sandbox via command line:

```bash
python -m tools.dev_environment.memory_sandbox_cli simulate --duration 10 --decay-rate 0.2
python -m tools.dev_environment.memory_sandbox_cli create --count 5 --type episodic
python -m tools.dev_environment.memory_sandbox_cli visualize --pattern activation
python -m tools.dev_environment.memory_sandbox_cli stats --category retrieval
```

## Schema Integration

The Schema Integration feature allows you to analyze memory structures and build knowledge graphs that represent relationships between memories.

### Key Capabilities

- **Build Schema Graphs**: Create graph representations of memory relationships
- **Detect Memory Clusters**: Identify groups of semantically related memories
- **Generate Concept Suggestions**: Discover emergent concepts from memory patterns
- **Visualize Knowledge Structures**: See how memories relate to each other
- **Export/Import Schema Graphs**: Save and restore knowledge structures

### Using Schema Integration

```python
from tools.dev_environment.memory_sandbox import MemorySandbox, MemoryStore
from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration

# Initialize
memory_store = MemoryStore()
sandbox = MemorySandbox(memory_store=memory_store)
schema = MemorySchemaIntegration(sandbox)

# Create test memories
for i in range(10):
    sandbox.create_synthetic_memory(
        content=f"Test memory {i}",
        memory_type="semantic" if i % 2 == 0 else "episodic",
        importance=0.5 + (i / 20)
    )

# Build schema graph
schema.build_schema_graph()

# Detect memory clusters
clusters = schema.detect_memory_clusters(eps=0.6, min_samples=2)
print(f"Detected {len(clusters)} clusters")

# Generate concept suggestions
concepts = schema.generate_concept_suggestions()
for concept_id, concept_data in concepts.items():
    print(f"Concept: {concept_data['concept_name']}")
    print(f"Keywords: {', '.join(concept_data['keywords'])}")

# Find related memories
first_memory_id = list(sandbox.memory_store.memories.keys())[0]
related = schema.find_related_memories(first_memory_id)
print(f"Found {len(related)} related memories")

# Visualize the schema graph
schema.visualize_schema_graph(output_path="schema_graph.png")

# Export the schema graph
schema.export_schema_graph("schema_export.json")

# Generate a knowledge report
report = schema.generate_knowledge_report()
print(f"Report summary: {report['summary']}")
```

### Demo

Try the Schema Integration demo:

```bash
python -m tools.dev_environment.demos.schema_integration_demo
```

This will demonstrate all the features of the Schema Integration component.

## Advanced Usage

### Memory Simulation

Simulate memory dynamics with custom parameters:

```python
# Set up decay parameters
decay_params = {
    "episodic": 0.2,
    "semantic": 0.05,
    "procedural": 0.1,
    "emotional": 0.15
}

# Simulate with custom parameters
sandbox.simulate_memory_dynamics(
    duration=7,
    decay_rates=decay_params,
    reinforcement_probability=0.3,
    random_seed=42
)
```

### Memory Comparison

Compare memory states before and after operations:

```python
# Take an initial snapshot
sandbox.take_snapshot("before_simulation")

# Run a simulation
sandbox.simulate_memory_dynamics(duration=10)

# Compare with the snapshot
diff = sandbox.compare_with_snapshot("before_simulation")
print(f"Changed memories: {len(diff['changed'])}")
print(f"New memories: {len(diff['added'])}")
print(f"Deleted memories: {len(diff['removed'])}")

# Analyze changes by memory type
type_changes = {}
for memory_id in diff['changed']:
    memory = sandbox.memory_store.memories.get(memory_id)
    if memory:
        memory_type = memory.memory_type
        type_changes[memory_type] = type_changes.get(memory_type, 0) + 1

print("Changes by memory type:")
for memory_type, count in type_changes.items():
    print(f"  - {memory_type}: {count}")
```

### Memory Statistics

Get detailed statistics about memory structures:

```python
# Get all statistics
all_stats = sandbox.calculate_memory_statistics()

# Get specific categories
retrieval_stats = sandbox.calculate_memory_statistics(category="retrieval")
type_stats = sandbox.calculate_memory_statistics(category="types")
temporal_stats = sandbox.calculate_memory_statistics(category="temporal")

# Print summary
print(f"Total memories: {all_stats['total_count']}")
print("Memory type distribution:")
for memory_type, count in all_stats['type_distribution'].items():
    print(f"  - {memory_type}: {count} ({count/all_stats['total_count']*100:.1f}%)")
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- NetworkX (for schema integration)
- scikit-learn (for memory clustering)

## Known Limitations

- Large memory stores (>10,000 memories) may experience performance issues
- Visualization of very complex schema graphs may be difficult to interpret
- Memory snapshots are stored in memory and not persisted between sessions 