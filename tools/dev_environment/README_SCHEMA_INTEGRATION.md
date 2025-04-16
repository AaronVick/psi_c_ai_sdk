# Memory Schema Integration

The Memory Schema Integration module bridges the ΨC-AI SDK Memory Sandbox with schema graph functionality, enabling powerful semantic analysis of agent memory structures.

## Features

- **Schema Graph Building**: Convert memory structures into semantic knowledge graphs
- **Memory Cluster Detection**: Identify related memory clusters using graph community detection
- **Concept Suggestion**: Generate higher-level concepts from memory clusters
- **Relationship Analysis**: Analyze semantic relationships between memories
- **Memory Visualization**: Visualize memory schemas as interactive graphs
- **JSON Export/Import**: Save and load schema graphs for persistent analysis

## Getting Started

### Prerequisites

- ΨC-AI SDK Development Environment
- Memory Sandbox initialized with memories

### Basic Usage

```python
from tools.dev_environment.launcher import DevEnvironmentLauncher

# Initialize the development environment with schema integration
launcher = DevEnvironmentLauncher(
    tools_enabled={
        'memory_sandbox': True,
        'schema_integration': True
    }
)

# Access the memory sandbox and schema integration
memory_sandbox = launcher.tools['memory_sandbox']
schema_integration = launcher.tools['schema_integration']

# Create some memories
memory_sandbox.create_synthetic_memory(
    memory_type='semantic',
    content='Neural networks are a fundamental component of deep learning',
    metadata={'tags': ['AI', 'neural networks', 'deep learning']}
)

memory_sandbox.create_synthetic_memory(
    memory_type='semantic',
    content='Convolutional neural networks are specialized for image processing',
    metadata={'tags': ['CNN', 'computer vision', 'deep learning']}
)

# Build a schema graph from memories
schema_integration.build_schema_graph()

# Detect memory clusters
clusters = schema_integration.detect_memory_clusters(min_cluster_size=2)

# Get concept suggestions for a cluster
suggested_concepts = schema_integration.suggest_concepts_from_cluster(clusters[0])

# Visualize the schema graph
schema_integration.visualize_schema_graph()

# Export the schema graph to JSON
schema_integration.export_schema_graph('schema_graph.json')
```

## Web Interface

The Memory Schema Integration is fully integrated with the web interface. To access:

1. Launch the development environment with `launcher.start_web_interface()`
2. Navigate to the Schema tab in the Memory Sandbox interface
3. Use the provided controls to build schemas, detect clusters, and visualize memory relationships

## Command Line Interface

The schema integration can be accessed through the command line:

```bash
python -m tools.dev_environment.launcher schema build
python -m tools.dev_environment.launcher schema detect-clusters
python -m tools.dev_environment.launcher schema suggest-concepts
python -m tools.dev_environment.launcher schema visualize
```

## Advanced Usage

### Custom Memory Analysis

```python
# Define memory filters for specific domains
science_filter = lambda mem: any(tag in mem.metadata.get('tags', []) for tag in 
                                ['science', 'physics', 'biology', 'chemistry'])
art_filter = lambda mem: any(tag in mem.metadata.get('tags', []) for tag in 
                            ['art', 'painting', 'music', 'literature'])

# Analyze relationships between science and art memories
relationships = schema_integration.analyze_relationships(
    filter_a=science_filter,
    filter_b=art_filter,
    relationship_types=['semantic_similarity', 'temporal_proximity']
)

# Find bridging concepts between domains
bridging_concepts = schema_integration.find_bridging_concepts(
    filter_a=science_filter,
    filter_b=art_filter,
    min_similarity=0.6
)
```

### Custom Graph Analysis

```python
# Access the underlying networkx graph
graph = schema_integration.schema_graph

# Calculate centrality measures
import networkx as nx
centrality = nx.betweenness_centrality(graph)

# Find most central memory nodes
central_memories = sorted(
    [(node, score) for node, score in centrality.items() if 'memory_id' in graph.nodes[node]],
    key=lambda x: x[1],
    reverse=True
)[:5]

# Extract subgraphs of interest
emotion_subgraph = schema_integration.extract_subgraph(
    node_filter=lambda n: graph.nodes[n].get('memory_type') == 'emotional'
)
```

### Schema Merging

```python
# Load a previously saved schema
schema_integration.import_schema_graph('previous_schema.json')

# Build a new schema from recent memories
schema_integration.build_schema_graph(
    memories=memory_sandbox.get_memories_by_recency(n=10)
)

# Merge the schemas
merged_schema = schema_integration.merge_schema_graphs(
    schema_a=schema_integration.schema_graph,
    schema_b=schema_integration.previous_schema_graph,
    resolution_strategy='newer_wins'
)
```

## Demo Script

A demonstration script is provided to showcase the core features of the Memory Schema Integration:

```bash
python -m tools.dev_environment.demos.memory_schema_demo
```

The demo includes:
- Creation of themed memory clusters
- Schema graph building and visualization
- Automatic cluster detection
- Concept suggestion
- Memory relationship analysis

## Mathematical Basis

The Memory Schema Integration employs several mathematical approaches:

1. **Graph Theory**: Memory relationships are modeled as a weighted graph where:
   - Memory nodes are connected based on semantic similarity
   - Edge weights are calculated as cos(θ) between memory embeddings
   - Community detection uses the Louvain method for modularity optimization

2. **Concept Extraction**: 
   - TF-IDF weighting to identify key terms
   - LDA (Latent Dirichlet Allocation) for topic modeling
   - Named entity recognition for identifying key entities

3. **Similarity Metrics**:
   - Cosine similarity between memory embeddings
   - Jaccard index for metadata tag overlap
   - Path-based measures for structural relationships in the graph

## API Reference

### MemorySchemaIntegration

The main class that provides schema integration functionality.

**Constructor**:
```python
MemorySchemaIntegration(memory_sandbox, embedding_model=None)
```

**Key Methods**:
- `build_schema_graph(memories=None, min_similarity=0.5)`: Build a schema graph from memories
- `detect_memory_clusters(min_cluster_size=3, resolution=1.0)`: Detect memory clusters in the graph
- `suggest_concepts_from_cluster(cluster)`: Generate concept suggestions for a memory cluster
- `add_concept_node(concept_name, connected_memories, metadata={})`: Add a concept node to the graph
- `analyze_relationships(filter_a, filter_b, relationship_types)`: Analyze relationships between memory sets
- `update_schema_with_new_memories(memories)`: Update existing schema with new memories
- `find_related_concepts(memory_id, max_distance=2)`: Find concepts related to a specific memory
- `visualize_schema_graph(highlight_nodes=None)`: Generate a visualization of the schema graph
- `export_schema_graph(filepath)`: Export the schema graph to a JSON file
- `import_schema_graph(filepath)`: Import a schema graph from a JSON file

## Known Limitations

- Performance may degrade with extremely large memory stores (>10,000 memories)
- Custom embedding models must implement the same interface as the default model
- Graph visualization becomes cluttered with large numbers of nodes (>100)
- Schema merging may encounter conflicts with complex overlapping schemas

## Contributing

Contributions to the Memory Schema Integration are welcome. To contribute:

1. Fork the repository
2. Create your feature branch
3. Add tests for your new functionality
4. Submit a pull request

## License

This project is licensed under the same license as the ΨC-AI SDK. 