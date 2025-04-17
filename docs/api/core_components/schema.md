# Schema API

The Schema module provides a graph-based representation of memories and concepts, allowing the ΨC-AI system to model relationships between pieces of information. The schema system forms the foundation for coherence calculation, knowledge representation, and reasoning capabilities.

## Key Classes

### `SchemaNode`

Represents a node in the schema graph, typically a memory or concept.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique identifier for the node |
| `memory_id` | Optional[str] | Reference to a memory UUID if this node represents a memory |
| `label` | str | Human-readable label for the node |
| `node_type` | str | Type of node (e.g., "memory", "concept") |
| `importance` | float | Importance score of this node (0.0 to 1.0) |
| `embedding` | Optional[List[float]] | Vector embedding of the node's content |
| `tags` | List[str] | Tags associated with this node |
| `metadata` | Dict[str, Any] | Additional metadata for the node |

#### Methods

```python
def __init__(self, id: str, memory_id: Optional[str] = None, 
             label: str = "", node_type: str = "memory",
             importance: float = 0.5, embedding: Optional[List[float]] = None,
             tags: List[str] = [], metadata: Dict[str, Any] = {})
```

Initialize a new schema node.

**Parameters:**
- `id`: Unique identifier for the node
- `memory_id`: Optional reference to a memory UUID
- `label`: Human-readable label for the node
- `node_type`: Type of node (default: "memory")
- `importance`: Importance score (default: 0.5)
- `embedding`: Optional vector embedding
- `tags`: Tags for categorization
- `metadata`: Additional metadata

```python
def to_dict(self) -> Dict[str, Any]
```

Convert the node to a dictionary for serialization.

**Returns:**
- Dictionary representation of the node

```python
@classmethod
def from_memory(cls, memory: Memory) -> 'SchemaNode'
```

Create a schema node from a memory object.

**Parameters:**
- `memory`: Memory object to convert

**Returns:**
- SchemaNode created from the memory

### `SchemaEdge`

Represents an edge (relationship) between nodes in the schema graph.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `source_id` | str | ID of the source node |
| `target_id` | str | ID of the target node |
| `weight` | float | Strength of the relationship (0.0 to 1.0) |
| `edge_type` | str | Type of relationship (e.g., "coherence", "causality") |
| `metadata` | Dict[str, Any] | Additional metadata for the edge |

#### Methods

```python
def __init__(self, source_id: str, target_id: str, 
             weight: float = 0.0, edge_type: str = "coherence",
             metadata: Dict[str, Any] = {})
```

Initialize a new schema edge.

**Parameters:**
- `source_id`: ID of the source node
- `target_id`: ID of the target node
- `weight`: Strength of the relationship (default: 0.0)
- `edge_type`: Type of relationship (default: "coherence")
- `metadata`: Additional metadata

```python
def to_dict(self) -> Dict[str, Any]
```

Convert the edge to a dictionary for serialization.

**Returns:**
- Dictionary representation of the edge

### `SchemaGraph`

A graph-based representation of memory and knowledge, capturing relationships between pieces of information.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | networkx.Graph | Underlying NetworkX graph |
| `memory_store` | MemoryStore | Reference to the memory store |
| `coherence_scorer` | CoherenceScorer | Reference to the coherence scorer |
| `min_edge_weight` | float | Minimum weight to create an edge |
| `tracked_memory_ids` | Set[str] | Set of memory IDs in the graph |

#### Methods

```python
def __init__(self, memory_store: MemoryStore, 
             coherence_scorer: CoherenceScorer,
             min_edge_weight: float = 0.3,
             auto_prune: bool = True,
             max_nodes: int = 500)
```

Initialize a schema graph.

**Parameters:**
- `memory_store`: Memory store to track
- `coherence_scorer`: Coherence scorer for calculating relationships
- `min_edge_weight`: Minimum coherence to create an edge (default: 0.3)
- `auto_prune`: Whether to automatically prune low-importance nodes (default: True)
- `max_nodes`: Maximum number of nodes to maintain (default: 500)

```python
def add_memory_node(self, memory: Memory) -> str
```

Add a memory to the schema graph.

**Parameters:**
- `memory`: Memory object to add

**Returns:**
- ID of the created node

```python
def update_memory_node(self, memory: Memory) -> str
```

Update an existing memory node in the schema.

**Parameters:**
- `memory`: Updated memory object

**Returns:**
- ID of the updated node

```python
def add_concept_node(self, label: str,
                     embedding: Optional[List[float]] = None,
                     importance: float = 0.5,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str
```

Add a concept node to the schema (not directly tied to a memory).

**Parameters:**
- `label`: Name or description of the concept
- `embedding`: Optional vector embedding
- `importance`: Importance score (default: 0.5)
- `tags`: Optional list of tags
- `metadata`: Optional additional metadata

**Returns:**
- ID of the created node

```python
def add_edge(self, source_id: str, target_id: str,
             weight: float, edge_type: str = "coherence",
             metadata: Optional[Dict[str, Any]] = None) -> bool
```

Add an edge between two nodes in the schema.

**Parameters:**
- `source_id`: ID of the source node
- `target_id`: ID of the target node
- `weight`: Strength of the relationship
- `edge_type`: Type of relationship (default: "coherence")
- `metadata`: Optional additional metadata

**Returns:**
- True if successful, False otherwise

```python
def remove_node(self, node_id: str) -> bool
```

Remove a node from the schema graph.

**Parameters:**
- `node_id`: ID of the node to remove

**Returns:**
- True if successful, False otherwise

```python
def calculate_memory_relationships(self, memory: Memory) -> List[Tuple[str, float]]
```

Calculate how a memory relates to existing nodes in the schema.

**Parameters:**
- `memory`: Memory to analyze

**Returns:**
- List of tuples (node_id, relationship_strength)

```python
def update_schema(self, max_memories: int = 100) -> Dict[str, int]
```

Update the schema graph by incorporating new memories and calculating relationships.

**Parameters:**
- `max_memories`: Maximum number of memories to process in this update

**Returns:**
- Dictionary with update statistics (nodes_added, edges_added, etc.)

```python
def get_subgraph(self, center_id: str, max_distance: int = 2) -> nx.Graph
```

Extract a subgraph centered around a specific node.

**Parameters:**
- `center_id`: ID of the central node
- `max_distance`: Maximum distance from center to include

**Returns:**
- NetworkX graph representing the subgraph

```python
def get_similar_nodes(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]
```

Find nodes most similar to a query embedding.

**Parameters:**
- `query_embedding`: Embedding vector to compare against
- `top_k`: Maximum number of results to return (default: 10)

**Returns:**
- List of tuples (node_id, similarity_score)

```python
def visualize(self, filename: Optional[str] = None,
              highlight_nodes: Optional[List[str]] = None,
              max_nodes: int = 100) -> None
```

Generate a visualization of the schema graph.

**Parameters:**
- `filename`: Optional path to save the visualization
- `highlight_nodes`: Optional list of node IDs to highlight
- `max_nodes`: Maximum number of nodes to include in visualization

```python
def to_dict(self) -> Dict[str, Any]
```

Convert the schema graph to a dictionary for serialization.

**Returns:**
- Dictionary representation of the schema graph

```python
def get_stats(self) -> Dict[str, Any]
```

Get statistics about the schema graph.

**Returns:**
- Dictionary with statistics (node_count, edge_count, etc.)

## Usage Examples

### Creating a Basic Schema Graph

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import SchemaGraph

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
schema_graph = SchemaGraph(memory_store, coherence_scorer)

# Add memories to the store
memory_id1 = memory_store.add("Earth is the third planet from the Sun.")
memory_id2 = memory_store.add("Mars is the fourth planet from the Sun.")
memory_id3 = memory_store.add("Venus is the second planet from the Sun.")

# Update the schema to establish relationships
update_stats = schema_graph.update_schema()
print(f"Added {update_stats['nodes_added']} nodes and {update_stats['edges_added']} edges")

# Get statistics about the schema
stats = schema_graph.get_stats()
print(f"Schema has {stats['node_count']} nodes and {stats['edge_count']} edges")
```

### Adding Concept Nodes

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import SchemaGraph

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
schema_graph = SchemaGraph(memory_store, coherence_scorer)

# Add concept nodes (not tied to specific memories)
solar_system_id = schema_graph.add_concept_node(
    label="Solar System",
    importance=0.9,
    tags=["astronomy", "space"],
    metadata={"description": "The planetary system that includes Earth"}
)

planet_id = schema_graph.add_concept_node(
    label="Planet",
    importance=0.8,
    tags=["astronomy"],
    metadata={"description": "A celestial body that orbits a star"}
)

# Add memories
earth_id = memory_store.add("Earth is the third planet from the Sun.")
mars_id = memory_store.add("Mars is the fourth planet from the Sun.")

# Update schema to incorporate memories
schema_graph.update_schema()

# Manually add relationships between concepts and memories
earth_node_id = f"memory_{earth_id}"
mars_node_id = f"memory_{mars_id}"

schema_graph.add_edge(planet_id, earth_node_id, weight=1.0, edge_type="is_a")
schema_graph.add_edge(planet_id, mars_node_id, weight=1.0, edge_type="is_a")
schema_graph.add_edge(solar_system_id, planet_id, weight=1.0, edge_type="contains")

# Visualize the schema
schema_graph.visualize("solar_system_schema.png", highlight_nodes=[solar_system_id])
```

### Finding Similar Nodes

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.embedding import EmbeddingEngine

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
embedding_engine = EmbeddingEngine()
schema_graph = SchemaGraph(memory_store, coherence_scorer)

# Add memories with related topics
memory_store.add("Python is a popular programming language.")
memory_store.add("JavaScript is used for web development.")
memory_store.add("Java is an object-oriented language.")
memory_store.add("Python has simple, easy-to-learn syntax.")
memory_store.add("Machine learning is often implemented in Python.")

# Update schema
schema_graph.update_schema()

# Create a query embedding
query = "What programming languages are good for beginners?"
query_embedding = embedding_engine.get_embedding(query)

# Find similar nodes in the schema
similar_nodes = schema_graph.get_similar_nodes(query_embedding, top_k=3)

print("Most relevant information:")
for node_id, score in similar_nodes:
    # If it's a memory node, get the original memory
    if node_id.startswith("memory_"):
        memory_id = node_id[7:]  # Remove "memory_" prefix
        memory = memory_store.get_memory(memory_id)
        print(f"- {memory.content} (Relevance: {score:.2f})")
    else:
        # It's a concept node
        node_data = schema_graph.graph.nodes[node_id]
        print(f"- {node_data['label']} (Relevance: {score:.2f})")
```

### Extracting a Subgraph

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import SchemaGraph
import networkx as nx

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
schema_graph = SchemaGraph(memory_store, coherence_scorer)

# Add memories about a specific topic
center_id = memory_store.add("Albert Einstein developed the theory of relativity.")
memory_store.add("Einstein was born in Germany in 1879.")
memory_store.add("The theory of relativity revolutionized physics.")
memory_store.add("Einstein was awarded the Nobel Prize in Physics in 1921.")
memory_store.add("The equation E=mc² is part of Einstein's theory.")
memory_store.add("Physics studies the fundamental nature of the universe.")

# Also add some unrelated memories
memory_store.add("The Eiffel Tower is in Paris, France.")
memory_store.add("Coffee contains caffeine.")
memory_store.add("The Pacific Ocean is the largest ocean on Earth.")

# Update the schema
schema_graph.update_schema()

# Extract a subgraph centered around Einstein
center_node_id = f"memory_{center_id}"
einstein_subgraph = schema_graph.get_subgraph(center_node_id, max_distance=2)

# Print information about the subgraph
print(f"Full schema has {schema_graph.graph.number_of_nodes()} nodes")
print(f"Einstein subgraph has {einstein_subgraph.number_of_nodes()} nodes")

# Print the nodes in the subgraph
print("\nNodes in Einstein subgraph:")
for node_id in einstein_subgraph.nodes:
    if node_id.startswith("memory_"):
        memory_id = node_id[7:]  # Remove "memory_" prefix
        memory = memory_store.get_memory(memory_id)
        print(f"- {memory.content}")
    else:
        node_data = schema_graph.graph.nodes[node_id]
        print(f"- Concept: {node_data['label']}")
``` 