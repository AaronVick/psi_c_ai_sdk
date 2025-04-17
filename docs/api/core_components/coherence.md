# Coherence API

The Coherence module is a fundamental component of the ΨC-AI SDK that measures how well different pieces of information relate and form a unified model. It provides tools for calculating semantic similarity between memories, measuring global coherence of the memory system, and detecting contradictions.

## Key Classes

### `CoherenceScorer`

Calculates coherence scores between memories and identifies contradictions.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `similarity_threshold` | float | Minimum similarity for memories to be considered related |
| `contradiction_threshold` | float | Similarity below this value indicates a contradiction |
| `min_context_size` | int | Minimum number of memories to consider for context |
| `global_weighting` | float | Weight of global coherence vs. local coherence |
| `importance_factor` | float | How much memory importance affects coherence calculations |
| `recent_coherence_scores` | List[float] | History of recent global coherence measurements |

#### Methods

```python
def __init__(self, similarity_threshold: float = 0.7,
             contradiction_threshold: float = -0.6,
             min_context_size: int = 5,
             global_weighting: float = 0.3,
             importance_factor: float = 0.5)
```

Initialize the coherence scorer.

**Parameters:**
- `similarity_threshold`: Minimum similarity for memories to be considered related (default: 0.7)
- `contradiction_threshold`: Similarity below this value indicates a contradiction (default: -0.6)
- `min_context_size`: Minimum number of memories to consider for context (default: 5)
- `global_weighting`: Weight of global coherence vs. local coherence (default: 0.3)
- `importance_factor`: How much memory importance affects coherence calculations (default: 0.5)

```python
def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float
```

Calculate cosine similarity between two embeddings.

**Parameters:**
- `embedding1`: First embedding vector
- `embedding2`: Second embedding vector

**Returns:**
- Cosine similarity score (-1 to 1)

```python
def calculate_memory_coherence(self, target_memory: Memory, context_memories: List[Memory]) -> float
```

Calculate how coherent a memory is with a set of context memories.

**Parameters:**
- `target_memory`: The memory to evaluate
- `context_memories`: List of memories to compare against

**Returns:**
- Coherence score from 0 to 1

```python
def calculate_global_coherence(self, memory_store: MemoryStore, 
                               recent_only: bool = False,
                               sample_size: int = 100) -> float
```

Calculate the overall coherence of the memory system.

**Parameters:**
- `memory_store`: The memory store to analyze
- `recent_only`: If True, only consider recent memories (default: False)
- `sample_size`: Maximum number of memories to sample for calculation (default: 100)

**Returns:**
- Global coherence score from 0 to 1

```python
def detect_contradictions(self, memory_store: MemoryStore,
                          max_contradictions: int = 10) -> List[Tuple[Memory, Memory, float]]
```

Find contradictory memories in the store.

**Parameters:**
- `memory_store`: The memory store to analyze
- `max_contradictions`: Maximum number of contradictions to return (default: 10)

**Returns:**
- List of tuples (memory1, memory2, contradiction_score)

```python
def calculate_entropy(self, memory_store: MemoryStore) -> float
```

Calculate the entropy (disorder) of the memory system.

**Parameters:**
- `memory_store`: The memory store to analyze

**Returns:**
- Entropy score (higher means more disorder/less coherence)

```python
def get_most_related_memories(self, target_memory: Memory, 
                              memory_store: MemoryStore,
                              min_similarity: float = 0.0,  
                              max_memories: int = 10) -> List[Tuple[Memory, float]]
```

Find memories most related to a target memory.

**Parameters:**
- `target_memory`: The memory to compare against
- `memory_store`: The memory store to search
- `min_similarity`: Minimum similarity threshold (default: 0.0)
- `max_memories`: Maximum number of memories to return (default: 10)

**Returns:**
- List of tuples (memory, similarity_score)

```python
def calculate_coherence_change(self, before: float, after: float) -> Dict[str, Any]
```

Calculate the change in coherence between two measurements.

**Parameters:**
- `before`: Coherence score before a change
- `after`: Coherence score after a change

**Returns:**
- Dictionary with change metrics (delta, percent_change, etc.)

## Usage Examples

### Basic Coherence Measurement

```python
from psi_c_ai_sdk.memory import MemoryStore, Memory
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.embedding import EmbeddingEngine

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
embedding_engine = EmbeddingEngine()

# Add some related memories
memory_store.add("Photosynthesis is how plants convert sunlight to energy.")
memory_store.add("Plants use chlorophyll to capture sunlight.")
memory_store.add("Plants release oxygen as a byproduct of photosynthesis.")
memory_store.add("Cellular respiration is the opposite of photosynthesis.")

# Add an unrelated memory
memory_store.add("The Great Wall of China is visible from space.")

# Generate embeddings (normally this would be handled by a higher-level component)
for memory in memory_store.get_all_memories():
    if not memory.embedding:
        memory.embedding = embedding_engine.get_embedding(memory.content)

# Calculate global coherence
coherence = coherence_scorer.calculate_global_coherence(memory_store)
print(f"Global coherence: {coherence:.2f}")
```

### Detecting Contradictions

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.embedding import EmbeddingEngine

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer(contradiction_threshold=-0.3)  # More sensitive
embedding_engine = EmbeddingEngine()

# Add memories with contradictions
memory_store.add("The Earth is approximately 4.5 billion years old.")
memory_store.add("The Earth is flat.")
memory_store.add("The Earth is roughly spherical in shape.")
memory_store.add("The Earth is the third planet from the Sun.")
memory_store.add("The Earth is the closest planet to the Sun.")
memory_store.add("Mercury is the closest planet to the Sun.")

# Generate embeddings (normally this would be handled by a higher-level component)
for memory in memory_store.get_all_memories():
    if not memory.embedding:
        memory.embedding = embedding_engine.get_embedding(memory.content)

# Detect contradictions
contradictions = coherence_scorer.detect_contradictions(memory_store)

print("Detected contradictions:")
for memory1, memory2, score in contradictions:
    print(f"- Contradiction (score: {score:.2f}):")
    print(f"  1. \"{memory1.content}\"")
    print(f"  2. \"{memory2.content}\"")
```

### Finding Related Memories

```python
from psi_c_ai_sdk.memory import MemoryStore, Memory
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.embedding import EmbeddingEngine

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
embedding_engine = EmbeddingEngine()

# Add a set of memories about various topics
memory_store.add("Paris is the capital of France.")
memory_store.add("The Eiffel Tower is located in Paris.")
memory_store.add("France is a country in Western Europe.")
memory_store.add("The Seine river runs through Paris.")
memory_store.add("The Louvre Museum is home to the Mona Lisa.")
memory_store.add("Coffee is a popular beverage worldwide.")
memory_store.add("The Great Barrier Reef is in Australia.")
memory_store.add("Leonardo da Vinci painted the Mona Lisa.")

# Generate embeddings (normally this would be handled by a higher-level component)
for memory in memory_store.get_all_memories():
    if not memory.embedding:
        memory.embedding = embedding_engine.get_embedding(memory.content)

# Create a query memory
query = Memory("I'm planning a trip to Paris and want to see famous landmarks.")
query.embedding = embedding_engine.get_embedding(query.content)

# Find related memories
related_memories = coherence_scorer.get_most_related_memories(
    query, memory_store, min_similarity=0.3
)

print(f"Query: {query.content}")
print("Related memories:")
for memory, similarity in related_memories:
    print(f"- {memory.content} (Similarity: {similarity:.2f})")
```

### Tracking Coherence Changes

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.embedding import EmbeddingEngine

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
embedding_engine = EmbeddingEngine()

# Add some initial memories
memory_store.add("Mammals are warm-blooded animals.")
memory_store.add("Whales are mammals that live in the ocean.")
memory_store.add("Mammals give birth to live young.")

# Generate embeddings
for memory in memory_store.get_all_memories():
    if not memory.embedding:
        memory.embedding = embedding_engine.get_embedding(memory.content)

# Measure initial coherence
initial_coherence = coherence_scorer.calculate_global_coherence(memory_store)
print(f"Initial coherence: {initial_coherence:.2f}")

# Add more coherent memories
memory_store.add("Bats are flying mammals.")
memory_store.add("Dolphins are intelligent marine mammals.")
memory_store.add("All mammals have fur or hair at some point in their lives.")

# Generate embeddings for new memories
for memory in memory_store.get_all_memories():
    if not memory.embedding:
        memory.embedding = embedding_engine.get_embedding(memory.content)

# Measure coherence after adding coherent memories
coherent_add_coherence = coherence_scorer.calculate_global_coherence(memory_store)
print(f"Coherence after adding coherent memories: {coherent_add_coherence:.2f}")

# Calculate change
coherent_change = coherence_scorer.calculate_coherence_change(
    initial_coherence, coherent_add_coherence
)
print(f"Change after coherent additions: {coherent_change['delta']:.2f} " +
      f"({coherent_change['percent_change']:.1f}%)")

# Add an incoherent memory
memory_store.add("Python is a programming language.")
memory = memory_store.get_all_memories()[-1]  # Get the last added memory
memory.embedding = embedding_engine.get_embedding(memory.content)

# Measure coherence after adding incoherent memory
incoherent_add_coherence = coherence_scorer.calculate_global_coherence(memory_store)
print(f"Coherence after adding incoherent memory: {incoherent_add_coherence:.2f}")

# Calculate change
incoherent_change = coherence_scorer.calculate_coherence_change(
    coherent_add_coherence, incoherent_add_coherence
)
print(f"Change after incoherent addition: {incoherent_change['delta']:.2f} " +
      f"({incoherent_change['percent_change']:.1f}%)")
```

### Calculating Entropy

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.embedding import EmbeddingEngine

# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
embedding_engine = EmbeddingEngine()

# Add some focused memories on a single topic
memory_store.add("Chess is a strategic board game.")
memory_store.add("In chess, the king can move one square in any direction.")
memory_store.add("The queen is the most powerful piece in chess.")
memory_store.add("A checkmate in chess occurs when the king is in check and cannot escape.")

# Generate embeddings
for memory in memory_store.get_all_memories():
    if not memory.embedding:
        memory.embedding = embedding_engine.get_embedding(memory.content)

# Calculate entropy for focused set
focused_entropy = coherence_scorer.calculate_entropy(memory_store)
print(f"Entropy for focused memories: {focused_entropy:.2f}")

# Clear and add diverse memories
memory_store = MemoryStore()  # Reset the store
memory_store.add("Chess is a strategic board game.")
memory_store.add("Mount Everest is the tallest mountain on Earth.")
memory_store.add("Python was created by Guido van Rossum.")
memory_store.add("The Mona Lisa was painted by Leonardo da Vinci.")
memory_store.add("Photosynthesis is the process used by plants to convert light into energy.")

# Generate embeddings
for memory in memory_store.get_all_memories():
    if not memory.embedding:
        memory.embedding = embedding_engine.get_embedding(memory.content)

# Calculate entropy for diverse set
diverse_entropy = coherence_scorer.calculate_entropy(memory_store)
print(f"Entropy for diverse memories: {diverse_entropy:.2f}")
print(f"Difference: {diverse_entropy - focused_entropy:.2f}")
``` 