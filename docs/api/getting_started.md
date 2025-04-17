# Getting Started with ΨC-AI SDK

This guide will help you quickly set up and start using the ΨC-AI SDK for developing cognitive AI systems with coherent belief networks.

## Installation

### Prerequisites

Before installing the ΨC-AI SDK, ensure you have:

- Python 3.8 or higher
- pip package manager

### Standard Installation

Install the ΨC-AI SDK using pip:

```bash
pip install psi-c-ai-sdk
```

### Development Installation

For development purposes, you can install the SDK directly from the source code:

```bash
git clone https://github.com/yourusername/psi_c_ai_sdk.git
cd psi_c_ai_sdk
pip install -e .
```

## Basic Setup

Here's a minimal example to set up a basic ΨC-AI system:

```python
import psi_c_ai_sdk as psi_c
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.embedding import EmbeddingEngine
from psi_c_ai_sdk.coherence import CoherenceScorer
from psi_c_ai_sdk.schema import SchemaGraph

# Initialize core components
memory_store = MemoryStore()
embedding_engine = EmbeddingEngine()
coherence_scorer = CoherenceScorer()

# Create a schema graph to represent knowledge
schema_graph = SchemaGraph(memory_store, coherence_scorer)

# Add a memory to the system
memory_id = memory_store.add(
    content="The sky is blue because of Rayleigh scattering of sunlight.",
    importance=0.8,
    tags=["science", "physics", "sky"],
    metadata={"source": "textbook", "confidence": 0.9}
)

# Retrieve the memory
memory = memory_store.get_memory(memory_id)
print(f"Added memory: {memory.content}")
```

## Core Components Overview

The ΨC-AI SDK is built around several key components:

1. **Memory System**: Stores and manages individual pieces of information
2. **Embedding Engine**: Converts textual information into numerical vectors
3. **Coherence Scorer**: Measures how well memories fit together
4. **Schema Graph**: Represents knowledge as a connected graph
5. **Reflection Engine**: Processes and refines existing knowledge

## Adding Memories

The foundation of any ΨC-AI system is the memory store. Here's how to add and manage memories:

```python
# Create a memory store
memory_store = MemoryStore()

# Add simple memories
memory_store.add("My name is Alice.")
memory_store.add("I live in New York City.")

# Add a memory with metadata and tags
memory_store.add(
    content="I enjoy hiking in the mountains.",
    importance=0.7,
    tags=["hobby", "outdoor"],
    metadata={"confidence": 0.9, "source": "personal"}
)

# Get all memories
all_memories = memory_store.get_all_memories()
for memory in all_memories:
    print(f"Memory: {memory.content} (Importance: {memory.importance})")
```

## Building a Schema

The schema graph represents relationships between memories:

```python
# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()
schema_graph = SchemaGraph(memory_store, coherence_scorer)

# Add memories
memory_store.add("Dogs are mammals.")
memory_store.add("Mammals are animals.")
memory_store.add("My dog Rex is a golden retriever.")

# Update the schema to establish relationships
update_stats = schema_graph.update_schema()
print(f"Added {update_stats['edges_added']} relationships between memories")

# Visualize the schema
schema_graph.visualize("my_schema.png")
```

## Measuring Coherence

Coherence is a fundamental aspect of the ΨC-AI system:

```python
# Initialize components
memory_store = MemoryStore()
coherence_scorer = CoherenceScorer()

# Add coherent memories
memory_store.add("Water boils at 100 degrees Celsius at sea level.")
memory_store.add("Water freezes at 0 degrees Celsius.")
memory_store.add("Water is a compound of hydrogen and oxygen.")

# Add potentially contradictory memory
memory_store.add("Water boils at 90 degrees Celsius at sea level.")

# Measure global coherence
global_coherence = coherence_scorer.calculate_global_coherence(memory_store)
print(f"Global coherence: {global_coherence:.2f}")

# Detect contradictions
contradictions = coherence_scorer.detect_contradictions(memory_store)
for mem1, mem2, score in contradictions:
    print(f"Contradiction found: '{mem1.content}' vs '{mem2.content}' (score: {score:.2f})")
```

## Next Steps

Now that you understand the basics of the ΨC-AI SDK, you can explore more advanced topics:

- Learn about the [Reflection Engine](reflection.md)
- Understand [Schema Versioning](schema_versioning.md) 
- Explore [Multi-Agent Communication](multi_agent.md)
- Implement [Coherence Optimization](coherence_optimization.md)

For more detailed examples, visit the [Examples](examples.md) section. 