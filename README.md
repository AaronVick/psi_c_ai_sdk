# ΨC-AI SDK

A comprehensive toolkit for implementing consciousness-informed AI systems with enhanced safety, reflection, and self-modeling capabilities.

## Overview

The ΨC-AI SDK provides tools and frameworks for developing AI systems with:

- **Self-modeling capabilities**: Schema graphs, belief systems, and identity modeling
- **Advanced reflection**: Metacognitive processing, contradiction detection, and memory coherence
- **Safety mechanisms**: Recursive stability monitoring, alignment boundaries, and behavior profiling
- **Cognitive enhancement**: Personal epistemic horizon, trust throttling, and ontology comparison

## Installation

### From PyPI (when available)
```bash
pip install psi-c-ai-sdk
```

### From Source
```bash
git clone https://github.com/psi-c-ai/psi-c-ai-sdk.git
cd psi-c-ai-sdk
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from psi_c_ai_sdk import MemoryStore, EmbeddingEngine, CoherenceScorer, ContradictionDetector

# Initialize the memory system
memory_store = MemoryStore()

# Add memories
memory_store.add("The sky is blue")
memory_store.add("Water is composed of hydrogen and oxygen")

# Initialize embedding engine and coherence scorer
embedding_engine = EmbeddingEngine()
coherence_scorer = CoherenceScorer(embedding_engine)

# Calculate coherence between memories
memories = memory_store.get_all_memories()
coherence_matrix = coherence_scorer.calculate_coherence_matrix(memories)
global_coherence = coherence_scorer.calculate_global_coherence(memories)
print(f"Global coherence: {global_coherence:.4f}")

# Find contradictions
contradiction_detector = ContradictionDetector(embedding_engine)
contradictions = contradiction_detector.find_contradictions(memories)
if contradictions:
    print(f"Found {len(contradictions)} contradictions")
else:
    print("No contradictions found")

# Export memories
memory_store.export("memories.json")
```

## Core Components

### Memory System
The memory system handles storing and retrieving memories, tracking their importance, and managing their lifecycle:

```python
from psi_c_ai_sdk import MemoryStore

# Create a memory store
memory_store = MemoryStore(decay_constant=0.01)

# Add memories with tags and importance
memory_id = memory_store.add(
    content="This is an important memory", 
    importance=2.0,
    tags=["important", "example"]
)

# Retrieve memories
memory = memory_store.get_memory(memory_id)
print(memory.content)

# Get memories by importance threshold
important_memories = memory_store.get_memories_by_importance(threshold=1.5)
```

### Embedding Engine
The embedding engine generates semantic vector representations of memories:

```python
from psi_c_ai_sdk import EmbeddingEngine

# Create an embedding engine with caching
embedding_engine = EmbeddingEngine(
    model_name="all-MiniLM-L6-v2",
    use_cache=True
)

# Generate embeddings
text = "This is a sample text"
embedding = embedding_engine.get_embedding(text)

# Calculate similarity between texts
similarity = embedding_engine.cosine_similarity(
    embedding_engine.get_embedding("I love cats"),
    embedding_engine.get_embedding("I like cats")
)
print(f"Similarity: {similarity:.4f}")
```

### Coherence Scoring
The coherence system measures the semantic coherence between memories:

```python
from psi_c_ai_sdk import CoherenceScorer

# Create a coherence scorer
coherence_scorer = CoherenceScorer(
    embedding_engine,
    tag_weight=0.2
)

# Calculate coherence
coherence = coherence_scorer.calculate_coherence(memory1, memory2)
print(f"Coherence: {coherence:.4f}")

# Find the highest coherence pair
highest_pair = coherence_scorer.get_highest_coherence_pair(memories)
memory_a, memory_b, score = highest_pair
```

### Contradiction Detection
The contradiction detector identifies semantic contradictions between memories:

```python
from psi_c_ai_sdk import ContradictionDetector

# Create a contradiction detector
contradiction_detector = ContradictionDetector(
    embedding_engine,
    similarity_threshold=0.7
)

# Check for contradictions
is_contradiction, confidence, explanation = contradiction_detector.detect_contradiction(memory1, memory2)
if is_contradiction:
    print(f"Contradiction found: {explanation}")
    
# Find all contradictions in a set of memories
contradictions = contradiction_detector.find_contradictions(memories)
```

### Reflection Engine
The reflection engine provides introspection capabilities for memory reorganization and processing:

```python
from psi_c_ai_sdk import ReflectionEngine, ReflectionScheduler

# Create a reflection engine
reflection_engine = ReflectionEngine(
    memory_store=memory_store,
    coherence_scorer=coherence_scorer,
    contradiction_detector=contradiction_detector
)

# Set up a reflection scheduler
scheduler = ReflectionScheduler(
    reflection_engine,
    coherence_threshold=0.7,
    memory_threshold=100,
    time_interval=3600  # seconds
)

# Perform a reflection cycle
outcome = reflection_engine.reflect()
print(f"Reflection outcome: {outcome.status}")
print(f"Resolved contradictions: {outcome.contradictions_resolved}")
print(f"Memories consolidated: {outcome.memories_consolidated}")
```

### Schema Graph Builder
The schema system builds and manages knowledge graphs:

```python
from psi_c_ai_sdk import SchemaGraph

# Create a schema graph
schema = SchemaGraph(
    memory_store=memory_store,
    embedding_engine=embedding_engine
)

# Add nodes and build connections
schema.add_memory_nodes(memory_store.get_all_memories())
schema.connect_nodes_by_coherence(threshold=0.7)

# Find related nodes
related_nodes = schema.find_similar_nodes("example concept")

# Visualize the schema
schema.visualize(filename="schema_graph.png", highlight_nodes=related_nodes)
```

### Schema Mutation System
The mutation system evolves schema graphs to improve knowledge organization:

```python
from psi_c_ai_sdk import SchemaMutationSystem

# Create a mutation system
mutation_system = SchemaMutationSystem(schema_graph=schema)

# Perform mutations
mutation_system.add_concept_nodes()
mutation_system.merge_similar_nodes(similarity_threshold=0.8)
mutation_system.prune_weak_connections(weight_threshold=0.3)

# Get mutation statistics
stats = mutation_system.get_stats()
print(f"Nodes merged: {stats['nodes_merged']}")
print(f"Connections pruned: {stats['connections_pruned']}")
```

### Bounded Cognitive Runtime
The complexity controller manages system resources and complexity:

```python
from psi_c_ai_sdk import ComplexityController

# Create a complexity controller
complexity_controller = ComplexityController(
    memory_store=memory_store,
    schema_graph=schema
)

# Check if a reflection is allowed
if complexity_controller.can_perform_reflection():
    reflection_engine.reflect()

# Get current complexity metrics
metrics = complexity_controller.get_complexity_metrics()
print(f"Memory complexity: {metrics.memory_complexity}")
print(f"Schema complexity: {metrics.schema_complexity}")
print(f"Current energy budget: {complexity_controller.energy_budget}")
```

### Reflection Credit System
The credit system manages cognitive resources during reflection:

```python
from psi_c_ai_sdk import ReflectionCreditSystem

# Create a credit system
credit_system = ReflectionCreditSystem(
    initial_credits=100,
    regeneration_rate=5,
    cooldown_period=3600  # seconds
)

# Track reflection effectiveness and manage credits
outcome = reflection_engine.reflect()
credit_adjustment = credit_system.process_reflection_outcome(outcome)
print(f"Credit adjustment: {credit_adjustment}")
print(f"Remaining credits: {credit_system.available_credits}")
print(f"Cognitive debt: {credit_system.calculate_cognitive_debt()}")
```

### Memory Legacy System
The legacy system enables agents to preserve and transfer their most valuable memories when reaching entropy thresholds:

```python
from psi_c_ai_sdk import LegacyManager, LegacyImporter

# Create a legacy manager
legacy_manager = LegacyManager(storage_path="legacy/")

# When agent reaches entropy threshold
if entropy >= 0.8 or coherence <= 0.3:
    # Create a legacy block with most valuable memories
    legacy_block = legacy_manager.create_legacy_block(
        agent_id="agent_123",
        agent_name="ΨC-Alpha",
        memories=memory_store.get_all_memories(),
        selector_type="emergent_value"
    )
    
    # Save the legacy block
    legacy_file = legacy_manager.save_legacy_block(legacy_block)
    print(f"Legacy created with {len(legacy_block.core_memories)} memories")
    
# New agent can import the legacy
successor_memory_store = MemoryStore()
count, imported_ids = LegacyImporter.import_legacy(
    memory_store=successor_memory_store,
    legacy_block=legacy_file,
    tag_as_inherited=True,
    preserve_lineage=True
)
print(f"Imported {count} legacy memories from predecessor")
```

### Safety Module

The Safety Module provides comprehensive safety monitoring and protection through:

- **Reflection Guard**: Prevents paradoxical reasoning loops and detects contradictions
- **Profile Analyzer**: Monitors AI behavior patterns against safety boundaries
- **Integration Manager**: Coordinates safety components and enforces protection policies
- **Epistemic Trust Throttler**: Dynamically adjusts trust in sources with high persuasion entropy

### Cognition Module

The Cognition Module enhances reasoning and belief management:

- **Epistemic Horizon**: Tracks confidence in beliefs and manages epistemic boundaries
- **Trust Throttler**: Protects against manipulation by monitoring source persuasion patterns
- **Coherence-Weighted Dilemma Engine**: Evaluates choices based on internal coherence

### Agent Module

The Agent Module provides identity and narrative tools:

- **Narrative Signature Generator**: Compresses an agent's journey into a human-readable signature
- **Schema Mutation Manager**: Tracks and controls ontological changes over time

## Examples
- [Simple Test](examples/simple_test.py): Basic verification of the memory system
- [Basic Example](examples/basic_example.py): Demonstrates core functionality including coherence and contradiction detection
- [Schema Mutation Demo](examples/schema_mutation_demo.py): Shows how to use the Schema Mutation System
- [Complexity Controller Demo](examples/complexity_controller_demo.py): Demonstrates the Bounded Cognitive Runtime
- [Reflection Credit Demo](examples/reflection_credit_demo.py): Examples of the Reflection Credit System in action
- [Memory Legacy Demo](examples/memory_legacy_demo.py): Demonstrates memory inheritance across agent generations
- [Safety Integration Example](examples/safety_integration_example.py): Demonstrates safety monitoring integration
- [Trust Throttle Example](examples/trust_throttle_example.py): Shows epistemic trust throttling in action
- [Recursive Safety Demo](examples/recursive_safety_demo.py): Illustrates recursive stability monitoring

## Documentation

### API Documentation
Comprehensive API documentation is available in the [docs/api/](docs/api/README.md) directory, including:

- **API Reference**: Detailed reference for all components, classes, and methods
- **Getting Started**: Basic installation and setup instructions in [docs/api/getting_started.md](docs/api/getting_started.md)
- **Core Components**: Documentation for fundamental components in the [docs/api/core_components/](docs/api/core_components/) directory
- **API Index**: A searchable index of all API documentation in [docs/api/index.md](docs/api/index.md)

### Guides and Technical Documentation
- [Schema Usage Examples](docs/Schema_Usage_Examples.md): Practical examples of schema graph usage
- [Coherence Optimization Tutorial](docs/Coherence_Optimization_Tutorial.md): Guide to optimizing coherence in your system
- [Safety Integration Guide](docs/safety_integration_guide.md): Implementing safety mechanisms in your application
- [Plugin Development Guide](docs/plugin_development.md): Creating plugins for the ΨC-AI SDK
- [Operational Understanding](docs/Operational_Understanding.md): Understanding the operational aspects of the SDK

### Conceptual Documentation
- [Understanding the ΨC-AI Framework](docs/Understanding_the_ΨC-AI_Framework.md): Conceptual overview of the framework
- [Foundation](docs/Foundation.md): Core principles and foundation of the ΨC-AI approach
- [Unified Formalism](docs/Unified_Formalism.md): Mathematical formalism behind the system
- [Underlying Math](docs/UnderlyingMath.md): Mathematical concepts used in the SDK

### Educational and Reference Materials
- [AI Governance](docs/AI_Governance.md): Guidelines for responsible AI governance
- [Architecture Models Comparison](docs/architecture/compare_models.md): Comparison of different architectural approaches

### Demo Documentation
- [Memory Sandbox Documentation](tools/dev_environment/README_MEMORY_SANDBOX.md): Guide to using the Memory Sandbox
- [Project Demo Documentation](docs/PROJECT_DEMO.md): Documentation for the ΨC-AI demonstration

For project status and roadmap, see the [PROGRESS.md](PROGRESS.md) file.

## Project Status
The ΨC-AI SDK is under active development. See the [PROGRESS.md](PROGRESS.md) file for current status and roadmap.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License

## Memory Sandbox

The Memory Sandbox provides tools for experimenting with agent memory structures and dynamics.

### Launching the Memory Sandbox

You can launch the Memory Sandbox in two ways:

**Web Interface:**
```bash
python -m tools.dev_environment.launcher --tool memory_sandbox
```

**Directly:**
```bash
python -m tools.dev_environment.memory_sandbox_cli
```

### Key Features

- **Memory Creation**: Create synthetic memories of different types with controlled properties
- **Memory Visualization**: Visualize memory activation patterns and distributions
- **Memory Simulation**: Simulate memory dynamics including decay, reinforcement, and retrieval
- **Memory Analysis**: Calculate and display detailed memory statistics and patterns
- **Schema Integration**: Analyze memory structures and build knowledge graphs
  - Build schema graphs from memories
  - Detect memory clusters and patterns
  - Generate concept suggestions from memory patterns
  - Visualize relationships between memories
  - Export and import schema graphs for persistent knowledge structures

### Schema Integration

To use the Memory Sandbox with Schema Integration:

```bash
python -m tools.dev_environment.demos.schema_integration_demo
```

This will demonstrate the full capabilities of the Schema Integration feature, including building knowledge graphs, detecting clusters, and generating concepts from your agent's memories.

### Demo

Try the Memory Sandbox demo to see the tool in action:

```bash
python -m tools.dev_environment.demos.memory_sandbox_demo
```

For detailed documentation, see [Memory Sandbox Documentation](tools/dev_environment/README_MEMORY_SANDBOX.md).

## Development Tools

The SDK includes several development tools to assist with building and testing ΨC agents:

### Memory Sandbox

The Memory Sandbox allows developers to create, manipulate, and test agent memory systems without integrating with a full agent:

```bash
# Launch the web interface
python -m tools.dev_environment.launcher --tool memory_sandbox

# Or use the CLI for quick operations
python -m tools.memory_sandbox_cli --help
```

Key features:
- Create and manipulate synthetic memory patterns
- Visualize memory activation and retrieval
- Simulate memory dynamics like decay and forgetting
- Compare memory states before and after operations

Try the included demo: `python -m tools.dev_environment.demos.memory_sandbox_demo`

### Schema Graph Integration

The Schema Graph Integration extends the Memory Sandbox with powerful tools for analyzing memory relationships and building knowledge structures:

```bash
# Launch with schema integration enabled
python -m tools.dev_environment.launcher --tool memory_sandbox --enable-schema
```

Key features:
- Build schema graphs from memory clusters
- Detect emergent concepts in memory
- Analyze memory relationships and patterns
- Visualize knowledge structures
- Suggest new conceptual links

Try the included demo: `python -m tools.dev_environment.demos.memory_schema_demo`

### ΨC Schema Integration Demo

This demo showcases the ΨC-AI SDK's cognitive coherence and schema integration capabilities. It provides an interactive interface for exploring how new memories are integrated into a schema graph and how this affects the system's coherence and entropy.

## Features

- Interactive memory input and processing
- Real-time visualization of schema graph evolution
- Coherence and entropy metrics with time-series tracking
- Phase transition detection
- Optional LLM integration for enhanced explanations
- Multiple demo profiles (healthcare, legal, narrative)
- Full persistence of system state between sessions

## Prerequisites

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`):
  - streamlit
  - networkx
  - matplotlib
  - numpy
  - pandas
  - (Optional for LLM integration) openai

## Quick Start

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the demo:
   ```
   streamlit run web_interface_demo.py
   ```

## Using the Demo

1. Select a profile from the sidebar (default, healthcare, legal, narrative)
2. Add a new memory in the text area and click "Process Memory"
3. Watch how the system processes the memory and updates its schema
4. Observe changes in coherence and entropy metrics
5. Explore the schema graph to see how concepts are related
6. If LLM integration is enabled, you can ask questions about the system's current state

## LLM Integration (Optional)

The demo can optionally use OpenAI or other LLM services to enhance explanations. To enable:

1. Add your API key to the environment:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```
2. Enable LLM integration using the toggle in the sidebar

## Profiles

- **Default**: Empty starting state
- **Healthcare**: Medical knowledge and health-related concepts
- **Legal**: Legal principles and jurisprudence
- **Narrative**: Story-based narrative understanding

## System Architecture

The demo consists of:

- `demo_runner.py`: Core integration with the ΨC-AI SDK
- `web_interface_demo.py`: Streamlit-based UI
- `llm_bridge.py`: Optional LLM integration
- `state/`: Persistent storage for system state
- `demo_config/`: Configuration and predefined case files

## Customization

You can customize the demo by:

1. Editing parameters in `demo_config/default_config.json`
2. Creating new profiles with predefined memories
3. Modifying prompt templates for LLM integration

## License

This demo is part of the ΨC-AI SDK and is provided under its license terms. 