# Orchestration API

The Orchestration module is the central control system of the ΨC-AI SDK that coordinates all components into a coherent cycle. It manages the flow of information through the system, including input ingestion, reflection processing, contradiction detection, memory updates, and coherence calculation.

## Key Classes

### `InputProcessor`

Handles the ingestion of new information with rich metadata tagging.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `memory_store` | MemoryStore | The memory store to add inputs to |
| `source_trust_levels` | Dict[str, float] | Dictionary mapping sources to trust levels |
| `preprocessors` | List[Callable] | List of preprocessing functions to apply to inputs |
| `stats` | Dict[str, Any] | Statistics about processed inputs |

#### Methods

```python
def __init__(self, memory_store: MemoryStore,
             source_trust_levels: Optional[Dict[str, float]] = None,
             preprocessors: Optional[List[Callable]] = None)
```

Initialize the input processor.

**Parameters:**
- `memory_store`: The memory store to add inputs to
- `source_trust_levels`: Dictionary mapping sources to trust levels (0.0-1.0)
- `preprocessors`: Optional list of preprocessing functions

```python
def ingest(self, content: str,
           metadata: Optional[Dict[str, Any]] = None,
           source: Optional[str] = None,
           domain: Optional[str] = None,
           importance: Optional[float] = None,
           timestamp: Optional[datetime] = None) -> str
```

Ingest new information into the system.

**Parameters:**
- `content`: The content to ingest
- `metadata`: Optional additional metadata
- `source`: Optional source of the information
- `domain`: Optional domain classification
- `importance`: Optional importance score (0.0-1.0)
- `timestamp`: Optional custom timestamp

**Returns:**
- Memory ID of the ingested content

```python
def get_stats(self) -> Dict[str, Any]
```

Get statistics about processed inputs.

**Returns:**
- Dictionary with input statistics

### `CycleController`

Core orchestration controller for the ΨC system cycle.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `memory_store` | MemoryStore | Memory storage system |
| `schema_graph` | SchemaGraph | Schema graph representation |
| `reflection_engine` | ReflectionEngine | Reflection processor |
| `contradiction_detector` | ContradictionDetector | Contradiction detection system |
| `coherence_calculator` | CoherenceCalculator | Coherence calculation system |
| `input_processor` | InputProcessor | Input processing system |
| `complexity_controller` | ComplexityController | Regulates processing complexity |
| `cycle_frequency` | Optional[float] | Fixed cycle interval in seconds |
| `cycle_count` | int | Number of cycles executed |
| `coherence_history` | List[Tuple[float, float]] | History of coherence measurements |

#### Methods

```python
def __init__(self, memory_store: MemoryStore,
             schema_graph: SchemaGraph,
             reflection_engine: ReflectionEngine,
             contradiction_detector: ContradictionDetector,
             coherence_calculator: CoherenceCalculator,
             input_processor: Optional[InputProcessor] = None,
             complexity_controller: Optional[ComplexityController] = None,
             cycle_frequency: Optional[float] = None,
             log_dir: Optional[str] = None)
```

Initialize the cycle controller.

**Parameters:**
- `memory_store`: Memory storage system
- `schema_graph`: Schema graph representation
- `reflection_engine`: Reflection processor
- `contradiction_detector`: Contradiction detection system
- `coherence_calculator`: Coherence calculation system
- `input_processor`: Optional custom input processor
- `complexity_controller`: Optional complexity controller
- `cycle_frequency`: Optional fixed cycle interval in seconds
- `log_dir`: Directory to store cycle logs

```python
def ingest(self, content: str,
           metadata: Optional[Dict[str, Any]] = None,
           source: Optional[str] = None,
           domain: Optional[str] = None,
           importance: Optional[float] = None) -> str
```

Ingest new information into the system.

**Parameters:**
- `content`: The content to ingest
- `metadata`: Optional additional metadata
- `source`: Optional source of the information
- `domain`: Optional domain classification
- `importance`: Optional importance score (0.0-1.0)

**Returns:**
- Memory ID of the ingested content

```python
def run_cycle(self, memory_id: Optional[str] = None) -> Dict[str, Any]
```

Run a single processing cycle.

**Parameters:**
- `memory_id`: Optional specific memory to focus on

**Returns:**
- Dictionary with cycle results and metrics

```python
def run_continuous(self, max_cycles: Optional[int] = None,
                   duration: Optional[float] = None,
                   coherence_target: Optional[float] = None) -> List[Dict[str, Any]]
```

Run continuous processing cycles.

**Parameters:**
- `max_cycles`: Maximum number of cycles to run
- `duration`: Maximum duration to run in seconds
- `coherence_target`: Stop when coherence reaches this value

**Returns:**
- List of cycle result dictionaries

```python
def reinforce_beliefs(self, reflection_result: Dict[str, Any],
                      contradictions: List[Tuple[str, str, float]]) -> Dict[str, Any]
```

Reinforce or revise beliefs based on reflection and contradictions.

**Parameters:**
- `reflection_result`: Results from the reflection process
- `contradictions`: List of detected contradictions

**Returns:**
- Dictionary with belief reinforcement results

```python
def get_coherence_history(self) -> List[Tuple[float, float]]
```

Get the history of coherence measurements.

**Returns:**
- List of tuples (timestamp, coherence_value)

```python
def get_cycle_stats(self) -> Dict[str, Any]
```

Get statistics about cycle execution.

**Returns:**
- Dictionary with cycle statistics

## Usage Examples

### Basic System Setup and Cycle Execution

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.coherence import CoherenceCalculator
from psi_c_ai_sdk.reflection import ReflectionEngine
from psi_c_ai_sdk.contradiction import ContradictionDetector
from psi_c_ai_sdk.core.orchestration import CycleController, InputProcessor

# Initialize core components
memory_store = MemoryStore()
schema_graph = SchemaGraph(memory_store, coherence_scorer)
coherence_calculator = CoherenceCalculator(memory_store, schema_graph)
reflection_engine = ReflectionEngine(memory_store, schema_graph)
contradiction_detector = ContradictionDetector(memory_store, schema_graph)

# Create input processor with source trust levels
source_trust_levels = {
    "user": 0.9,
    "web": 0.7,
    "inference": 0.6
}
input_processor = InputProcessor(memory_store, source_trust_levels=source_trust_levels)

# Initialize the cycle controller
controller = CycleController(
    memory_store=memory_store,
    schema_graph=schema_graph,
    reflection_engine=reflection_engine,
    contradiction_detector=contradiction_detector,
    coherence_calculator=coherence_calculator,
    input_processor=input_processor,
    log_dir="./cycle_logs"
)

# Add initial memories
controller.ingest("The sky is blue.", source="user")
controller.ingest("Water is composed of hydrogen and oxygen.", source="user")
controller.ingest("The Earth revolves around the Sun.", source="user")

# Run a processing cycle
cycle_result = controller.run_cycle()

# Print results
print(f"Cycle completed in {cycle_result['duration']:.2f} seconds")
print(f"Global coherence: {cycle_result['coherence_after']:.2f}")
print(f"Reflections generated: {len(cycle_result['reflections'])}")
print(f"Contradictions found: {len(cycle_result['contradictions'])}")

# Print reflections
if cycle_result['reflections']:
    print("\nReflections:")
    for ref in cycle_result['reflections']:
        print(f"- {ref['content']}")
```

### Running Continuous Cycles with Monitoring

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.coherence import CoherenceCalculator
from psi_c_ai_sdk.reflection import ReflectionEngine
from psi_c_ai_sdk.contradiction import ContradictionDetector
from psi_c_ai_sdk.core.orchestration import CycleController
import time
import matplotlib.pyplot as plt

# Initialize core components (as in previous example)
# ...

# Initialize the cycle controller
controller = CycleController(
    memory_store=memory_store,
    schema_graph=schema_graph,
    reflection_engine=reflection_engine,
    contradiction_detector=contradiction_detector,
    coherence_calculator=coherence_calculator
)

# Add a series of memories
memories = [
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
    "France is a country in Western Europe.",
    "Paris has a population of over 2 million people.",
    "The Seine river runs through Paris.",
    "The Louvre Museum houses the Mona Lisa painting."
]

for memory in memories:
    controller.ingest(memory)
    
# Run continuous cycles (max 10 cycles or until coherence reaches 0.8)
results = controller.run_continuous(max_cycles=10, coherence_target=0.8)

# Extract coherence values for plotting
timestamps = [r['timestamp'] for r in results]
coherence_values = [r['coherence_after'] for r in results]

# Plot coherence over time
plt.figure(figsize=(10, 6))
plt.plot(range(len(coherence_values)), coherence_values, marker='o')
plt.xlabel('Cycle Number')
plt.ylabel('Global Coherence')
plt.title('Coherence Evolution Over Processing Cycles')
plt.grid(True)
plt.tight_layout()
plt.savefig('coherence_evolution.png')
plt.close()

# Print final stats
print(f"Ran {len(results)} cycles")
print(f"Initial coherence: {results[0]['coherence_before']:.2f}")
print(f"Final coherence: {results[-1]['coherence_after']:.2f}")
print(f"Total reflections generated: {sum(len(r['reflections']) for r in results)}")
print(f"Total contradictions resolved: {sum(len(r['contradictions']) for r in results)}")
```

### Handling Contradictions

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.coherence import CoherenceCalculator
from psi_c_ai_sdk.reflection import ReflectionEngine
from psi_c_ai_sdk.contradiction import ContradictionDetector
from psi_c_ai_sdk.core.orchestration import CycleController

# Initialize core components (as in previous examples)
# ...

# Initialize the cycle controller
controller = CycleController(
    memory_store=memory_store,
    schema_graph=schema_graph,
    reflection_engine=reflection_engine,
    contradiction_detector=contradiction_detector,
    coherence_calculator=coherence_calculator
)

# Add initial consistent memories
controller.ingest("Water freezes at 0 degrees Celsius.")
controller.ingest("Water boils at 100 degrees Celsius at sea level.")
controller.ingest("H2O is the chemical formula for water.")

# Run a cycle to process initial information
controller.run_cycle()

# Get current coherence
initial_stats = controller.get_cycle_stats()
print(f"Initial coherence: {initial_stats['last_coherence']:.2f}")

# Now add contradictory information
controller.ingest("Water freezes at 10 degrees Celsius.", importance=0.7)
controller.ingest("Water boils at 90 degrees Celsius at sea level.", importance=0.7)

# Run another cycle to detect and process contradictions
result = controller.run_cycle()

# Check for contradictions
if result['contradictions']:
    print("\nDetected contradictions:")
    for mem1_id, mem2_id, score in result['contradictions']:
        mem1 = memory_store.get_memory(mem1_id)
        mem2 = memory_store.get_memory(mem2_id)
        print(f"- Between \"{mem1.content}\" and \"{mem2.content}\" (score: {score:.2f})")

# Check coherence after contradiction
print(f"Coherence after contradiction: {result['coherence_after']:.2f}")

# Run multiple cycles to try resolving contradictions
results = controller.run_continuous(max_cycles=5)

# Check final state
final_coherence = results[-1]['coherence_after']
print(f"Final coherence after resolution attempts: {final_coherence:.2f}")

# Check memory importance values to see which memories were prioritized
for memory in memory_store.get_all_memories():
    print(f"Memory: \"{memory.content}\" (Importance: {memory.importance:.2f})")
```

### Input Processing with Metadata

```python
from psi_c_ai_sdk.memory import MemoryStore
from psi_c_ai_sdk.core.orchestration import InputProcessor
from datetime import datetime

# Create a memory store
memory_store = MemoryStore()

# Define source trust levels
source_trust_levels = {
    "user": 0.95,
    "book": 0.85,
    "website": 0.7,
    "social_media": 0.5,
    "inference": 0.6
}

# Create a simple preprocessor function
def clean_text(text):
    """Simple preprocessor to clean input text."""
    return text.strip().replace("  ", " ")

# Initialize input processor
input_processor = InputProcessor(
    memory_store, 
    source_trust_levels=source_trust_levels,
    preprocessors=[clean_text]
)

# Ingest memories from different sources with different metadata
input_processor.ingest(
    content="The Earth is approximately 4.5 billion years old.",
    source="book",
    domain="science",
    importance=0.8,
    metadata={"title": "A Brief History of Earth", "author": "John Smith"}
)

input_processor.ingest(
    content="The Eiffel Tower was completed in 1889.",
    source="website",
    domain="history",
    importance=0.7,
    metadata={"url": "https://example.com/eiffel-tower-facts"}
)

input_processor.ingest(
    content="I saw a beautiful sunset yesterday.",
    source="user",
    domain="personal",
    importance=0.6,
    timestamp=datetime(2023, 5, 15, 18, 30)
)

input_processor.ingest(
    content="Everyone is saying the concert was amazing!",
    source="social_media",
    domain="entertainment",
    importance=0.3
)

# Get statistics about processed inputs
stats = input_processor.get_stats()
print(f"Total inputs processed: {stats['inputs_processed']}")
print(f"Average entropy: {stats['avg_entropy']:.2f}")
print("\nInputs by source:")
for source, count in stats['by_source'].items():
    print(f"- {source}: {count}")

print("\nInputs by domain:")
for domain, count in stats['by_domain'].items():
    print(f"- {domain}: {count}")

# Print memories with their trust levels
print("\nMemories with trust levels:")
for memory in memory_store.get_all_memories():
    source = memory.metadata.get("source", "unknown")
    trust_level = memory.metadata.get("trust_level", 0.0)
    print(f"- \"{memory.content}\" (Source: {source}, Trust: {trust_level:.2f})")
``` 