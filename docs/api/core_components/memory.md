# Memory API

The Memory module is a foundational component of the ΨC-AI SDK that handles the storage, retrieval, and management of information units. Each piece of information is represented as a "Memory" object that includes metadata for context and management.

## Key Classes

### `Memory`

Represents a single memory unit in the system.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | str | The textual content of the memory |
| `uuid` | str | Unique identifier for the memory |
| `created_at` | float | Timestamp when the memory was created |
| `importance` | float | Current importance score of the memory (0.0 to 1.0) |
| `initial_importance` | float | Initial importance score assigned at creation |
| `embedding` | Optional[List[float]] | Vector embedding of the memory content |
| `tags` | List[str] | Tags associated with this memory for categorization |
| `metadata` | Dict[str, Any] | Additional metadata for the memory |
| `last_accessed` | float | Timestamp when the memory was last accessed |
| `access_count` | int | Number of times the memory has been accessed |
| `is_pinned` | bool | Whether this memory is exempt from decay and culling |

#### Methods

```python
def __init__(self, content: str, importance: float = 1.0, 
             tags: Optional[List[str]] = None,
             metadata: Optional[Dict[str, Any]] = None, 
             is_pinned: bool = False)
```

Create a new memory with the given content and attributes.

**Parameters:**
- `content`: The textual content of the memory
- `importance`: Initial importance score (default: 1.0)
- `tags`: Optional list of tags for categorization
- `metadata`: Optional additional metadata
- `is_pinned`: Whether this memory should be pinned (default: False)

```python
def access(self) -> None
```

Mark this memory as accessed, updating `last_accessed` time and `access_count`.

```python
def update_importance(self, new_importance: float) -> None
```

Update the importance score of this memory.

**Parameters:**
- `new_importance`: The new importance score to assign (between 0.0 and 1.0)

```python
def add_tag(self, tag: str) -> None
```

Add a tag to this memory.

**Parameters:**
- `tag`: The tag to add

```python
def pin(self) -> None
```

Pin this memory to exempt it from decay and culling.

```python
def unpin(self) -> None
```

Unpin this memory, making it subject to normal decay and culling again.

```python
def to_dict(self) -> Dict[str, Any]
```

Convert this memory to a dictionary for serialization.

**Returns:**
- Dictionary representation of the memory

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'Memory'
```

Create a Memory object from a dictionary.

**Parameters:**
- `data`: Dictionary containing memory data

**Returns:**
- Memory object reconstructed from the dictionary

### `MemoryStore`

Manages a collection of memories, providing storage, retrieval, and management functions.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `memories` | Dict[str, Memory] | Dictionary of all memories indexed by UUID |
| `archive` | MemoryArchive | Archive for storing removed memories |
| `decay_constant` | float | Rate at which memory importance decays over time |

#### Methods

```python
def __init__(self, decay_constant: float = 0.01, archive_path: Optional[str] = None)
```

Initialize a memory store.

**Parameters:**
- `decay_constant`: Rate at which memory importance decays over time (default: 0.01)
- `archive_path`: Optional file path for the memory archive

```python
def add(self, content: str, importance: float = 1.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_pinned: bool = False) -> str
```

Add a new memory to the store.

**Parameters:**
- `content`: The textual content of the memory
- `importance`: Initial importance score (default: 1.0)
- `tags`: Optional list of tags for categorization
- `metadata`: Optional additional metadata
- `is_pinned`: Whether this memory should be pinned (default: False)

**Returns:**
- UUID of the created memory

```python
def add_memory(self, memory: Memory) -> str
```

Add an existing Memory object to the store.

**Parameters:**
- `memory`: The Memory object to add

**Returns:**
- UUID of the added memory

```python
def get_memory(self, uuid: str) -> Optional[Memory]
```

Retrieve a memory by UUID.

**Parameters:**
- `uuid`: The UUID of the memory to retrieve

**Returns:**
- The Memory object if found, None otherwise

```python
def get_all_memories(self) -> List[Memory]
```

Get all memories in the store.

**Returns:**
- List of all Memory objects

```python
def archive_memory(self, uuid: str) -> bool
```

Move a memory from the active store to the archive.

**Parameters:**
- `uuid`: The UUID of the memory to archive

**Returns:**
- True if successful, False otherwise

```python
def restore_from_archive(self, uuid: str) -> bool
```

Restore a memory from the archive to the active store.

**Parameters:**
- `uuid`: The UUID of the memory to restore

**Returns:**
- True if successful, False otherwise

```python
def pin_memory(self, uuid: str) -> bool
```

Pin a memory to exempt it from decay and culling.

**Parameters:**
- `uuid`: The UUID of the memory to pin

**Returns:**
- True if successful, False otherwise

```python
def unpin_memory(self, uuid: str) -> bool
```

Unpin a memory, making it subject to normal decay and culling.

**Parameters:**
- `uuid`: The UUID of the memory to unpin

**Returns:**
- True if successful, False otherwise

```python
def update_memory_importance(self, uuid: str, importance: float) -> bool
```

Update the importance score of a memory.

**Parameters:**
- `uuid`: The UUID of the memory to update
- `importance`: New importance score (between 0.0 and 1.0)

**Returns:**
- True if successful, False otherwise

```python
def apply_importance_decay(self) -> None
```

Apply time-based decay to all memory importance scores.

```python
def cull_memories(self, importance_threshold: float = 0.2, 
                 max_memories: Optional[int] = None,
                 archive: bool = True) -> Tuple[int, int]
```

Remove low-importance memories from the store, optionally archiving them.

**Parameters:**
- `importance_threshold`: Memories with importance below this will be culled
- `max_memories`: If set, keep only this many memories
- `archive`: Whether to archive culled memories

**Returns:**
- Tuple of (number of memories culled, number archived)

```python
def export(self, filepath: str) -> None
```

Save all memories to a file.

**Parameters:**
- `filepath`: Path to save the memories

```python
def load(self, filepath: str) -> None
```

Load memories from a file.

**Parameters:**
- `filepath`: Path to load the memories from

```python
def delete_memory(self, uuid: str) -> bool
```

Permanently delete a memory.

**Parameters:**
- `uuid`: The UUID of the memory to delete

**Returns:**
- True if successful, False otherwise

## Usage Examples

### Basic Memory Management

```python
from psi_c_ai_sdk.memory import MemoryStore, Memory

# Create a memory store
memory_store = MemoryStore()

# Add memories
memory_id1 = memory_store.add("The capital of France is Paris.")
memory_id2 = memory_store.add(
    content="Python is a high-level programming language.",
    importance=0.8,
    tags=["programming", "python"]
)

# Retrieve memories
memory1 = memory_store.get_memory(memory_id1)
memory2 = memory_store.get_memory(memory_id2)

print(f"Memory 1: {memory1.content} (Importance: {memory1.importance})")
print(f"Memory 2: {memory2.content} (Tags: {memory2.tags})")

# Update a memory's importance
memory_store.update_memory_importance(memory_id1, 0.9)

# Add a tag to a memory
memory1.add_tag("geography")

# Access a memory (updates last_accessed and access_count)
memory1.access()
```

### Memory Archiving and Restoration

```python
from psi_c_ai_sdk.memory import MemoryStore

# Create a memory store with a file-backed archive
memory_store = MemoryStore(archive_path="memory_archive.json")

# Add some memories
ids = []
for i in range(10):
    ids.append(memory_store.add(f"Test memory {i}", importance=i/10))

# Archive low-importance memories
memory_store.cull_memories(importance_threshold=0.5)

# List remaining memories
active_memories = memory_store.get_all_memories()
print(f"Active memories: {len(active_memories)}")

# Restore a memory from the archive
memory_store.restore_from_archive(ids[2])
```

### Memory Persistence

```python
from psi_c_ai_sdk.memory import MemoryStore

# Create a memory store
memory_store = MemoryStore()

# Add memories
memory_store.add("The Earth orbits the Sun.")
memory_store.add("The Moon orbits the Earth.")

# Save memories to a file
memory_store.export("astronomy_memories.json")

# Create a new memory store and load the saved memories
new_store = MemoryStore()
new_store.load("astronomy_memories.json")

# Verify memories were loaded
loaded_memories = new_store.get_all_memories()
for memory in loaded_memories:
    print(f"Loaded: {memory.content}")
```

### Memory Pinning

```python
from psi_c_ai_sdk.memory import MemoryStore

# Create a memory store
memory_store = MemoryStore()

# Add a critical memory that should never be forgotten
critical_id = memory_store.add(
    "Emergency contact: 911",
    importance=1.0,
    is_pinned=True  # Pin at creation
)

# Add some regular memories
for i in range(20):
    memory_store.add(f"Regular memory {i}", importance=0.3)

# Pin another memory after creation
another_id = memory_store.add("Also important: Medical allergies to penicillin")
memory_store.pin_memory(another_id)

# Apply decay to all memories
memory_store.apply_importance_decay()

# Cull low-importance memories
memory_store.cull_memories(importance_threshold=0.5)

# Pinned memories remain even after culling
pinned = [m for m in memory_store.get_all_memories() if m.is_pinned]
print(f"Pinned memories remaining: {len(pinned)}")
for memory in pinned:
    print(f"- {memory.content}")
``` 