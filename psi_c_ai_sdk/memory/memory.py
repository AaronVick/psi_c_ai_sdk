"""
Memory System: Core module for storing, retrieving, and managing memories.
"""

import json
import time
import uuid
import os
from typing import Dict, List, Union, Optional, Any, Tuple, Set


class Memory:
    """
    Represents a single memory unit in the system.
    
    Attributes:
        content (str): The textual content of the memory
        uuid (str): Unique identifier for the memory
        created_at (float): Timestamp when the memory was created
        importance (float): Current importance score of the memory
        initial_importance (float): Initial importance score assigned at creation
        embedding (Optional[List[float]]): Vector embedding of the memory content
        tags (List[str]): Tags associated with this memory for categorization
        metadata (Dict[str, Any]): Additional metadata for the memory
        last_accessed (float): Timestamp when the memory was last accessed
        access_count (int): Number of times the memory has been accessed
        is_pinned (bool): Whether this memory is pinned (exempt from decay and culling)
    """
    
    def __init__(
        self, 
        content: str, 
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_pinned: bool = False
    ):
        """
        Initialize a new memory.
        
        Args:
            content: The textual content of the memory
            importance: Initial importance score (default: 1.0)
            tags: Optional list of tags for categorization
            metadata: Optional additional metadata
            is_pinned: Whether this memory should be pinned (default: False)
        """
        self.content = content
        self.uuid = str(uuid.uuid4())
        self.created_at = time.time()
        self.importance = importance
        self.initial_importance = importance
        self.embedding = None
        self.tags = tags or []
        self.metadata = metadata or {}
        self.last_accessed = self.created_at
        self.access_count = 0
        self.is_pinned = is_pinned
    
    def access(self) -> None:
        """Mark this memory as accessed, updating last_accessed time and access_count."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def update_importance(self, new_importance: float) -> None:
        """
        Update the importance score of this memory.
        
        Args:
            new_importance: The new importance score to assign
        """
        self.importance = new_importance
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to this memory.
        
        Args:
            tag: The tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def pin(self) -> None:
        """
        Pin this memory to exempt it from decay and culling.
        Pinned memories are considered critical and will be preserved.
        """
        self.is_pinned = True
        # Optionally add a tag to indicate pinned status
        self.add_tag("pinned")
    
    def unpin(self) -> None:
        """
        Unpin this memory, making it subject to normal decay and culling again.
        """
        self.is_pinned = False
        # Remove the pinned tag if it exists
        if "pinned" in self.tags:
            self.tags.remove("pinned")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this memory to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the memory
        """
        return {
            "uuid": self.uuid,
            "content": self.content,
            "created_at": self.created_at,
            "importance": self.importance,
            "initial_importance": self.initial_importance,
            "embedding": self.embedding,
            "tags": self.tags,
            "metadata": self.metadata,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "is_pinned": self.is_pinned
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """
        Create a Memory object from a dictionary.
        
        Args:
            data: Dictionary containing memory data
            
        Returns:
            Memory object reconstructed from the dictionary
        """
        memory = cls(
            content=data["content"],
            importance=data["importance"],
            tags=data["tags"],
            metadata=data["metadata"],
            is_pinned=data.get("is_pinned", False)  # Default to False for backward compatibility
        )
        memory.uuid = data["uuid"]
        memory.created_at = data["created_at"]
        memory.initial_importance = data["initial_importance"]
        memory.embedding = data["embedding"]
        memory.last_accessed = data["last_accessed"]
        memory.access_count = data["access_count"]
        return memory


class MemoryArchive:
    """
    Storage for archived memories that have been removed from the active memory store.
    
    The archive allows old memories to be preserved for potential future reference without
    occupying space in the active memory store.
    """
    
    def __init__(self, archive_path: Optional[str] = None):
        """
        Initialize a memory archive.
        
        Args:
            archive_path: Optional file path where archive will be persisted
        """
        self.memories: Dict[str, Memory] = {}
        self.archive_path = archive_path
        self.last_archived_at: Optional[float] = None
    
    def add(self, memory: Memory) -> str:
        """
        Add a memory to the archive.
        
        Args:
            memory: The Memory object to archive
            
        Returns:
            The UUID of the archived memory
        """
        self.memories[memory.uuid] = memory
        self.last_archived_at = time.time()
        return memory.uuid
    
    def add_many(self, memories: List[Memory]) -> List[str]:
        """
        Add multiple memories to the archive.
        
        Args:
            memories: List of Memory objects to archive
            
        Returns:
            List of UUIDs of the archived memories
        """
        uuids = []
        for memory in memories:
            self.memories[memory.uuid] = memory
            uuids.append(memory.uuid)
        
        if memories:
            self.last_archived_at = time.time()
        
        return uuids
    
    def get(self, uuid: str) -> Optional[Memory]:
        """
        Retrieve a memory from the archive by UUID.
        
        Args:
            uuid: UUID of the memory to retrieve
            
        Returns:
            The Memory object if found, None otherwise
        """
        return self.memories.get(uuid)
    
    def get_all(self) -> List[Memory]:
        """
        Get all memories in the archive.
        
        Returns:
            List of all archived Memory objects
        """
        return list(self.memories.values())
    
    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Basic search for memories containing the query string.
        
        Args:
            query: Search string to match against memory content
            limit: Maximum number of results to return
            
        Returns:
            List of matching Memory objects
        """
        query = query.lower()
        results = []
        
        for memory in self.memories.values():
            if query in memory.content.lower():
                results.append(memory)
                if len(results) >= limit:
                    break
        
        return results
    
    def count(self) -> int:
        """
        Get the number of memories in the archive.
        
        Returns:
            Count of archived memories
        """
        return len(self.memories)
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save the archive to a file.
        
        Args:
            filepath: Path where to save the archive (defaults to self.archive_path)
        """
        target_path = filepath or self.archive_path
        if not target_path:
            raise ValueError("No archive path specified")
        
        memory_dicts = [memory.to_dict() for memory in self.memories.values()]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        
        with open(target_path, 'w') as f:
            json.dump({
                "memories": memory_dicts,
                "archived_at": self.last_archived_at or time.time()
            }, f, indent=2)
    
    def load(self, filepath: Optional[str] = None) -> None:
        """
        Load memories from an archive file.
        
        Args:
            filepath: Path to the archive file (defaults to self.archive_path)
        """
        target_path = filepath or self.archive_path
        if not target_path:
            raise ValueError("No archive path specified")
        
        if not os.path.exists(target_path):
            return  # No archive file exists yet
        
        with open(target_path, 'r') as f:
            data = json.load(f)
        
        self.memories = {}
        for memory_dict in data["memories"]:
            memory = Memory.from_dict(memory_dict)
            self.memories[memory.uuid] = memory
        
        self.last_archived_at = data.get("archived_at")
    
    def restore_to_store(self, memory_store: 'MemoryStore', uuid: str) -> bool:
        """
        Restore a memory from the archive to the active memory store.
        
        Args:
            memory_store: The MemoryStore to restore the memory to
            uuid: UUID of the memory to restore
            
        Returns:
            True if the memory was found and restored, False otherwise
        """
        memory = self.memories.get(uuid)
        if not memory:
            return False
        
        # Add to memory store
        memory_store.add_memory(memory)
        
        # Remove from archive
        del self.memories[uuid]
        
        return True
    
    def clear(self) -> None:
        """Clear all memories from the archive."""
        self.memories = {}


class MemoryStore:
    """
    A system for storing and managing memories with importance tracking.
    
    The MemoryStore implements the core memory functionality per the ΨC-AI SDK design,
    including memory decay over time according to the formula:
    I(t) = I_0 · e^(-λt) where:
    - I_0: Initial importance
    - λ: Decay constant
    - t: Time since memory creation or last access
    """
    
    def __init__(self, decay_constant: float = 0.01, archive_path: Optional[str] = None):
        """
        Initialize a new memory store.
        
        Args:
            decay_constant: Lambda value for importance decay (default: 0.01)
            archive_path: Optional path for memory archive storage
        """
        self.memories: Dict[str, Memory] = {}
        self.decay_constant = decay_constant
        self.archive = MemoryArchive(archive_path)
        if archive_path and os.path.exists(archive_path):
            self.archive.load()
    
    def add(
        self, 
        content: str, 
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_pinned: bool = False
    ) -> str:
        """
        Add a new memory to the store.
        
        Args:
            content: The textual content of the memory
            importance: Initial importance score (default: 1.0)
            tags: Optional list of tags for categorization
            metadata: Optional additional metadata
            is_pinned: Whether this memory should be pinned (default: False)
            
        Returns:
            UUID of the newly created memory
        """
        memory = Memory(
            content=content,
            importance=importance,
            tags=tags,
            metadata=metadata,
            is_pinned=is_pinned
        )
        self.memories[memory.uuid] = memory
        return memory.uuid
    
    def add_memory(self, memory: Memory) -> str:
        """
        Add an existing Memory object to the store.
        
        Args:
            memory: The Memory object to add
            
        Returns:
            UUID of the added memory
        """
        self.memories[memory.uuid] = memory
        return memory.uuid
    
    def get_memory(self, uuid: str) -> Optional[Memory]:
        """
        Retrieve a memory by its UUID.
        
        Args:
            uuid: The UUID of the memory to retrieve
            
        Returns:
            The Memory object if found, None otherwise
        """
        memory = self.memories.get(uuid)
        if memory:
            memory.access()
        return memory
    
    def get_all_memories(self) -> List[Memory]:
        """
        Get all memories in the store.
        
        Returns:
            List of all Memory objects
        """
        return list(self.memories.values())
    
    def archive_memory(self, uuid: str) -> bool:
        """
        Move a memory from the active store to the archive.
        
        Args:
            uuid: UUID of the memory to archive
            
        Returns:
            True if the memory was found and archived, False otherwise
        """
        memory = self.memories.get(uuid)
        if not memory:
            return False
        
        # Add to archive
        self.archive.add(memory)
        
        # Remove from active store
        del self.memories[uuid]
        
        return True
    
    def archive_memories(self, uuids: List[str]) -> int:
        """
        Move multiple memories from the active store to the archive.
        
        Args:
            uuids: List of UUIDs to archive
            
        Returns:
            Number of memories successfully archived
        """
        memories_to_archive = []
        valid_uuids = []
        
        # Find all valid memories
        for uuid in uuids:
            memory = self.memories.get(uuid)
            if memory:
                memories_to_archive.append(memory)
                valid_uuids.append(uuid)
        
        # Add to archive
        if memories_to_archive:
            self.archive.add_many(memories_to_archive)
            
            # Remove from active store
            for uuid in valid_uuids:
                del self.memories[uuid]
        
        return len(valid_uuids)
    
    def restore_from_archive(self, uuid: str) -> bool:
        """
        Restore a memory from the archive to the active store.
        
        Args:
            uuid: UUID of the memory to restore
            
        Returns:
            True if the memory was found and restored, False otherwise
        """
        return self.archive.restore_to_store(self, uuid)
    
    def search_archive(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Search for memories in the archive.
        
        Args:
            query: Search string to match against memory content
            limit: Maximum number of results to return
            
        Returns:
            List of matching Memory objects from the archive
        """
        return self.archive.search(query, limit)
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory archive.
        
        Returns:
            Dictionary with archive statistics
        """
        return {
            "count": self.archive.count(),
            "last_archived_at": self.archive.last_archived_at
        }
    
    def pin_memory(self, uuid: str) -> bool:
        """
        Pin a memory to exempt it from decay and culling.
        
        Args:
            uuid: The UUID of the memory to pin
            
        Returns:
            True if the memory was found and pinned, False otherwise
        """
        memory = self.memories.get(uuid)
        if memory:
            memory.pin()
            return True
        return False
    
    def unpin_memory(self, uuid: str) -> bool:
        """
        Unpin a previously pinned memory.
        
        Args:
            uuid: The UUID of the memory to unpin
            
        Returns:
            True if the memory was found and unpinned, False otherwise
        """
        memory = self.memories.get(uuid)
        if memory:
            memory.unpin()
            return True
        return False
    
    def get_pinned_memories(self) -> List[Memory]:
        """
        Get all pinned memories in the store.
        
        Returns:
            List of pinned Memory objects
        """
        return [memory for memory in self.memories.values() if memory.is_pinned]
    
    def update_memory_importance(self, uuid: str, importance: float) -> bool:
        """
        Update the importance of a memory.
        
        Args:
            uuid: The UUID of the memory to update
            importance: The new importance value
            
        Returns:
            True if the memory was found and updated, False otherwise
        """
        memory = self.memories.get(uuid)
        if memory:
            memory.update_importance(importance)
            return True
        return False
    
    def apply_importance_decay(self) -> None:
        """
        Apply importance decay to all non-pinned memories based on time elapsed.
        
        Uses the formula: I(t) = I_0 · e^(-λt)
        """
        current_time = time.time()
        
        for memory in self.memories.values():
            # Skip pinned memories - they don't decay
            if memory.is_pinned:
                continue
                
            # Calculate time since last access
            time_elapsed = current_time - memory.last_accessed
            
            # Apply decay formula
            decay_factor = 2.71828 ** (-self.decay_constant * time_elapsed)
            memory.importance = memory.initial_importance * decay_factor
    
    def get_memories_by_importance(self, threshold: float = 0.0) -> List[Memory]:
        """
        Get memories with importance above the specified threshold.
        
        Args:
            threshold: Minimum importance value (default: 0.0)
            
        Returns:
            List of memories with importance >= threshold
        """
        return [
            memory for memory in self.memories.values() 
            if memory.importance >= threshold or memory.is_pinned
        ]
    
    def cull_memories(self, importance_threshold: float = 0.2, max_memories: Optional[int] = None, 
                     archive: bool = True) -> Tuple[int, int]:
        """
        Remove low-importance memories from the store, optionally archiving them.
        
        This will never remove pinned memories regardless of their importance.
        
        Args:
            importance_threshold: Minimum importance to retain (default: 0.2)
            max_memories: Optional maximum number of memories to keep
            archive: Whether to archive culled memories (default: True)
            
        Returns:
            Tuple of (number of memories removed, number archived)
        """
        # First, apply decay to ensure importance values are current
        self.apply_importance_decay()
        
        # Store original count for return value
        original_count = len(self.memories)
        
        # Get memories to keep (pinned or above threshold)
        memories_to_keep = {
            uuid: memory for uuid, memory in self.memories.items()
            if memory.is_pinned or memory.importance >= importance_threshold
        }
        
        # Identify memories to remove
        memories_to_remove = {
            uuid: memory for uuid, memory in self.memories.items()
            if uuid not in memories_to_keep
        }
        
        # If we have a max_memories limit and still too many, sort by importance
        if max_memories and len(memories_to_keep) > max_memories:
            # Always keep pinned memories
            pinned = {uuid: mem for uuid, mem in memories_to_keep.items() if mem.is_pinned}
            unpinned = {uuid: mem for uuid, mem in memories_to_keep.items() if not mem.is_pinned}
            
            # Sort unpinned by importance (highest first)
            sorted_unpinned = sorted(
                unpinned.items(), 
                key=lambda x: x[1].importance,
                reverse=True
            )
            
            # Keep only top unpinned memories that fit within the limit
            unpinned_to_keep = dict(sorted_unpinned[:max(0, max_memories - len(pinned))])
            
            # Add the rest to memories_to_remove
            for i in range(max(0, max_memories - len(pinned)), len(sorted_unpinned)):
                uuid, memory = sorted_unpinned[i]
                memories_to_remove[uuid] = memory
            
            # Combine pinned and top unpinned as the final keep list
            memories_to_keep = {**pinned, **unpinned_to_keep}
        
        # Archive if requested
        archived_count = 0
        if archive and memories_to_remove:
            # Add to archive
            archived_count = len(memories_to_remove)
            self.archive.add_many(list(memories_to_remove.values()))
        
        # Calculate memories removed
        num_removed = original_count - len(memories_to_keep)
        
        # Update memories dictionary
        self.memories = memories_to_keep
        
        return num_removed, archived_count
    
    def save_archive(self, filepath: Optional[str] = None) -> None:
        """
        Save the memory archive to a file.
        
        Args:
            filepath: Optional path to save to (defaults to self.archive.archive_path)
        """
        self.archive.save(filepath)
    
    def load_archive(self, filepath: Optional[str] = None) -> None:
        """
        Load the memory archive from a file.
        
        Args:
            filepath: Optional path to load from (defaults to self.archive.archive_path)
        """
        self.archive.load(filepath)
    
    def export(self, filepath: str) -> None:
        """
        Export all memories to a JSON file.
        
        Args:
            filepath: Path to the output JSON file
        """
        memory_dicts = [memory.to_dict() for memory in self.memories.values()]
        
        with open(filepath, 'w') as f:
            json.dump({
                "memories": memory_dicts,
                "decay_constant": self.decay_constant
            }, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """
        Load memories from a JSON file.
        
        Args:
            filepath: Path to the JSON file to load
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.decay_constant = data.get("decay_constant", 0.01)
        
        self.memories = {}
        for memory_dict in data["memories"]:
            memory = Memory.from_dict(memory_dict)
            self.memories[memory.uuid] = memory
    
    def delete_memory(self, uuid: str) -> bool:
        """
        Delete a memory from the store.
        
        Args:
            uuid: The UUID of the memory to delete
            
        Returns:
            True if the memory was found and deleted, False otherwise
        """
        if uuid in self.memories:
            del self.memories[uuid]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        self.memories = {} 