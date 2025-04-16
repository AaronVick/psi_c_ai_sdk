"""
Memory Compression: Module for compressing and decompressing memories semantically.

This module provides functionality for semantic compression of redundant or similar memories,
tracking compression levels, and decompressing memories for inspection.
"""

import time
import json
import logging
import uuid
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple, Union

from psi_c_ai_sdk.memory.memory import Memory, MemoryStore

# Configure logger
logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression levels for memories."""
    NONE = 0        # No compression
    LOW = 1         # Light compression, preserves most details
    MEDIUM = 2      # Moderate compression, preserves key details
    HIGH = 3        # Heavy compression, preserves only essential information
    ARCHIVE = 4     # Maximum compression for archival, major detail loss


class CompressedMemory:
    """
    Represents a compressed memory that contains multiple original memories.
    
    Attributes:
        uuid (str): Unique identifier for the compressed memory
        content (str): The compressed textual content
        original_uuids (List[str]): UUIDs of the original memories
        compression_level (CompressionLevel): Level of compression applied
        created_at (float): Timestamp when the compression was created
        compression_metadata (Dict[str, Any]): Additional metadata about compression
        original_memories (Optional[List[Dict]]): Optional storage of original memories
    """
    
    def __init__(
        self,
        content: str,
        original_uuids: List[str],
        compression_level: CompressionLevel,
        original_memories: Optional[List[Dict[str, Any]]] = None,
        compression_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a compressed memory.
        
        Args:
            content: The compressed textual content
            original_uuids: UUIDs of the original memories
            compression_level: Level of compression applied
            original_memories: Optional storage of the original memories for decompression
            compression_metadata: Additional metadata about compression
        """
        self.uuid = str(uuid.uuid4())
        self.content = content
        self.original_uuids = original_uuids
        self.compression_level = compression_level
        self.created_at = time.time()
        self.compression_metadata = compression_metadata or {}
        self.original_memories = original_memories
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the compressed memory
        """
        return {
            "uuid": self.uuid,
            "content": self.content,
            "original_uuids": self.original_uuids,
            "compression_level": self.compression_level.name,
            "created_at": self.created_at,
            "compression_metadata": self.compression_metadata,
            "original_memories": self.original_memories
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressedMemory':
        """
        Create a CompressedMemory from a dictionary.
        
        Args:
            data: Dictionary containing compressed memory data
            
        Returns:
            CompressedMemory object
        """
        compressed = cls(
            content=data["content"],
            original_uuids=data["original_uuids"],
            compression_level=CompressionLevel[data["compression_level"]],
            original_memories=data.get("original_memories"),
            compression_metadata=data.get("compression_metadata", {})
        )
        compressed.uuid = data["uuid"]
        compressed.created_at = data["created_at"]
        return compressed


class MemoryCompressor:
    """
    System for compressing and decompressing memories based on semantic similarity.
    
    This class provides functionality to:
    1. Identify redundant or similar memories
    2. Compress them into concise summaries
    3. Track compression levels
    4. Decompress memories for inspection when needed
    """
    
    def __init__(self, memory_store: MemoryStore, similarity_threshold: float = 0.85):
        """
        Initialize a memory compressor.
        
        Args:
            memory_store: The memory store containing memories to compress
            similarity_threshold: Threshold for considering memories similar (default: 0.85)
        """
        self.memory_store = memory_store
        self.similarity_threshold = similarity_threshold
        self.compressed_memories: Dict[str, CompressedMemory] = {}
        
        # Compression statistics
        self.stats = {
            "total_compressions": 0,
            "total_memories_compressed": 0,
            "compression_by_level": {level.name: 0 for level in CompressionLevel},
            "space_saved": 0,  # Rough estimation in bytes
            "last_compression": None
        }
    
    def identify_compressible_clusters(
        self, min_cluster_size: int = 2, max_cluster_size: int = 10
    ) -> List[List[Memory]]:
        """
        Identify clusters of similar memories that can be compressed.
        
        Args:
            min_cluster_size: Minimum number of memories to form a cluster
            max_cluster_size: Maximum number of memories in a cluster
            
        Returns:
            List of memory clusters (lists of Memory objects)
        """
        memories = self.memory_store.get_all_memories()
        
        # Skip if not enough memories
        if len(memories) < min_cluster_size:
            return []
        
        clusters = []
        processed_uuids = set()
        
        # Simple clustering approach - for a real system this would use embeddings
        for i, memory in enumerate(memories):
            if memory.uuid in processed_uuids or memory.is_pinned:
                continue
                
            # Start a new cluster with this memory
            cluster = [memory]
            processed_uuids.add(memory.uuid)
            
            # Find similar memories
            for other_memory in memories[i+1:]:
                if (other_memory.uuid not in processed_uuids and 
                    not other_memory.is_pinned and 
                    self._are_memories_similar(memory, other_memory) and
                    len(cluster) < max_cluster_size):
                    cluster.append(other_memory)
                    processed_uuids.add(other_memory.uuid)
            
            # Only keep clusters of sufficient size
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
                
        return clusters
    
    def compress_memories(
        self,
        memories: List[Memory],
        compression_level: CompressionLevel = CompressionLevel.MEDIUM,
        preserve_originals: bool = True
    ) -> Optional[CompressedMemory]:
        """
        Compress a group of similar memories into a single compressed memory.
        
        Args:
            memories: List of memories to compress
            compression_level: Level of compression to apply
            preserve_originals: Whether to store original memories for decompression
            
        Returns:
            CompressedMemory object or None if compression failed
        """
        if not memories or len(memories) < 2:
            return None
            
        # Extract memory content and metadata
        contents = [memory.content for memory in memories]
        uuids = [memory.uuid for memory in memories]
        
        # Semantic compression - generate summary based on compression level
        try:
            compressed_content = self._generate_compressed_content(contents, compression_level)
        except Exception as e:
            logger.error(f"Memory compression failed: {str(e)}")
            return None
            
        # Prepare original memories for decompression if needed
        original_memories = None
        if preserve_originals:
            original_memories = [memory.to_dict() for memory in memories]
            
        # Calculate compression stats
        original_size = sum(len(content) for content in contents)
        compressed_size = len(compressed_content)
        space_saved = original_size - compressed_size
            
        # Create compression metadata
        compression_metadata = {
            "original_count": len(memories),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "space_saved": space_saved,
            "compression_ratio": original_size / max(1, compressed_size)
        }
            
        # Create compressed memory
        compressed = CompressedMemory(
            content=compressed_content,
            original_uuids=uuids,
            compression_level=compression_level,
            original_memories=original_memories,
            compression_metadata=compression_metadata
        )
        
        # Update compression statistics
        self.stats["total_compressions"] += 1
        self.stats["total_memories_compressed"] += len(memories)
        self.stats["compression_by_level"][compression_level.name] += 1
        self.stats["space_saved"] += space_saved
        self.stats["last_compression"] = time.time()
        
        # Store the compressed memory
        self.compressed_memories[compressed.uuid] = compressed
        
        # Remove original memories from store
        for memory in memories:
            self.memory_store.delete_memory(memory.uuid)
            
        # Add as a new memory with special metadata
        self.memory_store.add(
            content=compressed_content,
            importance=max(memory.importance for memory in memories),
            tags=["compressed"] + list(set().union(*[set(memory.tags) for memory in memories])),
            metadata={
                "is_compressed": True,
                "compressed_memory_id": compressed.uuid,
                "compression_level": compression_level.name,
                "original_count": len(memories)
            }
        )
            
        return compressed
    
    def decompress_memory(self, compressed_uuid: str) -> Optional[List[Memory]]:
        """
        Decompress a compressed memory back into its original form.
        
        Args:
            compressed_uuid: UUID of the compressed memory to decompress
            
        Returns:
            List of decompressed Memory objects or None if decompression failed
        """
        compressed = self.compressed_memories.get(compressed_uuid)
        if not compressed:
            logger.warning(f"Compressed memory {compressed_uuid} not found")
            return None
            
        # Check if original memories are available
        if not compressed.original_memories:
            logger.warning(f"Original memories not available for {compressed_uuid}")
            # Attempt reconstruction based on compression level
            return self._reconstruct_memories(compressed)
            
        # Recreate memories from stored originals
        memories = []
        for memory_dict in compressed.original_memories:
            memory = Memory.from_dict(memory_dict)
            memories.append(memory)
            
        return memories
    
    def restore_memories(self, compressed_uuid: str) -> bool:
        """
        Restore original memories to the memory store and remove the compressed version.
        
        Args:
            compressed_uuid: UUID of the compressed memory to restore
            
        Returns:
            True if restoration succeeded, False otherwise
        """
        # Get the decompressed memories
        memories = self.decompress_memory(compressed_uuid)
        if not memories:
            return False
            
        # Find and remove the compressed memory from the store
        for uuid, memory in list(self.memory_store.memories.items()):
            metadata = memory.metadata
            if metadata.get("is_compressed") and metadata.get("compressed_memory_id") == compressed_uuid:
                self.memory_store.delete_memory(uuid)
                break
                
        # Add the original memories back to the store
        for memory in memories:
            self.memory_store.add_memory(memory)
            
        # Remove from compressed memories
        if compressed_uuid in self.compressed_memories:
            del self.compressed_memories[compressed_uuid]
            
        return True
    
    def run_compression_cycle(
        self,
        compression_level: CompressionLevel = CompressionLevel.MEDIUM,
        min_cluster_size: int = 2,
        preserve_originals: bool = True
    ) -> Dict[str, Any]:
        """
        Run a full compression cycle to identify and compress redundant memories.
        
        Args:
            compression_level: Level of compression to apply
            min_cluster_size: Minimum number of memories to form a cluster
            preserve_originals: Whether to store original memories for decompression
            
        Returns:
            Statistics about the compression cycle
        """
        # Find memory clusters
        clusters = self.identify_compressible_clusters(min_cluster_size=min_cluster_size)
        
        # Compress each cluster
        compressed_count = 0
        original_count = 0
        
        for cluster in clusters:
            compressed = self.compress_memories(
                memories=cluster,
                compression_level=compression_level,
                preserve_originals=preserve_originals
            )
            
            if compressed:
                compressed_count += 1
                original_count += len(cluster)
        
        # Return statistics
        return {
            "clusters_found": len(clusters),
            "clusters_compressed": compressed_count,
            "memories_compressed": original_count,
            "timestamp": time.time()
        }
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory compression.
        
        Returns:
            Dictionary with compression statistics
        """
        return {
            **self.stats,
            "compressed_memories_count": len(self.compressed_memories),
            "compression_level_counts": {
                level.name: sum(1 for cm in self.compressed_memories.values() 
                               if cm.compression_level == level)
                for level in CompressionLevel
            }
        }
    
    def export_compressed_memories(self, filepath: str) -> None:
        """
        Export all compressed memories to a file.
        
        Args:
            filepath: Path where to save the export
        """
        compressed_dicts = [cm.to_dict() for cm in self.compressed_memories.values()]
        
        with open(filepath, 'w') as f:
            json.dump({
                "compressed_memories": compressed_dicts,
                "stats": self.stats,
                "timestamp": time.time()
            }, f, indent=2)
    
    def import_compressed_memories(self, filepath: str) -> int:
        """
        Import compressed memories from a file.
        
        Args:
            filepath: Path to the import file
            
        Returns:
            Number of compressed memories imported
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        count = 0
        for cm_dict in data.get("compressed_memories", []):
            compressed = CompressedMemory.from_dict(cm_dict)
            self.compressed_memories[compressed.uuid] = compressed
            count += 1
            
        # Update stats if available
        if "stats" in data:
            self.stats.update(data["stats"])
            
        return count
    
    def _are_memories_similar(self, memory1: Memory, memory2: Memory) -> bool:
        """
        Check if two memories are semantically similar.
        
        In a real implementation, this would use embedding similarity.
        This simplified version uses tag and content overlap.
        
        Args:
            memory1: First memory to compare
            memory2: Second memory to compare
            
        Returns:
            True if memories are similar, False otherwise
        """
        # Check for tag overlap
        tags1 = set(memory1.tags)
        tags2 = set(memory2.tags)
        tag_overlap = len(tags1.intersection(tags2)) / max(1, len(tags1.union(tags2)))
        
        # Simple text similarity (would use embeddings in real implementation)
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        word_overlap = len(words1.intersection(words2)) / max(1, len(words1.union(words2)))
        
        # Combined similarity score
        similarity = 0.4 * tag_overlap + 0.6 * word_overlap
        
        return similarity >= self.similarity_threshold
    
    def _generate_compressed_content(
        self, contents: List[str], compression_level: CompressionLevel
    ) -> str:
        """
        Generate compressed content based on original content and compression level.
        
        In a production system, this would use more sophisticated NLP techniques.
        This is a simplified version for demonstration.
        
        Args:
            contents: List of original memory contents
            compression_level: Level of compression to apply
            
        Returns:
            Compressed content
        """
        # Simple compression strategy based on level
        if compression_level == CompressionLevel.NONE:
            # No compression - return all contents combined
            return "\n\n".join(contents)
            
        elif compression_level == CompressionLevel.LOW:
            # Low compression - keep most of the content
            # In this simplified version, we just concatenate with a summary
            common_words = self._extract_common_topics(contents)
            summary = f"Group of {len(contents)} similar memories about: {', '.join(common_words)}"
            return f"{summary}\n\n" + "\n---\n".join(contents)
            
        elif compression_level == CompressionLevel.MEDIUM:
            # Medium compression - extract key information
            common_words = self._extract_common_topics(contents)
            unique_aspects = self._extract_unique_aspects(contents)
            
            summary = (
                f"Compressed group of {len(contents)} similar memories about: "
                f"{', '.join(common_words)}.\n\n"
                f"Key points:\n- " + "\n- ".join(unique_aspects)
            )
            return summary
            
        elif compression_level == CompressionLevel.HIGH:
            # High compression - just key topics and count
            common_words = self._extract_common_topics(contents)
            return (
                f"Highly compressed record of {len(contents)} memories about: "
                f"{', '.join(common_words)}."
            )
            
        elif compression_level == CompressionLevel.ARCHIVE:
            # Archive level - minimal information
            common_words = self._extract_common_topics(contents)
            return f"Archive: {len(contents)} memories on {', '.join(common_words[:3])}."
    
    def _extract_common_topics(self, contents: List[str], max_topics: int = 5) -> List[str]:
        """
        Extract common topics from a list of memory contents.
        
        Args:
            contents: List of memory contents
            max_topics: Maximum number of topics to return
            
        Returns:
            List of common topics
        """
        # Simple word frequency analysis
        # In a real system, this would use topic modeling or entity extraction
        all_words = []
        for content in contents:
            words = content.lower().split()
            all_words.extend(words)
            
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            # Skip short words and common stop words
            if len(word) <= 3 or word in {'the', 'and', 'for', 'that', 'this', 'with'}:
                continue
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top topics
        return [word for word, _ in sorted_words[:max_topics]]
    
    def _extract_unique_aspects(self, contents: List[str], max_aspects: int = 5) -> List[str]:
        """
        Extract unique aspects or key information from memory contents.
        
        Args:
            contents: List of memory contents
            max_aspects: Maximum number of aspects to return
            
        Returns:
            List of unique aspects
        """
        # Simple approach: extract first sentence from each content
        # In a real system, this would use more sophisticated extraction
        aspects = []
        for content in contents:
            sentences = content.split('.')
            if sentences:
                first_sentence = sentences[0].strip()
                if first_sentence and first_sentence not in aspects:
                    aspects.append(first_sentence)
                    if len(aspects) >= max_aspects:
                        break
                        
        return aspects
    
    def _reconstruct_memories(self, compressed: CompressedMemory) -> List[Memory]:
        """
        Attempt to reconstruct original memories from a compressed memory.
        
        This is used when original memories are not available.
        It's an imperfect reconstruction that tries to approximate the originals.
        
        Args:
            compressed: CompressedMemory to reconstruct from
            
        Returns:
            List of reconstructed Memory objects
        """
        # Extract information from compressed content
        content = compressed.content
        
        # Create a warning memory about the reconstruction
        warning_memory = Memory(
            content=(
                "WARNING: This is a reconstructed memory from a compressed version. "
                "Original details have been lost. Compressed content was:\n\n" + content
            ),
            importance=1.0,
            tags=["reconstructed", "compressed_memory"],
            metadata={
                "original_compressed_id": compressed.uuid,
                "compression_level": compressed.compression_level.name,
                "reconstruction_time": time.time(),
                "is_reconstruction": True
            }
        )
        
        # For severe compression levels, we can only return the warning
        if compressed.compression_level in {CompressionLevel.HIGH, CompressionLevel.ARCHIVE}:
            return [warning_memory]
            
        # For medium compression, try to extract key points
        memories = [warning_memory]
        if "Key points:" in content:
            points_section = content.split("Key points:")[1]
            points = [p.strip() for p in points_section.split("-") if p.strip()]
            
            for point in points:
                memory = Memory(
                    content=point,
                    importance=0.7,
                    tags=["reconstructed", "compressed_point"],
                    metadata={
                        "original_compressed_id": compressed.uuid,
                        "is_reconstruction": True
                    }
                )
                memories.append(memory)
                
        # For low compression, we might have some original contents
        elif "---" in content:
            sections = content.split("---")
            for section in sections[1:]:  # Skip the summary
                if section.strip():
                    memory = Memory(
                        content=section.strip(),
                        importance=0.8,
                        tags=["reconstructed", "compressed_section"],
                        metadata={
                            "original_compressed_id": compressed.uuid,
                            "is_reconstruction": True
                        }
                    )
                    memories.append(memory)
                    
        return memories 