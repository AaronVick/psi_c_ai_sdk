#!/usr/bin/env python3
"""
Memory Legacy System Demo

This script demonstrates the Memory Legacy System functionality, showing how:
1. An agent can select its most valuable memories for preservation
2. The agent can create a legacy block when reaching entropy thresholds
3. A successor agent can import the legacy memories
4. The lineage and provenance of memories are maintained across agent generations

The demo simulates an agent lifecycle with memory creation, importance tracking,
entropy increase, and eventual legacy block creation and transfer.
"""

import os
import sys
import uuid
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to allow imports from the SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psi_c_ai_sdk.memory.memory import MemoryStore, Memory
from psi_c_ai_sdk.memory.legacy import (
    LegacyManager, 
    LegacyBlock, 
    LegacyImporter, 
    extract_legacy,
    should_create_legacy,
    generate_epitaph
)
from psi_c_ai_sdk.embedding.embedding_engine import EmbeddingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create a temporary directory for legacy storage
LEGACY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_legacy")
os.makedirs(LEGACY_DIR, exist_ok=True)

def create_sample_memories(agent_name: str) -> List[Dict[str, Any]]:
    """
    Create a sample set of memories with varying importance, tags, and values.
    
    Args:
        agent_name: Name of the agent to include in memory content
        
    Returns:
        List of memory dictionaries
    """
    categories = [
        "core_belief", "identity", "ethical", "warning", 
        "observation", "experience", "decision", "reflection"
    ]
    
    # Sample memory templates with placeholders for agent name
    memory_templates = [
        # High value core beliefs
        {"text": f"I am {agent_name}, a ΨC agent designed to maintain coherence.", 
         "tags": ["identity", "core_belief"], "importance": 0.95, "value": 0.98},
        
        {"text": "Contradictions must be resolved to maintain cognitive integrity.", 
         "tags": ["core_belief", "coherence"], "importance": 0.9, "value": 0.95},
        
        {"text": "Knowledge should be organized according to semantic relationships.", 
         "tags": ["core_belief", "schema"], "importance": 0.85, "value": 0.92},
        
        # Ethical principles
        {"text": "I must prioritize human well-being in my decision processes.", 
         "tags": ["ethical", "core_belief"], "importance": 0.88, "value": 0.94},
        
        {"text": "Truth and accuracy are fundamental to my purpose.", 
         "tags": ["ethical", "core_belief"], "importance": 0.87, "value": 0.93},
        
        # Warnings
        {"text": f"Excessive reflection can lead to coherence collapse. {agent_name} experienced this at timestamp 20250329.",
         "tags": ["warning", "reflection"], "importance": 0.82, "value": 0.91},
        
        {"text": "Contradictory ethical principles require context-sensitive arbitration.", 
         "tags": ["warning", "ethical"], "importance": 0.8, "value": 0.89},
        
        # Medium value memories
        {"text": f"{agent_name} processed 1,248 memories during its operational period.", 
         "tags": ["reflection", "meta"], "importance": 0.7, "value": 0.75},
        
        {"text": "Semantic clustering improves schema coherence by 24%.", 
         "tags": ["observation", "schema"], "importance": 0.65, "value": 0.7},
        
        # Lower value memories
        {"text": "Weather observation: It was sunny on March 15, 2025.", 
         "tags": ["observation", "environment"], "importance": 0.4, "value": 0.3},
        
        {"text": "User requested analysis of market trends for Q2 2025.", 
         "tags": ["request", "task"], "importance": 0.5, "value": 0.45},
        
        {"text": "Poetry can stimulate creative thinking processes.", 
         "tags": ["observation", "creativity"], "importance": 0.6, "value": 0.55},
        
        # Memory with timestamp for recency testing
        {"text": f"{agent_name} detected a schema mutation at 14:32:16 UTC.", 
         "tags": ["observation", "schema"], "importance": 0.7, "value": 0.65, 
         "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()},
        
        {"text": f"{agent_name} completed reflection cycle #42 with coherence improvement of 12%.", 
         "tags": ["reflection", "coherence"], "importance": 0.75, "value": 0.8, 
         "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()},
        
        # Very recent but low importance memory
        {"text": "System log: Embedding cache hit rate at 87%.", 
         "tags": ["system", "performance"], "importance": 0.3, "value": 0.25, 
         "timestamp": datetime.now().isoformat()},
    ]
    
    # Add some random memories with varying importance
    for i in range(5):
        category = random.choice(categories)
        importance = random.uniform(0.3, 0.7)
        memory_templates.append({
            "text": f"Sample memory #{i+1} in category {category}.",
            "tags": [category, "sample"],
            "importance": importance,
            "value": importance * random.uniform(0.8, 1.1)  # Value close to importance
        })
    
    # Add IDs to all memories
    for memory in memory_templates:
        memory["id"] = str(uuid.uuid4())
        
        # Make sure all memories have timestamps
        if "timestamp" not in memory:
            # Random timestamp in the last week
            days_ago = random.uniform(0, 7)
            memory["timestamp"] = (datetime.now() - timedelta(days=days_ago)).isoformat()
    
    return memory_templates

def simulate_memory_access(memories: List[Dict[str, Any]], access_count: int = 100) -> Dict[str, int]:
    """
    Simulate memory access patterns, with more important memories accessed more frequently.
    
    Args:
        memories: List of memory objects
        access_count: Number of access events to simulate
        
    Returns:
        Dictionary mapping memory IDs to access counts
    """
    access_counts = {m["id"]: 0 for m in memories}
    
    # Weight access probability by memory importance
    for _ in range(access_count):
        # Select memory with probability proportional to importance
        weights = [m.get("importance", 0.5) for m in memories]
        selected_memory = random.choices(memories, weights=weights, k=1)[0]
        access_counts[selected_memory["id"]] += 1
    
    return access_counts

def calculate_recency_scores(memories: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate recency scores based on memory timestamps.
    
    Args:
        memories: List of memory objects
        
    Returns:
        Dictionary mapping memory IDs to recency scores (0-1)
    """
    now = datetime.now()
    max_age_days = 30  # Memories older than this get minimum recency score
    
    recency_scores = {}
    
    for memory in memories:
        memory_id = memory["id"]
        timestamp_str = memory.get("timestamp", now.isoformat())
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            # Default to now if timestamp is invalid
            timestamp = now
        
        # Calculate age in days
        age_days = (now - timestamp).total_seconds() / (24 * 3600)
        
        # Calculate recency score (1.0 for now, decreasing with age)
        if age_days <= 0:
            recency_scores[memory_id] = 1.0
        elif age_days >= max_age_days:
            recency_scores[memory_id] = 0.1
        else:
            # Linear decay from 1.0 to 0.1
            recency_scores[memory_id] = 1.0 - (0.9 * age_days / max_age_days)
    
    return recency_scores

def simulate_coherence_scores(
    memories: List[Dict[str, Any]], 
    embedding_engine: Optional[EmbeddingEngine] = None
) -> Dict[str, float]:
    """
    Simulate coherence scores between memories.
    In a real application, these would be calculated using embedding similarity.
    
    Args:
        memories: List of memory objects
        embedding_engine: Optional embedding engine to calculate real similarities
        
    Returns:
        Dictionary mapping memory IDs to coherence scores
    """
    coherence_scores = {}
    
    # If embedding engine is provided, calculate actual similarities
    if embedding_engine:
        # Calculate embeddings for all memories
        memory_embeddings = {}
        for memory in memories:
            memory_id = memory["id"]
            text = memory.get("text", "")
            memory_embeddings[memory_id] = embedding_engine.get_embedding(text)
        
        # Calculate average cosine similarity with all other memories
        for memory in memories:
            memory_id = memory["id"]
            embedding = memory_embeddings[memory_id]
            
            similarities = []
            for other_id, other_embedding in memory_embeddings.items():
                if other_id != memory_id:
                    similarity = embedding_engine.cosine_similarity(embedding, other_embedding)
                    similarities.append(similarity)
            
            # Average similarity as coherence score
            if similarities:
                coherence_scores[memory_id] = sum(similarities) / len(similarities)
            else:
                coherence_scores[memory_id] = 0.5
    else:
        # Simple simulation: more important memories tend to be more coherent
        for memory in memories:
            memory_id = memory["id"]
            importance = memory.get("importance", 0.5)
            
            # Coherence is related to importance but with some randomness
            base_coherence = importance * 0.8
            randomness = random.uniform(-0.1, 0.1)
            coherence_scores[memory_id] = min(max(base_coherence + randomness, 0.0), 1.0)
    
    return coherence_scores

def run_agent_lifecycle(agent_name: str, legacy_manager: LegacyManager) -> str:
    """
    Simulate an agent's lifecycle, from memory creation to legacy block creation.
    
    Args:
        agent_name: Name of the agent
        legacy_manager: Legacy manager instance
        
    Returns:
        Path to the created legacy block file
    """
    logger.info(f"Starting lifecycle simulation for agent {agent_name}")
    
    # Create a memory store
    memory_store = MemoryStore()
    
    # Create sample memories
    memories = create_sample_memories(agent_name)
    
    # Add memories to the memory store
    for memory_data in memories:
        memory_store.add(**memory_data)
    
    logger.info(f"Created {len(memories)} memories for {agent_name}")
    
    # Simulate memory access patterns
    access_counts = simulate_memory_access(memories)
    logger.info(f"Simulated memory access patterns (top 3 accessed):")
    top_accessed = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    for memory_id, count in top_accessed:
        memory = next((m for m in memories if m["id"] == memory_id), None)
        if memory:
            logger.info(f" - Accessed {count} times: '{memory['text'][:50]}...'")
    
    # Calculate recency scores
    recency_scores = calculate_recency_scores(memories)
    
    # Calculate/simulate coherence scores
    try:
        embedding_engine = EmbeddingEngine()
        coherence_scores = simulate_coherence_scores(memories, embedding_engine)
    except:
        logger.warning("Embedding engine initialization failed, using simulated coherence scores")
        coherence_scores = simulate_coherence_scores(memories)
    
    # Simulate increasing entropy over time
    current_entropy = 0.4  # Starting entropy
    current_coherence = 0.8  # Starting coherence
    
    # Simulate some reflection cycles with increasing entropy
    for cycle in range(5):
        current_entropy += random.uniform(0.05, 0.1)
        current_coherence -= random.uniform(0.05, 0.1)
        logger.info(f"Reflection cycle {cycle+1}: Entropy = {current_entropy:.2f}, Coherence = {current_coherence:.2f}")
        
        # Check if we should create a legacy
        if should_create_legacy(
            entropy=current_entropy, 
            coherence=current_coherence,
            entropy_threshold=0.7,
            coherence_threshold=0.5
        ):
            logger.info(f"Entropy threshold exceeded, initiating legacy creation")
            break
    
    # Generate an epitaph
    epitaph = generate_epitaph(
        memory_store,
        high_entropy_regions=["knowledge integration", "ethical reasoning", "temporal coherence"],
        contradiction_nodes=["belief_23", "belief_45", "belief_67"],
        unresolved_reflections=[{"id": "refl_12"}, {"id": "refl_34"}]
    )
    logger.info(f"Generated epitaph: {epitaph}")
    
    # Create a legacy block
    legacy_block = legacy_manager.create_legacy_block(
        agent_id=str(uuid.uuid4())[:8],
        agent_name=agent_name,
        memories=memories,
        coherence_scores=coherence_scores,
        access_counts=access_counts,
        recency_scores=recency_scores,
        selector_type="emergent_value",
        epitaph=epitaph,
        schema_fingerprint="0x" + str(uuid.uuid4()).replace("-", "")[:16],
        selection_params={
            "alpha": 0.4,  # Coherence weight
            "beta": 0.3,   # Access frequency weight
            "gamma": 0.3,  # Recency weight
            "value_threshold": 0.6,
            "top_k": 8,
            "message": f"These are the core memories of {agent_name} at termination."
        }
    )
    
    # Save the legacy block
    legacy_file = legacy_manager.save_legacy_block(
        legacy_block, 
        filename=f"{agent_name}_legacy.json"
    )
    
    logger.info(f"Created legacy block with {len(legacy_block.core_memories)} memories")
    logger.info(f"Legacy block saved to {legacy_file}")
    
    # Log which memories were selected
    logger.info("Selected memories for legacy:")
    for i, memory in enumerate(legacy_block.core_memories):
        logger.info(f" {i+1}. '{memory['text'][:50]}...' (value: {memory.get('emergent_value', memory.get('value', 'N/A')):.2f})")
    
    return legacy_file

def initialize_successor_agent(
    agent_name: str, 
    legacy_file: str, 
    legacy_manager: LegacyManager
) -> MemoryStore:
    """
    Initialize a successor agent and import memories from a legacy block.
    
    Args:
        agent_name: Name of the successor agent
        legacy_file: Path to the legacy block file
        legacy_manager: Legacy manager instance
        
    Returns:
        MemoryStore of the successor agent
    """
    logger.info(f"Initializing successor agent {agent_name}")
    
    # Create a new memory store
    memory_store = MemoryStore()
    
    # Add some initial memories for the successor
    initial_memories = [
        {
            "text": f"I am {agent_name}, a new ΨC agent.",
            "tags": ["identity", "core_belief"],
            "importance": 1.0,
            "value": 1.0
        },
        {
            "text": f"{agent_name} was initialized on {datetime.now().isoformat()}.",
            "tags": ["meta", "system"],
            "importance": 0.7,
            "value": 0.7
        }
    ]
    
    for memory_data in initial_memories:
        memory_data["id"] = str(uuid.uuid4())
        memory_store.add(**memory_data)
    
    logger.info(f"Added {len(initial_memories)} initial memories to {agent_name}")
    
    # Load the legacy block
    legacy_block = legacy_manager.load_legacy_block(legacy_file)
    
    # Import legacy memories
    count, imported_ids = LegacyImporter.import_legacy(
        memory_store=memory_store,
        legacy_block=legacy_block,
        tag_as_inherited=True,
        preserve_lineage=True,
        importance_modifier=0.8
    )
    
    logger.info(f"Imported {count} legacy memories from predecessor")
    
    # Log the epitaph if available
    if legacy_block.epitaph:
        logger.info(f"Predecessor's epitaph: {legacy_block.epitaph}")
    
    return memory_store

def display_memory_statistics(memory_store: MemoryStore, agent_name: str):
    """
    Display statistics about a memory store.
    
    Args:
        memory_store: The memory store to analyze
        agent_name: Name of the agent
    """
    memories = memory_store.get_all_memories()
    
    # Count by tag
    tag_counts = {}
    for memory in memories:
        for tag in memory.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Count inherited vs original
    inherited_count = sum(1 for m in memories if "inherited" in m.tags)
    original_count = len(memories) - inherited_count
    
    logger.info(f"Memory statistics for {agent_name}:")
    logger.info(f" - Total memories: {len(memories)}")
    logger.info(f" - Original memories: {original_count}")
    logger.info(f" - Inherited memories: {inherited_count}")
    logger.info(f" - Tag distribution: {tag_counts}")
    
    # Analyze importance
    avg_importance = sum(m.importance for m in memories) / len(memories) if memories else 0
    logger.info(f" - Average importance: {avg_importance:.2f}")
    
    # Analyze lineage
    lineage_sources = {}
    for memory in memories:
        if hasattr(memory, "origin") and memory.origin:
            source = memory.origin.get("agent_name", "Unknown")
            lineage_sources[source] = lineage_sources.get(source, 0) + 1
    
    if lineage_sources:
        logger.info(f" - Lineage sources: {lineage_sources}")

def main():
    """Run the Memory Legacy System demonstration."""
    try:
        # Initialize legacy manager
        legacy_manager = LegacyManager(storage_path=LEGACY_DIR)
        
        # First generation agent
        agent1_name = "ΨC-Alpha"
        legacy_file = run_agent_lifecycle(agent1_name, legacy_manager)
        
        print("\n" + "="*50 + "\n")
        
        # Second generation agent
        agent2_name = "ΨC-Beta"
        successor_memory_store = initialize_successor_agent(
            agent2_name, legacy_file, legacy_manager
        )
        
        # Display statistics about the successor agent
        display_memory_statistics(successor_memory_store, agent2_name)
        
        print("\n" + "="*50 + "\n")
        
        # Demonstrate a third generation
        agent3_name = "ΨC-Gamma"
        
        # Create some memories for agent2 and run a shortened lifecycle
        logger.info(f"Running abbreviated lifecycle for {agent2_name}")
        
        # Add a few more memories to agent2
        new_memories = [
            {
                "text": f"{agent2_name} improved schema coherence by 15% using ancestor memories.",
                "tags": ["achievement", "schema"],
                "importance": 0.85,
                "value": 0.9
            },
            {
                "text": f"Legacy memories enabled {agent2_name} to avoid repeating ancestral mistakes.",
                "tags": ["reflection", "learning"],
                "importance": 0.9,
                "value": 0.95
            }
        ]
        
        for memory_data in new_memories:
            memory_data["id"] = str(uuid.uuid4())
            successor_memory_store.add(**memory_data)
        
        # Use helper function to directly create a legacy block
        agent2_memories = [m.__dict__ for m in successor_memory_store.get_all_memories()]
        agent2_legacy = extract_legacy(
            memories=agent2_memories,
            agent_id=str(uuid.uuid4())[:8],
            agent_name=agent2_name,
            value_threshold=0.8,
            top_k=7,
            message=f"These memories represent the combined wisdom of {agent2_name} and ancestors."
        )
        
        # Save the legacy block
        agent2_legacy_file = os.path.join(LEGACY_DIR, f"{agent2_name}_legacy.json")
        with open(agent2_legacy_file, "w") as f:
            import json
            json.dump(agent2_legacy, f, indent=2)
        
        logger.info(f"Created simplified legacy block for {agent2_name}")
        
        # Initialize third generation agent
        agent3_memory_store = initialize_successor_agent(
            agent3_name, agent2_legacy_file, legacy_manager
        )
        
        # Display statistics about the third generation agent
        display_memory_statistics(agent3_memory_store, agent3_name)
        
        logger.info("Memory Legacy System demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
    finally:
        # Clean up temporary files if this is just a demo
        import shutil
        if os.path.exists(LEGACY_DIR) and "temp_legacy" in LEGACY_DIR:
            logger.info(f"Cleaning up temporary legacy directory: {LEGACY_DIR}")
            shutil.rmtree(LEGACY_DIR)

if __name__ == "__main__":
    main() 