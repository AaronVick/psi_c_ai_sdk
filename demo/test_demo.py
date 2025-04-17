#!/usr/bin/env python3
"""
ΨC Schema Integration Demo - Test Script
----------------------------------------
A simple test script to verify the functionality
of the ΨC Schema Integration Demo components.
"""

import os
import json
import logging
from pathlib import Path
from demo_runner import DemoRunner
from llm_bridge import LLMBridge, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_test")

# Constants
TEST_MEMORIES = [
    {
        "content": "The sky is blue because of Rayleigh scattering of sunlight.",
        "metadata": {
            "confidence": 0.9,
            "source": "test_script",
            "timestamp": 1700000000
        }
    },
    {
        "content": "Water is composed of hydrogen and oxygen atoms.",
        "metadata": {
            "confidence": 0.95,
            "source": "test_script",
            "timestamp": 1700000100
        }
    },
    {
        "content": "The Earth orbits around the Sun.",
        "metadata": {
            "confidence": 0.99,
            "source": "test_script",
            "timestamp": 1700000200
        }
    },
    {
        "content": "The sky appears red at sunset due to the angle of sunlight through the atmosphere.",
        "metadata": {
            "confidence": 0.85,
            "source": "test_script",
            "timestamp": 1700000300
        }
    }
]


def test_demo_runner():
    """Test the DemoRunner component."""
    logger.info("Testing DemoRunner...")
    
    # Test initialization
    runner = DemoRunner(profile="default")
    logger.info(f"DemoRunner initialized with coherence: {runner.get_current_coherence():.4f}")
    
    # Test adding memories
    for i, memory in enumerate(TEST_MEMORIES):
        content = memory["content"]
        metadata = memory["metadata"]
        
        logger.info(f"Adding memory {i+1}: {content}")
        result = runner.add_memory(content, metadata)
        
        logger.info(f"  - Coherence: {runner.get_current_coherence():.4f}")
        logger.info(f"  - Entropy: {runner.get_current_entropy():.4f}")
        logger.info(f"  - Contradictions: {result['contradictions']}")
        logger.info(f"  - Schema updated: {result['schema_updated']}")
        logger.info(f"  - Phase transition: {result['phase_transition']}")
    
    # Test schema graph data
    graph_data = runner.get_schema_graph_data()
    logger.info(f"Schema graph contains {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    # Test export functionality
    summary = runner.export_session_summary(format="json")
    logger.info(f"Session summary length: {len(summary)} characters")
    
    # Test system reset
    runner.reset_system()
    logger.info(f"System reset. New coherence: {runner.get_current_coherence():.4f}")
    
    return runner


def test_llm_bridge():
    """Test the LLMBridge component."""
    logger.info("Testing LLMBridge...")
    
    # Initialize with disabled LLM
    config = LLMConfig(is_enabled=False)
    bridge = LLMBridge(config=config)
    
    logger.info(f"LLM enabled: {bridge.is_enabled()}")
    
    # Test reflection enhancement
    memory_content = "The sky is blue because of Rayleigh scattering."
    relevant_memories = ["The sky often appears blue on clear days.", "Light is composed of different wavelengths."]
    
    reflection = bridge.enhance_reflection(memory_content, relevant_memories)
    logger.info(f"Reflection result: {reflection}")
    
    # Test change summary
    summary = bridge.generate_change_summary(
        memory_content, 0.75, 0.82, 0.3, 0.25, 0, True, False
    )
    logger.info(f"Change summary: {summary}")
    
    return bridge


def test_preloaded_cases():
    """Test the preloaded demo cases."""
    logger.info("Testing preloaded demo cases...")
    
    for profile in ["healthcare", "legal", "story"]:
        logger.info(f"Testing {profile} profile...")
        
        # Initialize runner with the profile
        runner = DemoRunner(profile=profile)
        
        # Get memory count
        memory_count = len(runner.memory_store.get_all_memories())
        logger.info(f"  - Loaded {memory_count} memories")
        
        # Get latest memories
        memories = runner.get_latest_memories(3)
        for i, mem in enumerate(memories[:3]):
            logger.info(f"  - Memory {i+1}: {mem.get('content', '')}")
        
        # Get coherence and entropy
        logger.info(f"  - Coherence: {runner.get_current_coherence():.4f}")
        logger.info(f"  - Entropy: {runner.get_current_entropy():.4f}")


def run_all_tests():
    """Run all demo tests."""
    logger.info("Starting ΨC Schema Integration Demo tests...")
    
    # Run tests
    runner = test_demo_runner()
    bridge = test_llm_bridge()
    test_preloaded_cases()
    
    logger.info("All tests completed successfully!")
    return runner, bridge


if __name__ == "__main__":
    # Run tests
    runner, bridge = run_all_tests()
    
    # Print final status
    print("\nTest Results Summary:")
    print("=====================")
    print(f"DemoRunner: Success")
    print(f"LLMBridge: Success")
    print(f"Preloaded Cases: Success")
    print("\nDemo is ready to use!")
    print("Run the demo with: streamlit run demo/web_interface_demo.py") 