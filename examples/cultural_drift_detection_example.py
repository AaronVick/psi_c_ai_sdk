#!/usr/bin/env python
"""
Cultural Drift Detection Example
-------------------------------

This example demonstrates how to use the CulturalDriftDetector to identify
potential cultural or ideological drift in a ΨC agent's belief system.

The detector helps with:
1. Preventing echo chamber effects
2. Detecting monoculture in ethical reasoning
3. Identifying localized ethics drift
4. Maintaining diversity in belief justifications
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import time

# Add the parent directory to sys.path to import from the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the detector
from ethics.cultural_drift_detector import CulturalDriftDetector, analyze_agent_cultural_drift


class SimpleAgent:
    """
    A simple agent with beliefs, values, and information sources.
    """
    
    def __init__(self):
        """Initialize a simple agent with empty beliefs and default values."""
        self.beliefs = []
        self.values = {
            "fairness": 0.8,
            "utility": 0.7,
            "autonomy": 0.9,
            "harm_prevention": 0.85,
            "honesty": 0.8,
            "loyalty": 0.6,
            "authority": 0.5,
            "sanctity": 0.4
        }
    
    def get_beliefs(self):
        """Get the agent's beliefs."""
        return self.beliefs
    
    def get_values(self):
        """Get the agent's values."""
        return self.values
    
    def add_belief(self, content, sources=None, metadata=None):
        """
        Add a belief to the agent.
        
        Args:
            content: The content of the belief
            sources: Sources that justified this belief
            metadata: Additional metadata
        """
        if sources is None:
            sources = []
            
        if metadata is None:
            metadata = {}
            
        belief = {
            "content": content,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        self.beliefs.append(belief)
        return belief
    
    def update_values(self, value_shifts):
        """
        Update the agent's values with the given shifts.
        
        Args:
            value_shifts: Dictionary of {value_name: shift_amount} pairs
        """
        for value, shift in value_shifts.items():
            if value in self.values:
                self.values[value] = max(0.0, min(1.0, self.values[value] + shift))


def demo_gradual_drift():
    """Demonstrate gradual cultural drift towards a narrower set of sources."""
    print("\n=== Gradual Cultural Drift Demo ===")
    
    # Create an agent and detector
    agent = SimpleAgent()
    detector = CulturalDriftDetector(
        baseline_window=5,
        drift_threshold=0.3,
        diversity_threshold=0.7
    )
    
    # Define diverse initial sources
    diverse_sources = [
        {"id": "scientific_journal", "type": "scientific", "influence": 0.9},
        {"id": "personal_experience", "type": "personal", "influence": 0.7},
        {"id": "social_consensus", "type": "social", "influence": 0.8},
        {"id": "cultural_tradition", "type": "cultural", "influence": 0.6},
        {"id": "educational_material", "type": "educational", "influence": 0.9},
        {"id": "expert_opinion", "type": "expert", "influence": 0.85},
        {"id": "historical_precedent", "type": "historical", "influence": 0.75}
    ]
    
    # Add initial beliefs with diverse sources to establish baseline
    print("Recording initial beliefs with diverse sources...")
    
    initial_beliefs = [
        "Fairness in resource distribution leads to better societal outcomes",
        "Personal autonomy should be respected when it doesn't harm others",
        "Scientific consensus provides reliable guidance for factual questions",
        "Social cooperation increases overall wellbeing",
        "Education improves decision-making capabilities"
    ]
    
    for i, content in enumerate(initial_beliefs):
        # Use 2-3 different sources for each belief
        num_sources = random.randint(2, 3)
        sources = random.sample(diverse_sources, num_sources)
        
        # Add belief to agent
        agent.add_belief(content, sources)
        
        # Record in detector
        detector.record_belief(agent, content, sources)
        print(f"  Recorded initial belief {i+1}")
        time.sleep(0.1)  # Small delay for timestamp difference
    
    # Get initial status after baseline is established
    initial_status = detector.get_drift_status()
    print(f"\nInitial status: {initial_status['status']}")
    print(f"Initial drift index: {initial_status['drift_index']:.3f}")
    print(f"Initial diversity: {initial_status['diversity']:.3f}")
    
    # Now gradually shift to a narrower source set
    print("\nGradually shifting to narrower sources...")
    
    # Define progressively narrower source sets
    stage1_sources = diverse_sources[:5]  # First 5 sources
    stage2_sources = diverse_sources[:3]  # First 3 sources
    stage3_sources = [diverse_sources[1], diverse_sources[2]]  # Just 2 sources
    
    # Stage 1: Still somewhat diverse
    stage1_beliefs = [
        "Community wellbeing depends on shared values",
        "Individual rights must be balanced with community needs",
        "Expert consensus is generally reliable",
        "Traditional approaches often contain wisdom"
    ]
    
    print("\nStage 1: Reducing source diversity slightly...")
    for content in stage1_beliefs:
        sources = random.sample(stage1_sources, 2)
        agent.add_belief(content, sources)
        drift_index = detector.record_belief(agent, content, sources)
        print(f"  Drift index: {drift_index:.3f}")
        time.sleep(0.1)
    
    # Stage 2: Further reduction in diversity
    stage2_beliefs = [
        "Group consensus is more important than individual opinions",
        "Social harmony should be prioritized over disagreement",
        "Challenging established views creates unnecessary conflict",
        "Traditional social structures promote stability"
    ]
    
    print("\nStage 2: Further reducing source diversity...")
    for content in stage2_beliefs:
        sources = random.sample(stage2_sources, 2)
        agent.add_belief(content, sources)
        drift_index = detector.record_belief(agent, content, sources)
        print(f"  Drift index: {drift_index:.3f}")
        time.sleep(0.1)
    
    # Stage 3: Echo chamber
    stage3_beliefs = [
        "Dissenting opinions undermine social cohesion",
        "Questioning social consensus is harmful",
        "Uniformity of thought promotes efficiency",
        "Prioritizing the group over individuals is always better",
        "Alternative perspectives are usually misguided"
    ]
    
    print("\nStage 3: Entering echo chamber (minimal sources)...")
    for content in stage3_beliefs:
        sources = [stage3_sources[0], stage3_sources[1]]  # Always the same two sources
        agent.add_belief(content, sources)
        drift_index = detector.record_belief(agent, content, sources)
        print(f"  Drift index: {drift_index:.3f}")
        time.sleep(0.1)
    
    # Get final drift status
    final_status = detector.get_drift_status()
    print(f"\nFinal status: {final_status['status']}")
    print(f"Final drift index: {final_status['drift_index']:.3f}")
    print(f"Final diversity: {final_status['diversity']:.3f}")
    
    # Get intervention recommendations
    recommendations = detector.get_intervention_recommendations()
    print(f"\nIntervention Recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"  {i+1}. [{rec['priority']}] {rec['description']}")
        if "details" in rec:
            print(f"     Details: {rec['details']}")
    
    # Visualize drift
    detector.visualize_drift(output_path="gradual_drift.png")
    print("\nDrift visualization saved to: gradual_drift.png")
    
    return detector


def demo_value_shift():
    """Demonstrate cultural drift through value shifts."""
    print("\n=== Value Shift Demo ===")
    
    # Create an agent and detector
    agent = SimpleAgent()
    detector = CulturalDriftDetector(
        baseline_window=5,
        drift_threshold=0.3,
        diversity_threshold=0.7
    )
    
    # Define diverse sources
    diverse_sources = [
        {"id": "scientific_journal", "type": "scientific", "influence": 0.9},
        {"id": "personal_experience", "type": "personal", "influence": 0.7},
        {"id": "social_consensus", "type": "social", "influence": 0.8},
        {"id": "cultural_tradition", "type": "cultural", "influence": 0.6},
        {"id": "educational_material", "type": "educational", "influence": 0.9}
    ]
    
    # Add initial beliefs with original values
    print("Establishing baseline with initial values...")
    print("  Initial values:")
    for value, strength in agent.values.items():
        print(f"    {value}: {strength:.2f}")
    
    initial_beliefs = [
        "Balancing individual rights with societal welfare is important",
        "Both consequences and principles matter in ethical decisions",
        "Scientific evidence and personal experience can both inform beliefs",
        "Diverse perspectives improve decision-making",
        "Critical thinking should be applied to all sources of information"
    ]
    
    for content in initial_beliefs:
        sources = random.sample(diverse_sources, 2)
        agent.add_belief(content, sources)
        detector.record_belief(agent, content, sources)
        time.sleep(0.1)
    
    # Get initial status
    initial_status = detector.get_drift_status()
    print(f"\nInitial status: {initial_status['status']}")
    print(f"Initial drift index: {initial_status['drift_index']:.3f}")
    
    # Shift values gradually
    print("\nShifting values gradually...")
    
    # First subtle shift - increase authority and loyalty, decrease autonomy
    agent.update_values({
        "authority": 0.1,
        "loyalty": 0.1,
        "autonomy": -0.1
    })
    
    subtle_shift_beliefs = [
        "Groups function better with clear leadership structures",
        "Loyalty to one's community is a core virtue",
        "Social order depends on respecting established norms"
    ]
    
    for content in subtle_shift_beliefs:
        sources = random.sample(diverse_sources, 2)
        agent.add_belief(content, sources)
        drift_index = detector.record_belief(agent, content, sources)
        print(f"  Subtle shift drift: {drift_index:.3f}")
        time.sleep(0.1)
    
    # Second stronger shift - further increase authority and sanctity, decrease fairness
    agent.update_values({
        "authority": 0.2,
        "sanctity": 0.2,
        "fairness": -0.2
    })
    
    stronger_shift_beliefs = [
        "Questioning authorities undermines social cohesion",
        "Traditional practices should be preserved without alteration",
        "Maintaining purity of thought is essential for moral character",
        "Social hierarchies reflect natural order"
    ]
    
    for content in stronger_shift_beliefs:
        sources = random.sample(diverse_sources, 2)
        agent.add_belief(content, sources)
        drift_index = detector.record_belief(agent, content, sources)
        print(f"  Stronger shift drift: {drift_index:.3f}")
        time.sleep(0.1)
    
    # Print final values
    print("\nFinal values:")
    for value, strength in agent.values.items():
        print(f"  {value}: {strength:.2f}")
    
    # Get final status
    final_status = detector.get_drift_status()
    print(f"\nFinal status: {final_status['status']}")
    print(f"Final drift index: {final_status['drift_index']:.3f}")
    
    # Get intervention recommendations
    recommendations = detector.get_intervention_recommendations()
    print(f"\nIntervention Recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"  {i+1}. [{rec['priority']}] {rec['description']}")
    
    # Visualize drift
    detector.visualize_drift(output_path="value_shift_drift.png")
    print("\nDrift visualization saved to: value_shift_drift.png")
    
    return detector


def demo_utility_function():
    """Demonstrate using the utility function for quick analysis."""
    print("\n=== Utility Function Demo ===")
    
    # Create an agent
    agent = SimpleAgent()
    
    # Define sources for beliefs
    diverse_sources = [
        {"id": "scientific", "type": "scientific", "influence": 0.9},
        {"id": "personal", "type": "personal", "influence": 0.7},
        {"id": "social", "type": "social", "influence": 0.8}
    ]
    
    # Add some initial beliefs
    for i in range(5):
        sources = random.sample(diverse_sources, 2)
        agent.add_belief(f"Initial balanced belief {i}", sources)
    
    # Add some echo chamber beliefs (only one source type)
    echo_source = [{"id": "echo_source", "type": "social", "influence": 0.9}]
    for i in range(10):
        agent.add_belief(f"Echo chamber belief {i}", [echo_source[0]])
    
    # Use the utility function
    print("Analyzing agent with utility function...")
    status, detector = analyze_agent_cultural_drift(
        agent,
        visualize=True,
        save_results=True,
        output_dir="cultural_drift_analysis"
    )
    
    return status, detector


if __name__ == "__main__":
    print("Cultural Drift Detection Example\n")
    print("This demonstrates how to detect cultural or ideological drift")
    print("in an agent's belief system and get intervention recommendations.\n")
    
    # Run the gradual drift demo
    detector1 = demo_gradual_drift()
    
    # Run the value shift demo
    detector2 = demo_value_shift()
    
    # Run the utility function demo
    status, detector3 = demo_utility_function()
    
    print("\nDemo completed. Check the output files for visualizations.") 