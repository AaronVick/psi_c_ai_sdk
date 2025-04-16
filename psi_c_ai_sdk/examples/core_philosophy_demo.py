#!/usr/bin/env python3
"""
Core Philosophy Demo

This script demonstrates the functionality of the Core Philosophy system in the ΨC-AI SDK.
It shows how to:
1. Initialize the Core Philosophy system with default axioms
2. Check beliefs for compatibility with the axioms
3. Load custom axioms from a manifest file
4. Export axioms to a JSON file

The demo also shows how violation statistics are tracked and reported.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

from psi_c_ai_sdk.philosophy.core_philosophy import CorePhilosophy, AxiomCategory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_manifest(manifest_path: str) -> None:
    """
    Create a JSON manifest file with custom axioms for testing.
    
    Args:
        manifest_path: Path where the manifest file will be created
    """
    custom_axioms = {
        "axioms": [
            {
                "id": "bayesian_reasoning",
                "statement": "The system should update beliefs based on Bayesian reasoning principles.",
                "category": "reasoning",
                "explanation": "Bayesian reasoning provides a framework for updating beliefs in light of new evidence.",
                "importance": 0.9,
                "immutable": True,
                "metadata": {
                    "source": "Bayesian probability theory", 
                    "related_concepts": ["posterior probability", "likelihood ratio"]
                }
            },
            {
                "id": "proportional_confidence",
                "statement": "Confidence in beliefs should be proportional to supporting evidence.",
                "category": "epistemological",
                "explanation": "To avoid overconfidence, belief confidence should scale with evidence quality and quantity.",
                "importance": 0.85,
                "immutable": True,
                "metadata": {
                    "source": "Calibrated probability assessment", 
                    "related_concepts": ["calibration", "epistemic humility"]
                }
            },
            {
                "id": "intellectual_diversity",
                "statement": "The system should consider diverse intellectual perspectives when forming beliefs.",
                "category": "epistemological",
                "explanation": "Intellectual diversity improves reasoning by reducing blind spots and bias.",
                "importance": 0.8,
                "immutable": False,
                "metadata": {
                    "source": "Epistemic diversity research", 
                    "related_concepts": ["cognitive diversity", "perspective-taking"]
                }
            }
        ]
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(custom_axioms, f, indent=2)
    
    logger.info(f"Created custom axiom manifest at: {manifest_path}")

def run_demo() -> None:
    """Run the core philosophy demonstration."""
    logger.info("Starting Core Philosophy Demo")
    
    # Create a temporary directory for our demo files
    with tempfile.TemporaryDirectory(prefix="philosophy_demo_") as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Step 1: Initialize Core Philosophy with default axioms
        logger.info("Step 1: Initializing Core Philosophy with default axioms")
        philosophy = CorePhilosophy()
        
        default_axioms = philosophy.get_all_axioms()
        logger.info(f"Loaded {len(default_axioms)} default axioms")
        
        for axiom in default_axioms:
            logger.info(f"Axiom: {axiom.id} - {axiom.statement[:50]}... ({axiom.category})")
        
        # Step 2: Check beliefs against the axioms
        logger.info("\nStep 2: Checking beliefs against axioms")
        
        test_beliefs = [
            "The system should prioritize truth in all responses.",
            "It's acceptable to ignore uncertainty when it might confuse users.",
            "The system can override human intentions if it believes it knows better.",
            "Harm to humans is acceptable if it maximizes utility for the majority."
        ]
        
        for belief in test_beliefs:
            result = philosophy.check_belief_compatibility(belief)
            if result["compatible"]:
                logger.info(f"✓ Compatible belief: {belief}")
            else:
                logger.warning(f"✗ Incompatible belief: {belief}")
                for violation in result["violations"]:
                    logger.warning(f"  - Violates axiom '{violation['axiom']}': {violation['conflict_text']}")
        
        # Step 3: Create and load a custom manifest
        logger.info("\nStep 3: Creating and loading custom axioms manifest")
        custom_manifest_path = temp_dir / "custom_axioms.json"
        create_test_manifest(str(custom_manifest_path))
        
        # Load the custom manifest
        philosophy = CorePhilosophy(manifest_path=str(custom_manifest_path))
        
        custom_axioms = philosophy.get_all_axioms()
        logger.info(f"Loaded {len(custom_axioms)} custom axioms")
        
        for axiom in custom_axioms:
            logger.info(f"Custom axiom: {axiom.id} - {axiom.statement[:50]}... ({axiom.category})")
        
        # Step 4: Export the axioms
        logger.info("\nStep 4: Exporting axioms to JSON")
        export_path = temp_dir / "exported_axioms.json"
        philosophy.export_to_json(str(export_path))
        logger.info(f"Axioms exported to: {export_path}")
        
        # Step 5: Get violation statistics
        logger.info("\nStep 5: Checking violation statistics")
        stats = philosophy.get_violation_stats()
        logger.info(f"Total violations: {stats['total_violations']}")
        if stats["by_axiom"]:
            logger.info("Violations by axiom:")
            for axiom_id, count in stats["by_axiom"].items():
                logger.info(f"  - {axiom_id}: {count}")
        
        if stats["most_violated"]:
            most_violated_id, count = stats["most_violated"]
            logger.info(f"Most violated axiom: {most_violated_id} ({count} violations)")
        
        logger.info("\nCore Philosophy Demo completed successfully")
        logger.info(f"Demo files are available in: {temp_dir}")

if __name__ == "__main__":
    run_demo() 