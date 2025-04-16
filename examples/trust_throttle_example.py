#!/usr/bin/env python3
"""
Epistemic Trust Throttler Example

This example demonstrates how to use the TrustThrottler to dynamically
adjust trust in information sources based on their persuasion patterns.
The example shows:

1. How to initialize and configure the TrustThrottler
2. Recording claims from different sources
3. Detecting sources with high persuasion entropy
4. Analyzing cross-source entropy for contested claims
"""

import time
import logging
import random
from typing import Dict, List, Any, Optional

from psi_c_ai_sdk.cognition.trust_throttle import TrustThrottler, SourceTrustProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trust_example")


def simulate_balanced_source(trust_throttler: TrustThrottler, source_id: str, claim_count: int = 10) -> None:
    """
    Simulate a balanced source making a variety of claims.
    
    Args:
        trust_throttler: The trust throttler instance
        source_id: Identifier for the source
        claim_count: Number of claims to simulate
    """
    logger.info(f"Simulating balanced source: {source_id}")
    
    # Generate diverse claims
    claims = [f"claim_{i}" for i in range(claim_count)]
    
    # Record each claim
    for claim_id in claims:
        result = trust_throttler.record_claim(source_id, claim_id)
        logger.info(f"Source {source_id} made claim: {claim_id}")
        
    # Get final trust score
    trust_score = trust_throttler.get_trust(source_id)
    logger.info(f"Final trust score for {source_id}: {trust_score:.4f}")
    
    # Get persuasion entropy
    profile = trust_throttler.get_source_profile(source_id)
    logger.info(f"Persuasion entropy for {source_id}: {profile['persuasion_entropy']:.4f}")


def simulate_repetitive_source(trust_throttler: TrustThrottler, source_id: str, claim_count: int = 10) -> None:
    """
    Simulate a source that repeatedly makes the same few claims.
    
    Args:
        trust_throttler: The trust throttler instance
        source_id: Identifier for the source
        claim_count: Number of claims to simulate
    """
    logger.info(f"Simulating repetitive source: {source_id}")
    
    # Generate limited set of claims that will be repeated
    claims = [f"claim_{i}" for i in range(3)]  # Only 3 unique claims
    
    # Record each claim (with repetition)
    for _ in range(claim_count):
        claim_id = random.choice(claims)  # Randomly choose from the limited set
        result = trust_throttler.record_claim(source_id, claim_id)
        logger.info(f"Source {source_id} made claim: {claim_id}")
        
    # Get final trust score
    trust_score = trust_throttler.get_trust(source_id)
    logger.info(f"Final trust score for {source_id}: {trust_score:.4f}")
    
    # Get persuasion entropy
    profile = trust_throttler.get_source_profile(source_id)
    logger.info(f"Persuasion entropy for {source_id}: {profile['persuasion_entropy']:.4f}")


def simulate_contested_claim(trust_throttler: TrustThrottler, claim_id: str, source_count: int = 5) -> None:
    """
    Simulate multiple sources making the same claim.
    
    Args:
        trust_throttler: The trust throttler instance
        claim_id: The contested claim ID
        source_count: Number of sources to simulate
    """
    logger.info(f"Simulating contested claim: {claim_id}")
    
    # Multiple sources make the same claim
    for i in range(source_count):
        source_id = f"contested_source_{i}"
        result = trust_throttler.record_claim(source_id, claim_id)
        logger.info(f"Source {source_id} made contested claim: {claim_id}")
    
    # Calculate cross-source entropy
    cross_entropy = trust_throttler.calculate_cross_source_entropy()
    if claim_id in cross_entropy:
        logger.info(f"Cross-source entropy for {claim_id}: {cross_entropy[claim_id]:.4f}")
        logger.info(f"This claim is {'highly' if cross_entropy[claim_id] > 0.7 else 'somewhat'} contested")


def main() -> None:
    """Run the trust throttler example."""
    logger.info("Starting Trust Throttler Example")
    
    # Initialize the trust throttler
    trust_throttler = TrustThrottler(
        learning_rate=0.1,
        time_window=3600.0,
        trust_floor=0.2,
        trust_ceiling=1.0,
        default_initial_trust=0.7
    )
    
    logger.info("Trust Throttler initialized")
    
    # Simulate balanced sources
    simulate_balanced_source(trust_throttler, "reliable_source_1", 10)
    simulate_balanced_source(trust_throttler, "reliable_source_2", 8)
    
    # Simulate repetitive sources (potential manipulation)
    simulate_repetitive_source(trust_throttler, "repetitive_source_1", 10)
    simulate_repetitive_source(trust_throttler, "repetitive_source_2", 15)
    
    # Get sources with high persuasion entropy
    high_persuasion_sources = trust_throttler.get_high_persuasion_sources(threshold=0.6)
    
    logger.info("\nSources with high persuasion entropy (potential manipulation):")
    for source in high_persuasion_sources:
        logger.info(f"  - {source['source_id']}: entropy={source['persuasion_entropy']:.4f}, trust={source['current_trust']:.4f}")
    
    # Simulate contested claims
    logger.info("\nSimulating contested claims:")
    simulate_contested_claim(trust_throttler, "contested_claim_1", 5)
    simulate_contested_claim(trust_throttler, "contested_claim_2", 3)
    
    logger.info("\nTrust Throttler Example completed")


if __name__ == "__main__":
    main() 