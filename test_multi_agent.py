#!/usr/bin/env python3

"""
Simple test script for multi-agent coherence function
"""

def calculate_multi_agent_coherence(agent_coherences, agent_weights):
    """
    Calculate the overall coherence in a multi-agent system (formula #8).
    
    R(S_t) = ∑_{i=1}^{N} w_i * C_i(t)
    
    Args:
        agent_coherences: Dictionary of {agent_id: coherence_value}
        agent_weights: Dictionary of {agent_id: weight}
        
    Returns:
        Overall multi-agent system coherence
    """
    if not agent_coherences:
        return 0.0
    
    # Simple, consistent approach that handles all cases correctly
    numerator = 0.0
    denominator = 0.0
    
    for agent_id, coherence in agent_coherences.items():
        weight = agent_weights.get(agent_id, 1.0)  # Default to 1.0 if not in weights
        numerator += weight * coherence
        denominator += weight
    
    # Return weighted average or 0 if denominator is 0
    return numerator / denominator if denominator > 0 else 0.0

def main():
    # Test case from the test file
    agent_coherences = {
        'agent1': 0.8,
        'agent2': 0.6,
        'agent3': 0.7
    }
    subset_weights = {
        'agent1': 2.0,
        'agent2': 0.5
    }

    result = calculate_multi_agent_coherence(agent_coherences, subset_weights)
    expected = (0.8*2.0 + 0.6*0.5 + 0.7*1.0) / (2.0 + 0.5 + 1.0)

    print(f'Result: {result}')
    print(f'Expected: {expected}')
    print(f'Match: {abs(result - expected) < 1e-10}')
    
    # Check intermediate calculations
    numerator = 0.0
    denominator = 0.0
    for agent_id, coherence in agent_coherences.items():
        weight = subset_weights.get(agent_id, 1.0)
        print(f'Agent {agent_id}: coherence={coherence}, weight={weight}')
        numerator += weight * coherence
        denominator += weight
    
    print(f'Numerator: {numerator}')
    print(f'Denominator: {denominator}')
    print(f'Numerator/Denominator = {numerator/denominator}')

if __name__ == "__main__":
    main() 