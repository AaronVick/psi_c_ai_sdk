#!/usr/bin/env python3
"""
Schema Mathematics Formulations Module

This module implements the advanced mathematical formulations from the ΨC-AI SDK
mathematical foundations that pertain to schema operations, cognitive processes,
and memory structure evaluation.

These implementations directly align with the formulations in the UnderlyingMath.md
reference document.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set, Union


def reflective_control_score(memory_importance: float, 
                             coherence: float, 
                             entropy: float, 
                             reflection_count: int,
                             global_coherence: float,
                             global_coherence_delta: float) -> Dict[str, float]:
    """
    Calculate the reflective control formula (formula #13) to determine
    the appropriate cognitive action.
    
    Ψ_t = {
        Reflect(M_i)     if ∂ℒ/∂M_i > θ_r
        Consolidate(M_i) if C_i < τ_c ∧ H_i < τ_h ∧ R_i > ρ
        Mutate(Σ)        if ΔC_global < θ_m ∧ dC/dt < 0 ∧ O_meta > ω
        Pause()          if |dΣ/dt| > ε_s
        Persist(M_i)     if I_i > τ_i ∧ A_i < α_a ∧ Φ_i > φ
        Terminate()      if C_global < θ_d ∧ dC/dt < 0 ∧ T_fatigue > τ_f
    }
    
    Args:
        memory_importance: Importance value of memory (I_i)
        coherence: Coherence value of memory (C_i)
        entropy: Entropy value of memory (H_i)
        reflection_count: Number of reflections on this memory (R_i)
        global_coherence: Global coherence of the system (C_global)
        global_coherence_delta: Change in global coherence (dC/dt)
        
    Returns:
        Dictionary with scores for each possible cognitive action
    """
    # Default thresholds (in a real system these would be configured)
    thresholds = {
        'reflect': 0.7,      # θ_r: Reflection threshold
        'consolidate_c': 0.5, # τ_c: Coherence threshold for consolidation
        'consolidate_h': 0.3, # τ_h: Entropy threshold for consolidation
        'consolidate_r': 2,   # ρ: Reflection count threshold for consolidation
        'mutate': 0.2,       # θ_m: Global coherence change threshold for mutation
        'persist_i': 0.6,    # τ_i: Importance threshold for persistence
        'terminate': 0.3     # θ_d: Global coherence threshold for termination
    }
    
    # Calculate scores for each action
    reflect_score = 0.0
    if entropy > thresholds['reflect'] or coherence < 0.5:
        reflect_score = memory_importance * (1 - coherence) * entropy
        
    consolidate_score = 0.0
    if (coherence < thresholds['consolidate_c'] and 
        entropy < thresholds['consolidate_h'] and 
        reflection_count > thresholds['consolidate_r']):
        consolidate_score = memory_importance * (1 - coherence) * (1 - entropy)
        
    mutate_score = 0.0
    if (abs(global_coherence_delta) < thresholds['mutate'] and 
        global_coherence_delta < 0):
        mutate_score = (1 - global_coherence) * abs(global_coherence_delta)
        
    pause_score = 0.0
    if abs(global_coherence_delta) > 0.5:  # ε_s: Schema change rate threshold
        pause_score = abs(global_coherence_delta)
        
    persist_score = 0.0
    if memory_importance > thresholds['persist_i']:
        persist_score = memory_importance * coherence
        
    terminate_score = 0.0
    if (global_coherence < thresholds['terminate'] and 
        global_coherence_delta < 0):
        terminate_score = (1 - global_coherence) * abs(global_coherence_delta)
    
    # Return scores for all possible actions
    return {
        'reflect': reflect_score,
        'consolidate': consolidate_score,
        'mutate': mutate_score,
        'pause': pause_score,
        'persist': persist_score,
        'terminate': terminate_score
    }


def meta_objective_function(schema_coherence: float, 
                           schema_entropy: float, 
                           reflection_cost: float,
                           context_importance: float,
                           schema_change_rate: float) -> Dict[str, float]:
    """
    Calculate the meta-objective function (formula #14) to determine
    the optimal cognitive behavior.
    
    Ψ_t^* = argmin_{ψ ∈ {Reflect, Consolidate, ...}} ℱ_ψ(t)
    Where:
    ℱ_ψ(t) = 𝔼[ℒ_future | ψ_t] + λ_ψ · ℂ_ψ
    
    Args:
        schema_coherence: Current coherence of the schema
        schema_entropy: Current entropy of the schema
        reflection_cost: Cost of reflection operations
        context_importance: Importance of current context
        schema_change_rate: Rate of schema mutation
        
    Returns:
        Dictionary with expected future costs for each cognitive action
    """
    # Lambda factors for different actions
    lambdas = {
        'reflect': 0.8,
        'consolidate': 0.5,
        'mutate': 1.2,
        'pause': 0.3,
        'persist': 0.4,
        'terminate': 2.0
    }
    
    # Calculate contextual costs for each action
    contextual_costs = {
        'reflect': reflection_cost * (1 + schema_entropy),
        'consolidate': reflection_cost * 0.5 * (1 - schema_coherence),
        'mutate': schema_change_rate * context_importance,
        'pause': 0.1 * context_importance,
        'persist': (1 - schema_coherence) * 0.2,
        'terminate': context_importance * 2.0
    }
    
    # Calculate expected future costs
    expected_costs = {}
    for action, lambda_factor in lambdas.items():
        if action == 'reflect':
            expected_future_cost = (1 - schema_coherence) * schema_entropy
        elif action == 'consolidate':
            expected_future_cost = schema_entropy * 0.8
        elif action == 'mutate':
            expected_future_cost = schema_entropy - schema_coherence * 0.5
        elif action == 'pause':
            expected_future_cost = context_importance * (1 - schema_coherence)
        elif action == 'persist':
            expected_future_cost = 0.1 * schema_entropy
        elif action == 'terminate':
            expected_future_cost = context_importance * 0.5
        
        # Calculate total cost with regularization
        total_cost = expected_future_cost + lambda_factor * contextual_costs[action]
        expected_costs[action] = total_cost
    
    return expected_costs


def cognitive_debt(reflection_count: int, 
                  schema_changes: int, 
                  coherence: float, 
                  lambda_factor: float = 0.5) -> float:
    """
    Calculate cognitive debt (formula #24) to measure
    the accumulation of unprocessed cognitive work.
    
    D_t = R_t - λ · (S_t + C_t)
    
    Args:
        reflection_count: Total reflections in recent cycles (R_t)
        schema_changes: Total schema changes (S_t)
        coherence: Average coherence (C_t)
        lambda_factor: Weight factor for schema changes and coherence
        
    Returns:
        Cognitive debt score
    """
    return reflection_count - lambda_factor * (schema_changes + coherence)


def reflective_confidence_heuristic(coherence: float, 
                                   entropy: float, 
                                   trust_level: float) -> float:
    """
    Calculate reflective confidence heuristic (formula #39) to
    determine confidence in a memory or belief.
    
    R_i = C_i · (1 - H_i) · T_k
    
    Args:
        coherence: Coherence value of memory (C_i)
        entropy: Entropy value of memory (H_i)
        trust_level: Trust level of source (T_k)
        
    Returns:
        Reflective confidence score
    """
    return coherence * (1 - entropy) * trust_level


def complexity_budget_function(active_memories: int, 
                              schema_depth: float, 
                              contradiction_count: int) -> float:
    """
    Calculate complexity budget (formula #18) to manage
    computational resources.
    
    C_t = α_1 · M_t + α_2 · S_t + α_3 · D_t
    
    Args:
        active_memories: Number of active memories (M_t)
        schema_depth: Average schema depth (S_t)
        contradiction_count: Number of recent contradictions (D_t)
        
    Returns:
        Complexity budget score
    """
    # Default alpha weights
    alpha1 = 0.4  # Weight for active memories
    alpha2 = 0.3  # Weight for schema depth
    alpha3 = 0.3  # Weight for contradictions
    
    return alpha1 * active_memories + alpha2 * schema_depth + alpha3 * contradiction_count


def memory_energy_budget(total_budget: float, 
                         feature_costs: Dict[str, float]) -> float:
    """
    Calculate memory energy budget (formula #20) to manage
    computational resources.
    
    E_available = B - ∑_i E_i
    
    Args:
        total_budget: Total compute/memory budget (B)
        feature_costs: Dictionary of feature costs {feature: cost}
        
    Returns:
        Available energy
    """
    used_energy = sum(feature_costs.values())
    return total_budget - used_energy


def schema_health_index(avg_coherence: float, 
                       avg_entropy: float, 
                       alignment_score: float, 
                       schema_drift_rate: float) -> float:
    """
    Calculate system health index (Ψ-index) (formula #44) to
    assess overall system health.
    
    Ψ_index = ω_1 · C̄ - ω_2 · H + ω_3 · A - ω_4 · D
    
    Args:
        avg_coherence: Average coherence (C̄)
        avg_entropy: Average entropy (H)
        alignment_score: Alignment score (A)
        schema_drift_rate: Rate of schema drift (D)
        
    Returns:
        System health index
    """
    # Default omega weights
    omega1 = 0.4  # Weight for coherence
    omega2 = 0.3  # Weight for entropy
    omega3 = 0.2  # Weight for alignment
    omega4 = 0.1  # Weight for drift
    
    return (omega1 * avg_coherence - 
            omega2 * avg_entropy + 
            omega3 * alignment_score - 
            omega4 * schema_drift_rate)


def calculate_memory_embedding_entropy(embedding: np.ndarray) -> float:
    """
    Calculate the entropy of a memory embedding (formula #10).
    
    H(e) = -∑_{i=1}^d p_i log p_i where p_i = |e_i|/∑|e|
    
    Args:
        embedding: Memory embedding vector
        
    Returns:
        Entropy of the embedding
    """
    # Normalize the embedding
    abs_embedding = np.abs(embedding)
    sum_abs = np.sum(abs_embedding)
    
    # Avoid division by zero
    if sum_abs == 0:
        return 0.0
    
    p = abs_embedding / sum_abs
    
    # Calculate entropy (using only non-zero probabilities)
    non_zeros = p > 0
    if not np.any(non_zeros):
        return 0.0
    
    entropy = -np.sum(p[non_zeros] * np.log(p[non_zeros]))
    return entropy


def calculate_graph_connectivity(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calculate connectivity metrics for a schema graph.
    
    Args:
        graph: NetworkX graph representing schema
        
    Returns:
        Dictionary with connectivity metrics
    """
    if len(graph) == 0:
        return {
            "average_degree": 0,
            "clustering_coefficient": 0,
            "connected_components": 0,
            "density": 0
        }
    
    # Calculate average degree
    degrees = [d for _, d in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    # Calculate clustering coefficient
    try:
        clustering = nx.average_clustering(graph)
    except:
        clustering = 0
    
    # Calculate number of connected components
    connected_components = nx.number_connected_components(graph)
    
    # Calculate graph density
    density = nx.density(graph)
    
    return {
        "average_degree": avg_degree,
        "clustering_coefficient": clustering,
        "connected_components": connected_components,
        "density": density
    }


def identity_drift_measure(anchored_beliefs: Dict[str, float]) -> float:
    """
    Calculate identity drift measure (formula #36) to track
    stability of identity.
    
    ℐ_drift = (1/|A|) · ∑_{i ∈ A} H_i
    
    Args:
        anchored_beliefs: Dictionary of {belief_id: entropy}
        
    Returns:
        Identity drift measure
    """
    if not anchored_beliefs:
        return 0.0
    
    return sum(anchored_beliefs.values()) / len(anchored_beliefs)


def schema_annealing_temperature(initial_temp: float, 
                                cooling_rate: float, 
                                iterations: int) -> float:
    """
    Calculate schema annealing temperature (formula #25) for
    schema optimization.
    
    T_t = T_0 · e^(-βt)
    
    Args:
        initial_temp: Initial temperature (T_0)
        cooling_rate: Cooling rate (β)
        iterations: Number of schema iterations (t)
        
    Returns:
        Current temperature
    """
    return initial_temp * np.exp(-cooling_rate * iterations) 


# ------------------- ENHANCED MATHEMATICAL FORMULATIONS -------------------

def calculate_coherence_accumulation(current_coherence: float,
                                     entropy: float,
                                     alpha: float = 0.1,
                                     max_coherence: float = 1.0,
                                     beta: float = 0.05,
                                     dt: float = 1.0) -> float:
    """
    Calculate coherence accumulation rate using the logistic growth model
    modified by entropy (formula #1).
    
    dC(t)/dt = α*C(t)*(1 - C(t)/K) - β*H(M(t))
    
    Args:
        current_coherence: Current coherence value C(t)
        entropy: Current entropy value H(M(t))
        alpha: Coherence growth rate (α)
        max_coherence: Maximum coherence (carrying capacity K)
        beta: Entropy influence factor (β)
        dt: Time step size for integration
        
    Returns:
        New coherence value after one time step
    """
    # Calculate the rate of change
    logistic_growth = alpha * current_coherence * (1 - current_coherence / max_coherence)
    entropy_influence = beta * entropy
    
    # Calculate change in coherence
    coherence_change = (logistic_growth - entropy_influence) * dt
    
    # Update coherence value
    new_coherence = current_coherence + coherence_change
    
    # Ensure coherence stays within bounds [0, max_coherence]
    new_coherence = max(0.0, min(new_coherence, max_coherence))
    
    return new_coherence


def calculate_shannon_entropy(probability_distribution: List[float]) -> float:
    """
    Calculate Shannon entropy for a given probability distribution (formula #2).
    
    H(M(t)) = -∑_{i=1}^{N} p(m_i) log(p(m_i))
    
    Args:
        probability_distribution: List of probabilities that sum to 1
        
    Returns:
        Shannon entropy value
    """
    # Avoid log(0) by filtering out zero probabilities
    non_zero_probs = [p for p in probability_distribution if p > 0]
    
    if not non_zero_probs:
        return 0.0
    
    # Calculate entropy using Shannon formula
    entropy = -sum(p * np.log2(p) for p in non_zero_probs)
    
    return entropy


def calculate_memory_probability_distribution(memories: Dict[str, Any],
                                             attribute: str = 'importance') -> List[float]:
    """
    Calculate a probability distribution over memories based on a specific attribute.
    
    Args:
        memories: Dictionary of memory objects with attributes
        attribute: The attribute to use for calculating probabilities
        
    Returns:
        Normalized probability distribution (sums to 1)
    """
    # Extract attribute values
    values = []
    for memory_id, memory in memories.items():
        if hasattr(memory, attribute):
            values.append(getattr(memory, attribute))
        elif isinstance(memory, dict) and attribute in memory:
            values.append(memory[attribute])
    
    if not values:
        return []
    
    # Convert to probabilities by normalizing
    total = sum(values)
    if total == 0:
        # If sum is zero, use uniform distribution
        return [1.0/len(values)] * len(values)
    
    probabilities = [value/total for value in values]
    return probabilities


def calculate_phase_transition_threshold(entropy_history: List[float],
                                        lambda_theta: float = 1.0) -> float:
    """
    Calculate the phase transition threshold based on entropy 
    expectation and variance (formula #3).
    
    θ = E[H(M(t))] + λ_θ * √Var(H(M(t)))
    
    Args:
        entropy_history: List of entropy values over time
        lambda_theta: Scaling factor for variance influence
        
    Returns:
        Phase transition threshold value
    """
    if not entropy_history:
        return 0.5  # Default threshold if no history available
    
    # Calculate mean and variance of entropy
    mean_entropy = np.mean(entropy_history)
    variance_entropy = np.var(entropy_history)
    
    # Calculate threshold
    threshold = mean_entropy + lambda_theta * np.sqrt(variance_entropy)
    
    return threshold


def calculate_memory_similarity_weight(embedding1: np.ndarray,
                                      embedding2: np.ndarray,
                                      co_reflection_frequency: float = 1.0,
                                      time_difference: float = 0,
                                      alpha: float = 0.1) -> float:
    """
    Calculate the similarity weight between two memory states (formula #4).
    
    w_{ij} = cos(z(m_i), z(m_j)) * f_{ij} / (1 + α|t_i - t_j|)
    
    Args:
        embedding1: Embedding vector for first memory
        embedding2: Embedding vector for second memory
        co_reflection_frequency: Frequency of co-reflection (f_{ij})
        time_difference: Absolute time difference between memories |t_i - t_j|
        alpha: Decay factor for temporal distance
        
    Returns:
        Similarity weight between the two memories
    """
    # Calculate cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm1 * norm2)
    
    # Calculate temporal decay factor
    temporal_factor = 1.0 / (1.0 + alpha * time_difference)
    
    # Calculate final weight
    weight = cosine_similarity * co_reflection_frequency * temporal_factor
    
    return weight


def calculate_self_loss(coherence_values: List[float],
                       optimal_coherence: float = 1.0) -> float:
    """
    Calculate the self-loss function that quantifies deviation 
    from optimal coherence (formula #5).
    
    L_self = ∑_i(C_i(t) - C_opt)²
    
    Args:
        coherence_values: List of coherence values for memories
        optimal_coherence: Optimal coherence value to aim for
        
    Returns:
        Self-loss value
    """
    # Sum of squared differences from optimal
    loss = sum((c - optimal_coherence)**2 for c in coherence_values)
    
    return loss


def calculate_coherence_kernel(edge_weights: Dict[Tuple[str, str], float],
                              frequencies: Dict[Tuple[str, str], float]) -> float:
    """
    Calculate the coherence kernel - overall system coherence (formula #6).
    
    R(S_t) = (1/|E_t|) * ∑_{(i,j)∈E_t} w_{ij} * f_{ij}
    
    Args:
        edge_weights: Dictionary of edge weights {(node_i, node_j): weight}
        frequencies: Dictionary of co-reflection frequencies {(node_i, node_j): freq}
        
    Returns:
        Overall system coherence value
    """
    if not edge_weights:
        return 0.0
    
    # Calculate weighted sum
    weighted_sum = 0.0
    for edge, weight in edge_weights.items():
        frequency = frequencies.get(edge, 1.0)
        weighted_sum += weight * frequency
    
    # Normalize by number of edges
    coherence = weighted_sum / len(edge_weights)
    
    return coherence


def gradient_descent_step(coherence_values: List[float],
                         self_loss_gradient: List[float],
                         learning_rate: float = 0.01) -> List[float]:
    """
    Perform one step of gradient descent to optimize coherence (formula #7).
    
    dC_i(t)/dt = -∇L_self(C_i(t))
    
    Args:
        coherence_values: Current coherence values for memories
        self_loss_gradient: Gradient of self-loss function
        learning_rate: Step size for gradient descent
        
    Returns:
        Updated coherence values after one gradient descent step
    """
    # Update coherence values using gradient descent
    updated_values = []
    for c_i, grad_i in zip(coherence_values, self_loss_gradient):
        # Update using gradient descent
        c_new = c_i - learning_rate * grad_i
        # Ensure value stays in [0, 1] range
        c_new = max(0.0, min(1.0, c_new))
        updated_values.append(c_new)
    
    return updated_values


def calculate_multi_agent_coherence(agent_coherences: Dict[str, float],
                                   agent_weights: Dict[str, float]) -> float:
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


def add_gaussian_noise(gradient: np.ndarray,
                      mean: float = 0.0,
                      std_dev: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to a gradient for robustness (formula #9).
    
    ∇L_self = μ + ξ, ξ ~ N(0, σ²I)
    
    Args:
        gradient: Original gradient vector
        mean: Mean of Gaussian noise (typically 0)
        std_dev: Standard deviation of Gaussian noise
        
    Returns:
        Noisy gradient vector
    """
    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, size=gradient.shape)
    
    # Add noise to gradient
    noisy_gradient = gradient + noise
    
    return noisy_gradient 