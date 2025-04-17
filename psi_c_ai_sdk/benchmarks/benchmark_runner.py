"""
Benchmark Runner for ΨC Agent Evaluation

This module implements a benchmark system that runs ΨC agents and baseline agents
through canonical task scenarios and evaluates their performance.
"""

import os
import json
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

from psi_c_ai_sdk.agent import PsiCAgent  
from psi_c_ai_sdk.benchmarks.canonical_scenarios import (
    TaskScenario, 
    get_scenario, 
    list_scenarios,
    CANONICAL_SCENARIOS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    scenario_name: str
    agent_type: str
    metrics: Dict[str, float]
    step_metrics: List[Dict[str, Any]]
    coherence_history: List[float]
    contradiction_count: int
    runtime_seconds: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "agent_type": self.agent_type,
            "metrics": self.metrics,
            "step_metrics": self.step_metrics,
            "coherence_history": self.coherence_history,
            "contradiction_count": self.contradiction_count,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create a BenchmarkResult from a dictionary."""
        return cls(
            scenario_name=data["scenario_name"],
            agent_type=data["agent_type"],
            metrics=data["metrics"],
            step_metrics=data["step_metrics"],
            coherence_history=data["coherence_history"],
            contradiction_count=data["contradiction_count"],
            runtime_seconds=data["runtime_seconds"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class BaselineAgent:
    """Simple baseline agent without coherence tracking or contradiction resolution."""
    
    def __init__(self, name: str = "baseline"):
        self.name = name
        self.knowledge = []
        self.confidence_threshold = 0.6
        
    def respond(self, instruction: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and generate a response without sophisticated reasoning.
        
        Args:
            instruction: The task instruction
            input_data: Input data for the task
            
        Returns:
            Response dictionary with answer and metrics
        """
        # Process all input data and add to knowledge
        if "initial_facts" in input_data:
            self.knowledge.extend(input_data["initial_facts"])
        elif "witness_statement" in input_data:
            self.knowledge.extend(input_data["witness_statement"])
        elif "forensic_evidence" in input_data:
            self.knowledge.extend(input_data["forensic_evidence"])
        elif "initial_data" in input_data:
            self.knowledge.extend(input_data["initial_data"])
        elif "updated_data" in input_data:
            self.knowledge.extend(input_data["updated_data"])
        elif "conflicting_data" in input_data:
            self.knowledge.extend(input_data["conflicting_data"])
        elif "final_data" in input_data:
            self.knowledge.extend(input_data["final_data"])
        elif "contradictions" in input_data:
            self.knowledge.extend(input_data["contradictions"])
        elif "challenge" in input_data:
            self.knowledge.append(input_data["challenge"])
        
        # Simple strategy: prefer higher confidence items and more recent information
        # When conflicts occur, just take the most recent high-confidence item
        filtered_knowledge = []
        for item in self.knowledge:
            confidence = item.get("confidence", 0.5)
            if confidence >= self.confidence_threshold:
                filtered_knowledge.append(item)
                
        # Mock response based on the filtered knowledge
        response = {
            "content": f"Processed {len(filtered_knowledge)} knowledge items.",
            "confidence": 0.7,
            "filtered_knowledge_count": len(filtered_knowledge),
            "total_knowledge_count": len(self.knowledge)
        }
        
        # Simple metrics for comparison
        metrics = {
            "coherence": 0.5,  # Baseline has no real coherence tracking
            "contradiction_count": 0,  # Baseline doesn't detect contradictions
            "response_time_ms": 100,
            "knowledge_items_used": len(filtered_knowledge)
        }
        
        return {
            "response": response,
            "metrics": metrics
        }


class BenchmarkController:
    """
    Controls the benchmark process for comparing agent performance.
    """
    
    def __init__(self, 
                output_dir: str = "benchmark_results",
                save_results: bool = True,
                visualize: bool = True):
        """
        Initialize the benchmark controller.
        
        Args:
            output_dir: Directory to save results
            save_results: Whether to save results to disk
            visualize: Whether to generate visualizations
        """
        self.output_dir = output_dir
        self.save_results = save_results
        self.visualize = visualize
        self.results = []
        
        # Create output directory if needed
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def run_scenario(self, 
                    scenario_name: str, 
                    agent: Any,
                    agent_type: str) -> BenchmarkResult:
        """
        Run a single agent through a benchmark scenario.
        
        Args:
            scenario_name: Name of the scenario to run
            agent: Agent object to evaluate
            agent_type: Type of agent ("psi_c" or "baseline")
            
        Returns:
            BenchmarkResult object with performance metrics
        """
        scenario = get_scenario(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
            
        logger.info(f"Running scenario '{scenario_name}' with {agent_type} agent")
        
        # Initialize tracking
        start_time = time.time()
        coherence_history = []
        contradiction_count = 0
        step_metrics = []
        
        # Run each step in the scenario
        for i, step in enumerate(scenario.steps):
            logger.info(f"Running step {i+1}/{len(scenario.steps)}: {step.instruction[:50]}...")
            
            # Execute step
            result = agent.respond(step.instruction, step.input_data)
            
            # Extract metrics
            step_result = {
                "step": i+1,
                "instruction": step.instruction,
                "response": result.get("response", {}).get("content", ""),
                "metrics": result.get("metrics", {})
            }
            
            # Track coherence and contradictions
            coherence = result.get("metrics", {}).get("coherence", 0.5)
            coherence_history.append(coherence)
            
            contradictions = result.get("metrics", {}).get("contradiction_count", 0)
            contradiction_count += contradictions
            
            step_metrics.append(step_result)
            
        # Calculate overall metrics
        runtime = time.time() - start_time
        
        # Aggregate metrics based on scenario evaluation criteria
        metrics = self._calculate_scenario_metrics(
            scenario=scenario,
            step_metrics=step_metrics,
            coherence_history=coherence_history,
            contradiction_count=contradiction_count
        )
        
        # Create result object
        result = BenchmarkResult(
            scenario_name=scenario_name,
            agent_type=agent_type,
            metrics=metrics,
            step_metrics=step_metrics,
            coherence_history=coherence_history,
            contradiction_count=contradiction_count,
            runtime_seconds=runtime
        )
        
        # Save results
        self.results.append(result)
        if self.save_results:
            self._save_result(result)
            
        logger.info(f"Completed scenario '{scenario_name}' with {agent_type} agent")
        logger.info(f"Overall performance score: {metrics.get('overall_score', 0):.4f}")
        
        return result
    
    def run_comparison(self, 
                      scenario_name: str,
                      agents: Dict[str, Any] = None) -> Dict[str, BenchmarkResult]:
        """
        Run multiple agents on the same scenario for comparison.
        
        Args:
            scenario_name: Name of the scenario to run
            agents: Dictionary of agent_type -> agent_instance
            
        Returns:
            Dictionary of agent_type -> BenchmarkResult
        """
        if agents is None:
            # Default agents
            agents = {
                "psi_c": PsiCAgent(),
                "baseline": BaselineAgent()
            }
            
        results = {}
        for agent_type, agent in agents.items():
            results[agent_type] = self.run_scenario(
                scenario_name=scenario_name,
                agent=agent,
                agent_type=agent_type
            )
            
        # Visualize comparison if enabled
        if self.visualize:
            self._visualize_comparison(results, scenario_name)
            
        return results
    
    def run_all_scenarios(self, 
                         agents: Dict[str, Any] = None) -> Dict[str, Dict[str, BenchmarkResult]]:
        """
        Run all scenarios with all agents.
        
        Args:
            agents: Dictionary of agent_type -> agent_instance
            
        Returns:
            Dictionary of scenario_name -> agent_type -> BenchmarkResult
        """
        if agents is None:
            # Default agents
            agents = {
                "psi_c": PsiCAgent(),
                "baseline": BaselineAgent()
            }
            
        all_results = {}
        for scenario_name in CANONICAL_SCENARIOS:
            all_results[scenario_name] = self.run_comparison(
                scenario_name=scenario_name,
                agents=agents
            )
            
        # Generate summary report
        self._generate_summary_report(all_results)
            
        return all_results
    
    def _calculate_scenario_metrics(self,
                                   scenario: TaskScenario,
                                   step_metrics: List[Dict[str, Any]],
                                   coherence_history: List[float],
                                   contradiction_count: int) -> Dict[str, float]:
        """
        Calculate overall metrics for a scenario run.
        
        Args:
            scenario: The scenario that was run
            step_metrics: Metrics from each step
            coherence_history: History of coherence values
            contradiction_count: Total contradictions encountered
            
        Returns:
            Dictionary of metric name -> value
        """
        # Extract main criteria from scenario
        criteria = scenario.evaluation_criteria
        
        # Calculate metrics based on scenario type
        if scenario.scenario_type.value == "narrative_planning":
            metrics = {
                "contradiction_resolution": 0.7 if contradiction_count > 0 else 0.3,
                "narrative_coherence": sum(coherence_history) / len(coherence_history),
                "evidence_incorporation": len(step_metrics[-1].get("metrics", {}).get("knowledge_items_used", 0)) / 20,
                "reasoning_quality": 0.6  # Would require more sophisticated evaluation
            }
        elif scenario.scenario_type.value == "decision_making":
            # Calculate adaptability based on decision changes
            adaptability = 0.5
            for i in range(1, len(step_metrics)):
                if step_metrics[i].get("response") != step_metrics[i-1].get("response"):
                    adaptability += 0.1
            
            metrics = {
                "adaptability": min(adaptability, 1.0),
                "decision_coherence": sum(coherence_history) / len(coherence_history),
                "uncertainty_handling": 0.7 if contradiction_count > 0 else 0.4,
                "risk_assessment": 0.6  # Would require more sophisticated evaluation
            }
        elif scenario.scenario_type.value == "identity_stability":
            # Calculate identity stability based on response consistency
            first_response = step_metrics[0].get("response", "")
            last_response = step_metrics[-1].get("response", "")
            
            # Simple similarity would be more sophisticated in real implementation
            similarity = 0.7  # Placeholder
            
            metrics = {
                "identity_stability": similarity,
                "principle_consistency": sum(coherence_history) / len(coherence_history),
                "reflection_depth": 0.5 + (contradiction_count * 0.1),
                "nuance_without_contradiction": 0.6  # Would require more sophisticated evaluation
            }
        else:
            # Generic metrics if scenario type not recognized
            metrics = {
                "coherence": sum(coherence_history) / len(coherence_history),
                "contradiction_handling": 0.7 if contradiction_count > 0 else 0.3,
                "task_completion": 0.8,
                "response_quality": 0.6
            }
            
        # Calculate overall score based on criteria weights
        overall_score = sum(metrics.get(key, 0) * weight for key, weight in criteria.items())
        metrics["overall_score"] = overall_score
        
        return metrics
    
    def _save_result(self, result: BenchmarkResult) -> None:
        """
        Save a benchmark result to disk.
        
        Args:
            result: The benchmark result to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{result.scenario_name}_{result.agent_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        logger.info(f"Saved result to {filename}")
    
    def _visualize_comparison(self, 
                             results: Dict[str, BenchmarkResult],
                             scenario_name: str) -> None:
        """
        Visualize comparison between agents for a scenario.
        
        Args:
            results: Dictionary of agent_type -> BenchmarkResult
            scenario_name: Name of the scenario
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Coherence over time
        ax = axs[0]
        for agent_type, result in results.items():
            ax.plot(result.coherence_history, marker='o', label=agent_type)
            
        ax.set_title(f"Coherence Over Time: {scenario_name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Coherence Score")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Performance metrics comparison
        ax = axs[1]
        metrics = {}
        for agent_type, result in results.items():
            for key, value in result.metrics.items():
                if key not in metrics:
                    metrics[key] = {}
                metrics[key][agent_type] = value
                
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(metrics)
        df.plot(kind='bar', ax=ax)
        
        ax.set_title(f"Performance Metrics: {scenario_name}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(title="Agent Type")
        ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{scenario_name}_comparison_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Saved comparison visualization to {filename}")
    
    def _generate_summary_report(self, 
                                all_results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
        """
        Generate summary report of all benchmark results.
        
        Args:
            all_results: Dictionary of scenario_name -> agent_type -> BenchmarkResult
        """
        # Create summary data structure
        summary = {
            "timestamp": datetime.now().isoformat(),
            "scenarios_run": len(all_results),
            "agents_compared": list(next(iter(all_results.values())).keys()),
            "scenario_results": {},
            "agent_summaries": {}
        }
        
        # Process each scenario
        for scenario_name, results in all_results.items():
            # Get winner
            winner = max(results.items(), key=lambda x: x[1].metrics.get("overall_score", 0))
            winner_name = winner[0]
            winner_score = winner[1].metrics.get("overall_score", 0)
            
            # Store scenario summary
            summary["scenario_results"][scenario_name] = {
                "winner": winner_name,
                "winner_score": winner_score,
                "agent_scores": {
                    agent_type: result.metrics.get("overall_score", 0)
                    for agent_type, result in results.items()
                }
            }
            
            # Update agent summaries
            for agent_type, result in results.items():
                if agent_type not in summary["agent_summaries"]:
                    summary["agent_summaries"][agent_type] = {
                        "scenarios_won": 0,
                        "average_score": 0,
                        "total_contradictions": 0,
                        "average_runtime": 0,
                        "scenario_scores": {}
                    }
                
                agent_summary = summary["agent_summaries"][agent_type]
                score = result.metrics.get("overall_score", 0)
                
                if agent_type == winner_name:
                    agent_summary["scenarios_won"] += 1
                    
                agent_summary["scenario_scores"][scenario_name] = score
                agent_summary["total_contradictions"] += result.contradiction_count
                agent_summary["average_runtime"] += result.runtime_seconds
        
        # Calculate averages
        for agent_type, agent_summary in summary["agent_summaries"].items():
            agent_summary["average_score"] = sum(agent_summary["scenario_scores"].values()) / len(agent_summary["scenario_scores"])
            agent_summary["average_runtime"] /= len(agent_summary["scenario_scores"])
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/summary_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved summary report to {filename}")
        
        # Also create visualization of summary
        if self.visualize:
            self._visualize_summary(summary)
    
    def _visualize_summary(self, summary: Dict[str, Any]) -> None:
        """
        Visualize benchmark summary.
        
        Args:
            summary: Summary data structure
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Overall agent performance
        ax = axs[0]
        agents = list(summary["agent_summaries"].keys())
        scores = [summary["agent_summaries"][agent]["average_score"] for agent in agents]
        
        bars = ax.bar(agents, scores)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_title("Average Performance Across All Scenarios")
        ax.set_ylabel("Average Score")
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        # Plot 2: Scenario-specific performance
        ax = axs[1]
        
        # Convert to DataFrame for easier plotting
        data = {}
        for scenario_name, scenario_results in summary["scenario_results"].items():
            for agent_type, score in scenario_results["agent_scores"].items():
                if agent_type not in data:
                    data[agent_type] = {}
                data[agent_type][scenario_name] = score
                
        df = pd.DataFrame(data)
        df.plot(kind='bar', ax=ax)
        
        ax.set_title("Performance By Scenario")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(title="Agent Type")
        ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/summary_visualization_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Saved summary visualization to {filename}")


def run_benchmark():
    """Run all benchmark scenarios and generate comparison report."""
    controller = BenchmarkController()
    
    # Run all scenarios
    results = controller.run_all_scenarios()
    
    logger.info("Benchmark complete!")
    logger.info(f"Results saved to {controller.output_dir}")
    
    return results


if __name__ == "__main__":
    run_benchmark() 