"""
Unified Comparison Dashboard for ΨC-AI SDK
------------------------------------------

This module provides tools to compare ΨC agents with other cognitive architectures
including SOAR, ACT-R, and IIT. It aggregates metrics, generates visualizations,
and exports comparison data for analysis.

The dashboard helps researchers understand how ΨC's characteristics differ from
established cognitive frameworks by comparing:
- Coherence trajectories
- Memory schema size
- Reflection vs. decision depth
- Entropy profiles
- Identity evolution

Usage:
    dashboard = ComparisonDashboard()
    dashboard.load_psi_metrics(agent)
    dashboard.load_baseline("SOAR", soar_data)
    dashboard.compare_all()
    dashboard.visualize()
    dashboard.export_json("comparison_results.json")
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class ComparisonDashboard:
    """
    Unified dashboard for comparing ΨC agents with other cognitive frameworks.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the comparison dashboard.
        
        Args:
            data_dir: Directory containing baseline comparison data.
                     If None, will use default path.
        """
        self.psi_metrics = {}
        self.baselines = {}
        self.comparison_results = {}
        self.data_dir = data_dir or self._get_default_data_dir()
        self.console = Console() if RICH_AVAILABLE else None

    def _get_default_data_dir(self) -> str:
        """Get the default directory for baseline data."""
        # First check if there's a data directory in the package
        package_dir = Path(__file__).parent.parent
        data_dir = package_dir / "data"
        if data_dir.exists():
            return str(data_dir)
        
        # If not, use a directory in the user's home
        home_dir = Path.home() / ".psi_c_ai_sdk" / "data"
        home_dir.mkdir(parents=True, exist_ok=True)
        return str(home_dir)

    def load_psi_metrics(self, agent=None, metrics_file: Optional[str] = None):
        """
        Load ΨC metrics either from a running agent or from a metrics file.
        
        Args:
            agent: A running ΨC agent instance, if available
            metrics_file: Path to a saved metrics file
        """
        if agent is not None:
            # Try to get metrics directly from the agent
            try:
                # This assumes the agent has a get_metrics method or similar
                self.psi_metrics = agent.get_metrics()
                logger.info(f"Loaded metrics directly from agent")
                return
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not load metrics directly from agent: {e}")
        
        # Fall back to loading from a file
        if metrics_file:
            metrics_path = metrics_file
        else:
            # Try to find metrics.py in the monitor directory
            try:
                from psi_c_ai_sdk.monitor import metrics
                self.psi_metrics = metrics.get_latest_metrics()
                logger.info(f"Loaded metrics from monitor.metrics module")
                return
            except ImportError:
                logger.warning("Could not import metrics from monitor.metrics")
                
                # Try to find reflection_metrics.py
                try:
                    from psi_c_ai_sdk import reflection_metrics
                    self.psi_metrics = reflection_metrics.get_latest_metrics()
                    logger.info(f"Loaded metrics from reflection_metrics module")
                    return
                except ImportError:
                    logger.warning("Could not import metrics from reflection_metrics")
                
                # Fall back to a default location
                metrics_path = os.path.join(self.data_dir, "psi_metrics.json")
        
        try:
            with open(metrics_path, 'r') as f:
                self.psi_metrics = json.load(f)
            logger.info(f"Loaded metrics from {metrics_path}")
        except FileNotFoundError:
            logger.warning(f"Metrics file not found: {metrics_path}")
            self.psi_metrics = self._generate_default_psi_metrics()
            logger.info("Generated default ΨC metrics")

    def _generate_default_psi_metrics(self) -> Dict[str, Any]:
        """Generate default ΨC metrics for demonstration."""
        return {
            "coherence": {
                "mean": 0.78,
                "trajectory": [0.5, 0.55, 0.62, 0.7, 0.75, 0.78],
                "variance": 0.02
            },
            "schema": {
                "size": 256,
                "depth": 3,
                "complexity": 45.2
            },
            "reflection": {
                "count": 12,
                "depth_distribution": [4, 5, 2, 1],
                "mean_recursive_depth": 1.83
            },
            "entropy": {
                "mean": 0.35,
                "trajectory": [0.6, 0.5, 0.45, 0.4, 0.38, 0.35],
                "profile": "decreasing"
            },
            "identity": {
                "stability": 0.85,
                "evolution_rate": 0.02,
                "mutation_count": 3
            }
        }

    def load_baseline(self, architecture: str, data: Optional[Dict] = None, file_path: Optional[str] = None):
        """
        Load comparison baseline for a cognitive architecture.
        
        Args:
            architecture: Name of the cognitive architecture (e.g., "SOAR", "ACT-R")
            data: Pre-loaded data dictionary
            file_path: Path to JSON file containing the baseline data
        """
        architecture = architecture.upper()
        
        if data is not None:
            self.baselines[architecture] = data
            logger.info(f"Loaded {architecture} baseline from provided data")
            return
            
        if file_path is not None:
            baseline_path = file_path
        else:
            baseline_path = os.path.join(self.data_dir, f"{architecture.lower()}_baseline.json")
        
        try:
            with open(baseline_path, 'r') as f:
                self.baselines[architecture] = json.load(f)
            logger.info(f"Loaded {architecture} baseline from {baseline_path}")
        except FileNotFoundError:
            logger.warning(f"Baseline file not found: {baseline_path}")
            self.baselines[architecture] = self._generate_default_baseline(architecture)
            logger.info(f"Generated default {architecture} baseline")

    def _generate_default_baseline(self, architecture: str) -> Dict[str, Any]:
        """Generate default baseline metrics for demonstration."""
        if architecture == "SOAR":
            return {
                "coherence": {
                    "mean": 0.65,
                    "trajectory": [0.63, 0.64, 0.65, 0.65, 0.65, 0.65],
                    "variance": 0.01
                },
                "schema": {
                    "size": 128,
                    "depth": 1,
                    "complexity": 25.0
                },
                "reflection": {
                    "count": 5,
                    "depth_distribution": [5, 0, 0, 0],
                    "mean_recursive_depth": 1.0
                },
                "entropy": {
                    "mean": 0.45,
                    "trajectory": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                    "profile": "flat"
                },
                "identity": {
                    "stability": 0.95,
                    "evolution_rate": 0.01,
                    "mutation_count": 1
                }
            }
        elif architecture == "ACT-R":
            return {
                "coherence": {
                    "mean": 0.7,
                    "trajectory": [0.68, 0.69, 0.7, 0.7, 0.7, 0.7],
                    "variance": 0.01
                },
                "schema": {
                    "size": 80,
                    "depth": 2,
                    "complexity": 18.5
                },
                "reflection": {
                    "count": 0,
                    "depth_distribution": [0, 0, 0, 0],
                    "mean_recursive_depth": 0.0
                },
                "entropy": {
                    "mean": 0.5,
                    "trajectory": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    "profile": "flat"
                },
                "identity": {
                    "stability": 0.98,
                    "evolution_rate": 0.005,
                    "mutation_count": 0
                }
            }
        elif architecture == "IIT":
            return {
                "coherence": {
                    "mean": 0.82,
                    "trajectory": [0.8, 0.81, 0.82, 0.82, 0.82, 0.82],
                    "variance": 0.01
                },
                "schema": {
                    "size": 320,
                    "depth": 4,
                    "complexity": 60.0
                },
                "reflection": {
                    "count": 15,
                    "depth_distribution": [5, 5, 3, 2],
                    "mean_recursive_depth": 2.13
                },
                "entropy": {
                    "mean": 0.25,
                    "trajectory": [0.3, 0.28, 0.26, 0.25, 0.25, 0.25],
                    "profile": "low"
                },
                "identity": {
                    "stability": 0.6,
                    "evolution_rate": 0.1,
                    "mutation_count": 8
                }
            }
        else:
            # Generic baseline
            return {
                "coherence": {"mean": 0.5, "trajectory": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "variance": 0.0},
                "schema": {"size": 100, "depth": 1, "complexity": 20.0},
                "reflection": {"count": 0, "depth_distribution": [0, 0, 0, 0], "mean_recursive_depth": 0.0},
                "entropy": {"mean": 0.5, "trajectory": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "profile": "flat"},
                "identity": {"stability": 0.9, "evolution_rate": 0.01, "mutation_count": 1}
            }

    def compare_all(self, weights: Optional[Dict[str, float]] = None):
        """
        Run comparison against all loaded baselines with optional custom weighting.
        
        Args:
            weights: Custom weights for different metrics in the final score
        """
        if not self.psi_metrics:
            raise ValueError("No ΨC metrics loaded. Call load_psi_metrics first.")
        
        if not self.baselines:
            logger.warning("No baselines loaded. Loading default baselines.")
            self.load_baseline("SOAR")
            self.load_baseline("ACT-R")
            self.load_baseline("IIT")
        
        # Default weights if not provided
        if weights is None:
            weights = {
                "coherence": 0.25,
                "schema": 0.2,
                "reflection": 0.25,
                "entropy": 0.15,
                "identity": 0.15
            }
        
        # Run comparisons for each baseline
        for arch, baseline in self.baselines.items():
            self.comparison_results[arch] = self._compare_to_baseline(baseline, weights)
            logger.info(f"Completed comparison with {arch}")

    def _compare_to_baseline(self, baseline: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare ΨC metrics with a baseline using the specified weights.
        
        Args:
            baseline: Baseline metrics to compare against
            weights: Weights for different metric categories
            
        Returns:
            Comparison results dictionary
        """
        result = {"delta_weighted": 0.0, "metrics": {}}
        
        # Compare coherence
        coherence_delta = abs(self.psi_metrics["coherence"]["mean"] - baseline["coherence"]["mean"])
        result["metrics"]["coherence"] = {
            "delta": coherence_delta,
            "psi": self.psi_metrics["coherence"]["mean"],
            "baseline": baseline["coherence"]["mean"]
        }
        result["delta_weighted"] += weights["coherence"] * coherence_delta
        
        # Compare schema size
        schema_size_delta = abs(self.psi_metrics["schema"]["size"] - baseline["schema"]["size"]) / max(self.psi_metrics["schema"]["size"], baseline["schema"]["size"])
        result["metrics"]["schema_size"] = {
            "delta": schema_size_delta,
            "psi": self.psi_metrics["schema"]["size"],
            "baseline": baseline["schema"]["size"]
        }
        result["delta_weighted"] += weights["schema"] * schema_size_delta
        
        # Compare reflection depth
        reflection_delta = abs(self.psi_metrics["reflection"]["mean_recursive_depth"] - baseline["reflection"]["mean_recursive_depth"])
        result["metrics"]["reflection_depth"] = {
            "delta": reflection_delta,
            "psi": self.psi_metrics["reflection"]["mean_recursive_depth"],
            "baseline": baseline["reflection"]["mean_recursive_depth"]
        }
        result["delta_weighted"] += weights["reflection"] * reflection_delta
        
        # Compare entropy
        entropy_delta = abs(self.psi_metrics["entropy"]["mean"] - baseline["entropy"]["mean"])
        result["metrics"]["entropy"] = {
            "delta": entropy_delta,
            "psi": self.psi_metrics["entropy"]["mean"],
            "baseline": baseline["entropy"]["mean"]
        }
        result["delta_weighted"] += weights["entropy"] * entropy_delta
        
        # Compare identity stability
        identity_delta = abs(self.psi_metrics["identity"]["stability"] - baseline["identity"]["stability"])
        result["metrics"]["identity_stability"] = {
            "delta": identity_delta,
            "psi": self.psi_metrics["identity"]["stability"],
            "baseline": baseline["identity"]["stability"]
        }
        result["delta_weighted"] += weights["identity"] * identity_delta
        
        return result

    def compare_psi_to_soar_metrics(self) -> Dict[str, Any]:
        """Run comparison specifically against SOAR baseline."""
        if "SOAR" not in self.baselines:
            self.load_baseline("SOAR")
        
        weights = {
            "coherence": 0.25,
            "schema": 0.2, 
            "reflection": 0.25,
            "entropy": 0.15,
            "identity": 0.15
        }
        
        return self._compare_to_baseline(self.baselines["SOAR"], weights)

    def compare_psi_to_actr_metrics(self) -> Dict[str, Any]:
        """Run comparison specifically against ACT-R baseline."""
        if "ACT-R" not in self.baselines:
            self.load_baseline("ACT-R")
        
        weights = {
            "coherence": 0.25,
            "schema": 0.2,
            "reflection": 0.25, 
            "entropy": 0.15,
            "identity": 0.15
        }
        
        return self._compare_to_baseline(self.baselines["ACT-R"], weights)

    def compare_psi_to_iit_metrics(self) -> Dict[str, Any]:
        """Run comparison specifically against IIT baseline."""
        if "IIT" not in self.baselines:
            self.load_baseline("IIT")
        
        weights = {
            "coherence": 0.3,  # IIT emphasizes coherence/integration more
            "schema": 0.2,
            "reflection": 0.2,
            "entropy": 0.15,
            "identity": 0.15
        }
        
        return self._compare_to_baseline(self.baselines["IIT"], weights)

    def visualize(self, output_dir: Optional[str] = None):
        """
        Generate visualizations comparing ΨC with baselines.
        
        Args:
            output_dir: Directory to save visualizations. If None, will display only.
        """
        if not self.comparison_results:
            logger.warning("No comparison results to visualize. Run compare_all first.")
            return
            
        # If Rich is available, display table in terminal
        if RICH_AVAILABLE:
            self._display_rich_table()
        else:
            self._display_simple_table()
            
        # Create and save/display visualizations
        self._plot_coherence_trajectories(output_dir)
        self._plot_reflection_depth_comparison(output_dir)
        self._plot_entropy_trajectories(output_dir)
        self._plot_schema_size_comparison(output_dir)
        self._plot_identity_stability_comparison(output_dir)
        
        # Display in Streamlit if available and running in Streamlit context
        if STREAMLIT_AVAILABLE and 'streamlit' in __name__:
            self._display_streamlit_dashboard()

    def _display_rich_table(self):
        """Display comparison results in a rich formatted table."""
        table = Table(title="ΨC vs. Other Cognitive Architectures")
        
        # Add columns
        table.add_column("Metric", style="cyan")
        table.add_column("ΨC", style="green")
        
        for arch in self.comparison_results.keys():
            table.add_column(arch, style="yellow")
            
        # Add rows for each metric
        metrics = ["coherence", "schema_size", "reflection_depth", "entropy", "identity_stability"]
        metric_names = ["Coherence", "Schema Size", "Reflection Depth", "Entropy", "Identity Stability"]
        
        for i, metric in enumerate(metrics):
            row = [metric_names[i], f"{self.psi_metrics[metric.split('_')[0]]['mean']:.3f}" if '_' not in metric else f"{self.psi_metrics[metric.split('_')[0]][metric.split('_')[1]]:.3f}"]
            
            for arch in self.comparison_results.keys():
                baseline_value = self.comparison_results[arch]["metrics"][metric]["baseline"]
                delta = self.comparison_results[arch]["metrics"][metric]["delta"]
                row.append(f"{baseline_value:.3f} (Δ: {delta:.3f})")
                
            table.add_row(*row)
            
        # Add weighted delta row
        row = ["Weighted Delta", ""]
        for arch in self.comparison_results.keys():
            row.append(f"{self.comparison_results[arch]['delta_weighted']:.3f}")
            
        table.add_row(*row)
        
        # Display the table
        self.console.print(table)

    def _display_simple_table(self):
        """Display comparison results in a simple text table."""
        print("\nΨC vs. Other Cognitive Architectures")
        print("-" * 80)
        
        # Print header
        header = ["Metric", "ΨC"]
        for arch in self.comparison_results.keys():
            header.append(arch)
        print("{:<20} {:<10} {}".format(header[0], header[1], " ".join(["{:<20}".format(h) for h in header[2:]])))
        print("-" * 80)
        
        # Print rows for each metric
        metrics = ["coherence", "schema_size", "reflection_depth", "entropy", "identity_stability"]
        metric_names = ["Coherence", "Schema Size", "Reflection Depth", "Entropy", "Identity Stability"]
        
        for i, metric in enumerate(metrics):
            row = [metric_names[i]]
            
            # ΨC value
            if '_' not in metric:
                row.append(f"{self.psi_metrics[metric]['mean']:.3f}")
            else:
                parts = metric.split('_')
                row.append(f"{self.psi_metrics[parts[0]][parts[1]]:.3f}")
            
            # Comparison architectures
            for arch in self.comparison_results.keys():
                baseline_value = self.comparison_results[arch]["metrics"][metric]["baseline"]
                delta = self.comparison_results[arch]["metrics"][metric]["delta"]
                row.append(f"{baseline_value:.3f} (Δ: {delta:.3f})")
                
            print("{:<20} {:<10} {}".format(row[0], row[1], " ".join(["{:<20}".format(r) for r in row[2:]])))
        
        # Print weighted delta row
        print("-" * 80)
        row = ["Weighted Delta", ""]
        for arch in self.comparison_results.keys():
            row.append(f"{self.comparison_results[arch]['delta_weighted']:.3f}")
            
        print("{:<20} {:<10} {}".format(row[0], row[1], " ".join(["{:<20}".format(r) for r in row[2:]])))

    def _plot_coherence_trajectories(self, output_dir: Optional[str] = None):
        """Plot coherence trajectories for ΨC and baselines."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.psi_metrics["coherence"]["trajectory"], label="ΨC", linewidth=2)
        
        for arch, baseline in self.baselines.items():
            plt.plot(baseline["coherence"]["trajectory"], label=arch, linestyle="--")
            
        plt.title("Coherence Trajectories")
        plt.xlabel("Time Steps")
        plt.ylabel("Coherence")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "coherence_trajectories.png"), dpi=300)
        else:
            plt.show()

    def _plot_reflection_depth_comparison(self, output_dir: Optional[str] = None):
        """Plot reflection depth distribution for ΨC and baselines."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        architectures = ["ΨC"] + list(self.baselines.keys())
        depths = ["Depth 1", "Depth 2", "Depth 3", "Depth 4"]
        
        x = np.arange(len(depths))
        width = 0.8 / len(architectures)
        
        for i, arch in enumerate(architectures):
            if arch == "ΨC":
                data = self.psi_metrics["reflection"]["depth_distribution"]
            else:
                data = self.baselines[arch]["reflection"]["depth_distribution"]
                
            ax.bar(x + i * width - 0.4, data, width, label=arch)
            
        ax.set_title("Reflection Depth Distribution")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(depths)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7, axis="y")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "reflection_depth_comparison.png"), dpi=300)
        else:
            plt.show()

    def _plot_entropy_trajectories(self, output_dir: Optional[str] = None):
        """Plot entropy trajectories for ΨC and baselines."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.psi_metrics["entropy"]["trajectory"], label="ΨC", linewidth=2)
        
        for arch, baseline in self.baselines.items():
            plt.plot(baseline["entropy"]["trajectory"], label=arch, linestyle="--")
            
        plt.title("Entropy Trajectories")
        plt.xlabel("Time Steps")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "entropy_trajectories.png"), dpi=300)
        else:
            plt.show()

    def _plot_schema_size_comparison(self, output_dir: Optional[str] = None):
        """Plot schema size comparison for ΨC and baselines."""
        plt.figure(figsize=(10, 6))
        
        architectures = ["ΨC"] + list(self.baselines.keys())
        sizes = [self.psi_metrics["schema"]["size"]]
        for arch in self.baselines.keys():
            sizes.append(self.baselines[arch]["schema"]["size"])
            
        colors = ["green"] + ["lightblue"] * len(self.baselines)
        
        plt.bar(architectures, sizes, color=colors)
        plt.title("Schema Size Comparison")
        plt.ylabel("Schema Size")
        plt.grid(True, linestyle="--", alpha=0.7, axis="y")
        
        # Add value labels
        for i, v in enumerate(sizes):
            plt.text(i, v + 5, str(v), ha="center")
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "schema_size_comparison.png"), dpi=300)
        else:
            plt.show()

    def _plot_identity_stability_comparison(self, output_dir: Optional[str] = None):
        """Plot identity stability comparison for ΨC and baselines."""
        plt.figure(figsize=(10, 6))
        
        architectures = ["ΨC"] + list(self.baselines.keys())
        stabilities = [self.psi_metrics["identity"]["stability"]]
        for arch in self.baselines.keys():
            stabilities.append(self.baselines[arch]["identity"]["stability"])
            
        colors = ["green"] + ["lightblue"] * len(self.baselines)
        
        plt.bar(architectures, stabilities, color=colors)
        plt.title("Identity Stability Comparison")
        plt.ylabel("Stability Score")
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle="--", alpha=0.7, axis="y")
        
        # Add value labels
        for i, v in enumerate(stabilities):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "identity_stability_comparison.png"), dpi=300)
        else:
            plt.show()

    def _display_streamlit_dashboard(self):
        """Display interactive dashboard in Streamlit."""
        st.title("ΨC-AI Cognitive Architecture Comparison")
        
        # Summary metrics
        st.header("Comparison Summary")
        
        # Create a DataFrame for the metrics
        data = {
            "Metric": ["Coherence", "Schema Size", "Reflection Depth", "Entropy", "Identity Stability"]
        }
        
        data["ΨC"] = [
            f"{self.psi_metrics['coherence']['mean']:.3f}",
            f"{self.psi_metrics['schema']['size']}",
            f"{self.psi_metrics['reflection']['mean_recursive_depth']:.2f}",
            f"{self.psi_metrics['entropy']['mean']:.3f}",
            f"{self.psi_metrics['identity']['stability']:.3f}"
        ]
        
        for arch in self.comparison_results.keys():
            data[arch] = [
                f"{self.comparison_results[arch]['metrics']['coherence']['baseline']:.3f} (Δ: {self.comparison_results[arch]['metrics']['coherence']['delta']:.3f})",
                f"{self.comparison_results[arch]['metrics']['schema_size']['baseline']} (Δ: {self.comparison_results[arch]['metrics']['schema_size']['delta']:.3f})",
                f"{self.comparison_results[arch]['metrics']['reflection_depth']['baseline']:.2f} (Δ: {self.comparison_results[arch]['metrics']['reflection_depth']['delta']:.2f})",
                f"{self.comparison_results[arch]['metrics']['entropy']['baseline']:.3f} (Δ: {self.comparison_results[arch]['metrics']['entropy']['delta']:.3f})",
                f"{self.comparison_results[arch]['metrics']['identity_stability']['baseline']:.3f} (Δ: {self.comparison_results[arch]['metrics']['identity_stability']['delta']:.3f})"
            ]
            
        df = pd.DataFrame(data)
        st.dataframe(df.set_index("Metric"))
        
        # Overall difference scores
        st.subheader("Overall Weighted Difference")
        delta_data = {"Architecture": [], "Weighted Delta": []}
        for arch in self.comparison_results.keys():
            delta_data["Architecture"].append(arch)
            delta_data["Weighted Delta"].append(self.comparison_results[arch]["delta_weighted"])
            
        delta_df = pd.DataFrame(delta_data)
        st.bar_chart(delta_df.set_index("Architecture"))
        
        # Interactive visualizations
        st.header("Detailed Comparisons")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Coherence", "Reflection", "Entropy", "Schema Size", "Identity"
        ])
        
        with tab1:
            st.subheader("Coherence Trajectories")
            # Create DataFrame for coherence trajectories
            coherence_data = {"Time Step": list(range(len(self.psi_metrics["coherence"]["trajectory"])))}
            coherence_data["ΨC"] = self.psi_metrics["coherence"]["trajectory"]
            
            for arch in self.baselines.keys():
                coherence_data[arch] = self.baselines[arch]["coherence"]["trajectory"]
                
            coherence_df = pd.DataFrame(coherence_data)
            st.line_chart(coherence_df.set_index("Time Step"))
            
        with tab2:
            st.subheader("Reflection Depth Distribution")
            reflection_data = {"Depth": ["Depth 1", "Depth 2", "Depth 3", "Depth 4"]}
            reflection_data["ΨC"] = self.psi_metrics["reflection"]["depth_distribution"]
            
            for arch in self.baselines.keys():
                reflection_data[arch] = self.baselines[arch]["reflection"]["depth_distribution"]
                
            reflection_df = pd.DataFrame(reflection_data)
            st.bar_chart(reflection_df.set_index("Depth"))
            
        with tab3:
            st.subheader("Entropy Trajectories")
            entropy_data = {"Time Step": list(range(len(self.psi_metrics["entropy"]["trajectory"])))}
            entropy_data["ΨC"] = self.psi_metrics["entropy"]["trajectory"]
            
            for arch in self.baselines.keys():
                entropy_data[arch] = self.baselines[arch]["entropy"]["trajectory"]
                
            entropy_df = pd.DataFrame(entropy_data)
            st.line_chart(entropy_df.set_index("Time Step"))
            
        with tab4:
            st.subheader("Schema Size Comparison")
            schema_data = {"Architecture": ["ΨC"] + list(self.baselines.keys())}
            schema_data["Size"] = [self.psi_metrics["schema"]["size"]]
            
            for arch in self.baselines.keys():
                schema_data["Size"].append(self.baselines[arch]["schema"]["size"])
                
            schema_df = pd.DataFrame(schema_data)
            st.bar_chart(schema_df.set_index("Architecture"))
            
        with tab5:
            st.subheader("Identity Stability Comparison")
            identity_data = {"Architecture": ["ΨC"] + list(self.baselines.keys())}
            identity_data["Stability"] = [self.psi_metrics["identity"]["stability"]]
            
            for arch in self.baselines.keys():
                identity_data["Stability"].append(self.baselines[arch]["identity"]["stability"])
                
            identity_df = pd.DataFrame(identity_data)
            st.bar_chart(identity_df.set_index("Architecture"))

    def export_json(self, file_path: str):
        """
        Export comparison results to JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        if not self.comparison_results:
            logger.warning("No comparison results to export. Run compare_all first.")
            return
            
        output = {
            "psi_metrics": self.psi_metrics,
            "baselines": self.baselines,
            "comparison_results": self.comparison_results,
            "export_time": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Exported comparison results to {file_path}")
        
        return file_path


def main():
    """Command-line interface for the comparison dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ΨC-AI SDK Comparison Dashboard")
    parser.add_argument("--metrics", help="Path to ΨC metrics JSON file")
    parser.add_argument("--data-dir", help="Directory containing baseline data")
    parser.add_argument("--output", help="Directory to save visualizations")
    parser.add_argument("--export", help="File to export comparison results as JSON")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit dashboard")
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = ComparisonDashboard(data_dir=args.data_dir)
    
    # Load ΨC metrics
    dashboard.load_psi_metrics(metrics_file=args.metrics)
    
    # Load baselines
    dashboard.load_baseline("SOAR")
    dashboard.load_baseline("ACT-R")
    dashboard.load_baseline("IIT")
    
    # Run comparison
    dashboard.compare_all()
    
    # Visualize results
    dashboard.visualize(output_dir=args.output)
    
    # Export results if requested
    if args.export:
        dashboard.export_json(args.export)
        
    # Launch Streamlit dashboard if requested
    if args.streamlit and STREAMLIT_AVAILABLE:
        import subprocess
        import sys
        
        # Create a temporary Streamlit script
        temp_script = os.path.join(os.getcwd(), "comparison_dashboard_streamlit.py")
        with open(temp_script, "w") as f:
            f.write("""
import streamlit as st
import pickle
import sys
import os
sys.path.append(os.getcwd())
from psi_c_ai_sdk.tools.comparison_dashboard import ComparisonDashboard

# Load dashboard data
with open('dashboard_data.pkl', 'rb') as f:
    dashboard = pickle.load(f)
    
# Display dashboard
dashboard._display_streamlit_dashboard()
""")
        
        # Save dashboard to file
        with open("dashboard_data.pkl", "wb") as f:
            import pickle
            pickle.dump(dashboard, f)
            
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", temp_script])
        
        # Clean up
        os.remove(temp_script)
        os.remove("dashboard_data.pkl")


if __name__ == "__main__":
    main() 