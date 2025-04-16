"""
Empirical ΨC Calibration Logger

This module implements a data collection and analysis system to empirically calibrate
the ΨC activation thresholds based on observed correlations between ΨC scores and
meaningful cognitive changes in the system.

The calibration logger tracks:
- ΨC scores over time
- When schema changes occur
- Entropy deltas
- Reflection outcomes

This empirical data allows:
- Data-driven threshold calibration
- Sigmoid slope optimization
- Correlation analysis between ΨC scores and cognitive changes
"""

import time
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from datetime import datetime


class CalibrationLogger:
    """
    Empirical calibration logger that tracks ΨC activations and system changes
    to determine optimal thresholds and correlations.
    
    This class collects data points over time and provides analysis methods to
    determine what ΨC scores correlate with meaningful internal changes in the
    cognitive system.
    
    Attributes:
        log_dir (str): Directory where calibration logs are saved
        session_id (str): Unique identifier for the current session
        log_interval (float): Minimum time between log entries in seconds
        events (List[Dict]): List of recorded events
        last_log_time (float): Timestamp of the last log entry
        metrics_callback (Callable): Function to call to get current metrics
    """
    
    def __init__(
        self,
        log_dir: str = "logs/calibration",
        session_id: Optional[str] = None,
        log_interval: float = 1.0,
        metrics_callback: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        """
        Initialize the calibration logger.
        
        Args:
            log_dir: Directory to store calibration logs
            session_id: Optional unique ID for the session (default: timestamp)
            log_interval: Minimum time between log entries in seconds
            metrics_callback: Optional function to call to get current metrics
        """
        self.log_dir = log_dir
        self.session_id = session_id or f"session_{int(time.time())}"
        self.log_interval = log_interval
        self.events: List[Dict[str, Any]] = []
        self.last_log_time = 0
        self.metrics_callback = metrics_callback
        
        # Set up logging
        self.logger = logging.getLogger("calibration_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger.info(f"Calibration logger initialized with session ID: {self.session_id}")
    
    def log_state(
        self,
        psi_score: float,
        entropy: float,
        schema_changed: bool,
        reflection_occurred: bool,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log the current state of the system for calibration purposes.
        
        Args:
            psi_score: Current ΨC score
            entropy: Current entropy value
            schema_changed: Whether a schema change occurred
            reflection_occurred: Whether a reflection cycle occurred
            additional_metrics: Optional additional metrics to include
        """
        # Rate limiting - don't log too frequently
        current_time = time.time()
        if current_time - self.last_log_time < self.log_interval:
            return
        
        self.last_log_time = current_time
        
        # Create event record
        event = {
            "timestamp": current_time,
            "psi_score": psi_score,
            "entropy": entropy,
            "schema_changed": schema_changed,
            "reflection_occurred": reflection_occurred,
            "human_time": datetime.now().isoformat()
        }
        
        # Add additional metrics
        if additional_metrics:
            event.update(additional_metrics)
        
        # If we have a metrics callback, add those metrics too
        if self.metrics_callback:
            try:
                callback_metrics = self.metrics_callback()
                if callback_metrics:
                    event.update(callback_metrics)
            except Exception as e:
                self.logger.warning(f"Error getting metrics from callback: {e}")
        
        # Add to events list
        self.events.append(event)
        
        # Periodically save to disk
        if len(self.events) % 20 == 0:
            self.save_logs()
    
    def log_toolkit_state(self, toolkit) -> None:
        """
        Convenience method to log state directly from a PsiToolkit instance.
        
        Args:
            toolkit: A PsiToolkit instance
        """
        try:
            # Get metrics from toolkit
            metrics = toolkit.psi_operator.psi_index()
            threshold_metrics = toolkit.get_threshold_metrics()
            
            # Extract key metrics
            psi_score = metrics.get("psi_c_score", 0.0)
            entropy = 1.0 - metrics.get("stability", 0.0)  # Simple approximation
            
            # Check if this is a reflection event
            reflection_occurred = False
            schema_changed = False
            
            # Get recent activation logs if available
            if hasattr(toolkit, "get_activation_log"):
                recent_logs = toolkit.get_activation_log(limit=1)
                if recent_logs:
                    log_entry = recent_logs[0]
                    if log_entry.get("event_type") == "reflection":
                        reflection_occurred = True
                        schema_changed = log_entry.get("schema_changed", False)
            
            # Combine all metrics
            additional_metrics = {
                "threshold": threshold_metrics.get("current_threshold", 0.7),
                "threshold_type": threshold_metrics.get("type", "static"),
                "state": toolkit.get_psi_state().value,
                "stability": metrics.get("stability", 0.0),
                "coherence": metrics.get("coherence", 0.0)
            }
            
            # Log the state
            self.log_state(
                psi_score=psi_score,
                entropy=entropy,
                schema_changed=schema_changed,
                reflection_occurred=reflection_occurred,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error logging toolkit state: {e}")
    
    def save_logs(self) -> str:
        """
        Save the current events to disk as a JSON file.
        
        Returns:
            Path to the saved log file
        """
        if not self.events:
            return ""
        
        log_path = os.path.join(self.log_dir, f"{self.session_id}_calibration.json")
        
        try:
            with open(log_path, 'w') as f:
                json.dump(self.events, f, indent=2)
            
            self.logger.info(f"Saved {len(self.events)} calibration events to {log_path}")
            return log_path
        except Exception as e:
            self.logger.error(f"Error saving calibration logs: {e}")
            return ""
    
    def load_logs(self, session_id: Optional[str] = None) -> bool:
        """
        Load calibration logs from disk.
        
        Args:
            session_id: Session ID to load (default: current session)
            
        Returns:
            True if logs were loaded successfully, False otherwise
        """
        session_to_load = session_id or self.session_id
        log_path = os.path.join(self.log_dir, f"{session_to_load}_calibration.json")
        
        if not os.path.exists(log_path):
            self.logger.warning(f"Calibration log not found: {log_path}")
            return False
        
        try:
            with open(log_path, 'r') as f:
                self.events = json.load(f)
            
            self.logger.info(f"Loaded {len(self.events)} calibration events from {log_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading calibration logs: {e}")
            return False
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert the events to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame containing all logged events
        """
        if not self.events:
            return pd.DataFrame()
        
        return pd.DataFrame(self.events)
    
    def calculate_correlations(self) -> Dict[str, float]:
        """
        Calculate correlations between ΨC scores and cognitive changes.
        
        Returns:
            Dictionary of correlation coefficients
        """
        df = self.get_dataframe()
        if df.empty:
            return {}
        
        correlations = {}
        
        # Correlation between ΨC score and schema changes
        if "schema_changed" in df.columns:
            correlations["psi_schema_change"] = df["psi_score"].corr(df["schema_changed"].astype(float))
        
        # Correlation between ΨC score and entropy
        if "entropy" in df.columns:
            correlations["psi_entropy"] = df["psi_score"].corr(df["entropy"])
        
        # Correlation between ΨC score and reflection occurrence
        if "reflection_occurred" in df.columns:
            correlations["psi_reflection"] = df["psi_score"].corr(df["reflection_occurred"].astype(float))
        
        return correlations
    
    def find_optimal_threshold(self) -> Dict[str, float]:
        """
        Find the optimal ΨC threshold based on empirical data.
        
        This method analyzes the relationship between ΨC scores and meaningful
        cognitive changes to determine what threshold values are most predictive
        of real changes in the system.
        
        Returns:
            Dictionary with optimal threshold values and statistics
        """
        df = self.get_dataframe()
        if df.empty or len(df) < 10:
            return {"error": "Not enough data for threshold calibration"}
        
        # Sort by ΨC score to find thresholds
        df_sorted = df.sort_values("psi_score")
        
        # Initialize results
        results = {
            "schema_change_threshold": 0.0,
            "reflection_threshold": 0.0,
            "entropy_reduction_threshold": 0.0,
            "confidence": 0.0
        }
        
        # Find threshold for schema changes
        if "schema_changed" in df.columns and df["schema_changed"].sum() > 0:
            # Find distribution of ΨC scores when schema changes occurred
            schema_change_scores = df[df["schema_changed"]]["psi_score"]
            results["schema_change_threshold"] = schema_change_scores.quantile(0.25)  # 25th percentile
        
        # Find threshold for reflection occurrences
        if "reflection_occurred" in df.columns and df["reflection_occurred"].sum() > 0:
            reflection_scores = df[df["reflection_occurred"]]["psi_score"]
            results["reflection_threshold"] = reflection_scores.quantile(0.25)
        
        # Find threshold for entropy reduction
        if "entropy" in df.columns and len(df) > 5:
            # Calculate entropy delta (negative delta = reduction)
            df["entropy_delta"] = df["entropy"].diff()
            # Find distribution of scores when entropy was reduced
            entropy_reduction_scores = df[df["entropy_delta"] < -0.01]["psi_score"]
            if not entropy_reduction_scores.empty:
                results["entropy_reduction_threshold"] = entropy_reduction_scores.quantile(0.25)
        
        # Calculate overall suggested threshold
        thresholds = [
            v for k, v in results.items() 
            if k.endswith("_threshold") and v > 0
        ]
        
        if thresholds:
            results["suggested_threshold"] = sum(thresholds) / len(thresholds)
            
            # Calculate confidence based on amount of data
            results["confidence"] = min(0.95, 0.5 + (len(df) / 200))
            
            # Suggested sigmoid slope
            if "suggested_threshold" in results:
                # Steeper slope = sharper transition
                results["suggested_sigmoid_steepness"] = 4.0 + (results["confidence"] * 6.0)
        
        return results
    
    def plot_calibration_data(self, save_path: Optional[str] = None) -> None:
        """
        Create visualizations of calibration data.
        
        Args:
            save_path: Optional path to save the visualization
        """
        df = self.get_dataframe()
        if df.empty:
            self.logger.warning("No data to plot")
            return
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Convert timestamp to relative time in minutes
        if "timestamp" in df.columns:
            start_time = df["timestamp"].min()
            df["minutes"] = (df["timestamp"] - start_time) / 60
        
        # Plot ΨC score over time
        axs[0].plot(df["minutes"], df["psi_score"], 'b-', label='ΨC Score')
        axs[0].set_ylabel('ΨC Score')
        axs[0].set_title('ΨC Calibration Data')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot entropy over time
        if "entropy" in df.columns:
            axs[1].plot(df["minutes"], df["entropy"], 'r-', label='Entropy')
            axs[1].set_ylabel('Entropy')
            axs[1].legend()
            axs[1].grid(True)
        
        # Plot reflection and schema changes
        if "reflection_occurred" in df.columns:
            reflection_times = df[df["reflection_occurred"]]["minutes"]
            axs[2].scatter(reflection_times, [0.5] * len(reflection_times), 
                         marker='^', color='green', s=100, label='Reflection')
        
        if "schema_changed" in df.columns:
            schema_change_times = df[df["schema_changed"]]["minutes"]
            axs[2].scatter(schema_change_times, [0.8] * len(schema_change_times), 
                         marker='*', color='orange', s=100, label='Schema Change')
        
        axs[2].set_xlabel('Time (minutes)')
        axs[2].set_ylabel('Events')
        axs[2].set_yticks([])
        axs[2].legend()
        axs[2].grid(True)
        
        # Add optimal threshold if available
        optimal = self.find_optimal_threshold()
        if "suggested_threshold" in optimal:
            for ax in axs:
                ax.axhline(y=optimal["suggested_threshold"], color='purple', 
                          linestyle='--', alpha=0.7,
                          label=f'Suggested Threshold: {optimal["suggested_threshold"]:.3f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved calibration plot to {save_path}")
        
        plt.show()
    
    def generate_calibration_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive calibration report with statistics and recommendations.
        
        Returns:
            Dictionary containing calibration report data
        """
        if not self.events:
            return {"error": "No calibration data available"}
        
        df = self.get_dataframe()
        
        # Basic statistics
        stats = {
            "total_events": len(df),
            "session_id": self.session_id,
            "duration_minutes": (df["timestamp"].max() - df["timestamp"].min()) / 60,
            "psi_score": {
                "mean": df["psi_score"].mean(),
                "median": df["psi_score"].median(),
                "min": df["psi_score"].min(),
                "max": df["psi_score"].max(),
                "std": df["psi_score"].std()
            }
        }
        
        # Cognitive event statistics
        if "reflection_occurred" in df.columns:
            stats["reflection_count"] = df["reflection_occurred"].sum()
        
        if "schema_changed" in df.columns:
            stats["schema_change_count"] = df["schema_changed"].sum()
        
        # Calculate correlations
        correlations = self.calculate_correlations()
        
        # Find optimal thresholds
        thresholds = self.find_optimal_threshold()
        
        # Combine into report
        report = {
            "statistics": stats,
            "correlations": correlations,
            "thresholds": thresholds,
            "timestamp": time.time(),
            "generated": datetime.now().isoformat()
        }
        
        # Save the report
        report_path = os.path.join(self.log_dir, f"{self.session_id}_report.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Saved calibration report to {report_path}")
        except Exception as e:
            self.logger.error(f"Error saving calibration report: {e}")
        
        return report 