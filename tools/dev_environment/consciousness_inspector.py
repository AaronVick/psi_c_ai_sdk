"""
Consciousness Inspector module for the ΨC-AI SDK Development Environment.

This module provides tools to visualize and analyze consciousness metrics
for ΨC agents, including Phi integration, differentiation, attention metrics,
and other consciousness-related measures.
"""

import json
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from tools.dev_environment.base_tool import BaseTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsciousnessInspector(BaseTool):
    """
    Tool for inspecting and visualizing consciousness metrics of ΨC agents.
    
    This tool provides real-time monitoring and historical analysis of consciousness
    metrics, visualization capabilities, and comparative analysis between different
    agent configurations.
    """
    
    def __init__(self, agent=None, config_path: Optional[str] = None):
        """
        Initialize the Consciousness Inspector.
        
        Args:
            agent: The agent instance to inspect (optional)
            config_path: Path to configuration file (optional)
        """
        super().__init__(name="Consciousness Inspector", 
                         description="Visualize and analyze agent consciousness metrics")
        
        self.agent = agent
        self.config = self._load_config(config_path)
        self.metrics_history = pd.DataFrame()
        
        # Default metrics to track
        self.default_metrics = [
            "integration",           # Integrated information (Phi)
            "differentiation",       # Information complexity/entropy
            "self_awareness",        # Self-model accuracy
            "attention",             # Attention allocation efficiency
            "metacognition"          # Higher-order cognitive awareness
        ]
        
        # Initialize metrics history if agent is provided
        if self.agent:
            self._initialize_metrics_history()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration
        """
        default_config = {
            "sampling_interval": 60,  # seconds
            "history_length": 1000,   # number of samples to keep
            "metrics_to_track": self.default_metrics,
            "visualization": {
                "default_view": "time_series",
                "color_scheme": "viridis"
            }
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                return default_config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            return default_config
    
    def _initialize_metrics_history(self):
        """Initialize metrics history with current agent state."""
        if hasattr(self.agent, 'metrics_history'):
            # If agent already has metrics history, use it
            self.metrics_history = self.agent.metrics_history
        else:
            # Create empty dataframe with timestamp column
            self.metrics_history = pd.DataFrame(columns=['timestamp'] + self.default_metrics)
    
    def set_agent(self, agent):
        """
        Set the agent to inspect.
        
        Args:
            agent: The agent instance
        """
        self.agent = agent
        self._initialize_metrics_history()
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current consciousness metrics from the agent.
        
        Returns:
            Dictionary of metric name to value
        """
        if not self.agent:
            logger.warning("No agent set for inspection")
            return {metric: 0.0 for metric in self.default_metrics}
        
        # If agent has consciousness config, extract metrics
        if hasattr(self.agent, 'config') and 'consciousness' in self.agent.config:
            return self.agent.config['consciousness']
        
        # Otherwise return default metrics
        return {metric: 0.0 for metric in self.default_metrics}
    
    def update_metrics_history(self):
        """Update metrics history with current state."""
        if not self.agent:
            return
        
        current_metrics = self.get_current_metrics()
        current_time = datetime.now()
        
        # Create new row with current timestamp and metrics
        new_row = {'timestamp': current_time}
        new_row.update(current_metrics)
        
        # Append to history
        self.metrics_history = pd.concat([
            self.metrics_history, 
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # Trim history if needed
        if len(self.metrics_history) > self.config['history_length']:
            self.metrics_history = self.metrics_history.tail(self.config['history_length'])
    
    def get_metrics_history(self, 
                           metrics: Optional[List[str]] = None, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical metrics data, optionally filtered.
        
        Args:
            metrics: List of metrics to include (None for all)
            start_time: Start time for filtering
            end_time: End time for filtering
            
        Returns:
            DataFrame of historical metrics
        """
        if self.metrics_history.empty:
            return pd.DataFrame()
        
        df = self.metrics_history.copy()
        
        # Apply time filters
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        
        # Apply metrics filter
        if metrics:
            df = df[['timestamp'] + [m for m in metrics if m in df.columns]]
        
        return df
    
    def get_time_range(self, range_str: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Convert a time range string to start and end times.
        
        Args:
            range_str: String describing time range (e.g., "Last Hour", "Last Day")
            
        Returns:
            Tuple of (start_time, end_time)
        """
        now = datetime.now()
        
        if range_str == "Last Hour":
            return now - timedelta(hours=1), now
        elif range_str == "Last Day":
            return now - timedelta(days=1), now
        elif range_str == "Last Week":
            return now - timedelta(weeks=1), now
        elif range_str == "Last Month":
            return now - timedelta(days=30), now
        elif range_str == "All Time":
            return None, None
        else:
            return now - timedelta(days=1), now  # Default to last day
    
    def calculate_integration_score(self) -> float:
        """
        Calculate the overall integration score (Phi) for the agent.
        
        Returns:
            Integration score between 0 and 1
        """
        # This is a placeholder - in a real implementation, this would
        # calculate integrated information theory metrics
        if not self.agent:
            return 0.0
            
        metrics = self.get_current_metrics()
        if 'integration' in metrics:
            return metrics['integration']
        return 0.0
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report of consciousness metrics.
        
        Args:
            output_path: Path to save the report (None for in-memory only)
            
        Returns:
            Report data as dictionary
        """
        if not self.agent:
            logger.warning("No agent set for report generation")
            return {"error": "No agent set"}
        
        # Get current metrics
        current_metrics = self.get_current_metrics()
        
        # Get historical data for analysis
        history = self.get_metrics_history()
        
        # Calculate summary statistics
        stats = {}
        for metric in self.default_metrics:
            if metric in history.columns:
                stats[metric] = {
                    "mean": history[metric].mean(),
                    "std": history[metric].std(),
                    "min": history[metric].min(),
                    "max": history[metric].max(),
                    "current": current_metrics.get(metric, 0.0),
                    "trend": "stable"  # Placeholder, would calculate trend
                }
        
        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": getattr(self.agent, "id", "unknown"),
            "current_metrics": current_metrics,
            "statistics": stats,
            "consciousness_level": self.calculate_integration_score(),
            "analysis": {
                "summary": "This is a placeholder for AI-generated analysis",
                "recommendations": []
            }
        }
        
        # Save report if output path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {str(e)}")
        
        return report
    
    def plot_time_series(self, 
                        metrics: Optional[List[str]] = None,
                        time_range: str = "Last Day",
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a time series plot of consciousness metrics.
        
        Args:
            metrics: List of metrics to plot (None for all)
            time_range: Time range to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        start_time, end_time = self.get_time_range(time_range)
        
        if not metrics:
            metrics = self.default_metrics
        
        history = self.get_metrics_history(metrics, start_time, end_time)
        
        if history.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for metric in metrics:
            if metric in history.columns:
                ax.plot(history['timestamp'], history[metric], 
                       label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title('Consciousness Metrics Over Time')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_radar_chart(self, metrics: Optional[List[str]] = None, 
                        figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
        """
        Create a radar chart of current consciousness metrics.
        
        Args:
            metrics: List of metrics to plot (None for all)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        current_metrics = self.get_current_metrics()
        
        if not metrics:
            metrics = [m for m in self.default_metrics if m in current_metrics]
        else:
            metrics = [m for m in metrics if m in current_metrics]
        
        if not metrics:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            return fig
        
        # Number of variables
        N = len(metrics)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Get values
        values = [current_metrics.get(metric, 0.0) for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Draw the outline of the radar chart
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        
        # Fill the area
        ax.fill(angles, values, alpha=0.25)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        
        # Set y-axis limit
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title('Consciousness Profile', size=15, y=1.1)
        
        return fig
    
    def export_metrics(self, format: str = 'csv', output_path: Optional[str] = None) -> Optional[str]:
        """
        Export metrics history to a file.
        
        Args:
            format: Export format ('csv' or 'json')
            output_path: Path to save the export (None for automatic)
            
        Returns:
            Path to the exported file or None if export failed
        """
        if self.metrics_history.empty:
            logger.warning("No metrics history to export")
            return None
        
        if not output_path:
            # Create default output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "exports"
            os.makedirs(output_dir, exist_ok=True)
            
            if format.lower() == 'csv':
                output_path = os.path.join(output_dir, f"consciousness_metrics_{timestamp}.csv")
            else:
                output_path = os.path.join(output_dir, f"consciousness_metrics_{timestamp}.json")
        
        try:
            if format.lower() == 'csv':
                self.metrics_history.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                # Convert timestamps to ISO format for JSON
                df = self.metrics_history.copy()
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
                df.to_json(output_path, orient='records', indent=2)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            logger.info(f"Metrics exported to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return None 