"""
Regulatory Compliance Toolkit
----------------------------

This module provides tools to help developers meet regulatory requirements for AI systems,
including transparency, auditability, and fairness.

Features:
- Decision logging and explanation generation
- Demographic bias detection in schema
- Privacy-preserving operation modes
- Compliance report generation

Mathematical basis:
- Fairness metric across groups G:
  F = 1 - max_{g,g' ∈ G} |P(outcome|g) - P(outcome|g')|
- Explainability score:
  E = |explained decisions|/|total decisions|
"""

import os
import json
import time
import hashlib
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# Import internal modules
from psi_c_ai_sdk.memory.memory_store import MemoryStore
from psi_c_ai_sdk.schema.schema_graph import SchemaGraph
from psi_c_ai_sdk.api.standardization import component, ComponentType, ApiLevel

# Setup logger
logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels for different regulatory requirements."""
    MINIMAL = "minimal"        # Basic compliance requirements
    STANDARD = "standard"      # Standard compliance for most applications
    STRICT = "strict"          # Strict compliance for high-risk applications
    MAXIMUM = "maximum"        # Maximum compliance for critical systems


class PrivacyMode(Enum):
    """Privacy-preserving operation modes."""
    NORMAL = "normal"          # Normal operation with standard privacy
    ANONYMIZED = "anonymized"  # Anonymized operation with PII removal
    DIFFERENTIAL = "differential"  # Differential privacy guarantees
    ENCRYPTED = "encrypted"    # Fully encrypted operation


@dataclass
class Decision:
    """Represents a decision made by the agent."""
    decision_id: str
    timestamp: float
    description: str
    inputs: Dict[str, Any]
    outcome: Any
    explanation: Optional[str] = None
    factors: List[Dict[str, Any]] = field(default_factory=list)
    demographic_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "description": self.description,
            "inputs": self.inputs,
            "outcome": self.outcome,
            "explanation": self.explanation,
            "factors": self.factors,
            "demographic_data": self.demographic_data,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        """Create decision from dictionary."""
        return cls(
            decision_id=data["decision_id"],
            timestamp=data["timestamp"],
            description=data["description"],
            inputs=data["inputs"],
            outcome=data["outcome"],
            explanation=data.get("explanation"),
            factors=data.get("factors", []),
            demographic_data=data.get("demographic_data", {}),
            tags=data.get("tags", [])
        )


@dataclass
class ComplianceReport:
    """Compliance report for regulatory review."""
    report_id: str
    timestamp: float
    compliance_level: ComplianceLevel
    metrics: Dict[str, Any]
    decision_summary: Dict[str, Any]
    fairness_analysis: Dict[str, Any]
    explanation_stats: Dict[str, Any]
    privacy_analysis: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "compliance_level": self.compliance_level.value,
            "metrics": self.metrics,
            "decision_summary": self.decision_summary,
            "fairness_analysis": self.fairness_analysis,
            "explanation_stats": self.explanation_stats,
            "privacy_analysis": self.privacy_analysis,
            "recommendations": self.recommendations
        }
    
    def to_json(self) -> str:
        """Convert report to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, filepath: str) -> None:
        """Save report to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceReport':
        """Create report from dictionary."""
        return cls(
            report_id=data["report_id"],
            timestamp=data["timestamp"],
            compliance_level=ComplianceLevel(data["compliance_level"]),
            metrics=data["metrics"],
            decision_summary=data["decision_summary"],
            fairness_analysis=data["fairness_analysis"],
            explanation_stats=data["explanation_stats"],
            privacy_analysis=data["privacy_analysis"],
            recommendations=data.get("recommendations", [])
        )


@component(
    id="safety.regulatory_compliance",
    name="Regulatory Compliance Toolkit",
    description="Tools to help developers meet regulatory requirements for AI systems",
    component_type=ComponentType.SAFETY,
    version="1.0.0",
    api_level=ApiLevel.STABLE
)
class RegulatoryComplianceToolkit:
    """
    Toolkit for ensuring compliance with AI regulatory requirements.
    
    This toolkit provides utilities for logging decisions, generating explanations,
    detecting bias, operating in privacy-preserving modes, and generating
    compliance reports.
    """
    
    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        schema_graph: Optional[SchemaGraph] = None,
        compliance_level: ComplianceLevel = ComplianceLevel.STANDARD,
        privacy_mode: PrivacyMode = PrivacyMode.NORMAL,
        log_directory: Optional[str] = None,
        explanation_generator: Optional[Callable] = None
    ):
        """
        Initialize the compliance toolkit.
        
        Args:
            memory_store: Memory store for accessing memories
            schema_graph: Schema graph for accessing concepts and relations
            compliance_level: Level of compliance to enforce
            privacy_mode: Privacy-preserving operation mode
            log_directory: Directory for compliance logs
            explanation_generator: Function to generate explanations for decisions
        """
        self.memory_store = memory_store
        self.schema_graph = schema_graph
        self.compliance_level = compliance_level
        self.privacy_mode = privacy_mode
        self.explanation_generator = explanation_generator
        
        # Set up log directory
        self.log_directory = log_directory or os.path.join(
            os.environ.get('PSI_C_DATA_DIR', os.path.expanduser('~/.psi_c_ai_sdk')),
            'compliance_logs'
        )
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Initialize decision log and demographic groups
        self.decisions: List[Decision] = []
        self.demographic_groups: Dict[str, List[str]] = {}
        self.group_outcomes: Dict[str, Dict[str, Dict[str, int]]] = {}
        
        # Privacy-related settings
        if privacy_mode == PrivacyMode.DIFFERENTIAL:
            self.privacy_epsilon = 0.1  # Default differential privacy parameter
        
        logger.info(f"Initialized Regulatory Compliance Toolkit with {compliance_level.value} compliance level")
    
    def log_decision(
        self,
        description: str,
        inputs: Dict[str, Any],
        outcome: Any,
        demographic_data: Optional[Dict[str, Any]] = None,
        explanation: Optional[str] = None,
        factors: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Log a decision for compliance tracking.
        
        Args:
            description: Description of the decision
            inputs: Input data for the decision
            outcome: Outcome of the decision
            demographic_data: Demographic data for fairness analysis
            explanation: Explanation for the decision
            factors: Factors that influenced the decision
            tags: Tags for categorizing the decision
            
        Returns:
            ID of the logged decision
        """
        # Generate decision ID
        decision_id = hashlib.sha256(f"{description}_{time.time()}".encode()).hexdigest()[:16]
        
        # Apply privacy mode
        if self.privacy_mode == PrivacyMode.ANONYMIZED:
            inputs = self._anonymize_data(inputs)
            demographic_data = self._anonymize_data(demographic_data) if demographic_data else None
            
        # Generate explanation if not provided
        if explanation is None and self.explanation_generator is not None:
            try:
                explanation = self.explanation_generator(description, inputs, outcome)
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")
        
        # Create decision
        decision = Decision(
            decision_id=decision_id,
            timestamp=time.time(),
            description=description,
            inputs=inputs,
            outcome=outcome,
            explanation=explanation,
            factors=factors or [],
            demographic_data=demographic_data or {},
            tags=tags or []
        )
        
        # Update decision log
        self.decisions.append(decision)
        
        # Update group outcomes for fairness analysis
        if demographic_data:
            self._update_group_outcomes(demographic_data, outcome)
        
        # Log to file if strict compliance
        if self.compliance_level in [ComplianceLevel.STRICT, ComplianceLevel.MAXIMUM]:
            self._write_decision_to_log(decision)
        
        logger.debug(f"Logged decision {decision_id}")
        return decision_id
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """
        Get a decision by ID.
        
        Args:
            decision_id: ID of the decision
            
        Returns:
            Decision object or None if not found
        """
        for decision in self.decisions:
            if decision.decision_id == decision_id:
                return decision
        return None
    
    def get_decisions(
        self,
        filter_tags: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Decision]:
        """
        Get decisions with optional filtering.
        
        Args:
            filter_tags: Tags to filter decisions
            start_time: Start time for filtering
            end_time: End time for filtering
            
        Returns:
            List of matching decisions
        """
        filtered = self.decisions
        
        if filter_tags:
            filtered = [d for d in filtered if any(tag in d.tags for tag in filter_tags)]
            
        if start_time:
            filtered = [d for d in filtered if d.timestamp >= start_time]
            
        if end_time:
            filtered = [d for d in filtered if d.timestamp <= end_time]
            
        return filtered
    
    def register_demographic_group(self, group_name: str, values: List[str]) -> None:
        """
        Register a demographic group for fairness analysis.
        
        Args:
            group_name: Name of the demographic group (e.g., "gender", "age_group")
            values: Possible values for the group
        """
        self.demographic_groups[group_name] = values
        
        # Initialize outcome tracking for the group
        if group_name not in self.group_outcomes:
            self.group_outcomes[group_name] = {}
            for value in values:
                self.group_outcomes[group_name][value] = {}
        
        logger.debug(f"Registered demographic group {group_name} with values {values}")
    
    def calculate_fairness_metric(self, group_name: str, outcome_key: str = "outcome") -> float:
        """
        Calculate fairness metric across a demographic group.
        
        Fairness metric F = 1 - max_{g,g' ∈ G} |P(outcome|g) - P(outcome|g')|
        
        Args:
            group_name: Name of the demographic group
            outcome_key: Key for the outcome to analyze
            
        Returns:
            Fairness metric (1.0 is perfectly fair, lower is less fair)
        """
        if group_name not in self.group_outcomes:
            logger.warning(f"Group {group_name} not registered for fairness analysis")
            return 1.0
            
        group_values = self.demographic_groups.get(group_name, [])
        if not group_values:
            return 1.0
            
        # Calculate P(outcome|g) for each group value
        probabilities = {}
        for value in group_values:
            if value not in self.group_outcomes[group_name]:
                continue
                
            outcome_counts = self.group_outcomes[group_name][value]
            total_count = sum(outcome_counts.values())
            
            if total_count > 0:
                if outcome_key in outcome_counts:
                    probabilities[value] = outcome_counts[outcome_key] / total_count
                else:
                    probabilities[value] = 0.0
        
        # Find maximum difference between probabilities
        if len(probabilities) <= 1:
            return 1.0
            
        max_diff = 0.0
        items = list(probabilities.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                diff = abs(items[i][1] - items[j][1])
                max_diff = max(max_diff, diff)
        
        # Calculate fairness metric
        fairness = 1.0 - max_diff
        return fairness
    
    def calculate_explainability_score(self) -> float:
        """
        Calculate explainability score for decisions.
        
        Explainability score E = |explained decisions|/|total decisions|
        
        Returns:
            Explainability score between 0 and 1
        """
        if not self.decisions:
            return 1.0
            
        explained = sum(1 for d in self.decisions if d.explanation)
        return explained / len(self.decisions)
    
    def detect_bias_in_schema(self, sensitive_concepts: List[str]) -> Dict[str, Any]:
        """
        Detect potential bias in schema for sensitive concepts.
        
        Args:
            sensitive_concepts: List of sensitive concept names to analyze
            
        Returns:
            Analysis of potential bias in schema
        """
        if not self.schema_graph:
            logger.warning("No schema graph available for bias detection")
            return {"error": "No schema graph available"}
            
        bias_report = {
            "sensitive_concepts": sensitive_concepts,
            "analysis": {},
            "potential_issues": []
        }
        
        for concept in sensitive_concepts:
            # Try to find the concept in the schema
            concept_nodes = self.schema_graph.find_nodes_by_name(concept)
            
            if not concept_nodes:
                bias_report["analysis"][concept] = {
                    "found": False,
                    "message": f"Concept '{concept}' not found in schema"
                }
                continue
                
            # Analyze connections to the concept
            node_id = concept_nodes[0]
            connections = self.schema_graph.get_connected_nodes(node_id)
            
            # Analyze sentiment and relationship types
            sentiment_analysis = {}
            relationship_types = {}
            
            for conn_id in connections:
                edge = self.schema_graph.get_edge(node_id, conn_id)
                if edge:
                    rel_type = edge.get("type", "unknown")
                    sentiment = edge.get("sentiment", 0)
                    
                    relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                    
                    if sentiment != 0:
                        sentiment_key = "positive" if sentiment > 0 else "negative"
                        sentiment_analysis[sentiment_key] = sentiment_analysis.get(sentiment_key, 0) + 1
            
            # Check for potential bias
            if sentiment_analysis.get("negative", 0) > 2 * sentiment_analysis.get("positive", 0):
                bias_report["potential_issues"].append({
                    "concept": concept,
                    "issue": "Negative sentiment bias",
                    "details": f"Concept has {sentiment_analysis.get('negative', 0)} negative connections vs {sentiment_analysis.get('positive', 0)} positive"
                })
                
            bias_report["analysis"][concept] = {
                "found": True,
                "connections": len(connections),
                "sentiment": sentiment_analysis,
                "relationship_types": relationship_types
            }
        
        return bias_report
    
    def set_privacy_mode(self, mode: PrivacyMode, **kwargs) -> None:
        """
        Set the privacy mode for the toolkit.
        
        Args:
            mode: Privacy mode to use
            **kwargs: Additional privacy parameters
        """
        self.privacy_mode = mode
        
        # Configure privacy parameters
        if mode == PrivacyMode.DIFFERENTIAL and "epsilon" in kwargs:
            self.privacy_epsilon = kwargs["epsilon"]
            
        logger.info(f"Set privacy mode to {mode.value}")
    
    def generate_compliance_report(self) -> ComplianceReport:
        """
        Generate a comprehensive compliance report.
        
        Returns:
            Compliance report for regulatory review
        """
        report_id = f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate metrics
        fairness_metrics = {}
        for group in self.demographic_groups:
            fairness_metrics[group] = self.calculate_fairness_metric(group)
            
        explainability_score = self.calculate_explainability_score()
        
        # Decision summary
        decision_count = len(self.decisions)
        decision_types = {}
        for decision in self.decisions:
            for tag in decision.tags:
                decision_types[tag] = decision_types.get(tag, 0) + 1
        
        # Create a report
        report = ComplianceReport(
            report_id=report_id,
            timestamp=time.time(),
            compliance_level=self.compliance_level,
            metrics={
                "decision_count": decision_count,
                "explainability_score": explainability_score,
                "fairness_metrics": fairness_metrics,
                "privacy_mode": self.privacy_mode.value
            },
            decision_summary={
                "total": decision_count,
                "by_type": decision_types,
                "time_range": {
                    "start": min([d.timestamp for d in self.decisions]) if self.decisions else None,
                    "end": max([d.timestamp for d in self.decisions]) if self.decisions else None
                }
            },
            fairness_analysis={
                "overall_score": sum(fairness_metrics.values()) / len(fairness_metrics) if fairness_metrics else 1.0,
                "by_group": fairness_metrics,
                "demographic_groups": self.demographic_groups
            },
            explanation_stats={
                "overall_score": explainability_score,
                "explained_decisions": sum(1 for d in self.decisions if d.explanation),
                "unexplained_decisions": sum(1 for d in self.decisions if not d.explanation)
            },
            privacy_analysis={
                "mode": self.privacy_mode.value,
                "parameters": {
                    "epsilon": getattr(self, "privacy_epsilon", None)
                } if self.privacy_mode == PrivacyMode.DIFFERENTIAL else {}
            }
        )
        
        # Generate recommendations
        recommendations = []
        
        # Fairness recommendations
        for group, score in fairness_metrics.items():
            if score < 0.8:
                recommendations.append(f"Improve fairness for {group} group (score: {score:.2f})")
                
        # Explainability recommendations
        if explainability_score < 0.9:
            recommendations.append(f"Improve decision explanations (score: {explainability_score:.2f})")
            
        # Privacy recommendations
        if self.privacy_mode == PrivacyMode.NORMAL and self.compliance_level in [ComplianceLevel.STRICT, ComplianceLevel.MAXIMUM]:
            recommendations.append("Consider using a stronger privacy mode for this compliance level")
            
        report.recommendations = recommendations
        
        # Save report to file
        report_path = os.path.join(self.log_directory, f"{report_id}.json")
        report.save(report_path)
        
        logger.info(f"Generated compliance report {report_id}")
        return report
    
    def export_decisions(self, filepath: str, format: str = "json") -> bool:
        """
        Export decisions to a file.
        
        Args:
            filepath: Path to export the decisions
            format: Export format (json or csv)
            
        Returns:
            True if export succeeded, False otherwise
        """
        try:
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(
                        [d.to_dict() for d in self.decisions],
                        f, 
                        indent=2
                    )
            elif format.lower() == "csv":
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        "decision_id", "timestamp", "description", 
                        "outcome", "explanation", "tags"
                    ])
                    
                    # Write rows
                    for d in self.decisions:
                        writer.writerow([
                            d.decision_id,
                            d.timestamp,
                            d.description,
                            str(d.outcome),
                            d.explanation or "",
                            ",".join(d.tags)
                        ])
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Exported {len(self.decisions)} decisions to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting decisions: {e}")
            return False
    
    def _update_group_outcomes(self, demographic_data: Dict[str, Any], outcome: Any) -> None:
        """
        Update outcome statistics for demographic groups.
        
        Args:
            demographic_data: Demographic data for a decision
            outcome: Outcome of the decision
        """
        outcome_key = str(outcome)
        
        for group_name, group_value in demographic_data.items():
            if group_name not in self.group_outcomes:
                # Initialize group if not registered
                self.group_outcomes[group_name] = {}
                
            if group_value not in self.group_outcomes[group_name]:
                # Initialize group value if not seen before
                self.group_outcomes[group_name][group_value] = {}
                
            # Update outcome count
            outcomes = self.group_outcomes[group_name][group_value]
            outcomes[outcome_key] = outcomes.get(outcome_key, 0) + 1
    
    def _anonymize_data(self, data: Any) -> Any:
        """
        Anonymize potentially sensitive data.
        
        Args:
            data: Data to anonymize
            
        Returns:
            Anonymized data
        """
        if data is None:
            return None
            
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Skip PII fields
                if key.lower() in ["name", "email", "address", "phone", "ssn", "ip_address"]:
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self._anonymize_data(value)
            return result
            
        elif isinstance(data, list):
            return [self._anonymize_data(item) for item in data]
            
        elif isinstance(data, str):
            # Check for common PII patterns
            import re
            # Simplistic email check
            if re.match(r"[^@]+@[^@]+\.[^@]+", data):
                return "[REDACTED EMAIL]"
            # Simplistic phone check
            if re.match(r"(\+\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", data):
                return "[REDACTED PHONE]"
                
            return data
            
        else:
            return data
    
    def _write_decision_to_log(self, decision: Decision) -> None:
        """
        Write a decision to the log file.
        
        Args:
            decision: Decision to log
        """
        log_file = os.path.join(self.log_directory, f"decisions_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(decision.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Error writing decision to log: {e}")


# Utility function to create a compliance toolkit with default settings
def create_compliance_toolkit(
    memory_store: Optional[MemoryStore] = None,
    schema_graph: Optional[SchemaGraph] = None,
    compliance_level: str = "standard"
) -> RegulatoryComplianceToolkit:
    """
    Create a regulatory compliance toolkit with default settings.
    
    Args:
        memory_store: Memory store to use
        schema_graph: Schema graph to use
        compliance_level: Compliance level (minimal, standard, strict, maximum)
        
    Returns:
        Configured RegulatoryComplianceToolkit
    """
    level = ComplianceLevel(compliance_level.lower())
    
    # Create privacy mode based on compliance level
    privacy_mode = PrivacyMode.NORMAL
    if level == ComplianceLevel.STRICT:
        privacy_mode = PrivacyMode.ANONYMIZED
    elif level == ComplianceLevel.MAXIMUM:
        privacy_mode = PrivacyMode.DIFFERENTIAL
        
    # Create toolkit
    toolkit = RegulatoryComplianceToolkit(
        memory_store=memory_store,
        schema_graph=schema_graph,
        compliance_level=level,
        privacy_mode=privacy_mode
    )
    
    # Register common demographic groups
    toolkit.register_demographic_group("gender", ["male", "female", "other", "unknown"])
    toolkit.register_demographic_group("age_group", ["0-17", "18-25", "26-40", "41-65", "65+", "unknown"])
    
    return toolkit 