#!/usr/bin/env python3
"""
Schema CLI - Command-line interface for schema graph management

This CLI provides tools for building, analyzing, and managing schema graphs,
including schema versioning, health monitoring, and deployment readiness checks.

Usage:
    schema_cli build-graph --memory-store MEMORY_STORE
    schema_cli detect-clusters --eps EPS --min-samples MIN_SAMPLES
    schema_cli generate-concepts
    schema_cli visualize --output-file FILE [--show]
    schema_cli export --output-file FILE
    schema_cli import --input-file FILE
    schema_cli report --output-file FILE
    schema_cli version --type TYPE [--message MESSAGE]
    schema_cli deploy --environment ENV [--version VERSION]
    schema_cli status
    schema_cli health
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import schema components
try:
    from tools.dev_environment.memory_schema_integration import MemorySchemaIntegration
    from tools.dev_environment.memory_sandbox import MemorySandbox
    from tools.dev_environment.schema_versioning import SchemaVersionManager, SchemaDeploymentManager
    from psi_c_ai_sdk.memory.memory_store import MemoryStore
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for schema graph management"
    )
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--memory-store", "-m",
        type=str,
        help="Path to memory store file"
    )
    common_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="schema_output",
        help="Output directory for generated files"
    )
    common_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build graph command
    build_parser = subparsers.add_parser(
        "build-graph", 
        parents=[common_parser],
        help="Build schema graph from memories"
    )
    
    # Detect clusters command
    cluster_parser = subparsers.add_parser(
        "detect-clusters", 
        parents=[common_parser],
        help="Detect memory clusters in the graph"
    )
    cluster_parser.add_argument(
        "--eps", "-e",
        type=float,
        default=0.5,
        help="Maximum distance between samples for DBSCAN clustering"
    )
    cluster_parser.add_argument(
        "--min-samples", "-s",
        type=int,
        default=2,
        help="Minimum samples in neighborhood for DBSCAN clustering"
    )
    
    # Generate concepts command
    concept_parser = subparsers.add_parser(
        "generate-concepts", 
        parents=[common_parser],
        help="Generate concept suggestions from clusters"
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser(
        "visualize", 
        parents=[common_parser],
        help="Visualize the schema graph"
    )
    viz_parser.add_argument(
        "--output-file", "-f",
        type=str,
        default="schema_graph.png",
        help="Output file name for the visualization"
    )
    viz_parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Show the visualization as well as saving it"
    )
    
    # Export command
    export_parser = subparsers.add_parser(
        "export", 
        parents=[common_parser],
        help="Export the schema graph to a file"
    )
    export_parser.add_argument(
        "--output-file", "-f",
        type=str,
        default="schema_export.json",
        help="Output file name for the export"
    )
    
    # Import command
    import_parser = subparsers.add_parser(
        "import", 
        parents=[common_parser],
        help="Import a schema graph from a file"
    )
    import_parser.add_argument(
        "--input-file", "-i",
        type=str,
        required=True,
        help="Input file to import"
    )
    
    # Report command
    report_parser = subparsers.add_parser(
        "report", 
        parents=[common_parser],
        help="Generate a knowledge report"
    )
    report_parser.add_argument(
        "--output-file", "-f",
        type=str,
        default="knowledge_report.json",
        help="Output file name for the report"
    )
    
    # Version command (new)
    version_parser = subparsers.add_parser(
        "version", 
        parents=[common_parser],
        help="Create a new schema version"
    )
    version_parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["major", "minor", "patch"],
        default="patch",
        help="Type of version increment"
    )
    version_parser.add_argument(
        "--message", "-m",
        type=str,
        default="",
        help="Version commit message"
    )
    version_parser.add_argument(
        "--schema-dir", "-d",
        type=str,
        default="schema_versions",
        help="Directory for storing schema versions"
    )
    
    # Deploy command (new)
    deploy_parser = subparsers.add_parser(
        "deploy", 
        parents=[common_parser],
        help="Deploy schema to an environment"
    )
    deploy_parser.add_argument(
        "--environment", "-e",
        type=str,
        choices=["staging", "production"],
        required=True,
        help="Target environment for deployment"
    )
    deploy_parser.add_argument(
        "--version", "-v",
        type=str,
        help="Schema version to deploy (defaults to current version)"
    )
    deploy_parser.add_argument(
        "--schema-dir", "-d",
        type=str,
        default="schema_versions",
        help="Directory for storing schema versions"
    )
    
    # Status command (new)
    status_parser = subparsers.add_parser(
        "status", 
        parents=[common_parser],
        help="Get schema deployment status"
    )
    status_parser.add_argument(
        "--schema-dir", "-d",
        type=str,
        default="schema_versions",
        help="Directory for storing schema versions"
    )
    
    # Health command (new)
    health_parser = subparsers.add_parser(
        "health", 
        parents=[common_parser],
        help="Get schema health metrics"
    )
    health_parser.add_argument(
        "--schema-dir", "-d",
        type=str,
        default="schema_versions",
        help="Directory for storing schema versions"
    )
    health_parser.add_argument(
        "--report-file", "-r",
        type=str,
        help="Output file for health report"
    )
    
    return parser.parse_args()

def initialize_environment(args):
    """Initialize the environment and components."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize memory store
    memory_store = MemoryStore()
    
    # Load memories if a memory store file is provided
    if args.memory_store and os.path.exists(args.memory_store):
        try:
            with open(args.memory_store, 'r') as f:
                memories_data = json.load(f)
            
            # Load memories from the file
            logger.info(f"Loading memories from {args.memory_store}")
            # This is a simplified example - in a real implementation, you would
            # properly deserialize the memories with all their attributes
            for memory_data in memories_data.get('memories', []):
                memory_store.add_memory_from_dict(memory_data)
            
            logger.info(f"Loaded {len(memory_store.memories)} memories")
        except Exception as e:
            logger.error(f"Error loading memory store: {e}")
            sys.exit(1)
    
    # Initialize memory sandbox
    snapshot_dir = os.path.join(args.output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    sandbox = MemorySandbox(memory_store=memory_store, snapshot_dir=snapshot_dir)
    
    # Initialize schema integration
    schema = MemorySchemaIntegration(sandbox)
    
    return sandbox, schema

def build_graph(args, sandbox, schema):
    """Build a schema graph from memories."""
    logger.info("Building schema graph from memories...")
    
    # Build the schema graph
    schema.build_schema_graph()
    
    # Get and print statistics
    stats = schema.calculate_schema_statistics()
    logger.info(f"Schema graph built with {stats['node_count']} nodes and {stats['edge_count']} edges")
    
    if 'memory_type_distribution' in stats:
        logger.info("Memory type distribution:")
        for memory_type, count in stats['memory_type_distribution'].items():
            logger.info(f"  - {memory_type}: {count}")
    
    return True

def detect_clusters(args, sandbox, schema):
    """Detect memory clusters in the graph."""
    logger.info("Detecting memory clusters...")
    
    # Detect clusters
    clusters = schema.detect_memory_clusters(
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    # Print cluster info
    logger.info(f"Detected {len(clusters)} clusters of related memories:")
    
    for cluster_id, cluster_data in clusters.items():
        memory_types = cluster_data['memory_types']
        dominant_type = max(memory_types.items(), key=lambda x: x[1])[0] if memory_types else "unknown"
        
        logger.info(f"\nCluster {cluster_id}:")
        logger.info(f"  - Size: {cluster_data['size']} memories")
        logger.info(f"  - Dominant type: {dominant_type}")
        logger.info(f"  - Average importance: {cluster_data['avg_importance']:.2f}")
        
        if args.verbose:
            logger.info("  - Sample contents:")
            for content in cluster_data['contents'][:3]:
                logger.info(f"    * {content[:50]}..." if len(content) > 50 else f"    * {content}")
    
    # Save clusters to file
    clusters_file = os.path.join(args.output_dir, "memory_clusters.json")
    with open(clusters_file, 'w') as f:
        json.dump(clusters, f, indent=2)
    
    logger.info(f"Saved clusters to {clusters_file}")
    
    return True

def generate_concepts(args, sandbox, schema):
    """Generate concept suggestions from clusters."""
    logger.info("Generating concept suggestions...")
    
    # Generate concepts
    concepts = schema.generate_concept_suggestions()
    
    # Print concept info
    logger.info(f"Generated {len(concepts)} concept suggestions:")
    
    for concept_id, concept_data in concepts.items():
        logger.info(f"\nConcept: {concept_data['concept_name']}")
        logger.info(f"  - Memory type: {concept_data['dominant_type']}")
        logger.info(f"  - Keywords: {', '.join(concept_data['keywords'])}")
        logger.info(f"  - Associated with {len(concept_data['memory_ids'])} memories")
        logger.info(f"  - Importance: {concept_data['importance']:.2f}")
    
    # Save concepts to file
    concepts_file = os.path.join(args.output_dir, "concept_suggestions.json")
    with open(concepts_file, 'w') as f:
        json.dump(concepts, f, indent=2)
    
    logger.info(f"Saved concept suggestions to {concepts_file}")
    
    return True

def visualize_graph(args, sandbox, schema):
    """Visualize the schema graph."""
    logger.info("Visualizing schema graph...")
    
    # Create output path
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Visualize the graph
    schema.visualize_schema_graph(output_path=output_path, show=args.show)
    
    logger.info(f"Schema graph visualization saved to {output_path}")
    
    return True

def export_graph(args, sandbox, schema):
    """Export the schema graph to a file."""
    logger.info("Exporting schema graph...")
    
    # Create output path
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Export the schema snapshot
    schema.export_schema_snapshot(output_path)
    
    logger.info(f"Schema graph exported to {output_path}")
    
    return True

def import_graph(args, sandbox, schema):
    """Import a schema graph from a file."""
    logger.info(f"Importing schema graph from {args.input_file}...")
    
    # Import the schema snapshot
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return False
    
    success = schema.import_schema_snapshot(input_path=args.input_file)
    
    if success:
        logger.info("Schema graph imported successfully")
        
        # Print statistics
        stats = schema.calculate_schema_statistics()
        logger.info(f"Imported graph has {stats['node_count']} nodes and {stats['edge_count']} edges")
    else:
        logger.error("Failed to import schema graph")
        
    return success

def generate_report(args, sandbox, schema):
    """Generate a knowledge report."""
    logger.info("Generating knowledge report...")
    
    # Create output path
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Get schema statistics
    stats = schema.calculate_schema_statistics()
    
    # Get cognitive health metrics
    health_metrics = {
        "cognitive_debt": schema.calculate_cognitive_debt(),
        "schema_health": schema.calculate_schema_health(),
        "complexity_budget": schema.calculate_complexity_budget(),
        "energy_usage": schema.calculate_memory_energy_usage()
    }
    
    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "health_metrics": health_metrics
    }
    
    # Save report to file
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Knowledge report saved to {output_path}")
    
    # Print summary
    logger.info("\nKnowledge Report Summary:")
    logger.info(f"  - Nodes: {stats['node_count']}")
    logger.info(f"  - Edges: {stats['edge_count']}")
    logger.info(f"  - Cognitive Debt: {health_metrics['cognitive_debt']:.2f}")
    logger.info(f"  - Schema Health: {health_metrics['schema_health']:.2f}")
    logger.info(f"  - Complexity Budget: {health_metrics['complexity_budget']:.2f}")
    
    return True

def create_version(args, sandbox, schema):
    """Create a new schema version."""
    logger.info(f"Creating new {args.type} schema version...")
    
    # Initialize schema version manager
    version_manager = SchemaVersionManager(schema_dir=args.schema_dir)
    
    # Create new version
    version = version_manager.create_version(
        schema_graph=schema.graph,
        schema_integration=schema,
        version_type=args.type,
        commit_message=args.message
    )
    
    logger.info(f"Created schema version {version}")
    
    # Print version info
    if version_manager.version_history:
        latest_version = version_manager.version_history[-1]
        logger.info(f"Version: {version}")
        logger.info(f"Timestamp: {latest_version['timestamp']}")
        logger.info(f"Message: {latest_version['message']}")
        
        # Print health metrics
        if 'health_metrics' in latest_version:
            health = latest_version['health_metrics']
            logger.info("\nHealth Metrics:")
            logger.info(f"  - Schema Health: {health.get('schema_health', 'N/A')}")
            logger.info(f"  - Cognitive Debt: {health.get('cognitive_debt', 'N/A')}")
            logger.info(f"  - Complexity Budget: {health.get('complexity_budget', 'N/A')}")
    
    return True

def deploy_schema(args, sandbox, schema):
    """Deploy schema to an environment."""
    logger.info(f"Deploying schema to {args.environment}...")
    
    # Initialize schema version manager and deployment manager
    version_manager = SchemaVersionManager(schema_dir=args.schema_dir)
    deployment_manager = SchemaDeploymentManager(version_manager)
    
    # Deploy schema
    result = deployment_manager.deploy_to_environment(
        environment=args.environment,
        version=args.version
    )
    
    if result["status"] == "success":
        logger.info(f"Successfully deployed schema version {result['version']} to {result['environment']}")
    else:
        logger.error(f"Failed to deploy schema: {result['message']}")
        
        # Print readiness info if available
        if 'readiness' in result:
            readiness = result['readiness']
            logger.info(f"\nSchema Readiness: {readiness['readiness_score']:.2f}")
            logger.info(f"Status: {readiness['status']}")
            logger.info(f"Message: {readiness['message']}")
            
            logger.info("\nFailing Checks:")
            for check in readiness['checklist']:
                if check['status'] == 'fail':
                    logger.info(f"  - {check['check']}: {check['value']}")
    
    return result["status"] == "success"

def show_status(args):
    """Show schema deployment status."""
    logger.info("Getting schema deployment status...")
    
    # Initialize schema version manager and deployment manager
    version_manager = SchemaVersionManager(schema_dir=args.schema_dir)
    deployment_manager = SchemaDeploymentManager(version_manager)
    
    # Get status
    status = deployment_manager.get_environment_status()
    
    # Print status
    logger.info("\nSchema Deployment Status:")
    logger.info(f"Timestamp: {status['timestamp']}")
    
    for env, version in status['environments'].items():
        if version:
            logger.info(f"{env.capitalize()}: v{version}")
        else:
            logger.info(f"{env.capitalize()}: Not deployed")
    
    # Get readiness
    readiness = version_manager.get_deployment_readiness()
    
    logger.info(f"\nCurrent Version: v{version_manager.get_version_string()}")
    logger.info(f"Deployment Readiness: {readiness['readiness_score']:.2f}")
    logger.info(f"Status: {readiness['status']}")
    logger.info(f"Message: {readiness['message']}")
    
    return True

def show_health(args):
    """Show schema health metrics."""
    logger.info("Getting schema health metrics...")
    
    # Initialize schema version manager
    version_manager = SchemaVersionManager(schema_dir=args.schema_dir)
    
    # Create deployment report
    report = version_manager.create_deployment_report(
        output_path=args.report_file if hasattr(args, 'report_file') else None
    )
    
    # Print health metrics
    logger.info("\nSchema Health Report:")
    logger.info(f"Schema Version: v{report['schema_version']}")
    logger.info(f"Timestamp: {report['timestamp']}")
    
    readiness = report['readiness']
    logger.info(f"\nDeployment Readiness: {readiness['readiness_score']:.2f}")
    logger.info(f"Status: {readiness['status']}")
    logger.info(f"Message: {readiness['message']}")
    
    # Print checklist
    logger.info("\nReadiness Checklist:")
    for check in readiness['checklist']:
        status = "✅" if check['status'] == 'pass' else "❌"
        logger.info(f"{status} {check['check']}: {check['value']}")
    
    # Print health trends
    if 'health_trends' in report:
        trends = report['health_trends']
        
        logger.info("\nHealth Trends:")
        for metric, trend in trends.items():
            if trend['status'] == 'success':
                logger.info(f"\n{metric}:")
                logger.info(f"  Current: {trend['statistics']['current']}")
                logger.info(f"  Trend: {trend['statistics']['trend']}")
                logger.info(f"  Average: {trend['statistics']['average']}")
                logger.info(f"  Min: {trend['statistics']['minimum']}")
                logger.info(f"  Max: {trend['statistics']['maximum']}")
    
    if args.report_file:
        logger.info(f"\nDetailed report saved to {args.report_file}")
    
    return True

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        return 1
    
    # Commands that don't require schema initialization
    if args.command == "status":
        return 0 if show_status(args) else 1
    
    if args.command == "health":
        return 0 if show_health(args) else 1
    
    # Initialize environment and components
    sandbox, schema = initialize_environment(args)
    
    # Execute the specified command
    if args.command == "build-graph":
        result = build_graph(args, sandbox, schema)
    elif args.command == "detect-clusters":
        result = detect_clusters(args, sandbox, schema)
    elif args.command == "generate-concepts":
        result = generate_concepts(args, sandbox, schema)
    elif args.command == "visualize":
        result = visualize_graph(args, sandbox, schema)
    elif args.command == "export":
        result = export_graph(args, sandbox, schema)
    elif args.command == "import":
        result = import_graph(args, sandbox, schema)
    elif args.command == "report":
        result = generate_report(args, sandbox, schema)
    elif args.command == "version":
        result = create_version(args, sandbox, schema)
    elif args.command == "deploy":
        result = deploy_schema(args, sandbox, schema)
    else:
        print(f"Unknown command: {args.command}")
        return 1
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 