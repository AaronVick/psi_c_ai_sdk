#!/usr/bin/env python3
"""
Launcher module for the ΨC-AI SDK Development Environment.

This module provides functions to launch individual development tools or the
complete suite of development tools, either via command line or from Python.

Usage:
    # Launch all tools
    python -m tools.dev_environment.launcher
    
    # Launch specific tools
    python -m tools.dev_environment.launcher --tools schema memory reflection
    
    # Launch with a specific agent
    python -m tools.dev_environment.launcher --agent-path /path/to/agent.json
    
    # Launch with custom configuration
    python -m tools.dev_environment.launcher --config /path/to/config.yaml
"""

import argparse
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import tools
try:
    from tools.dev_environment.web_interface import DevWebInterface
    from tools.dev_environment.memory_sandbox import MemorySandbox
    from tools.dev_environment.memory_schema_integration import integrate_with_memory_sandbox
    from psi_c_ai_sdk.memory.memory_store import MemoryStore
    TOOLS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing tools: {e}")
    TOOLS_AVAILABLE = False

# List of available tools
AVAILABLE_TOOLS = [
    "schema_editor",
    "memory_sandbox",
    "reflection_debugger",
    "consciousness_inspector",
    "stress_test_generator"
]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ΨC-AI SDK Development Environment Launcher")
    
    parser.add_argument(
        "tool", 
        nargs="?", 
        choices=AVAILABLE_TOOLS + ["all", "web"],
        default="web",
        help="Tool to launch (default: web interface)"
    )
    
    parser.add_argument(
        "--agent-path", 
        type=str,
        help="Path to agent file to load"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--host", 
        type=str,
        default="localhost",
        help="Host to run web interface on (default: localhost)"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=8501,
        help="Port to run web interface on (default: 8501)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--enable-schema", 
        action="store_true",
        help="Enable schema integration with Memory Sandbox"
    )
    
    return parser.parse_args()

def launch_web_interface(host: str = "localhost", port: int = 8501, 
                        agent_path: Optional[str] = None,
                        config_path: Optional[str] = None) -> None:
    """
    Launch the web interface for all tools.
    
    Args:
        host: Host to run on
        port: Port to run on
        agent_path: Path to agent file to load
        config_path: Path to configuration file
    """
    try:
        from tools.dev_environment.web_interface import main
        import streamlit.web.cli as stcli
        
        # Build Streamlit command
        sys.argv = [
            "streamlit", 
            "run", 
            os.path.abspath(main.__code__.co_filename),
            "--server.address", host,
            "--server.port", str(port)
        ]
        
        if agent_path:
            sys.argv.extend(["--", "--agent-path", agent_path])
            
        if config_path:
            sys.argv.extend(["--config", config_path])
            
        logger.info(f"Launching web interface on {host}:{port}")
        sys.exit(stcli.main())
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure streamlit is installed: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to launch web interface: {e}")
        sys.exit(1)

def launch_schema_editor(agent_path: Optional[str] = None,
                        config_path: Optional[str] = None) -> Any:
    """
    Launch the Schema Editor tool.
    
    Args:
        agent_path: Path to agent file to load
        config_path: Path to configuration file
        
    Returns:
        SchemaEditor instance
    """
    try:
        from tools.dev_environment.schema_editor import SchemaEditor
        
        logger.info("Launching Schema Editor")
        editor = SchemaEditor(config_path=config_path)
        
        if agent_path:
            # In a real implementation, this would load the agent's schema
            logger.info(f"Loading schema from {agent_path}")
            
        return editor
    except ImportError:
        logger.error("SchemaEditor is not available")
        return None

def launch_memory_sandbox(args):
    """
    Launch the Memory Sandbox tool.
    
    Args:
        args: Command line arguments
    """
    if not TOOLS_AVAILABLE:
        logger.error("Required modules not available")
        return 1
        
    # Create sandbox directory
    sandbox_dir = os.path.join(os.getcwd(), "memory_snapshots")
    os.makedirs(sandbox_dir, exist_ok=True)
    
    # Create memory store
    memory_store = MemoryStore()
    
    # Create and initialize sandbox
    sandbox = MemorySandbox(
        memory_store=memory_store,
        snapshot_dir=sandbox_dir
    )
    
    # Initialize schema integration if enabled
    if args.enable_schema:
        schema = MemorySchemaIntegration(sandbox)
        print("Schema Integration enabled for Memory Sandbox")
        
        # Add schema integration methods to sandbox for easy access
        sandbox.schema = schema
        sandbox.build_schema_graph = schema.build_schema_graph
        sandbox.detect_memory_clusters = schema.detect_memory_clusters
        sandbox.generate_concept_suggestions = schema.generate_concept_suggestions
        sandbox.find_related_memories = schema.find_related_memories
        sandbox.visualize_schema_graph = schema.visualize_schema_graph
        sandbox.export_schema_graph = schema.export_schema_graph
        sandbox.generate_knowledge_report = schema.generate_knowledge_report
    
    # Launch web interface with sandbox
    web = DevWebInterface()
    web.add_tool(sandbox)
    
    logger.info(f"Memory Sandbox initialized with snapshot directory: {sandbox_dir}")
    logger.info("Launching web interface with Memory Sandbox...")
    
    return web.run(port=args.port, debug=args.debug)

def launch_reflection_debugger(agent_path: Optional[str] = None,
                              config_path: Optional[str] = None) -> Any:
    """
    Launch the Reflection Debugger tool.
    
    Args:
        agent_path: Path to agent file to load
        config_path: Path to configuration file
        
    Returns:
        ReflectionDebugger instance
    """
    try:
        from tools.dev_environment.reflection_debugger import ReflectionDebugger
        
        logger.info("Launching Reflection Debugger")
        debugger = ReflectionDebugger(config_path=config_path)
        
        if agent_path:
            logger.info(f"Loading agent from {agent_path}")
            # In a real implementation, this would load the agent
        
        return debugger
    except ImportError:
        logger.error("ReflectionDebugger is not available")
        return None

def launch_consciousness_inspector(agent_path: Optional[str] = None,
                                  config_path: Optional[str] = None) -> Any:
    """
    Launch the Consciousness Inspector tool.
    
    Args:
        agent_path: Path to agent file to load
        config_path: Path to configuration file
        
    Returns:
        ConsciousnessInspector instance
    """
    try:
        from tools.dev_environment.consciousness_inspector import ConsciousnessInspector
        
        logger.info("Launching Consciousness Inspector")
        inspector = ConsciousnessInspector(config_path=config_path)
        
        if agent_path:
            logger.info(f"Loading agent from {agent_path}")
            # In a real implementation, this would load the agent
        
        return inspector
    except ImportError:
        logger.error("ConsciousnessInspector is not available")
        return None

def launch_stress_test_generator(agent_path: Optional[str] = None,
                                config_path: Optional[str] = None) -> Any:
    """
    Launch the Stress Test Generator tool.
    
    Args:
        agent_path: Path to agent file to load
        config_path: Path to configuration file
        
    Returns:
        StressTestGenerator instance
    """
    try:
        from tools.dev_environment.stress_test_generator import StressTestGenerator
        
        logger.info("Launching Stress Test Generator")
        generator = StressTestGenerator(config_path=config_path)
        
        if agent_path:
            logger.info(f"Loading agent from {agent_path}")
            # In a real implementation, this would load the agent
        
        return generator
    except ImportError:
        logger.error("StressTestGenerator is not available")
        return None

def launch_tool(tool_name: str, agent_path: Optional[str] = None,
               config_path: Optional[str] = None) -> Any:
    """
    Launch a specific development tool.
    
    Args:
        tool_name: Name of the tool to launch
        agent_path: Path to agent file to load
        config_path: Path to configuration file
        
    Returns:
        Tool instance, or None if the tool is not available
    """
    tool_launchers = {
        "schema_editor": launch_schema_editor,
        "memory_sandbox": launch_memory_sandbox,
        "reflection_debugger": launch_reflection_debugger,
        "consciousness_inspector": launch_consciousness_inspector,
        "stress_test_generator": launch_stress_test_generator,
        "web": launch_web_interface
    }
    
    launcher = tool_launchers.get(tool_name)
    if launcher:
        return launcher(agent_path=agent_path, config_path=config_path)
    else:
        logger.error(f"Unknown tool: {tool_name}")
        return None

def launch_all_tools(agent_path: Optional[str] = None,
                    config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Launch all available development tools.
    
    Args:
        agent_path: Path to agent file to load
        config_path: Path to configuration file
        
    Returns:
        Dictionary mapping tool names to tool instances
    """
    tools = {}
    
    for tool_name in AVAILABLE_TOOLS:
        tool = launch_tool(tool_name, agent_path=agent_path, config_path=config_path)
        if tool:
            tools[tool_name] = tool
    
    return tools

def main() -> None:
    """Main entry point for the launcher."""
    args = parse_args()
    
    if args.tool == "all":
        launch_all_tools(agent_path=args.agent_path, config_path=args.config)
    elif args.tool == "web":
        launch_web_interface(
            host=args.host,
            port=args.port,
            agent_path=args.agent_path,
            config_path=args.config
        )
    else:
        launch_tool(args.tool, agent_path=args.agent_path, config_path=args.config)

if __name__ == "__main__":
    main() 