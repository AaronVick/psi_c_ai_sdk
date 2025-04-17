#!/usr/bin/env python3
"""
ΨC Demo Runner Script
---------------------
Simplified launcher for the ΨC-AI SDK demonstration.
This script handles environment setup and launches the Streamlit interface.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        "streamlit",
        "networkx",
        "matplotlib",
        "pandas"
    ]
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements(packages):
    """Install missing requirements."""
    print(f"Installing missing packages: {', '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("Installation completed successfully.")

def setup_environment():
    """Set up the environment for running the demo."""
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    sys.path.append(str(script_dir))
    
    # Add parent directory to path for imports
    parent_dir = script_dir.parent.absolute()
    sys.path.append(str(parent_dir))
    
    # Create necessary directories
    demo_data_dir = script_dir / "demo_data" / "history"
    demo_data_dir.mkdir(parents=True, exist_ok=True)
    
    return script_dir

def run_demo(script_dir):
    """Run the demo using Streamlit."""
    print("Starting ΨC Demo Interface...")
    
    # Path to the web interface demo
    demo_path = script_dir / "web_interface_demo.py"
    
    if not demo_path.exists():
        print(f"Error: Demo interface not found at {demo_path}")
        return False
    
    # Run Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_path)])
        return True
    except Exception as e:
        print(f"Error running demo: {str(e)}")
        return False

def main():
    """Main function to launch the demo."""
    print("ΨC Schema Integration Demo Launcher")
    print("===================================")
    
    # Check requirements
    print("Checking requirements...")
    missing_packages = check_requirements()
    if missing_packages:
        try:
            install_requirements(missing_packages)
        except Exception as e:
            print(f"Error installing requirements: {str(e)}")
            print("Please install the following packages manually:")
            for package in missing_packages:
                print(f"  - {package}")
            return
    
    # Set up environment
    script_dir = setup_environment()
    
    # Run the demo
    run_demo(script_dir)

if __name__ == "__main__":
    main() 