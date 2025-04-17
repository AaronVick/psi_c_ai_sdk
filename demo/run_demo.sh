#!/bin/bash

# ΨC Schema Integration Demo Runner

echo "ΨC Schema Integration Demo"
echo "--------------------------"
echo "Checking requirements..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not found."
    exit 1
fi

# Check if requirements are installed
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found."
    exit 1
fi

# Install requirements if needed
echo "Installing requirements..."
pip3 install -r requirements.txt

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Error: Failed to import Streamlit. Please install it manually: pip install streamlit"
    exit 1
fi

# Run the demo
echo "Starting ΨC Schema Integration Demo..."
echo "To stop the demo, press Ctrl+C"
echo ""

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory (parent of demo directory)
cd "$SCRIPT_DIR/.."

# Run Streamlit
streamlit run "$SCRIPT_DIR/web_interface_demo.py" 