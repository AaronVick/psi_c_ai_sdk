#!/bin/bash

# ΨC Schema Integration Demo Runner Script

echo "ΨC Schema Integration Demo"
echo "--------------------------"

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

# Define the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEMO_DIR="$SCRIPT_DIR/demo"

# Install required packages if needed
echo "Checking requirements..."
if [ -f "$DEMO_DIR/requirements.txt" ]; then
    pip3 install -r "$DEMO_DIR/requirements.txt"
else
    echo "Warning: requirements.txt not found in demo directory."
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Error: Failed to import Streamlit. Please install it manually: pip install streamlit"
    exit 1
fi

# Check for web_interface_demo.py
if [ -f "$DEMO_DIR/web_interface_demo.py" ]; then
    DEMO_PATH="$DEMO_DIR/web_interface_demo.py"
else
    echo "Error: Could not find web_interface_demo.py in the demo directory."
    exit 1
fi

# Create state directory if it doesn't exist
mkdir -p "$DEMO_DIR/state"
mkdir -p "$DEMO_DIR/demo_config"

# Run tests if requested
if [ "$1" == "--test" ] || [ "$1" == "-t" ]; then
    echo "Running tests..."
    if [ -f "$DEMO_DIR/test_demo.py" ]; then
        python3 "$DEMO_DIR/test_demo.py"
    else
        echo "Warning: Could not find test_demo.py. Skipping tests."
    fi
    # Remove the test argument so it doesn't get passed to streamlit
    shift
fi

# Run the demo
echo ""
echo "Starting ΨC Schema Integration Demo..."
echo "To stop the demo, press Ctrl+C"
echo ""

# Run Streamlit
streamlit run "$DEMO_PATH" "$@" 