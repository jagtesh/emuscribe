#!/bin/bash
# Enhanced wrapper script to run the video transcriber

set -e  # Exit on error

# Function to activate the right virtual environment
activate_venv() {
    # Try mise first if available and configured
    if command -v mise &> /dev/null && [ -f "mise.toml" ]; then
        echo "🔧 Using mise-managed environment..."
        eval "$(mise activate bash)"
        return 0
    fi
    
    # Check if .venv exists (mise-managed or uv-managed), otherwise use venv
    if [ -d ".venv" ]; then
        echo "🐍 Activating .venv virtual environment..."
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        echo "🐍 Activating venv virtual environment..."
        source venv/bin/activate
    else
        echo "❌ No virtual environment found! Please run ./install.sh first."
        exit 1
    fi
}

# Activate virtual environment
activate_venv

# Check if package is installed
if ! python -c "import emuscribe" 2>/dev/null; then
    echo "❌ emuscribe package not found! Please install with:"
    echo "  pip install -e ."
    exit 1
fi

# Run the transcriber with all arguments passed through
echo "🚀 Running video transcriber..."
python -m emuscribe.cli "$@"