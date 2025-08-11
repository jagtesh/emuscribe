#!/bin/bash
# install.sh - Modern installation script

set -e  # Exit on error

echo "🚀 Installing Video Transcription Tool Dependencies..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo "🔧 Installing system dependencies..."
brew install ffmpeg portaudio

# Check for uv first (preferred)
if command -v uv &> /dev/null; then
    echo "⚡ Using uv for fast Python environment setup..."
    
    # Create .venv with uv
    uv venv .venv
    
    # Install dependencies using pyproject.toml (preferred)
    if [ -f "pyproject.toml" ]; then
        echo "📋 Installing from pyproject.toml..."
        uv pip install -e .
    else
        echo "📋 Installing from requirements.txt..."
        uv pip install -r requirements.txt
    fi
    
    echo "✅ Installation complete! Use ./transcribe to run."
    
else
    echo "🐍 Using traditional pip installation..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install Python packages from requirements.txt
    pip install --upgrade pip
    
    if [ -f "pyproject.toml" ]; then
        echo "📋 Installing from pyproject.toml..."
        pip install -e .
    else
        echo "📋 Installing from requirements.txt..."
        pip install -r requirements.txt
    fi
    
    echo "✅ Installation complete! Use ./run.sh to run."
    echo "💡 For faster startup, consider installing uv: brew install uv"
fi