#!/bin/bash
# install_dependencies.sh

echo "Installing Video Transcription Tool Dependencies..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
# brew install ffmpeg python@3.13 portaudio
brew install ffmpeg portaudio

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install openai-whisper
pip install pyannote-audio
pip install opencv-python
pip install transformers
pip install sentence-transformers
pip install clip-by-openai
pip install markdown
pip install weasyprint
pip install Pillow
pip install numpy
pip install scipy
pip install librosa
pip install soundfile

echo "Installation complete!"