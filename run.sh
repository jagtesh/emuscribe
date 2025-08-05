#!/bin/bash
# Simple wrapper script to run the video transcriber

# Check if .venv exists (mise-managed), otherwise use venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    source venv/bin/activate
fi

# Run the transcriber with all arguments passed through
python main.py "$@"