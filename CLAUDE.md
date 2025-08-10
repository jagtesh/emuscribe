# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is a comprehensive video transcription tool that processes videos through an AI pipeline to create rich, structured transcripts with speaker identification and visual context.

### Core Processing Pipeline
```
Video Input → Audio Extraction → Whisper Transcription → Speaker Diarization
     ↓              ↓                    ↓                    ↓
Frame Extraction → Visual Analysis → Context Matching → Export Generation
```

### Key Components

**main.py** - Single-file application containing the complete `VideoTranscriber` class with:
- Audio extraction using FFmpeg
- Multiple Whisper backends (faster-whisper for Apple Silicon optimization, openai-whisper fallback)
- Speaker diarization (temporarily disabled due to dependency issues)
- Visual analysis and frame extraction 
- Export engines for Markdown, HTML, and PDF

**Processing Data Flow:**
1. Extract audio from video (16kHz mono WAV)
2. Transcribe using Whisper (supports word-level timestamps)
3. Perform speaker diarization (pyannote-audio)
4. Extract video frames at configurable intervals
5. Match visual content with transcript segments using keyword detection
6. Save all data to structured JSON for reusable exports
7. Export to requested format(s)

## Development Commands

### Setup and Installation
```bash
# Install system dependencies and Python packages
./install.sh

# Alternative setup using mise (if present)
mise install
mise run install
```

### Running the Application
```bash
# Quick command (fastest - uses uv or mise)
./transcribe process video.mp4

# Enhanced wrapper (with environment detection)
./run.sh process video.mp4

# Direct python (manual venv activation required)
python main.py process video.mp4

# With custom settings
./transcribe process video.mp4 --format html --interval 20 --output ./results

# Re-export from processed data (fast)
./transcribe export output/video_processed.json --format pdf
```

### Key CLI Commands
- `process` - Full video processing pipeline
- `export` - Re-export from stored JSON data without reprocessing
- `--backend faster-whisper` - Use optimized Apple Silicon backend
- `--no-diarization` - Disable speaker identification
- `--interval N` - Screenshot every N seconds

## Configuration

**config.json** - Main configuration file with:
- Whisper model selection (tiny/base/small/medium/large)
- Device preferences (auto/cpu/mps/cuda) 
- Processing intervals and quality settings
- Backend selection (faster-whisper vs openai-whisper)

**Critical Settings:**
- `whisper_backend`: "faster-whisper" (recommended for Apple Silicon) or "openai-whisper"
- `device`: "auto" (detects MPS/CUDA/CPU), "mps", "cuda", or "cpu"
- `compute_type`: "auto", "int8", "float16", "float32"
- `screenshot_interval`: seconds between frame captures

## Data Storage Format

The tool generates a comprehensive JSON file (`{video_name}_processed.json`) containing:
- **metadata**: Video info, processing settings, timestamps
- **transcript**: Complete segments with speaker labels and timing
- **visual**: Frame data and visual-text relevance matching

This allows instant re-export to different formats without reprocessing.

## Apple Silicon Optimization

The codebase is optimized for Apple Silicon (M1/M2/M3) with:
- **faster-whisper backend**: 5x faster model loading, CTranslate2 optimizations
- **MPS device detection**: Automatic Metal Performance Shaders usage
- **int8 quantization**: Default for Apple Silicon for speed/quality balance
- **Fallback mechanisms**: Graceful degradation to CPU when needed

## Dependencies and Environment

**Python Environment:**
- Managed via virtual environment (venv/ or .venv/)
- Python 3.9+ required
- Supports both manual venv and mise tool management

**Key Dependencies:**
- **whisper/faster-whisper**: Speech recognition
- **torch**: PyTorch for ML models
- **opencv-python**: Video frame processing
- **ffmpeg**: Audio/video manipulation (system dependency)
- **pyannote-audio**: Speaker diarization (temporarily disabled)
- **reportlab**: PDF generation
- **markdown**: HTML conversion

**System Requirements:**
- macOS 10.15+ (Catalina or later)
- FFmpeg installed via Homebrew
- 8GB RAM minimum (16GB+ for large models)

## Testing and Quality

Currently no formal test suite. Quality assurance through:
- Processing sample videos with known characteristics
- Verifying export format integrity
- Cross-checking timestamp accuracy
- Manual verification of speaker diarization results

## Known Issues and Temporary Disables

1. **Speaker Diarization**: Temporarily disabled due to pyannote-audio dependency conflicts
2. **CLIP Visual Analysis**: Disabled due to transformers library issues
3. **WeasyPrint PDF**: Replaced with ReportLab due to system library dependencies

The application includes comprehensive error handling and fallback mechanisms for these disabled features.

## Performance Characteristics

**Processing Speed** (depends on Whisper model):
- tiny: ~2x real-time
- base: ~1x real-time  
- large: ~0.3x real-time

**Export Speed** (from stored JSON):
- Markdown: <1s
- HTML: ~2s
- PDF: ~3s (includes image embedding)

**Typical Output Sizes** (6-minute video):
- Processed JSON: ~254KB
- Markdown: ~8KB
- HTML: ~15MB (embedded images)
- PDF: ~3MB