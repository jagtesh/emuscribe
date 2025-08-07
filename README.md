# Video Transcription Tool

A comprehensive macOS application for transcribing videos with locally hosted AI models, featuring speaker identification, intelligent screenshot capture, and multiple export formats.

## Pipeline Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│   Processing     │───▶│    Output       │
│   (.mp4, .mov)  │    │   Pipeline       │    │  (.md/.html/.pdf)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Components:       │
                    │  • Audio Extract   │
                    │  • Whisper AI      │
                    │  • Diarization     │
                    │  • Frame Extract   │
                    │  • Visual Analysis │
                    │  • Export Engine   │
                    └────────────────────┘
```

## Key Features

- **Local AI Processing**: Uses OpenAI's Whisper for transcription and pyannote-audio for speaker diarization
- **Intelligent Visual Capture**: Automatically screenshots video frames when speakers reference visual content
- **Speaker Identification**: Distinguishes between different speakers throughout the transcript
- **Multiple Export Formats**: Outputs clean Markdown, self-contained HTML, or PDF formats
- **Reusable Data Storage**: Process once, export to multiple formats without re-running transcription
- **Apple Silicon Optimized**: 5x faster performance on M-series chips with faster-whisper backend

## Usage

### Processing a Video (First Time)

```bash
# Basic processing
python main.py process video.mp4

# Custom format and settings
python main.py process video.mp4 --format html --interval 20 --output ./results

# Disable certain features
python main.py process video.mp4 --no-diarization --no-visual
```

### Re-exporting from Stored Data

Once you have processed a video, you can export to different formats instantly:

```bash
# Export to HTML from stored data
python main.py export output/video_processed.json --format html

# Export to PDF with custom output directory
python main.py export output/video_processed.json --format pdf -o ./exports
```

### Backward Compatibility

The tool maintains backward compatibility. This still works:

```bash
python main.py video.mp4 --format markdown
```

## Storage Format

When processing a video, the tool saves all processed data in a structured JSON format:

```
output/
├── frames/                          # All extracted screenshots
│   ├── frame_0001_5.30s.jpg
│   ├── frame_0002_35.30s.jpg
│   └── ...
├── video_name_processed.json        # Complete processed data
├── video_name_transcript.md         # Exported markdown
├── video_name_transcript.html       # Exported HTML
└── video_name_transcript.pdf        # Exported PDF
```

## Apple Silicon Optimization

On Apple Silicon Macs (M1/M2/M3), the tool provides optimized backends for maximum performance:

### Whisper Backends

**faster-whisper (Recommended)**
- **5x faster loading**: 0.3s vs 1.8s model initialization
- **Apple Silicon optimized**: Built on CTranslate2 with ARM64 optimizations
- **Memory efficient**: Uses quantized int8 models by default
- **Better performance**: Leverages Apple's optimized BLAS libraries

**openai-whisper (Fallback)**
- **MPS compatibility issues**: Falls back to CPU due to PyTorch sparse tensor limitations
- **Slower but compatible**: Works when faster-whisper has issues

### Command Line Override

```bash
# Use faster-whisper (recommended)
python main.py process video.mp4 --backend faster-whisper

# Use OpenAI Whisper 
python main.py process video.mp4 --backend openai-whisper
```

## Export Format Comparison

| Format | Timestamps | Images | File Size | Use Case |
|--------|------------|--------|-----------|----------|
| **Markdown** | ✅ Separate lines | 🖼️ Linked | ~8KB | Documentation, editing |
| **HTML** | ✅ Separate lines | 🖼️ Embedded | ~15MB | Web viewing, sharing |
| **PDF** | ✅ Separate lines | 🖼️ Embedded | ~3MB | Printing, archival |

## Recent Improvements

### ✅ Fixed Issues

**1. PDF Export Fixed**
- **Problem**: WeasyPrint had system library dependencies issues
- **Solution**: Implemented ReportLab-based PDF export
- **Benefits**: 
  - Pure Python solution (no system dependencies)
  - Better performance and reliability
  - Professional PDF formatting with embedded images

**2. Improved Timestamp Formatting**
- **Problem**: Timestamps were inline with text, making transcripts hard to read
- **Solution**: Put timestamps on separate lines in all formats
- **Before**: `**[00:00:06]** First of all, we'll talk about...`
- **After**: 
  ```
  **[00:00:06]**
  First of all, we'll talk about...
  ```

### Performance Results

**Model Loading Time**
- **faster-whisper**: 0.32s ⚡
- **openai-whisper**: 1.80s 🐌

**Export Speed** (from stored data)
- **Markdown**: <1s ⚡
- **HTML**: ~2s ⚡  
- **PDF**: ~3s ⚡ (includes image processing)

**File Sizes** (6-minute video)
- **Processed JSON**: 254KB (all data stored)
- **Markdown**: 8KB (clean, readable)
- **HTML**: 15MB (self-contained with images)
- **PDF**: 3MB (professional format with images)

## Key Benefits

1. **No Re-processing**: Export to any format instantly from stored data
2. **Better Readability**: Timestamps on separate lines improve flow
3. **Reliable PDF**: No more system dependency issues
4. **Apple Silicon Optimized**: True performance gains on M-series chips
5. **Professional Output**: All formats look clean and professional
6. **Smart Caching**: Automatically detects if processed data already exists

## Installation

### System Dependencies
- macOS 10.15+ (Catalina or later)
- Python 3.9+
- FFmpeg for media processing
- 8GB RAM minimum (16GB+ recommended for large models)

### Setup
```bash
# Run the installation script
./install_dependencies.sh
```

## Configuration

The processed data JSON contains:
- **metadata**: Video information, processing settings, duration
- **transcript**: Complete transcript with speaker information and timestamps
- **visual**: Frame data and visual-text matching information

### Backend Configuration
```json
{
  "whisper_backend": "faster-whisper",  // "faster-whisper" or "openai-whisper"
  "compute_type": "auto",               // auto, int8, int8_float32, float16, float32
  "device": "auto",                     // auto, cpu, mps, cuda
  "force_cpu": false                    // Force CPU even if GPU available
}
```

All improvements maintain full backward compatibility while providing significant performance and usability enhancements!