# Video Transcription Tool - Usage Guide

## Overview

This tool now supports reusable data storage, allowing you to process a video once and export to multiple formats without re-running the expensive transcription and AI analysis.

## Storage Format

When processing a video, the tool now saves all processed data in a structured JSON format:

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

## Usage Commands

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

## Processed Data Structure

The JSON file contains:

- **metadata**: Video information, processing settings, duration
- **transcript**: Complete transcript with speaker information and timestamps
- **visual**: Frame data and visual-text matching information

## Benefits

1. **No Re-processing**: Export to multiple formats without running transcription again
2. **Fast Format Changes**: Switch between markdown, HTML, and PDF instantly
3. **Data Preservation**: All processing work is saved and reusable
4. **Selective Re-processing**: Delete JSON file to reprocess from scratch
5. **Asset Management**: All screenshots organized in frames/ subdirectory

## Smart Caching

The tool automatically detects if processed data already exists and uses it instead of reprocessing, making subsequent runs much faster.