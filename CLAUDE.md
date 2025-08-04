# Video Transcription Tool with AI and Diarization

A comprehensive macOS application for transcribing videos with locally hosted AI models, featuring speaker identification, intelligent screenshot capture, and multiple export formats.

## Project Overview

This tool processes video files to create rich, structured transcripts that combine speech-to-text, speaker identification, and contextual visual information. It's designed for professionals who need accurate, searchable records of video content with visual context preserved.

### Key Capabilities

- **Local AI Processing**: Uses OpenAI's Whisper for transcription and pyannote-audio for speaker diarization, ensuring privacy and offline capability
- **Intelligent Visual Capture**: Automatically screenshots video frames when speakers reference visual content or at configurable intervals
- **Speaker Identification**: Distinguishes between different speakers and labels their contributions throughout the transcript
- **Multiple Export Formats**: Outputs clean Markdown, self-contained HTML, or PDF formats optimized for different use cases
- **LLM-Optimized Output**: Structured format specifically designed for consumption by language models

## Technical Architecture

### Core Components

1. **Media Processing Pipeline**
   - FFmpeg for video/audio extraction and processing
   - OpenCV for frame extraction and manipulation
   - Librosa for advanced audio analysis

2. **AI Models**
   - Whisper (tiny/base/small/medium/large) for speech recognition
   - pyannote-audio for speaker diarization
   - CLIP for visual-text relevance scoring

3. **Export Engine**
   - Markdown generation with embedded media references
   - HTML export with base64-encoded images
   - PDF generation via WeasyPrint

### Processing Flow

```
Video Input → Audio Extraction → Transcription → Speaker Diarization
     ↓              ↓                ↓              ↓
Frame Extraction → Visual Analysis → Context Matching → Export Generation
```

## Use Cases

### Professional Applications
- **Meeting Documentation**: Transcribe team meetings with speaker identification and visual aids
- **Educational Content**: Process lecture recordings with slide captures and speaker notes
- **Interview Analysis**: Create searchable transcripts of interviews with contextual screenshots
- **Training Materials**: Convert video training content into structured documentation

### Content Creation
- **Podcast Production**: Generate show notes with timestamps and visual elements
- **Video Editing**: Create detailed scripts with visual cues for post-production
- **Research Documentation**: Process research interviews or focus groups with speaker attribution

## Installation Requirements

### System Dependencies
- macOS 10.15+ (Catalina or later)
- Python 3.9+
- FFmpeg for media processing
- 8GB RAM minimum (16GB+ recommended for large models)
- 5GB storage for AI models and dependencies

### AI Model Setup
- HuggingFace account for speaker diarization models
- Local model storage for Whisper variants
- Optional GPU acceleration for faster processing

## Configuration Options

### Processing Settings
- **Whisper Model Size**: Balance between speed and accuracy (tiny → large)
- **Screenshot Intervals**: Configurable capture frequency (seconds)
- **Speaker Sensitivity**: Minimum duration for speaker identification
- **Visual Analysis**: Enable/disable CLIP-based screenshot intelligence

### Output Customization
- **Format Selection**: Markdown, HTML, or PDF export
- **Image Quality**: Configurable JPEG compression for screenshots
- **Timestamp Precision**: Second or sub-second level timestamps
- **Speaker Labeling**: Automatic or manual speaker name assignment

## File Structure

```
video-transcription-tool/
├── video_transcriber.py      # Main application
├── install_dependencies.sh   # Setup script
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── output/                  # Generated transcripts and media
│   ├── frames/             # Extracted screenshots
│   ├── transcripts/        # Markdown outputs
│   └── exports/            # HTML/PDF exports
└── logs/                   # Processing logs
```

## Output Format Example

### Markdown Structure
```markdown
# Transcript: quarterly_meeting

**Generated:** 2025-08-04 14:30:25
**Duration:** 01:23:45

## Speakers
- **SPEAKER_00** (John - CEO)
- **SPEAKER_01** (Sarah - CTO)

## Transcript

### SPEAKER_00
**[00:00:03]** Welcome everyone to today's quarterly review. 
Let me share the performance dashboard.

![Screenshot at 00:00:03](frame_0001_3.45s.jpg)

**[00:00:15]** As you can see in this chart, our user growth 
has exceeded targets by 23%.

### SPEAKER_01
**[00:00:32]** That's fantastic! Can you break down the 
metrics by geographic region?
```

## Development Workflow

### Setup Process
1. Clone repository and run installation script
2. Configure HuggingFace authentication for diarization
3. Test with sample video file
4. Customize configuration for specific use cases

### Processing Pipeline
1. Video validation and format checking
2. Audio extraction and preprocessing
3. Parallel transcription and frame extraction
4. Speaker diarization and segment alignment
5. Visual content analysis and matching
6. Output generation in requested format

### Quality Assurance
- Automated testing with sample videos
- Manual verification of speaker accuracy
- Visual relevance scoring validation
- Export format integrity checks

## Performance Considerations

### Processing Speed
- **Tiny Model**: ~2x real-time processing
- **Base Model**: ~1x real-time processing
- **Large Model**: ~0.3x real-time processing

### Accuracy Trade-offs
- Smaller models process faster but may miss nuances
- Larger models provide better accuracy for technical content
- Speaker diarization accuracy depends on audio quality and speaker distinctiveness

### Resource Management
- Temporary file cleanup after processing
- Configurable memory usage limits
- Optional GPU acceleration support

## Integration Possibilities

### Workflow Integration
- Command-line interface for automation
- Batch processing capabilities for multiple files
- Integration with video conferencing platforms
- API endpoints for programmatic access

### Output Consumption
- Direct import into documentation systems
- LLM prompt integration for content analysis
- Search indexing for large video libraries
- Integration with note-taking applications

## Future Development

### Planned Enhancements
- Real-time processing during recording
- Web-based interface for easier access
- Cloud storage integration (S3, Google Drive)
- Advanced speaker recognition with voice profiles
- Automated summary generation using LLMs
- Multi-language interface support

### Potential Integrations
- Zoom/Teams plugin development
- Notion/Obsidian export capabilities
- Vector database integration for semantic search
- Custom speaker training for improved accuracy

## Technical Notes

### Dependencies Management
- Virtual environment isolation for Python packages
- Homebrew for system-level dependencies
- Model caching for faster subsequent runs
- Graceful fallbacks when optional features fail

### Error Handling
- Comprehensive logging for debugging
- Graceful degradation when models unavailable
- Input validation for supported video formats
- Recovery mechanisms for interrupted processing

### Security Considerations
- Local-only processing for privacy
- No cloud API dependencies for core functionality
- Secure credential handling for optional services
- Output sanitization for web exports

This tool represents a complete solution for video content analysis and documentation, designed to bridge the gap between raw video content and structured, searchable text with preserved visual context.