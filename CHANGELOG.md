# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PyPI publishing setup with automated workflows
- Proper Python package structure with `src/emuscribe/` layout
- setuptools-scm for automatic version management from git tags
- GitHub Actions workflow for testing and publishing
- Type checking support (py.typed marker)
- Development dependencies (black, ruff, pytest)

### Changed
- Moved from single `main.py` to proper package structure
- CLI now available as `emuscribe` command after installation
- Updated all scripts to use new package structure
- Modern pyproject.toml with comprehensive metadata

### Fixed
- Package can now be properly installed and distributed

## [1.0.0] - 2025-08-10

### Added
- Initial release of emuscribe
- Video transcription with OpenAI Whisper and faster-whisper backends
- Speaker diarization support (currently disabled due to dependencies)
- Intelligent screenshot capture at configurable intervals
- Visual content analysis and matching with transcript
- Multiple export formats: Markdown, HTML, PDF
- Apple Silicon optimization with MPS and int8 quantization
- Reusable data storage - process once, export multiple times
- uv support for fast Python environment management
- Smart installation script with automatic environment detection

### Features
- **Core Processing**: Audio extraction, Whisper transcription, speaker identification
- **Visual Analysis**: Frame extraction with keyword-based relevance matching
- **Export Formats**: Clean Markdown, self-contained HTML, professional PDF
- **Apple Silicon**: 5x faster loading with faster-whisper backend
- **Smart Caching**: Automatic detection of existing processed data
- **Multiple Backends**: faster-whisper (recommended) and openai-whisper fallback
- **Configuration**: JSON-based configuration with CLI overrides
- **Command Line**: Process and export subcommands with legacy compatibility

### Technical Details
- Python 3.9+ required
- macOS 10.15+ (Catalina or later)
- FFmpeg and portaudio system dependencies
- Virtual environment support (venv, .venv, mise)
- Comprehensive error handling and logging
- ReportLab-based PDF generation (replacing WeasyPrint)

### Performance
- **Model Loading**: 0.3s (faster-whisper) vs 1.8s (openai-whisper)
- **Export Speed**: <1s (Markdown), ~2s (HTML), ~3s (PDF)
- **File Sizes**: 8KB (Markdown), 15MB (HTML), 3MB (PDF) for 6-min video