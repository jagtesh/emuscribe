# Contributing to emuscribe

Thank you for your interest in contributing to emuscribe! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- macOS 10.15+ (Catalonia or later)
- Python 3.9+
- FFmpeg and portaudio (installed via Homebrew)
- uv (recommended) or pip

### Quick Start
```bash
# Clone the repository
git clone https://github.com/jagtesh/emuscribe.git
cd emuscribe

# Install uv if not already installed
brew install uv

# Setup development environment
uv venv .venv
uv pip install -e ".[dev]"

# Test the installation
emuscribe --help
```

## Development Workflow

### Code Quality
We use several tools to maintain code quality:

```bash
# Format code with black
black src/

# Lint with ruff
ruff check src/

# Type checking (when configured)
# mypy src/
```

### Testing
Currently, testing is done manually with sample videos. We welcome contributions to add a proper test suite.

```bash
# Test basic functionality
emuscribe process sample.mp4

# Test different formats
emuscribe export output/sample_processed.json --format html
emuscribe export output/sample_processed.json --format pdf
```

### Package Structure
```
emuscribe/
├── src/emuscribe/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── cli.py              # Command-line interface
│   ├── transcriber.py      # Core VideoTranscriber class
│   └── py.typed           # Type checking marker
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Legacy compatibility
├── README.md              # Project documentation
└── CHANGELOG.md           # Version history
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 style guide (enforced by black and ruff)
- Use type hints where possible
- Write docstrings for classes and functions
- Keep functions focused and modular

### Commit Messages
Use conventional commit format:
```
type(scope): description

Examples:
feat(transcriber): add support for new audio format
fix(cli): handle missing file error gracefully
docs(readme): update installation instructions
```

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and test them
4. Run code quality checks: `black src/ && ruff check src/`
5. Commit your changes with descriptive messages
6. Push to your fork and submit a pull request

### Issues
When reporting issues, please include:
- Operating system and version
- Python version
- Complete error messages
- Steps to reproduce the issue
- Sample files (if applicable and safe to share)

## Areas for Contribution

### High Priority
- **Test Suite**: Add pytest-based tests for core functionality
- **Type Annotations**: Add comprehensive type hints throughout codebase
- **Error Handling**: Improve error messages and recovery mechanisms
- **Documentation**: API documentation and usage examples

### Medium Priority
- **Performance**: Optimize processing speed and memory usage
- **Features**: Additional export formats or processing options
- **Dependencies**: Re-enable pyannote-audio and CLIP when dependencies stabilize
- **Cross-platform**: Windows and Linux support

### Lower Priority
- **UI**: Web interface or GUI application
- **Cloud**: Support for cloud storage and processing
- **Languages**: Multi-language interface support

## Release Process

Releases are automated through GitHub Actions:

1. **Version Bumping**: Uses setuptools-scm for automatic versioning from git tags
2. **Testing**: Runs tests on multiple Python versions (3.9-3.12)
3. **Building**: Creates wheel and source distributions
4. **Publishing**: Uploads to TestPyPI first, then PyPI on release

### Creating a Release
```bash
# Tag a new version
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0

# Or create a GitHub release, which will trigger the workflow
```

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: Contact the maintainer for security issues

## License

By contributing to emuscribe, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to emuscribe! 🎉