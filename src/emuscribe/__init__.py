"""
emuscribe - A comprehensive video transcription tool with AI and speaker diarization

A macOS application for transcribing videos with locally hosted AI models,
featuring speaker identification, intelligent screenshot capture, and multiple export formats.
"""

from importlib.metadata import PackageNotFoundError, version

from .transcriber import VideoTranscriber

try:
    __version__ = version("emuscribe")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+dev"

__author__ = "Jagtesh Chadha"
__email__ = "jagtesh@example.com"

__all__ = ["VideoTranscriber", "__version__"]