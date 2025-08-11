#!/usr/bin/env python3
"""
VideoTranscriber class for video transcription with AI and diarization
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import markdown
import numpy as np
import torch
import whisper
from faster_whisper import WhisperModel
from PIL import Image
from reportlab.lib.colors import gray

# import weasyprint  # Temporarily disabled due to system lib issues
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# from pyannote.audio import Pipeline  # Temporarily disabled
# from transformers import CLIPModel, CLIPProcessor  # Temporarily disabled


class VideoTranscriber:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.models_initialized = False

    def load_config(self, config_path):
        """Load configuration settings"""
        default_config = {
            "whisper_model": "base",  # tiny, base, small, medium, large
            "screenshot_interval": 30,  # seconds
            "screenshot_quality": 95,
            "output_format": "markdown",  # markdown, html, pdf
            "enable_diarization": True,
            "enable_visual_analysis": True,
            "visual_similarity_threshold": 0.3,
            "min_speaker_duration": 1.0,  # seconds
            "output_dir": "./output",
            "device": "auto",  # auto, cpu, mps, cuda
            "force_cpu": False,  # Force CPU even if GPU available
            "whisper_backend": "faster-whisper",  # "openai-whisper" or "faster-whisper"
            "compute_type": "auto",  # auto, int8, int8_float32, float16, float32
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("transcription.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def save_processed_data(
        self, video_path, transcript_segments, frames, visual_matches, output_dir
    ):
        """Save all processed data to JSON format for later export"""
        video_name = Path(video_path).stem
        json_file = os.path.join(output_dir, f"{video_name}_processed.json")

        # Create the comprehensive data structure
        # Compute duration safely as the max end among segments
        duration_seconds = (
            max((seg.get("end") or 0) for seg in transcript_segments)
            if transcript_segments
            else 0
        )
        processed_data = {
            "metadata": {
                "video_name": video_name,
                "video_path": str(video_path),
                "generated_at": datetime.now().isoformat(),
                "duration": duration_seconds,
                "duration_formatted": self.format_timestamp(duration_seconds),
                "processing_config": {
                    "whisper_model": self.config["whisper_model"],
                    "screenshot_interval": self.config["screenshot_interval"],
                    "diarization_enabled": self.config["enable_diarization"],
                    "visual_analysis_enabled": self.config["enable_visual_analysis"],
                },
            },
            "transcript": {
                "segments": transcript_segments,
                "speakers": sorted(
                    {seg.get("speaker", "Unknown") for seg in transcript_segments}
                ),
            },
            "visual": {
                "frames": [
                    {
                        "index": frame["index"],
                        "timestamp": frame["timestamp"],
                        "timestamp_formatted": self.format_timestamp(
                            frame["timestamp"]
                        ),
                        "filepath": frame["filepath"],
                        "relative_path": os.path.relpath(frame["filepath"], output_dir),
                    }
                    for frame in frames
                ],
                "matches": [
                    {
                        "segment": match["segment"],
                        "frame": {
                            "index": match["frame"]["index"],
                            "timestamp": match["frame"]["timestamp"],
                            "timestamp_formatted": self.format_timestamp(
                                match["frame"]["timestamp"]
                            ),
                            "filepath": match["frame"]["filepath"],
                            "relative_path": os.path.relpath(
                                match["frame"]["filepath"], output_dir
                            ),
                        },
                        "relevance_score": match["relevance_score"],
                    }
                    for match in visual_matches
                ],
            },
        }

        # Save to JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Processed data saved to {json_file}")
        return json_file

    def load_processed_data(self, json_file):
        """Load processed data from JSON file"""
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def export_from_json(self, json_file, output_format, output_dir=None):
        """Export processed data to specified format"""
        processed_data = self.load_processed_data(json_file)

        if output_dir is None:
            output_dir = os.path.dirname(json_file)

        if output_format == "markdown":
            return self.export_to_markdown(processed_data, output_dir)
        elif output_format == "html":
            return self.export_to_html(processed_data, output_dir)
        elif output_format == "pdf":
            return self.export_to_pdf(processed_data, output_dir)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def setup_models(self):
        """Initialize AI models"""
        self.logger.info("Loading AI models...")

        # Detect best available device
        self.device = self.get_best_device()
        self.logger.info(f"Using device: {self.device}")

        # Load Whisper model based on backend selection
        backend = self.config.get("whisper_backend", "faster-whisper")
        self.logger.info(f"Using Whisper backend: {backend}")

        if backend == "faster-whisper":
            self.setup_faster_whisper()
        else:
            self.setup_openai_whisper()

        # Temporarily disable diarization
        self.logger.warning("Diarization temporarily disabled due to dependency issues")
        self.config["enable_diarization"] = False

        # Temporarily disable CLIP visual analysis
        self.logger.warning(
            "Visual analysis temporarily disabled due to dependency issues"
        )
        self.config["enable_visual_analysis"] = False

        # Mark models as initialized
        self.models_initialized = True

    def setup_faster_whisper(self):
        """Setup faster-whisper backend with Apple Silicon optimization"""
        try:
            # Determine compute type
            compute_type = self.get_compute_type()

            # faster-whisper doesn't use MPS directly, but CTranslate2 is optimized for Apple Silicon
            # Prefer CUDA if available; otherwise CPU
            device = "cuda" if getattr(self, "device", "cpu") == "cuda" else "cpu"
            cpu_threads = max(1, (os.cpu_count() or 1))

            self.logger.info(
                f"Loading faster-whisper model: {self.config['whisper_model']} on {device} with {compute_type}"
            )

            self.whisper_model = WhisperModel(
                self.config["whisper_model"],
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )

            self.actual_device = device
            self.backend_type = "faster-whisper"
            self.logger.info(
                f"faster-whisper model loaded successfully with {compute_type} precision"
            )

        except Exception as e:
            self.logger.error(f"Failed to load faster-whisper: {e}")
            self.logger.info("Falling back to OpenAI Whisper...")
            self.setup_openai_whisper()

    def setup_openai_whisper(self):
        """Setup OpenAI Whisper backend with fallback mechanism"""
        try:
            self.whisper_model = whisper.load_model(
                self.config["whisper_model"], device=self.device
            )
            self.logger.info(
                f"OpenAI Whisper model loaded successfully on {self.device}"
            )
            self.actual_device = self.device
            self.backend_type = "openai-whisper"
        except Exception as e:
            if self.device == "mps":
                self.logger.warning(
                    f"MPS loading failed for Whisper ({str(e)[:100]}...), falling back to CPU"
                )
                self.whisper_model = whisper.load_model(
                    self.config["whisper_model"], device="cpu"
                )
                self.actual_device = "cpu"
                self.backend_type = "openai-whisper"
                self.logger.info(
                    "Note: CLIP models (when enabled) may still use MPS for better performance"
                )
            else:
                raise e

    def get_compute_type(self):
        """Determine the best compute type for faster-whisper"""
        compute_type = self.config.get("compute_type", "auto")

        if compute_type != "auto":
            return compute_type

        # Auto-detect best compute type for Apple Silicon
        import platform

        if platform.machine() == "arm64":  # Apple Silicon
            # int8 provides good speed/quality balance on Apple Silicon
            return "int8"
        else:
            # x86 systems
            return "int8"

    def get_best_device(self):
        """Determine the best available device for PyTorch operations"""
        # Check if user forced CPU usage
        if self.config.get("force_cpu", False):
            self.logger.info("CPU forced via configuration")
            return "cpu"

        # Check if user specified a device
        device_config = self.config.get("device", "auto").lower()
        if device_config != "auto":
            if (
                device_config == "mps"
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                self.logger.info("Using MPS (Metal Performance Shaders) as configured")
                return "mps"
            elif device_config == "cuda" and torch.cuda.is_available():
                self.logger.info("Using CUDA as configured")
                return "cuda"
            elif device_config == "cpu":
                self.logger.info("Using CPU as configured")
                return "cpu"
            else:
                self.logger.warning(
                    f"Configured device '{device_config}' not available, falling back to auto-detection"
                )

        # Auto-detect best device
        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            self.logger.info(
                "Apple Silicon MPS (Metal Performance Shaders) detected and selected"
            )
            return "mps"
        elif torch.cuda.is_available():
            self.logger.info("CUDA GPU detected and selected")
            return "cuda"
        else:
            self.logger.info("Using CPU (no GPU acceleration available)")
            return "cpu"

    def extract_audio(self, video_path, output_path):
        """Extract audio from video file"""
        self.logger.info("Extracting audio from video...")
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Audio extraction failed: {result.stderr}")
        return output_path

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper (supports both backends)"""
        self.logger.info(f"Transcribing audio using {self.backend_type}...")

        if self.backend_type == "faster-whisper":
            return self.transcribe_with_faster_whisper(audio_path)
        else:
            return self.transcribe_with_openai_whisper(audio_path)

    def transcribe_with_faster_whisper(self, audio_path):
        """Transcribe audio using faster-whisper backend"""
        segments, info = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None,  # Auto-detect
        )

        # Convert faster-whisper format to OpenAI Whisper format for compatibility
        segments_list = []
        for i, segment in enumerate(segments):
            segment_dict = {
                "id": i,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [],
            }

            # Add word-level timestamps if available
            if hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    segment_dict["words"].append(
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": getattr(word, "probability", 1.0),
                        }
                    )

            segments_list.append(segment_dict)

        return {
            "segments": segments_list,
            "language": info.language,
            "language_probability": info.language_probability,
        }

    def transcribe_with_openai_whisper(self, audio_path):
        """Transcribe audio using OpenAI Whisper backend"""
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None,  # Auto-detect
        )
        return result

    def perform_diarization(self, audio_path):
        """Perform speaker diarization"""
        if not self.config["enable_diarization"]:
            return None

        self.logger.info("Performing speaker diarization...")
        try:
            # Check if diarization pipeline is available
            if not hasattr(self, "diarization_pipeline"):
                self.logger.warning("Diarization pipeline not available")
                return None

            diarization = self.diarization_pipeline(audio_path)

            # Convert to list of segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.duration >= self.config["min_speaker_duration"]:
                    speaker_segments.append(
                        {
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker,
                            "duration": turn.duration,
                        }
                    )

            return speaker_segments
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return None

    def extract_frames(self, video_path, timestamps, output_dir):
        """Extract frames at specified timestamps"""
        self.logger.info("Extracting video frames...")

        # Create frames subdirectory
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = []

        for i, timestamp in enumerate(timestamps):
            # Seek by frame number if FPS is known, otherwise fallback to time-based seek
            if fps > 1e-6:
                frame_number = int(float(timestamp) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ret, frame = cap.read()

            if ret:
                filename = f"frame_{i:04d}_{timestamp:.2f}s.jpg"
                filepath = os.path.join(frames_dir, filename)

                # Save frame
                cv2.imwrite(
                    filepath,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config["screenshot_quality"]],
                )

                frames.append(
                    {
                        "index": i,
                        "timestamp": float(timestamp),
                        "filename": filename,
                        "filepath": filepath,
                        "frame": frame,
                    }
                )

        cap.release()
        return frames

    def analyze_visual_content(self, frames, transcript_segments):
        """Analyze visual content and match with transcript"""
        # Even if CLIP visual analysis is disabled, we can still do keyword-based matching
        self.logger.info("Analyzing visual content (keyword-based matching)...")
        visual_matches = []
        if not frames:
            return visual_matches

        # Extract text snippets that might reference visual content
        visual_keywords = [
            "look",
            "see",
            "show",
            "display",
            "screen",
            "image",
            "picture",
            "chart",
            "graph",
            "diagram",
            "slide",
            "video",
            "here",
            "this",
        ]

        for segment in transcript_segments:
            text = segment["text"].lower()
            if any(keyword in text for keyword in visual_keywords):
                # Find closest frame
                segment_time = segment["start"]
                try:
                    closest_frame = min(
                        frames, key=lambda x: abs(x["timestamp"] - segment_time)
                    )
                except ValueError:
                    continue

                if (
                    abs(closest_frame["timestamp"] - segment_time) <= 10
                ):  # Within 10 seconds
                    visual_matches.append(
                        {
                            "segment": segment,
                            "frame": closest_frame,
                            "relevance_score": self.calculate_visual_relevance(
                                segment["text"], closest_frame
                            ),
                        }
                    )

        return visual_matches

    def calculate_visual_relevance(self, text, frame):
        """Calculate relevance between text and visual content using CLIP"""
        try:
            # Check if CLIP models are available
            if not hasattr(self, "clip_processor") or not hasattr(self, "clip_model"):
                # Only log once by checking if we've already logged this warning
                if not hasattr(self, "_clip_warning_logged"):
                    self.logger.warning(
                        "CLIP models not available - using keyword-based relevance fallback"
                    )
                    self._clip_warning_logged = True
                return 0.5

            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame["frame"], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Process with CLIP - move inputs to device
            inputs = self.clip_processor(
                text=[text], images=image, return_tensors="pt", padding=True
            )

            # Move inputs to device if available
            if hasattr(self, "device") and self.device != "cpu":
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

            outputs = self.clip_model(**inputs)

            # Calculate similarity
            logits_per_image = outputs.logits_per_image
            similarity = torch.nn.functional.softmax(logits_per_image, dim=1)[0][
                0
            ].item()

            return similarity
        except Exception as e:
            self.logger.warning(f"Visual relevance calculation failed: {e}")
            return 0.5

    def merge_transcript_with_speakers(self, transcript, speaker_segments):
        """Merge transcript with speaker information"""
        if not speaker_segments:
            return transcript["segments"]

        merged_segments = []

        for segment in transcript["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]

            # Find overlapping speaker
            current_speaker = "Unknown"
            max_overlap = 0

            for speaker_seg in speaker_segments:
                overlap_start = max(segment_start, speaker_seg["start"])
                overlap_end = min(segment_end, speaker_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    current_speaker = speaker_seg["speaker"]

            segment["speaker"] = current_speaker
            merged_segments.append(segment)

        return merged_segments

    def export_to_markdown(self, processed_data, output_dir):
        """Export processed data to markdown format"""
        self.logger.info("Exporting to markdown...")

        metadata = processed_data["metadata"]
        transcript = processed_data["transcript"]
        visual = processed_data["visual"]

        video_name = metadata["video_name"]
        markdown_content = []

        # Header
        markdown_content.append(f"# Transcript: {video_name}")
        markdown_content.append(
            f"**Generated:** {metadata['generated_at'][:19].replace('T', ' ')}"
        )
        markdown_content.append(f"**Duration:** {metadata['duration_formatted']}")
        markdown_content.append("")

        # Summary section
        markdown_content.append("## Summary")
        markdown_content.append("*Auto-generated summary would go here*")
        markdown_content.append("")

        # Speakers section
        speakers = transcript["speakers"]
        if len(speakers) > 1:
            markdown_content.append("## Speakers")
            for speaker in sorted(speakers):
                markdown_content.append(f"- **{speaker}**")
            markdown_content.append("")

        # Transcript with timestamps and screenshots
        markdown_content.append("## Transcript")
        markdown_content.append("")

        current_speaker = None

        for segment in transcript["segments"]:
            timestamp = self.format_timestamp(segment["start"])
            speaker = segment.get("speaker", "Unknown")
            text = segment["text"].strip()

            # Add speaker change
            if speaker != current_speaker:
                if current_speaker is not None:
                    markdown_content.append("")
                markdown_content.append(f"### {speaker}")
                current_speaker = speaker

            # Add timestamp on its own line, then text
            markdown_content.append(f"**[{timestamp}]**")
            markdown_content.append(f"{text}")
            markdown_content.append("")

            # Add relevant screenshots
            relevant_matches = [
                match
                for match in visual["matches"]
                if match["segment"]["start"] == segment["start"]
            ]

            for match in relevant_matches:
                rel_path = match["frame"]["relative_path"]
                markdown_content.append(f"![Screenshot at {timestamp}]({rel_path})")
                markdown_content.append("")

        # Write markdown file
        markdown_file = os.path.join(output_dir, f"{video_name}_transcript.md")
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))

        return markdown_file

    def export_to_html(self, processed_data, output_dir):
        """Export processed data to HTML format"""
        self.logger.info("Exporting to HTML...")

        # First create markdown, then convert to HTML
        markdown_file = self.export_to_markdown(processed_data, output_dir)

        with open(markdown_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=["extra", "codehilite"])

        # Embed images as base64
        html_with_embedded = self.embed_images_in_html(html_content, output_dir)

        # Create complete HTML document
        video_name = processed_data["metadata"]["video_name"]
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Transcript: {video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
{html_with_embedded}
</body>
</html>"""

        # Write HTML file
        html_file = os.path.join(output_dir, f"{video_name}_transcript.html")
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(full_html)

        return html_file

    def export_to_pdf(self, processed_data, output_dir):
        """Export processed data to PDF format using ReportLab"""
        self.logger.info("Exporting to PDF using ReportLab...")

        try:
            metadata = processed_data["metadata"]
            transcript = processed_data["transcript"]
            visual = processed_data["visual"]

            video_name = metadata["video_name"]
            pdf_file = os.path.join(output_dir, f"{video_name}_transcript.pdf")

            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_file,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )

            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=16,
                spaceAfter=30,
            )
            heading_style = ParagraphStyle(
                "CustomHeading",
                parent=styles["Heading2"],
                fontSize=14,
                spaceAfter=12,
            )
            timestamp_style = ParagraphStyle(
                "Timestamp",
                parent=styles["Normal"],
                fontSize=10,
                textColor=gray,
                fontName="Helvetica-Bold",
                spaceAfter=6,
            )
            content_style = ParagraphStyle(
                "Content",
                parent=styles["Normal"],
                fontSize=11,
                spaceAfter=12,
            )

            # Build story
            story = []

            # Title
            story.append(Paragraph(f"Transcript: {video_name}", title_style))
            story.append(
                Paragraph(
                    f"Generated: {metadata['generated_at'][:19].replace('T', ' ')}",
                    styles["Normal"],
                )
            )
            story.append(
                Paragraph(
                    f"Duration: {metadata['duration_formatted']}", styles["Normal"]
                )
            )
            story.append(Spacer(1, 20))

            # Summary
            story.append(Paragraph("Summary", heading_style))
            story.append(
                Paragraph("Auto-generated summary would go here", styles["Italic"])
            )
            story.append(Spacer(1, 20))

            # Speakers
            speakers = transcript["speakers"]
            if len(speakers) > 1:
                story.append(Paragraph("Speakers", heading_style))
                for speaker in speakers:
                    story.append(Paragraph(f"• {speaker}", styles["Normal"]))
                story.append(Spacer(1, 20))

            # Transcript
            story.append(Paragraph("Transcript", heading_style))

            current_speaker = None
            for segment in transcript["segments"]:
                timestamp = self.format_timestamp(segment["start"])
                speaker = segment.get("speaker", "Unknown")
                text = segment["text"].strip()

                # Add speaker change
                if speaker != current_speaker:
                    if current_speaker is not None:
                        story.append(Spacer(1, 12))
                    story.append(Paragraph(speaker, heading_style))
                    current_speaker = speaker

                # Add timestamp and text on separate lines
                story.append(Paragraph(f"[{timestamp}]", timestamp_style))
                story.append(Paragraph(text, content_style))

                # Add relevant screenshots
                relevant_matches = [
                    match
                    for match in visual["matches"]
                    if match["segment"]["start"] == segment["start"]
                ]

                for match in relevant_matches:
                    img_path = match["frame"]["filepath"]
                    if os.path.exists(img_path):
                        try:
                            # Add image with caption
                            story.append(
                                RLImage(img_path, width=4 * inch, height=3 * inch)
                            )
                            story.append(
                                Paragraph(
                                    f"Screenshot at {timestamp}", styles["Normal"]
                                )
                            )
                            story.append(Spacer(1, 12))
                        except Exception as e:
                            self.logger.warning(f"Could not add image to PDF: {e}")

            # Build PDF
            doc.build(story)
            self.logger.info(f"PDF exported successfully: {pdf_file}")
            return pdf_file

        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            return None

    def embed_images_in_html(self, html_content, base_path):
        """Embed images as base64 in HTML (robust to attribute order)"""
        import base64
        import re

        def replace_img_tag(match):
            tag = match.group(0)
            src_m = re.search(r'src="([^"]+)"', tag, flags=re.IGNORECASE)
            if not src_m:
                return tag
            alt_m = re.search(r'alt="([^"]*)"', tag, flags=re.IGNORECASE)
            img_path = src_m.group(1)
            alt_text = alt_m.group(1) if alt_m else ""
            full_path = os.path.join(base_path, img_path)
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                return f'<img src="data:image/jpeg;base64,{img_data}" alt="{alt_text}">'
            return tag

        return re.sub(
            r"<img\s+[^>]*>", replace_img_tag, html_content, flags=re.IGNORECASE
        )

    def format_timestamp(self, seconds):
        """Format timestamp as HH:MM:SS (handles 24h+ and float seconds)"""
        try:
            total = int(round(float(seconds)))
        except Exception:
            return "00:00:00"
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def process_video(self, video_path, output_dir=None):
        """Main processing pipeline"""
        # Initialize models if not already done (allows for config overrides)
        if not self.models_initialized:
            self.setup_models()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_dir is None:
            output_dir = self.config["output_dir"]

        os.makedirs(output_dir, exist_ok=True)

        video_name = Path(video_path).stem
        temp_dir = tempfile.mkdtemp()

        try:
            # Check if processed data already exists
            json_file = os.path.join(output_dir, f"{video_name}_processed.json")
            if os.path.exists(json_file):
                self.logger.info(f"Found existing processed data: {json_file}")
                self.logger.info(
                    "Using existing data. Delete the JSON file to reprocess from scratch."
                )

                # Export to requested format using existing data
                output_file = self.export_from_json(
                    json_file, self.config["output_format"], output_dir
                )

                results = {
                    "processed_data": json_file,
                    self.config["output_format"]: output_file,
                }

                return results

            # Extract audio
            audio_path = os.path.join(temp_dir, f"{video_name}.wav")
            self.extract_audio(video_path, audio_path)

            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)

            # Perform speaker diarization
            speaker_segments = self.perform_diarization(audio_path)

            # Merge transcript with speaker information
            merged_segments = self.merge_transcript_with_speakers(
                transcript, speaker_segments
            )

            # Extract frames at regular intervals and when referenced
            frame_timestamps = []

            # Regular interval screenshots
            if transcript.get("segments"):
                try:
                    duration = (
                        max((seg.get("end") or 0) for seg in transcript["segments"])
                        if isinstance(transcript["segments"], list)
                        else 0
                    )
                except Exception:
                    duration = 0
            else:
                duration = 0
            for t in np.arange(0, duration, self.config["screenshot_interval"]):
                frame_timestamps.append(t)

            # Extract frames
            frames = self.extract_frames(video_path, frame_timestamps, output_dir)

            # Analyze visual content
            visual_matches = self.analyze_visual_content(frames, merged_segments)

            # Save processed data to JSON for future re-exports
            json_file = self.save_processed_data(
                video_path, merged_segments, frames, visual_matches, output_dir
            )

            # Export to requested format using the stored data
            output_file = self.export_from_json(
                json_file, self.config["output_format"], output_dir
            )

            results = {
                "processed_data": json_file,
                self.config["output_format"]: output_file,
            }

            self.logger.info(f"Processing complete! Output files: {results}")
            return results

        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
