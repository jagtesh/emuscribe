#!/usr/bin/env python3
"""
Video Transcription Tool with AI and Diarization
A comprehensive tool for transcribing videos with speaker identification
and intelligent screenshot capture.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import librosa
import markdown
import numpy as np
import soundfile as sf
import torch
import weasyprint
import whisper
from PIL import Image
from pyannote.audio import Pipeline
from transformers import CLIPModel, CLIPProcessor


class VideoTranscriber:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_models()

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

    def setup_models(self):
        """Initialize AI models"""
        self.logger.info("Loading AI models...")

        # Load Whisper model
        self.whisper_model = whisper.load_model(self.config["whisper_model"])

        # Load diarization pipeline (requires HuggingFace token)
        if self.config["enable_diarization"]:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
                )
            except Exception as e:
                self.logger.warning(f"Could not load diarization model: {e}")
                self.config["enable_diarization"] = False

        # Load CLIP for visual analysis
        if self.config["enable_visual_analysis"]:
            try:
                self.clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self.clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            except Exception as e:
                self.logger.warning(f"Could not load CLIP model: {e}")
                self.config["enable_visual_analysis"] = False

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
        """Transcribe audio using Whisper"""
        self.logger.info("Transcribing audio...")
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

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                filename = f"frame_{i:04d}_{timestamp:.2f}s.jpg"
                filepath = os.path.join(output_dir, filename)

                # Save frame
                cv2.imwrite(
                    filepath,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config["screenshot_quality"]],
                )

                frames.append(
                    {
                        "timestamp": timestamp,
                        "filename": filename,
                        "filepath": filepath,
                        "frame": frame,
                    }
                )

        cap.release()
        return frames

    def analyze_visual_content(self, frames, transcript_segments):
        """Analyze visual content and match with transcript"""
        if not self.config["enable_visual_analysis"]:
            return []

        self.logger.info("Analyzing visual content...")
        visual_matches = []

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
                closest_frame = min(
                    frames, key=lambda x: abs(x["timestamp"] - segment_time)
                )

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
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame["frame"], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Process with CLIP
            inputs = self.clip_processor(
                text=[text], images=image, return_tensors="pt", padding=True
            )
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

    def generate_markdown_output(
        self, transcript_segments, frames, visual_matches, video_path, output_dir
    ):
        """Generate markdown output with embedded screenshots"""
        self.logger.info("Generating markdown output...")

        video_name = Path(video_path).stem
        markdown_content = []

        # Header
        markdown_content.append(f"# Transcript: {video_name}")
        markdown_content.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        markdown_content.append(
            f"**Duration:** {self.format_timestamp(transcript_segments[-1]['end'])}"
        )
        markdown_content.append("")

        # Summary section
        markdown_content.append("## Summary")
        markdown_content.append("*Auto-generated summary would go here*")
        markdown_content.append("")

        # Speakers section
        speakers = set(seg.get("speaker", "Unknown") for seg in transcript_segments)
        if len(speakers) > 1:
            markdown_content.append("## Speakers")
            for i, speaker in enumerate(sorted(speakers), 1):
                markdown_content.append(f"- **{speaker}**")
            markdown_content.append("")

        # Transcript with timestamps and screenshots
        markdown_content.append("## Transcript")
        markdown_content.append("")

        current_speaker = None

        for segment in transcript_segments:
            timestamp = self.format_timestamp(segment["start"])
            speaker = segment.get("speaker", "Unknown")
            text = segment["text"].strip()

            # Add speaker change
            if speaker != current_speaker:
                if current_speaker is not None:
                    markdown_content.append("")
                markdown_content.append(f"### {speaker}")
                current_speaker = speaker

            # Add timestamp and text
            markdown_content.append(f"**[{timestamp}]** {text}")

            # Add relevant screenshots
            relevant_frames = [
                vm
                for vm in visual_matches
                if vm["segment"]["start"] == segment["start"]
            ]

            for match in relevant_frames:
                frame_path = match["frame"]["filepath"]
                rel_path = os.path.relpath(frame_path, output_dir)
                markdown_content.append(f"![Screenshot at {timestamp}]({rel_path})")
                markdown_content.append("")

        # Write markdown file
        markdown_file = os.path.join(output_dir, f"{video_name}_transcript.md")
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))

        return markdown_file

    def export_to_html(self, markdown_file, output_dir):
        """Export markdown to HTML with embedded images"""
        self.logger.info("Exporting to HTML...")

        with open(markdown_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=["extra", "codehilite"])

        # Embed images as base64
        html_with_embedded = self.embed_images_in_html(
            html_content, os.path.dirname(markdown_file)
        )

        # Create complete HTML document
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Video Transcript</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
               max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        h1, h2, h3 {{ color: #333; }}
        .timestamp {{ color: #666; font-weight: bold; }}
    </style>
</head>
<body>
{html_with_embedded}
</body>
</html>
        """

        html_file = markdown_file.replace(".md", ".html")
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(full_html)

        return html_file

    def embed_images_in_html(self, html_content, base_path):
        """Embed images as base64 in HTML"""
        import base64
        import re

        def replace_img(match):
            img_path = match.group(1)
            full_path = os.path.join(base_path, img_path)

            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                    return f'<img src="data:image/jpeg;base64,{img_data}" alt="{match.group(2)}">'
            return match.group(0)

        return re.sub(
            r'<img src="([^"]+)" alt="([^"]*)"[^>]*>', replace_img, html_content
        )

    def export_to_pdf(self, html_file, output_dir):
        """Export HTML to PDF"""
        self.logger.info("Exporting to PDF...")

        pdf_file = html_file.replace(".html", ".pdf")

        try:
            weasyprint.HTML(filename=html_file).write_pdf(pdf_file)
            return pdf_file
        except Exception as e:
            self.logger.error(f"PDF export failed: {e}")
            return None

    def format_timestamp(self, seconds):
        """Format timestamp as HH:MM:SS"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def process_video(self, video_path, output_dir=None):
        """Main processing pipeline"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_dir is None:
            output_dir = self.config["output_dir"]

        os.makedirs(output_dir, exist_ok=True)

        video_name = Path(video_path).stem
        temp_dir = tempfile.mkdtemp()

        try:
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
            duration = transcript["segments"][-1]["end"]
            for t in np.arange(0, duration, self.config["screenshot_interval"]):
                frame_timestamps.append(t)

            # Extract frames
            frames = self.extract_frames(video_path, frame_timestamps, output_dir)

            # Analyze visual content
            visual_matches = self.analyze_visual_content(frames, merged_segments)

            # Generate markdown output
            markdown_file = self.generate_markdown_output(
                merged_segments, frames, visual_matches, video_path, output_dir
            )

            # Export to other formats
            results = {"markdown": markdown_file}

            if self.config["output_format"] in ["html", "pdf"]:
                html_file = self.export_to_html(markdown_file, output_dir)
                results["html"] = html_file

                if self.config["output_format"] == "pdf":
                    pdf_file = self.export_to_pdf(html_file, output_dir)
                    if pdf_file:
                        results["pdf"] = pdf_file

            self.logger.info(f"Processing complete! Output files: {results}")
            return results

        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Video Transcription Tool with AI and Diarization"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output directory", default="./output")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "pdf"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--no-diarization", action="store_true", help="Disable speaker diarization"
    )
    parser.add_argument(
        "--no-visual", action="store_true", help="Disable visual analysis"
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="Screenshot interval in seconds"
    )

    args = parser.parse_args()

    # Create transcriber
    transcriber = VideoTranscriber(args.config)

    # Override config with command line arguments
    if args.no_diarization:
        transcriber.config["enable_diarization"] = False
    if args.no_visual:
        transcriber.config["enable_visual_analysis"] = False
    transcriber.config["output_format"] = args.format
    transcriber.config["screenshot_interval"] = args.interval

    try:
        results = transcriber.process_video(args.video, args.output)
        print(f"\n✅ Processing complete!")
        print(f"📁 Output directory: {args.output}")
        for format_type, file_path in results.items():
            print(f"📄 {format_type.upper()}: {file_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
