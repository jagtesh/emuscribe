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
# import weasyprint  # Temporarily disabled due to system lib issues
import whisper
from PIL import Image
# from pyannote.audio import Pipeline  # Temporarily disabled
# from transformers import CLIPModel, CLIPProcessor  # Temporarily disabled


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

    def save_processed_data(self, video_path, transcript_segments, frames, visual_matches, output_dir):
        """Save all processed data to JSON format for later export"""
        video_name = Path(video_path).stem
        json_file = os.path.join(output_dir, f"{video_name}_processed.json")
        
        # Create the comprehensive data structure
        processed_data = {
            "metadata": {
                "video_name": video_name,
                "video_path": str(video_path),
                "generated_at": datetime.now().isoformat(),
                "duration": transcript_segments[-1]["end"] if transcript_segments else 0,
                "duration_formatted": self.format_timestamp(transcript_segments[-1]["end"]) if transcript_segments else "00:00:00",
                "processing_config": {
                    "whisper_model": self.config["whisper_model"],
                    "screenshot_interval": self.config["screenshot_interval"],
                    "diarization_enabled": self.config["enable_diarization"],
                    "visual_analysis_enabled": self.config["enable_visual_analysis"]
                }
            },
            "transcript": {
                "segments": transcript_segments,
                "speakers": sorted(list(set(seg.get("speaker", "Unknown") for seg in transcript_segments)))
            },
            "visual": {
                "frames": [
                    {
                        "index": frame["index"],
                        "timestamp": frame["timestamp"],
                        "timestamp_formatted": self.format_timestamp(frame["timestamp"]),
                        "filepath": frame["filepath"],
                        "relative_path": os.path.relpath(frame["filepath"], output_dir)
                    }
                    for frame in frames
                ],
                "matches": [
                    {
                        "segment": match["segment"],
                        "frame": {
                            "index": match["frame"]["index"],
                            "timestamp": match["frame"]["timestamp"],
                            "timestamp_formatted": self.format_timestamp(match["frame"]["timestamp"]),
                            "filepath": match["frame"]["filepath"],
                            "relative_path": os.path.relpath(match["frame"]["filepath"], output_dir)
                        },
                        "relevance_score": match["relevance_score"]
                    }
                    for match in visual_matches
                ]
            }
        }
        
        # Save to JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Processed data saved to {json_file}")
        return json_file

    def load_processed_data(self, json_file):
        """Load processed data from JSON file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def export_from_json(self, json_file, output_format, output_dir=None):
        """Export processed data to specified format"""
        processed_data = self.load_processed_data(json_file)
        
        if output_dir is None:
            output_dir = os.path.dirname(json_file)
        
        video_name = processed_data["metadata"]["video_name"]
        
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

        # Load Whisper model
        self.whisper_model = whisper.load_model(self.config["whisper_model"])

        # Temporarily disable diarization
        self.logger.warning("Diarization temporarily disabled due to dependency issues")
        self.config["enable_diarization"] = False

        # Temporarily disable CLIP visual analysis
        self.logger.warning("Visual analysis temporarily disabled due to dependency issues")
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
            # Check if diarization pipeline is available
            if not hasattr(self, 'diarization_pipeline'):
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
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
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
        # Even if CLIP visual analysis is disabled, we can still do keyword-based matching
        self.logger.info("Analyzing visual content (keyword-based matching)...")
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
            # Check if CLIP models are available
            if not hasattr(self, 'clip_processor') or not hasattr(self, 'clip_model'):
                # Only log once by checking if we've already logged this warning
                if not hasattr(self, '_clip_warning_logged'):
                    self.logger.warning("CLIP models not available - using keyword-based relevance fallback")
                    self._clip_warning_logged = True
                return 0.5
                
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
        markdown_content.append(f"**Generated:** {metadata['generated_at'][:19].replace('T', ' ')}")
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

            # Add timestamp and text
            markdown_content.append(f"**[{timestamp}]** {text}")

            # Add relevant screenshots
            relevant_matches = [
                match for match in visual["matches"]
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
        """Export processed data to PDF format"""
        try:
            import weasyprint
        except ImportError:
            self.logger.error("WeasyPrint not available for PDF export")
            return None

        self.logger.info("Exporting to PDF...")
        
        # First create HTML, then convert to PDF
        html_file = self.export_to_html(processed_data, output_dir)
        
        video_name = processed_data["metadata"]["video_name"]
        pdf_file = os.path.join(output_dir, f"{video_name}_transcript.pdf")
        
        try:
            weasyprint.HTML(filename=html_file).write_pdf(pdf_file)
            return pdf_file
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            return None



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
            # Check if processed data already exists
            json_file = os.path.join(output_dir, f"{video_name}_processed.json")
            if os.path.exists(json_file):
                self.logger.info(f"Found existing processed data: {json_file}")
                self.logger.info("Using existing data. Delete the JSON file to reprocess from scratch.")
                
                # Export to requested format using existing data
                output_file = self.export_from_json(
                    json_file, self.config["output_format"], output_dir
                )
                
                results = {
                    "processed_data": json_file,
                    self.config["output_format"]: output_file
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
            if transcript["segments"]:
                segments = transcript["segments"]
                if isinstance(segments, list) and len(segments) > 0:
                    last_segment = segments[-1]
                    duration = last_segment.get("end", 0)
                else:
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
                self.config["output_format"]: output_file
            }

            self.logger.info(f"Processing complete! Output files: {results}")
            return results

        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Video Transcription Tool with AI and Diarization"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command (default behavior)
    process_parser = subparsers.add_parser("process", help="Process a video file")
    process_parser.add_argument("video", help="Path to video file")
    process_parser.add_argument("-o", "--output", help="Output directory", default="./output")
    process_parser.add_argument("-c", "--config", help="Configuration file path")
    process_parser.add_argument(
        "--format",
        choices=["markdown", "html", "pdf"],
        default="markdown",
        help="Output format",
    )
    process_parser.add_argument(
        "--no-diarization", action="store_true", help="Disable speaker diarization"
    )
    process_parser.add_argument(
        "--no-visual", action="store_true", help="Disable visual analysis"
    )
    process_parser.add_argument(
        "--interval", type=int, default=30, help="Screenshot interval in seconds"
    )
    
    # Export command (for re-exporting from stored data)
    export_parser = subparsers.add_parser("export", help="Export from previously processed data")
    export_parser.add_argument("json_file", help="Path to processed JSON file")
    export_parser.add_argument(
        "--format",
        choices=["markdown", "html", "pdf"],
        required=True,
        help="Export format",
    )
    export_parser.add_argument("-o", "--output", help="Output directory (defaults to JSON file directory)")
    export_parser.add_argument("-c", "--config", help="Configuration file path")
    
    # Legacy support: if first argument looks like a video file, use process command
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in ['process', 'export']:
        # Insert 'process' command for backward compatibility
        sys.argv.insert(1, 'process')
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create transcriber
    transcriber = VideoTranscriber(args.config)
    
    try:
        if args.command == "process":
            # Override config with command line arguments
            if hasattr(args, 'no_diarization') and args.no_diarization:
                transcriber.config["enable_diarization"] = False
            if hasattr(args, 'no_visual') and args.no_visual:
                transcriber.config["enable_visual_analysis"] = False
            transcriber.config["output_format"] = args.format
            if hasattr(args, 'interval'):
                transcriber.config["screenshot_interval"] = args.interval
            
            results = transcriber.process_video(args.video, args.output)
            print("\n✅ Processing complete!")
            print(f"📁 Output directory: {args.output}")
            for format_type, file_path in results.items():
                print(f"📄 {format_type.upper()}: {file_path}")
                
        elif args.command == "export":
            if not os.path.exists(args.json_file):
                print(f"❌ Error: JSON file not found: {args.json_file}")
                sys.exit(1)
                
            output_file = transcriber.export_from_json(
                args.json_file, args.format, args.output
            )
            
            print("\n✅ Export complete!")
            print(f"📄 {args.format.upper()}: {output_file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
