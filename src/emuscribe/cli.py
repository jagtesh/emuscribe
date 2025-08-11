#!/usr/bin/env python3
"""
Command-line interface for emuscribe video transcription tool
"""

import argparse
import os
import sys

from .transcriber import VideoTranscriber


def main():
    parser = argparse.ArgumentParser(
        description="Video Transcription Tool with AI and Diarization"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command (default behavior)
    process_parser = subparsers.add_parser("process", help="Process a video file")
    process_parser.add_argument("video", help="Path to video file")
    process_parser.add_argument(
        "-o", "--output", help="Output directory", default="./output"
    )
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
    process_parser.add_argument(
        "--backend",
        choices=["openai-whisper", "faster-whisper"],
        help="Whisper backend to use (overrides config)",
    )

    # Export command (for re-exporting from stored data)
    export_parser = subparsers.add_parser(
        "export", help="Export from previously processed data"
    )
    export_parser.add_argument("json_file", help="Path to processed JSON file")
    export_parser.add_argument(
        "--format",
        choices=["markdown", "html", "pdf"],
        required=True,
        help="Export format",
    )
    export_parser.add_argument(
        "-o", "--output", help="Output directory (defaults to JSON file directory)"
    )
    export_parser.add_argument("-c", "--config", help="Configuration file path")

    # Legacy support: if first argument looks like a video file, use process command
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and sys.argv[1] not in ["process", "export"]
    ):
        # Insert 'process' command for backward compatibility
        sys.argv.insert(1, "process")

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
            if hasattr(args, "no_diarization") and args.no_diarization:
                transcriber.config["enable_diarization"] = False
            if hasattr(args, "no_visual") and args.no_visual:
                transcriber.config["enable_visual_analysis"] = False
            transcriber.config["output_format"] = args.format
            if hasattr(args, "interval"):
                transcriber.config["screenshot_interval"] = args.interval
            if hasattr(args, "backend") and args.backend:
                transcriber.config["whisper_backend"] = args.backend

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

