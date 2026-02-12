"""CLI entry point for KugelAudio UI."""

import argparse

from . import launch_app


def main():
    parser = argparse.ArgumentParser(description="Launch KugelAudio Gradio UI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server hostname (default: 127.0.0.1, use 0.0.0.0 for network access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    
    args = parser.parse_args()
    
    print(f"🎙️ Starting KugelAudio UI on {args.host}:{args.port}")
    if args.share:
        print("📡 Creating public share link...")
    
    launch_app(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
