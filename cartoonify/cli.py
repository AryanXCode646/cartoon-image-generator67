"""
Command-line interface for Cartoonify.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

# Fix Windows console UTF-8 encoding for emojis
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from cartoonify.engine import STYLES, CartoonEngine
from cartoonify.utils import HistoryManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartoonify",
        description="Cartoonify - Professional AI & Computer Vision Image Stylizer",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: process
    proc_parser = subparsers.add_parser("process", help="Transform a single image into cartoon artwork")
    proc_parser.add_argument("input", type=str, help="Path to input image file")
    proc_parser.add_argument("-o", "--output", type=str, default=None, help="Output image file path")
    proc_parser.add_argument(
        "-s",
        "--style",
        type=str,
        default="ghibli_pro",
        choices=list(STYLES.keys()),
        help="Cartoon style preset",
    )
    proc_parser.add_argument("--strength", type=float, default=0.8, help="Stylization strength (0.1 - 1.0)")
    proc_parser.add_argument("--face-align", action="store_true", help="Enable face-aligned feathered blend")
    proc_parser.add_argument("--max-dim", type=int, default=1600, help="Maximum image dimension clamp")

    # Command: batch
    batch_parser = subparsers.add_parser("batch", help="Batch process an entire directory of images")
    batch_parser.add_argument("input_dir", type=str, help="Directory containing input images")
    batch_parser.add_argument("-o", "--output-dir", type=str, default="cartoon_output", help="Output directory")
    batch_parser.add_argument(
        "-s",
        "--style",
        type=str,
        default="ghibli_pro",
        choices=list(STYLES.keys()),
        help="Cartoon style preset",
    )
    batch_parser.add_argument("--strength", type=float, default=0.8, help="Stylization strength (0.1 - 1.0)")
    batch_parser.add_argument("--face-align", action="store_true", help="Enable face alignment")

    # Command: web
    web_parser = subparsers.add_parser("web", help="Launch the interactive Web Studio")
    web_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    web_parser.add_argument("--port", type=int, default=8000, help="Port number")
    web_parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    # Command: gui
    subparsers.add_parser("gui", help="Launch the Desktop GUI application")

    # Command: list-styles
    subparsers.add_parser("list-styles", help="List all available cartoon styles and descriptions")

    return parser


def cmd_process(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file does not exist: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else in_path.parent / f"cartoon_{args.style}_{in_path.name}"

    print(f"🎨 Cartoonifying '{in_path.name}' with style '{args.style}'...")
    engine = CartoonEngine()
    try:
        saved = engine.process_file(
            in_path,
            out_path,
            style=args.style,
            strength=args.strength,
            use_face_align=args.face_align,
            max_dimension=args.max_dim,
        )
        print(f"✅ Successfully created: {saved}")
        return 0
    except Exception as e:
        print(f"❌ Error during processing: {e}", file=sys.stderr)
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"Error: Input directory does not exist: {in_dir}", file=sys.stderr)
        return 1

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files = [f for ext in extensions for f in in_dir.glob(ext) if not f.name.startswith("cartoon_")]
    if not files:
        print(f"No image files found in {in_dir}")
        return 0

    print(f"🚀 Found {len(files)} images to process in '{in_dir}'...")
    engine = CartoonEngine()

    def on_progress(current: int, total: int, out_file: str):
        print(f"[{current}/{total}] Wrote {Path(out_file).name}")

    results = engine.process_batch(
        files,
        args.output_dir,
        style=args.style,
        strength=args.strength,
        use_face_align=args.face_align,
        progress_callback=on_progress,
    )
    print(f"✨ Batch complete! {len(results)} images saved to '{args.output_dir}'.")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    try:
        import uvicorn
        from cartoonify.api.app import app
    except ImportError:
        print("FastAPI / Uvicorn not installed.", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}"
    print(f"🌐 Starting Cartoonify Web Studio at {url}")
    if not args.no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_gui() -> int:
    try:
        import tkinter as tk
        from cartoonify.gui.app import ModernCartoonifierApp
    except ImportError as e:
        print(f"GUI dependencies missing: {e}", file=sys.stderr)
        return 1

    root = tk.Tk()
    app = ModernCartoonifierApp(root)
    root.mainloop()
    return 0


def cmd_list_styles() -> int:
    print("\n🌟 Available Cartoonify Styles:")
    print("=" * 80)
    for style in STYLES.values():
        neural_tag = "[Neural AI]" if style.is_neural else "[Artistic CV]"
        print(f"{style.icon}  {style.key:<16} | {style.name:<22} {neural_tag:<14}")
        print(f"    └─ {style.description}\n")
    return 0


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        # Default to launching Web Studio if run without arguments
        print("No command provided. Defaulting to Web Studio...")
        sys.exit(cmd_web(argparse.Namespace(host="127.0.0.1", port=8000, no_browser=False)))

    args = parser.parse_args()

    if args.command == "process":
        sys.exit(cmd_process(args))
    elif args.command == "batch":
        sys.exit(cmd_batch(args))
    elif args.command == "web":
        sys.exit(cmd_web(args))
    elif args.command == "gui":
        sys.exit(cmd_gui())
    elif args.command == "list-styles":
        sys.exit(cmd_list_styles())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
