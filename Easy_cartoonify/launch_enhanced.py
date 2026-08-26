#!/usr/bin/env python3
"""
Cartoon Image Generator Pro - Enhanced Launcher
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def main():
    print("Launching Cartoonify Studio Pro...")
    try:
        from cartoonify.gui.app import main as run_gui
        run_gui()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        # Run Web fallback
        from cartoonify.cli import main as cli_main
        cli_main()


if __name__ == '__main__':
    main()
