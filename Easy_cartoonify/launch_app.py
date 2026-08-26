#!/usr/bin/env python3
"""
Cartoon Image Generator Launcher.
"""
import sys
import subprocess
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
        print(f"Error launching GUI directly: {e}")
        # Fallback to subprocess
        gui_script = root_dir / "cartoonify" / "gui" / "app.py"
        subprocess.run([sys.executable, str(gui_script)])


if __name__ == '__main__':
    main()
