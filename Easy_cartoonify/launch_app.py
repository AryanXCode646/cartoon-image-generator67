#!/usr/bin/env python3
"""
Launcher for the Cartoon Image Generator GUI App
Runs app_gui.py with proper error handling
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app_gui import CartoonifierApp
    import tkinter as tk
    
    if __name__ == '__main__':
        root = tk.Tk()
        app = CartoonifierApp(root)
        root.mainloop()
except ImportError as e:
    print(f"Error: Missing required module - {e}")
    print("Please ensure you have installed: opencv-python, torch, pillow")
    sys.exit(1)
except Exception as e:
    print(f"Error starting app: {e}")
    sys.exit(1)
