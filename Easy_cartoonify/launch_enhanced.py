#!/usr/bin/env python3
"""
Cartoon Image Generator Pro - Enhanced Launcher
Launch the upgraded app with all premium features
"""
import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

def main():
    """Launch the enhanced app"""
    root = tk.Tk()
    root.withdraw()
    
    # Check dependencies
    print("Checking dependencies...")
    missing = []
    
    try:
        import cv2
    except:
        missing.append("opencv-python")
    
    try:
        import torch
    except:
        missing.append("torch")
    
    try:
        import PIL
    except:
        missing.append("pillow")
    
    if missing:
        messagebox.showerror(
            'Missing Dependencies',
            f'Please install: {", ".join(missing)}\n\nRun: pip install {" ".join(missing)}'
        )
        root.destroy()
        return
    
    # Check enhanced processors
    app_dir = Path(__file__).parent
    if not (app_dir / 'enhanced_processor.py').exists():
        messagebox.showerror(
            'Missing Files',
            'enhanced_processor.py not found in app directory'
        )
        root.destroy()
        return
    
    if not (app_dir / 'chatgpt_processor.py').exists():
        messagebox.showerror(
            'Missing Files',
            'chatgpt_processor.py not found in app directory'
        )
        root.destroy()
        return
    
    # Launch app
    print("Launching Cartoon Image Generator Pro...")
    root.destroy()
    
    try:
        subprocess.run([sys.executable, str(app_dir / 'app_gui_enhanced.py')], check=True)
    except Exception as e:
        messagebox.showerror('Error', f'Failed to launch app: {e}')

if __name__ == '__main__':
    main()
