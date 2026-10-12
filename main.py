"""
DAÏO Main Application Launcher.
Author: Al-Amin Abubakar

Usage:
    python main.py                   # Launch Desktop GUI (ttkbootstrap)
    python main.py --gui             # Launch Desktop GUI
    python main.py --video <file>    # Run detector on video file
    python main.py --webcam          # Run detector on live webcam
    python main.py --benchmark       # Run Section 4.2 Ablation Study Benchmark
"""

import sys
import argparse
import os

from TestFallDetection import process_video_stream
from tools.benchmark_ablation import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="DAÏO: High-Specificity Real-Time Vision Fall Detection with Extended Kalman Filter"
    )
    parser.add_argument("--gui", action="store_true", help="Launch interactive Desktop GUI")
    parser.add_argument("--video", type=str, default=None, help="Path to video file to process")
    parser.add_argument("--webcam", action="store_true", help="Process live camera stream (Webcam 0)")
    parser.add_argument("--benchmark", action="store_true", help="Run Ablation Study and benchmark test cases")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.webcam:
        process_video_stream("0")
    elif args.video:
        process_video_stream(args.video)
    else:
        # Default behavior: Launch desktop GUI
        try:
            import ttkbootstrap as tb
            from app_gui import FallDetectionApp

            app_window = tb.Window(themename="darkly")
            _ = FallDetectionApp(app_window)
            app_window.mainloop()
        except Exception as e:
            print(f"[DAÏO] Unable to start GUI ({e}). Falling back to CLI mode...")
            demo_path = os.path.join("DataVideos", "fall_forward.mp4")
            if os.path.exists(demo_path):
                process_video_stream(demo_path)
            else:
                parser.print_help()


if __name__ == "__main__":
    main()
