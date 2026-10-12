"""
TestFallDetection.py
Preserves the exact script entrypoint referenced in Figure 18 of the project report:
subprocess.run(['python', 'TestFallDetection.py', video_path])

Runs the DAÏO Extended Kalman Filter Fall Detection pipeline on the specified video,
overlays real-time HUD with skeletal points and velocity vectors,
and saves confirmed fall snapshots to Sauvegrade/.
"""

import sys
import os
import cv2
from daio.pipeline import DAIOFallDetector
from daio.state_machine import FallState


def process_video_stream(video_path: str):
    if not os.path.exists(video_path) and not video_path.isdigit():
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)

    src = int(video_path) if video_path.isdigit() else video_path
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"Error: Failed to open video source '{video_path}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = DAIOFallDetector(fps=fps, save_dir="Sauvegrade", enable_audio=True)

    window_name = "DAÏO - Real-Time Fall Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    print(f"[DAÏO] Starting video processing on: {video_path}")
    print("[DAÏO] Press 'q' to quit, 'r' to recalibrate standing baseline.")

    fall_detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DAÏO] End of video stream reached.")
            break

        annotated_frame, state, telemetry = detector.process_frame(frame, render_hud=True)

        if state == FallState.FALL_CONFIRMED:
            fall_detected_count += 1

        cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(int(1000.0 / fps)) & 0xFF

        if key == ord("q"):
            print("[DAÏO] Execution stopped by user.")
            break
        elif key == ord("r"):
            detector.reset()
            print("[DAÏO] Recalibrating standing baseline...")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DAÏO] Video processing completed. Confirmed fall events logged: {fall_detected_count}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        v_path = sys.argv[1]
    else:
        # Default fallback to generated demo video if none provided
        v_path = os.path.join("DataVideos", "fall_forward.mp4")
        if not os.path.exists(v_path):
            print("Usage: python TestFallDetection.py <path_to_video_file>")
            sys.exit(1)

    process_video_stream(v_path)
