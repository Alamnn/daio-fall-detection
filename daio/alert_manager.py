"""
Alert Manager and Snapshot Logger for DAÏO.
Maintains the 'Sauvegrade' event repository, records timestamped frame captures,
and logs event telemetry.
"""

from typing import Dict, Any, Optional
import os
import time
import json
import csv
import cv2
import numpy as np


class AlertManager:
    """
    Manages fall notification events, saves snapshots to 'Sauvegrade/',
    records audit telemetry, and handles optional audio or network alerts.
    """

    def __init__(
        self,
        save_dir: str = "Sauvegrade",
        enable_audio: bool = True,
    ):
        self.save_dir = save_dir
        self.enable_audio = enable_audio
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_json_path = os.path.join(self.save_dir, "fall_events_log.json")
        self.log_csv_path = os.path.join(self.save_dir, "fall_events_log.csv")

        # Initialize CSV header if missing
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "EventID",
                        "Timestamp",
                        "FrameIndex",
                        "ConfidenceScore",
                        "TorsoAngleDeg",
                        "DownwardVelocityMps",
                        "HeightDropRatio",
                        "SnapshotFile",
                    ]
                )

    def trigger_fall_alert(
        self,
        frame: np.ndarray,
        frame_idx: int,
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Executes full fall response:
        1. Generates timestamped snapshot file in Sauvegrade/
        2. Logs event details to JSON and CSV
        3. Emits audio beep
        """
        event_time_str = time.strftime("%Y%m%d_%H%M%S")
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        event_id = f"FALL_{event_time_str}_F{frame_idx}"
        snapshot_filename = f"{event_id}.jpg"
        snapshot_path = os.path.join(self.save_dir, snapshot_filename)

        # Save frame
        cv2.imwrite(snapshot_path, frame)

        event_data = {
            "event_id": event_id,
            "timestamp": timestamp_iso,
            "frame_index": frame_idx,
            "confidence_score": float(telemetry.get("confidence_score", 1.0)),
            "torso_angle_deg": float(telemetry.get("torso_angle_deg", 0.0)),
            "vy_mps": float(telemetry.get("vy_mps", 0.0)),
            "height_ratio": float(telemetry.get("height_ratio", 0.0)),
            "snapshot_file": snapshot_path,
        }

        # Append to CSV
        try:
            with open(self.log_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        event_data["event_id"],
                        event_data["timestamp"],
                        event_data["frame_index"],
                        f"{event_data['confidence_score']:.3f}",
                        f"{event_data['torso_angle_deg']:.2f}",
                        f"{event_data['vy_mps']:.2f}",
                        f"{event_data['height_ratio']:.2f}",
                        snapshot_filename,
                    ]
                )
        except Exception as e:
            print(f"[AlertManager] Error updating CSV: {e}")

        # Append to JSON
        try:
            existing_events = []
            if os.path.exists(self.log_json_path):
                with open(self.log_json_path, "r", encoding="utf-8") as f:
                    existing_events = json.load(f)
            existing_events.append(event_data)
            with open(self.log_json_path, "w", encoding="utf-8") as f:
                json.dump(existing_events, f, indent=2)
        except Exception as e:
            print(f"[AlertManager] Error updating JSON: {e}")

        # Optional audio alert
        if self.enable_audio:
            self._play_alert_sound()

        return event_data

    def _play_alert_sound(self) -> None:
        """Emits audio alert without crashing if audio device is unavailable."""
        try:
            # Print bell character for terminal acoustic cue
            print("\a", end="", flush=True)
        except Exception:
            pass

    def get_all_events(self) -> list:
        """Retrieves list of all logged events."""
        if os.path.exists(self.log_json_path):
            try:
                with open(self.log_json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
