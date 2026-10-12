"""
DAÏO Desktop Graphical User Interface (GUI).
Built with ttkbootstrap and Tkinter.
Preserves the user's original FallDetectionApp architecture while adding live HUD display,
telemetry meters, event logs, and snapshot inspection.
"""

from typing import Optional
import os
import sys
import threading
import time
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

from daio.pipeline import DAIOFallDetector
from daio.state_machine import FallState


class FallDetectionApp:
    """
    Main Tkinter / ttkbootstrap application for DAÏO Fall Detection System.
    """

    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title("DAÏO - AI-Assisted Fall Detection System (EKF)")
        self.root.geometry("1280x850")
        self.root.minsize(1024, 720)

        # Pipeline attributes
        self.video_source: Optional[str] = None
        self.is_running: bool = False
        self.is_paused: bool = False
        self.worker_thread: Optional[threading.Thread] = None
        self.detector: Optional[DAIOFallDetector] = None

        # Data directories
        self.data_dir = os.path.abspath("DataVideos")
        self.save_dir = os.path.abspath("Sauvegrade")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self._build_ui()
        self._load_sample_videos()

    def _build_ui(self) -> None:
        # Header frame
        header_frame = tb.Frame(self.root, bootstyle="secondary", padding=12)
        header_frame.pack(fill=X, side=TOP)

        title_lbl = tb.Label(
            header_frame,
            text="DAÏO: AI-Assisted Fall Detection System",
            font=("Helvetica", 18, "bold"),
            bootstyle="inverse-secondary",
        )
        title_lbl.pack(anchor=W)

        subtitle_lbl = tb.Label(
            header_frame,
            text="Extended Kalman Filter (EKF) Kinematic Tracking | High-Specificity Triple-Check Engine",
            font=("Helvetica", 11),
            bootstyle="inverse-secondary",
        )
        subtitle_lbl.pack(anchor=W, pady=(2, 0))

        # Main horizontal split
        main_content = tb.Frame(self.root, padding=10)
        main_content.pack(fill=BOTH, expand=True)

        # Left Column: Video display + Live HUD (approx 65% width)
        left_col = tb.Frame(main_content)
        left_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        video_card = tb.Labelframe(left_col, text="Live Kinematic Video Feed", padding=5, bootstyle="info")
        video_card.pack(fill=BOTH, expand=True)

        self.video_canvas = tb.Label(video_card, text="No video loaded\nSelect a sample or upload a video file to begin.", justify=CENTER, font=("Helvetica", 14))
        self.video_canvas.pack(fill=BOTH, expand=True)

        # Playback control bar
        controls_frame = tb.Frame(left_col, padding=(0, 8, 0, 0))
        controls_frame.pack(fill=X)

        self.btn_run = tb.Button(controls_frame, text="▶ Run Fall Detection", bootstyle="success", command=self.on_start_clicked)
        self.btn_run.pack(side=LEFT, padx=(0, 5))

        self.btn_pause = tb.Button(controls_frame, text="⏸ Pause", bootstyle="warning-outline", command=self.on_pause_clicked, state=DISABLED)
        self.btn_pause.pack(side=LEFT, padx=5)

        self.btn_stop = tb.Button(controls_frame, text="⏹ Stop", bootstyle="danger-outline", command=self.on_stop_clicked, state=DISABLED)
        self.btn_stop.pack(side=LEFT, padx=5)

        self.btn_reset_calib = tb.Button(controls_frame, text="⟳ Recalibrate", bootstyle="secondary-outline", command=self.on_reset_calib_clicked, state=DISABLED)
        self.btn_reset_calib.pack(side=LEFT, padx=5)

        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_lbl = tb.Label(controls_frame, textvariable=self.status_var, font=("Helvetica", 10, "italic"))
        self.status_lbl.pack(side=RIGHT, padx=5)

        # Right Column: Source Controls + Telemetry + Events Log (approx 35% width)
        right_col = tb.Frame(main_content, width=380)
        right_col.pack(side=RIGHT, fill=BOTH)
        right_col.pack_propagate(False)

        # 1. Source Selection Card
        src_card = tb.Labelframe(right_col, text="Source Selection", padding=10, bootstyle="primary")
        src_card.pack(fill=X, pady=(0, 8))

        tb.Label(src_card, text="Quick Benchmark Sample:").pack(anchor=W)
        self.sample_combo = tb.Combobox(src_card, state="readonly")
        self.sample_combo.pack(fill=X, pady=(2, 6))
        self.sample_combo.bind("<<ComboboxSelected>>", self.on_sample_selected)

        btn_browse = tb.Button(src_card, text="📂 Browse Video File...", bootstyle="info-outline", command=self.on_browse_clicked)
        btn_browse.pack(fill=X, pady=(2, 4))

        btn_webcam = tb.Button(src_card, text="📷 Use Live Webcam", bootstyle="secondary-outline", command=self.on_webcam_clicked)
        btn_webcam.pack(fill=X)

        # 2. Live Telemetry Card
        telem_card = tb.Labelframe(right_col, text="Real-Time EKF Telemetry", padding=10, bootstyle="success")
        telem_card.pack(fill=X, pady=(0, 8))

        # State badge
        state_row = tb.Frame(telem_card)
        state_row.pack(fill=X, pady=(0, 6))
        tb.Label(state_row, text="Current State:", font=("Helvetica", 10, "bold")).pack(side=LEFT)
        self.state_badge = tb.Label(state_row, text="READY", font=("Helvetica", 10, "bold"), bootstyle="inverse-secondary", padding=(8, 2))
        self.state_badge.pack(side=RIGHT)

        # Metrics rows
        self.metric_vars = {
            "angle": tk.StringVar(value="90.0°"),
            "drop": tk.StringVar(value="0.00x"),
            "vel": tk.StringVar(value="0.00 m/s"),
            "latch": tk.StringVar(value="IDLE"),
            "fps": tk.StringVar(value="0.0"),
        }

        def add_metric_line(parent, label_text, str_var):
            row = tb.Frame(parent)
            row.pack(fill=X, pady=2)
            tb.Label(row, text=label_text).pack(side=LEFT)
            tb.Label(row, textvariable=str_var, font=("Helvetica", 10, "bold")).pack(side=RIGHT)

        add_metric_line(telem_card, "Torso Tilt Angle (θ):", self.metric_vars["angle"])
        add_metric_line(telem_card, "Height Drop Ratio:", self.metric_vars["drop"])
        add_metric_line(telem_card, "Hip Downward Velocity:", self.metric_vars["vel"])
        add_metric_line(telem_card, "Kinematic Latch:", self.metric_vars["latch"])
        add_metric_line(telem_card, "Pipeline FPS:", self.metric_vars["fps"])

        # Fall Confidence Progress Bar
        tb.Label(telem_card, text="Fall Confidence Meter:").pack(anchor=W, pady=(6, 2))
        self.conf_bar = tb.Progressbar(telem_card, maximum=100, bootstyle="success-striped")
        self.conf_bar.pack(fill=X)
        self.conf_pct_lbl = tb.Label(telem_card, text="0%", font=("Helvetica", 9))
        self.conf_pct_lbl.pack(anchor=E)

        # 3. Confirmed Fall Events Log
        events_card = tb.Labelframe(right_col, text="Confirmed Fall Incidents", padding=8, bootstyle="danger")
        events_card.pack(fill=BOTH, expand=True)

        columns = ("id", "time", "conf", "angle", "file")
        self.tree = tb.Treeview(events_card, columns=columns, show="headings", height=5)
        self.tree.heading("id", text="ID")
        self.tree.heading("time", text="Timestamp")
        self.tree.heading("conf", text="Conf")
        self.tree.heading("angle", text="Angle")
        self.tree.heading("file", text="Snapshot")

        self.tree.column("id", width=60, anchor=CENTER)
        self.tree.column("time", width=120, anchor=CENTER)
        self.tree.column("conf", width=55, anchor=CENTER)
        self.tree.column("angle", width=55, anchor=CENTER)
        self.tree.column("file", width=70, anchor=W)

        tree_scroll = tb.Scrollbar(events_card, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        tree_scroll.pack(side=RIGHT, fill=Y)

        ev_btn_row = tb.Frame(right_col, padding=(0, 6, 0, 0))
        ev_btn_row.pack(fill=X)

        btn_open_snap = tb.Button(ev_btn_row, text="🖼 View Snapshot", bootstyle="info-outline", command=self.on_view_snapshot_clicked)
        btn_open_snap.pack(side=LEFT, fill=X, expand=True, padx=(0, 4))

        btn_open_dir = tb.Button(ev_btn_row, text="📁 Sauvegrade Folder", bootstyle="secondary-outline", command=self.on_open_sauvegrade_clicked)
        btn_open_dir.pack(side=RIGHT, fill=X, expand=True, padx=(4, 0))

        # Footer
        footer = tb.Frame(self.root, padding=4)
        footer.pack(fill=X, side=BOTTOM)
        tb.Label(footer, text="© 2024-2026 Al-Amin Abubakar. DAÏO Real-Time Vision Fall Detection with Extended Kalman Filter.", font=("Helvetica", 9), bootstyle="secondary").pack(side=LEFT)

    def _load_sample_videos(self) -> None:
        """Populates the benchmark videos combo box."""
        if os.path.exists(self.data_dir):
            files = [f for f in os.listdir(self.data_dir) if f.endswith((".mp4", ".avi", ".mov"))]
            self.sample_combo["values"] = files
            if files:
                self.sample_combo.set(files[0])
                self.video_source = os.path.join(self.data_dir, files[0])

    def on_sample_selected(self, event=None) -> None:
        selected = self.sample_combo.get()
        if selected:
            self.video_source = os.path.join(self.data_dir, selected)
            self.status_var.set(f"Loaded sample: {selected}")

    def on_browse_clicked(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if filepath:
            self.video_source = filepath
            self.status_var.set(f"Selected: {os.path.basename(filepath)}")

    def on_webcam_clicked(self) -> None:
        self.video_source = 0
        self.status_var.set("Selected: Live Webcam (Index 0)")

    def on_start_clicked(self) -> None:
        if self.video_source is None:
            messagebox.showwarning("No Video Source", "Please select a sample video or upload a video file first.")
            return

        self.is_running = True
        self.is_paused = False
        self.btn_run.config(state=DISABLED)
        self.btn_pause.config(state=NORMAL, text="⏸ Pause")
        self.btn_stop.config(state=NORMAL)
        self.btn_reset_calib.config(state=NORMAL)

        self.worker_thread = threading.Thread(target=self._process_video_loop, daemon=True)
        self.worker_thread.start()

    def on_pause_clicked(self) -> None:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ Resume", bootstyle="success-outline")
            self.status_var.set("Playback paused")
        else:
            self.btn_pause.config(text="⏸ Pause", bootstyle="warning-outline")
            self.status_var.set("Processing...")

    def on_stop_clicked(self) -> None:
        self.is_running = False
        self.is_paused = False
        self.btn_run.config(state=NORMAL)
        self.btn_pause.config(state=DISABLED)
        self.btn_stop.config(state=DISABLED)
        self.btn_reset_calib.config(state=DISABLED)
        self.status_var.set("Processing stopped")

    def on_reset_calib_clicked(self) -> None:
        if self.detector:
            self.detector.reset()
            self.status_var.set("Calibration baseline reset. Stand upright for 100 frames.")

    def _on_fall_confirmed_callback(self, event_data: dict) -> None:
        """Callback fired on background thread when fall is confirmed."""
        self.root.after(0, self._insert_event_to_tree, event_data)

    def _insert_event_to_tree(self, ev: dict) -> None:
        ev_id = ev.get("event_id", "FALL")
        ts = ev.get("timestamp", "").split("T")[-1]
        conf = f"{int(ev.get('confidence_score', 1.0) * 100)}%"
        ang = f"{ev.get('torso_angle_deg', 0.0):.1f}°"
        fn = os.path.basename(ev.get("snapshot_file", ""))
        self.tree.insert("", 0, values=(ev_id, ts, conf, ang, fn))

    def _process_video_loop(self) -> None:
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Video Error", f"Could not open source: {self.video_source}"))
            self.on_stop_clicked()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.detector = DAIOFallDetector(
            fps=fps,
            save_dir=self.save_dir,
            enable_audio=True,
            on_fall_confirmed=self._on_fall_confirmed_callback,
        )

        self.root.after(0, lambda: self.status_var.set("Processing video stream..."))

        delay = 1.0 / fps

        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            start_t = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, state, telem = self.detector.process_frame(frame, render_hud=True)

            # Update GUI elements on main thread
            self.root.after(0, self._update_gui_frame, annotated_frame, state, telem)

            elapsed = time.time() - start_t
            sleep_t = max(0.001, delay - elapsed)
            time.sleep(sleep_t)

        cap.release()
        self.root.after(0, self._on_playback_finished)

    def _update_gui_frame(self, frame: np.ndarray, state: FallState, telem: dict) -> None:
        # 1. Update Video Canvas
        disp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to fit display smoothly
        disp_h, disp_w = 480, 640
        disp_frame = cv2.resize(disp_frame, (disp_w, disp_h))
        img_pil = Image.fromarray(disp_frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.video_canvas.configure(image=img_tk)
        self.video_canvas.image = img_tk  # Keep reference

        # 2. Update State Badge
        state_styles = {
            FallState.CALIBRATING: ("CALIBRATING", "inverse-info"),
            FallState.MONITORING: ("MONITORING", "inverse-success"),
            FallState.FALL_SUSPECTED: ("FALL SUSPECTED", "inverse-warning"),
            FallState.FALL_CONFIRMED: ("FALL CONFIRMED!", "inverse-danger"),
            FallState.RECOVERY: ("RECOVERY", "inverse-primary"),
        }
        text, style = state_styles.get(state, (state.value, "inverse-secondary"))
        self.state_badge.configure(text=text, bootstyle=style)

        # 3. Update Telemetry Labels
        ang = telem.get("torso_angle_deg", 90.0)
        self.metric_vars["angle"].set(f"{ang:.1f}°")

        drop = telem.get("height_ratio", 0.0)
        self.metric_vars["drop"].set(f"{drop:.2f}x")

        vel = telem.get("vy_mps", 0.0)
        self.metric_vars["vel"].set(f"{vel:.2f} m/s")

        latched = telem.get("velocity_latched", False)
        self.metric_vars["latch"].set("ACTIVE" if latched else "IDLE")

        self.metric_vars["fps"].set(f"{self.detector.current_fps:.1f}")

        # 4. Update Confidence Meter
        conf = telem.get("confidence_score", 0.0)
        conf_pct = int(conf * 100)
        self.conf_bar["value"] = conf_pct
        self.conf_pct_lbl.configure(text=f"{conf_pct}%")

        if conf_pct > 70:
            self.conf_bar.configure(bootstyle="danger")
        elif conf_pct > 40:
            self.conf_bar.configure(bootstyle="warning")
        else:
            self.conf_bar.configure(bootstyle="success")

    def _on_playback_finished(self) -> None:
        self.is_running = False
        self.btn_run.config(state=NORMAL)
        self.btn_pause.config(state=DISABLED)
        self.btn_stop.config(state=DISABLED)
        self.status_var.set("Video processing finished.")

    def on_view_snapshot_clicked(self) -> None:
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showinfo("No Selection", "Please click on a fall event in the table to view its snapshot.")
            return

        values = self.tree.item(selected_item[0], "values")
        filename = values[4]
        snap_path = os.path.join(self.save_dir, filename)
        if os.path.exists(snap_path):
            if sys.platform == "darwin":
                subprocess.run(["open", snap_path])
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", snap_path])
            elif sys.platform.startswith("win"):
                os.startfile(snap_path)
        else:
            messagebox.showwarning("File Missing", f"Could not find snapshot: {snap_path}")

    def on_open_sauvegrade_clicked(self) -> None:
        if os.path.exists(self.save_dir):
            if sys.platform == "darwin":
                subprocess.run(["open", self.save_dir])
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", self.save_dir])
            elif sys.platform.startswith("win"):
                os.startfile(self.save_dir)


def main():
    app_window = tb.Window(themename="darkly")
    app = FallDetectionApp(app_window)
    app_window.mainloop()


if __name__ == "__main__":
    main()
