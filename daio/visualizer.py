"""
HUD Visualizer and Graphic Designer for DAÏO Fall Detection System.
Preserves classes Legend and Drawing while incorporating advanced EKF vectors,
temporal rolling charts, and state indicators.
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np
from daio.state_machine import FallState


class Legend:
    """
    Renders reference axes and threshold lines on video frames.
    Directly compatible with Al-Amin Abubakar's Legend class.
    """

    def __init__(self, img: np.ndarray):
        self.img = img
        self.height, self.width = img.shape[:2]

    def draw_vline(
        self,
        x: int,
        label: str,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 1,
    ) -> None:
        cv2.line(self.img, (x, 0), (x, self.height), color, thickness)
        cv2.putText(
            self.img,
            label,
            (x + 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_hline(
        self,
        y: int,
        label: str,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 1,
    ) -> None:
        cv2.line(self.img, (0, y), (self.width, y), color, thickness)
        cv2.putText(
            self.img,
            label,
            (10, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


class Drawing:
    """
    Renders skeletal overlays, velocity vectors, and coordinate indicators.
    Directly compatible with Al-Amin Abubakar's Drawing class.
    """

    @staticmethod
    def draw_axes(img: np.ndarray) -> None:
        """Draws top-left coordinate reference axes."""
        h, w = img.shape[:2]
        # Y axis (downwards in screen coordinates)
        cv2.arrowedLine(img, (20, 20), (20, 80), (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(img, "+Y (Down)", (25, 75), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)

        # X axis (rightwards)
        cv2.arrowedLine(img, (20, 20), (80, 20), (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(img, "+X", (85, 25), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)

    @staticmethod
    def draw_kf_vector(
        img: np.ndarray,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        label: str = "v",
        color: Tuple[int, int, int] = (0, 255, 255),
        scale: float = 0.5,
    ) -> None:
        """Visualizes Kalman filter estimated velocity vector."""
        px, py = int(pos[0]), int(pos[1])
        vx, vy = vel
        end_x = int(px + (vx * scale))
        end_y = int(py + (vy * scale))

        speed = np.sqrt(vx**2 + vy**2)
        if speed > 5.0:
            cv2.arrowedLine(img, (px, py), (end_x, end_y), color, 2, tipLength=0.25)
            cv2.putText(
                img,
                f"{label}:{int(speed)}px/s",
                (end_x + 5, end_y),
                cv2.FONT_HERSHEY_PLAIN,
                0.9,
                color,
                1,
            )


class DAIOVisualizer:
    """
    Main visualization and HUD compositor for DAÏO.
    """

    def __init__(self, history_len: int = 120):
        self.history_len = history_len
        self.vy_history: List[float] = [0.0] * history_len
        self.angle_history: List[float] = [90.0] * history_len
        self.conf_history: List[float] = [0.0] * history_len

    def update_history(self, vy_mps: float, angle_deg: float, conf: float) -> None:
        self.vy_history.pop(0)
        self.vy_history.append(vy_mps)

        self.angle_history.pop(0)
        self.angle_history.append(angle_deg)

        self.conf_history.pop(0)
        self.conf_history.append(conf)

    def render_hud(
        self,
        frame: np.ndarray,
        state: FallState,
        telemetry: Dict[str, Any],
        shoulder_smoothed: Tuple[float, float],
        hip_smoothed: Tuple[float, float],
        fps: float,
    ) -> np.ndarray:
        """
        Overlays complete DAÏO HUD onto frame.
        """
        h, w = frame.shape[:2]
        hud = frame.copy()

        # Update rolling trend curves
        vy_mps = telemetry.get("vy_mps", 0.0)
        angle = telemetry.get("torso_angle_deg", 90.0)
        conf = telemetry.get("confidence_score", 0.0)
        self.update_history(vy_mps, angle, conf)

        # 1. Coordinate axes
        Drawing.draw_axes(hud)

        # 2. Torso Line & Joints (EKF Smoothed)
        s_pt = (int(shoulder_smoothed[0]), int(shoulder_smoothed[1]))
        h_pt = (int(hip_smoothed[0]), int(hip_smoothed[1]))
        
        # Color torso line based on angle
        abs_ang = abs(angle)
        torso_color = (0, 255, 0) if abs_ang >= 45.0 else (0, 0, 255)
        cv2.line(hud, s_pt, h_pt, torso_color, 4)
        cv2.circle(hud, s_pt, 7, (255, 200, 0), cv2.FILLED)
        cv2.circle(hud, h_pt, 7, (0, 200, 255), cv2.FILLED)

        # Draw velocity vectors for shoulders & hips
        vy_pixels = telemetry.get("vy_pixels_sec", 0.0)
        Drawing.draw_kf_vector(hud, hip_smoothed, (0.0, vy_pixels), label="v_hip", color=(0, 255, 255))

        # 3. Top Banner & State Badge
        state_colors = {
            FallState.CALIBRATING: (200, 150, 0),    # Blue/Gold
            FallState.MONITORING: (50, 200, 50),     # Vibrant Green
            FallState.FALL_SUSPECTED: (0, 165, 255), # Orange
            FallState.FALL_CONFIRMED: (0, 0, 255),   # Red Alert
            FallState.RECOVERY: (255, 180, 0),       # Cyan
        }
        b_color = state_colors.get(state, (100, 100, 100))

        # Header bar background
        cv2.rectangle(hud, (0, 0), (w, 52), (20, 20, 25), cv2.FILLED)
        cv2.line(hud, (0, 52), (w, 52), b_color, 2)

        # App Title
        cv2.putText(
            hud,
            "DAIO | Extended Kalman Filter Fall Detection",
            (105, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # State pill badge
        badge_text = f"STATE: {state.value}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        badge_x = w - tw - 25
        cv2.rectangle(hud, (badge_x - 10, 10), (badge_x + tw + 10, 42), b_color, cv2.FILLED)
        cv2.putText(hud, badge_text, (badge_x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Telemetry Glassmorphism Card (Top-Left)
        card_w, card_h = 240, 185
        card_x, card_y = 15, 65
        overlay = hud.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (15, 15, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
        cv2.rectangle(hud, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 60, 70), 1)

        cv2.putText(hud, "KINEMATIC TELEMETRY", (card_x + 10, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1, cv2.LINE_AA)

        # Metrics rows
        y_off = card_y + 45
        line_spacing = 24

        def draw_metric_row(label: str, val_str: str, passed: Optional[bool] = None):
            nonlocal y_off
            cv2.putText(hud, label, (card_x + 10, y_off), cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)
            val_col = (255, 255, 255)
            if passed is True:
                val_col = (0, 255, 128)
            elif passed is False:
                val_col = (100, 100, 240)
            cv2.putText(hud, val_str, (card_x + 145, y_off), cv2.FONT_HERSHEY_PLAIN, 1.1, val_col, 2)
            y_off += line_spacing

        draw_metric_row("Torso Angle:", f"{angle:.1f}deg", telemetry.get("check_angle"))
        draw_metric_row("Height Drop:", f"{telemetry.get('height_ratio', 0.0):.2f}x", telemetry.get("check_position"))
        draw_metric_row("Down Velocity:", f"{vy_mps:.2f} m/s", telemetry.get("check_velocity"))
        
        latch_status = "LATCHED" if telemetry.get("velocity_latched") else "IDLE"
        draw_metric_row("Velocity Latch:", latch_status, telemetry.get("velocity_latched"))
        draw_metric_row("Pipeline FPS:", f"{fps:.1f}", None)

        # 5. Fall Confidence Bar Meter (Bottom Right)
        meter_w, meter_h = 220, 48
        meter_x = w - meter_w - 20
        meter_y = h - meter_h - 20
        cv2.rectangle(hud, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (20, 20, 25), cv2.FILLED)
        cv2.rectangle(hud, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (60, 60, 70), 1)

        conf_pct = int(conf * 100)
        bar_fill = int((meter_w - 20) * conf)
        bar_col = (0, 200, 0) if conf < 0.4 else ((0, 165, 255) if conf < 0.7 else (0, 0, 255))
        
        cv2.putText(hud, f"FALL CONFIDENCE: {conf_pct}%", (meter_x + 10, meter_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(hud, (meter_x + 10, meter_y + 25), (meter_x + 10 + bar_fill, meter_y + 38), bar_col, cv2.FILLED)

        # 6. Temporal Graph Overlay (Bottom-Left)
        # Visualizes the temporal evolution of downward velocity (v_y) and threshold line
        gw, gh = 240, 80
        gx, gy = 15, h - gh - 20
        cv2.rectangle(hud, (gx, gy), (gx + gw, gy + gh), (15, 15, 20), cv2.FILLED)
        cv2.rectangle(hud, (gx, gy), (gx + gw, gy + gh), (60, 60, 70), 1)
        cv2.putText(hud, "EKF Velocity Trend (m/s)", (gx + 8, gy + 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (180, 180, 180), 1)

        # Threshold horizontal line (1.5 m/s)
        thresh_mps = 1.5
        max_scale_mps = 3.0
        thresh_py = int(gy + gh - (thresh_mps / max_scale_mps * (gh - 25)))
        cv2.line(hud, (gx, thresh_py), (gx + gw, thresh_py), (0, 0, 255), 1)
        cv2.putText(hud, "1.5m/s seuil", (gx + gw - 75, thresh_py - 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)

        # Trend points
        step = max(1, gw // len(self.vy_history))
        pts = []
        for i, v_val in enumerate(self.vy_history):
            px = gx + (i * step)
            norm_v = np.clip(v_val / max_scale_mps, 0.0, 1.0)
            py = int((gy + gh - 5) - (norm_v * (gh - 25)))
            pts.append((px, py))
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(hud, pts[i], pts[i + 1], (0, 255, 255), 1)

        # 7. Big Screen Alert when FALL_CONFIRMED
        if state == FallState.FALL_CONFIRMED:
            alert_box_h = 70
            alert_box_y = h // 2 - alert_box_h // 2
            cv2.rectangle(hud, (0, alert_box_y), (w, alert_box_y + alert_box_h), (0, 0, 200), cv2.FILLED)
            cv2.line(hud, (0, alert_box_y), (w, alert_box_y), (255, 255, 255), 2)
            cv2.line(hud, (0, alert_box_y + alert_box_h), (w, alert_box_y + alert_box_h), (255, 255, 255), 2)
            
            alert_text = "CRITICAL ALERT: FALL CONFIRMED!"
            (atw, ath), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
            cv2.putText(hud, alert_text, (w // 2 - atw // 2, alert_box_y + 46), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        return hud
