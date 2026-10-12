"""
Triple-Check Logic Engine
Based on DAÏO Section 3.3 by Al-Amin Abubakar.
Validates falls across spatial, rotational, and kinematic dimensions in parallel:
Fall Flag(k) = Check_Position(k) AND Check_Angle(k) AND Check_Velocity(k)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class TripleCheckEngine:
    """
    Implements the Triple-Check Logic Engine with kinematic velocity latching:
    - Check 1: Dynamic Position Check (Hips drop below 50% baseline standing height)
    - Check 2: Torso Angle Analysis (|theta| < 45 deg relative to horizontal ground)
    - Check 3: Kinematic Velocity Latch (Downwards velocity > threshold for >= 3 frames, latched)
    - Weighted Confidence Scoring: C = 0.25*S_pos + 0.25*S_ang + 0.50*S_vel
    """

    def __init__(
        self,
        position_drop_threshold: float = 0.50,
        torso_angle_threshold: float = 45.0,
        velocity_threshold_mps: float = 1.5,
        velocity_sustain_frames: int = 3,
        velocity_latch_duration_frames: int = 75,
        w_pos: float = 0.25,
        w_ang: float = 0.25,
        w_vel: float = 0.50,
    ):
        self.position_drop_threshold = position_drop_threshold
        self.torso_angle_threshold = torso_angle_threshold
        self.velocity_threshold_mps = velocity_threshold_mps
        self.velocity_sustain_frames = velocity_sustain_frames
        self.velocity_latch_duration_frames = velocity_latch_duration_frames

        self.w_pos = w_pos
        self.w_ang = w_ang
        self.w_vel = w_vel

        # Calibration reference
        self.is_calibrated: bool = False
        self.y_base: float = 0.0
        self.ref_torso_length: float = 100.0
        self.calibration_accumulator_y: list = []
        self.calibration_accumulator_len: list = []

        # Kinematic velocity latch internal state
        self.high_velocity_counter: int = 0
        self.velocity_latch_timer: int = 0
        self.peak_latched_velocity: float = 0.0

    def calibrate_step(self, hip_y: float, torso_length: float) -> bool:
        """Accumulates standing baseline over frames."""
        self.calibration_accumulator_y.append(hip_y)
        self.calibration_accumulator_len.append(torso_length)
        return len(self.calibration_accumulator_y) >= 100

    def finalize_calibration(self) -> None:
        """Finalizes baseline metrics after calibration window."""
        if self.calibration_accumulator_y:
            self.y_base = float(np.median(self.calibration_accumulator_y))
            self.ref_torso_length = float(np.median(self.calibration_accumulator_len))
            self.is_calibrated = True

    def reset_calibration(self) -> None:
        """Resets calibration state."""
        self.is_calibrated = False
        self.y_base = 0.0
        self.ref_torso_length = 100.0
        self.calibration_accumulator_y.clear()
        self.calibration_accumulator_len.clear()
        self.reset_latch()

    def reset_latch(self) -> None:
        """Resets the velocity latch mechanism."""
        self.high_velocity_counter = 0
        self.velocity_latch_timer = 0
        self.peak_latched_velocity = 0.0

    def evaluate(
        self,
        hip_y: float,
        torso_angle_deg: float,
        vy_hip_pixels_per_sec: float,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Evaluates the frame across the three independent checks.

        Returns:
            Dictionary with booleans for each check, fall_flag, and confidence score.
        """
        # Convert pixel velocity to approximate metric scale (m/s)
        # Using calibrated torso length: average adult torso ~ 0.50 meters
        pixels_per_meter = max(self.ref_torso_length / 0.50, 50.0)
        vy_mps = vy_hip_pixels_per_sec / pixels_per_meter

        # --- Check 1: Dynamic Position Check ---
        # In screen coordinates, moving downward means hip_y > y_base.
        # Height drop relative to torso length:
        vertical_drop_pixels = max(0.0, hip_y - self.y_base)
        height_ratio = vertical_drop_pixels / max(self.ref_torso_length, 1.0)
        check_position = height_ratio > self.position_drop_threshold

        # --- Check 2: Torso Angle Analysis ---
        # Torso angle relative to horizontal ground: < 45 degrees indicates horizontal lying
        abs_angle = abs(torso_angle_deg)
        check_angle = abs_angle < self.torso_angle_threshold

        # --- Check 3: Kinematic Velocity Latch ---
        # Downward velocity check (positive vy indicates downward motion in screen coordinates)
        if vy_mps > self.velocity_threshold_mps:
            self.high_velocity_counter += 1
        else:
            self.high_velocity_counter = max(0, self.high_velocity_counter - 1)

        # Trigger latch if sustained for >= tau frames
        if self.high_velocity_counter >= self.velocity_sustain_frames:
            self.velocity_latch_timer = self.velocity_latch_duration_frames
            self.peak_latched_velocity = max(self.peak_latched_velocity, vy_mps)

        # Check if latch is currently active
        check_velocity = self.velocity_latch_timer > 0
        if self.velocity_latch_timer > 0:
            self.velocity_latch_timer -= 1
            if self.velocity_latch_timer == 0:
                self.peak_latched_velocity = 0.0

        # --- Parallel Triple-Check Logic Gate ---
        fall_flag = check_position and check_angle and check_velocity

        # --- Weighted Confidence Scoring ---
        # S_pos: deviation beyond position threshold [0, 1]
        s_pos = float(np.clip((height_ratio - self.position_drop_threshold) / 0.50, 0.0, 1.0))

        # S_ang: deviation below angle threshold [0, 1] (closer to 0 deg = higher score)
        s_ang = float(np.clip((self.torso_angle_threshold - abs_angle) / self.torso_angle_threshold, 0.0, 1.0))

        # S_vel: presence and intensity of velocity impact [0, 1]
        if check_velocity:
            s_vel = float(np.clip(self.peak_latched_velocity / (self.velocity_threshold_mps * 1.8), 0.5, 1.0))
        else:
            s_vel = 0.0

        confidence_score = (self.w_pos * s_pos) + (self.w_ang * s_ang) + (self.w_vel * s_vel)
        if not (check_position and check_angle):
            # If body is upright, suppress overall fall confidence
            confidence_score *= 0.3

        confidence_score = float(np.clip(confidence_score, 0.0, 1.0))

        return {
            "fall_flag": bool(fall_flag),
            "check_position": bool(check_position),
            "check_angle": bool(check_angle),
            "check_velocity": bool(check_velocity),
            "height_ratio": float(height_ratio),
            "torso_angle_deg": float(torso_angle_deg),
            "vy_mps": float(vy_mps),
            "vy_pixels_sec": float(vy_hip_pixels_per_sec),
            "velocity_latched": bool(self.velocity_latch_timer > 0),
            "latch_remaining_frames": int(self.velocity_latch_timer),
            "confidence_score": confidence_score,
            "s_pos": s_pos,
            "s_ang": s_ang,
            "s_vel": s_vel,
        }
