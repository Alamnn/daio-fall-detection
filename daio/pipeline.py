"""
DAÏO End-to-End Fall Detection Pipeline Orchestrator.
Connects MediaPipe Pose -> Dual EKFs -> Triple-Check Logic -> FSM -> HUD Visualizer -> Alert Manager.
"""

from typing import Optional, Dict, Any, Tuple, Callable
import time
import cv2
import numpy as np

from daio.ekf import ExtendedKalmanFilter, TorsoKinematics
from daio.pose_detector import PoseDetector
from daio.triple_check import TripleCheckEngine
from daio.state_machine import FallDetectionFSM, FallState
from daio.visualizer import DAIOVisualizer
from daio.alert_manager import AlertManager


class DAIOFallDetector:
    """
    Main pipeline controller integrating all DAÏO system components.
    """

    def __init__(
        self,
        fps: float = 30.0,
        calibration_frames: int = 100,
        save_dir: str = "Sauvegrade",
        enable_audio: bool = True,
        on_fall_confirmed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_state_change: Optional[Callable[[FallState, FallState], None]] = None,
    ):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)
        self.frame_idx: int = 0
        self.prev_time: float = time.time()
        self.current_fps: float = fps

        # Submodules
        self.pose_detector = PoseDetector(
            mode=False,
            smooth=True,
            detection_con=0.5,
            track_con=0.5,
        )

        # Dual EKF tracking instances for Shoulders midpoint and Hips midpoint
        self.ekf_shoulder = ExtendedKalmanFilter(dt=self.dt)
        self.ekf_hip = ExtendedKalmanFilter(dt=self.dt)
        self.ekf_initialized = False

        # Triple-Check Engine
        self.triple_check = TripleCheckEngine(
            position_drop_threshold=0.50,
            torso_angle_threshold=45.0,
            velocity_threshold_mps=1.5,
            velocity_sustain_frames=3,
        )

        # Alert Manager
        self.alert_manager = AlertManager(save_dir=save_dir, enable_audio=enable_audio)

        # External callbacks
        self.user_on_fall_confirmed = on_fall_confirmed
        self.user_on_state_change = on_state_change

        # Finite State Machine
        self.fsm = FallDetectionFSM(
            calibration_frames=calibration_frames,
            suspected_confirmation_frames=30,
            recovery_normalization_frames=150,
            on_fall_confirmed=self._handle_fall_confirmed,
            on_state_change=self.user_on_state_change,
        )

        # HUD Visualizer
        self.visualizer = DAIOVisualizer()

        # Telemetry cache
        self.latest_telemetry: Dict[str, Any] = {}
        self.latest_frame_annotated: Optional[np.ndarray] = None
        self.latest_fall_event: Optional[Dict[str, Any]] = None

    def _handle_fall_confirmed(self, telemetry: Dict[str, Any]) -> None:
        """Internal callback when FSM declares FALL_CONFIRMED."""
        if self.latest_frame_annotated is not None:
            # Re-render HUD to show FALL_CONFIRMED badge in top right
            s_smoothed = self.ekf_shoulder.position if self.ekf_initialized else (0, 0)
            h_smoothed = self.ekf_hip.position if self.ekf_initialized else (0, 0)
            snapshot = self.visualizer.render_hud(
                frame=self.latest_frame_annotated.copy(),
                state=FallState.FALL_CONFIRMED,
                telemetry=telemetry,
                shoulder_smoothed=s_smoothed,
                hip_smoothed=h_smoothed,
                fps=self.current_fps,
            )

            h, w = snapshot.shape[:2]
            alert_box_h = 70
            alert_box_y = h // 2 - alert_box_h // 2
            cv2.rectangle(snapshot, (0, alert_box_y), (w, alert_box_y + alert_box_h), (0, 0, 200), cv2.FILLED)
            cv2.line(snapshot, (0, alert_box_y), (w, alert_box_y), (255, 255, 255), 2)
            cv2.line(snapshot, (0, alert_box_y + alert_box_h), (w, alert_box_y + alert_box_h), (255, 255, 255), 2)
            alert_text = "CRITICAL ALERT: FALL CONFIRMED!"
            (atw, ath), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
            cv2.putText(snapshot, alert_text, (w // 2 - atw // 2, alert_box_y + 46), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            event_data = self.alert_manager.trigger_fall_alert(
                frame=snapshot,
                frame_idx=self.frame_idx,
                telemetry=telemetry,
            )
            self.latest_fall_event = event_data
            if self.user_on_fall_confirmed:
                self.user_on_fall_confirmed(event_data)

    def process_frame(
        self,
        frame: np.ndarray,
        render_hud: bool = True,
    ) -> Tuple[np.ndarray, FallState, Dict[str, Any]]:
        """
        Processes a single video frame through the full DAÏO pipeline.

        Returns:
            annotated_frame: Frame with HUD & skeleton overlay
            current_state: Current FallState
            telemetry: Detailed metrics dictionary
        """
        self.frame_idx += 1
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff > 0:
            self.current_fps = (self.current_fps * 0.9) + ((1.0 / time_diff) * 0.1)
        self.prev_time = curr_time

        h, w = frame.shape[:2]

        # 1. Pose Detection
        raw_frame = frame.copy()
        _ = self.pose_detector.find_pose(raw_frame, draw=False)
        self.pose_detector.find_position(raw_frame, draw=False)
        torso_data = self.pose_detector.get_torso_landmarks(frame.shape)

        is_pose_detected = torso_data is not None

        if is_pose_detected and torso_data:
            s_raw = torso_data["shoulder_mid"]
            s_vis = torso_data["shoulder_vis"]
            h_raw = torso_data["hip_mid"]
            h_vis = torso_data["hip_vis"]
            torso_len = torso_data["torso_length"]

            # Initialize EKFs on first valid detection
            if not self.ekf_initialized:
                self.ekf_shoulder = ExtendedKalmanFilter(dt=self.dt, initial_point=s_raw)
                self.ekf_hip = ExtendedKalmanFilter(dt=self.dt, initial_point=h_raw)
                self.ekf_initialized = True

            # 2. EKF Tracking: Predict & Update with Adaptive Noise
            self.ekf_shoulder.predict()
            self.ekf_hip.predict()

            self.ekf_shoulder.update(s_raw, visibility=s_vis)
            self.ekf_hip.update(h_raw, visibility=h_vis)

            s_smoothed = self.ekf_shoulder.position
            h_smoothed = self.ekf_hip.position
            _, vy_hip_pixels = self.ekf_hip.velocity

            # 3. Torso Kinematics & Analytical Jacobian
            theta_deg, J_theta, sigma_theta = TorsoKinematics.calculate_angle_and_jacobian(
                shoulder_pos=s_smoothed,
                hip_pos=h_smoothed,
                shoulder_cov=self.ekf_shoulder.P,
                hip_cov=self.ekf_hip.P,
            )

            # 4. Calibration handling during CALIBRATING state
            if self.fsm.current_state == FallState.CALIBRATING:
                is_calib_done = self.triple_check.calibrate_step(
                    hip_y=h_smoothed[1],
                    torso_length=torso_len,
                )
                if is_calib_done:
                    self.triple_check.finalize_calibration()

            # 5. Triple-Check Evaluation
            triple_result = self.triple_check.evaluate(
                hip_y=h_smoothed[1],
                torso_angle_deg=theta_deg,
                vy_hip_pixels_per_sec=vy_hip_pixels,
                fps=self.fps,
            )
            triple_result["sigma_theta"] = float(sigma_theta)
            triple_result["shoulder_vis"] = float(s_vis)
            triple_result["hip_vis"] = float(h_vis)

        else:
            # Joint Occlusion or missing skeleton:
            # If EKFs are initialized, execute predictive coasting
            if self.ekf_initialized:
                self.ekf_shoulder.predict()
                self.ekf_hip.predict()
                s_smoothed = self.ekf_shoulder.position
                h_smoothed = self.ekf_hip.position
                _, vy_hip_pixels = self.ekf_hip.velocity
                theta_deg, _, sigma_theta = TorsoKinematics.calculate_angle_and_jacobian(
                    shoulder_pos=s_smoothed,
                    hip_pos=h_smoothed,
                )
            else:
                s_smoothed = (w / 2.0, h / 3.0)
                h_smoothed = (w / 2.0, h / 2.0)
                vy_hip_pixels = 0.0
                theta_deg = 90.0
                sigma_theta = 0.0

            triple_result = {
                "fall_flag": False,
                "check_position": False,
                "check_angle": False,
                "check_velocity": False,
                "height_ratio": 0.0,
                "torso_angle_deg": theta_deg,
                "vy_mps": 0.0,
                "vy_pixels_sec": 0.0,
                "velocity_latched": False,
                "latch_remaining_frames": 0,
                "confidence_score": 0.0,
                "sigma_theta": float(sigma_theta),
                "shoulder_vis": 0.0,
                "hip_vis": 0.0,
            }

        # 6. Render HUD overlay
        if render_hud:
            annotated_frame = self.visualizer.render_hud(
                frame=raw_frame,
                state=self.fsm.current_state,
                telemetry=triple_result,
                shoulder_smoothed=s_smoothed,
                hip_smoothed=h_smoothed,
                fps=self.current_fps,
            )
        else:
            annotated_frame = raw_frame

        self.latest_frame_annotated = annotated_frame

        # 7. FSM State Transition
        current_state = self.fsm.process_frame(
            triple_check_result=triple_result,
            is_pose_detected=is_pose_detected,
        )

        # Re-render with confirmed banner if state transitioned to FALL_CONFIRMED
        if current_state == FallState.FALL_CONFIRMED and render_hud:
            annotated_frame = self.visualizer.render_hud(
                frame=raw_frame,
                state=current_state,
                telemetry=triple_result,
                shoulder_smoothed=s_smoothed,
                hip_smoothed=h_smoothed,
                fps=self.current_fps,
            )
            self.latest_frame_annotated = annotated_frame

        self.latest_telemetry = triple_result
        return annotated_frame, current_state, triple_result

    def reset(self) -> None:
        """Resets the entire pipeline."""
        self.frame_idx = 0
        self.ekf_initialized = False
        self.triple_check.reset_calibration()
        self.fsm.reset()
        self.latest_fall_event = None
