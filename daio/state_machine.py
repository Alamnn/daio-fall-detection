"""
Finite State Machine (FSM) for Fall Detection
Based on DAÏO Section 3.4 by Al-Amin Abubakar.
States: CALIBRATING -> MONITORING -> FALL_SUSPECTED -> FALL_CONFIRMED -> RECOVERY
"""

from enum import Enum
from typing import Dict, Any, Callable, Optional, List
import time


class FallState(str, Enum):
    CALIBRATING = "CALIBRATING"
    MONITORING = "MONITORING"
    FALL_SUSPECTED = "FALL_SUSPECTED"
    FALL_CONFIRMED = "FALL_CONFIRMED"
    RECOVERY = "RECOVERY"


class FallDetectionFSM:
    """
    Finite State Machine managing temporal transition dynamics:
    - Eliminates single-frame tracking glitches
    - Requires 30 consecutive frames of positive Triple-Check to declare FALL_CONFIRMED
    - Requires 150 consecutive frames of upright normalization to declare RECOVERY
    """

    def __init__(
        self,
        calibration_frames: int = 100,
        suspected_confirmation_frames: int = 30,
        recovery_normalization_frames: int = 150,
        upright_angle_threshold: float = 70.0,
        on_fall_confirmed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_state_change: Optional[Callable[[FallState, FallState], None]] = None,
    ):
        self.calibration_frames = calibration_frames
        self.suspected_confirmation_frames = suspected_confirmation_frames
        self.recovery_normalization_frames = recovery_normalization_frames
        self.upright_angle_threshold = upright_angle_threshold

        self.current_state = FallState.CALIBRATING
        self.state_enter_timestamp = time.time()
        self.on_fall_confirmed = on_fall_confirmed
        self.on_state_change = on_state_change

        # Counters
        self.frame_index: int = 0
        self.calibration_count: int = 0
        self.suspected_count: int = 0
        self.recovery_count: int = 0

        # Event log
        self.transition_history: List[Dict[str, Any]] = []

    def set_state(self, new_state: FallState, reason: str = "") -> None:
        """Transitions state and triggers notification callbacks."""
        old_state = self.current_state
        if old_state != new_state:
            self.current_state = new_state
            self.state_enter_timestamp = time.time()
            record = {
                "frame": self.frame_index,
                "timestamp": self.state_enter_timestamp,
                "from_state": old_state.value,
                "to_state": new_state.value,
                "reason": reason,
            }
            self.transition_history.append(record)
            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    def process_frame(
        self,
        triple_check_result: Dict[str, Any],
        is_pose_detected: bool,
    ) -> FallState:
        """
        Executes FSM transition logic for each incoming frame.
        """
        self.frame_index += 1
        fall_flag = triple_check_result.get("fall_flag", False)
        torso_angle = abs(triple_check_result.get("torso_angle_deg", 90.0))
        check_pos = triple_check_result.get("check_position", False)

        if self.current_state == FallState.CALIBRATING:
            if is_pose_detected:
                self.calibration_count += 1
                if self.calibration_count >= self.calibration_frames:
                    self.set_state(FallState.MONITORING, "Calibration window completed")

        elif self.current_state == FallState.MONITORING:
            if fall_flag:
                self.suspected_count = 1
                self.set_state(FallState.FALL_SUSPECTED, "Triple-Check passed initial frame")

        elif self.current_state == FallState.FALL_SUSPECTED:
            if fall_flag:
                self.suspected_count += 1
                if self.suspected_count >= self.suspected_confirmation_frames:
                    self.set_state(
                        FallState.FALL_CONFIRMED,
                        f"Fall sustained for {self.suspected_confirmation_frames} frames",
                    )
                    if self.on_fall_confirmed:
                        self.on_fall_confirmed(triple_check_result)
            else:
                # Glitch rejection / reset
                self.suspected_count = 0
                self.set_state(FallState.MONITORING, "Triple-Check condition aborted prematurely")

        elif self.current_state == FallState.FALL_CONFIRMED:
            # Check if user began recovery: torso upright (> 70 deg) and hips returned to baseline
            is_upright = torso_angle > self.upright_angle_threshold and not check_pos
            if is_upright:
                self.recovery_count += 1
                if self.recovery_count >= 15:  # Initial upright posture detected
                    self.set_state(FallState.RECOVERY, "User beginning to stand upright")
            else:
                self.recovery_count = 0

        elif self.current_state == FallState.RECOVERY:
            is_upright = torso_angle > self.upright_angle_threshold and not check_pos
            if is_upright:
                self.recovery_count += 1
                if self.recovery_count >= self.recovery_normalization_frames:
                    self.recovery_count = 0
                    self.suspected_count = 0
                    self.set_state(
                        FallState.MONITORING,
                        f"User upright and motion stabilized for {self.recovery_normalization_frames} frames",
                    )
            else:
                # If subject collapses back down during recovery attempt
                if fall_flag:
                    self.set_state(FallState.FALL_CONFIRMED, "Subject collapsed during recovery")

        return self.current_state

    def reset(self) -> None:
        """Resets the state machine back to CALIBRATING."""
        self.calibration_count = 0
        self.suspected_count = 0
        self.recovery_count = 0
        self.set_state(FallState.CALIBRATING, "Manual reset")
