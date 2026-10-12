"""
MediaPipe Pose Estimation Module with visibility confidence extraction.
Compatible with Al-Amin Abubakar's PoseDetector class and DAÏO architecture.
"""

from typing import List, Tuple, Dict, Optional, Any
import math
import cv2
import numpy as np
import mediapipe as mp


class PoseDetector:
    """
    Encapsulates Google MediaPipe BlazePose to detect and extract 33 skeletal landmarks
    along with normalized coordinates and visibility probabilities.
    """

    def __init__(
        self,
        mode: bool = False,
        smooth: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        detection_con: float = 0.5,
        track_con: float = 0.5,
        model_complexity: int = 1,
    ):
        self.mode = mode
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_con = detection_con
        self.track_con = track_con
        self.model_complexity = model_complexity

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.results: Any = None
        self.lm_list: List[Dict[str, Any]] = []

    def find_pose(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Processes image frame using MediaPipe Pose.
        Args:
            img: BGR image from OpenCV
            draw: whether to draw default skeleton landmarks
        Returns:
            Processed BGR image
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results and self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 180, 0), thickness=2, circle_radius=2),
            )
        return img

    def find_position(
        self,
        img: np.ndarray,
        draw: bool = False,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> List[Dict[str, Any]]:
        """
        Extracts all 33 landmark positions from the detected pose.
        Returns a list of dicts with:
        id, cx, cy (pixel coordinates), nx, ny (normalized [0, 1]), visibility (confidence [0, 1])
        """
        self.lm_list = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                visibility = float(getattr(lm, "visibility", 1.0))
                self.lm_list.append(
                    {
                        "id": lm_id,
                        "cx": cx,
                        "cy": cy,
                        "nx": float(lm.x),
                        "ny": float(lm.y),
                        "visibility": visibility,
                    }
                )
                if draw:
                    cv2.circle(img, (cx, cy), 4, color, cv2.FILLED)
        return self.lm_list

    def get_joint(self, joint_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve landmark by ID if present."""
        if 0 <= joint_id < len(self.lm_list):
            return self.lm_list[joint_id]
        return None

    def get_torso_landmarks(
        self, img_shape: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Computes key anatomical midpoints for DAÏO:
        - Midpoint of Shoulders (L:11, R:12)
        - Midpoint of Hips / Pelvis (L:23, R:24)
        - Ankles (L:27, R:28)
        - Head / Nose (0)
        """
        if len(self.lm_list) < 29:
            return None

        h, w = img_shape[:2]
        l_shoulder = self.lm_list[11]
        r_shoulder = self.lm_list[12]
        l_hip = self.lm_list[23]
        r_hip = self.lm_list[24]
        l_ankle = self.lm_list[27]
        r_ankle = self.lm_list[28]
        nose = self.lm_list[0]

        # Shoulders midpoint
        s_cx = (l_shoulder["cx"] + r_shoulder["cx"]) / 2.0
        s_cy = (l_shoulder["cy"] + r_shoulder["cy"]) / 2.0
        s_vis = min(l_shoulder["visibility"], r_shoulder["visibility"])

        # Hips midpoint
        h_cx = (l_hip["cx"] + r_hip["cx"]) / 2.0
        h_cy = (l_hip["cy"] + r_hip["cy"]) / 2.0
        h_vis = min(l_hip["visibility"], r_hip["visibility"])

        # Ankles midpoint
        a_cx = (l_ankle["cx"] + r_ankle["cx"]) / 2.0
        a_cy = (l_ankle["cy"] + r_ankle["cy"]) / 2.0
        a_vis = min(l_ankle["visibility"], r_ankle["visibility"])

        # Torso height in pixels
        torso_len = math.sqrt((s_cx - h_cx) ** 2 + (s_cy - h_cy) ** 2)

        return {
            "shoulder_mid": (s_cx, s_cy),
            "shoulder_vis": s_vis,
            "hip_mid": (h_cx, h_cy),
            "hip_vis": h_vis,
            "ankle_mid": (a_cx, a_cy),
            "ankle_vis": a_vis,
            "head": (float(nose["cx"]), float(nose["cy"])),
            "head_vis": nose["visibility"],
            "torso_length": max(torso_len, 10.0),
            "l_shoulder": (l_shoulder["cx"], l_shoulder["cy"]),
            "r_shoulder": (r_shoulder["cx"], r_shoulder["cy"]),
            "l_hip": (l_hip["cx"], l_hip["cy"]),
            "r_hip": (r_hip["cx"], r_hip["cy"]),
            "l_ankle": (l_ankle["cx"], l_ankle["cy"]),
            "r_ankle": (r_ankle["cx"], r_ankle["cy"]),
        }

    def find_angle(
        self, img: np.ndarray, p1: int, p2: int, p3: int, draw: bool = True
    ) -> float:
        """Calculates angle between 3 landmarks (p1-p2-p3)."""
        if len(self.lm_list) <= max(p1, p2, p3):
            return 0.0

        x1, y1 = self.lm_list[p1]["cx"], self.lm_list[p1]["cy"]
        x2, y2 = self.lm_list[p2]["cx"], self.lm_list[p2]["cy"]
        x3, y3 = self.lm_list[p3]["cx"], self.lm_list[p3]["cy"]

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 2)
            cv2.circle(img, (x2, y2), 6, (0, 0, 255), cv2.FILLED)
            cv2.putText(
                img,
                f"{int(angle)}deg",
                (x2 - 40, y2 - 20),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                (0, 0, 255),
                2,
            )
        return angle
