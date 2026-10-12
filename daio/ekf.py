"""
Extended Kalman Filter (EKF) with Adaptive Measurement Noise & Torso Kinematics
Based on DAÏO architecture by Al-Amin Abubakar.
"""

from typing import Tuple, Optional
import numpy as np


class ExtendedKalmanFilter:
    """
    4D State-space Kalman Filter tracking planar joint dynamics:
    State: x = [p_x, p_y, v_x, v_y]^T
    Observations: z = [p_x, p_y]^T
    
    Includes DAÏO Adaptive Measurement Noise:
    R(k) = (R_base / (v(k) + eps)) * I_2
    where v(k) is the MediaPipe joint visibility confidence score.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        initial_point: Optional[Tuple[float, float]] = None,
        sigma_p_sq: float = 0.01,
        sigma_v_sq: float = 0.1,
        r_base: float = 0.05,
        eps: float = 1e-5,
    ):
        self.dt = dt
        self.r_base = r_base
        self.eps = eps

        # Initial state vector x = [px, py, vx, vy]^T
        if initial_point is not None:
            self.x = np.array([[initial_point[0]], [initial_point[1]], [0.0], [0.0]], dtype=np.float64)
        else:
            self.x = np.zeros((4, 1), dtype=np.float64)

        # State transition matrix F
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Process noise covariance Q
        self.Q = np.array(
            [
                [sigma_p_sq, 0.0, 0.0, 0.0],
                [0.0, sigma_p_sq, 0.0, 0.0],
                [0.0, 0.0, sigma_v_sq, 0.0],
                [0.0, 0.0, 0.0, sigma_v_sq],
            ],
            dtype=np.float64,
        )

        # Measurement mapping matrix H
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Initial state covariance P
        self.P = np.eye(4, dtype=np.float64) * 0.1

        # Diagnostic metrics
        self.last_kalman_gain: np.ndarray = np.zeros((4, 2))
        self.last_visibility: float = 1.0
        self.last_innovation: np.ndarray = np.zeros((2, 1))

    def predict(self) -> np.ndarray:
        """
        EKF Prediction Step:
        x_hat^-(k) = F * x_hat(k-1)
        P^-(k) = F * P(k-1) * F^T + Q
        """
        self.x = self.F @ self.x
        self.P = (self.F @ self.P @ self.F.T) + self.Q
        return self.x

    def update(self, z: Tuple[float, float], visibility: float = 1.0) -> np.ndarray:
        """
        EKF Correction/Update Step with Adaptive Measurement Noise:
        R(k) = (R_base / (v(k) + eps)) * I_2
        
        If visibility is near 0 (occlusion), R -> infinity, Kalman Gain K -> 0,
        bypassing measurement update and relying entirely on process model.
        """
        self.last_visibility = float(np.clip(visibility, 0.0, 1.0))

        # Clamp tiny visibility to prevent numerical overflow, but allow R to scale high
        v_clamped = max(self.last_visibility, 1e-4)
        r_val = self.r_base / (v_clamped + self.eps)
        R_k = np.eye(2, dtype=np.float64) * r_val

        z_vec = np.array([[z[0]], [z[1]]], dtype=np.float64)
        
        # Innovation (measurement residual)
        y = z_vec - (self.H @ self.x)
        self.last_innovation = y

        # Innovation covariance S = H * P^- * H^T + R(k)
        S = (self.H @ self.P @ self.H.T) + R_k

        # Kalman Gain K = P^- * H^T * inv(S)
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ self.H.T @ S_inv
        except np.linalg.LinAlgError:
            K = np.zeros((4, 2), dtype=np.float64)

        self.last_kalman_gain = K

        # State and Covariance Update
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float64)
        self.P = (I - (K @ self.H)) @ self.P

        return self.x

    @property
    def position(self) -> Tuple[float, float]:
        """Returns smoothed (px, py) coordinates."""
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Returns smoothed (vx, vy) velocities in units/sec."""
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def speed(self) -> float:
        """Euclidean velocity magnitude."""
        vx, vy = self.velocity
        return float(np.sqrt(vx**2 + vy**2))


class TorsoKinematics:
    """
    Computes human torso geometric tilt and analytical Jacobian derivatives
    between the shoulder midpoint and hip midpoint as defined in DAÏO Section 3.2.3.
    """

    @staticmethod
    def calculate_angle_and_jacobian(
        shoulder_pos: Tuple[float, float],
        hip_pos: Tuple[float, float],
        shoulder_cov: Optional[np.ndarray] = None,
        hip_cov: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Calculate torso angle theta relative to the ground plane,
        along with the first-order analytical Jacobian and propagated variance.

        In screen coordinates:
        (x=0, y=0) is top-left.
        Upright body: shoulder_y < hip_y.
        Vector: dx = shoulder_x - hip_x, dy = hip_y - shoulder_y (pointing upwards).
        Angle theta relative to horizontal ground plane:
        theta = atan2(dy, dx) in degrees.
        
        Returns:
            theta_deg: Angle in degrees in [-90, 90] or [0, 90] relative to horizontal.
            Jacobian J_theta: 1x4 matrix [dtheta/dpx_s, dtheta/dpy_s, dtheta/dpx_h, dtheta/dpy_h]
            sigma_theta_sq: Propagated angle variance in rad^2.
        """
        px_s, py_s = shoulder_pos
        px_h, py_h = hip_pos

        dx = px_s - px_h
        # In image coordinates, y increases downwards.
        # Height difference: dy = py_h - py_s (positive when head/shoulders are above hips)
        dy = py_h - py_s

        denom = dx**2 + dy**2
        if denom < 1e-6:
            return 90.0, np.zeros((1, 4)), 0.0

        # Angle relative to horizontal ground axis in degrees:
        # 90 degrees = perfectly upright standing
        # 0 degrees = perfectly horizontal (lying on ground)
        theta_rad = np.arctan2(dy, np.abs(dx))
        theta_deg = float(np.degrees(theta_rad))

        # Analytical Jacobian derivatives:
        # theta = atan2(py_h - py_s, px_s - px_h)
        # d(theta)/d(px_s) = dy / denom
        # d(theta)/d(py_s) = -dx / denom
        # d(theta)/d(px_h) = -dy / denom
        # d(theta)/d(py_h) = dx / denom
        J_theta = np.array(
            [[dy / denom, -dx / denom, -dy / denom, dx / denom]],
            dtype=np.float64,
        )

        sigma_theta_sq = 0.0
        if shoulder_cov is not None and hip_cov is not None:
            # Combined positional covariance matrix (4x4)
            P_combined = np.zeros((4, 4), dtype=np.float64)
            P_combined[0:2, 0:2] = shoulder_cov[0:2, 0:2]
            P_combined[2:4, 2:4] = hip_cov[0:2, 0:2]
            sigma_theta_sq = float((J_theta @ P_combined @ J_theta.T)[0, 0])

        return theta_deg, J_theta, sigma_theta_sq
