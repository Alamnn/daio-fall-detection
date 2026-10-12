"""
DAÏO: A High-Specificity Real-Time Vision-Based Fall Detection System
Using Advanced Kinematic Tracking and Extended Kalman Filter (EKF)

Author: Al-Amin Abubakar
Architecture:
- MediaPipe BlazePose Extraction (33 Landmarks + Visibility Scores)
- Extended Kalman Filter with Adaptive Measurement Noise
- Analytical Torso Angle Jacobian & Covariance Propagation
- Triple-Check Logic Engine (Position, Torso Angle, Kinematic Velocity Latch)
- 5-State Finite State Machine (Calibrating, Monitoring, Suspected, Confirmed, Recovery)
"""

__version__ = "2.0.0"
__author__ = "Al-Amin Abubakar"
