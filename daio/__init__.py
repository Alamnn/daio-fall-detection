"""
DAÏO: A High-Specificity Real-Time Vision-Based Fall Detection System
Using Advanced Kinematic Tracking and Extended Kalman Filter (EKF)

Author: Alamin Abubakar Nataala (SEGi University)
Publication: https://www.researchgate.net/publication/410377796_DAIO_A_High-Specificity_Real-TimeVision-Based_Fall_Detection_SystemUsing_Advanced_Kinematic_Tracking_andComputer_Vision
Architecture:
- MediaPipe BlazePose Extraction (33 Landmarks + Visibility Scores)
- Extended Kalman Filter with Adaptive Measurement Noise
- Analytical Torso Angle Jacobian & Covariance Propagation
- Triple-Check Logic Engine (Position, Torso Angle, Kinematic Velocity Latch)
- 5-State Finite State Machine (Calibrating, Monitoring, Suspected, Confirmed, Recovery)
"""

__version__ = "2.0.0"
__author__ = "Alamin Abubakar Nataala"
__institution__ = "SEGi University"
__url__ = "https://www.researchgate.net/publication/410377796_DAIO_A_High-Specificity_Real-TimeVision-Based_Fall_Detection_SystemUsing_Advanced_Kinematic_Tracking_andComputer_Vision"
