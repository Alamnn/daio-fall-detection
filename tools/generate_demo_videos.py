"""
Generates synthetic test benchmark videos for DAÏO Fall Detection System.
Creates test scenarios in DataVideos/:
1. fall_forward.mp4         - Uncontrolled rapid fall forward (True Positive)
2. normal_walking.mp4       - Upright walking (True Negative ADL)
3. deep_squat_adl.mp4       - Fast squatting / rapid sitting (ADL rejection via Check 2)
4. prostration_prayer_adl.mp4 - Controlled bowing & prostration (ADL rejection via Check 3)
5. temporary_occlusion.mp4   - Occluded tracking test for EKF adaptive noise
"""

from typing import Tuple
import os
import cv2
import numpy as np


def draw_humanoid(
    img: np.ndarray,
    head_pos: Tuple[int, int],
    shoulder_pos: Tuple[int, int],
    hip_pos: Tuple[int, int],
    l_knee: Tuple[int, int],
    r_knee: Tuple[int, int],
    l_ankle: Tuple[int, int],
    r_ankle: Tuple[int, int],
    l_hand: Tuple[int, int],
    r_hand: Tuple[int, int],
    color: Tuple[int, int, int] = (230, 230, 230),
) -> None:
    """Renders an articulated human figure with realistic thickness."""
    # Head
    cv2.circle(img, head_pos, 22, (200, 180, 150), cv2.FILLED)
    cv2.circle(img, head_pos, 22, (100, 100, 100), 2)

    # Torso (Chest to Pelvis)
    cv2.line(img, shoulder_pos, hip_pos, (50, 120, 200), 26)

    # Arms
    cv2.line(img, shoulder_pos, l_hand, (200, 180, 150), 10)
    cv2.line(img, shoulder_pos, r_hand, (200, 180, 150), 10)

    # Legs
    cv2.line(img, hip_pos, l_knee, (60, 60, 70), 14)
    cv2.line(img, l_knee, l_ankle, (60, 60, 70), 12)

    cv2.line(img, hip_pos, r_knee, (60, 60, 70), 14)
    cv2.line(img, r_knee, r_ankle, (60, 60, 70), 12)

    # Feet
    cv2.line(img, l_ankle, (l_ankle[0] + 15, l_ankle[1]), (30, 30, 30), 8)
    cv2.line(img, r_ankle, (r_ankle[0] + 15, r_ankle[1]), (30, 30, 30), 8)


def create_room_background(width: int = 640, height: int = 480) -> np.ndarray:
    """Creates a realistic room scene with floor and walls."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Wall (top 70%)
    wall_h = int(height * 0.70)
    bg[:wall_h, :] = (235, 230, 220)
    # Floor (bottom 30%)
    bg[wall_h:, :] = (140, 160, 175)
    # Baseboard line
    cv2.line(bg, (0, wall_h), (width, wall_h), (90, 80, 70), 4)
    # Grid lines on floor for perspective
    for x in range(0, width, 80):
        cv2.line(bg, (x, wall_h), (x - 60, height), (120, 140, 155), 1)
    return bg


def generate_fall_forward_video(filepath: str, num_frames: int = 180, fps: int = 30) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, (640, 480))

    for f in range(num_frames):
        frame = create_room_background(640, 480)
        # Sequence:
        # 0-60: Standing & calibrating
        # 60-80: Walking slowly
        # 80-95: Rapid fall forward (sudden drop in y, angle rotates to 0 deg)
        # 95-180: Motionless on floor
        if f < 60:
            # Steady standing
            x = 320
            head = (x, 160)
            shoulder = (x, 200)
            hip = (x, 270)
            l_knee, r_knee = (x - 15, 335), (x + 15, 335)
            l_ankle, r_ankle = (x - 15, 400), (x + 15, 400)
            l_hand, r_hand = (x - 35, 260), (x + 35, 260)
        elif f < 80:
            # Walking
            step = f - 60
            x = 320 + int(step * 1.5)
            bob = int(np.sin(step * 0.5) * 6)
            head = (x, 160 + bob)
            shoulder = (x, 200 + bob)
            hip = (x, 270 + bob)
            l_knee = (x - 15 + int(np.sin(step * 0.5) * 15), 335)
            r_knee = (x + 15 - int(np.sin(step * 0.5) * 15), 335)
            l_ankle = (x - 15 + int(np.sin(step * 0.5) * 20), 400)
            r_ankle = (x + 15 - int(np.sin(step * 0.5) * 20), 400)
            l_hand = (x - 30 - int(np.sin(step * 0.5) * 15), 260)
            r_hand = (x + 30 + int(np.sin(step * 0.5) * 15), 260)
        elif f < 95:
            # Fast Fall Forward (15 frames = 0.5s rapid downward acceleration)
            fall_prog = (f - 80) / 15.0
            x = 350 + int(fall_prog * 80)
            head_y = int(160 + fall_prog * (410 - 160))
            shoulder_y = int(200 + fall_prog * (415 - 200))
            hip_y = int(270 + fall_prog * (420 - 270))
            
            head = (x + 70, head_y)
            shoulder = (x + 40, shoulder_y)
            hip = (x - 40, hip_y)
            l_knee, r_knee = (x - 70, 420), (x - 60, 420)
            l_ankle, r_ankle = (x - 110, 420), (x - 100, 420)
            l_hand, r_hand = (x + 50, head_y + 10), (x + 30, head_y + 15)
        else:
            # Lying motionless on the floor
            x = 430
            head = (x + 70, 410)
            shoulder = (x + 40, 415)
            hip = (x - 40, 420)
            l_knee, r_knee = (x - 70, 420), (x - 60, 420)
            l_ankle, r_ankle = (x - 110, 420), (x - 100, 420)
            l_hand, r_hand = (x + 50, 420), (x + 30, 425)

        draw_humanoid(frame, head, shoulder, hip, l_knee, r_knee, l_ankle, r_ankle, l_hand, r_hand)
        out.write(frame)

    out.release()
    print(f"Generated: {filepath}")


def generate_deep_squat_video(filepath: str, num_frames: int = 180, fps: int = 30) -> None:
    """Rapid deep squat ADL: hips drop low, BUT torso remains strictly upright (>60 deg)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, (640, 480))

    for f in range(num_frames):
        frame = create_room_background(640, 480)
        x = 320
        if f < 60:
            # Standing steady
            head = (x, 160)
            shoulder = (x, 200)
            hip = (x, 270)
            l_knee, r_knee = (x - 15, 335), (x + 15, 335)
            l_ankle, r_ankle = (x - 15, 400), (x + 15, 400)
            l_hand, r_hand = (x - 35, 260), (x + 35, 260)
        elif f < 100:
            # Squatting down rapidly
            prog = (f - 60) / 40.0
            squat_offset = int(prog * 100)
            head = (x, 160 + squat_offset)
            shoulder = (x, 200 + squat_offset)
            hip = (x, 270 + squat_offset)
            l_knee, r_knee = (x - 35, 370), (x + 35, 370)
            l_ankle, r_ankle = (x - 15, 400), (x + 15, 400)
            l_hand, r_hand = (x - 30, 260 + squat_offset), (x + 30, 260 + squat_offset)
        else:
            # Holding squat then standing back up
            prog = max(0.0, 1.0 - (f - 100) / 40.0)
            squat_offset = int(prog * 100)
            head = (x, 160 + squat_offset)
            shoulder = (x, 200 + squat_offset)
            hip = (x, 270 + squat_offset)
            l_knee, r_knee = (x - 35, 370), (x + 35, 370)
            l_ankle, r_ankle = (x - 15, 400), (x + 15, 400)
            l_hand, r_hand = (x - 30, 260 + squat_offset), (x + 30, 260 + squat_offset)

        draw_humanoid(frame, head, shoulder, hip, l_knee, r_knee, l_ankle, r_ankle, l_hand, r_hand)
        out.write(frame)

    out.release()
    print(f"Generated: {filepath}")


def generate_prayer_prostration_video(filepath: str, num_frames: int = 180, fps: int = 30) -> None:
    """Slow, controlled bowing & prostration: low angle & height, but very slow velocity (<1.0 m/s)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, (640, 480))

    for f in range(num_frames):
        frame = create_room_background(640, 480)
        x = 320
        if f < 60:
            # Standing steady
            head = (x, 160)
            shoulder = (x, 200)
            hip = (x, 270)
            l_knee, r_knee = (x - 15, 335), (x + 15, 335)
            l_ankle, r_ankle = (x - 15, 400), (x + 15, 400)
            l_hand, r_hand = (x - 35, 260), (x + 35, 260)
        elif f < 130:
            # Very slow, gentle prostration transition over 70 frames (slow velocity < 0.8 m/s)
            prog = (f - 60) / 70.0
            head_x = x + int(prog * 80)
            head_y = int(160 + prog * 240)
            shoulder_x = x + int(prog * 50)
            shoulder_y = int(200 + prog * 190)
            hip_x = x - int(prog * 10)
            hip_y = int(270 + prog * 110)

            head = (head_x, head_y)
            shoulder = (shoulder_x, shoulder_y)
            hip = (hip_x, hip_y)
            l_knee, r_knee = (x - 20, 385), (x + 10, 385)
            l_ankle, r_ankle = (x - 25, 400), (x + 5, 400)
            l_hand, r_hand = (head_x - 15, head_y), (head_x + 15, head_y)
        else:
            # Prostrating quietly
            head = (400, 400)
            shoulder = (370, 390)
            hip = (310, 380)
            l_knee, r_knee = (300, 390), (330, 390)
            l_ankle, r_ankle = (295, 400), (325, 400)
            l_hand, r_hand = (385, 405), (415, 405)

        draw_humanoid(frame, head, shoulder, hip, l_knee, r_knee, l_ankle, r_ankle, l_hand, r_hand)
        out.write(frame)

    out.release()
    print(f"Generated: {filepath}")


def generate_occlusion_video(filepath: str, num_frames: int = 180, fps: int = 30) -> None:
    """Subject walks past a foreground table / wall that temporarily occludes hips & knees."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, (640, 480))

    for f in range(num_frames):
        frame = create_room_background(640, 480)
        # Walk left to right
        x = 100 + int((f / float(num_frames)) * 440)
        head = (x, 160)
        shoulder = (x, 200)
        hip = (x, 270)
        bob = int(np.sin(f * 0.4) * 15)
        l_knee = (x - 15 + bob, 335)
        r_knee = (x + 15 - bob, 335)
        l_ankle = (x - 15 + bob, 400)
        r_ankle = (x + 15 - bob, 400)
        l_hand = (x - 35 - bob, 260)
        r_hand = (x + 35 + bob, 260)

        draw_humanoid(frame, head, shoulder, hip, l_knee, r_knee, l_ankle, r_ankle, l_hand, r_hand)

        # Draw foreground occlusion table in middle of room (x: 240 to 380, y: 240 to 440)
        table_x1, table_x2 = 240, 380
        table_y1, table_y2 = 240, 440
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), (80, 50, 30), cv2.FILLED)
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), (50, 30, 20), 3)
        cv2.putText(frame, "Occlusion Barrier", (table_x1 + 10, table_y1 + 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"Generated: {filepath}")


if __name__ == "__main__":
    data_dir = "DataVideos"
    os.makedirs(data_dir, exist_ok=True)
    generate_fall_forward_video(os.path.join(data_dir, "fall_forward.mp4"))
    generate_deep_squat_video(os.path.join(data_dir, "deep_squat_adl.mp4"))
    generate_prayer_prostration_video(os.path.join(data_dir, "prostration_prayer_adl.mp4"))
    generate_occlusion_video(os.path.join(data_dir, "temporary_occlusion.mp4"))
    print("All demo benchmark videos successfully generated in DataVideos/!")
