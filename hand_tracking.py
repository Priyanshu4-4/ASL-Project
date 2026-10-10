"""
Shared helper for using MediaPipe's Tasks HandLandmarker (the API that
replaced the old, now-removed mp.solutions.hands in MediaPipe 1.0+).
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# Standard 21-point hand skeleton connections, used since drawing_utils
# was removed along with the old solutions API.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def create_landmarker(running_mode=None):
    """Create a HandLandmarker.

    running_mode defaults to VIDEO (for webcam scripts, one call per frame
    in order). Pass vision.RunningMode.IMAGE for processing a folder of
    independent still images (e.g. a downloaded dataset).
    """
    if running_mode is None:
        running_mode = vision.RunningMode.VIDEO
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run download_model.py first."
        )
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def frame_to_mp_image(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_landmarks(frame, hand_landmarks):
    """hand_landmarks: list of NormalizedLandmark for one hand."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    _draw_points_and_connections(frame, points)


def draw_skeleton_xyz(canvas, landmarks_xyz):
    """landmarks_xyz: (21, 3) array-like of normalized (x, y, z) coordinates.
    Used for playback of recorded/averaged landmarks (not live detection
    results), e.g. in text_to_gesture.py."""
    h, w = canvas.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, _ in landmarks_xyz]
    _draw_points_and_connections(canvas, points)


def _draw_points_and_connections(canvas, points):
    for start, end in HAND_CONNECTIONS:
        cv2.line(canvas, points[start], points[end], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)
