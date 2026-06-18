"""
tracker.py — Hand detection & landmark extraction via MediaPipe Tasks API.

Milestone 2: Hand Tracking & ROI Extraction
--------------------------------------------
This module wraps the MediaPipe HandLandmarker (Tasks API, v0.10.35+) to:
  1. Detect hands in each webcam frame.
  2. Return the 21 normalised landmarks per hand.
  3. Draw the landmark skeleton on the frame.
  4. Calculate a tight bounding box around the *palm centre* using
     landmarks 0 (wrist), 1 (thumb CMC), 5 (index MCP), 9 (middle MCP),
     13 (ring MCP) and 17 (pinky MCP).  This deliberately *excludes*
     finger tips so the ROI captures only the palm surface where the
     major lines live.

Design decisions:
  - We use RunningMode.VIDEO so MediaPipe can leverage inter-frame
    tracking rather than re-detecting from scratch every frame — much
    faster.
  - `num_hands=1` because palmistry reads one hand at a time.
    If two hands appear, MediaPipe picks the most confident one.
  - A configurable padding factor lets us expand the bounding box slightly
    so the palm edges are not clipped.
  - The model file (hand_landmarker.task) is auto-downloaded on first run
    if not already present.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from typing import Optional


# Path to the hand landmarker model file.
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
# Google-hosted URL for the float16 hand landmarker model.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model() -> None:
    """
    Download the hand landmarker model if it doesn't already exist.

    Uses urllib from the standard library so we don't need requests or
    any additional dependency.  The model is ~7.8 MB — typically downloads
    in a few seconds.
    """
    if os.path.isfile(_MODEL_PATH):
        return

    print("[INFO] Hand landmarker model not found.  Downloading (~7.8 MB)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[INFO] Model saved to: {_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        print(f"[ERROR] Please download it manually from:\n  {_MODEL_URL}")
        print(f"[ERROR] And place it at: {_MODEL_PATH}")
        sys.exit(1)


class HandTracker:
    """Thin wrapper around MediaPipe HandLandmarker for palm detection & ROI extraction."""

    # The six landmarks that outline the palm (excluding finger tips).
    # 0  = Wrist
    # 1  = Thumb CMC (base of thumb, palm side)
    # 5  = Index finger MCP (knuckle)
    # 9  = Middle finger MCP
    # 13 = Ring finger MCP
    # 17 = Pinky MCP
    PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        presence_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
        roi_padding: float = 0.15,
    ):
        """
        Parameters
        ----------
        max_hands : int
            Maximum number of hands to detect.  We default to 1 because
            palmistry analyses one hand at a time.
        detection_confidence : float
            Minimum confidence for the palm-detection model to consider a
            detection valid.  0.7 is a good balance between false positives
            and missed detections.
        presence_confidence : float
            Minimum confidence that a hand is actually present in the frame.
        tracking_confidence : float
            Minimum confidence to continue tracking a hand between frames
            before falling back to re-detection.
        roi_padding : float
            Fractional padding added around the palm bounding box.  0.15
            means the box is expanded by 15 % on each side.
        """
        self.roi_padding = roi_padding

        # Auto-download the model if missing.
        _ensure_model()

        # Configure the HandLandmarker with the Tasks API.
        # RunningMode.VIDEO enables inter-frame tracking (faster than IMAGE
        # mode, which re-detects from scratch every frame).
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Monotonically increasing timestamp — required by VIDEO mode.
        self._timestamp_ms = 0

        # Store the hand connections for drawing.
        self._hand_connections = HandLandmarksConnections.HAND_CONNECTIONS

    def process(self, frame: np.ndarray) -> HandLandmarkerResult:
        """
        Run hand detection on *frame* (BGR).

        Returns a HandLandmarkerResult containing:
          - hand_landmarks: list of NormalizedLandmarkList (one per hand)
          - hand_world_landmarks: world coordinates
          - handedness: left/right classification

        We convert to RGB internally because MediaPipe expects RGB, but the
        caller always works with BGR (OpenCV's default).
        """
        # Convert BGR → RGB for MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Wrap as a MediaPipe Image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires a monotonically increasing timestamp.
        self._timestamp_ms += 33  # ≈30 FPS increment
        results = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        return results

    def draw_landmarks(self, frame: np.ndarray, results: HandLandmarkerResult) -> np.ndarray:
        """
        Draw the 21-point hand skeleton on *frame* (in-place) for every
        detected hand.

        Returns the same frame (mutated) for convenience.
        """
        if not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand_landmarks in results.hand_landmarks:
            # The new Tasks API returns a list of NormalizedLandmark objects.
            # We need to draw them manually since the drawing_utils expects
            # a specific format.
            landmark_points = []
            for lm in hand_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                landmark_points.append((px, py))
                # Draw each landmark as a small filled circle.
                cv2.circle(frame, (px, py), 5, (0, 0, 255), cv2.FILLED)

            # Draw connections between landmarks.
            for connection in self._hand_connections:
                start_idx = connection.start
                end_idx = connection.end
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    cv2.line(
                        frame,
                        landmark_points[start_idx],
                        landmark_points[end_idx],
                        (0, 255, 0), 2,
                    )

        return frame

    def get_palm_roi(
        self, frame: np.ndarray, results: HandLandmarkerResult
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """
        Extract the palm Region of Interest from *frame*.

        We use landmarks 0, 1, 5, 9, 13, 17 to define the palm polygon,
        compute its axis-aligned bounding box, pad it, clamp to frame
        bounds, and crop.

        Returns
        -------
        (roi_image, (x1, y1, x2, y2))  if a hand is detected.
        None                            if no hand is found.

        The bounding-box tuple is in pixel coordinates and can be used to
        draw a rectangle on the original frame.
        """
        if not results.hand_landmarks:
            return None

        # Use the first (most confident) detected hand.
        hand_landmarks = results.hand_landmarks[0]
        h, w, _ = frame.shape

        # Collect the pixel coordinates of the six palm landmarks.
        palm_points = []
        for idx in self.PALM_LANDMARKS:
            lm = hand_landmarks[idx]
            # MediaPipe landmarks are normalised to [0, 1].
            px = int(lm.x * w)
            py = int(lm.y * h)
            palm_points.append((px, py))

        palm_points = np.array(palm_points)

        # Axis-aligned bounding box around those six points.
        x_min, y_min = palm_points.min(axis=0)
        x_max, y_max = palm_points.max(axis=0)

        # Expand by the padding factor so we don't clip the palm edges.
        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_x = int(box_w * self.roi_padding)
        pad_y = int(box_h * self.roi_padding)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        # Safety: if the box is degenerate (e.g., landmarks collapsed to a
        # single point), return None rather than a 0-pixel crop.
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        roi = frame[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.landmarker.close()
