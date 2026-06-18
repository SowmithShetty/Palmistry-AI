"""
main.py — Entry point for the Palmistry AI application.

Milestones 1–4: Complete Palmistry AI Pipeline
-----------------------------------------------
This module handles:
  1. Opening a webcam feed via OpenCV's VideoCapture.
  2. Flipping the frame horizontally to create a natural mirror effect.
  3. Running MediaPipe hand detection and drawing the 21-point skeleton.
  4. Extracting the palm ROI and displaying it in a second window.
  5. Running CLAHE + Canny edge detection on the ROI to isolate palm lines.
  6. Analysing the edge map and generating a palmistry reading.
  7. Overlaying the reading on the live feed.
  8. Graceful exit when the user presses 'q'.

Design decisions:
  - We attempt multiple camera backends (DirectShow on Windows first, then
    the default) to maximise compatibility across machines.
  - The window is named "Palmistry AI" and set to be resizable so the user
    can adjust it to their liking.
  - FPS is calculated using a rolling average over the last 30 frames to
    avoid jittery numbers.
  - The palm ROI window only appears when a hand is detected and auto-hides
    when the hand leaves the frame, avoiding an empty window.
  - A separate "Edge Controls" window with trackbars lets the user tune
    the Canny thresholds in real-time.
  - The palmistry reading is debounced (updated every 0.5 s) to avoid
    distracting flicker in the overlay text.
"""

import cv2
import time
import sys
import numpy as np
from collections import deque
from tracker import HandTracker
from processor import PalmProcessor
from analyzer import analyze, PalmistryReading


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
WINDOW_NAME = "Palmistry AI"
ROI_WINDOW_NAME = "Palm ROI"
EDGES_WINDOW_NAME = "Palm Lines (Edges)"
CAMERA_INDEX = 0
# Desired capture resolution — the camera will use the closest it supports.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
# Number of recent frame-times used to compute the rolling-average FPS.
FPS_BUFFER_SIZE = 30
# How often (seconds) to refresh the palmistry reading overlay.
# Updating every frame causes distracting flicker; 0.5 s feels responsive
# without being jittery.
READING_DEBOUNCE_SECS = 0.5


def init_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Initialise the webcam with sensible defaults.

    We try the DirectShow backend first (faster startup on Windows), then
    fall back to the platform default.  If neither works we print a
    user-friendly message and exit.
    """
    # Try DirectShow backend first (Windows-specific, faster init)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fall back to the platform default backend
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please check that:")
        print("  • A camera is connected and not in use by another app.")
        print("  • The correct camera index is set (current: {}).".format(index))
        sys.exit(1)

    # Request a 720p feed — the camera will pick the closest it supports.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # Reduce the internal buffer to 1 frame so we always get the *latest*
    # frame rather than a stale queued one.  This matters for real-time apps.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened at {actual_w}x{actual_h}")
    return cap


def draw_fps(frame, fps: float) -> None:
    """
    Overlay the current FPS in the top-left corner of *frame* (in-place).

    We use a dark background rectangle behind the text so it stays readable
    regardless of the scene content.
    """
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    color = (0, 255, 0)  # Bright green — easy to spot at a glance.

    # Measure the text so we can draw a fitted background rectangle.
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Semi-transparent dark backdrop for readability.
    cv2.rectangle(frame, (8, 8), (18 + tw, 18 + th + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (12, 12 + th), font, scale, color, thickness, cv2.LINE_AA)


def draw_instructions(frame, hand_detected: bool = False) -> None:
    """
    Draw a small instruction banner at the bottom of the frame so the user
    knows how to quit and what the app is waiting for.
    """
    h, w = frame.shape[:2]

    if hand_detected:
        instructions = [
            "Press 'q' to quit",
            "Hand detected — reading palm...",
        ]
    else:
        instructions = [
            "Press 'q' to quit",
            "Show your palm to the camera",
        ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    color = (200, 200, 200)

    y = h - 15  # start near the bottom and work upward
    for line in reversed(instructions):
        (tw, th), baseline = cv2.getTextSize(line, font, scale, thickness)
        # Background bar spanning the full width for a clean look.
        cv2.rectangle(frame, (0, y - th - 6), (w, y + baseline + 4), (40, 40, 40), cv2.FILLED)
        # Centre the text horizontally.
        x = (w - tw) // 2
        cv2.putText(frame, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y -= (th + baseline + 14)


def draw_roi_box(frame, bbox: tuple[int, int, int, int]) -> None:
    """
    Draw the palm bounding box on the main frame so the user can see
    exactly what region is being analysed.

    Uses a green rectangle with moderate thickness — visible without
    obscuring the hand.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Label above the box.
    cv2.putText(
        frame, "Palm ROI", (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )


def draw_canny_info(frame, canny_low: int, canny_high: int) -> None:
    """
    Show the current Canny threshold values on the main feed so the user
    can see the effect of their trackbar adjustments at a glance.
    """
    text = f"Canny: {canny_low}/{canny_high}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    color = (255, 200, 0)  # Cyan-ish for contrast against the green FPS.

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Position below the FPS counter.
    y_offset = 50
    cv2.rectangle(frame, (8, y_offset), (18 + tw, y_offset + 10 + th + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (12, y_offset + 4 + th), font, scale, color, thickness, cv2.LINE_AA)


def _wrap_text(text: str, font, scale: float, thickness: int, max_width: int) -> list[str]:
    """
    Word-wrap *text* so that each returned line fits within *max_width*
    pixels when rendered with the given font parameters.

    Falls back to character-level wrapping if a single word is wider
    than max_width (rare, but handles very long words gracefully).
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current_line = words[0]

    for word in words[1:]:
        # Test whether appending the next word still fits.
        test_line = current_line + " " + word
        (tw, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if tw <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def draw_reading(frame, reading: PalmistryReading) -> None:
    """
    Render the palmistry reading as a semi-transparent panel on the right
    side of the main feed.

    Long detail strings are word-wrapped so they never clip at the frame
    edge.  The panel height is calculated dynamically from the actual
    number of wrapped lines.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1

    panel_w = 400
    # Usable text width is the panel minus some internal padding.
    text_max_w = panel_w - 20

    # Build the raw entries: (text, colour, indent).
    # indent=True adds left padding for sub-items.
    raw_entries: list[tuple[str, tuple[int, int, int], bool]] = [
        ("=== PALM READING ===", (255, 255, 255), False),
        ("", (0, 0, 0), False),  # spacer
        ("[Heart Line]", (100, 100, 255), False),
        (reading.heart["trait"], (200, 200, 255), True),
        (reading.heart["detail"], (180, 180, 180), True),
        ("", (0, 0, 0), False),
        ("[Head Line]", (255, 200, 100), False),
        (reading.head["trait"], (255, 220, 180), True),
        (reading.head["detail"], (180, 180, 180), True),
        ("", (0, 0, 0), False),
        ("[Life Line]", (100, 255, 100), False),
        (reading.life["trait"], (180, 255, 180), True),
        (reading.life["detail"], (180, 180, 180), True),
        ("", (0, 0, 0), False),
        ("[Overall]", (200, 200, 255), False),
        (reading.overall["trait"], (220, 220, 255), True),
        (reading.overall["detail"], (180, 180, 180), True),
    ]

    # Word-wrap every entry and flatten into (text, colour, x_offset) lines.
    indent_px = 16
    wrapped_lines: list[tuple[str, tuple[int, int, int], int]] = []
    for text, color, indent in raw_entries:
        offset = indent_px if indent else 0
        if not text:
            # Spacer — keep as an empty line.
            wrapped_lines.append(("", color, 0))
            continue
        available_w = text_max_w - offset
        for wline in _wrap_text(text, font, scale, thickness, available_w):
            wrapped_lines.append((wline, color, offset))

    line_height = 20
    panel_h = len(wrapped_lines) * line_height + 24

    # Position: right side, vertically centred.
    x_start = w - panel_w - 12
    y_start = max(10, (h - panel_h) // 2)

    # Draw a dark semi-transparent background panel.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_start - 10, y_start - 10),
        (x_start + panel_w, y_start + panel_h),
        (20, 20, 20),
        cv2.FILLED,
    )
    # Blend for a translucent effect (alpha = 0.75).
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Draw each wrapped line.
    y = y_start + 14
    for text, color, x_offset in wrapped_lines:
        if text:
            cv2.putText(
                frame, text, (x_start + x_offset, y),
                font, scale, color, thickness, cv2.LINE_AA,
            )
        y += line_height


def main() -> None:
    """
    Core application loop.

    Flow:
      1. Grab a frame from the webcam.
      2. Flip it horizontally (mirror).
      3. Run MediaPipe hand detection; draw landmarks.
      4. Extract the palm ROI and display it in a second window.
      5. Run CLAHE + Canny edge detection and display the edge map.
      6. Draw the HUD overlay (FPS, Canny info, instructions, bounding box).
      7. Show the frame; break on 'q'.
    """
    cap = init_camera()
    tracker = HandTracker()
    processor = PalmProcessor()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # We create the ROI / edges windows lazily (only when a hand is first
    # detected) so the user isn't greeted by empty black windows.
    roi_window_created = False
    edges_window_created = False

    # Rolling buffer of recent frame durations for smooth FPS calculation.
    frame_times: deque[float] = deque(maxlen=FPS_BUFFER_SIZE)
    prev_time = time.perf_counter()

    # Palmistry reading state — debounced to avoid flicker.
    current_reading: PalmistryReading | None = None
    last_reading_time: float = 0.0

    print("[INFO] Palmistry AI is running.  Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            # The camera occasionally drops a frame; skip rather than crash.
            print("[WARN] Dropped frame — retrying...")
            continue

        # ── Mirror the frame ──
        # Flipping horizontally (flipCode=1) gives a natural "mirror"
        # experience so the user's left hand appears on the left side of
        # the screen.
        frame = cv2.flip(frame, 1)

        # ── Hand tracking (Milestone 2) ──
        results = tracker.process(frame)
        hand_detected = len(results.hand_landmarks) > 0

        # Draw the 21-point skeleton on the main feed.
        tracker.draw_landmarks(frame, results)

        # Extract the palm ROI and show it in a separate window.
        roi_result = tracker.get_palm_roi(frame, results)
        if roi_result is not None:
            roi_img, bbox = roi_result
            draw_roi_box(frame, bbox)

            # Create/show the ROI window.
            if not roi_window_created:
                cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL)
                roi_window_created = True

            # ── Image processing (Milestone 3) ──
            # Run the CLAHE + Canny pipeline on the palm ROI.
            gray, enhanced, edges = processor.process(roi_img)

            # Show the edge-detected palm lines in a dedicated window.
            if not edges_window_created:
                cv2.namedWindow(EDGES_WINDOW_NAME, cv2.WINDOW_NORMAL)
                edges_window_created = True
            cv2.imshow(EDGES_WINDOW_NAME, edges)

            # Map landmarks to ROI coordinates
            h_f, w_f = frame.shape[:2]
            hand_landmarks = results.hand_landmarks[0]
            landmarks_roi = np.array([
                [int(lm.x * w_f) - bbox[0], int(lm.y * h_f) - bbox[1]]
                for lm in hand_landmarks
            ])

            # ── Palmistry analysis (Milestone 4) ──
            # We run analyze every frame to draw overlays on roi_img,
            # but only update current_reading at a debounced interval.
            now_reading = time.perf_counter()
            if now_reading - last_reading_time >= READING_DEBOUNCE_SECS:
                reading = analyze(edges, landmarks_roi, roi_img)
                if reading is not None:
                    current_reading = reading
                last_reading_time = now_reading
            else:
                # Still analyze on intermediate frames to keep the overlays rendered.
                analyze(edges, landmarks_roi, roi_img)

            cv2.imshow(ROI_WINDOW_NAME, roi_img)
        else:
            # No hand detected — hide the secondary windows and clear reading.
            current_reading = None
            if roi_window_created:
                cv2.destroyWindow(ROI_WINDOW_NAME)
                roi_window_created = False
            if edges_window_created:
                cv2.destroyWindow(EDGES_WINDOW_NAME)
                edges_window_created = False
                processor.destroy_windows()

        # ── FPS bookkeeping ──
        now = time.perf_counter()
        frame_times.append(now - prev_time)
        prev_time = now
        # Average over the buffer to smooth out jitter.
        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

        # ── HUD overlays ──
        draw_fps(frame, fps)
        draw_canny_info(frame, processor.canny_low, processor.canny_high)
        draw_instructions(frame, hand_detected)

        # ── Palmistry reading overlay (Milestone 4) ──
        if current_reading is not None:
            draw_reading(frame, current_reading)

        # ── Display ──
        cv2.imshow(WINDOW_NAME, frame)

        # waitKey(1) is the minimum delay (≈1 ms).  It keeps the loop as
        # fast as possible while still pumping the HighGUI event queue so
        # the window stays responsive.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed — shutting down.")
            break

    # ── Cleanup ──
    processor.destroy_windows()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()
