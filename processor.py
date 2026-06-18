"""
processor.py — Image preprocessing & edge detection for palm line isolation.

Milestone 3: Image Preprocessing & Line Detection
---------------------------------------------------
This module takes the cropped palm ROI and runs a multi-stage pipeline to
isolate the major palm lines (Heart, Head, Life):

  1. **Grayscale conversion** — Removes colour information so we can work
     purely on intensity gradients, which is what palm lines are.

  2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — Unlike
     regular histogram equalization, CLAHE operates on small tiles and
     clips the histogram to avoid amplifying noise.  This is critical for
     palms because the lines are often faint against similarly-coloured
     skin.  The clip limit and tile size are tunable.

  3. **Gaussian Blur** — A low-pass filter that smooths out fine skin
     texture (pores, wrinkles) while preserving the broader palm lines.
     The kernel size controls the trade-off: larger = more noise removed
     but also more line detail lost.

  4. **Canny Edge Detection** — A two-threshold hysteresis algorithm that
     finds strong edges (above the high threshold), weak edges (between
     the thresholds), and keeps weak edges only if they're connected to
     strong ones.  This is ideal for palm lines because it suppresses
     isolated noise pixels while preserving continuous line structures.

Design decisions:
  - We expose the Canny thresholds as OpenCV trackbars so the user can
    tune them in real-time without restarting the app.
  - CLAHE clip limit and blur kernel size are also configurable but via
    constructor args rather than trackbars (they change less often).
  - The window for trackbars is created lazily on first call to avoid
    empty windows when no hand is in view.
"""

import cv2
import numpy as np


class PalmProcessor:
    """
    Multi-stage image processing pipeline for palm line extraction.

    Usage:
        processor = PalmProcessor()
        edges = processor.process(roi_image)
    """

    # Default Canny thresholds — good starting points for palm lines.
    DEFAULT_CANNY_LOW = 30
    DEFAULT_CANNY_HIGH = 100

    # Trackbar window name.
    TRACKBAR_WINDOW = "Edge Controls"

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_tile_size: tuple[int, int] = (8, 8),
        blur_kernel_size: int = 5,
    ):
        """
        Parameters
        ----------
        clahe_clip_limit : float
            Contrast clipping limit for CLAHE.  Higher values give more
            contrast but also more noise.  2.0–3.0 works well for skin.
        clahe_tile_size : tuple[int, int]
            Size of the tiles CLAHE divides the image into.  Smaller tiles
            give more localised enhancement.  8×8 is a sensible default.
        blur_kernel_size : int
            Side length of the Gaussian blur kernel.  Must be odd.
            5 gives a mild blur that preserves major lines.
        """
        # Ensure the blur kernel is odd (OpenCV requires it).
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size,
        )
        self._blur_ksize = blur_kernel_size

        # Canny thresholds — will be controlled by trackbars.
        self._canny_low = self.DEFAULT_CANNY_LOW
        self._canny_high = self.DEFAULT_CANNY_HIGH

        # Lazy init flag for the trackbar window.
        self._trackbar_created = False

    # ──────────────────────────────────────────────────
    # Trackbar management
    # ──────────────────────────────────────────────────

    def _ensure_trackbars(self) -> None:
        """
        Create the trackbar window and sliders once, on first call.

        We use a dedicated named window so the trackbars don't clutter
        the main feed or the ROI window.
        """
        if self._trackbar_created:
            return

        cv2.namedWindow(self.TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TRACKBAR_WINDOW, 400, 150)

        # Canny low threshold: 0–300
        cv2.createTrackbar(
            "Canny Low", self.TRACKBAR_WINDOW,
            self._canny_low, 300,
            self._on_canny_low,
        )
        # Canny high threshold: 0–500
        cv2.createTrackbar(
            "Canny High", self.TRACKBAR_WINDOW,
            self._canny_high, 500,
            self._on_canny_high,
        )

        self._trackbar_created = True

    def _on_canny_low(self, val: int) -> None:
        """Trackbar callback — update the low Canny threshold."""
        self._canny_low = val

    def _on_canny_high(self, val: int) -> None:
        """Trackbar callback — update the high Canny threshold."""
        self._canny_high = val

    # ──────────────────────────────────────────────────
    # Core processing pipeline
    # ──────────────────────────────────────────────────

    def process(self, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the full preprocessing pipeline on a palm ROI image.

        Parameters
        ----------
        roi : np.ndarray
            The cropped palm region in BGR colour (from tracker.get_palm_roi).

        Returns
        -------
        (grayscale, clahe_enhanced, edges)
            - grayscale: the raw grayscale conversion
            - clahe_enhanced: after CLAHE + Gaussian blur
            - edges: the final Canny edge map (binary, white-on-black)

        All returned images have the same spatial dimensions as the input.
        """
        # Ensure trackbars exist so the user can adjust thresholds.
        self._ensure_trackbars()

        # Step 1: Grayscale
        # Colour adds no useful information for line detection and
        # would triple the computation.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Step 2: CLAHE
        # Adaptive histogram equalisation enhances the faint palm lines
        # without blowing out already-bright regions (which regular
        # equalisation would do).
        enhanced = self._clahe.apply(gray)

        # Step 3: Gaussian Blur
        # Smooths out fine skin texture (pores, tiny wrinkles) that would
        # otherwise produce noisy edges.  sigmaX=0 lets OpenCV compute the
        # optimal sigma from the kernel size.
        blurred = cv2.GaussianBlur(
            enhanced,
            (self._blur_ksize, self._blur_ksize),
            sigmaX=0,
        )

        # Step 4: Canny Edge Detection
        # The two-threshold hysteresis approach is perfect for palm lines:
        #   - Pixels with gradient > canny_high are "strong" edges (kept).
        #   - Pixels between canny_low and canny_high are "weak" edges
        #     (kept only if connected to a strong edge).
        #   - Pixels below canny_low are discarded.
        # This naturally filters out isolated noise while preserving the
        # continuous structure of palm lines.
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)

        return gray, enhanced, edges

    @property
    def canny_low(self) -> int:
        """Current Canny low threshold (for external inspection)."""
        return self._canny_low

    @property
    def canny_high(self) -> int:
        """Current Canny high threshold (for external inspection)."""
        return self._canny_high

    def destroy_windows(self) -> None:
        """Destroy the trackbar window if it was created."""
        if self._trackbar_created:
            cv2.destroyWindow(self.TRACKBAR_WINDOW)
            self._trackbar_created = False
