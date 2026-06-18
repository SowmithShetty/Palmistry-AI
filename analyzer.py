"""
analyzer.py — Palmistry heuristic engine.

Milestone 4: The Palmistry Engine
----------------------------------
This module analyses the Canny edge output from the palm ROI and maps
heuristic measurements to palmistry traits defined in rules.json.

Strategy:
  The palm ROI is divided into three horizontal zones, each corresponding
  to one of the three major palm lines in traditional palmistry:

    ┌──────────────────┐
    │   Top third       │  ← Heart Line zone (emotions, relationships)
    ├──────────────────┤
    │   Middle third    │  ← Head Line zone (intellect, decision-making)
    ├──────────────────┤
    │   Bottom third    │  ← Life Line zone (vitality, physical well-being)
    └──────────────────┘

  For each zone we measure:
    1. **Edge density** — the ratio of white (edge) pixels to total pixels.
       Higher density means more prominent, well-defined lines.
    2. **Edge extent** — how much of the zone's *width* the edges span.
       This approximates line length (long vs short).

  These two metrics are then classified into categories (long/medium/short,
  strong/faint) and looked up in rules.json to produce human-readable
  trait descriptions.

Design decisions:
  - We use simple pixel-counting heuristics rather than contour tracing
    because the Canny output from a webcam is noisy and contour-based
    approaches are fragile.  Density is surprisingly robust.
  - Thresholds for "long"/"short" and "strong"/"faint" are tuned
    empirically — they can be adjusted without touching the code by
    editing rules.json.
  - The analyzer is stateless: each call to `analyze()` is independent.
    This avoids temporal artefacts (readings from previous frames
    bleeding into the current one).
  - We add a cooldown/debounce mechanism in the main loop (not here) to
    avoid the reading flickering every frame.
"""

import json
import os
import cv2
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# Load the palmistry rules once at import time
# ─────────────────────────────────────────────
_RULES_PATH = os.path.join(os.path.dirname(__file__), "rules.json")

with open(_RULES_PATH, "r", encoding="utf-8") as f:
    RULES: dict = json.load(f)


# ─────────────────────────────────────────────
# Classification thresholds (empirically tuned for landmark-masked zones)
# ─────────────────────────────────────────────

# Edge density: ratio of white pixels to total mask pixels in a zone.
DENSITY_STRONG = 0.022   # >2.2% of mask pixels are edges
DENSITY_FAINT = 0.007    # <0.7% of mask pixels are edges

# Edge extent/length score: longest contour length / reference distance.
EXTENT_LONG = 0.55       # longest line length / reference distance >= 55%
EXTENT_SHORT = 0.25      # longest line length / reference distance <= 25%

# Overall palm classification: total edge density across the full palm mask.
OVERALL_MANY = 0.035     # Abundant edges in the palm area
OVERALL_CLEAR = 0.012    # Clear primary lines


class PalmistryReading:
    """
    Container for a single palmistry reading.

    Attributes
    ----------
    heart : dict   — {"length": str, "strength": str, "trait": str, "detail": str}
    head  : dict   — same structure
    life  : dict   — same structure
    overall : dict — {"category": str, "trait": str, "detail": str}
    """

    def __init__(self, heart: dict, head: dict, life: dict, overall: dict):
        self.heart = heart
        self.head = head
        self.life = life
        self.overall = overall

    def summary_lines(self) -> list[str]:
        """
        Return a list of short one-line strings suitable for cv2.putText
        overlay on the main feed.
        """
        lines = []
        lines.append(f"Heart: {self.heart['trait']}")
        lines.append(f"  -> {self.heart['detail']}")
        lines.append(f"Head:  {self.head['trait']}")
        lines.append(f"  -> {self.head['detail']}")
        lines.append(f"Life:  {self.life['trait']}")
        lines.append(f"  -> {self.life['detail']}")
        lines.append(f"Overall: {self.overall['trait']}")
        return lines


def _shrink_polygon(poly: np.ndarray, factor: float = 0.85) -> np.ndarray:
    """
    Shrink the vertices of a polygon towards its centroid.
    This helps pull the region boundaries away from outer skin boundaries.
    """
    centroid = poly.mean(axis=0)
    return centroid + factor * (poly - centroid)


def _classify_zone(
    edges: np.ndarray,
    polygon_points: np.ndarray,
    ref_dist: float,
    roi_img: np.ndarray = None,
    overlay_color: tuple[int, int, int] = (0, 255, 0),
    line_name: str = ""
) -> tuple[str, str, float, float]:
    """
    Analyse a specific region of the palm defined by the landmark polygon.
    
    We create a mask from the polygon, calculate the edge density inside it,
    and find the longest contour (primary line) to estimate line length.
    """
    h, w = edges.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points.astype(np.int32)], 255)
    
    zone_edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    mask_area = np.count_nonzero(mask)
    if mask_area == 0:
        return "short", "faint", 0.0, 0.0
        
    edge_count = np.count_nonzero(zone_edges)
    density = edge_count / mask_area
    
    # Extract contours to trace lines
    contours, _ = cv2.findContours(zone_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours: we only want reasonably long continuous segments, not small noise dots
    min_contour_len = 15.0
    valid_contours = []
    for c in contours:
        length = cv2.arcLength(c, closed=False)
        if length >= min_contour_len:
            valid_contours.append((c, length))
            
    if not valid_contours:
        longest_len = 0.0
    else:
        # Sort by length descending
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        longest_c, longest_len = valid_contours[0]
        
        # Draw the primary detected line contour in yellow on the ROI image
        if roi_img is not None:
            cv2.drawContours(roi_img, [longest_c], -1, (0, 255, 255), 2)
            
    # Draw zone overlay
    if roi_img is not None:
        cv2.polylines(roi_img, [polygon_points.astype(np.int32)], isClosed=True, color=overlay_color, thickness=1)
        overlay = roi_img.copy()
        cv2.fillPoly(overlay, [polygon_points.astype(np.int32)], overlay_color)
        cv2.addWeighted(overlay, 0.15, roi_img, 0.85, 0, roi_img)
        
        # Label the zone near its centroid
        cx, cy = polygon_points.mean(axis=0).astype(int)
        cv2.putText(roi_img, line_name, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    # Scale-invariant length score: length relative to hand reference distance
    length_score = longest_len / ref_dist if ref_dist > 0 else 0.0
    
    # Classify length
    if length_score >= EXTENT_LONG:
        length_class = "long"
    elif length_score <= EXTENT_SHORT:
        length_class = "short"
    else:
        length_class = "medium"
        
    # Classify strength
    if density >= DENSITY_STRONG:
        strength_class = "strong"
    elif density <= DENSITY_FAINT:
        strength_class = "faint"
    else:
        strength_class = "medium"
        
    return length_class, strength_class, density, length_score


def _lookup_trait(line_key: str, length: str, strength: str) -> dict:
    """
    Look up the palmistry trait for a given line from rules.json.

    We prefer the length-based trait (long/medium/short) as the primary
    reading, and fall back to strength-based (strong/faint) if the length
    is "medium" (i.e., unremarkable).
    """
    line_rules = RULES.get(line_key, {})

    # Primary: use the length classification.
    if length != "medium" and length in line_rules:
        entry = line_rules[length]
        return {
            "length": length,
            "strength": strength,
            "trait": entry.get("trait", "Unknown"),
            "detail": entry.get("detail", ""),
        }

    # Fallback: use the strength classification if length is medium.
    if strength != "medium" and strength in line_rules:
        entry = line_rules[strength]
        return {
            "length": length,
            "strength": strength,
            "trait": entry.get("trait", "Unknown"),
            "detail": entry.get("detail", ""),
        }

    # Double fallback: length is medium, strength is medium → use "medium".
    if "medium" in line_rules:
        entry = line_rules["medium"]
        return {
            "length": length,
            "strength": strength,
            "trait": entry.get("trait", "Balanced"),
            "detail": entry.get("detail", ""),
        }

    return {
        "length": length,
        "strength": strength,
        "trait": "Balanced",
        "detail": "Your traits are well-balanced.",
    }


def _classify_overall(edges: np.ndarray, palm_mask: np.ndarray) -> dict:
    """Classify the overall palm based on total edge density inside the palm mask."""
    mask_area = np.count_nonzero(palm_mask)
    if mask_area == 0:
        return RULES.get("overall", {}).get("faint_lines", {})

    masked_edges = cv2.bitwise_and(edges, edges, mask=palm_mask)
    density = np.count_nonzero(masked_edges) / mask_area

    if density >= OVERALL_MANY:
        key = "many_lines"
    elif density >= OVERALL_CLEAR:
        key = "clear_lines"
    else:
        key = "faint_lines"

    entry = RULES.get("overall", {}).get(key, {})
    return {
        "category": key,
        "trait": entry.get("trait", "Unknown"),
        "detail": entry.get("detail", ""),
    }


def analyze(
    edges: np.ndarray,
    landmarks_roi: np.ndarray,
    roi_img: np.ndarray = None
) -> Optional[PalmistryReading]:
    """
    Perform a full palmistry analysis on a Canny edge map of the palm ROI.
    
    Uses landmark-aligned zones and contour tracing for high accuracy.
    Optionally draws shaded overlays and detected lines on the BGR roi_img.
    """
    h, w = edges.shape[:2]
    if h < 30 or w < 30 or landmarks_roi is None or len(landmarks_roi) < 21:
        return None

    # Retrieve key landmarks (x, y) relative to the ROI
    L0 = landmarks_roi[0]   # Wrist
    L1 = landmarks_roi[1]   # Thumb CMC
    L2 = landmarks_roi[2]   # Thumb MCP
    L5 = landmarks_roi[5]   # Index knuckle
    L9 = landmarks_roi[9]   # Middle knuckle
    L13 = landmarks_roi[13] # Ring knuckle
    L17 = landmarks_roi[17] # Pinky knuckle

    # Define reference scale distances
    ref_heart = np.linalg.norm(L17 - L5)
    ref_head = np.linalg.norm(L17 - L5)
    ref_life = np.linalg.norm(L5 - L0)

    # 1. Heart Line Region (upper palm strip under knuckles)
    poly_heart = np.array([
        L17, L13, L9, L5,
        L5 + 0.3 * (L0 - L5),
        L17 + 0.35 * (L0 - L17)
    ])
    poly_heart_shrunk = _shrink_polygon(poly_heart, 0.85)

    # 2. Head Line Region (middle palm strip)
    poly_head = np.array([
        L5 + 0.2 * (L0 - L5),
        L9 + 0.2 * (L0 - L9),
        L17 + 0.3 * (L0 - L17),
        L17 + 0.6 * (L0 - L17),
        L5 + 0.5 * (L0 - L5)
    ])
    poly_head_shrunk = _shrink_polygon(poly_head, 0.85)

    # 3. Life Line Region (curves around the thumb eminence)
    poly_life = np.array([
        L5 + 0.25 * (L0 - L5),
        L9 + 0.4 * (L0 - L9),
        L0,
        L1,
        L2
    ])
    poly_life_shrunk = _shrink_polygon(poly_life, 0.85)

    # 4. Overall Palm Region (for overall complexity/density analysis)
    poly_overall = np.array([L0, L1, L2, L5, L9, L13, L17])
    poly_overall_shrunk = _shrink_polygon(poly_overall, 0.85)

    # Classify each line zone
    # Heart line: Red overlay
    heart_len, heart_str, _, _ = _classify_zone(
        edges, poly_heart_shrunk, ref_heart, roi_img, (0, 0, 255), "Heart Line"
    )
    # Head line: Blue overlay
    head_len, head_str, _, _ = _classify_zone(
        edges, poly_head_shrunk, ref_head, roi_img, (255, 0, 0), "Head Line"
    )
    # Life line: Green overlay
    life_len, life_str, _, _ = _classify_zone(
        edges, poly_life_shrunk, ref_life, roi_img, (0, 255, 0), "Life Line"
    )

    # Construct the overall palm mask
    palm_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(palm_mask, [poly_overall_shrunk.astype(np.int32)], 255)

    # Look up traits in rules.json
    heart = _lookup_trait("heart_line", heart_len, heart_str)
    head = _lookup_trait("head_line", head_len, head_str)
    life = _lookup_trait("life_line", life_len, life_str)
    overall = _classify_overall(edges, palm_mask)

    return PalmistryReading(heart, head, life, overall)
