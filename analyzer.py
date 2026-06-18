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
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# Load the palmistry rules once at import time
# ─────────────────────────────────────────────
_RULES_PATH = os.path.join(os.path.dirname(__file__), "rules.json")

with open(_RULES_PATH, "r", encoding="utf-8") as f:
    RULES: dict = json.load(f)


# ─────────────────────────────────────────────
# Classification thresholds (empirically tuned)
# ─────────────────────────────────────────────

# Edge density: ratio of white pixels to total pixels in a zone.
# These thresholds classify how "strong" or "faint" the lines are.
DENSITY_STRONG = 0.06   # >6 % of pixels are edges → strong lines
DENSITY_FAINT = 0.02    # <2 % → faint lines
# (Between 2 % and 6 % is "normal" — we don't flag it either way.)

# Edge extent: fraction of the zone's width that edges span.
# This approximates the *length* of the line.
EXTENT_LONG = 0.65      # Edges span >65 % of the width → long line
EXTENT_SHORT = 0.35     # Edges span <35 % of the width → short line
# (Between 35 % and 65 % is "medium".)

# Overall palm classification: total edge density across the full ROI.
OVERALL_MANY = 0.08     # Lots of fine lines everywhere
OVERALL_CLEAR = 0.04    # Strong, clear primary lines
# Below OVERALL_CLEAR → faint lines overall.


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


def _classify_zone(
    edges: np.ndarray, y_start: int, y_end: int
) -> tuple[str, str, float, float]:
    """
    Analyse a horizontal slice of the edge map.

    Parameters
    ----------
    edges : np.ndarray
        The full Canny edge image (single-channel, 0/255).
    y_start, y_end : int
        The row range defining this zone.

    Returns
    -------
    (length_class, strength_class, density, extent)
        length_class  : "long" | "medium" | "short"
        strength_class: "strong" | "faint" | "medium"  (medium = normal)
        density       : raw density value (for debugging)
        extent        : raw extent value (for debugging)
    """
    zone = edges[y_start:y_end, :]
    h, w = zone.shape

    total_pixels = h * w
    if total_pixels == 0:
        return "short", "faint", 0.0, 0.0

    # Count edge pixels (Canny outputs 255 for edges, 0 for background).
    edge_count = np.count_nonzero(zone)
    density = edge_count / total_pixels

    # Calculate horizontal extent: find the leftmost and rightmost columns
    # that contain at least one edge pixel.
    col_has_edge = np.any(zone > 0, axis=0)  # boolean array of width w
    if not np.any(col_has_edge):
        # No edges at all in this zone.
        return "short", "faint", density, 0.0

    edge_cols = np.where(col_has_edge)[0]
    left = edge_cols[0]
    right = edge_cols[-1]
    extent = (right - left + 1) / w

    # Classify length based on extent.
    if extent >= EXTENT_LONG:
        length_class = "long"
    elif extent <= EXTENT_SHORT:
        length_class = "short"
    else:
        length_class = "medium"

    # Classify strength based on density.
    if density >= DENSITY_STRONG:
        strength_class = "strong"
    elif density <= DENSITY_FAINT:
        strength_class = "faint"
    else:
        strength_class = "medium"

    return length_class, strength_class, density, extent


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


def _classify_overall(edges: np.ndarray) -> dict:
    """Classify the overall palm based on total edge density."""
    total = edges.size
    if total == 0:
        return RULES.get("overall", {}).get("faint_lines", {})

    density = np.count_nonzero(edges) / total

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


def analyze(edges: np.ndarray) -> Optional[PalmistryReading]:
    """
    Perform a full palmistry analysis on a Canny edge map of the palm ROI.

    Parameters
    ----------
    edges : np.ndarray
        Single-channel binary edge image (from processor.process()).

    Returns
    -------
    PalmistryReading or None if the image is too small to analyse.
    """
    h, w = edges.shape[:2]
    if h < 30 or w < 30:
        return None

    # Divide into three equal horizontal zones.
    third = h // 3
    top_start, top_end = 0, third                    # Heart line
    mid_start, mid_end = third, 2 * third            # Head line
    bot_start, bot_end = 2 * third, h                # Life line

    # Classify each zone.
    heart_len, heart_str, _, _ = _classify_zone(edges, top_start, top_end)
    head_len, head_str, _, _ = _classify_zone(edges, mid_start, mid_end)
    life_len, life_str, _, _ = _classify_zone(edges, bot_start, bot_end)

    # Look up traits.
    heart = _lookup_trait("heart_line", heart_len, heart_str)
    head = _lookup_trait("head_line", head_len, head_str)
    life = _lookup_trait("life_line", life_len, life_str)
    overall = _classify_overall(edges)

    return PalmistryReading(heart, head, life, overall)
