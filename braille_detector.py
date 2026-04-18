"""
Braille Detector — OpenCV-based braille dot detection and decoding.

Pipeline:
  Frame → Grayscale → Blur → Adaptive Threshold → Contour Detection
  → Filter circular blobs → Cluster into 3×2 cells → Decode to text

Supports Grade-1 English Braille (letters, digits, common punctuation).
"""

import cv2
import numpy as np
from collections import defaultdict


# ─── Grade 1 Braille map ────────────────────────────────────────────
# Each braille cell has 6 dots arranged in a 3×2 grid:
#   1 4
#   2 5
#   3 6
# We represent a braille cell as a tuple of active dot indices (1-6).

BRAILLE_TO_CHAR = {
    (1,):           'a',
    (1, 2):         'b',
    (1, 4):         'c',
    (1, 4, 5):      'd',
    (1, 5):         'e',
    (1, 2, 4):      'f',
    (1, 2, 4, 5):   'g',
    (1, 2, 5):      'h',
    (2, 4):         'i',
    (2, 4, 5):      'j',
    (1, 3):         'k',
    (1, 2, 3):      'l',
    (1, 3, 4):      'm',
    (1, 3, 4, 5):   'n',
    (1, 3, 5):      'o',
    (1, 2, 3, 4):   'p',
    (1, 2, 3, 4, 5): 'q',
    (1, 2, 3, 5):   'r',
    (2, 3, 4):      's',
    (2, 3, 4, 5):   't',
    (1, 3, 6):      'u',
    (1, 2, 3, 6):   'v',
    (2, 4, 5, 6):   'w',
    (1, 3, 4, 6):   'x',
    (1, 3, 4, 5, 6): 'y',
    (1, 3, 5, 6):   'z',
    # Numbers (preceded by number indicator ⠼ = dots 3,4,5,6)
    (3, 4, 5, 6):   '#',  # number indicator
    # Space
    ():             ' ',
}

# Reverse map: also build Unicode braille for display
BRAILLE_UNICODE_BASE = 0x2800  # ⠀

def _dots_to_unicode(dots):
    """Convert a tuple of dot indices (1-6) to a Unicode braille character."""
    val = 0
    for d in dots:
        val |= (1 << (d - 1))
    return chr(BRAILLE_UNICODE_BASE + val)


class BrailleDetector:
    """Real-time braille detection from camera frames."""

    def __init__(self, min_dot_area=30, max_dot_area=2000,
                 circularity_thresh=0.55, adaptive_block=15,
                 adaptive_c=8):
        self.min_dot_area = min_dot_area
        self.max_dot_area = max_dot_area
        self.circularity_thresh = circularity_thresh
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c

        # State
        self.detected_text = ""
        self.dot_count = 0
        self.cell_count = 0
        self._buffer = []
        self._buffer_size = 8
        self._last_confirmed = ""
        self.confidence = 0.0

    # ─── Public API ──────────────────────────────────────────────────
    def process_frame(self, frame_rgb):
        """Process one RGB frame.

        Returns
        -------
        annotated : np.ndarray  — frame with detection overlays
        decoded_text : str      — decoded braille text (may be empty)
        """
        h, w = frame_rgb.shape[:2]
        annotated = frame_rgb.copy()

        # 1. Preprocess
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive threshold to find dots
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adaptive_block, self.adaptive_c
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 2. Find candidate dots
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        dot_centers = []
        dot_radii = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_dot_area or area > self.max_dot_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.circularity_thresh:
                continue

            # It's a dot!
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            radius = int(np.sqrt(area / np.pi))

            dot_centers.append((cx, cy))
            dot_radii.append(radius)

            # Draw detected dot
            cv2.circle(annotated, (cx, cy), radius + 3, (56, 189, 248), 2)
            cv2.circle(annotated, (cx, cy), 2, (0, 255, 100), -1)

        self.dot_count = len(dot_centers)

        if len(dot_centers) < 2:
            self._draw_status(annotated, "Scanning for braille dots...", (120, 120, 160))
            self.confidence = 0.0
            return annotated, ""

        # 3. Cluster dots into cells
        cells = self._cluster_into_cells(dot_centers, dot_radii)
        self.cell_count = len(cells)

        if not cells:
            self._draw_status(annotated, f"{self.dot_count} dots found, analyzing...",
                            (100, 180, 255))
            return annotated, ""

        # 4. Decode cells
        decoded = ""
        braille_display = ""

        for cell_info in cells:
            dots_pattern = cell_info["pattern"]
            char = BRAILLE_TO_CHAR.get(tuple(sorted(dots_pattern)), "?")
            unicode_br = _dots_to_unicode(dots_pattern) if dots_pattern else "⠀"
            decoded += char
            braille_display += unicode_br

            # Draw cell boundary
            bbox = cell_info["bbox"]
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 255, 100), 1)

            # Draw decoded char above cell
            cv2.putText(annotated, char.upper(), (bbox[0], bbox[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2, cv2.LINE_AA)

        # Stability buffer
        self._buffer.append(decoded)
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)

        stable = self._get_stable()
        if stable:
            self.detected_text = stable
            self.confidence = 0.85
        elif decoded:
            self.confidence = 0.5

        # Draw info bar
        self._draw_info(annotated, decoded, braille_display, w, h)

        return annotated, self.detected_text

    def get_text(self):
        """Return the last stable detected text."""
        return self.detected_text

    def clear(self):
        """Reset detection state."""
        self.detected_text = ""
        self._buffer.clear()
        self._last_confirmed = ""

    # ─── Dot clustering ──────────────────────────────────────────────
    def _cluster_into_cells(self, centers, radii):
        """Group dots into 3×2 braille cells based on spatial proximity.

        Uses a grid-based approach: find the median dot spacing, then
        build a grid with cell_width ≈ 2×spacing, cell_height ≈ 3×spacing.
        """
        if len(centers) < 2:
            return []

        centers = np.array(centers, dtype=np.float32)
        avg_radius = np.mean(radii) if radii else 5

        # Estimate spacing: median nearest-neighbor distance
        from scipy.spatial.distance import cdist
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        median_spacing = np.median(nn_dists)

        if median_spacing < 1:
            return []

        # Cell dimensions (a braille cell is 2 columns × 3 rows)
        col_spacing = median_spacing
        row_spacing = median_spacing
        cell_w = col_spacing * 2.5
        cell_h = row_spacing * 3.5

        # Sort centers left-to-right, then top-to-bottom
        sorted_idx = np.lexsort((centers[:, 1], centers[:, 0]))
        sorted_centers = centers[sorted_idx]

        # Group into columns first
        cells = []
        used = set()

        # Simple greedy clustering
        for i in range(len(sorted_centers)):
            if i in used:
                continue

            seed = sorted_centers[i]
            cell_dots = [(seed[0], seed[1])]
            used.add(i)

            # Find neighbors within cell bounds
            for j in range(len(sorted_centers)):
                if j in used:
                    continue
                other = sorted_centers[j]
                dx = abs(other[0] - seed[0])
                dy = abs(other[1] - seed[1])
                if dx < cell_w and dy < cell_h:
                    cell_dots.append((other[0], other[1]))
                    used.add(j)

                    if len(cell_dots) >= 6:
                        break

            if len(cell_dots) >= 1:
                pattern = self._dots_to_pattern(cell_dots, col_spacing, row_spacing)
                xs = [d[0] for d in cell_dots]
                ys = [d[1] for d in cell_dots]
                bbox = (int(min(xs) - 10), int(min(ys) - 10),
                        int(max(xs) + 10), int(max(ys) + 10))
                cells.append({"pattern": pattern, "bbox": bbox, "dots": cell_dots})

        return cells

    @staticmethod
    def _dots_to_pattern(cell_dots, col_spacing, row_spacing):
        """Map 2D dot positions within a cell to braille dot indices (1-6).

        Standard braille cell layout:
          1 4
          2 5
          3 6
        """
        if not cell_dots:
            return ()

        pts = np.array(cell_dots)
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2

        pattern = []
        for px, py in cell_dots:
            # Determine column (left=1,2,3 or right=4,5,6)
            col = 0 if px < cx else 1

            # Determine row (top=0, mid=1, bottom=2)
            rel_y = py - (pts[:, 1].min())
            range_y = pts[:, 1].max() - pts[:, 1].min()
            if range_y < row_spacing * 0.5:
                row = 0
            elif rel_y < range_y * 0.33:
                row = 0
            elif rel_y < range_y * 0.67:
                row = 1
            else:
                row = 2

            # Map to dot index: col*3 + row + 1
            dot_idx = col * 3 + row + 1
            if 1 <= dot_idx <= 6:
                pattern.append(dot_idx)

        return tuple(sorted(set(pattern)))

    # ─── Stability ───────────────────────────────────────────────────
    def _get_stable(self):
        """Return decoded text only if stable across buffer frames."""
        if len(self._buffer) < self._buffer_size // 2:
            return None

        counts = {}
        for t in self._buffer:
            if t:
                counts[t] = counts.get(t, 0) + 1

        if not counts:
            return None

        best = max(counts, key=counts.get)
        ratio = counts[best] / len(self._buffer)

        if ratio >= 0.5:
            return best
        return None

    # ─── Drawing ─────────────────────────────────────────────────────
    @staticmethod
    def _draw_status(frame, text, color):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 44), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (16, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_info(frame, decoded, braille_str, w, h):
        """Draw info bar at bottom of frame."""
        overlay = frame.copy()
        bar_h = 60
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Decoded text
        cv2.putText(frame, f"Decoded: {decoded.upper()}", (16, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2, cv2.LINE_AA)

        # Braille unicode display
        cv2.putText(frame, f"Braille: {braille_str}", (16, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 180), 1, cv2.LINE_AA)
