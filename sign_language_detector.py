"""
Sign Language Detector — MediaPipe Hand Landmarker (Tasks API) + Rule-based
Gesture Recognition.

Uses MediaPipe's HandLandmarker (Tasks API, >=0.10) to extract 21 hand
landmarks per frame, then classifies gestures using geometric rules
(finger extension, angles, distances).

Recognises ~13 common gestures out-of-the-box without any trained model.
"""

import os
import math
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarkerResult,
        HandLandmarksConnections,
        RunningMode,
    )
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# ─── Default model path ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "assets", "hand_landmarker.task")

# ─── Hand connection list for drawing ───────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle  (0→9 via 5→9 palm)
    (0, 13), (13, 14), (14, 15), (15, 16), # ring    (0→13 via 9→13)
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm connections
    (1, 5),                                 # palm base
]

# ─── Landmark indices ───────────────────────────────────────────────
WRIST = 0
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]


class SignLanguageDetector:
    """Real-time sign-language / gesture detector using MediaPipe Hands."""

    def __init__(self, max_hands=1, min_detection_confidence=0.7,
                 min_tracking_confidence=0.6, model_path=None):
        if not MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )

        model = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(model):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model}\n"
                f"Download from: https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        # Create HandLandmarker with IMAGE mode (synchronous)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # ── Stability buffer ─────────────────────────────────────────
        self._buffer = []
        self._buffer_size = 12          # frames to accumulate
        self._confirmed_text = ""       # last confirmed gesture
        self._sentence = ""             # accumulated sentence
        self._last_confirmed = ""

        # Detection state
        self.hand_detected = False
        self.confidence = 0.0
        self.raw_prediction = ""

    # ─── Public API ──────────────────────────────────────────────────
    def process_frame(self, frame_rgb):
        """Process one RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray  — RGB image (H×W×3, uint8)

        Returns
        -------
        annotated : np.ndarray  — frame with landmarks drawn
        prediction : str        — current accumulated sentence
        """
        h, w, _ = frame_rgb.shape
        annotated = frame_rgb.copy()

        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            self.hand_detected = False
            self.confidence = 0.0
            self.raw_prediction = ""
            self._draw_status(annotated, "No hand detected", (120, 120, 140))
            return annotated, self._sentence

        self.hand_detected = True

        for hand_landmarks in result.hand_landmarks:
            # Convert landmarks to numpy array
            lm = self._extract_landmarks(hand_landmarks)

            # Draw landmarks on frame
            self._draw_hand(annotated, hand_landmarks, w, h)

            # Classify gesture
            gesture, conf = self._classify_rule_based(lm)
            self.confidence = conf
            self.raw_prediction = gesture

            # Push to stability buffer
            self._buffer.append(gesture)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

            stable = self._get_stable_prediction()
            if stable:
                self._confirmed_text = stable
                if stable != self._last_confirmed:
                    if stable in ("Space", " "):
                        self._sentence += " "
                    elif stable == "Backspace":
                        self._sentence = self._sentence[:-1]
                    elif stable == "Clear":
                        self._sentence = ""
                    elif len(stable) <= 3:
                        # Single letter / short symbol
                        self._sentence += stable
                    else:
                        # Word-level gesture (e.g. "Hello")
                        if self._sentence and not self._sentence.endswith(" "):
                            self._sentence += " "
                        self._sentence += stable
                    self._last_confirmed = stable

            # Draw prediction overlay
            self._draw_prediction(annotated, gesture, conf, w, h)

        return annotated, self._sentence

    def get_sentence(self):
        """Return the accumulated sentence so far."""
        return self._sentence

    def clear_sentence(self):
        """Reset the sentence buffer."""
        self._sentence = ""
        self._last_confirmed = ""
        self._buffer.clear()

    def release(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    # ─── Landmark extraction ─────────────────────────────────────────
    @staticmethod
    def _extract_landmarks(hand_landmarks):
        """Return (21, 3) numpy array of normalised landmarks."""
        pts = []
        for lm in hand_landmarks:
            pts.append([lm.x, lm.y, lm.z])
        pts = np.array(pts, dtype=np.float32)

        # Normalise relative to wrist
        wrist = pts[WRIST].copy()
        pts -= wrist

        # Scale so max dist from wrist = 1
        max_dist = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
        pts /= max_dist
        return pts

    # ─── Rule-based classifier ───────────────────────────────────────
    def _classify_rule_based(self, lm):
        """Classify gesture using geometric heuristics on the 21 landmarks.

        Returns (gesture_name, confidence 0-1).
        """
        fingers_up = self._fingers_extended(lm)
        thumb_up = fingers_up[0]
        num_fingers = sum(fingers_up)

        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]
        pinky_tip = lm[20]

        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)

        # ── Gesture rules ────────────────────────────────────────

        # 1. FIST — all fingers closed
        if num_fingers == 0:
            return "Fist ✊", 0.90

        # 2. THUMBS UP — only thumb extended, thumb pointing up
        if thumb_up and num_fingers == 1:
            if lm[4][1] < lm[3][1] and lm[4][1] < lm[2][1]:
                return "Thumbs Up 👍", 0.92

        # 3. THUMBS DOWN — only thumb extended, thumb pointing down
        if thumb_up and num_fingers == 1:
            if lm[4][1] > lm[3][1] and lm[4][1] > lm[2][1]:
                return "Thumbs Down 👎", 0.88

        # 4. POINTING UP — only index extended
        if fingers_up == [False, True, False, False, False]:
            if lm[8][1] < lm[6][1]:
                return "Point Up ☝️", 0.90

        # 5. PEACE / VICTORY — index + middle extended
        if fingers_up == [False, True, True, False, False]:
            spread = np.linalg.norm(index_tip - middle_tip)
            if spread > 0.15:
                return "Peace ✌️", 0.91
            else:
                return "V Sign", 0.85

        # 6. OK SIGN — thumb + index form circle, others extended
        if thumb_index_dist < 0.12 and fingers_up[2] and fingers_up[3] and fingers_up[4]:
            return "OK 👌", 0.89

        # 7. OPEN PALM / STOP — all fingers extended, spread
        if num_fingers == 5:
            spread = np.linalg.norm(index_tip - pinky_tip)
            if spread > 0.4:
                return "Hello / Stop ✋", 0.93
            else:
                return "Open Hand", 0.80

        # 8. I LOVE YOU (ILY) — thumb + index + pinky extended
        if fingers_up == [True, True, False, False, True]:
            return "I Love You 🤟", 0.92

        # 9. ROCK ON / HORNS — index + pinky extended
        if fingers_up == [False, True, False, False, True]:
            return "Rock On 🤘", 0.87

        # 10. THREE — index + middle + ring extended
        if fingers_up == [False, True, True, True, False]:
            return "Three", 0.85

        # 11. FOUR — all except thumb
        if fingers_up == [False, True, True, True, True]:
            return "Four", 0.85

        # 12. CALL ME — thumb + pinky extended
        if fingers_up == [True, False, False, False, True]:
            return "Call Me 🤙", 0.88

        # 13. GUN / L — thumb + index extended
        if fingers_up == [True, True, False, False, False]:
            return "L / Gun", 0.82

        # Fallback — count fingers
        if num_fingers == 1 and not thumb_up:
            for i, up in enumerate(fingers_up):
                if up:
                    names = ["Thumb", "Point", "Middle", "Ring", "Pinky"]
                    return names[i], 0.75

        return f"{num_fingers} Fingers", 0.6

    def _fingers_extended(self, lm):
        """Return [thumb, index, middle, ring, pinky] booleans."""
        extended = []

        # Thumb — tip farther from palm center than MCP
        palm_center = (lm[0] + lm[5] + lm[17]) / 3
        thumb_tip_dist = np.linalg.norm(lm[4] - palm_center)
        thumb_mcp_dist = np.linalg.norm(lm[2] - palm_center)
        extended.append(thumb_tip_dist > thumb_mcp_dist * 1.1)

        # Other 4 fingers — tip above PIP joint (in y, lower y = higher)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            extended.append(lm[tip][1] < lm[pip][1])

        return extended

    # ─── Stability buffer ────────────────────────────────────────────
    def _get_stable_prediction(self):
        """Return a prediction only if consistent over the buffer."""
        if len(self._buffer) < self._buffer_size:
            return None

        counts = {}
        for g in self._buffer:
            if g:
                counts[g] = counts.get(g, 0) + 1

        if not counts:
            return None

        best = max(counts, key=counts.get)
        ratio = counts[best] / len(self._buffer)

        if ratio >= 0.7:
            return best
        return None

    # ─── Drawing utilities ───────────────────────────────────────────
    def _draw_hand(self, frame, hand_landmarks, w, h):
        """Draw hand skeleton + landmarks on frame."""
        # Convert normalised landmarks to pixel coords
        pts = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append((px, py))

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(frame, pts[start_idx], pts[end_idx],
                         (56, 189, 248), 2, cv2.LINE_AA)

        # Draw landmark dots
        for i, (px, py) in enumerate(pts):
            # Fingertips get a larger dot
            if i in FINGER_TIPS:
                cv2.circle(frame, (px, py), 6, (0, 255, 160), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 3, (56, 189, 248), -1, cv2.LINE_AA)

        # Draw bounding box with corner accents
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        x_min = max(0, min(x_coords) - 20)
        y_min = max(0, min(y_coords) - 20)
        x_max = min(w, max(x_coords) + 20)
        y_max = min(h, max(y_coords) + 20)

        corner_len = 20
        color = (56, 189, 248)
        thickness = 2

        # Top-left
        cv2.line(frame, (x_min, y_min), (x_min + corner_len, y_min), color, thickness)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x_max, y_min), (x_max - corner_len, y_min), color, thickness)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x_min, y_max), (x_min + corner_len, y_max), color, thickness)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x_max, y_max), (x_max - corner_len, y_max), color, thickness)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_len), color, thickness)

    def _draw_prediction(self, frame, gesture, confidence, w, h):
        """Draw prediction label + confidence bar on frame."""
        if not gesture:
            return

        # Semi-transparent background bar
        overlay = frame.copy()
        bar_h = 56
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Gesture text
        cv2.putText(frame, gesture, (16, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (56, 189, 248), 2, cv2.LINE_AA)

        # Confidence bar
        conf_pct = int(confidence * 100)
        bar_x = w - 180
        bar_w = 150
        bar_y = h - 38
        bar_height = 14

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_height),
                      (40, 40, 60), -1)
        fill_w = int(bar_w * confidence)
        bar_color = (56, 189, 248) if confidence > 0.7 else \
                    (0, 165, 255) if confidence > 0.5 else (0, 0, 220)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_height),
                      bar_color, -1)
        cv2.putText(frame, f"{conf_pct}%", (bar_x + bar_w + 6, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_status(frame, text, color):
        """Draw a status text at bottom of frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, text, (16, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
