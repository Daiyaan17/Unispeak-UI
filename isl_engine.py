"""
ISL Recognition Engine — adapted for MediaPipe Tasks API (>=0.10).

Pipeline:
  Webcam → MediaPipe HandLandmarker + optional pose → 150-dim features
  → Sliding window (64 frames) → Transformer classifier → ISL sign label
  → Token buffer → SentenceComposer (Claude API) → Natural English

Works with OpenHands (AI4Bharat) checkpoints or in mock mode for demo.
"""

import os
import cv2
import numpy as np
import time
import threading
from collections import deque
from typing import Optional, Callable

# ── PyTorch (optional — mock mode if unavailable) ────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── MediaPipe Tasks API ─────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# ── Anthropic (optional) ────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HAND_MODEL = os.path.join(SCRIPT_DIR, "assets", "hand_landmarker.task")

# Hand connection indices for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(1,5),
]
FINGER_TIPS = [4, 8, 12, 16, 20]


# ─────────────────────────────────────────────
# 1. KEYPOINT EXTRACTOR (Tasks API)
# ─────────────────────────────────────────────

class KeypointExtractor:
    """
    Extracts feature vector per frame using MediaPipe Tasks API.
    With two hands: 21 landmarks × 3 coords × 2 = 126 features.
    Padded to 150 with zeros (pose placeholder) for model compatibility.
    """

    def __init__(self, model_path=None):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not installed")

        model = model_path or DEFAULT_HAND_MODEL
        if not os.path.isfile(model):
            raise FileNotFoundError(f"Hand model not found: {model}")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns a 150-dim numpy array for the given BGR frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        hand_data = {"Left": np.zeros(63), "Right": np.zeros(63)}
        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"
                coords = []
                for lm in landmarks:
                    coords.extend([lm.x, lm.y, lm.z])
                hand_data[label] = np.array(coords[:63])

        features = np.concatenate([
            hand_data["Left"],   # 63
            hand_data["Right"],  # 63
            np.zeros(24),        # pose placeholder (24)
        ])
        return features.astype(np.float32)  # (150,)

    def extract_landmarks_for_drawing(self, frame_bgr):
        """Return raw landmarks + handedness for overlay drawing."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect(mp_image)

    def close(self):
        self.landmarker.close()


# ─────────────────────────────────────────────
# 2. TRANSFORMER MODEL (OpenHands-compatible)
# ─────────────────────────────────────────────

if TORCH_AVAILABLE:
    class ISLTransformerModel(nn.Module):
        def __init__(self, input_dim=150, num_classes=263, seq_len=64,
                     d_model=256, nhead=4, num_layers=3, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=512,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.input_proj(x)
            x = x + self.pos_embedding[:, :x.size(1), :]
            x = self.dropout(x)
            x = self.transformer(x)
            x = self.norm(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    def load_openhands_model(checkpoint_path, num_classes=263, device="cpu"):
        model = ISLTransformerModel(num_classes=num_classes)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model


# ─────────────────────────────────────────────
# 3. ISL LABEL MAP
# ─────────────────────────────────────────────

ISL_LABELS_50 = [
    "After", "Again", "Against", "Age", "All", "Asl", "Before", "Boy",
    "Can", "Cannot", "Come", "Cry", "Deaf", "Different", "Dont-Like",
    "Drink", "Eat", "Finish", "From", "Go", "Good", "Happy", "Have",
    "He", "Hello", "Help", "Home", "How", "Hurt", "I", "Know", "Later",
    "Learn", "Like", "Make", "Maybe", "More", "Mother", "My", "Name",
    "No", "None", "Not", "Now", "Please", "Same", "See", "She", "Sign",
    "Slow", "Sorry", "Stop", "Study", "Tell", "Thank-You", "Think",
    "Time", "Understand", "Want", "Water", "We", "What", "Where",
    "Who", "Why", "With", "Work", "Write", "Yes", "You", "Your"
]

def get_label_map(label_list=None):
    labels = label_list or ISL_LABELS_50
    return {i: lbl for i, lbl in enumerate(labels)}


# ─────────────────────────────────────────────
# 4. MOCK MODEL (demo without checkpoint)
# ─────────────────────────────────────────────

class MockISLModel:
    """Fires random ISL signs for UI testing without a real checkpoint."""
    import random as _rng

    def __init__(self, labels=None):
        self._labels = labels or ISL_LABELS_50
        self._frame_count = 0
        self._fire_every = 90  # ~3 seconds at 30fps

    def __call__(self, x):
        import torch as _t
        logits = _t.zeros(1, len(self._labels))
        self._frame_count += 1
        if self._frame_count >= self._fire_every:
            self._frame_count = 0
            logits[0, self._rng.randint(0, len(self._labels) - 1)] = 10.0
        return logits

    def eval(self): return self
    def to(self, d): return self


# ─────────────────────────────────────────────
# 5. SIGN RECOGNIZER (sliding window)
# ─────────────────────────────────────────────

class SignRecognizer:
    def __init__(self, model, label_map, seq_len=64,
                 confidence_threshold=0.65,
                 on_sign_detected=None, device="cpu"):
        self.model = model
        self.label_map = label_map
        self.seq_len = seq_len
        self.threshold = confidence_threshold
        self.on_sign_detected = on_sign_detected
        self.device = device

        self.frame_buffer = deque(maxlen=seq_len)
        self._last_sign = None
        self._last_sign_time = 0
        self._debounce_sec = 1.5

        # Public state
        self.current_sign = ""
        self.current_confidence = 0.0

    def push_frame(self, keypoints):
        self.frame_buffer.append(keypoints)
        if len(self.frame_buffer) == self.seq_len:
            self._run_inference()

    def _run_inference(self):
        if not TORCH_AVAILABLE:
            return
        import torch as _t
        seq = np.stack(list(self.frame_buffer), axis=0)
        tensor = _t.from_numpy(seq).unsqueeze(0).to(self.device)

        with _t.no_grad():
            logits = self.model(tensor)
            probs = _t.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            conf, idx = conf.item(), idx.item()

        self.current_confidence = conf
        label = self.label_map.get(idx, "Unknown")
        self.current_sign = label if conf >= self.threshold else ""

        if conf < self.threshold:
            return

        now = time.time()
        if label == self._last_sign and (now - self._last_sign_time) < self._debounce_sec:
            return

        self._last_sign = label
        self._last_sign_time = now
        if self.on_sign_detected:
            self.on_sign_detected(label, round(conf * 100, 1))


# ─────────────────────────────────────────────
# 6. SENTENCE COMPOSER (Claude NLP)
# ─────────────────────────────────────────────

class SentenceComposer:
    SYSTEM_PROMPT = (
        "You receive a sequence of Indian Sign Language (ISL) sign tokens "
        "detected by a computer vision system. Convert them into natural, "
        "conversational English as if a person typed it in a chat message. "
        "Keep it concise, warm, and natural. Output ONLY the final sentence."
    )

    def __init__(self, api_key="", pause_threshold=2.5,
                 on_sentence_ready=None):
        self.api_key = api_key
        self.pause_threshold = pause_threshold
        self.on_sentence_ready = on_sentence_ready

        self.token_buffer = []
        self._flush_timer = None
        self._lock = threading.Lock()

        self._has_api = bool(api_key and api_key != "YOUR_API_KEY_HERE"
                            and ANTHROPIC_AVAILABLE)
        if self._has_api:
            self.client = anthropic.Anthropic(api_key=api_key)

    def add_sign(self, label):
        with self._lock:
            self.token_buffer.append(label.upper())
            self._reset_flush_timer()

    def _reset_flush_timer(self):
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(self.pause_threshold, self._flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def flush_now(self):
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush()

    def _flush(self):
        with self._lock:
            if not self.token_buffer:
                return
            tokens = list(self.token_buffer)
            self.token_buffer.clear()

        token_str = " ".join(tokens)

        if self._has_api:
            try:
                resp = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": token_str}],
                )
                sentence = resp.content[0].text.strip()
            except Exception as e:
                print(f"[Composer] API error: {e}")
                sentence = token_str.title()
        else:
            # No API — just join tokens naturally
            sentence = " ".join(t.capitalize() for t in tokens)

        if self.on_sentence_ready:
            self.on_sentence_ready(sentence)

    def clear(self):
        with self._lock:
            self.token_buffer.clear()
        if self._flush_timer:
            self._flush_timer.cancel()

    def get_pending_tokens(self):
        with self._lock:
            return list(self.token_buffer)


# ─────────────────────────────────────────────
# 7. DRAWING UTILITIES
# ─────────────────────────────────────────────

def draw_hand_landmarks(frame, result, w, h):
    """Draw hand landmarks from MediaPipe result onto frame (RGB)."""
    if not result.hand_landmarks:
        return frame

    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

        for s, e in HAND_CONNECTIONS:
            if s < len(pts) and e < len(pts):
                cv2.line(frame, pts[s], pts[e], (56, 189, 248), 2, cv2.LINE_AA)

        for i, (px, py) in enumerate(pts):
            if i in FINGER_TIPS:
                cv2.circle(frame, (px, py), 5, (0, 255, 160), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 3, (56, 189, 248), -1, cv2.LINE_AA)

    return frame
