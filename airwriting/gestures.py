"""Hand gestures from the 21 MediaPipe hand landmarks.

    index finger up                -> DRAW   (pen down)
    index + middle finger up       -> HOVER  (pen up: move, or press toolbar buttons)
    all four fingers up (open palm) -> PALM  (hold to clear the canvas)
    no fingers up (fist)           -> FIST   (hold to recognise the character now)

The thumb is ignored on purpose: people stick it out in different ways.
A finger counts as "up" when it is straight and its tip is further from the
wrist than its middle joint. Using 3-D landmarks keeps this working when the
finger points at the camera. Hysteresis + a few frames of agreement stop the
gesture from flickering, which would otherwise break strokes.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

WRIST = 0
INDEX_TIP = 8
FINGERS = {  # name: (MCP, PIP, DIP, TIP)
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class Gesture(str, Enum):
    NONE = "none"      # no hand visible
    DRAW = "draw"
    HOVER = "hover"
    PALM = "palm"
    FIST = "fist"
    OTHER = "other"


def finger_metrics(lm: np.ndarray, finger: str) -> tuple[float, float]:
    """(straightness cosine, tip-distance / pip-distance from the wrist) for one finger."""
    mcp, pip, dip, tip = (lm[i] for i in FINGERS[finger])
    v1, v2 = pip - mcp, tip - pip
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    wrist = lm[WRIST]
    ratio = float(np.linalg.norm(tip - wrist) / (np.linalg.norm(pip - wrist) + 1e-6))
    return cos, ratio


def classify(up: dict[str, bool]) -> Gesture:
    i, m, r, p = up["index"], up["middle"], up["ring"], up["pinky"]
    if i and not m and not r and not p:
        return Gesture.DRAW
    if i and m and not r and not p:
        return Gesture.HOVER
    if i and m and r and p:
        return Gesture.PALM
    if not (i or m or r or p):
        return Gesture.FIST
    return Gesture.OTHER


class GestureDetector:
    UP_COS, UP_RATIO = 0.55, 1.12        # thresholds to become "up"
    DOWN_COS, DOWN_RATIO = 0.25, 1.0     # thresholds to become "down"

    def __init__(self, stable_frames: int = 3):
        self.stable_frames = max(1, stable_frames)
        self.reset()

    def reset(self):
        self.up = {f: False for f in FINGERS}
        self.stable = Gesture.NONE
        self.candidate = Gesture.NONE
        self.count = 0

    def fingers_up(self, lm: np.ndarray) -> dict[str, bool]:
        for f in FINGERS:
            cos, ratio = finger_metrics(lm, f)
            if self.up[f]:
                if cos < self.DOWN_COS or ratio < self.DOWN_RATIO:
                    self.up[f] = False
            elif cos > self.UP_COS and ratio > self.UP_RATIO:
                self.up[f] = True
        return dict(self.up)

    def update(self, lm: np.ndarray | None) -> Gesture:
        """Feed one frame of landmarks (21x3, pixels) or None; returns the stable gesture."""
        if lm is None:
            raw = Gesture.NONE
            self.up = {f: False for f in FINGERS}
        else:
            raw = classify(self.fingers_up(lm))
        if raw == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = raw, 1
        # losing the hand ends things immediately; other changes need agreement
        if raw == Gesture.NONE or self.count >= self.stable_frames:
            self.stable = self.candidate
        return self.stable
