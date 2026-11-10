"""MediaPipe Hand Landmarker wrapper (MediaPipe Tasks API).

Recent MediaPipe releases (0.10.30+) removed the old `mp.solutions.hands`
API that most tutorials use, so this project uses the Tasks API, which works
on every 0.10.x version. The model file (hand_landmarker.task, ~7.5 MB) is
downloaded automatically the first time the app runs.
"""
from __future__ import annotations

import os
import shutil
import ssl
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import HAND_MODEL_PATH, HAND_MODEL_URL


@dataclass
class Hand:
    landmarks: np.ndarray   # (21, 3) pixel x, pixel y, depth (same scale as x)
    handedness: str = ""
    score: float = 1.0

    @property
    def index_tip(self) -> np.ndarray:
        return self.landmarks[8, :2]


def _ssl_context():
    try:
        import certifi  # often present; fixes "certificate verify failed" on macOS Python
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def ensure_hand_model(path: Path = HAND_MODEL_PATH, url: str = HAND_MODEL_URL) -> Path:
    """Return the path of the hand model, downloading it once if needed."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the MediaPipe hand model (~7.5 MB) to {path} ...")
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".part")
    tmp.close()
    try:
        with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as resp, open(tmp.name, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done / total * 100:5.1f}%")
                    sys.stdout.flush()
        print()
        shutil.move(tmp.name, path)
    except Exception as exc:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise RuntimeError(
            "Could not download the hand model automatically "
            f"({exc}).\nDownload it in your browser from:\n  {url}\n"
            f"and save it as:\n  {path}") from exc
    return path


class HandTracker:
    """Finds one hand per frame and returns its landmarks in pixel coordinates."""

    def __init__(self, model_path: Path | str | None = None, num_hands: int = 1,
                 min_detection_confidence: float = 0.6, min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = Path(model_path) if model_path else ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self._t0 = time.monotonic()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> Hand | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = int(timestamp_ms) if timestamp_ms is not None else int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)        # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
        label, score = "", 1.0
        if result.handedness and result.handedness[0]:
            label = result.handedness[0][0].category_name
            score = float(result.handedness[0][0].score)
        return Hand(pts, label, score)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
