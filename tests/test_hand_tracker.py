"""HandTracker conversion logic with a fake MediaPipe landmarker (no model file needed)."""
from types import SimpleNamespace

import numpy as np
import pytest

from airwriting import hand_tracker
from airwriting.hand_tracker import HandTracker, ensure_hand_model


class FakeLandmarker:
    def __init__(self):
        self.timestamps = []

    def detect_for_video(self, image, ts):
        self.timestamps.append(ts)
        lms = [SimpleNamespace(x=0.5 + i * 0.01, y=0.25, z=-0.02) for i in range(21)]
        return SimpleNamespace(hand_landmarks=[lms],
                               handedness=[[SimpleNamespace(category_name="Right", score=0.97)]])

    def close(self):
        pass


def make_tracker():
    import mediapipe as mp
    t = HandTracker.__new__(HandTracker)
    t._mp, t._landmarker, t._last_ts, t._t0 = mp, FakeLandmarker(), -1, 0.0
    return t


def test_landmarks_are_converted_to_pixels_and_timestamps_increase():
    t = make_tracker()
    frame = np.zeros((480, 640, 3), np.uint8)
    hand = t.process(frame, 100)
    t.process(frame, 100)          # same timestamp twice must still be increasing for MediaPipe
    assert hand.landmarks.shape == (21, 3)
    assert np.allclose(hand.landmarks[0], [320, 120, -0.02 * 640])
    assert hand.handedness == "Right" and hand.score == pytest.approx(0.97)
    assert t._landmarker.timestamps == [100, 101]


def test_real_mediapipe_api_is_available():
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions  # the class names this project relies on
    assert hasattr(vision, "HandLandmarker") and hasattr(vision.RunningMode, "VIDEO") and opts


def test_missing_model_gives_clear_instructions(tmp_path):
    with pytest.raises(RuntimeError) as err:
        ensure_hand_model(tmp_path / "hand.task", url=(tmp_path / "nope.task").as_uri())
    assert "save it as" in str(err.value)


def test_existing_model_is_not_downloaded_again(tmp_path, monkeypatch):
    p = tmp_path / "hand.task"
    p.write_bytes(b"x" * 1_500_000)
    monkeypatch.setattr(hand_tracker.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert ensure_hand_model(p) == p
