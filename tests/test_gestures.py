import numpy as np
import pytest

from airwriting.filters import OneEuroFilter
from airwriting.gestures import Gesture, GestureDetector
from simulate import SimHand


@pytest.mark.parametrize("gesture", [Gesture.DRAW, Gesture.HOVER, Gesture.PALM, Gesture.FIST])
@pytest.mark.parametrize("tilt", [-35, 0, 35])
@pytest.mark.parametrize("scale", [0.6, 1.0, 1.8])
def test_gestures_recognised_for_rotated_and_scaled_hands(gesture, tilt, scale):
    hand = SimHand(scale=scale, seed=1, noise=0.5 * scale)
    det = GestureDetector(stable_frames=3)
    a = np.radians(tilt)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
    for _ in range(5):
        lm = hand.shape(gesture) @ rot.T + np.array([640, 400, 0], np.float32)
        lm += hand.rng.normal(0, hand.noise, lm.shape).astype(np.float32)
        out = det.update(lm)
    assert out == gesture


def test_gesture_needs_a_few_frames_and_hand_loss_is_immediate():
    hand = SimHand(seed=0)
    det = GestureDetector(stable_frames=3)
    for _ in range(4):
        det.update(hand.landmarks((500, 300), Gesture.HOVER))
    assert det.stable == Gesture.HOVER
    det.update(hand.landmarks((500, 300), Gesture.DRAW))
    assert det.stable == Gesture.HOVER          # one frame is not enough
    det.update(hand.landmarks((500, 300), Gesture.DRAW))
    det.update(hand.landmarks((500, 300), Gesture.DRAW))
    assert det.stable == Gesture.DRAW
    assert det.update(None) == Gesture.NONE


def test_one_euro_filter_smooths_jitter_but_follows_motion():
    f = OneEuroFilter(min_cutoff=1.5, beta=0.01)
    rng = np.random.default_rng(0)
    still = [f(np.array([100.0, 100.0]) + rng.normal(0, 3, 2), t / 30) for t in range(60)]
    assert np.std(np.array(still[20:]), axis=0).max() < 1.5      # jitter of 3 px is damped
    f.reset()
    moving = [f(np.array([10.0 * t, 0.0]), t / 30) for t in range(60)]
    assert abs(moving[-1][0] - 590) < 25                          # lag stays small when moving fast
