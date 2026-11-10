import numpy as np

from airwriting.datagen import Augmenter
from airwriting.recognizer import TWINS, Prediction, apply_context
from airwriting.templates import CLASSES, TEMPLATES


def test_reads_distorted_handwriting(recognizer):
    rng = np.random.default_rng(123)
    aug = Augmenter(rng, strength=0.8)
    correct = total = 0
    for ch in CLASSES:
        for w, strokes in TEMPLATES[ch]:
            for _ in range(4):
                s = [p * 250 + 100 for p in aug.strokes(strokes)]   # screen-sized strokes
                pred = recognizer.predict(s)
                ok = pred.char == ch or (TWINS.get(ch) == pred.char)
                correct += ok
                total += 1
    assert correct / total > 0.93


def test_charset_restriction(recognizer):
    zero = [np.array([[0.3 + 0.3 * np.cos(t), 0.5 + 0.5 * np.sin(t)] for t in np.linspace(-1.6, 4.8, 40)]) * 300]
    assert recognizer.predict(zero, allowed=list("0123456789")).char == "0"
    assert recognizer.predict(zero, allowed=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")).char in "OQDC"


def test_segments_two_characters(recognizer):
    one = [np.array([[0, 0], [0, 200]], np.float32) + [100, 100]]
    seven = [np.array([[0, 0], [130, 0], [45, 200]], np.float32) + [300, 100]]
    preds = recognizer.recognize(one + seven, allowed=list("0123456789"))
    assert [p.char for p in preds] == ["1", "7"]


def test_context_resolves_look_alikes():
    p = Prediction("O", 0.55, [("0", 0.40)], probs={"O": 0.55, "0": 0.40})
    assert apply_context([p], "2")[0].char == "0"
    q = Prediction("0", 0.6, [("O", 0.35)], probs={"0": 0.6, "O": 0.35})
    assert apply_context([q], "L")[0].char == "O"
    r = Prediction("0", 0.6, [("O", 0.35)], probs={"0": 0.6, "O": 0.35})
    assert apply_context([r], " ")[0].char == "0"        # no context after a space
