"""Character recognition: strokes -> 28x28 image -> CNN -> letter/digit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cnn import CNN
from .preprocess import DEFAULT_THICKNESS, bounding_box, group_strokes, render_strokes


@dataclass
class Prediction:
    char: str
    confidence: float
    alternatives: list = field(default_factory=list)   # [(char, prob), ...] after the best one
    image: np.ndarray | None = None                   # the 28x28 image the network saw
    box: tuple | None = None                          # (x0, y0, x1, y1) of the strokes
    probs: dict = field(default_factory=dict)         # probability of every allowed character
    by_context: bool = False                          # True if a look-alike was swapped (0/O, 1/I...)


class Recognizer:
    def __init__(self, model_path: Path | str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Character model not found: {model_path}\n"
                                    "Train one with:  python train_model.py")
        self.net, self.meta = CNN.load(model_path)
        self.classes: list[str] = list(self.meta["classes"])
        self.thickness = float(self.meta.get("thickness", DEFAULT_THICKNESS))

    def predict(self, strokes, allowed=None, topk: int = 3) -> Prediction:
        img = render_strokes(strokes=strokes, thickness=self.thickness)
        probs = self.net.predict_proba(img[None])[0]
        if allowed:
            mask = np.array([c in allowed for c in self.classes])
            probs = np.where(mask, probs, 0.0)
            probs = probs / max(probs.sum(), 1e-9)
        order = np.argsort(-probs)
        best = int(order[0])
        alts = [(self.classes[i], float(probs[i])) for i in order[1:topk]]
        lo, hi = bounding_box(strokes)
        return Prediction(self.classes[best], float(probs[best]), alts, img,
                          (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
                          {c: float(q) for c, q in zip(self.classes, probs) if q > 0})

    def recognize(self, strokes, allowed=None, min_size: float = 0.0, segment: bool = True) -> list[Prediction]:
        """Read the strokes as one or more characters (left to right)."""
        strokes = [np.asarray(s, np.float32).reshape(-1, 2) for s in strokes if len(s)]
        if not strokes:
            return []
        groups = group_strokes(strokes) if segment else [strokes]
        out = []
        for g in groups:
            lo, hi = bounding_box(g)
            if float((hi - lo).max()) < min_size:
                continue            # a stray mark, not a character
            out.append(self.predict(g, allowed))
        return out


# Shapes that look the same as a letter and as a digit.
TWINS = {"0": "O", "O": "0", "1": "I", "I": "1", "5": "S", "S": "5", "2": "Z", "Z": "2", "8": "B", "B": "8"}


def apply_context(preds: list[Prediction], previous: str, min_twin_prob: float = 0.1) -> list[Prediction]:
    """Resolve look-alikes from the neighbouring character: after a digit, an
    ambiguous O becomes 0; after a letter, a 0 becomes O, and so on."""
    prev = previous
    for p in preds:
        twin = TWINS.get(p.char)
        if twin and prev and not prev.isspace() and prev.isdigit() != p.char.isdigit():
            q = p.probs.get(twin, 0.0)
            if q >= min_twin_prob:
                p.alternatives = [(p.char, p.confidence)] + [a for a in p.alternatives if a[0] != twin]
                p.char, p.confidence, p.by_context = twin, p.confidence + q, True
        prev = p.char
    return preds
