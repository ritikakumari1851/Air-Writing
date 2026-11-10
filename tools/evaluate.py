#!/usr/bin/env python3
"""Measure how well the character model reads air-writing.

    python tools/evaluate.py                      # models/char_cnn.npz
    python tools/evaluate.py --model my_model.npz --seeds 10

Two tests, both without a webcam:

1. Whole app: a simulated hand writes every character in the air and the
   real app (gestures, smoothing, canvas, splitting, CNN) reads it.
2. Model only: hand-designed stroke templates, randomly distorted the way
   fingertip writing is, are given straight to the recogniser.

"Look-alikes OK" counts 0/O, 1/I, 5/S, 2/Z and 8/B as correct, because those
shapes are the same when written; the A-Z and 0-9 modes remove that problem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from airwriting.config import CHAR_MODEL_PATH, Settings  # noqa: E402
from airwriting.datagen import Augmenter  # noqa: E402
from airwriting.recognizer import TWINS, Recognizer  # noqa: E402
from airwriting.templates import CLASSES, DIGITS, LETTERS, TEMPLATES  # noqa: E402
from simulate import simulate  # noqa: E402


def score(pairs):
    pairs = list(pairs)
    strict = sum(t == p for t, p in pairs) / len(pairs)
    loose = sum(t == p or TWINS.get(t) == p for t, p in pairs) / len(pairs)
    return strict, loose, len(pairs)


def whole_app(rec, seeds, chars, charset_name=None):
    pairs = []
    for s in range(seeds):
        cfg = Settings()
        if charset_name:                 # make this character set the active one
            cfg.charsets = {charset_name: cfg.charsets[charset_name]}
        app = simulate("".join(chars), width=960, height=540, seed=1000 + s, settings=cfg,
                       recognizer=rec, render_hand=False)
        got = app.text.ljust(len(chars))[:len(chars)]
        pairs += list(zip(chars, got))
    return pairs


def model_only(rec, per_variant, strength=1.0, seed=7):
    aug, pairs = Augmenter(np.random.default_rng(seed), strength), []
    for ch in CLASSES:
        for _, strokes in TEMPLATES[ch]:
            for _ in range(per_variant):
                s = [p * 250 + 100 for p in aug.strokes(strokes)]
                pairs.append((ch, rec.predict(s).char))
    return pairs


def mnist_test(rec, path):
    """Real handwritten digits (MNIST test set), redrawn the way the app draws strokes."""
    from airwriting.datagen import load_mnist, raster_to_cloud
    from airwriting.preprocess import render_strokes
    _, _, xte, yte = load_mnist(path)
    imgs, labels = [], []
    for img, lab in zip(xte, yte):
        cloud = raster_to_cloud(img)
        if cloud is not None:
            imgs.append(render_strokes(clouds=[cloud], thickness=rec.thickness))
            labels.append(str(int(lab)))
    probs = rec.net.predict_proba(np.stack(imgs))
    mixed = [rec.classes[i] for i in probs.argmax(axis=1)]
    digit_idx = [rec.classes.index(d) for d in DIGITS]
    digits = [rec.classes[digit_idx[i]] for i in probs[:, digit_idx].argmax(axis=1)]
    return list(zip(labels, mixed)), list(zip(labels, digits))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(CHAR_MODEL_PATH))
    ap.add_argument("--seeds", type=int, default=5, help="simulated writers (each writes all 36 characters)")
    ap.add_argument("--per-variant", type=int, default=10, help="distorted samples per template variant")
    ap.add_argument("--mnist", help="also test on real handwritten digits: mnist.npz or folder with the IDX files")
    args = ap.parse_args()
    rec = Recognizer(args.model)

    rows = []
    if args.mnist:
        print("Reading the MNIST test digits...")
        mixed_d, digits_d = mnist_test(rec, args.mnist)
        rows.append(("MNIST test digits (real handwriting), mode A-Z 0-9", *score(mixed_d)))
        rows.append(("MNIST test digits (real handwriting), mode 0-9", *score(digits_d)))
    print("Simulating air-writing through the whole app...")
    mixed = whole_app(rec, args.seeds, CLASSES)
    rows.append(("Whole app, simulated hand, mode A-Z 0-9", *score(mixed)))
    split = whole_app(rec, args.seeds, DIGITS, "0-9") + whole_app(rec, args.seeds, LETTERS, "A-Z")
    rows.append(("Whole app, simulated hand, modes 0-9 / A-Z", *score(split)))
    print("Distorted templates straight into the model...")
    rows.append(("Model only, distorted stroke templates", *score(model_only(rec, args.per_variant))))

    print()
    print("| Test | Exact | Look-alikes OK | Characters |")
    print("|---|---|---|---|")
    for name, strict, loose, n in rows:
        print(f"| {name} | {strict * 100:.1f}% | {loose * 100:.1f}% | {n:,} |")
    errors = {}
    for t, p in mixed:
        if t != p:
            errors[f"{t}->{p}"] = errors.get(f"{t}->{p}", 0) + 1
    if errors:
        print("\nMistakes in the whole-app test:", ", ".join(f"{k} x{v}" for k, v in sorted(errors.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
