"""Training-data generation for the character recogniser (used by train_model.py).

Sources (all are turned into thin centre-lines, then redrawn with the same
renderer the live app uses, so every source looks alike to the network):

* stroke templates (templates.py) distorted to look like fingertip paths,
* font glyphs: OpenCV's built-in Hershey fonts (always available) and any
  TrueType/OpenType fonts found on the computer (needs Pillow),
* optional real handwriting: MNIST digits, EMNIST letters,
* optional samples you recorded yourself with `air_writing.py --collect`.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import struct
from pathlib import Path

import cv2
import numpy as np

from .preprocess import render_strokes
from .templates import CLASSES, TEMPLATES

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Thinning (skeletonisation): reduces a filled glyph to its 1-pixel centre line
# ---------------------------------------------------------------------------

def _zhang_suen(img: np.ndarray) -> np.ndarray:
    I = np.pad((img > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P2, P3, P4 = I[:-2, 1:-1], I[:-2, 2:], I[1:-1, 2:]
            P5, P6, P7 = I[2:, 2:], I[2:, 1:-1], I[2:, :-2]
            P8, P9, Cc = I[1:-1, :-2], I[:-2, :-2], I[1:-1, 1:-1]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(n.astype(np.int16) for n in nb[:8])
            Acnt = sum(((nb[k] == 0) & (nb[k + 1] == 1)).astype(np.int16) for k in range(8))
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            m = (Cc == 1) & (B >= 2) & (B <= 6) & (Acnt == 1) & cond
            if m.any():
                Cc[m] = 0
                changed = True
        if not changed:
            break
    return I[1:-1, 1:-1] * 255


def thin(binary: np.ndarray) -> np.ndarray:
    """Skeletonise a 0/255 uint8 image (uses opencv-contrib when available)."""
    binary = np.ascontiguousarray(binary, dtype=np.uint8)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return ximgproc.thinning(binary)
    return _zhang_suen(binary).astype(np.uint8)


def mask_to_cloud(binary: np.ndarray, min_points: int = 8):
    sk = thin(binary)
    ys, xs = np.nonzero(sk)
    if len(xs) < min_points:
        return None
    return np.stack([xs, ys], axis=1).astype(np.float32)


def raster_to_cloud(img: np.ndarray, up: int = 4, threshold: float = 0.35):
    """Grey handwriting image (e.g. 28x28 MNIST, white on black) -> skeleton point cloud."""
    big = cv2.resize(img, (img.shape[1] * up, img.shape[0] * up), interpolation=cv2.INTER_CUBIC)
    binary = (big > threshold * 255).astype(np.uint8) * 255
    return mask_to_cloud(binary)


# ---------------------------------------------------------------------------
# Font glyphs
# ---------------------------------------------------------------------------

HERSHEY_FACES = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
]


def hershey_clouds(chars=CLASSES) -> dict[str, list]:
    out: dict[str, list] = {c: [] for c in chars}
    for face in HERSHEY_FACES:
        for italic in (0, cv2.FONT_ITALIC):
            f = face | italic
            (_, h1), _ = cv2.getTextSize("H", f, 1.0, 1)
            scale = 80.0 / max(h1, 1)
            for ch in chars:
                (w, h), _ = cv2.getTextSize(ch, f, scale, 3)
                img = np.zeros((220, 220), np.uint8)
                cv2.putText(img, ch, (110 - w // 2, 110 + h // 2), f, scale, 255, 3, cv2.LINE_8)
                cloud = mask_to_cloud(img)
                if cloud is not None:
                    out[ch].append(cloud)
    return out


def ttf_clouds(font_paths, chars=CLASSES, size: int = 96) -> dict[str, list]:
    """Skeletons of glyphs from TrueType/OpenType fonts (requires Pillow)."""
    out: dict[str, list] = {c: [] for c in chars}
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  Pillow not installed - skipping TrueType fonts")
        return out
    for path in font_paths:
        try:
            font = ImageFont.truetype(str(path), size)
        except Exception:
            continue
        for ch in chars:
            img = Image.new("L", (size * 2, size * 2), 0)
            d = ImageDraw.Draw(img)
            x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=font)
            if x1 - x0 < 2 or y1 - y0 < 8:
                continue
            d.text((size - (x0 + x1) / 2, size - (y0 + y1) / 2), ch, font=font, fill=255)
            cloud = mask_to_cloud((np.asarray(img) > 127).astype(np.uint8) * 255)
            if cloud is not None:
                out[ch].append(cloud)
    return out


# Plain Latin fonts that commonly exist on Windows / macOS / Linux. Symbol fonts
# (Wingdings...) would teach nonsense, so only these name prefixes are used.
FONT_WHITELIST = (
    "arial", "calibri", "cambria", "candara", "consola", "constan", "corbel", "cour", "georgia",
    "segoeui", "segoepr", "segoesc", "tahoma", "times", "trebuc", "verdana", "comic", "lucon",
    "helvetica", "avenir", "futura", "gillsans", "optima", "palatino", "menlo", "monaco",
    "dejavusans", "dejavuserif", "liberationsans", "liberationserif", "liberationmono",
    "freesans", "freeserif", "freemono", "carlito", "caladea", "inter-", "poppins", "lora",
    "texgyre", "ubuntu", "notosans-", "notoserif-", "opensans", "roboto", "lato",
)
FONT_DIRS = (
    "C:/Windows/Fonts", os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
    "/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
)


def find_system_fonts(limit: int = 40) -> list[str]:
    """Pick regular/italic weights of well-known Latin fonts installed on this computer."""
    found = []
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "**", "*.[ot]tf"), recursive=True):
            name = os.path.basename(p).lower()
            if not name.startswith(FONT_WHITELIST):
                continue
            if any(k in name for k in ("bold", "black", "heavy", "semibold", "extrabold", "thin", "symbol",
                                       "math", "emoji", "display", "cond", "narrow", "medium")):
                continue
            found.append(p)
    uniq = sorted(set(found), key=lambda p: os.path.basename(p).lower())
    return uniq[:limit]


# ---------------------------------------------------------------------------
# Real handwriting datasets (optional)
# ---------------------------------------------------------------------------

def _read_idx(path: str) -> np.ndarray:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", data[:4])
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path} is not a uint8 IDX file")
    shape = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _find(folder: str, *patterns: str):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(folder, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


def load_mnist(path: str):
    """MNIST from a Keras-style mnist.npz or a folder with the four IDX(.gz) files.

    Returns (x_train, y_train, x_test, y_test) as uint8 arrays (white digit on black).
    """
    if os.path.isfile(path) and path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    folder = path
    xs = _find(folder, "train-images-idx3-ubyte*", "train-images.idx3-ubyte*")
    ys = _find(folder, "train-labels-idx1-ubyte*", "train-labels.idx1-ubyte*")
    xt = _find(folder, "t10k-images-idx3-ubyte*", "t10k-images.idx3-ubyte*")
    yt = _find(folder, "t10k-labels-idx1-ubyte*", "t10k-labels.idx1-ubyte*")
    if not all((xs, ys, xt, yt)):
        raise FileNotFoundError(f"MNIST files not found in {folder}")
    return _read_idx(xs), _read_idx(ys), _read_idx(xt), _read_idx(yt)


def load_emnist(folder: str):
    """EMNIST 'letters' (or 'balanced'/'byclass' with mapping file) -> (images, chars).

    Images are transposed back to the normal orientation. Only digits and
    letters are kept; EMNIST 'letters' mixes upper and lower case under one label.
    """
    for split in ("letters", "balanced", "byclass"):
        xi = _find(folder, f"emnist-{split}-train-images-idx3-ubyte*")
        yi = _find(folder, f"emnist-{split}-train-labels-idx1-ubyte*")
        if xi and yi:
            break
    else:
        raise FileNotFoundError(f"EMNIST letters/balanced/byclass IDX files not found in {folder}")
    x = _read_idx(xi).transpose(0, 2, 1)
    y = _read_idx(yi).astype(np.int64)
    if split == "letters":
        chars = np.array([chr(ord("A") + int(v) - 1) for v in y])
    else:
        mapping_file = _find(folder, f"emnist-{split}-mapping.txt")
        if not mapping_file:
            raise FileNotFoundError(f"emnist-{split}-mapping.txt is needed next to the IDX files")
        mp = {}
        for line in Path(mapping_file).read_text().split("\n"):
            parts = line.split()
            if len(parts) >= 2:
                mp[int(parts[0])] = chr(int(parts[1]))
        chars = np.array([mp.get(int(v), "?") for v in y])
    keep = np.isin(chars, CLASSES)
    return x[keep], chars[keep]


def load_user_samples(path: str) -> dict[str, list]:
    """Samples recorded with `air_writing.py --collect` (JSON lines of strokes)."""
    out: dict[str, list] = {c: [] for c in CLASSES}
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = str(rec.get("label", "")).upper()
                strokes = [np.asarray(s, np.float32).reshape(-1, 2) for s in rec.get("strokes", []) if len(s) >= 2]
                if label in out and strokes:
                    out[label].append(strokes)
    return out


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _arclength(p: np.ndarray) -> np.ndarray:
    if len(p) < 2:
        return np.zeros(len(p), np.float32)
    return np.concatenate([[0.0], np.cumsum(np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1)))]).astype(np.float32)


def _trim_end(q: np.ndarray, amount: float) -> np.ndarray:
    """Shorten a polyline by `amount` of path length at its end (never by more than a third)."""
    s = _arclength(q)
    target = float(s[-1]) - amount
    if len(q) < 2 or target < float(s[-1]) * 0.67:
        return q
    k = int(np.searchsorted(s, target))
    k = min(max(k, 1), len(q) - 1)
    t = (target - s[k - 1]) / max(float(s[k] - s[k - 1]), 1e-9)
    cut = q[k - 1] + t * (q[k] - q[k - 1])
    return np.vstack([q[:k], cut[None]]).astype(np.float32)


def _resample(p: np.ndarray, spacing: float) -> np.ndarray:
    s = _arclength(p)
    total = float(s[-1]) if len(s) else 0.0
    if total < 1e-6:
        return p[:1].copy()
    n = max(2, int(np.ceil(total / spacing)) + 1)
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1).astype(np.float32)


class Augmenter:
    """Random distortions that imitate a fingertip writing in the air."""

    def __init__(self, rng: np.random.Generator, strength: float = 1.0):
        self.rng = rng
        self.k = strength

    # -- geometry shared by strokes and clouds ---------------------------------
    def _global_warp(self, parts):
        r, k = self.rng, self.k
        allp = np.vstack(parts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        h = max(float(hi[1] - lo[1]), float(hi[0] - lo[0]), 1e-6)
        c = (lo + hi) / 2
        ang = np.radians(np.clip(r.normal(0, 7 * k), -18, 18))
        shear = r.uniform(-0.3, 0.3) * k
        aspect = np.exp(np.clip(r.normal(0, 0.14 * k), -0.35, 0.35))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        M = rot @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([aspect, 1.0 / aspect])
        waves = []
        for _ in range(4):
            th = r.uniform(0, 2 * np.pi)
            waves.append((np.array([np.cos(th), np.sin(th)]), r.uniform(0.6, 2.2), r.uniform(0, 2 * np.pi),
                          r.normal(0, 0.022 * k, size=2)))
        out = []
        for p in parts:
            q = (p - c) / h
            q = q @ M.T
            for u, f, ph, a in waves:
                q = q + np.sin(2 * np.pi * f * (q @ u) + ph)[:, None] * a
            out.append(q.astype(np.float32))
        return out

    # -- strokes (ordered fingertip paths) -------------------------------------
    def strokes(self, strokes):
        r, k = self.rng, self.k
        parts = [np.asarray(s, np.float32).copy() for s in strokes if len(s)]
        allp = np.vstack(parts)
        span = max(float(np.ptp(allp[:, 1])), float(np.ptp(allp[:, 0])), 1e-6)
        parts = [p / span for p in parts]

        if len(parts) > 1:                                  # imprecise placement of each stroke
            parts = [p + r.normal(0, 0.022 * k, size=2).astype(np.float32) for p in parts]

        new = []
        for p in parts:                                     # over/undershoot and little hooks
            if len(p) >= 2:
                for end in (0, 1):
                    q = p if end == 1 else p[::-1]
                    d = q[-1] - q[-2]
                    n = np.linalg.norm(d)
                    if n > 1e-6 and r.random() < 0.45:
                        ext = r.uniform(-0.04, 0.06) * k
                        if ext > 0:
                            q = np.vstack([q, q[-1] + d / n * ext])
                        else:
                            q = _trim_end(q, -ext)
                    if r.random() < 0.12 * k:
                        th = r.uniform(0, 2 * np.pi)
                        q = np.vstack([q, q[-1] + np.array([np.cos(th), np.sin(th)], np.float32) * r.uniform(0.03, 0.09)])
                    p = q if end == 1 else q[::-1]
            new.append(np.asarray(p, np.float32))
        parts = new

        if len(parts) > 1 and r.random() < 0.18 * k:        # pen not lifted between strokes
            i = int(r.integers(0, len(parts) - 1))
            parts = parts[:i] + [np.vstack([parts[i], parts[i + 1]])] + parts[i + 2:]

        wobbly = []
        for p in parts:                                     # slow hand wobble along the path
            q = _resample(p, 0.006)
            s = _arclength(q)
            for _ in range(2):
                f, ph = r.uniform(0.6, 3.0), r.uniform(0, 2 * np.pi, size=2)
                q = q + (np.sin(2 * np.pi * f * s[:, None] + ph) * r.normal(0, 0.011 * k, size=2)).astype(np.float32)
            wobbly.append(q)
        parts = self._global_warp(wobbly)

        out = []
        for p in parts:                                     # camera-rate sampling, jitter, smoothing
            q = _resample(p, r.uniform(0.02, 0.07))
            q = q + r.normal(0, r.uniform(0, 0.007) * k, size=q.shape).astype(np.float32)
            alpha = r.uniform(0.5, 1.0)
            for i in range(1, len(q)):
                q[i] = alpha * q[i] + (1 - alpha) * q[i - 1]
            if len(q) > 6 and r.random() < 0.06 * k:      # a short tracking dropout
                a = int(r.integers(1, len(q) - 3))
                out.append(q[:a])
                q = q[a + max(1, len(q) // 25):]
            out.append(q)
        return [q for q in out if len(q)]

    # -- clouds (skeletons of fonts / scanned handwriting) ---------------------
    def cloud(self, cloud):
        q = self._global_warp([np.asarray(cloud, np.float32)])[0]
        return q + self.rng.normal(0, 0.003 * self.k, size=q.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _pick_template(rng, ch):
    variants = TEMPLATES[ch]
    w = np.array([v[0] for v in variants], np.float64)
    return variants[int(rng.choice(len(variants), p=w / w.sum()))][1]


def make_sample(rng, aug, source, item, thickness_range=(1.7, 3.3)):
    t = rng.uniform(*thickness_range)
    if source in ("template", "user"):
        return render_strokes(strokes=aug.strokes(item), thickness=t)
    return render_strokes(clouds=[aug.cloud(item)], thickness=t)


def build_dataset(n_per_class: int, rng: np.random.Generator, fonts=None, mnist=None,
                  emnist=None, user=None, strength: float = 1.0, verbose: bool = True):
    """Return (X uint8 [N,28,28], y int64 [N], src list) with roughly n_per_class per class.

    fonts/mnist/emnist: dict char -> list of point clouds; user: dict char -> list of stroke lists.
    """
    aug = Augmenter(rng, strength)
    X, Y, SRC = [], [], []
    for ch in CLASSES:
        pools = {"template": None}
        if fonts and fonts.get(ch):
            pools["font"] = fonts[ch]
        if mnist and mnist.get(ch):
            pools["mnist"] = mnist[ch]
        if emnist and emnist.get(ch):
            pools["emnist"] = emnist[ch]
        if user and user.get(ch):
            pools["user"] = user[ch]
        weights = {"template": 0.55, "font": 0.45, "mnist": 0.45, "emnist": 0.45, "user": 0.35}
        if "mnist" in pools or "emnist" in pools:
            weights.update(template=0.35, font=0.25)
        names = list(pools)
        w = np.array([weights[n] for n in names])
        counts = np.floor(w / w.sum() * n_per_class).astype(int)
        counts[0] += n_per_class - counts.sum()
        for name, cnt in zip(names, counts):
            pool = pools[name]
            for _ in range(int(cnt)):
                item = _pick_template(rng, ch) if name == "template" else pool[int(rng.integers(len(pool)))]
                src = "user" if name == "user" else ("template" if name == "template" else "cloud")
                img = make_sample(rng, aug, src, item)
                X.append(np.round(img * 255).astype(np.uint8))
                Y.append(CLASS_INDEX[ch])
                SRC.append(name)
        if verbose:
            print(f"\r  generated class {ch}", end="", flush=True)
    if verbose:
        print()
    X = np.stack(X)
    Y = np.array(Y, np.int64)
    perm = rng.permutation(len(Y))
    return X[perm], Y[perm], [SRC[i] for i in perm]


def clouds_from_images(images, labels_chars, per_class: int, rng, threshold: float = 0.35):
    """Convert a sample of real handwriting images into skeleton clouds, per class."""
    out: dict[str, list] = {c: [] for c in CLASSES}
    labels_chars = np.asarray(labels_chars)
    for ch in CLASSES:
        idx = np.nonzero(labels_chars == ch)[0]
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for i in idx:
            c = raster_to_cloud(images[i], threshold=threshold)
            if c is not None:
                out[ch].append(c)
    return out
