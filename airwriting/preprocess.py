"""Turn pen strokes into the 28x28 image the recogniser reads.

The same function is used for the training data and for live air-writing, so
what the network sees at run time looks exactly like what it learned from:

* the character is fitted into a 20x20 box (aspect ratio kept) and centred
  in a 28x28 image, like MNIST/EMNIST;
* every stroke is redrawn with the same fixed thickness, so it does not
  matter how big you write in the air or which brush size you use.

Strokes are lists of (x, y) points in any coordinate system (screen pixels,
glyph units...). "Clouds" are unordered point sets (used for skeletons of
font glyphs and MNIST digits during training); they are drawn as dots.
"""
from __future__ import annotations

import numpy as np
import cv2

IMG_SIZE = 28          # network input size
BOX_SIZE = 20          # the character is fitted inside this box
DEFAULT_THICKNESS = 2.5  # stroke width in pixels of the 28x28 image
SUPERSAMPLE = 4        # render at 4x and downsample for anti-aliasing


def densify(points: np.ndarray, step: float) -> np.ndarray:
    """Insert points along a polyline so neighbours are at most `step` apart."""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(p) < 2:
        return p
    seg = np.diff(p, axis=0)
    dist = np.sqrt((seg ** 2).sum(axis=1))
    n = np.maximum(1, np.ceil(dist / step).astype(np.int64))
    idx = np.repeat(np.arange(len(seg)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(n.sum()) - start) / np.repeat(n, n)
    out = p[idx] + seg[idx] * frac[:, None].astype(np.float32)
    return np.vstack([out, p[-1:]])


def bounding_box(strokes) -> tuple[np.ndarray, np.ndarray] | None:
    pts = [np.asarray(s, np.float32).reshape(-1, 2) for s in strokes if len(s)]
    if not pts:
        return None
    allp = np.vstack(pts)
    return allp.min(axis=0), allp.max(axis=0)


def render_strokes(strokes=(), clouds=(), size: int = IMG_SIZE, box: int = BOX_SIZE,
                   thickness: float = DEFAULT_THICKNESS, ss: int = SUPERSAMPLE) -> np.ndarray:
    """Render strokes/clouds into a normalised float32 image in [0, 1] (white ink on black)."""
    strokes = [np.asarray(s, np.float32).reshape(-1, 2) for s in strokes if len(s)]
    clouds = [np.asarray(c, np.float32).reshape(-1, 2) for c in clouds if len(c)]
    bb = bounding_box(strokes + clouds)
    if bb is None:
        return np.zeros((size, size), np.float32)
    lo, hi = bb
    extent = float(max((hi - lo).max(), 1e-6))
    scale = (box * ss) / extent
    centre = (lo + hi) / 2.0
    canvas = size * ss
    offset = canvas / 2.0

    img = np.zeros((canvas, canvas), np.uint8)

    def plot(q):
        ij = np.round(q).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < canvas) & (ij[:, 1] >= 0) & (ij[:, 1] < canvas)
        ij = ij[ok]
        img[ij[:, 1], ij[:, 0]] = 255

    for s in strokes:
        plot(densify((s - centre) * scale + offset, 0.7))
    for c in clouds:
        plot((c - centre) * scale + offset)

    radius = max(1, int(round(thickness * ss / 2.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    img = cv2.dilate(img, kernel)
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def group_strokes(strokes, gap_ratio: float = 0.12, fragment_ratio: float = 0.4, fragment_gap_ratio: float = 0.5):
    """Split strokes into characters written side by side.

    Strokes whose horizontal ranges overlap (or nearly touch) belong to the same
    character. A small piece next to a character (smaller than `fragment_ratio`
    of the tallest character and closer than `fragment_gap_ratio` of its height),
    e.g. the start of a stroke cut off when the tracker lost the hand for a moment,
    joins its nearest neighbour instead of becoming a character of its own.
    Returns a list of stroke lists sorted left to right.
    """
    items = []
    for s in strokes:
        p = np.asarray(s, np.float32).reshape(-1, 2)
        if len(p) == 0:
            continue
        items.append([p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(), [p]])
    if not items:
        return []
    height = max(it[3] for it in items) - min(it[2] for it in items)
    tol = gap_ratio * max(height, 1.0)
    items.sort(key=lambda it: it[0])
    groups = [items[0]]
    for it in items[1:]:
        g = groups[-1]
        if it[0] <= g[1] + tol:            # overlaps the current group horizontally
            _merge(g, it)
        else:
            groups.append(it)

    while len(groups) > 1:                 # fold small fragments into their nearest neighbour
        ref = max(g[3] - g[2] for g in groups)
        for i, g in enumerate(groups):
            if max(g[1] - g[0], g[3] - g[2]) >= fragment_ratio * ref:
                continue
            near = [(g[0] - groups[i - 1][1], i - 1)] if i > 0 else []
            if i + 1 < len(groups):
                near.append((groups[i + 1][0] - g[1], i + 1))
            gap, j = min(near)
            if gap <= fragment_gap_ratio * ref:
                _merge(groups[j], g)
                del groups[i]
                break
        else:
            break
    return [g[4] for g in groups]


def _merge(g, other):
    """Grow group `g` ([x0, x1, y0, y1, strokes]) by `other`."""
    g[0], g[1] = min(g[0], other[0]), max(g[1], other[1])
    g[2], g[3] = min(g[2], other[2]), max(g[3], other[3])
    g[4].extend(other[4])
