#!/usr/bin/env python3
"""Simulate a hand writing in the air - no webcam needed.

Generates MediaPipe-style hand landmarks for each gesture, moves the index
fingertip along handwriting strokes, and feeds everything through the real
AirWritingApp (gestures, smoothing, canvas, recognition, UI). Used by the
tests and to make the demo video:

    python tools/simulate.py --text "HELLO 2026" --gif docs/demo.gif --mp4 docs/demo.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from airwriting.app import AirWritingApp  # noqa: E402
from airwriting.config import Settings  # noqa: E402
from airwriting.datagen import Augmenter  # noqa: E402
from airwriting.gestures import HAND_CONNECTIONS, Gesture  # noqa: E402
from airwriting.hand_tracker import Hand  # noqa: E402
from airwriting.templates import TEMPLATES  # noqa: E402

# Right hand, palm towards the camera, fingers up. Units: pixels for a ~200 px tall hand.
EXTENDED = {
    0: (0, 0, 0), 1: (-30, -20, -5), 2: (-55, -45, -10), 3: (-75, -68, -14), 4: (-92, -90, -18),
    5: (-28, -95, -5), 6: (-32, -135, -9), 7: (-34, -160, -12), 8: (-36, -182, -14),
    9: (-2, -100, -5), 10: (-2, -145, -9), 11: (-2, -172, -12), 12: (-2, -196, -14),
    13: (22, -95, -5), 14: (26, -135, -9), 15: (28, -158, -12), 16: (30, -178, -13),
    17: (42, -82, -4), 18: (50, -110, -7), 19: (54, -128, -9), 20: (57, -145, -10),
}
CURLED_TIPS = {  # (PIP, DIP, TIP) of a folded finger
    "index": {6: (-32, -128, -15), 7: (-30, -108, -28), 8: (-28, -94, -30)},
    "middle": {10: (-2, -135, -15), 11: (-1, -112, -28), 12: (0, -98, -30)},
    "ring": {14: (25, -128, -14), 15: (25, -108, -26), 16: (24, -95, -28)},
    "pinky": {18: (48, -105, -12), 19: (47, -92, -22), 20: (45, -82, -24)},
}
THUMB_TUCKED = {2: (-45, -50, -15), 3: (-40, -75, -25), 4: (-25, -88, -30)}
UP = {
    Gesture.DRAW: ("index",),
    Gesture.HOVER: ("index", "middle"),
    Gesture.PALM: ("index", "middle", "ring", "pinky"),
    Gesture.FIST: (),
}


class SimHand:
    def __init__(self, scale=1.0, seed=0, noise=1.0):
        self.scale, self.noise = scale, noise
        self.rng = np.random.default_rng(seed)
        self.tilt = 0.0

    def shape(self, gesture: Gesture) -> np.ndarray:
        pts = {k: np.array(v, np.float32) for k, v in EXTENDED.items()}
        for finger in ("index", "middle", "ring", "pinky"):
            if finger not in UP[gesture]:
                for k, v in CURLED_TIPS[finger].items():
                    pts[k] = np.array(v, np.float32)
        if gesture != Gesture.PALM:
            for k, v in THUMB_TUCKED.items():
                pts[k] = np.array(v, np.float32)
        return np.stack([pts[i] for i in range(21)]) * self.scale

    def landmarks(self, tip_xy, gesture: Gesture) -> np.ndarray:
        self.tilt = 0.9 * self.tilt + 0.1 * self.rng.normal(0, 6)
        a = np.radians(self.tilt)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
        lm = self.shape(gesture) @ rot.T
        lm[:, :2] += np.asarray(tip_xy, np.float32) - lm[8, :2]      # put the index tip on the target
        lm += self.rng.normal(0, self.noise, lm.shape).astype(np.float32)   # detector jitter
        return lm


def _path(points, speed, fps, rng):
    """Resample a polyline into per-frame positions at roughly `speed` px/s."""
    p = np.asarray(points, np.float32)
    seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(seg)])
    total = float(s[-1])
    n = max(2, int(total / (speed * rng.uniform(0.8, 1.2)) * fps))
    t = np.linspace(0, total, n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def script_for_text(text, width, height, fps=30, seed=0, char_height=None, augment=True, settings=None):
    """List of (fingertip xy, gesture) per frame that writes `text`, one character at a time.

    Like a person would: left to right across the screen, and when the line is full the
    open palm clears it (with --auto-clear behaviour every character goes in the middle).
    """
    rng = np.random.default_rng(seed)
    aug = Augmenter(rng, strength=0.8)
    cfg = settings or Settings()
    s = height / 720.0
    top, bottom = 78 * s, height - 110 * s           # free area between the toolbar and the text bar
    ch_h = char_height or height * 0.33
    cell = ch_h * 0.9                                  # horizontal room per character
    left, right = width * 0.07, width * 0.93
    cy = (top + bottom) / 2
    x = left + cell / 2                                # centre of the next character
    frames = []
    pos = np.array([width * 0.75, height * 0.6], np.float32)

    def hold(g, seconds, at=None):
        nonlocal pos
        for _ in range(int(seconds * fps)):
            frames.append((pos + rng.normal(0, 0.6, 2) if at is None else at, g))

    def move(target, g=Gesture.HOVER, speed=900):
        nonlocal pos
        for q in _path([pos, target], speed, fps, rng)[1:]:
            frames.append((q, g))
        pos = np.asarray(target, np.float32)

    space_btn = None
    for ch in text:
        if ch == " ":
            if space_btn is None:
                from airwriting.ui import Toolbar
                b = [b for b in Toolbar(width, height).buttons if b.name == "space"][0]
                space_btn = np.array([(b.rect[0] + b.rect[2]) / 2, (b.rect[1] + b.rect[3]) / 2], np.float32)
            move(space_btn)
            hold(Gesture.HOVER, cfg.dwell_seconds + 0.25)
            move(np.array([min(x, right - cell / 2), cy + ch_h * 0.45]))
            x += cell * 0.6                            # leave a gap between words
            continue
        if cfg.keep_ink and x + cell / 2 > right:      # line full: wipe it with an open palm
            hold(Gesture.HOVER, 0.15)
            hold(Gesture.PALM, cfg.palm_hold_seconds + 0.35)
            hold(Gesture.HOVER, 0.2)
            x = left + cell / 2
        variants = TEMPLATES[ch]
        w = np.array([v[0] for v in variants])
        strokes = variants[int(rng.choice(len(variants), p=w / w.sum()))][1]
        if augment:
            strokes = aug.strokes(strokes)
        allp = np.vstack(strokes)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        sc = ch_h / max(float(hi[1] - lo[1]), float(hi[0] - lo[0]) * 0.8, 1e-6)
        cx = x if cfg.keep_ink else width * 0.5
        off = np.array([cx, cy]) - (lo + hi) / 2 * sc + rng.normal(0, 10, 2)
        for st in strokes:
            st = np.asarray(st) * sc + off
            if len(st) < 2:
                continue
            move(st[0], Gesture.HOVER)
            hold(Gesture.DRAW, 0.12, at=st[0])        # finger goes up, then moves
            for q in _path(st, rng.uniform(450, 700), fps, rng)[1:]:
                frames.append((q, Gesture.DRAW))
            pos = st[-1].astype(np.float32)
            hold(Gesture.HOVER, 0.15)
        if cfg.keep_ink:                               # rest beside the character while it is read
            rest = np.array([min(x + cell * 0.55, right), min(cy + ch_h * 0.45, bottom - 20 * s)])
        else:
            rest = np.array([width * 0.78, height * 0.55])
        move(rest)
        hold(Gesture.HOVER, cfg.recognize_delay + 0.6)
        x += cell
    hold(Gesture.HOVER, 0.8)
    return frames


def fake_background(width, height, seed=0):
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height)[:, None, None]
    x = np.linspace(0, 1, width)[None, :, None]
    base = (np.array([70, 60, 52]) * (1 - y) + np.array([40, 42, 48]) * y + 18 * x).astype(np.float32)
    img = np.broadcast_to(base, (height, width, 3)).copy()
    for _ in range(6):
        c = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        cv2.circle(img, c, int(rng.uniform(60, 220)), tuple(float(v) for v in rng.uniform(40, 90, 3)), -1)
    return cv2.GaussianBlur(img, (0, 0), 25).astype(np.uint8)


def draw_hand(img, lm, scale):
    skin = (140, 170, 215)
    pts = [(int(p[0]), int(p[1])) for p in lm]
    palm = np.array([pts[i] for i in (0, 1, 5, 9, 13, 17)], np.int32)
    cv2.fillConvexPoly(img, cv2.convexHull(palm), skin, cv2.LINE_AA)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], skin, int(22 * scale), cv2.LINE_AA)


def simulate(text, width=1280, height=720, fps=30, seed=0, settings=None, recognizer=None, render_hand=True,
             frame_callback=None, augment=True):
    cfg = settings or Settings()
    if recognizer is None:
        from airwriting.config import CHAR_MODEL_PATH
        from airwriting.recognizer import Recognizer
        recognizer = Recognizer(CHAR_MODEL_PATH)
    app = AirWritingApp(width, height, recognizer, cfg)
    hand = SimHand(scale=height / 720 * 1.1, seed=seed)
    bg = fake_background(width, height, seed)
    for i, (tip, g) in enumerate(script_for_text(text, width, height, fps, seed, settings=cfg, augment=augment)):
        lm = hand.landmarks(tip, g)
        frame = bg.copy()
        if render_hand:
            draw_hand(frame, lm, hand.scale)
        out = app.step(frame, Hand(lm), now=i / fps)
        if frame_callback is not None:
            frame_callback(i, out, app)
    return app


def save_gif(path, frames_rgb, fps=10, colors=256):
    """Animated GIF with one shared palette: unchanged pixels stay identical from frame to
    frame, so the GIF only stores what moved and stays small."""
    from PIL import Image

    from airwriting import config
    step = max(1, len(frames_rgb) // 24)
    sample = list(frames_rgb[::step][:24])
    # the app's own colours get big swatches, so small coloured buttons keep their exact colour
    ui = [config.BLUE, config.GREEN, config.RED, config.YELLOW, config.ACCENT, config.WHITE, config.BLACK,
          config.DARK, (140, 170, 215)]
    w = sample[0].shape[1]
    swatch = np.zeros((w // len(ui), w, 3), np.uint8)
    for i, bgr in enumerate(ui):
        swatch[:, i * w // len(ui):(i + 1) * w // len(ui)] = bgr[::-1]
    mosaic = Image.fromarray(np.concatenate(sample + [swatch] * 6, axis=0))
    palette = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames_rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default="HELLO 2026")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mp4", help="write the simulated session to this video file")
    ap.add_argument("--gif", help="write a small animated GIF (needs Pillow)")
    ap.add_argument("--gif-width", type=int, default=640, help="width of the GIF frames")
    ap.add_argument("--snapshot", help="save one frame (PNG): the last character written, just before it is read")
    ap.add_argument("--model", help="character model (.npz); default models/char_cnn.npz")
    ap.add_argument("--auto-clear", action="store_true", help="clear each character once read (old behaviour)")
    args = ap.parse_args()

    writer, gif_frames, snap = None, [], {}
    if args.mp4:
        writer = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*"mp4v"), 30, (args.width, args.height))
    gif_size = (args.gif_width, int(round(args.gif_width * args.height / args.width)))

    def on_frame(i, out, app):
        if writer is not None:
            writer.write(out)
        if args.gif and i % 3 == 0:                 # 30 fps -> 10 fps
            small = cv2.resize(out, gif_size, interpolation=cv2.INTER_AREA)
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if args.snapshot and app.canvas.has_pending() and not app.canvas.drawing:
            snap["frame"] = out.copy()             # keeps the last such frame

    recognizer = None
    if args.model:
        from airwriting.recognizer import Recognizer
        recognizer = Recognizer(args.model)
    app = simulate(args.text, args.width, args.height, seed=args.seed, recognizer=recognizer,
                   settings=Settings(keep_ink=not args.auto_clear), frame_callback=on_frame)
    if writer is not None:
        writer.release()
    if args.gif and gif_frames:
        save_gif(args.gif, gif_frames)
    if args.snapshot and "frame" in snap:
        cv2.imwrite(args.snapshot, snap["frame"])
    print(f"typed:      {args.text!r}")
    print(f"recognised: {app.text!r}")
    print("confidence:", " ".join(f"{c}:{p * 100:.0f}%" for c, p in app.history))


if __name__ == "__main__":
    main()
