"""The air-writing application logic, independent of the camera.

`AirWritingApp.step(frame, hand)` takes one video frame plus the detected hand
(or None) and returns the picture to show. Keeping the camera and MediaPipe
outside this class means it can be tested with simulated hands.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .canvas import Canvas
from .config import BLUE, GREEN, OUTPUT_DIR, PALETTE, RED, USER_DATA_DIR, YELLOW, Settings
from .filters import OneEuroFilter
from .gestures import Gesture, GestureDetector
from .recognizer import apply_context
from .ui import HUD, Toolbar, ring

BACKGROUNDS = ["camera", "dim", "blackboard", "whiteboard"]


@dataclass
class ViewState:
    mode: str
    charset: str
    color: tuple
    eraser: bool
    gesture: Gesture
    text: str
    last_prediction: object
    message: str
    message_until: float
    collect_prompt: str = ""
    hint: str = ""           # replaces the gesture name in the status line (e.g. how to press a button)


class Collector:
    """Records your own handwriting samples for fine-tuning (air_writing.py --collect)."""

    def __init__(self, labels: str, per_char: int = 5, folder: Path = USER_DATA_DIR):
        self.labels = [c for c in labels.upper() if c.strip()]
        self.per_char = max(1, per_char)
        self.index, self.count = 0, 0
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.records: list[dict] = []

    @property
    def done(self) -> bool:
        return self.index >= len(self.labels)

    @property
    def label(self) -> str:
        return "" if self.done else self.labels[self.index]

    def prompt(self) -> str:
        if self.done:
            return f"All done - {len(self.records)} samples saved. Press Q."
        return f"Write:  {self.label}     ({self.count + 1}/{self.per_char})    saved {len(self.records)}"

    def _write(self):
        with open(self.path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")

    def save(self, strokes, size):
        if self.done:
            return
        self.records.append({"label": self.label, "frame": list(size), "time": datetime.now().isoformat(timespec="seconds"),
                             "strokes": [[[round(float(x), 1), round(float(y), 1)] for x, y in s] for s in strokes]})
        self._write()
        self.count += 1
        if self.count >= self.per_char:
            self.index, self.count = self.index + 1, 0

    def skip(self):
        if not self.done:
            self.index, self.count = self.index + 1, 0

    def undo(self) -> bool:
        if not self.records:
            return False
        rec = self.records.pop()
        self._write()
        self.index = self.labels.index(rec["label"]) if rec["label"] in self.labels else self.index
        self.count = sum(1 for r in self.records if r["label"] == rec["label"]) % self.per_char
        return True


class AirWritingApp:
    def __init__(self, width: int, height: int, recognizer=None, settings: Settings | None = None,
                 collector: Collector | None = None, start_mode: str | None = None):
        self.cfg = settings or Settings()
        self.w, self.h = width, height
        self.canvas = Canvas(width, height)
        self.toolbar = Toolbar(width, height)
        self.hud = HUD(width, height)
        self.detector = GestureDetector(self.cfg.gesture_stable_frames)
        self.filter = OneEuroFilter(self.cfg.filter_min_cutoff, self.cfg.filter_beta)
        self.recognizer = recognizer
        self.collector = collector
        self.mode = start_mode or ("WRITE" if (recognizer is not None or collector is not None) else "DRAW")
        self.charset_names = list(self.cfg.charsets)
        self.charset = self.charset_names[0]
        self.color, self.eraser, self.brush = BLUE, False, self.cfg.brush_thickness
        self.text = ""
        self.last_prediction = None
        self.flash = None
        self.gesture = Gesture.NONE
        self.pointer = None
        self.last_ink_time = None
        self.hold_start = {Gesture.PALM: None, Gesture.FIST: None}
        self.hold_fired = {Gesture.PALM: False, Gesture.FIST: False}
        self.show_help = False
        self.show_landmarks = self.cfg.show_landmarks
        self.background = self.cfg.background
        self.message, self.message_until = "", 0.0
        self.fps, self._last_t = 0.0, None
        self.now = 0.0                   # app clock (time of the latest frame)
        self.last_screen = None
        self.history: list = []          # every recognised character: (char, confidence)
        # Pointing at the toolbar with one finger: after a button is pressed the pen stays up
        # until the finger is lifted, so moving back down to the canvas doesn't draw a line.
        self.pen_locked = False
        self.pen_down = False
        self._toolbar_draw_since = None
        self._approach = None            # (stroke, time): a line that ran up into the toolbar

    # ------------------------------------------------------------------ helpers
    def say(self, msg: str, seconds: float = 1.6):
        # timed on the app clock, so messages also behave in --video mode and in simulations
        self.message, self.message_until = msg, self.now + seconds

    def state(self) -> ViewState:
        hint = ""
        if self.toolbar.hover_name is not None and self.toolbar.armed:
            hint = "HOLD STILL TO PRESS"
        elif self.pen_locked:
            hint = "PEN UP - show two fingers, then one finger to draw"
        return ViewState(self.mode, self.charset, self.color, self.eraser, self.gesture, self.text,
                         self.last_prediction, self.message, self.message_until,
                         self.collector.prompt() if self.collector else "", hint)

    def _end_stroke(self):
        return self.canvas.end(min_length=self.cfg.min_stroke_length)

    def clear_canvas(self):
        self.canvas.clear()
        self.last_ink_time = None

    def _done_with_ink(self):
        """The pending strokes have been read: keep them on screen, or clear them (--auto-clear)."""
        if self.cfg.keep_ink:
            self.canvas.mark_read()
            self.last_ink_time = None
        else:
            self.clear_canvas()

    def _drop_approach_line(self, now):
        """A button was pressed with the pen down: remove the line drawn on the way up to it."""
        if self._approach is None:
            return
        stroke, t = self._approach
        self._approach = None
        if now - t <= 3.0 and not stroke.read and self.canvas.strokes and self.canvas.strokes[-1] is stroke:
            self.canvas.remove(stroke)
            if not self.canvas.has_pending():
                self.last_ink_time = None

    # ------------------------------------------------------------------ main step
    def step(self, frame: np.ndarray, hand, now: float | None = None) -> np.ndarray:
        now = time.monotonic() if now is None else now
        if self._last_t is not None:
            dt = max(now - self._last_t, 1e-6)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self._last_t = self.now = now

        lm = hand.landmarks if hand is not None else None
        g = self.detector.update(lm)
        self.gesture = g
        if lm is not None:
            self.pointer = self.filter(lm[8, :2], now)
        else:
            self.pointer = None
            self.filter.reset()
        pt = self.pointer
        in_toolbar = pt is not None and pt[1] < self.toolbar.height
        in_ui = in_toolbar or (pt is not None and pt[1] > self.hud.top)

        # one finger kept on the toolbar: lock the pen until the finger is lifted
        if g != Gesture.DRAW:
            self.pen_locked = False
        if g == Gesture.DRAW and in_toolbar:
            if self._toolbar_draw_since is None:
                self._toolbar_draw_since = now
            elif now - self._toolbar_draw_since >= 0.3:
                self.pen_locked = True
        else:
            self._toolbar_draw_since = None

        # drawing / erasing
        pen_down = g == Gesture.DRAW and pt is not None and not in_ui and not self.pen_locked
        self.pen_down = pen_down
        if pen_down and self.eraser:
            self._end_stroke()
            if self.canvas.erase(pt, self.cfg.eraser_radius):
                self.last_ink_time = now
        elif pen_down:
            if not self.canvas.drawing:
                self.canvas.begin(pt, self.color, self.brush)
            else:
                last = np.asarray(self.canvas.current.points[-1])
                if np.linalg.norm(pt - last) > self.cfg.max_jump_ratio * np.hypot(self.w, self.h):
                    self._end_stroke()                    # tracking glitch: don't draw a long line
                    self.canvas.begin(pt, self.color, self.brush)
                else:
                    self.canvas.add(pt)
            self.last_ink_time = now
        elif self.canvas.drawing:
            s = self._end_stroke()
            if s is not None and in_toolbar:
                self._approach = (s, now)             # this line ran up into the toolbar

        # toolbar buttons: point at one with one or two fingers and hold still. Any pointing
        # gesture counts, so a flicker between one and two fingers doesn't restart the hold.
        pointing = g in (Gesture.DRAW, Gesture.HOVER, Gesture.OTHER)
        pressed = self.toolbar.update(pt, pointing, now, self.cfg.dwell_seconds)
        if pressed is not None:
            if g == Gesture.DRAW:
                self._drop_approach_line(now)
                self.pen_locked = True
            self.press(pressed.name)

        self._update_holds(g, now)

        # read the character after a pause
        if (self.mode == "WRITE" and not self.canvas.drawing and self.last_ink_time is not None
                and not pen_down and now - self.last_ink_time >= self.cfg.recognize_delay):
            self.recognize(now)

        out = self.render(frame, lm, now)
        self.last_screen = out
        return out

    def _update_holds(self, g, now):
        for hg, seconds in ((Gesture.PALM, self.cfg.palm_hold_seconds), (Gesture.FIST, self.cfg.fist_hold_seconds)):
            if g != hg:
                self.hold_start[hg] = None
                continue
            if self.hold_start[hg] is None:
                self.hold_start[hg], self.hold_fired[hg] = now, False
            if not self.hold_fired[hg] and now - self.hold_start[hg] >= seconds:
                self.hold_fired[hg] = True
                if hg == Gesture.PALM:
                    if not self.canvas.is_empty():
                        self.clear_canvas()
                        self.say("Canvas cleared")
                elif self.canvas.has_pending():
                    self.recognize(now)

    # ------------------------------------------------------------------ actions
    def recognize(self, now: float | None = None):
        now = self.now if now is None else now
        self._end_stroke()
        strokes = self.canvas.pending_arrays()        # only what was written since the last read
        self.last_ink_time = None
        if not strokes:
            return []
        if self.collector is not None:
            label = self.collector.label
            if label:
                self.collector.save(strokes, (self.w, self.h))
                lo, hi = np.vstack(strokes).min(axis=0), np.vstack(strokes).max(axis=0)
                self.flash = (label, (*lo, *hi), now, GREEN)
                self.say(f"Saved '{label}'", 1.0)
            self.clear_canvas()
            return []
        if self.recognizer is None:
            return []
        preds = self.recognizer.recognize(strokes, allowed=self.cfg.charsets[self.charset],
                                          min_size=self.cfg.min_char_size_ratio * self.h,
                                          segment=self.cfg.segment_characters)
        if not preds:
            self.canvas.discard_pending()             # only marks too small to be characters
            self.say("Too small - write bigger")
            return []
        self._done_with_ink()
        if self.charset == self.charset_names[0]:          # letters and digits mixed
            preds = apply_context(preds, self.text[-1:])
        chars = "".join(p.char for p in preds)
        self.text += chars
        self.history.extend((p.char, p.confidence) for p in preds)
        self.last_prediction = preds[-1]
        conf = min(p.confidence for p in preds)
        color = GREEN if conf >= 0.8 else (YELLOW if conf >= 0.5 else RED)
        boxes = np.array([p.box for p in preds])
        self.flash = (chars, (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()), now, color)
        return preds

    def press(self, name: str):
        if name in PALETTE:
            self.color, self.eraser = PALETTE[name], False
        elif name == "eraser":
            self.eraser = not self.eraser
        elif name == "clear":
            self.clear_canvas()
            self.say("Canvas cleared")
        elif name == "space":
            if self.mode == "WRITE" and self.collector is None and self.canvas.has_pending():
                self.recognize()             # read the pending character first, then add the space
            self.text += " "
        elif name == "delete":
            if self.collector is not None:
                if self.collector.undo():
                    self.say("Removed the last sample")
            elif self.mode == "WRITE" and self.canvas.has_pending():
                self.canvas.discard_pending()   # cancel the character being written
                self.last_ink_time = None
                self.say("Cancelled")
            else:
                self.text = self.text[:-1]
        elif name == "charset":
            self.charset = self.charset_names[(self.charset_names.index(self.charset) + 1) % len(self.charset_names)]
            self.say(f"Recognising {self.charset}")
        elif name == "mode":
            if self.recognizer is None and self.collector is None:
                self.say("No character model loaded - drawing only")
                return
            self.mode = "DRAW" if self.mode == "WRITE" else "WRITE"
            self.last_ink_time = None
            self.say("Write mode: characters become text" if self.mode == "WRITE" else "Draw mode: free drawing")

    def handle_key(self, key: int) -> bool:
        """Keyboard shortcuts. Returns False when the app should quit."""
        if key < 0 or key == 255:
            return True
        ch = chr(key).lower() if 32 <= key < 127 else ""
        if key == 27 or ch == "q":
            return False
        if key in (13, 10):
            if not self.canvas.has_pending():
                self.say("Nothing to read yet")
            else:
                self.recognize()
        elif key in (8, 127):
            self.press("delete")
        elif ch == " ":
            self.press("space")
        elif ch == "x":
            self.text = ""
            self.say("Text cleared")
        elif ch == "c":
            self.press("clear")
        elif ch == "u":
            self.canvas.undo()
        elif ch == "m":
            self.press("mode")
        elif ch == "k":
            self.press("charset")
        elif ch in "1234":
            self.press(["blue", "green", "red", "yellow"][int(ch) - 1])
        elif ch == "e":
            self.press("eraser")
        elif ch in "+=":
            self.brush = min(40, self.brush + 2)
            self.say(f"Brush {self.brush}px", 0.8)
        elif ch == "-":
            self.brush = max(2, self.brush - 2)
            self.say(f"Brush {self.brush}px", 0.8)
        elif ch == "b":
            self.background = BACKGROUNDS[(BACKGROUNDS.index(self.background) + 1) % len(BACKGROUNDS)]
            self.say(f"Background: {self.background}", 1.0)
        elif ch == "l":
            self.show_landmarks = not self.show_landmarks
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "s":
            self.save()
        elif ch == "n" and self.collector is not None:
            self.collector.skip()
            self.clear_canvas()
        return True

    def save(self) -> dict:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}
        if not self.canvas.is_empty():
            files["drawing"] = OUTPUT_DIR / f"drawing_{stamp}.png"
            cv2.imwrite(str(files["drawing"]), self.canvas.ink_image())
        if self.last_screen is not None:
            files["screen"] = OUTPUT_DIR / f"screen_{stamp}.png"
            cv2.imwrite(str(files["screen"]), self.last_screen)
        if self.text:
            files["text"] = OUTPUT_DIR / f"text_{stamp}.txt"
            files["text"].write_text(self.text + "\n", encoding="utf-8")
        self.say("Saved to outputs/ folder" if files else "Nothing to save yet")
        return files

    # ------------------------------------------------------------------ drawing
    def _background(self, frame):
        if self.background == "dim":
            return (frame * 0.4).astype(np.uint8)
        if self.background == "blackboard":
            return np.full_like(frame, (32, 36, 34))
        if self.background == "whiteboard":
            return np.full_like(frame, (245, 245, 245))
        return frame.copy()

    def render(self, frame, lm, now):
        out = self._background(frame)
        self.canvas.composite(out)
        s = self.toolbar.s
        if self.show_landmarks and lm is not None:
            self.hud.draw_landmarks(out, lm)
        st = self.state()
        self.toolbar.draw(out, st, now, self.cfg.dwell_seconds)
        self.hud.draw_status(out, self.toolbar.height + int(28 * s), st, self.fps, now)
        self.hud.draw_text_panel(out, st, now)

        pt = self.pointer
        if (self.mode == "WRITE" and self.last_ink_time is not None and not self.canvas.drawing
                and self.canvas.has_pending() and not self.pen_down):
            progress = (now - self.last_ink_time) / self.cfg.recognize_delay
            anchor = pt
            if anchor is None:
                p = np.vstack(self.canvas.pending_arrays())
                anchor = (p[:, 0].max() + 30 * s, p[:, 1].min())
            ring(out, anchor, 26 * s, progress, (255, 190, 60), max(2, int(4 * s)))
        if lm is not None:
            for hg, seconds, color in ((Gesture.PALM, self.cfg.palm_hold_seconds, RED),
                                       (Gesture.FIST, self.cfg.fist_hold_seconds, GREEN)):
                t0 = self.hold_start.get(hg)
                # the palm clears everything on screen; the fist reads what is not read yet
                target = not self.canvas.is_empty() if hg == Gesture.PALM else self.canvas.has_pending()
                if t0 is not None and not self.hold_fired[hg] and target:
                    ring(out, lm[9, :2], 50 * s, (now - t0) / seconds, color, max(3, int(6 * s)))
        # outside the canvas, or with the pen locked, the cursor shows "pen up"
        cursor_gesture = Gesture.HOVER if (self.gesture == Gesture.DRAW and not self.pen_down) else self.gesture
        self.hud.draw_cursor(out, pt, cursor_gesture, self.color, self.eraser, self.cfg.eraser_radius, self.brush)
        self.hud.draw_flash(out, self.flash, now)
        if self.show_help:
            self.hud.draw_help(out)
        return out
