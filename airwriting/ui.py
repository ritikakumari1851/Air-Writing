"""Everything that is drawn on top of the video: toolbar, text bar, cursor, hints."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ACCENT, BLACK, DARK, GREEN, PALETTE, RED, WHITE, YELLOW
from .gestures import HAND_CONNECTIONS, Gesture

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_PLAIN = cv2.FONT_HERSHEY_SIMPLEX


def blend_rect(img, x0, y0, x1, y1, color, alpha):
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


_OUTLINE = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]


def text(img, s, org, scale, color=WHITE, thick=1, font=FONT, outline=True):
    x, y = int(org[0]), int(org[1])
    if outline:
        # same-thickness copies around the text: a thicker stroke changes letter
        # spacing in OpenCV 5's text renderer, so it cannot be used for the outline
        for dx, dy in _OUTLINE:
            cv2.putText(img, s, (x + dx, y + dy), font, scale, BLACK, thick, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)


def text_size(s, scale, thick=1, font=FONT):
    (w, h), base = cv2.getTextSize(s, font, scale, thick)
    return w, h, base


def ring(img, center, radius, progress, color, thick=4):
    """Circular progress indicator, progress in [0, 1]."""
    c = (int(center[0]), int(center[1]))
    cv2.circle(img, c, int(radius), (80, 80, 80), max(1, thick // 2), cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(img, c, (int(radius), int(radius)), -90, 0, 360 * min(1.0, progress), color, thick, cv2.LINE_AA)


@dataclass
class Button:
    name: str
    label: str
    rect: tuple          # x0, y0, x1, y1
    color: tuple | None = None

    def contains(self, pt) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


class Toolbar:
    NAMES = [("blue", ""), ("green", ""), ("red", ""), ("yellow", ""), ("eraser", "ERASE"),
             ("clear", "CLEAR"), ("space", "SPACE"), ("delete", "DEL"), ("charset", "A-Z 0-9"), ("mode", "WRITE")]

    def __init__(self, width: int, height: int):
        self.s = height / 720.0
        self.height = int(78 * self.s)
        pad = int(8 * self.s)
        n = len(self.NAMES)
        bw = (width - pad * (n + 1)) / n
        self.buttons = []
        for i, (name, label) in enumerate(self.NAMES):
            x0 = pad + i * (bw + pad)
            self.buttons.append(Button(name, label, (int(x0), pad, int(x0 + bw), self.height - pad), PALETTE.get(name)))
        self.hover_name = None
        self.hover_since = 0.0
        self.armed = True        # a button fires once per visit

    def button_at(self, pt):
        if pt is None:
            return None
        for b in self.buttons:
            if b.contains(pt):
                return b
        return None

    def update(self, pt, pointing: bool, now: float, dwell: float):
        """Return the button pressed this frame (point at it and hold still), if any."""
        b = self.button_at(pt) if pointing else None
        name = b.name if b else None
        if name != self.hover_name:
            self.hover_name, self.hover_since, self.armed = name, now, True
            return None
        if b is not None and self.armed and now - self.hover_since >= dwell:
            self.armed = False
            return b
        return None

    def progress(self, now: float, dwell: float) -> float:
        if self.hover_name is None or not self.armed:
            return 0.0
        return min(1.0, (now - self.hover_since) / dwell)

    def draw(self, img, state, now, dwell):
        s = self.s
        blend_rect(img, 0, 0, img.shape[1], self.height, DARK, 0.72)
        for b in self.buttons:
            x0, y0, x1, y1 = b.rect
            active = (b.color is not None and state.color == b.color and not state.eraser) or \
                     (b.name == "eraser" and state.eraser)
            hovered = b.name == self.hover_name
            blend_rect(img, x0, y0, x1, y1, (70, 70, 70) if hovered else (50, 50, 50), 0.85)
            cv2.rectangle(img, (x0, y0), (x1, y1), ACCENT if active else (110, 110, 110), 2 if active else 1, cv2.LINE_AA)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if b.color is not None:
                cv2.circle(img, (cx, cy), int(min(x1 - x0, y1 - y0) * 0.32), b.color, -1, cv2.LINE_AA)
                if active:
                    cv2.circle(img, (cx, cy), int(min(x1 - x0, y1 - y0) * 0.42), WHITE, 2, cv2.LINE_AA)
            else:
                label = b.label
                if b.name == "mode":
                    label = state.mode
                elif b.name == "charset":
                    label = state.charset
                sc = 0.55 * s
                tw, th, _ = text_size(label, sc)
                while tw > (x1 - x0) - 8 and sc > 0.3:
                    sc -= 0.05
                    tw, th, _ = text_size(label, sc)
                text(img, label, (cx - tw / 2, cy + th / 2), sc, ACCENT if active else WHITE, 1, outline=False)
            if hovered:
                p = self.progress(now, dwell)
                if p > 0:
                    cv2.rectangle(img, (x0, y1 - int(5 * s)), (int(x0 + (x1 - x0) * p), y1), ACCENT, -1)


class HUD:
    """Bottom text bar, status line, cursor, hand skeleton and overlays."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.s = height / 720.0
        self.panel_h = int(110 * self.s)
        self.top = height - self.panel_h

    # -- pieces --------------------------------------------------------------
    def draw_landmarks(self, img, lm):
        pts = [(int(p[0]), int(p[1])) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (200, 200, 200), max(1, int(2 * self.s)), cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(img, p, max(2, int(4 * self.s)), (0, 200, 255) if i in (4, 8, 12, 16, 20) else (255, 255, 255), -1, cv2.LINE_AA)

    def draw_cursor(self, img, pt, gesture, color, eraser, eraser_radius, brush):
        if pt is None:
            return
        p = (int(pt[0]), int(pt[1]))
        if eraser and gesture == Gesture.DRAW:
            cv2.circle(img, p, int(eraser_radius), WHITE, 2, cv2.LINE_AA)
            cv2.circle(img, p, 3, WHITE, -1, cv2.LINE_AA)
        elif gesture == Gesture.DRAW:
            cv2.circle(img, p, max(4, brush // 2 + 3), color, -1, cv2.LINE_AA)
            cv2.circle(img, p, max(4, brush // 2 + 3), WHITE, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, p, int(12 * self.s), WHITE, 2, cv2.LINE_AA)
            cv2.circle(img, p, 3, WHITE, -1, cv2.LINE_AA)

    def draw_status(self, img, y, state, fps, now):
        s = self.s
        labels = {Gesture.DRAW: "DRAWING", Gesture.HOVER: "MOVING", Gesture.PALM: "OPEN PALM - hold to clear",
                  Gesture.FIST: "FIST - hold to read now", Gesture.OTHER: "PEN UP", Gesture.NONE: "show your hand"}
        msg = labels.get(state.gesture, "")
        if state.eraser and state.gesture == Gesture.DRAW:
            msg = "ERASING"
        if state.hint:
            msg = state.hint
        text(img, msg, (int(12 * s), y), 0.6 * s, ACCENT if state.gesture == Gesture.DRAW else WHITE, 1)
        right = f"{state.mode}  |  {state.charset}  |  {fps:4.1f} fps"
        tw, _, _ = text_size(right, 0.5 * s)
        text(img, right, (self.w - tw - int(12 * s), y), 0.5 * s, (220, 220, 220), 1)
        if state.message and now < state.message_until:
            tw, _, _ = text_size(state.message, 0.7 * s)
            text(img, state.message, ((self.w - tw) / 2, y + int(34 * s)), 0.7 * s, YELLOW, 1)

    def draw_text_panel(self, img, state, now):
        s, top = self.s, self.top
        blend_rect(img, 0, top, self.w, self.h, DARK, 0.78)
        pad = int(14 * s)
        preview = int(84 * s)
        right_w = preview + int(230 * s)
        if state.collect_prompt:
            text(img, "COLLECT MODE", (pad, top + int(30 * s)), 0.55 * s, ACCENT, 1)
            text(img, state.collect_prompt, (pad, top + int(78 * s)), 1.1 * s, WHITE, 2)
        else:
            text(img, "TEXT", (pad, top + int(26 * s)), 0.5 * s, ACCENT, 1)
            shown = state.text.replace("\n", " ")
            sc = 1.25 * s
            max_w = self.w - right_w - 2 * pad
            while shown and text_size(shown + "_", sc, 2)[0] > max_w:
                shown = shown[1:]
            if shown != state.text and shown:
                shown = "..." + shown[3:]
            cursor = "_" if int(now * 2) % 2 == 0 else " "
            text(img, shown + cursor, (pad, top + int(80 * s)), sc, WHITE, 2)

        # last prediction (big letter, confidence, alternatives) and what the network saw
        x = self.w - right_w
        p = state.last_prediction
        if p is not None:
            col = GREEN if p.confidence >= 0.8 else (YELLOW if p.confidence >= 0.5 else RED)
            text(img, p.char, (x + int(6 * s), top + int(84 * s)), 2.2 * s, col, 3)
            text(img, f"{p.confidence * 100:.0f}%", (x + int(70 * s), top + int(48 * s)), 0.7 * s, col, 1)
            alt = "   ".join(f"{c}: {q * 100:.0f}%" for c, q in p.alternatives if q >= 0.01)
            if getattr(p, "by_context", False):
                alt = "chosen from context"
            if alt:
                text(img, alt, (x + int(70 * s), top + int(80 * s)), 0.45 * s, (200, 200, 200), 1)
            if p.image is not None:
                im = cv2.resize((p.image * 255).astype(np.uint8), (preview, preview), interpolation=cv2.INTER_NEAREST)
                y0, x0 = top + (self.panel_h - preview) // 2, self.w - preview - pad
                img[y0:y0 + preview, x0:x0 + preview] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + preview, y0 + preview), (120, 120, 120), 1)
        else:
            text(img, "write a letter or digit", (x + int(6 * s), top + int(60 * s)), 0.5 * s, (170, 170, 170), 1)

    def draw_flash(self, img, flash, now):
        if not flash:
            return
        chars, box, t0, color = flash
        age = now - t0
        if age > 0.9:
            return
        x0, y0, x1, y1 = box
        h = max(40.0, y1 - y0)
        scale = h / 30.0
        tw, th, _ = text_size(chars, scale, 4)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        layer = img.copy()
        text(layer, chars, (cx - tw / 2, cy + th / 2), scale, color, max(2, int(scale * 1.5)), outline=False)
        a = 1.0 - age / 0.9
        cv2.addWeighted(layer, a, img, 1 - a, 0, dst=img)

    def draw_help(self, img):
        s = self.s
        lines = [
            ("GESTURES", None),
            ("index finger up", "draw (pen down)"),
            ("index + middle up", "move without drawing"),
            ("point at a button", "hold still ~0.5 s to press it"),
            ("fist (hold)", "read the character now"),
            ("open palm (hold)", "clear the screen"),
            ("", ""),
            ("KEYS", None),
            ("Enter", "read now          Space  add space"),
            ("Backspace", "delete letter     X  clear text"),
            ("C / U", "clear canvas / undo stroke"),
            ("M / K", "write-draw mode / letters-digits"),
            ("1-4 / E", "colour / eraser     + -  brush size"),
            ("B / L", "background / hand skeleton"),
            ("S", "save picture and text to outputs/"),
            ("H / Q", "this help / quit"),
        ]
        w, h = int(620 * s), int((len(lines) * 30 + 40) * s)
        x0, y0 = (self.w - w) // 2, (self.h - h) // 2
        blend_rect(img, x0, y0, x0 + w, y0 + h, (20, 20, 20), 0.88)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), ACCENT, 1)
        y = y0 + int(34 * s)
        for a, b in lines:
            if b is None:
                text(img, a, (x0 + int(20 * s), y), 0.62 * s, ACCENT, 1)
            elif a or b:
                text(img, a, (x0 + int(20 * s), y), 0.52 * s, WHITE, 1, outline=False)
                text(img, b, (x0 + int(220 * s), y), 0.52 * s, (210, 210, 210), 1, font=FONT_PLAIN, outline=False)
            y += int(30 * s)
