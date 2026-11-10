"""The drawing surface: strokes are kept as vectors (for recognition, undo and
the eraser) and also painted onto a colour layer that is blended over the video."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

POINT_SPACING = 3.0   # px between stored points, so the eraser cuts precisely


@dataclass(eq=False)          # strokes are compared by identity
class Stroke:
    color: tuple
    thickness: int
    points: list = field(default_factory=list)   # [(x, y), ...] floats
    read: bool = False                           # already turned into text (stays on screen as ink)

    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        p = np.asarray(self.points, np.float32)
        return float(np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1)).sum())

    def array(self) -> np.ndarray:
        return np.asarray(self.points, np.float32).reshape(-1, 2)


class Canvas:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.strokes: list[Stroke] = []
        self.current: Stroke | None = None
        self.color_layer = np.zeros((height, width, 3), np.uint8)
        self.alpha = np.zeros((height, width), np.uint8)
        self._bbox = None       # region that contains ink: [x0, y0, x1, y1]

    # -- drawing -------------------------------------------------------------
    def begin(self, pt, color, thickness):
        self.end()
        self.current = Stroke(tuple(int(c) for c in color), int(thickness), [tuple(map(float, pt))])
        self.strokes.append(self.current)
        self._paint_dot(pt, self.current)

    def add(self, pt):
        if self.current is None:
            return
        last = np.asarray(self.current.points[-1], np.float32)
        pt = np.asarray(pt, np.float32)
        dist = float(np.linalg.norm(pt - last))
        if dist < 1.0:
            return
        n = max(1, int(np.ceil(dist / POINT_SPACING)))
        for k in range(1, n + 1):
            self.current.points.append(tuple(map(float, last + (pt - last) * (k / n))))
        self._paint_segment(last, pt, self.current)

    def end(self, min_length: float = 0.0):
        """Finish the current stroke. Strokes shorter than min_length are discarded."""
        s, self.current = self.current, None
        if s is not None and min_length > 0 and s.length() < min_length:
            self.strokes.remove(s)
            self.redraw()
            return None
        return s

    @property
    def drawing(self) -> bool:
        return self.current is not None

    # -- editing ---------------------------------------------------------------
    def erase(self, pt, radius: float) -> bool:
        """Remove ink within `radius` of pt. Returns True if anything was erased."""
        self.end()
        c = np.asarray(pt, np.float32)
        changed, kept = False, []
        for s in self.strokes:
            p = s.array()
            if len(p) == 0:
                continue
            inside = np.sqrt(((p - c) ** 2).sum(axis=1)) <= radius + s.thickness / 2
            if not inside.any():
                kept.append(s)
                continue
            changed = True
            run = []
            for point, gone in zip(s.points, inside):    # split into the pieces that survive
                if gone:
                    if len(run) >= 2:
                        kept.append(Stroke(s.color, s.thickness, run, s.read))
                    run = []
                else:
                    run.append(point)
            if len(run) >= 2:
                kept.append(Stroke(s.color, s.thickness, run, s.read))
        if changed:
            self.strokes = kept
            self.redraw()
        return changed

    def undo(self) -> bool:
        self.end()
        if not self.strokes:
            return False
        self.strokes.pop()
        self.redraw()
        return True

    def clear(self):
        self.current = None
        self.strokes = []
        self.color_layer[:] = 0
        self.alpha[:] = 0
        self._bbox = None

    def remove(self, stroke: Stroke) -> bool:
        """Delete one stroke (e.g. the line drawn on the way up to the toolbar)."""
        if self.current is stroke:
            self.current = None
        n = len(self.strokes)
        self.strokes = [s for s in self.strokes if s is not stroke]
        if len(self.strokes) != n:
            self.redraw()
        return len(self.strokes) != n

    def is_empty(self) -> bool:
        """True if there is no ink at all on the screen."""
        return not any(len(s.points) for s in self.strokes)

    def has_pending(self) -> bool:
        """True if there is ink that has not been read as text yet."""
        return any(len(s.points) and not s.read for s in self.strokes)

    def stroke_arrays(self) -> list[np.ndarray]:
        return [s.array() for s in self.strokes if len(s.points) >= 1]

    def pending_arrays(self) -> list[np.ndarray]:
        """The strokes written since the last character was read."""
        return [s.array() for s in self.strokes if len(s.points) >= 1 and not s.read]

    def mark_read(self):
        """Everything on the canvas has been read: keep it on screen, but don't read it again."""
        self.end()
        for s in self.strokes:
            s.read = True

    def discard_pending(self) -> bool:
        """Remove only the strokes that have not been read yet (cancel the character being written)."""
        self.end()
        keep = [s for s in self.strokes if s.read]
        if len(keep) == len(self.strokes):
            return False
        self.strokes = keep
        self.redraw()
        return True

    # -- rendering -----------------------------------------------------------
    def _grow_bbox(self, x0, y0, x1, y1):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.w, int(np.ceil(x1))), min(self.h, int(np.ceil(y1)))
        if self._bbox is None:
            self._bbox = [x0, y0, x1, y1]
        else:
            b = self._bbox
            self._bbox = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]

    def _paint_segment(self, a, b, s: Stroke):
        p0 = (int(round(a[0])), int(round(a[1])))
        p1 = (int(round(b[0])), int(round(b[1])))
        cv2.line(self.color_layer, p0, p1, s.color, s.thickness + 2, cv2.LINE_8)
        cv2.line(self.alpha, p0, p1, 255, s.thickness, cv2.LINE_AA)
        r = s.thickness
        self._grow_bbox(min(p0[0], p1[0]) - r, min(p0[1], p1[1]) - r, max(p0[0], p1[0]) + r, max(p0[1], p1[1]) + r)

    def _paint_dot(self, pt, s: Stroke):
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(self.color_layer, p, s.thickness // 2 + 1, s.color, -1, cv2.LINE_8)
        cv2.circle(self.alpha, p, max(1, s.thickness // 2), 255, -1, cv2.LINE_AA)
        r = s.thickness
        self._grow_bbox(p[0] - r, p[1] - r, p[0] + r, p[1] + r)

    def redraw(self):
        self.color_layer[:] = 0
        self.alpha[:] = 0
        self._bbox = None
        for s in self.strokes:
            if not s.points:
                continue
            self._paint_dot(s.points[0], s)
            pts = np.round(np.asarray(s.points)).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(self.color_layer, [pts], False, s.color, s.thickness + 2, cv2.LINE_8)
                cv2.polylines(self.alpha, [pts], False, 255, s.thickness, cv2.LINE_AA)
                r = s.thickness
                self._grow_bbox(pts[:, 0].min() - r, pts[:, 1].min() - r, pts[:, 0].max() + r, pts[:, 1].max() + r)

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Blend the ink over `frame` (in place) and return it."""
        if self._bbox is None:
            return frame
        x0, y0, x1, y1 = self._bbox
        if x1 <= x0 or y1 <= y0:
            return frame
        a = self.alpha[y0:y1, x0:x1].astype(np.float32)[..., None] * (1.0 / 255.0)
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        ink = self.color_layer[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (roi + (ink - roi) * a).astype(np.uint8)
        return frame

    def ink_image(self, background=(255, 255, 255)) -> np.ndarray:
        """The drawing alone on a plain background (used when saving)."""
        img = np.full((self.h, self.w, 3), background, np.uint8)
        saved = self._bbox
        self._bbox = [0, 0, self.w, self.h]
        out = self.composite(img)
        self._bbox = saved
        return out
