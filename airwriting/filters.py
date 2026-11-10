"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012) for the fingertip position.

It smooths a lot when the finger moves slowly (no jitter) and very little
when it moves fast (no lag), which is exactly what writing in the air needs.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = None
        self.t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = _alpha(dt, cutoff)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()
