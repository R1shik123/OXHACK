"""Per-responder movement history and movement features.

Responder *activity* (stationary / walking / running) comes from Apple's pre-trained Core Motion
classifier on the phone, and falls from the phone's accelerometer (free fall + impact) plus the
height-drop rule in twin.Vitals. This module only keeps the recent track and computes a few plain
physical measures from it, e.g. whether the phone is lying perfectly still (put down / dropped).
"""

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np

RATE = 5.0  # Hz, features are computed on a resampled 5 Hz track
WIN = 6.0  # seconds of history the short-term features look at
FEATURES = ["speed_mean", "speed_max", "rel_height", "drop_3s", "y_std_1s", "xz_std", "turn_rate",
            "disp_10s", "disp_30s", "tremor", "vy_min"]


def resample(samples: list[tuple], t_end: float, seconds: float) -> np.ndarray | None:
    """samples: (t, x, y, z, yaw). Returns an (n, 5) array on a regular RATE grid ending at t_end."""
    if len(samples) < 4:
        return None
    a = np.array(samples, float)
    t0 = t_end - seconds
    if a[0, 0] > t_end - 8.0:  # need at least 8 s of movement before classifying
        return None
    if a[0, 0] > t0:  # shorter history than the window: hold the first sample (as if they stood there)
        a = np.vstack([np.r_[t0, a[0, 1:]][None], a])
    grid = np.arange(t0, t_end + 1e-9, 1 / RATE)
    out = np.empty((len(grid), 5))
    out[:, 0] = grid
    for c in range(1, 4):
        out[:, c] = np.interp(grid, a[:, 0], a[:, c])
    out[:, 4] = np.interp(grid, a[:, 0], np.unwrap(a[:, 4]))
    return out


def features(track: np.ndarray, baseline_eye: float) -> np.ndarray:
    """track: 30 s at RATE Hz (t, x, y, z, yaw)."""
    n_short = int(WIN * RATE)
    s = track[-n_short:]
    xz = s[:, [1, 3]]
    v = np.linalg.norm(np.diff(xz, axis=0), axis=1) * RATE
    y = s[:, 2]
    last1 = s[-int(RATE):]
    k3 = int(3 * RATE)
    drop = (y[-k3:-int(RATE)].max() if len(y) > k3 else y.max()) - y[-4:].mean()
    yaw = s[:, 4]
    turn = np.abs(np.diff(yaw)).sum() / WIN
    disp10 = np.linalg.norm(track[-1, [1, 3]] - track[-int(10 * RATE), [1, 3]])
    disp30 = np.linalg.norm(track[-1, [1, 3]] - track[0, [1, 3]])
    acc = np.diff(s[-int(2 * RATE):, 1:4], n=2, axis=0)  # second differences ~ hand tremor
    tremor = float(np.abs(acc).mean()) * 1000  # mm per step^2
    vy = np.diff(y) * RATE
    return np.array([v.mean(), v.max(), y[-1] - baseline_eye, drop, last1[:, 2].std(), xz.std(axis=0).sum(),
                     turn, disp10, disp30, tremor, vy.min()], float)


class Tracker:
    """Per-responder motion history + standing-eye-height baseline, feeding the model."""

    def __init__(self) -> None:
        self.samples: deque = deque(maxlen=400)  # (t, x, y, z, yaw) at up to 10 Hz, ~40 s
        self.eye_hist: deque = deque(maxlen=300)

    def add(self, t: float, pos: list[float], fwd: list[float] | None) -> None:
        yaw = math.atan2(fwd[0], fwd[2]) if fwd else (self.samples[-1][4] if self.samples else 0.0)
        self.samples.append((t, pos[0], pos[1], pos[2], yaw))
        if len(self.samples) > 10:
            prev = self.samples[-10]
            if math.dist((prev[1], prev[3]), (pos[0], pos[2])) > 0.3:  # moving -> upright: learn eye height
                self.eye_hist.append(pos[1])

    def baseline(self) -> float | None:
        if len(self.eye_hist) >= 10:
            return float(np.median(self.eye_hist))
        return float(np.median([s[2] for s in self.samples])) if self.samples else None

    def features(self, t_end: float) -> np.ndarray | None:
        tr = resample(list(self.samples), t_end, 30.0)
        base = self.baseline()
        return None if tr is None or base is None else features(tr, base)


def set_down(feats: np.ndarray | None) -> bool:
    """No hand tremor, no height wobble and no movement for the last few seconds: the phone isn't held."""
    if feats is None:
        return False
    f = dict(zip(FEATURES, feats))
    return bool(f["tremor"] < 0.35 and f["y_std_1s"] < 0.002 and f["speed_max"] < 0.05)
