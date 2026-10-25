"""Per-phone fine alignment: removes the small rotation / offset between phones that makes walls appear twice.

Every phone locks onto the same entrance marker, but a degree of yaw error there becomes ~17 cm of offset
10 m away. The laptop compares each phone's AR wall pieces with the reference phone's (the first one with
enough walls) and estimates a small rigid correction on the floor plane: a rotation about the marker plus a
shift. It is applied to everything that phone sends afterwards (poses, planes, points, depth frames).

Conservative on purpose: needs >= 3 matching wall pairs and >= 3 m of overlap, moves at most half the
estimated error per update (so it converges smoothly instead of jumping), and caps each step.
"""

from __future__ import annotations

import math

import numpy as np

MIN_PAIRS, MIN_OVERLAP = 3, 3.0
MAX_STEP_DEG, MAX_STEP_M = 2.0, 0.25
DAMPING = 0.5


def apply(corr: tuple[float, float, float], x: float, z: float) -> tuple[float, float]:
    th, tx, tz = corr
    c, s = math.cos(th), math.sin(th)
    return c * x - s * z + tx, s * x + c * z + tz


def matrix(corr: tuple[float, float, float]) -> np.ndarray:
    """4x4 transform (rotation about the vertical axis + floor shift) for poses."""
    th, tx, tz = corr
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, -s, tx], [0, 1, 0, 0], [s, 0, c, tz], [0, 0, 0, 1]], dtype=np.float64)


def compose(inc: tuple[float, float, float], old: tuple[float, float, float]) -> tuple[float, float, float]:
    """First `old`, then `inc`."""
    th = old[0] + inc[0]
    tx, tz = apply((inc[0], 0.0, 0.0), old[1], old[2])
    return th, tx + inc[1], tz + inc[2]


def estimate(own: list, ref: list) -> tuple[float, float, float] | None:
    """Correction that moves `own` wall segments ((x1, z1), (x2, z2)) onto `ref` ones, or None if unsure."""
    def pairs(segs):
        out = []
        for p, q in segs:
            ang = math.atan2(q[1] - p[1], q[0] - p[0])
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            best = None
            for rp, rq in ref:
                L = math.dist(rp, rq)
                if L < 0.5:
                    continue
                rang = math.atan2(rq[1] - rp[1], rq[0] - rp[0])
                d = (rang - ang + math.pi / 2) % math.pi - math.pi / 2
                if abs(d) > math.radians(6):
                    continue
                ux, uz = (rq[0] - rp[0]) / L, (rq[1] - rp[1]) / L
                off = (mid[0] - rp[0]) * -uz + (mid[1] - rp[1]) * ux
                ta = (p[0] - rp[0]) * ux + (p[1] - rp[1]) * uz
                tb = (q[0] - rp[0]) * ux + (q[1] - rp[1]) * uz
                overlap = min(max(ta, tb), L) - max(min(ta, tb), 0)
                if abs(off) < 0.6 and overlap > 0.3 and (best is None or abs(off) < abs(best[1])):
                    best = (d, off, overlap, (-uz, ux))
            if best:
                out.append(best)
        return out

    found = pairs(own)
    if len(found) < MIN_PAIRS or sum(f[2] for f in found) < MIN_OVERLAP:
        return None
    order = sorted(found, key=lambda f: f[0])  # weighted median rotation
    half, acc, dth = sum(f[2] for f in order) / 2, 0.0, 0.0
    for f in order:
        acc += f[2]
        if acc >= half:
            dth = f[0]
            break
    rot = [((apply((dth, 0, 0), *p), apply((dth, 0, 0), *q))) for p, q in own]
    found = pairs(rot)
    if len(found) < MIN_PAIRS:
        return None
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for _, off, w, n in found:  # least squares: n . t = -off for every pair
        nv = np.array(n)
        A += w * np.outer(nv, nv)
        b += -w * off * nv
    A += 0.05 * sum(f[2] for f in found) * np.eye(2)  # one wall direction only: don't slide along it
    t = np.linalg.solve(A, b)
    return dth, float(t[0]), float(t[1])


def damped(inc: tuple[float, float, float]) -> tuple[float, float, float]:
    th = max(-math.radians(MAX_STEP_DEG), min(math.radians(MAX_STEP_DEG), inc[0] * DAMPING))
    tx, tz = inc[1] * DAMPING, inc[2] * DAMPING
    n = math.hypot(tx, tz)
    if n > MAX_STEP_M:
        tx, tz = tx / n * MAX_STEP_M, tz / n * MAX_STEP_M
    return th, tx, tz
