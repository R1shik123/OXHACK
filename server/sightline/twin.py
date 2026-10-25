"""Digital-twin layer on top of the live map: search coverage and responder vitals (fall / no movement).
All in memory, all in the shared frame."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from .geometry import ray_distance

# ---- search coverage ---------------------------------------------------------------
COV_CELL = 0.25  # metres per coverage cell (top-down)
VIEW_RANGE = 5.0  # metres a camera usefully "searches"
HALF_FOV = math.radians(28)  # phone camera, portrait
RAYS = 13


class Coverage:
    """Which floor cells some responder's camera has actually looked at, with walls blocking the view."""

    def __init__(self) -> None:
        self.seen: dict[tuple[int, int], int] = {}  # cell -> looks (capped at 255)
        self.new: dict[tuple[int, int], int] = {}
        self.bbox: tuple[float, float, float, float] | None = None  # building outline (from walls)

    def observe(self, pos: list[float], fwd: list[float] | None, walls: list[tuple]) -> None:
        if fwd is None:
            return
        h = math.hypot(fwd[0], fwd[2])
        if h < 0.3:  # looking straight at the floor/ceiling: no useful horizontal sweep
            return
        yaw = math.atan2(fwd[0], fwd[2])
        segs = [(p, q) for p, q, *_ in walls]
        o = (pos[0], pos[2])
        cells: set[tuple[int, int]] = set()
        for i in range(RAYS):
            a = yaw - HALF_FOV + 2 * HALF_FOV * i / (RAYS - 1)
            d = (math.sin(a), math.cos(a))
            reach = ray_distance(o, d, segs, VIEW_RANGE)
            t = 0.0
            while t <= reach:
                cells.add((math.floor((o[0] + d[0] * t) / COV_CELL), math.floor((o[1] + d[1] * t) / COV_CELL)))
                t += COV_CELL * 0.5
        for c in cells:
            n = min(255, self.seen.get(c, 0) + 1)
            self.seen[c] = n
            self.new[c] = n

    def set_outline(self, walls: list[tuple]) -> None:
        pts = [pt for p, q, *_ in walls for pt in (p, q)]
        self.bbox = (min(x for x, _ in pts), min(z for _, z in pts), max(x for x, _ in pts), max(z for _, z in pts)) if pts else None

    def percent(self) -> float | None:
        """Share of cells inside the building outline that have been looked at."""
        if not self.bbox:
            return None
        x0, z0, x1, z1 = self.bbox
        ix0, iz0, ix1, iz1 = math.floor(x0 / COV_CELL), math.floor(z0 / COV_CELL), math.floor(x1 / COV_CELL), math.floor(z1 / COV_CELL)
        total = (ix1 - ix0 + 1) * (iz1 - iz0 + 1)
        inside = sum(1 for (ix, iz) in self.seen if ix0 <= ix <= ix1 and iz0 <= iz <= iz1)
        return round(100 * inside / total, 1) if total > 0 else None

    def message(self, full: bool) -> dict | None:
        cells = self.seen if full else self.new
        if not full and not cells:
            return None
        msg = {"t": "coverage", "cell": COV_CELL, "reset": full, "bbox": self.bbox, "pct": self.percent(),
               "seen": [[ix, iz, n] for (ix, iz), n in cells.items()]}
        if not full:  # a full snapshot for one new viewer must not swallow everyone's pending delta
            self.new = {}
        return msg


# ---- responder vitals --------------------------------------------------------------
FALL_DROP = 0.7  # metres the camera drops ...
FALL_WINDOW = 2.5  # ... within this many seconds, then stays low
STILL_SECS = 30.0  # no movement for this long -> alert
STILL_RADIUS = 0.25  # metres counted as "not moving"
CLEAR_MOVE = 0.5  # moving this far from the alert spot clears it


@dataclass
class Vitals:
    samples: deque = field(default_factory=lambda: deque(maxlen=600))  # (t, x, y, z)
    alert: dict | None = None
    window_start: float = 0.0  # still-detection only looks at samples after this (reset by "I'm OK")
    impact_at: float = 0.0  # last accelerometer free-fall + impact reported by the phone
    impact_g: float = 0.0

    def impact(self, g: float, now: float) -> None:
        self.impact_at, self.impact_g = now, g

    def update(self, pos: list[float], now: float) -> bool:
        """Feed a pose; returns True if the alert state changed."""
        self.samples.append((now, *pos))
        while self.samples and self.samples[0][0] < now - STILL_SECS - 5:
            self.samples.popleft()
        if self.alert:
            if math.dist(pos, self.alert["pos"]) > CLEAR_MOVE:
                self.alert = None
                self.window_start = now
                return True
            return False
        kind = self._fall(now) or self._impact_fall(now) or self._still(pos, now)
        if kind:
            source = "impact" if kind == "fall" and now - self.impact_at < 5 else "height" if kind == "fall" else "stillness"
            self.alert = {"kind": kind, "since": now, "acked": False, "pos": list(pos), "source": source}
            return True
        return False

    def _impact_fall(self, now: float) -> str | None:
        """Free fall + impact from the accelerometer, then hardly moving for a second: a fall,
        even if the height drop was small (e.g. falling from kneeling)."""
        if not (1.0 <= now - self.impact_at <= 5.0):
            return None
        recent = [s for s in self.samples if s[0] >= now - 1.0]
        if len(recent) < 3:
            return None
        span = max(math.dist(recent[0][1:], s[1:]) for s in recent)
        drop = max((s[2] for s in self.samples if self.impact_at - 2 <= s[0] <= self.impact_at), default=recent[-1][2]) - recent[-1][2]
        return "fall" if span < 0.2 and drop > 0.25 else None

    def _fall(self, now: float) -> str | None:
        before = [s[2] for s in self.samples if now - FALL_WINDOW <= s[0] <= now - 1.0]
        recent = [s[2] for s in self.samples if s[0] >= now - 0.8]
        if len(before) < 3 or len(recent) < 3:
            return None
        if max(before) - sum(recent) / len(recent) > FALL_DROP and max(recent) - min(recent) < 0.15:
            return "fall"
        return None

    def _still(self, pos: list[float], now: float) -> str | None:
        start = max(self.window_start, now - STILL_SECS)
        if not self.samples or self.samples[0][0] > now - STILL_SECS or self.window_start > now - STILL_SECS:
            return None
        window = [s for s in self.samples if s[0] >= start]
        if window and all(math.dist(pos, s[1:]) < STILL_RADIUS for s in window):
            return "still"
        return None

    def ok(self, now: float) -> None:
        """Responder tapped "I'm OK"."""
        self.alert = None
        self.window_start = now

    def public(self, now: float) -> dict | None:
        if not self.alert:
            return None
        return {"kind": self.alert["kind"], "for": round(now - self.alert["since"], 1), "acked": self.alert["acked"],
                "source": self.alert.get("source")}


# ---- hazard approach ------------------------------------------------------------------
HAZARD_WARN_M = 3.0  # heading towards a hazard closer than this -> the phone asks "avoid or proceed?"
HAZARD_FLAG_M = 1.5  # this close (warning ignored), or "proceed" tapped -> the commander is flagged
HAZARD_RESET_M = 4.0  # walked away again: the next approach warns afresh


@dataclass
class HazardWatch:
    """Warns a responder walking towards a hazard pin; flags the commander if they go on anyway."""

    ref: dict = field(default_factory=dict)  # pin id -> (t, distance) about a second ago
    avoided: set = field(default_factory=set)
    proceeded: set = field(default_factory=set)
    flagged: dict = field(default_factory=dict)  # pin id -> what the commander sees
    warn: dict | None = None  # what this phone should ask about now

    def update(self, pos: list[float], hazards: list[dict], now: float) -> list[str]:
        """Feed a pose and the current hazard pins; returns timeline texts for newly flagged approaches."""
        events, best = [], None
        live = {h["id"] for h in hazards}
        for k in [k for k in self.flagged if k not in live]:
            self.flagged.pop(k)
        for h in hazards:
            hid, label = h["id"], h.get("label") or "hazard"
            d = math.hypot(pos[0] - h["pos"][0], pos[2] - h["pos"][2])
            t0, d0 = self.ref.get(hid, (now, d))
            if now - t0 >= 1.0 or hid not in self.ref:
                self.ref[hid] = (now, d)
            approaching = d < d0 - 0.15
            if d > HAZARD_RESET_M:
                self.avoided.discard(hid)
                self.proceeded.discard(hid)
                self.flagged.pop(hid, None)
                continue
            if hid in self.flagged:
                self.flagged[hid]["dist"] = round(d, 1)
            elif d <= HAZARD_FLAG_M:
                why = "went on after choosing to proceed" if hid in self.proceeded else "kept going despite the warning"
                self.flagged[hid] = {"pin": hid, "label": label, "dist": round(d, 1), "why": why, "since": now}
                events.append(f"is {d:.1f} m from hazard \"{label}\" ({why})")
            if d <= HAZARD_WARN_M and (approaching or d <= HAZARD_FLAG_M) and hid not in self.avoided and hid not in self.proceeded:
                if best is None or d < best["dist"]:
                    best = {"id": hid, "label": label, "dist": round(d, 1)}
        self.warn = best
        return events

    def respond(self, pin_id: str, action: str, now: float) -> str | None:
        """The responder answered the warning. "proceed" flags the commander straight away."""
        w = self.warn if self.warn and self.warn["id"] == pin_id else None
        self.warn = None
        if action == "avoid":
            self.avoided.add(pin_id)
            return None
        self.proceeded.add(pin_id)
        label, dist = (w["label"], w["dist"]) if w else ("hazard", None)
        self.flagged[pin_id] = {"pin": pin_id, "label": label, "dist": dist, "why": "chose to proceed", "since": now}
        return f"chose to proceed towards hazard \"{label}\"" + (f" ({dist:.1f} m away)" if dist is not None else "")

    def public(self) -> list[dict] | None:
        return [dict(v) for v in self.flagged.values()] or None
