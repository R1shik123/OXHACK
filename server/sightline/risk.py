"""Fall-risk zones: mild (yellow) warnings, not hazards.

Three kinds, all from data the system already measures:
  - stairs: the flight plus a margin at its top and bottom (stairs detection in fusion.py)
  - drops: floor edges where the surface falls away more than `drop_depth` (fusion: "drops")
  - clutter: low objects (12-35 cm tall) standing in walkways, away from walls (fusion furniture "low")
  - falls: places where a fall alert went off earlier this session

Zones are plain data ({"kind", "x", "z", "r"} circles or {"kind", "poly"} polygons) so the planner, the map
and the offline evaluation (evaluate.py) all use exactly the same definition. The numbers in PARAMS are the
starting values; evaluate.py tunes them on the training area of the labelled real-building data.
"""

from __future__ import annotations

import math

PARAMS = {
    "stairs_margin": 0.6,  # m added around a flight of stairs (top and bottom landings)
    "drop_radius": 0.5,  # m around a detected drop
    "clutter_max_top": 0.35,  # m: only objects this low count as trip clutter
    "clutter_wall_gap": 0.3,  # m: objects closer to a wall than this aren't in a walkway
    "clutter_margin": 0.35,  # m added around clutter
    "fall_radius": 1.0,  # m around a place where someone fell
}
RED_RADIUS = 1.5  # m around a hazard pin (red: routes avoid it unless there is no other way)


def _grow(poly: list, d: float) -> list:
    """Push a convex polygon's corners outward from its centre by d metres."""
    cx = sum(p[0] for p in poly) / len(poly)
    cz = sum(p[1] for p in poly) / len(poly)
    out = []
    for x, z in poly:
        L = math.hypot(x - cx, z - cz) or 1.0
        out.append([round(x + (x - cx) / L * d * 1.41, 2), round(z + (z - cz) / L * d * 1.41, 2)])
    return out


def _seg_dist(p, a, b) -> float:
    ax, az, bx, bz = a[0], a[1], b[0], b[1]
    L2 = (bx - ax) ** 2 + (bz - az) ** 2
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - az) * (bz - az)) / L2))
    return math.hypot(p[0] - ax - t * (bx - ax), p[1] - az - t * (bz - az))


def fall_risk_zones(stairs: list, drops: list, objects: list, walls: list, fall_spots: list, params: dict | None = None) -> list[dict]:
    P = {**PARAMS, **(params or {})}
    zones: list[dict] = []
    for st in stairs:
        zones.append({"kind": "stairs", "poly": _grow([list(c) for c in st["corners"]], P["stairs_margin"])})
    for x, z in drops:
        zones.append({"kind": "drop", "x": round(x, 2), "z": round(z, 2), "r": P["drop_radius"]})
    segs = [(w[0], w[1]) for w in walls if w[4] == "wall"]
    for o in objects:
        if o.get("kind") != "low" or o.get("top", 1) > P["clutter_max_top"]:
            continue
        cx, cz = (o["x0"] + o["x1"]) / 2, (o["z0"] + o["z1"]) / 2
        half = max(o["x1"] - o["x0"], o["z1"] - o["z0"]) / 2
        if any(_seg_dist((cx, cz), a, b) < P["clutter_wall_gap"] + half for a, b in segs):
            continue  # against a wall: not in anyone's way
        zones.append({"kind": "clutter", "x": round(cx, 2), "z": round(cz, 2), "r": round(half + P["clutter_margin"], 2)})
    for x, z in fall_spots:
        zones.append({"kind": "fall", "x": round(x, 2), "z": round(z, 2), "r": P["fall_radius"]})
    return zones


def inside(zone: dict, x: float, z: float, tol: float = 0.0) -> bool:
    """Is (x, z) in the zone (or within tol metres of it)?"""
    if "r" in zone:
        return math.hypot(x - zone["x"], z - zone["z"]) <= zone["r"] + tol
    poly = zone["poly"]
    hit = False
    for i in range(len(poly)):  # ray casting
        (ax, az), (bx, bz) = poly[i], poly[(i + 1) % len(poly)]
        if (az > z) != (bz > z) and x < (bx - ax) * (z - az) / ((bz - az) or 1e-12) + ax:
            hit = not hit
    if hit or tol <= 0:
        return hit
    return min(_seg_dist((x, z), poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))) <= tol
