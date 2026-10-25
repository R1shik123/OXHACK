"""Rooms from the mapped walls, and how well each one has been searched.

Rooms: the floor inside the building outline is rasterised (20 cm cells), walls *and doors* drawn as
barriers, and each connected open region big enough to stand in is a room. Regions touching the outline
are outside.

Status per room, from what the system actually measured:
  - searched: someone stood inside it and the cameras saw most of its floor
  - glanced:  the cameras saw part of it (e.g. from the doorway) but it isn't searched
  - unsearched: hardly seen

PARAMS are starting values; evaluate.py tunes them on the training area of the labelled real-building data.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as cc_label

CELL = 0.2
PARAMS = {
    "min_area": 2.0,  # m^2: smaller pockets are wall gaps / alcoves, not rooms
    "wall_thick": 1,  # cells each side of a wall line (barrier thickness)
    "searched_cover": 0.6,  # share of the room's floor the cameras saw, plus someone stood inside
    "glanced_cover": 0.2,  # share seen for "glanced"
}


def segment_rooms(walls: list, params: dict | None = None) -> list[dict]:
    """walls: [(p, q, y0, y1, kind)]. Returns rooms: {"id", "x", "z", "area", "cells" (set of (i, j) on CELL grid),
    "poly" (outline rectangle for drawing / labels)}; ids are stable while the room's centre stays put."""
    P = {**PARAMS, **(params or {})}
    segs = [(p, q) for p, q, y0, y1, k in walls if k in ("wall", "door", "window", "vertical")]
    if len(segs) < 3:
        return []
    xs = [c for p, q in segs for c in (p[0], q[0])]
    zs = [c for p, q in segs for c in (p[1], q[1])]
    x0, z0 = min(xs) - CELL * 2, min(zs) - CELL * 2
    nx, nz = int((max(xs) - x0) / CELL) + 3, int((max(zs) - z0) / CELL) + 3
    if nx * nz > 600_000:
        return []
    wall = np.zeros((nx, nz), bool)
    for p, q in segs:
        n = max(2, int(math.dist(p, q) / (CELL / 2)))
        for s in np.linspace(0, 1, n):
            i = int((p[0] + (q[0] - p[0]) * s - x0) / CELL)
            j = int((p[1] + (q[1] - p[1]) * s - z0) / CELL)
            if 0 <= i < nx and 0 <= j < nz:
                wall[i, j] = True
    if P["wall_thick"] > 0:
        wall = binary_dilation(wall, iterations=int(P["wall_thick"]))
    lab, n = cc_label(~wall)
    border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))) - {0}
    rooms = []
    for r in range(1, n + 1):
        if r in border:
            continue
        cells = np.argwhere(lab == r)
        area = len(cells) * CELL * CELL
        if area < P["min_area"]:
            continue
        c = cells.mean(0)
        x, z = x0 + (c[0] + 0.5) * CELL, z0 + (c[1] + 0.5) * CELL
        i0, j0 = cells.min(0)
        i1, j1 = cells.max(0) + 1
        rooms.append({"id": f"r{round(x / 0.5)}_{round(z / 0.5)}", "x": round(x, 2), "z": round(z, 2), "area": round(area, 1),
                      "cells": {(int(a), int(b)) for a, b in cells}, "origin": (x0, z0),
                      "poly": [[round(x0 + i0 * CELL, 2), round(z0 + j0 * CELL, 2)], [round(x0 + i1 * CELL, 2), round(z0 + j0 * CELL, 2)],
                               [round(x0 + i1 * CELL, 2), round(z0 + j1 * CELL, 2)], [round(x0 + i0 * CELL, 2), round(z0 + j1 * CELL, 2)]]})
    return rooms


def room_status(room: dict, seen: dict, visited: set, cov_cell: float, params: dict | None = None) -> tuple[str, float]:
    """(status, share of floor seen). seen: coverage cells {(i, j): looks}; visited: 40 cm cells stood in."""
    P = {**PARAMS, **(params or {})}
    x0, z0 = room["origin"]
    hit, entered = 0, False
    for i, j in room["cells"]:
        x, z = x0 + (i + 0.5) * CELL, z0 + (j + 0.5) * CELL
        if (math.floor(x / cov_cell), math.floor(z / cov_cell)) in seen:
            hit += 1
        if not entered and (math.floor(x / 0.4), math.floor(z / 0.4)) in visited:
            entered = True
    share = hit / max(1, len(room["cells"]))
    if entered and share >= P["searched_cover"]:
        return "searched", share
    if share >= P["glanced_cover"] or entered:
        return "glanced", share
    return "unsearched", share


def contains(room: dict, x: float, z: float) -> bool:
    x0, z0 = room["origin"]
    return (int((x - x0) / CELL), int((z - z0) / CELL)) in room["cells"]
