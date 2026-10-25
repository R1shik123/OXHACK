"""Floor occupancy with uncertainty, gap filling, and A* pathfinding.

Every 20 cm floor cell gets a probability of being blocked (wall / large object) and an uncertainty, from
evidence the system actually measured, combined in log-odds (the standard occupancy-grid method):
  + walls (AR planes and walls from 3D depth), windows, very large obstructions  -> blocked
  - doors, floor the cameras looked at (more looks = surer), places someone stood -> open
Cells nobody measured stay at 50/50 with full uncertainty. Then the gaps are filled by inference:
  1. Wall gaps: a short gap (<= GAP_MAX) between two lined-up wall pieces, where nobody walked through and
     the cameras saw no open floor, is probably wall (e.g. a dark or glass panel the depth missed).
  2. Unseen floor: unknown cells take the average open-floor belief of their neighbours (Laplace
     in-painting; walls don't spread), so floor enclosed by seen floor reads as open. Uncertainty grows with
     distance from anything measured.
Filled cells are marked `inferred` and keep a high uncertainty, so they are drawn and treated differently
from measured ones. A* then finds routes that are short, avoid blocked cells, and prefer certain cells:
cost per metre = 1 + UNCERTAINTY_WEIGHT * uncertainty + zone penalty (fall-risk yellow, hazard red).
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

CELL = 0.2
GAP_MAX = 1.2  # m: longer gaps are openings (doorways, corridors), never filled
BLOCKED_P = 0.6  # probability of blocked at which a cell can't be walked through
UNCERTAINTY_WEIGHT = 1.5  # a fully unknown metre costs 2.5 m: certain floor is preferred, unknown still usable
L_WALL, L_DOOR, L_OBJ, L_LOOK, L_STOOD, L_GAP = 3.0, -3.0, 2.5, -0.5, -4.0, 1.6


def _line_cells(p, q, x0, z0, nx, nz):
    """Grid cells under the segment p-q, each once."""
    n = max(2, int(math.dist(p, q) / (CELL / 2)))
    seen = set()
    for s in np.linspace(0, 1, n):
        i = int((p[0] + (q[0] - p[0]) * s - x0) / CELL)
        j = int((p[1] + (q[1] - p[1]) * s - z0) / CELL)
        if 0 <= i < nx and 0 <= j < nz and (i, j) not in seen:
            seen.add((i, j))
            yield i, j


class Occupancy:
    def __init__(self, walls: list, objects: list[dict], seen: dict, cov_cell: float, visited: set,
                 extra_points: list = (), margin: float = 2.0) -> None:
        pts = [pt for p, q, *_ in walls for pt in (p, q)] + list(extra_points) + [(0.0, 0.0)]
        self.x0 = min(p[0] for p in pts) - margin
        self.z0 = min(p[1] for p in pts) - margin
        self.nx = int((max(p[0] for p in pts) + margin - self.x0) / CELL) + 1
        self.nz = int((max(p[1] for p in pts) + margin - self.z0) / CELL) + 1
        nx, nz = self.nx, self.nz
        L = np.zeros((nx, nz), np.float32)
        measured = np.zeros((nx, nz), bool)
        wall = np.zeros((nx, nz), bool)
        for p, q, y0, y1, kind in walls:  # walls and windows block, doors are open
            cells = list(_line_cells(p, q, self.x0, self.z0, nx, nz))
            for i, j in cells:
                if kind == "door":
                    L[i, j] += L_DOOR
                else:
                    L[i, j] += L_WALL
                    wall[i, j] = True
                measured[i, j] = True
        wall_thick = binary_dilation(wall, iterations=1) & ~wall  # walls are ~20 cm thick on the grid
        L[wall_thick] += L_WALL * 0.6
        measured |= wall_thick
        for o in objects:  # only very large obstructions block (people step round chairs)
            if o.get("large") and o.get("kind") in ("table", "seat", "obstacle"):
                i0, j0 = int((o["x0"] - self.x0) / CELL), int((o["z0"] - self.z0) / CELL)
                i1, j1 = int((o["x1"] - self.x0) / CELL) + 1, int((o["z1"] - self.z0) / CELL) + 1
                L[max(0, i0):i1, max(0, j0):j1] += L_OBJ
                measured[max(0, i0):i1, max(0, j0):j1] = True
        X = self.x0 + (np.arange(nx) + 0.5) * CELL
        Z = self.z0 + (np.arange(nz) + 0.5) * CELL
        ci = np.floor(X / cov_cell).astype(int)
        cj = np.floor(Z / cov_cell).astype(int)
        looks = np.zeros((nx, nz), np.float32)
        for (a, b), n in seen.items():  # floor the cameras looked at: open (more looks, surer)
            ii = np.nonzero(ci == a)[0]
            jj = np.nonzero(cj == b)[0]
            if len(ii) and len(jj):
                looks[np.ix_(ii, jj)] = max(looks[ii[0], jj[0]], min(n, 6))
        L += np.where(wall | wall_thick, 0, looks * L_LOOK)
        measured |= looks > 0
        for a, b in visited:  # stood here: certainly open (40 cm cells)
            i0, j0 = int((a * 0.4 - self.x0) / CELL), int((b * 0.4 - self.z0) / CELL)
            L[max(0, i0):i0 + 2, max(0, j0):j0 + 2] += L_STOOD
            measured[max(0, i0):i0 + 2, max(0, j0):j0 + 2] = True
        self.measured = measured
        self.inferred = np.zeros((nx, nz), bool)
        self.gaps: list = []
        self._fill_wall_gaps(walls, L, looks)
        self._inpaint(L)
        self.L = L
        self.p = 1 / (1 + np.exp(-L))
        dist = distance_transform_edt(~measured) * CELL  # metres to the nearest measured cell
        conf = np.abs(self.p - 0.5) * 2
        self.unc = np.clip(np.maximum(1 - conf, np.where(measured, 0, np.minimum(1.0, 0.4 + dist / 2.0))), 0, 1).astype(np.float32)
        self.blocked = self.p >= BLOCKED_P

    # ---- gap filling ----
    def _fill_wall_gaps(self, walls: list, L: np.ndarray, looks: np.ndarray) -> None:
        segs = [(p, q) for p, q, y0, y1, k in walls if k in ("wall", "window", "vertical")]
        ends = []
        for idx, (p, q) in enumerate(segs):
            ang = math.atan2(q[1] - p[1], q[0] - p[0])
            ends += [(p, ang, idx), (q, ang, idx)]
        doors = [(p, q) for p, q, y0, y1, k in walls if k == "door"]
        for a in range(len(ends)):
            pa, anga, ia = ends[a]
            for b in range(a + 1, len(ends)):
                pb, angb, ib = ends[b]
                if ia == ib:
                    continue
                g = math.dist(pa, pb)
                if not 0.05 < g <= GAP_MAX:
                    continue
                d = abs((anga - angb + math.pi / 2) % math.pi - math.pi / 2)
                gang = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
                along = abs((gang - anga + math.pi / 2) % math.pi - math.pi / 2)
                if d > 0.06 or along > 0.12:  # both pieces and the gap on one straight line
                    continue
                mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
                if any(math.dist(mid, ((dp[0] + dq[0]) / 2, (dp[1] + dq[1]) / 2)) < 0.8 for dp, dq in doors):
                    continue  # it's a door
                cells = list(_line_cells(pa, pb, self.x0, self.z0, self.nx, self.nz))
                if any(L[i, j] < -2.5 for i, j in cells):  # someone walked through: an opening, not a wall
                    continue
                if np.mean([looks[i, j] for i, j in cells]) >= 1.5:  # well seen open floor in the gap: opening
                    continue
                # narrow gaps (< 70 cm) can't be doorways: fill as wall. Door-sized gaps only get expensive, never
                # sealed: wrongly closing a real doorway would break the routes home.
                add = L_GAP if g < 0.7 else 0.3
                for i, j in cells:
                    L[i, j] += add
                    self.inferred[i, j] = True
                self.gaps.append([[round(pa[0], 2), round(pa[1], 2)], [round(pb[0], 2), round(pb[1], 2)]])

    def _inpaint(self, L: np.ndarray, iters: int = 40) -> None:
        """Unknown cells take the average belief of their neighbours (measured cells stay fixed)."""
        unknown = ~self.measured & ~self.inferred
        if not unknown.any():
            return
        V = np.minimum(L, 0.0)  # only open-floor evidence spreads: walls are thin, they don't "leak" into rooms
        for _ in range(iters):
            P = np.pad(V, 1, mode="edge")
            avg = (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]) / 4
            V[unknown] = avg[unknown] * 0.98  # slight pull back to 50/50: far from evidence stays unsure
        L[unknown] = V[unknown]
        self.inferred |= unknown & (V < -0.4)

    # ---- queries ----
    def cell(self, xz) -> tuple[int, int] | None:
        i, j = int((xz[0] - self.x0) / CELL), int((xz[1] - self.z0) / CELL)
        if not (0 <= i < self.nx and 0 <= j < self.nz):
            return None
        if self.blocked[i, j]:  # standing next to a wall: nearest walkable cell
            win = ~self.blocked[max(0, i - 3):i + 4, max(0, j - 3):j + 4]
            free = np.argwhere(win)
            if not len(free):
                return None
            k = ((free - [min(3, i), min(3, j)]) ** 2).sum(1).argmin()
            i, j = max(0, i - 3) + int(free[k][0]), max(0, j - 3) + int(free[k][1])
        return i, j

    def xz(self, c) -> list[float]:
        return [round(self.x0 + (c[0] + 0.5) * CELL, 2), round(self.z0 + (c[1] + 0.5) * CELL, 2)]

    def step_cost(self, zone_cost: np.ndarray | None = None) -> np.ndarray:
        """Cost multiplier per cell (>= 1)."""
        c = 1.0 + UNCERTAINTY_WEIGHT * self.unc
        return c * zone_cost if zone_cost is not None else c

    def astar(self, start, goal, cost: np.ndarray) -> list[tuple[int, int]] | None:
        """A* on the 8-connected grid; diagonals may not cut wall corners. Heuristic: straight-line metres
        (admissible, every cell costs at least 1 per metre). Returns the cell path or None."""
        if start is None or goal is None:
            return None
        nx, nz, blocked = self.nx, self.nz, self.blocked
        g = {start: 0.0}
        came: dict = {}
        h = lambda c: CELL * math.hypot(c[0] - goal[0], c[1] - goal[1])
        open_ = [(h(start), start)]
        closed = set()
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
        while open_:
            _, c = heapq.heappop(open_)
            if c == goal:
                path = [c]
                while c in came:
                    c = came[c]
                    path.append(c)
                return path[::-1]
            if c in closed:
                continue
            closed.add(c)
            ci, cj = c
            for di, dj, d in steps:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < nx and 0 <= nj < nz) or blocked[ni, nj]:
                    continue
                if di and dj and (blocked[ci + di, cj] or blocked[ci, cj + dj]):
                    continue
                n = (ni, nj)
                ng = g[c] + CELL * d * (cost[ci, cj] + cost[ni, nj]) / 2
                if ng < g.get(n, 1e18):
                    g[n] = ng
                    came[n] = c
                    heapq.heappush(open_, (ng + h(n), n))
        return None
