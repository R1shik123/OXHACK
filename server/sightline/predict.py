"""Predictions and decision support on top of the live twin.

- Kalman filter per responder: smoothed velocity, and a predicted position (+ uncertainty) when stale.
- Walkable grid from walls (doors open) and furniture; shortest routes with Dijkstra (scipy).
- Exit route + time, turnaround time against an air/time budget, isolation, search-complete ETA.
- Decisions: ranked, explainable suggestions, including an optimal assignment of responders to
  unsearched areas (Hungarian algorithm on route times).
Everything here is advice for the commander; nothing is enforced.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

GRID = 0.2  # metres per routing cell
WALL_CLEARANCE = 0.15
SAFE_SPEED = 0.6  # m/s assumed when we don't know someone's pace (or in smoke)
EXIT_MARGIN = 1.3  # route time safety factor
TURN_MARGIN = 60.0  # seconds of reserve on top of the walk out
YELLOW_COST = 6.0  # a metre inside a fall-risk zone counts like 6 m: avoided whenever there's a reasonable way round
RED_COST = 250.0  # a metre near a hazard counts like 250 m: crossed only if there is no other way at all
HOME = (0.0, 0.6)  # the command post: just inside the entrance marker


@dataclass
class Kalman:
    """Constant-velocity filter on the floor plane: state (x, z, vx, vz)."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 10)
    t: float = 0.0
    ready: bool = False

    def step(self, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = 0.5  # m/s^2 of unmodelled acceleration
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q * q

    def update(self, t: float, xz: tuple[float, float]) -> None:
        if not self.ready:
            self.x[:] = [xz[0], xz[1], 0, 0]
            self.P = np.diag([0.05, 0.05, 1, 1])
            self.t, self.ready = t, True
            return
        self.step(max(1e-3, t - self.t))
        self.t = t
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1
        R = np.eye(2) * 0.03 ** 2
        y = np.array(xz) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def predict(self, t: float) -> tuple[list[float], float]:
        """Predicted (x, z) at time t and a 2-sigma radius. Velocity decays: people stop when lost."""
        dt = max(0.0, t - self.t)
        decay = min(dt, 3.0)  # assume they keep going for at most ~3 s, then hold position
        x = self.x[0] + self.x[2] * decay
        z = self.x[1] + self.x[3] * decay
        sigma = math.sqrt(max(self.P[0, 0], self.P[1, 1])) + 0.45 * dt  # walking pace of spread
        return [x, z], 2 * sigma

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))


class Router:
    """Walkable grid from the occupancy-with-uncertainty map: A* for routes, Dijkstra for one-to-many distances."""

    def __init__(self) -> None:
        self.key = None
        self.origin = (0.0, 0.0)
        self.shape = (0, 0)
        self.blocked: np.ndarray | None = None
        self.level: np.ndarray | None = None
        self.occ = None
        self.cost2d = None
        self._astar_cache: dict = {}
        self.graph = None

    def build(self, walls: list, objects: list[dict], extra_points: list[tuple[float, float]],
              risk: list[dict] = (), red: list[dict] = (), seen: dict | None = None, cov_cell: float = 0.25,
              visited: set | None = None) -> None:
        """Occupancy with uncertainty and gap filling (occupancy.py) -> walkable cells and a cost per cell:
        uncertainty, fall-risk (yellow) and hazard (red) zones make cells dearer, never forbidden."""
        from .occupancy import CELL, Occupancy
        seen, visited = seen or {}, visited or set()
        zsig = tuple(sorted(str(sorted(z.items())) for z in list(risk) + list(red)))
        key = (len(walls), round(sum(p[0] + q[1] for p, q, *_ in walls), 2), len(objects), hash(zsig), len(seen) // 40, len(visited) // 10)
        if key == self.key and self.graph is not None:
            return
        occ = Occupancy(walls, objects, seen, cov_cell, visited, extra_points)
        nx, nz, x0, z0 = occ.nx, occ.nz, occ.x0, occ.z0
        if nx * nz > 250_000:  # keep it fast; huge sites get coarse routing
            return
        assert abs(CELL - GRID) < 1e-9
        blocked = occ.blocked
        # zone level per cell: 0 clear, 1 fall-risk (yellow), 2 hazard (red)
        level = np.zeros((nx, nz), np.int8)
        X = x0 + (np.arange(nx) + 0.5)[:, None] * GRID
        Z = z0 + (np.arange(nz) + 0.5)[None, :] * GRID
        for zones, lv in ((risk, 1), (red, 2)):
            for zn in zones:
                if "r" in zn:
                    m = (X - zn["x"]) ** 2 + (Z - zn["z"]) ** 2 <= zn["r"] ** 2
                else:
                    from .risk import inside
                    xs, zs = [p[0] for p in zn["poly"]], [p[1] for p in zn["poly"]]
                    i0, i1 = max(0, int((min(xs) - x0) / GRID)), min(nx, int((max(xs) - x0) / GRID) + 1)
                    j0, j1 = max(0, int((min(zs) - z0) / GRID)), min(nz, int((max(zs) - z0) / GRID) + 1)
                    m = np.zeros((nx, nz), bool)
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            m[i, j] = inside(zn, X[i, 0], Z[0, j])
                level[m] = np.maximum(level[m], lv)
        cost2d = occ.step_cost(np.where(level == 2, RED_COST, np.where(level == 1, YELLOW_COST, 1.0)))
        cost = cost2d.ravel()
        # 8-connected graph between free cells (for one-to-many distances); diagonals may not cut wall corners
        idx = np.arange(nx * nz).reshape(nx, nz)
        free_all = ~blocked
        rows, cols, wts = [], [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            ilo, ihi = 0, nx - di
            jlo, jhi = max(0, -dj), nz - max(0, dj)
            a = idx[ilo:ihi, jlo:jhi]
            b = idx[ilo + di:ihi + di, jlo + dj:jhi + dj]
            ok = free_all[ilo:ihi, jlo:jhi] & free_all[ilo + di:ihi + di, jlo + dj:jhi + dj]
            if di and dj:
                ok &= free_all[ilo + di:ihi + di, jlo:jhi] & free_all[ilo:ihi, jlo + dj:jhi + dj]
            rows.append(a[ok])
            cols.append(b[ok])
            wts.append(GRID * math.hypot(di, dj) * (cost[a[ok]] + cost[b[ok]]) / 2)
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        self.graph = coo_matrix((np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(nx * nz, nx * nz)).tocsr()
        self.blocked, self.origin, self.shape, self.key = blocked, (x0, z0), (nx, nz), key
        self.level, self.occ, self.cost2d = level, occ, cost2d
        self._astar_cache = {}

    def astar(self, start: int | None, goal: int | None) -> tuple[list[list[float]], float, int, float] | None:
        """A* route between two cells: (path for drawing, length in metres, worst zone, mean uncertainty)."""
        if start is None or goal is None or self.graph is None:
            return None
        k = (start, goal)
        if k in self._astar_cache:
            return self._astar_cache[k]
        sc, gc = divmod(int(start), self.shape[1]), divmod(int(goal), self.shape[1])
        cells = self.occ.astar(sc, gc, self.cost2d)
        if cells is None:
            out = None
        else:
            xy = [self.occ.xz(c) for c in cells]
            length = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
            worst = int(max((self.level[c] for c in cells), default=0))
            unc = float(np.mean([self.occ.unc[c] for c in cells])) if cells else 0.0
            out = (xy[::2] + ([xy[-1]] if len(xy) % 2 == 0 else []), length, worst, unc)
        if len(self._astar_cache) > 200:
            self._astar_cache.clear()
        self._astar_cache[k] = out
        return out

    def cell(self, xz) -> int | None:
        if self.graph is None:
            return None
        i, j = int((xz[0] - self.origin[0]) / GRID), int((xz[1] - self.origin[1]) / GRID)
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return None
        if self.blocked[i, j]:  # standing next to a wall: snap to the nearest free cell
            free = np.argwhere(~self.blocked[max(0, i - 3):i + 4, max(0, j - 3):j + 4])
            if not len(free):
                return None
            d = ((free - [min(3, i), min(3, j)]) ** 2).sum(1).argmin()
            i, j = max(0, i - 3) + free[d][0], max(0, j - 3) + free[d][1]
        return i * self.shape[1] + j

    def xz(self, c: int) -> list[float]:
        i, j = divmod(int(c), self.shape[1])
        return [round(self.origin[0] + (i + 0.5) * GRID, 2), round(self.origin[1] + (j + 0.5) * GRID, 2)]

    def from_sources(self, sources: list[int]):
        """Distances (and predecessors) from each source cell to everywhere."""
        return dijkstra(self.graph, directed=False, indices=sources, return_predecessors=True)

    def trace(self, pred_row: np.ndarray, target: int, max_len: int = 2000) -> list[int]:
        """Cells from the Dijkstra source to `target`."""
        out, c = [], target
        while c >= 0 and len(out) < max_len:
            out.append(int(c))
            c = pred_row[c]
        return out[::-1]

    def path(self, pred_row: np.ndarray, target: int, max_len: int = 400) -> list[list[float]]:
        return [self.xz(c) for c in self.trace(pred_row, target)][::2][:max_len]  # every other cell is plenty for drawing

    def route(self, pred_row: np.ndarray, target: int) -> tuple[list[list[float]], float, int]:
        """(path for drawing, true length in metres, worst zone crossed: 0 clear, 1 fall-risk, 2 hazard)."""
        cells = self.trace(pred_row, target)
        xy = [self.xz(c) for c in cells]
        length = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
        worst = int(max((self.level.ravel()[c] for c in cells), default=0))
        return xy[::2] + ([xy[-1]] if len(xy) % 2 == 0 else []), length, worst


def search_eta(history: list[tuple[float, float]], target: float = 95.0) -> float | None:
    """Seconds until coverage reaches `target`%, from the recent coverage curve.

    Fits the saturating model p(t) = 100 (1 - exp(-k (t - t0))), i.e. -ln(1 - p/100) is linear in t,
    on the last 3 minutes; falls back to the recent linear rate.
    """
    pts = [(t, p) for t, p in history if p is not None and 0 < p < 99.5 and t >= history[-1][0] - 180]
    if len(pts) < 5:
        return None
    t = np.array([a for a, _ in pts])
    p = np.array([b for _, b in pts])
    if p[-1] >= target:
        return 0.0
    yv = -np.log(1 - p / 100)
    k, b = np.polyfit(t, yv, 1)
    if k > 1e-5:
        t_hit = (-math.log(1 - target / 100) - b) / k
        eta = t_hit - t[-1]
        if 0 <= eta < 7200:
            return float(eta)
    rate = (p[-1] - p[0]) / max(1.0, t[-1] - t[0])
    return float((target - p[-1]) / rate) if rate > 0.01 else None


def unsearched_regions(coverage, min_area: float = 1.0, limit: int = 6) -> list[dict]:
    """Connected unsearched patches inside the building outline, biggest first."""
    if not coverage.bbox:
        return []
    from .twin import COV_CELL

    x0, z0, x1, z1 = coverage.bbox
    i0, j0 = math.floor(x0 / COV_CELL), math.floor(z0 / COV_CELL)
    nx, nz = math.floor(x1 / COV_CELL) - i0 + 1, math.floor(z1 / COV_CELL) - j0 + 1
    if nx <= 0 or nz <= 0 or nx * nz > 400_000:
        return []
    un = np.ones((nx, nz), bool)
    for (i, j) in coverage.seen:
        if 0 <= i - i0 < nx and 0 <= j - j0 < nz:
            un[i - i0, j - j0] = False
    lab, n = cc_label(un)
    out = []
    for r in range(1, n + 1):
        cells = np.argwhere(lab == r)
        area = len(cells) * COV_CELL * COV_CELL
        if area < min_area:
            continue
        c = cells.mean(0)
        out.append({"x": round((i0 + c[0] + 0.5) * COV_CELL, 2), "z": round((j0 + c[1] + 0.5) * COV_CELL, 2), "area": round(area, 1)})
    return sorted(out, key=lambda r: -r["area"])[:limit]


def exit_doors(walls: list, edge: float = 0.75) -> list[tuple[float, float]]:
    """Doors in the building's outside walls (within `edge` m of the mapped outline): ways out."""
    segs = [(p, q) for p, q, y0, y1, k in walls if k == "wall"]
    if not segs:
        return []
    xs = [c for p, q in segs for c in (p[0], q[0])]
    zs = [c for p, q in segs for c in (p[1], q[1])]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    out = []
    for p, q, y0, y1, k in walls:
        if k != "door":
            continue
        mx, mz = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        if min(mx - x0, x1 - mx, mz - z0, z1 - mz) < edge:
            out.append((mx, mz))
    return out


class Planner:
    def __init__(self) -> None:
        self.known_exits: set[tuple[int, int]] = set()  # exit doors already announced on the timeline
        self.kalman: dict[str, Kalman] = {}
        self.router = Router()
        self.cov_history: list[tuple[float, float]] = []
        self.budget_s = 20 * 60.0  # air / time-on-task budget per responder (commander can change it)
        self.first_seen: dict[str, float] = {}
        self.isolated_since: dict[str, float] = {}
        self.speed: dict[str, float] = {}

    def observe_pose(self, pid: str, t: float, pos: list[float], walking: bool) -> None:
        k = self.kalman.setdefault(pid, Kalman())
        k.update(t, (pos[0], pos[2]))
        self.first_seen.setdefault(pid, t)
        if walking and 0.3 < k.speed < 2.5:  # learn each person's pace while they walk
            self.speed[pid] = 0.9 * self.speed.get(pid, k.speed) + 0.1 * k.speed

    def forget(self, pid: str) -> None:
        for d in (self.kalman, self.first_seen, self.isolated_since, self.speed):
            d.pop(pid, None)

    def plan(self, world) -> dict:
        """Run every ~1 s off the event loop. Returns {'people': {...}, 'search': {...}, 'decisions': [...]}."""
        now = time.time()
        pct = world.coverage.percent()
        if pct is not None:
            self.cov_history.append((now, pct))
            self.cov_history = self.cov_history[-400:]
        people = {p.id: p for p in list(world.phones.values()) if p.pos and p.connected}
        walls = world.merged_walls()
        # ways out: the entrance (marker), rally points, and doors found in the building's outside walls
        doors_out = exit_doors(walls)
        for mx, mz in doors_out:
            key = (round(mx / 0.5), round(mz / 0.5))
            if key not in self.known_exits:
                self.known_exits.add(key)
                world.event("exit", f"Exit door found at ({mx:.1f}, {mz:.1f}) m: exit routes now use it")
        exits = [(0.0, 0.6)] + [(pin["pos"][0], pin["pos"][2]) for pin in list(world.pins.values()) if pin["kind"] == "rally"] + doors_out
        from .risk import RED_RADIUS
        risk = world.risk_zones()
        red = [{"kind": "hazard", "x": pin["pos"][0], "z": pin["pos"][2], "r": RED_RADIUS} for pin in list(world.pins.values()) if pin["kind"] == "hazard"]
        from .twin import COV_CELL
        self.router.build(walls, world.objects, [(p.pos[0], p.pos[2]) for p in people.values()] + exits + [HOME], risk, red,
                          dict(world.coverage.seen), COV_CELL, set(world.visited))  # snapshots: other threads keep adding
        out_people: dict[str, dict] = {}
        decisions: list[dict] = []

        # predicted positions for stale responders
        for p in list(world.phones.values()):
            k = self.kalman.get(p.id)
            if k and k.ready and (not p.connected or now - p.pose_ts > 3):
                xz, r = k.predict(now)
                out_people[p.id] = {"predicted": {"x": round(xz[0], 2), "z": round(xz[1], 2), "r": round(r, 2)}}

        R = self.router
        if R.graph is None or not people:
            return {"people": out_people, "search": self._search(pct), "decisions": decisions}

        # exit routes: one Dijkstra per exit, keep the best for each person
        exit_cells = [c for c in (R.cell(e) for e in exits) if c is not None]
        dist_e, pred_e = R.from_sources(exit_cells) if exit_cells else (None, None)
        person_cells = {pid: R.cell((p.pos[0], p.pos[2])) for pid, p in people.items()}
        for pid, p in people.items():
            info = out_people.setdefault(pid, {})
            c = person_cells[pid]
            speed = max(SAFE_SPEED, min(1.4, self.speed.get(pid, 0.9)))
            info["speed"] = round(speed, 2)
            if c is not None and dist_e is not None:
                d = dist_e[:, c]
                e = int(np.argmin(d))
                if np.isfinite(d[e]):
                    ast = R.astar(c, exit_cells[e])  # the drawn exit route: A* on the uncertainty map
                    exit_path, exit_len, _ = (ast[0], ast[1], ast[2]) if ast else R.route(pred_e[e], c)
                    if not ast:
                        exit_path = exit_path[::-1]  # Dijkstra traces exit -> person; routes always start at the person
                    d = np.array([exit_len if i == e else np.inf for i in range(len(d))])  # true metres, not route cost
                    route_t = d[e] / speed * EXIT_MARGIN
                    inside = now - self.first_seen.get(pid, now)
                    turn = self.budget_s - inside - route_t - TURN_MARGIN
                    info.update({"exit_m": round(float(d[e]), 1), "exit_s": round(route_t), "inside_s": round(inside),
                                 "turnaround_s": round(turn), "exit_path": exit_path})
                    if turn <= 0:
                        decisions.append({"prio": 1, "kind": "turnaround", "who": pid,
                                          "text": f"{p.name}: turn back now ({route_t:.0f} s to the exit, budget used)"})
                    elif turn < 120:
                        decisions.append({"prio": 2, "kind": "turnaround", "who": pid,
                                          "text": f"{p.name}: turn back in {turn / 60:.0f} min ({d[e]:.0f} m to the exit)"})
                else:
                    info["exit_m"] = None
                    decisions.append({"prio": 2, "kind": "noroute", "who": pid,
                                      "text": f"{p.name}: no known route to an exit through the mapped walls"})

        # everyone's safest way back to the command post (entrance marker), shown on the map at all times
        home_cell = R.cell(HOME)
        if home_cell is not None:
            for pid, p in people.items():
                c, info = person_cells[pid], out_people.setdefault(pid, {})
                ast = R.astar(c, home_cell)  # A*: short, certain floor preferred, yellow avoided, red last resort
                if ast is None:
                    info["home"] = None
                    decisions.append({"prio": 2, "kind": "nohome", "who": pid, "text": f"{p.name}: no known route back to command"})
                    continue
                path, length, worst, unc = ast
                info["home"] = {"path": path, "m": round(length, 1), "zone": worst, "unc": round(unc, 2)}  # responder -> command
                if worst == 2:
                    decisions.append({"prio": 1, "kind": "home_hazard", "who": pid,
                                      "text": f"{p.name}: the only way back to command passes a hazard ({length:.0f} m)"})
                elif worst == 1:
                    decisions.append({"prio": 4, "kind": "home_risk", "who": pid,
                                      "text": f"{p.name}: the way back crosses a fall-risk area: take care ({length:.0f} m)"})

        # routes from every responder (for helping, victims, assignment)
        pids = [pid for pid in people if person_cells[pid] is not None]
        dist_p, pred_p = R.from_sources([person_cells[pid] for pid in pids]) if pids else (None, None)

        def nearest_helper(target_xz, exclude: str | None):
            tc = R.cell(target_xz)
            if tc is None or dist_p is None:
                return None
            best = None
            for i, pid in enumerate(pids):
                if pid == exclude or people[pid].vitals.alert:
                    continue
                d = dist_p[i, tc]
                if np.isfinite(d) and (best is None or d < best[1]):
                    best = (pid, float(d), i, tc)
            return best

        # (sending a colleague to someone who is down, "converge", is switched off for now: its routes
        # weren't reliable. The fall / no-movement alarm itself is unchanged.)

        # victims pinned by the commander or a responder -> nearest responder by route
        for pin in list(world.pins.values()):
            if pin["kind"] != "victim":
                continue
            h = nearest_helper((pin["pos"][0], pin["pos"][2]), None)
            if h:
                sp = out_people.get(h[0], {}).get("speed", 0.9)
                what = f"victim{': ' + pin['label'] if pin.get('label') else ''}"
                decisions.append({"prio": 1, "kind": "victim", "who": h[0], "pin": pin["id"],
                                  "path": R.path(pred_p[h[2]], h[3]), "text": f"{people[h[0]].name} → {what}: {h[1]:.0f} m, ~{h[1] / sp:.0f} s"})

        # isolation: nobody within 10 m by route (or 2+ walls) for over a minute
        for i, pid in enumerate(pids):
            others = [dist_p[i, person_cells[o]] for o in pids if o != pid]
            alone = bool(others) and min(others) > 10
            if alone:
                since = self.isolated_since.setdefault(pid, now)
                if now - since > 60:
                    decisions.append({"prio": 3, "kind": "isolated", "who": pid,
                                      "text": f"{people[pid].name} has been alone for {(now - since) / 60:.0f} min (nearest teammate {min(others):.0f} m)"})
                out_people.setdefault(pid, {})["alone_s"] = round(now - since)
            else:
                self.isolated_since.pop(pid, None)

        # search management: rooms from the walls; each free responder gets a next room (not searched yet), or
        # a prompt to re-check a room that was only seen from the doorway. Optimal assignment on route length.
        from .rooms import contains, room_status
        from .twin import COV_CELL
        rooms = self._rooms(walls)
        seen_now, visited_now = dict(world.coverage.seen), set(world.visited)
        room_info = []
        for rm in rooms:
            status, share = room_status(rm, seen_now, visited_now, COV_CELL)
            occupied = any(contains(rm, p.pos[0], p.pos[2]) for p in people.values())
            room_info.append({"id": rm["id"], "poly": rm["poly"], "area": rm["area"], "status": status, "seen": round(share, 2),
                              "x": rm["x"], "z": rm["z"], "occupied": occupied})
        busy = {d.get("who") for d in decisions if d["kind"] == "victim" or (d["kind"] == "turnaround" and d["prio"] == 1)}
        free = [i for i, pid in enumerate(pids) if pid not in busy and not people[pid].vitals.alert]
        targets = [(ri, R.cell((ri["x"], ri["z"]))) for ri in room_info if ri["status"] != "searched" and not ri["occupied"]]
        targets = [(ri, c) for ri, c in targets if c is not None]
        if free and targets:
            cost = np.full((len(free), len(targets)), 1e9)
            for a_, i in enumerate(free):
                for b_, (ri, c) in enumerate(targets):
                    d = dist_p[i, c]
                    if np.isfinite(d):
                        cost[a_, b_] = d * (1.0 if ri["status"] == "unsearched" else 1.15)  # fresh rooms first, re-checks next
            rows, cols = linear_sum_assignment(cost)
            for a_, b_ in zip(rows, cols):
                if cost[a_, b_] >= 1e9:
                    continue
                i = free[a_]
                ri, c = targets[b_]
                ast = R.astar(person_cells[pids[i]], c)
                path, length, worst = (ast[0], ast[1], ast[2]) if ast else R.route(pred_p[i], c)
                pid = pids[i]
                care = " (route crosses a fall-risk area)" if worst == 1 else " (route passes a hazard!)" if worst == 2 else ""
                if ri["status"] == "unsearched":
                    decisions.append({"prio": 4, "kind": "next_room", "who": pid, "path": path, "room": ri["id"],
                                      "text": f"{people[pid].name} → next room: {ri['area']:.0f} m², {length:.0f} m away, not searched yet{care}"})
                else:
                    decisions.append({"prio": 3, "kind": "research", "who": pid, "path": path, "room": ri["id"],
                                      "text": f"{people[pid].name} → re-check the room {length:.0f} m away: only {ri['seen'] * 100:.0f}% "
                                              f"seen so far, nobody went inside properly{care}"})
        decisions.sort(key=lambda d: d["prio"])
        done = sum(1 for ri in room_info if ri["status"] == "searched")
        route = self._route(world, R, people, person_cells, out_people)
        occ = R.occ
        return {"people": out_people, "search": {**self._search(pct), "rooms": room_info, "rooms_done": done},
                "decisions": decisions[:12], "inferred_walls": occ.gaps if occ is not None else [], "route": route}

    # ---- the commander's waypoint route ----
    def _route(self, world, R, people, person_cells, out_people) -> dict | None:
        rt = world.route
        stops = [world.pins[k] for k in rt["order"] if k in world.pins]
        if not stops:
            return None
        who = rt["assignee"] if rt["assignee"] in people else None
        if who:  # progress: within 1 m of the next stop = reached
            nxt = next((s for s in stops if s["id"] not in rt["reached"]), None)
            pos = people[who].pos
            if nxt and math.hypot(pos[0] - nxt["pos"][0], pos[2] - nxt["pos"][2]) <= 1.0:
                rt["reached"].append(nxt["id"])
                world.event("route", f"{people[who].name} reached waypoint {nxt['label']} of {len(stops)}", who=who)
        todo = [s for s in stops if s["id"] not in rt["reached"]]
        start = (people[who].pos[0], people[who].pos[2]) if who else HOME
        pts = [start] + [(s["pos"][0], s["pos"][2]) for s in todo] + ([HOME] if rt["home"] else [])
        legs, total, worst_all = [], 0.0, 0
        for a, b in zip(pts, pts[1:]):
            ast = R.astar(R.cell(a), R.cell(b))
            if ast is None:
                legs.append({"path": [list(a), list(b)], "m": None, "zone": 0, "blocked": True})
                continue
            path, length, worst, unc = ast
            legs.append({"path": path, "m": round(length, 1), "zone": worst})
            total += length
            worst_all = max(worst_all, worst)
        speed = out_people.get(who, {}).get("speed", 0.9) if who else 0.9
        out = {"stops": [{"id": s["id"], "n": int(s["label"] or 0), "x": round(s["pos"][0], 2), "z": round(s["pos"][2], 2),
                          "reached": s["id"] in rt["reached"]} for s in stops],
               "legs": legs, "total_m": round(total, 1), "eta_s": round(total / max(speed, 0.3)), "assignee": who,
               "home": rt["home"], "worst_zone": worst_all, "done": not todo}
        if who and todo and legs and legs[0].get("m") is not None:  # guidance for the responder's phone
            path = legs[0]["path"]
            pos = people[who].pos
            aim = next((pt for pt in path if math.hypot(pt[0] - pos[0], pt[1] - pos[2]) >= 1.5), path[-1])
            out_people.setdefault(who, {})["guide"] = {"next": int(todo[0]["label"] or 0), "total": len(stops),
                                                       "m": legs[0]["m"], "aim": aim, "x": todo[0]["pos"][0], "z": todo[0]["pos"][2]}
        elif who and not todo:
            out_people.setdefault(who, {})["guide"] = {"done": True, "total": len(stops)}
        return out

    def best_order(self, world) -> list[str] | None:
        """Shortest walking order through the waypoints (A* distances), starting at the assignee (or command)
        and ending at command if the route returns there. Exact for <= 8 stops, 2-opt above that."""
        import itertools
        R = self.router
        rt = world.route
        stops = [world.pins[k] for k in rt["order"] if k in world.pins and k not in rt["reached"]]
        if len(stops) < 2 or R.graph is None:
            return None
        who = rt["assignee"]
        p = world.phones.get(who) if who else None
        start = (p.pos[0], p.pos[2]) if p and p.pos else HOME
        pts = [start] + [(s["pos"][0], s["pos"][2]) for s in stops] + ([HOME] if rt["home"] else [])
        cells = [R.cell(x) for x in pts]
        n = len(pts)
        D = np.full((n, n), 1e6)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ast = R.astar(cells[i], cells[j])
                    if ast is not None:
                        D[i, j] = ast[1] * (1 + 0.5 * ast[2])  # a little extra for legs through risk zones
        k = len(stops)
        end = n - 1 if rt["home"] else None

        def cost(order):
            seq = [0] + [o + 1 for o in order] + ([end] if end is not None else [])
            return sum(D[a, b] for a, b in zip(seq, seq[1:]))

        if k <= 8:
            best = min(itertools.permutations(range(k)), key=cost)
        else:
            left, cur, best = set(range(k)), 0, []
            while left:  # nearest neighbour, then 2-opt
                nxt_ = min(left, key=lambda o: D[cur, o + 1])
                best.append(nxt_)
                left.discard(nxt_)
                cur = nxt_ + 1
            improved = True
            while improved:
                improved = False
                for i in range(k - 1):
                    for j in range(i + 1, k):
                        cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                        if cost(cand) < cost(best) - 1e-6:
                            best, improved = cand, True
        done = [k_ for k_ in rt["order"] if k_ in rt["reached"]]
        return done + [stops[o]["id"] for o in best]

    def _rooms(self, walls: list) -> list[dict]:
        from .rooms import segment_rooms
        key = (len(walls), round(sum(p[0] + q[1] for p, q, *_ in walls), 2))
        if getattr(self, "_rooms_key", None) != key:
            self._rooms_key, self._rooms_cache = key, segment_rooms(walls)
        return self._rooms_cache

    def _search(self, pct) -> dict:
        eta = search_eta(self.cov_history) if self.cov_history else None
        return {"pct": pct, "eta_s": round(eta) if eta is not None else None}
