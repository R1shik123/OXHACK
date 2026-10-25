"""2D top-down geometry in the shared frame (x right, z toward the room)."""

from __future__ import annotations

import math

Pt = tuple[float, float]


def _orient(a: Pt, b: Pt, c: Pt) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a: Pt, b: Pt, c: Pt, d: Pt) -> bool:
    """Proper intersection of segments ab and cd (touching endpoints don't count)."""
    d1, d2 = _orient(c, d, a), _orient(c, d, b)
    d3, d4 = _orient(a, b, c), _orient(a, b, d)
    return d1 * d2 < 0 and d3 * d4 < 0


def wall_segment(corners: list[list[float]]) -> tuple[Pt, Pt, float, float] | None:
    """Footprint of a vertical plane: the two most distant xz corners, plus its y range."""
    if len(corners) < 2:
        return None
    pts = [(c[0], c[2]) for c in corners]
    best, pair = -1.0, (pts[0], pts[1])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dd = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if dd > best:
                best, pair = dd, (pts[i], pts[j])
    ys = [c[1] for c in corners]
    return pair[0], pair[1], min(ys), max(ys)


def _cross_t(a: Pt, b: Pt, c: Pt, d: Pt) -> float:
    """Parameter along a->b where it meets line cd (assumes they cross)."""
    rx, rz, sx, sz = b[0] - a[0], b[1] - a[1], d[0] - c[0], d[1] - c[1]
    return ((c[0] - a[0]) * sz - (c[1] - a[1]) * sx) / (rx * sz - rz * sx)


def walls_between(a: Pt, b: Pt, eye_y: float, walls: list[tuple[Pt, Pt, float, float]], merge: float = 0.4) -> int:
    """How many physical walls the sight line a->b crosses at roughly eye height.

    Several phones upload their own copy of the same wall, so crossings closer
    than `merge` metres along the line count as one wall.
    """
    length = max(1e-6, ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
    ts = sorted(_cross_t(a, b, p, q) * length for p, q, y0, y1 in walls if y0 - 0.5 <= eye_y <= y1 + 0.5 and segments_cross(a, b, p, q))
    return sum(1 for i, t in enumerate(ts) if i == 0 or t - ts[i - 1] > merge)


def merge_walls(segs: list[tuple[Pt, Pt, float, float, str]], angle_tol: float = 0.17, offset_tol: float = 0.18, gap_tol: float = 0.35) -> list[tuple[Pt, Pt, float, float, str]]:
    """Merge overlapping, near-collinear wall pieces (many phones, many plane anchors) into clean walls.

    Segments of the same kind group whose direction differs by < angle_tol rad, whose midpoints sit within
    offset_tol m of the other's line, and that overlap or nearly touch (gap < gap_tol m) become one wall.
    Longest pieces seed the clusters, so their direction wins.
    """
    order = sorted(segs, key=lambda s: -math.dist(s[0], s[1]))
    clusters: list[dict] = []
    for p, q, y0, y1, kind in order:
        length = math.dist(p, q)
        if length < 0.15:
            continue
        group = "wall" if kind in ("wall", "vertical") else kind
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        for c in clusters:
            if c["group"] != group:
                continue
            d = abs((ang - c["ang"] + math.pi / 2) % math.pi - math.pi / 2)
            if d > angle_tol:
                continue
            ux, uz = c["u"]
            soff = (mid[0] - c["o"][0]) * -uz + (mid[1] - c["o"][1]) * ux
            if abs(soff) > offset_tol:
                continue
            ta = (p[0] - c["o"][0]) * ux + (p[1] - c["o"][1]) * uz
            tb = (q[0] - c["o"][0]) * ux + (q[1] - c["o"][1]) * uz
            lo, hi = min(ta, tb), max(ta, tb)
            if lo > c["t1"] + gap_tol or hi < c["t0"] - gap_tol:
                continue
            c["t0"], c["t1"] = min(c["t0"], lo), max(c["t1"], hi)
            c["y0"], c["y1"] = min(c["y0"], y0), max(c["y1"], y1)
            c["off"] += soff * length  # length-weighted: the merged line sits between both faces of a wall
            c["w"] += length
            break
        else:
            u = (math.cos(ang), math.sin(ang))
            clusters.append({"group": group, "kind": "wall" if group == "wall" else kind, "ang": ang, "u": u, "o": p, "t0": 0.0, "t1": length,
                             "y0": y0, "y1": y1, "off": 0.0, "w": length})
    out = []
    for c in clusters:
        (ox, oz), (ux, uz) = c["o"], c["u"]
        shift = c["off"] / c["w"]
        ox, oz = ox - uz * shift, oz + ux * shift
        out.append(((ox + ux * c["t0"], oz + uz * c["t0"]), (ox + ux * c["t1"], oz + uz * c["t1"]), c["y0"], c["y1"], c["kind"]))
    return out


def stabilize_walls(new: list, prev: list, move: float = 0.08, turn: float = 0.035) -> list:
    """Keep last time's geometry for walls that only moved a few cm (no visible wobble on every update);
    snap the rest to 2 cm."""
    out = []
    for p, q, y0, y1, k in new:
        keep = None
        for pp, qq, py0, py1, pk in prev:
            if pk != k:
                continue
            same = (math.dist(p, pp) < move and math.dist(q, qq) < move) or (math.dist(p, qq) < move and math.dist(q, pp) < move)
            if same:
                keep = (pp, qq, py0, py1, pk)
                break
        if keep is None:
            snap = lambda v: (round(v[0] / 0.02) * 0.02, round(v[1] / 0.02) * 0.02)
            keep = (snap(p), snap(q), y0, y1, k)
        out.append(keep)
    return out


def room_axis(segs: list[tuple[Pt, Pt, float, float, str]]) -> float | None:
    """The building's main wall direction (radians, modulo 90 degrees), length-weighted over all walls."""
    sx = sy = 0.0
    for p, q, *_ in segs:
        L = math.dist(p, q)
        a = 4 * math.atan2(q[1] - p[1], q[0] - p[0])  # x4: walls at 0/90/180/270 degrees all agree
        sx, sy = sx + L * math.cos(a), sy + L * math.sin(a)
    return math.atan2(sy, sx) / 4 if math.hypot(sx, sy) > 1.0 else None


def straighten_walls(segs: list[tuple[Pt, Pt, float, float, str]], snap_deg: float = 12.0, min_len: float = 0.8,
                     odd_len: float = 1.5) -> list[tuple[Pt, Pt, float, float, str]]:
    """Clean map walls: rotate walls within snap_deg of the room's main directions onto them (about their
    midpoint), drop walls that are off-axis unless they're long (odd_len: a real angled wall), drop short
    wall pieces (min_len; doors and windows are kept), then merge again so straightened pieces join up."""
    axis = room_axis([s for s in segs if s[4] == "wall"])
    out = []
    for p, q, y0, y1, kind in segs:
        L = math.dist(p, q)
        if axis is not None:
            ang = math.atan2(q[1] - p[1], q[0] - p[0])
            d = (ang - axis + math.pi / 4) % (math.pi / 2) - math.pi / 4
            if abs(d) <= math.radians(snap_deg):
                ang -= d
                mx, mz = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
                ux, uz = math.cos(ang) * L / 2, math.sin(ang) * L / 2
                p, q = (mx - ux, mz - uz), (mx + ux, mz + uz)
            elif kind == "wall" and L < odd_len:
                continue
        out.append((p, q, y0, y1, kind))
    return [w for w in merge_walls(out) if w[4] != "wall" or math.dist(w[0], w[1]) >= min_len]


def ray_distance(o: Pt, d: Pt, segs: list[tuple[Pt, Pt]], max_d: float) -> float:
    """Distance along unit ray o + t*d to the first segment it hits (or max_d)."""
    best = max_d
    for (ax, az), (bx, bz) in segs:
        ex, ez = bx - ax, bz - az
        den = d[0] * ez - d[1] * ex
        if abs(den) < 1e-9:
            continue
        t = ((ax - o[0]) * ez - (az - o[1]) * ex) / den
        u = ((ax - o[0]) * d[1] - (az - o[1]) * d[0]) / den
        if 1e-4 < t < best and 0 <= u <= 1:
            best = t
    return best


def _cross_interior(a, b, margin: float = 0.35) -> bool:
    """Do segments a and b cross each other away from their ends (an X, not an L or T)?"""
    (p, q), (r, s) = a, b
    d1x, d1z, d2x, d2z = q[0] - p[0], q[1] - p[1], s[0] - r[0], s[1] - r[1]
    den = d1x * d2z - d1z * d2x
    if abs(den) < 1e-9:
        return False
    t = ((r[0] - p[0]) * d2z - (r[1] - p[1]) * d2x) / den
    u = ((r[0] - p[0]) * d1z - (r[1] - p[1]) * d1x) / den
    la, lb = math.dist(p, q), math.dist(r, s)
    return margin / la < t < 1 - margin / la and margin / lb < u < 1 - margin / lb


def regularize_walls(segs: list[tuple[Pt, Pt, float, float, str]], snap_deg: float = 12.0, merge_tol: float = 0.45,
                     join_gap: float = 0.6, min_len: float = 0.6, odd_len: float = 2.0, corner: float = 0.5) -> list:
    """Floor-plan clean-up ("Manhattan" regularisation) of merged wall pieces:
    1. walls within snap_deg of the building's two main directions are made exactly straight;
       off-axis walls stay only if long (odd_len) and not cutting across straight walls,
    2. parallel lines closer than merge_tol (two faces of one wall, two phones slightly off) become one
       line at their length-weighted average offset; pieces along a line join across gaps < join_gap,
    3. fragments under min_len go; of two walls crossing in an X (not physical) a clearly shorter one goes,
    4. wall ends within `corner` m of a perpendicular wall meet it exactly (clean L / T corners),
    5. doors and windows are moved onto their wall line; pillars stay only if free-standing."""
    walls = [s for s in segs if s[4] == "wall"]
    axis = room_axis(walls)
    if axis is None:
        return list(segs)
    dirs = [(math.cos(axis), math.sin(axis)), (-math.sin(axis), math.cos(axis))]  # group 0 and group 1 directions
    groups: list[list] = [[], []]
    odd = []

    def orient(p, q):
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        d = (ang - axis + math.pi / 4) % (math.pi / 2) - math.pi / 4
        g = int(round(((ang - d - axis) % math.pi) / (math.pi / 2))) % 2
        return d, g

    for p, q, y0, y1, k in walls:
        L = math.dist(p, q)
        d, g = orient(p, q)
        if abs(d) <= math.radians(snap_deg):
            ux, uz = dirs[g]
            nx_, nz_ = -uz, ux
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            off = mid[0] * nx_ + mid[1] * nz_
            t = mid[0] * ux + mid[1] * uz
            groups[g].append({"off": off, "t0": t - L / 2, "t1": t + L / 2, "w": L, "y0": y0, "y1": y1})
        elif L >= odd_len:
            odd.append((p, q, y0, y1, k))
    lines: list = []  # (group, offset, t0, t1, y0, y1)
    for g, items in enumerate(groups):
        items.sort(key=lambda it: it["off"])
        clusters: list[list] = []
        for it in items:
            if clusters and it["off"] - sum(c["off"] * c["w"] for c in clusters[-1]) / sum(c["w"] for c in clusters[-1]) <= merge_tol:
                clusters[-1].append(it)
            else:
                clusters.append([it])
        for cl in clusters:
            off = sum(c["off"] * c["w"] for c in cl) / sum(c["w"] for c in cl)
            for c in sorted(cl, key=lambda c: c["t0"]):  # join pieces along the line
                if lines and lines[-1][0] == g and abs(lines[-1][1] - off) < 1e-9 and c["t0"] <= lines[-1][3] + join_gap:
                    gg, o, a, b, y0, y1 = lines[-1]
                    lines[-1] = (gg, o, a, max(b, c["t1"]), min(y0, c["y0"]), max(y1, c["y1"]))
                else:
                    lines.append((g, off, c["t0"], c["t1"], c["y0"], c["y1"]))
    lines = [ln for ln in lines if ln[3] - ln[2] >= min_len]

    def ends(ln):
        g, off, t0, t1 = ln[:4]
        (ux, uz), (nx_, nz_) = dirs[g], (-dirs[g][1], dirs[g][0])
        return (ux * t0 + nx_ * off, uz * t0 + nz_ * off), (ux * t1 + nx_ * off, uz * t1 + nz_ * off)

    # X crossings between the two groups are not physical: keep the longer wall
    drop = set()
    for i, a in enumerate(lines):
        for j, b in enumerate(lines):
            if a[0] == 0 and b[0] == 1 and i not in drop and j not in drop and _cross_interior(ends(a), ends(b), 0.6):
                la, lb = a[3] - a[2], b[3] - b[2]
                if min(la, lb) < 0.6 * max(la, lb):  # only when one is clearly the minor (noise) line
                    drop.add(i if la < lb else j)
    lines = [ln for i, ln in enumerate(lines) if i not in drop]
    # clean corners: a wall end near a perpendicular wall meets it exactly (extending that wall if needed)
    lines = [list(ln) for ln in lines]
    for a in lines:
        for b in lines:
            if a[0] == b[0]:
                continue
            # in a's coordinates, b is the line t_a = b_off (b's offset measured along a's direction, up to sign)
            (ux, uz) = dirs[a[0]]
            bn = (-dirs[b[0]][1], dirs[b[0]][0])
            s = ux * bn[0] + uz * bn[1]  # +1 or -1
            t_cross = b[1] / s
            (vx, vz) = dirs[b[0]]
            an = (-uz, ux)
            s2 = vx * an[0] + vz * an[1]
            u_cross = a[1] / s2  # where a's line meets b's line, in b's coordinates
            for e in (2, 3):
                if abs(a[e] - t_cross) <= corner and b[2] - corner <= u_cross <= b[3] + corner:
                    a[e] = t_cross
                    b[2], b[3] = min(b[2], u_cross), max(b[3], u_cross)
    out = []
    for ln in lines:
        p, q = ends(ln)
        out.append((p, q, ln[4], ln[5], "wall"))
    straight = [(p, q) for p, q, *_ in out]
    kept_odd: list = []
    for o in sorted(odd, key=lambda o: -math.dist(o[0], o[1])):  # longest first; crossings with anything kept: drop
        if not any(_cross_interior((o[0], o[1]), s) for s in straight + [(k[0], k[1]) for k in kept_odd]):
            kept_odd.append(o)
    out += kept_odd
    for p, q, y0, y1, k in segs:  # doors / windows onto their wall line; pillars only if free-standing
        if k in ("door", "window"):
            d, g = orient(p, q)
            if abs(d) <= math.radians(snap_deg):
                ux, uz = dirs[g]
                nx_, nz_ = -uz, ux
                mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                off = mid[0] * nx_ + mid[1] * nz_
                near = [ln[1] for ln in lines if ln[0] == g and abs(ln[1] - off) <= 0.35]
                if near:
                    off = min(near, key=lambda o: abs(o - off))
                t = mid[0] * ux + mid[1] * uz
                L = math.dist(p, q)
                p = (ux * (t - L / 2) + nx_ * off, uz * (t - L / 2) + nz_ * off)
                q = (ux * (t + L / 2) + nx_ * off, uz * (t + L / 2) + nz_ * off)
            out.append((p, q, y0, y1, k))
        elif k == "pillar":
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            if all(_pt_seg(mid, a, b) > 0.45 for a, b in straight):
                out.append((p, q, y0, y1, k))
    return out


def _pt_seg(p, a, b) -> float:
    L2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / L2))
    return math.hypot(p[0] - a[0] - t * (b[0] - a[0]), p[1] - a[1] - t * (b[1] - a[1]))
