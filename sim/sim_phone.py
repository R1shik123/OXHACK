#!/usr/bin/env python3
"""Fake Sightline phones walking a virtual building.

Each fake phone has its own random "ARKit world" (random yaw + offset, like a real
phone that started its session somewhere else), sees the entrance marker in that
local frame, and uses the same S/T math as the iOS app to convert everything into
the shared frame before sending. So the server and web view get exactly what real
phones would send, and --check verifies the whole pipeline.

  python sim/sim_phone.py                      # 3 phones touring the building
  python sim/sim_phone.py --scenario xray      # Alice looks through a wall at Bob
  python sim/sim_phone.py --check --duration 8 # self-test, exit code 0 on success
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.client
import io
import json
import os
import ssl
import urllib.parse
import math
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from sightline.frame import mat_mul, to_shared_from_marker, transform_dir, transform_point, yaw_rotation  # noqa: E402
from sightline.protocol import pack_depth_frame, pack_points  # noqa: E402

import numpy as np  # noqa: E402

import websockets  # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:  # frames are optional
    Image = None

EYE = 1.45  # camera height above floor (m)
MARKER_Y = 1.4  # marker centre height above floor; shared y=0 is the marker, floor is y=-1.4
WALL_H = 2.6
FLOOR_Y = -MARKER_Y
SEE_RANGE = 6.0
HFOV = math.radians(55)

# ---------------- the virtual building (ground truth, shared frame, top-down x/z) -------------
# Marker is on the entrance wall z=0 at x=0, facing +z (into the building).


def _build() -> list[tuple[tuple[float, float], tuple[float, float], str]]:
    segs: list = []

    def wall(a, b, kind="wall"):
        segs.append((a, b, kind))

    def wall_with_gaps(a, b, gaps, kind="door"):
        """Wall from a to b (axis-aligned) with openings given as (t0, t1) distances from a."""
        L = math.dist(a, b)
        ux, uz = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        t = 0.0
        for g0, g1 in sorted(gaps):
            if g0 > t:
                wall((a[0] + ux * t, a[1] + uz * t), (a[0] + ux * g0, a[1] + uz * g0))
            wall((a[0] + ux * g0, a[1] + uz * g0), (a[0] + ux * g1, a[1] + uz * g1), kind)
            t = g1
        if t < L:
            wall((a[0] + ux * t, a[1] + uz * t), b)

    wall_with_gaps((-6, 0), (6, 0), [(7.2, 8.4)])  # entrance door right of the marker
    wall_with_gaps((-6, 0), (-6, 16), [(5.0, 7.0), (11.0, 13.0)], "window")
    wall_with_gaps((6, 0), (6, 16), [(9.0, 11.0)], "window")
    wall((-6, 16), (6, 16))
    wall_with_gaps((-1.5, 0), (-1.5, 16), [(4.0, 5.0), (11.0, 12.0)])  # corridor left wall
    wall_with_gaps((1.5, 1.5), (1.5, 16), [(1.5, 2.5), (8.5, 9.5)])  # corridor right wall
    wall((1.5, 0), (1.5, 1.5))
    wall((-6, 8), (-1.5, 8))
    wall((1.5, 7), (6, 7))
    return segs


BUILDING = _build()

TOURS = {
    "Alice": [(0, 1.5), (0, 4.5), (-3.5, 4.5), (-4.5, 2), (-3, 6.5), (-1.5, 4.5), (0, 4.5), (0, 11.5), (-4, 11.5), (-4, 14.5), (-2.5, 9.5), (0, 11.5), (0, 1.5)],
    "Bob": [(0, 1.5), (0, 3.5), (3.5, 3.5), (4.5, 1.5), (4.5, 5.5), (2.5, 3.5), (0, 3.5), (0, 10), (3.5, 10), (4.5, 14.5), (2.5, 12), (0, 10), (0, 1.5)],
    "Chen": [(0, 1.5), (0, 14.5), (-0.8, 15), (0.8, 8), (0, 1.5)],
    "Dana": [(0, 1.5), (0, 11.5), (-3, 11.5), (-5, 9), (-3, 14), (0, 11.5), (0, 1.5)],
}
# furniture as boxes (x0, z0, x1, z1, height): what the depth frames see and fusion should find
BOXES = [(-4.5, 3.0, -3.0, 3.8, 0.75), (-4.0, 4.1, -3.6, 4.5, 0.45), (-3.6, 2.4, -3.2, 2.8, 0.45),
         (3.0, 12.5, 4.5, 13.3, 0.75), (3.4, 12.0, 3.8, 12.4, 0.45), (-4.2, 9.0, -2.8, 9.9, 0.75)]

# tables/desks (footprint edge, height) - only show up in the depth point cloud
FURNITURE = [((-4.5, 3.0), (-3.0, 3.0), 0.75), ((-4.5, 3.0), (-4.5, 3.8), 0.75), ((3.0, 12.5), (4.5, 12.5), 0.75), ((-4.0, 13.5), (-3.2, 13.5), 0.45)]
COLORS = {"wall": (200, 205, 214), "door": (139, 90, 43), "window": (120, 180, 230)}


def ray_hit(ox, oz, dx, dz, segs, max_d=40.0):
    """Nearest hit along a ray: (distance, kind) or (None, None)."""
    best, kind = max_d, None
    for (ax, az), (bx, bz), k in segs:
        ex, ez = bx - ax, bz - az
        den = dx * ez - dz * ex
        if abs(den) < 1e-9:
            continue
        t = ((ax - ox) * ez - (az - oz) * ex) / den
        u = ((ax - ox) * dz - (az - oz) * dx) / den
        if 1e-4 < t < best and 0 <= u <= 1:
            best, kind = t, k
    return (best, kind) if kind else (None, None)


def blocked(a, b, segs) -> bool:
    d = math.dist(a, b)
    if d < 1e-6:
        return False
    hit, kind = ray_hit(a[0], a[1], (b[0] - a[0]) / d, (b[1] - a[1]) / d, segs, d - 1e-3)
    return hit is not None and kind != "door"


# ---------------- one fake phone ----------------


@dataclass
class SimPhone:
    name: str
    route: list[tuple[float, float]]
    speed: float = 1.0
    hold: tuple[float, float] | None = None  # stand still here (xray scenario)
    fall_at: float | None = None  # seconds after start when this responder falls and stays down
    look_at: tuple[float, float] | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self.id = "sim-" + self.name.lower()
        # The phone's private ARKit world: local = L * shared. Unknown to everybody else.
        self.L = yaw_rotation(self.rng.uniform(-math.pi, math.pi), (self.rng.uniform(-5, 5), self.rng.uniform(-1, 1), self.rng.uniform(-5, 5)))
        self.T = None
        self.pos = list(self.hold or self.route[0])
        self.yaw = math.pi / 2  # facing +z
        self.leg = 0
        self.t0 = time.time()
        self.sent_planes: dict[str, tuple] = {}
        self.floor_cells: set[tuple[int, int]] = set()
        self.peers: list[dict] = []
        self.xray: str | None = None
        self.truth = (0.0, 0.0, 0.0)
        self.token = self.server = self.fingerprint = self.host_override = None
        self.coarse_bytes = 0

    # --- "ARKit" ------------------------------------------------------
    def scan_marker(self) -> None:
        """What happens when the phone sees the printed marker: compute T from M (local)."""
        upright_on_wall = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]  # image normal = +z
        m_local = mat_mul(self.L, upright_on_wall)  # marker sits at shared origin
        # add a little detection noise, like real image tracking
        m_local = mat_mul(m_local, yaw_rotation(self.rng.gauss(0, 0.003), (self.rng.gauss(0, 0.004), 0, self.rng.gauss(0, 0.004))))
        self.T = to_shared_from_marker(m_local)

    def local(self, x, y, z):
        return transform_point(self.L, (x, y, z))

    def to_shared(self, p_local):
        return [round(v, 4) for v in transform_point(self.T, p_local)]

    # --- motion -------------------------------------------------------
    @property
    def fallen(self) -> bool:
        return self.fall_at is not None and time.time() - self.t0 > self.fall_at

    def step(self, dt: float) -> None:
        if self.fallen:
            return
        if self.hold:
            target_yaw = math.atan2(self.look_at[0] - self.pos[0], self.look_at[1] - self.pos[1]) if self.look_at else self.yaw
            target_yaw += 0.12 * math.sin((time.time() - self.t0) * 0.7)
        else:
            tgt = self.route[(self.leg + 1) % len(self.route)]
            dx, dz = tgt[0] - self.pos[0], tgt[1] - self.pos[1]
            d = math.hypot(dx, dz)
            if d < 0.05:
                self.leg = (self.leg + 1) % len(self.route)
                return
            s = min(d, self.speed * dt)
            self.pos[0] += dx / d * s
            self.pos[1] += dz / d * s
            target_yaw = math.atan2(dx, dz) + 0.35 * math.sin((time.time() - self.t0) * 0.9)
        dy = (target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += dy * min(1.0, dt * 3)

    def fwd(self):
        return (math.sin(self.yaw), 0.0, math.cos(self.yaw))

    def pose_msg(self) -> dict:
        bob = 0.03 * math.sin((time.time() - self.t0) * 6) if not self.hold else 0.0
        eye = 0.3 if self.fallen else EYE + bob  # phone ends up near the floor
        truth = (self.pos[0], FLOOR_Y + eye, self.pos[1])
        self.truth = truth
        p_local = self.local(*truth)
        f_local = transform_dir(self.L, self.fwd())
        act = "stationary" if self.fallen or self.hold else "walking"  # what Apple's Core Motion classifier would say
        return {"t": "pose", "pos": self.to_shared(p_local), "fwd": [round(v, 4) for v in transform_dir(self.T, f_local)], "tracking": "normal",
                "act": act, "act_conf": 2}

    def discover_planes(self) -> list[dict]:
        """Wall pieces in view become ARKit-like plane anchors (split into <=2 m chunks)."""
        out = []
        me = tuple(self.pos)
        for i, ((ax, az), (bx, bz), kind) in enumerate(BUILDING):
            L = math.dist((ax, az), (bx, bz))
            n = max(1, math.ceil(L / 2.0))
            for k in range(n):
                pid = f"w{i}.{k}"
                if pid in self.sent_planes:
                    continue
                p0 = (ax + (bx - ax) * k / n, az + (bz - az) * k / n)
                p1 = (ax + (bx - ax) * (k + 1) / n, az + (bz - az) * (k + 1) / n)
                mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
                d = math.dist(me, mid)
                ang = abs((math.atan2(mid[0] - me[0], mid[1] - me[1]) - self.yaw + math.pi) % (2 * math.pi) - math.pi)
                if d > SEE_RANGE or ang > HFOV / 2 + 0.3:
                    continue
                # visible if the middle isn't hidden behind another segment
                dd = math.dist(me, mid)
                hit, _ = ray_hit(me[0], me[1], (mid[0] - me[0]) / dd, (mid[1] - me[1]) / dd, BUILDING, dd - 0.05)
                if hit is not None:
                    continue
                j = lambda: self.rng.gauss(0, 0.02)  # noqa: E731
                y0, y1 = FLOOR_Y + 0.3, FLOOR_Y + (2.05 if kind == "door" else WALL_H - 0.3)
                if kind == "window":
                    y0, y1 = FLOOR_Y + 0.9, FLOOR_Y + 2.1
                corners = [(p0[0] + j(), y0, p0[1] + j()), (p1[0] + j(), y0, p1[1] + j()), (p1[0] + j(), y1, p1[1] + j()), (p0[0] + j(), y1, p0[1] + j())]
                shared = [self.to_shared(self.local(*c)) for c in corners]
                self.sent_planes[pid] = tuple(map(tuple, shared))
                out.append({"t": "plane", "id": pid, "kind": kind, "corners": shared})
        cell = (math.floor(self.pos[0] / 2), math.floor(self.pos[1] / 2))
        if cell not in self.floor_cells:
            self.floor_cells.add(cell)
            x0, z0 = cell[0] * 2, cell[1] * 2
            corners = [(x0, FLOOR_Y, z0), (x0 + 2, FLOOR_Y, z0), (x0 + 2, FLOOR_Y, z0 + 2), (x0, FLOOR_Y, z0 + 2)]
            out.append({"t": "plane", "id": f"f{cell[0]}_{cell[1]}", "kind": "floor", "corners": [self.to_shared(self.local(*c)) for c in corners]})
        return out

    def pick_xray(self) -> str | None:
        """Same rule as the iOS app: nearest-to-centre colleague within a 30 degree cone."""
        best, best_ang = None, math.radians(30)
        f = self.fwd()
        for p in self.peers:
            if not p.get("pos") or p.get("stale"):
                continue
            dx, dz = p["pos"][0] - self.truth[0], p["pos"][2] - self.truth[2]
            d = math.hypot(dx, dz)
            if d < 0.5 or d > 25:
                continue
            ang = math.acos(max(-1, min(1, (dx * f[0] + dz * f[2]) / d)))
            if ang < best_ang:
                best, best_ang = p["id"], ang
        return best

    def depth_points(self, cols: int = 40, rows: int = 24) -> list[tuple]:
        """What the phone's AI depth would produce: coloured 3D points of walls, floor and furniture."""
        out = []
        f_px = 1 / math.tan(HFOV / 2)
        vfov = HFOV * 1.3
        for ci in range(cols):
            a = self.yaw + math.atan(((ci + 0.5) / cols * 2 - 1) / f_px)
            dx, dz = math.sin(a), math.cos(a)
            d_wall, kind = ray_hit(self.pos[0], self.pos[1], dx, dz, [s for s in BUILDING if s[2] != "door"], SEE_RANGE)
            d_furn, fh = None, 0.0
            for (ax, az), (bx, bz), h in FURNITURE:
                d, _ = ray_hit(self.pos[0], self.pos[1], dx, dz, [((ax, az), (bx, bz), "wall")], SEE_RANGE)
                if d is not None and (d_furn is None or d < d_furn):
                    d_furn, fh = d, h
            for ri in range(rows):
                pitch = ((ri + 0.5) / rows - 0.5) * vfov
                t = math.tan(pitch)
                hit = None
                if d_furn is not None and (d_wall is None or d_furn < d_wall) and EYE + d_furn * t <= fh:
                    hit, col = d_furn, (150, 95, 55)
                elif d_wall is not None and 0 <= EYE + d_wall * t <= WALL_H:
                    hit = d_wall
                    col = COLORS[kind]
                elif t < 0:
                    hit, col = EYE / -t, (90, 90, 96)
                if hit is None or hit > SEE_RANGE:
                    continue
                n = self.rng.gauss(0, 0.02 * hit)  # depth noise grows with distance, like monocular depth
                hit += n
                x, z = self.pos[0] + dx * hit, self.pos[1] + dz * hit
                y = FLOOR_Y + EYE + hit * t
                p = self.to_shared(self.local(x, y, z))
                out.append((p[0], p[1], p[2], *col))
        return out

    def depth_frame(self, w: int = 64, h: int = 48) -> bytes:
        """A synthetic depth frame (what the iPhone's calibrated AI depth would send), vectorised with numpy."""
        f = 0.5 * w / math.tan(HFOV / 2)
        pitch = -0.17  # holding the phone slightly down, so floor and table tops are in view
        fwd = np.array([math.sin(self.yaw) * math.cos(pitch), math.sin(pitch), math.cos(self.yaw) * math.cos(pitch)])
        right = np.cross(fwd, [0, 1.0, 0])
        right /= np.linalg.norm(right)
        zc = -fwd
        up = np.cross(zc, right)
        o = np.array([self.pos[0], FLOOR_Y + EYE, self.pos[1]])
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        dcam = np.stack([(u - w / 2) / f, -(v - h / 2) / f, -np.ones_like(u)], -1).reshape(-1, 3)
        R = np.stack([right, up, zc], 1)
        d = dcam @ R.T  # world rays; t along d == camera depth
        best = np.full(len(d), np.inf)
        color = np.zeros((len(d), 3))
        # floor
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (FLOOR_Y - o[1]) / d[:, 1]
        hit = (t > 0) & (t < best)
        best[hit], color[hit] = t[hit], (95, 92, 88)
        # walls (vertical segments, doorways open)
        for (ax, az), (bx, bz), kind in BUILDING:
            if kind == "door":
                continue
            ex, ez = bx - ax, bz - az
            den = d[:, 0] * ez - d[:, 2] * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ax - o[0]) * ez - (az - o[2]) * ex) / den
                s_ = ((ax - o[0]) * d[:, 2] - (az - o[2]) * d[:, 0]) / den
            y = o[1] + t * d[:, 1]
            hit = (t > 0.05) & (t < best) & (s_ >= 0) & (s_ <= 1) & (y >= FLOOR_Y) & (y <= FLOOR_Y + WALL_H)
            best[hit], color[hit] = t[hit], COLORS[kind]
        # furniture boxes (slab test); remember which box each pixel saw, to outline them like YOLO would
        which = np.full(len(d), -1)
        for bi, (x0, z0, x1, z1, hgt) in enumerate(BOXES):
            lo = np.array([x0, FLOOR_Y, z0])
            hi = np.array([x1, FLOOR_Y + hgt, z1])
            with np.errstate(divide="ignore", invalid="ignore"):
                ta, tb = (lo - o) / d, (hi - o) / d
            tmin = np.nanmax(np.minimum(ta, tb), axis=1)
            tmax = np.nanmin(np.maximum(ta, tb), axis=1)
            hit = (tmax >= tmin) & (tmin > 0.05) & (tmin < best)
            best[hit], color[hit] = tmin[hit], (150, 95, 55)
            which[hit] = bi
        scale = 1 + self.rng.gauss(0, 0.03)  # per-frame scale error, like monocular calibration
        depth = best * scale * (1 + np.array([self.rng.gauss(0, 0.04) for _ in range(len(best))]))
        depth = np.where(np.isfinite(depth) & (depth < 6.5), depth, 0)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, zc, o
        mm = (depth * 1000).astype(np.uint16).reshape(h, w)
        grid = which.reshape(h, w)
        outlines = []
        for bi in range(len(BOXES)):
            ys, xs = np.nonzero(grid == bi)
            if len(xs) >= 6 and self.rng.random() < 0.8:  # YOLO doesn't catch every frame
                outlines.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        return pack_depth_frame(pose, (f, f, w / 2, h / 2), mm, color.astype(np.uint8).reshape(h, w, 3), boxes=outlines)

    # --- camera ---------------------------------------------------------
    def render(self, others: list["SimPhone"]) -> bytes | None:
        if Image is None:
            return None
        Wd, Ht = 180, 240
        img = Image.new("RGB", (Wd, Ht), (24, 26, 30))
        dr = ImageDraw.Draw(img)
        for y in range(Ht // 2, Ht):  # floor gradient
            g = int(40 + 50 * (y - Ht / 2) / (Ht / 2))
            dr.line([(0, y), (Wd, y)], fill=(g, g - 4, g - 10))
        f_px = (Wd / 2) / math.tan(HFOV / 2)
        horizon = Ht / 2
        depth = [99.0] * Wd
        for col in range(Wd):
            a = self.yaw + math.atan((col - Wd / 2) / f_px)
            dx, dz = math.sin(a), math.cos(a)
            d, kind = ray_hit(self.pos[0], self.pos[1], dx, dz, [s for s in BUILDING if s[2] != "door"])
            if d is None:
                continue
            perp = d * math.cos(a - self.yaw)
            depth[col] = d
            top = horizon - (WALL_H - EYE) / perp * f_px
            bot = horizon + EYE / perp * f_px
            shade = max(0.25, 1 - perp / 14)
            c = tuple(int(v * shade) for v in COLORS[kind])
            dr.line([(col, top), (col, bot)], fill=c)
            if kind == "window":
                dr.line([(col, horizon - 0.6 / perp * f_px), (col, horizon + 0.5 / perp * f_px)], fill=(int(160 * shade), int(210 * shade), 255))
        for o in others:  # colleagues as coloured capsules
            dx, dz = o.pos[0] - self.pos[0], o.pos[1] - self.pos[1]
            d = math.hypot(dx, dz)
            rel = (math.atan2(dx, dz) - self.yaw + math.pi) % (2 * math.pi) - math.pi
            if d < 0.3 or abs(rel) > HFOV / 2:
                continue
            col = int(Wd / 2 + math.tan(rel) * f_px)
            if 0 <= col < Wd and depth[col] > d:
                w = 0.25 / d * f_px
                dr.rounded_rectangle([col - w, horizon - 0.3 / d * f_px, col + w, horizon + EYE / d * f_px], radius=int(w), fill=o.rgb)
        dr.rectangle([0, 0, Wd, 16], fill=(0, 0, 0))
        dr.text((4, 2), f"{self.name} {time.strftime('%H:%M:%S')} SIM", fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=60)
        return buf.getvalue()


# ---------------- networking (TLS with certificate pinning, like the real apps) ----------------


def pinned_context() -> ssl.SSLContext:
    """No CA check (self-signed); trust comes from comparing the fingerprint ourselves."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def check_pin(ssl_object, fingerprint: str) -> None:
    der = ssl_object.getpeercert(binary_form=True)
    if hashlib.sha256(der).hexdigest() != fingerprint:
        raise ssl.SSLError("certificate fingerprint mismatch: not our server")


def admin_request(admin: str, token: str, method: str, path: str) -> dict:
    req = urllib.request.Request(f"{admin}{path}", method=method, headers={"Cookie": f"sl_admin={token}", "X-Sightline": "1"})
    return json.load(urllib.request.urlopen(req, timeout=5))


def pair_phone(ph: SimPhone, admin: str, token: str) -> None:
    """Exactly what a phone does after scanning the QR code."""
    q = urllib.parse.parse_qs(urllib.parse.urlparse(admin_request(admin, token, "POST", "/api/admin/pair")["uri"]).query)
    host, port, code, fp = q["h"][0], int(q["p"][0]), q["c"][0], q["f"][0]
    if ph.host_override:
        host = ph.host_override
    conn = http.client.HTTPSConnection(host, port, context=pinned_context(), timeout=5)
    conn.connect()
    check_pin(conn.sock, fp)
    conn.request("POST", "/api/pair", json.dumps({"code": code, "name": ph.name, "platform": "sim"}), {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        raise RuntimeError(f"pairing failed: {r.status} {r.read()[:200]!r}")
    d = json.loads(r.read())
    ph.id, ph.token, ph.server, ph.fingerprint = d["device_id"], d["token"], f"{host}:{port}", fp


async def run_phone(ph: SimPhone, everyone: list[SimPhone], frames: bool, stop: asyncio.Event, drift: float) -> None:
    ctx = pinned_context()
    async with websockets.connect(f"wss://{ph.server}/ws/phone", ssl=ctx, max_size=2**23,
                                  additional_headers={"Authorization": f"Bearer {ph.token}"}) as ws:
        check_pin(ws.transport.get_extra_info("ssl_object"), ph.fingerprint)
        await ws.send(json.dumps({"t": "status", "aligned": False, "tracking": "normal", "battery": round(ph.rng.uniform(0.4, 1), 2)}))
        await asyncio.sleep(0.3 + ph.rng.random())
        ph.scan_marker()  # walk past the marker at the entrance
        await ws.send(json.dumps({"t": "status", "aligned": True, "tracking": "normal"}))

        async def receiver():
            async for raw in ws:
                if isinstance(raw, str):
                    m = json.loads(raw)
                    if m.get("t") == "state":
                        ph.peers = m["peers"]
                else:
                    ph.coarse_bytes += len(raw)

        rx = asyncio.create_task(receiver())
        last, last_frame, last_planes, last_points = time.time(), 0.0, 0.0, 0.0
        try:
            while not stop.is_set():
                now = time.time()
                ph.step(now - last)
                if drift:  # slowly rotate the phone's private frame -> visible drift on the map
                    ph.L = mat_mul(yaw_rotation(drift * (now - last)), ph.L)
                last = now
                await ws.send(json.dumps(ph.pose_msg()))
                if ph.fallen and not getattr(ph, "impact_sent", False):  # accelerometer: free fall then impact
                    ph.impact_sent = True
                    await ws.send(json.dumps({"t": "impact", "g": 3.6, "freefall_ms": 240}))
                if now - last_planes > 0.5:
                    last_planes = now
                    for m in ph.discover_planes():
                        await ws.send(json.dumps(m))
                    x = ph.pick_xray()
                    if x != ph.xray:
                        ph.xray = x
                        await ws.send(json.dumps({"t": "xray", "target": x}))
                if now - last_points > 0.5:
                    last_points = now
                    pts = ph.depth_points()
                    if pts:
                        await ws.send(pack_points(pts))
                    await ws.send(ph.depth_frame())
                if frames and now - last_frame > 0.33:
                    last_frame = now
                    jpg = ph.render([o for o in everyone if o is not ph])
                    if jpg:
                        await ws.send(b"\x01" + jpg)
                await asyncio.sleep(0.1)
        finally:
            rx.cancel()


def check(admin: str, token: str, phones: list[SimPhone], scenario: str) -> bool:
    req = urllib.request.Request(f"{admin}/api/admin/state", headers={"Cookie": f"sl_admin={token}"})
    st = json.load(urllib.request.urlopen(req, timeout=5))
    by_id = {p["id"]: p for p in st["phones"]}
    ok = True
    for ph in phones:
        p = by_id.get(ph.id)
        if not p or not p["pos"]:
            print(f"FAIL {ph.name}: no position on server")
            ok = False
            continue
        err = math.dist(p["pos"], ph.truth)
        good = err < 0.25  # pose is ~0.1 s old + marker noise
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {ph.name}: server pos {[round(v, 2) for v in p['pos']]} vs truth {[round(v, 2) for v in ph.truth]} (err {err * 100:.1f} cm)")
    walls = [pl for pl in st["planes"] if pl["kind"] == "wall"]
    print(f"{'ok  ' if walls else 'FAIL'} {len(st['planes'])} planes on server ({len(walls)} walls)")
    ok &= bool(walls)
    got = sum(ph.coarse_bytes for ph in phones)
    print(f"{'ok  ' if got else 'FAIL'} phones received {got} bytes of shared 3D points")
    ok &= got > 0
    if len(phones) > 1:
        worst = 0.0
        for pid in set(phones[0].sent_planes) & set(phones[1].sent_planes):
            a, b = phones[0].sent_planes[pid], phones[1].sent_planes[pid]
            worst = max(worst, max(math.dist(x, y) for x, y in zip(a, b)))
        print(f"{'ok  ' if worst < 0.15 else 'FAIL'} same wall seen by two phones agrees within {worst * 100:.1f} cm")
        ok &= worst < 0.15
    fu = st.get("fusion", {})
    good = fu.get("frames", 0) > 20 and fu.get("meshes", 0) > 10
    print(f"{'ok  ' if good else 'FAIL'} fusion: {fu.get('frames')} depth frames -> {fu.get('blocks')} blocks, {fu.get('meshes')} meshes, {fu.get('depth_walls')} walls from depth")
    ok &= good
    if scenario == "tour":
        objs = st.get("objects", [])
        def near(o, box):
            cx, cz = (o["x0"] + o["x1"]) / 2, (o["z0"] + o["z1"]) / 2
            return box[0] - 0.4 <= cx <= box[2] + 0.4 and box[1] - 0.4 <= cz <= box[3] + 0.4
        tables = [b for b in BOXES if b[4] > 0.6]
        found_t = sum(1 for b in tables if any(o["kind"] == "table" and near(o, b) for o in objs))
        seats = [b for b in BOXES if b[4] < 0.6]
        found_s = sum(1 for b in seats if any(o["kind"] in ("seat", "table", "object") and near(o, b) for o in objs))
        spurious = [o for o in objs if o["kind"] in ("table", "seat") and not any(near(o, b) for b in BOXES)]
        conf = sum(1 for o in objs if o.get("confirmed"))
        shown = [o for o in objs if o.get("confirmed") or o["kind"] != "obstacle"]
        print(f"{'ok  ' if found_t >= 2 else 'FAIL'} furniture: {found_t}/{len(tables)} tables, {found_s}/{len(seats)} chairs found; "
              f"{len(spurious)} spurious; {conf} confirmed by outlines; {len(shown)} drawn of {len(objs)}")
        ok &= found_t >= 2
    if scenario == "mandown":
        dana = by_id.get(phones[-1].id, {})
        others = [by_id.get(p.id, {}).get("alert") for p in phones[:-1]]
        good = (dana.get("alert") or {}).get("kind") == "fall" and not any(others)
        print(f"{'ok  ' if good else 'FAIL'} fall alert for {phones[-1].name}: {dana.get('alert')}; others: {others}")
        ok &= good
    if st.get("coverage_pct") is not None or scenario != "xray":
        pct = st.get("coverage_pct")
        good = pct is not None and pct > 0
        print(f"{'ok  ' if good else 'FAIL'} search coverage {pct}% of the building outline")
        ok &= good
    if scenario == "xray":
        a, b = phones[0], phones[1]
        target = by_id.get(a.id, {}).get("xray")
        peer = next((p for p in a.peers if p["id"] == b.id), {})
        good = target == b.id and peer.get("walls_between", 0) >= 1
        print(f"{'ok  ' if good else 'FAIL'} {a.name} x-ray target={target}, walls between={peer.get('walls_between')}")
        ok &= good
    print("PASS" if ok else "FAIL")
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=None, help="phone-side server address (default: from the QR payload)")
    ap.add_argument("--admin", default="http://127.0.0.1:8000", help="commander listener, used to mint pairing codes")
    ap.add_argument("--admin-token", default=os.environ.get("SIGHTLINE_ADMIN_TOKEN"), help="admin token printed by the server (or $SIGHTLINE_ADMIN_TOKEN)")
    ap.add_argument("--phones", type=int, default=3, help="number of touring phones (max 4)")
    ap.add_argument("--scenario", choices=["tour", "xray", "mandown"], default="tour")
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = forever)")
    ap.add_argument("--no-frames", action="store_true", help="don't send camera frames")
    ap.add_argument("--drift", type=float, default=0.0, help="simulated ARKit yaw drift, rad/s")
    ap.add_argument("--check", action="store_true", help="verify server state at the end, exit 1 on failure")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    if a.scenario == "xray":
        # Alice in the corridor facing the left wall; Bob in the room behind it; Chen walking.
        phones = [
            SimPhone("Alice", [(0, 6)], hold=(0.2, 6.0), look_at=(-4, 6.5), rng=random.Random(rng.random())),
            SimPhone("Bob", [(-4, 6.5)], hold=(-4.0, 6.5), look_at=(-5.5, 3), rng=random.Random(rng.random())),
            SimPhone("Chen", TOURS["Chen"], rng=random.Random(rng.random())),
        ]
    elif a.scenario == "mandown":
        # Alice and Bob tour; Dana walks in and falls after 6 s
        phones = [SimPhone(n, TOURS[n], speed=1.0, rng=random.Random(rng.random())) for n in ("Alice", "Bob")]
        phones.append(SimPhone("Dana", TOURS["Dana"], speed=1.0, fall_at=6.0, rng=random.Random(rng.random())))
    else:
        names = list(TOURS)[: max(1, min(4, a.phones))]
        phones = [SimPhone(n, TOURS[n], speed=0.9 + 0.2 * i, rng=random.Random(rng.random())) for i, n in enumerate(names)]
    palette = [(255, 90, 95), (63, 167, 255), (255, 180, 0), (46, 204, 113)]
    for i, ph in enumerate(phones):
        ph.rgb = palette[i % 4]

    if not a.admin_token:
        ap.error("--admin-token is required (it is printed by the server at startup)")
    for ph in phones:
        ph.host_override = a.host
        pair_phone(ph, a.admin.rstrip("/"), a.admin_token)
    stop = asyncio.Event()
    tasks = [asyncio.create_task(run_phone(ph, phones, not a.no_frames, stop, a.drift)) for ph in phones]
    print(f"{len(phones)} sim phones paired over pinned TLS -> {phones[0].server} ({a.scenario}). Ctrl-C to stop.")
    try:
        if a.duration:
            await asyncio.sleep(a.duration)
            ok = check(a.admin.rstrip("/"), a.admin_token, phones, a.scenario) if a.check else True
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            return 0 if ok else 1
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        stop.set()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
