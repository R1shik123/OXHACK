"""Photo-textured walls and floor, painted from the phones' real camera snapshots.

The 3D mesh can only colour each 4 cm corner once, which blurs posters, doors, skirting and floor
patterns. Walls and the floor are flat and we know exactly where they are (merged walls + floor height),
so each gets its own image instead: every texel (2.5 cm on walls, 4 cm on the floor) is projected into
the camera snapshots (0x05, sent with their pose) and takes the colour of the views that saw it most
directly and closest. Walls between the camera and a texel block it.

Honesty rules:
  - Only colours a camera actually recorded are used. Texels nobody saw are filled from the nearest seen
    texels ("push-pull" fill) so surfaces look solid; `seen` keeps which are measured and which are filled.
  - Until a snapshot covers a spot, the colour comes from the 3D scan's own voxel colours (also real).
Runs on the laptop; a few MB of memory per building floor.
"""

from __future__ import annotations

import io
import logging
import math
import struct
import threading
import zlib

import numpy as np

log = logging.getLogger("sightline")
WALL_PPM = 40  # texels per metre on walls (2.5 cm)
FLOOR_PPM = 25  # texels per metre on the floor (4 cm)
WALL_H = 2.6  # metres of wall above the floor that get a texture
MAX_RANGE = 5.5  # metres: farther, a 360 px snapshot is too coarse
MSG = struct.Struct("<BBI12f")  # 0x06 | kind (1 wall, 2 floor) | id | 4 corners xyz (bottom-left, bottom-right, top-right, top-left)


def push_pull(acc: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fill texels with no weight from their nearest weighted neighbours (coarse-to-fine)."""
    out = np.where(w[..., None] > 0, acc / np.maximum(w, 1e-6)[..., None], 0)
    if (w > 0).all() or not (w > 0).any():
        return out
    levels = [(out * (w > 0)[..., None], (w > 0).astype(np.float32))]
    while min(levels[-1][1].shape) > 1:
        c, k = levels[-1]
        H, W = k.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        cp = np.zeros((H2 * 2, W2 * 2, 3), np.float32)
        kp = np.zeros((H2 * 2, W2 * 2), np.float32)
        cp[:H, :W], kp[:H, :W] = c, k
        cs = cp.reshape(H2, 2, W2, 2, 3).sum(axis=(1, 3))
        ks = kp.reshape(H2, 2, W2, 2).sum(axis=(1, 3))
        levels.append((cs, ks))
    fill = levels[-1][0] / np.maximum(levels[-1][1], 1e-6)[..., None]
    for c, k in reversed(levels[:-1]):
        H, W = k.shape
        up = np.repeat(np.repeat(fill, 2, axis=0), 2, axis=1)[:H, :W]
        fill = np.where(k[..., None] > 0, c / np.maximum(k, 1e-6)[..., None], up)
    return np.where(w[..., None] > 0, out, fill)


class Surface:
    """One textured flat surface: a wall (along a line, from the floor up) or the floor (a box)."""

    def __init__(self, sid: int, kind: str, origin, u, v, nu: int, nv: int, normal) -> None:
        self.id, self.kind = sid, kind
        self.origin, self.u, self.v = np.array(origin, float), np.array(u, float), np.array(v, float)  # u, v: metres per texel
        self.nu, self.nv = nu, nv
        self.normal = np.array(normal, float)
        self.acc = np.zeros((nv, nu, 3), np.float32)
        self.w = np.zeros((nv, nu), np.float32)
        self.seeded = False
        self.version = 0
        self.sent = -1
        ii, jj = np.meshgrid(np.arange(nu) + 0.5, np.arange(nv) + 0.5)
        self.pts = self.origin + ii[..., None] * self.u + jj[..., None] * self.v  # texel centres, world

    def corners(self) -> list[float]:
        U, V = self.u * self.nu, self.v * self.nv
        o = self.origin
        return [*o, *(o + U), *(o + U + V), *(o + V)]


class Textures:
    def __init__(self, frames_from, fusion) -> None:
        self.frames_from = frames_from  # ObjectConfirmer: keeps each phone's latest posed snapshot
        self.fusion = fusion
        self.surfaces: dict[tuple, Surface] = {}
        self.walls2d: list = []
        self.done: set[float] = set()  # snapshot timestamps already painted in
        self.lock = threading.Lock()
        self._next = 1

    # ---------------- geometry ----------------
    def set_geometry(self, walls: list, floor_y: float | None, extent) -> None:
        """walls: [((x1, z1), (x2, z2))]. Keeps the paint of surfaces whose geometry didn't change."""
        if floor_y is None:
            return
        keep: dict[tuple, Surface] = {}
        for (ax, az), (bx, bz) in walls:
            L = math.hypot(bx - ax, bz - az)
            if L < 0.4:
                continue
            key = ("w", round(ax, 1), round(az, 1), round(bx, 1), round(bz, 1), round(floor_y, 2))
            s = self.surfaces.get(key)
            if s is None:
                nu, nv = max(2, int(L * WALL_PPM)), int(WALL_H * WALL_PPM)
                ux, uz = (bx - ax) / L, (bz - az) / L
                s = Surface(self._new_id(), "wall", (ax, floor_y, az), (ux * L / nu, 0, uz * L / nu), (0, WALL_H / nv, 0), nu, nv, (-uz, 0, ux))
            keep[key] = s
        if extent is not None:
            x0, z0, x1, z1 = (round(v, 1) for v in extent)
            key = ("f", x0, z0, x1, z1, round(floor_y, 2))
            s = self.surfaces.get(key)
            if s is None:
                nu, nv = max(2, int((x1 - x0) * FLOOR_PPM)), max(2, int((z1 - z0) * FLOOR_PPM))
                old = next((o for k, o in self.surfaces.items() if k[0] == "f"), None)
                s = Surface(old.id if old else self._new_id(), "floor", (x0, floor_y, z0), ((x1 - x0) / nu, 0, 0), (0, 0, (z1 - z0) / nv), nu, nv, (0, 1, 0))
                if old is not None:  # the floor grew: carry over what was already painted
                    self._carry(old, s)
            keep[key] = s
        with self.lock:
            self.surfaces = keep
            self.walls2d = [((ax, az), (bx, bz)) for (ax, az), (bx, bz) in walls]

    def _new_id(self) -> int:
        self._next += 1
        return self._next

    @staticmethod
    def _carry(old: Surface, new: Surface) -> None:
        ox = np.round((old.pts[0, :, 0] - new.origin[0]) / new.u[0] - 0.5).astype(int)
        oz = np.round((old.pts[:, 0, 2] - new.origin[2]) / new.v[2] - 0.5).astype(int)
        okx, okz = (ox >= 0) & (ox < new.nu), (oz >= 0) & (oz < new.nv)
        if okx.any() and okz.any():
            new.acc[np.ix_(oz[okz], ox[okx])] = old.acc[np.ix_(okz, okx)]
            new.w[np.ix_(oz[okz], ox[okx])] = old.w[np.ix_(okz, okx)]
            new.seeded = old.seeded

    # ---------------- painting ----------------
    def paint(self, max_frames: int = 3) -> int:
        """Paint the freshest unseen snapshots into all surfaces. Returns how many surfaces changed."""
        from PIL import Image

        with self.lock:
            surfaces = list(self.surfaces.values())
            walls = list(self.walls2d)
        if not surfaces:
            return 0
        for s in surfaces:
            if not s.seeded:
                self._seed(s)
        with self.frames_from.lock:
            frames = [f for f in self.frames_from.frames.values() if f["ts"] not in self.done]
        frames = sorted(frames, key=lambda f: -f["ts"])[:max_frames]
        changed: set[int] = set()
        for fr in frames:
            self.done.add(fr["ts"])
            try:
                img = np.asarray(Image.open(io.BytesIO(fr["jpeg"])).convert("RGB"), np.float32)
            except Exception as e:
                log.info("texture: skipped a snapshot: %s", e)
                continue
            for s in surfaces:
                if self._paint_one(s, fr, img, walls):
                    changed.add(s.id)
        if len(self.done) > 400:
            self.done = set(sorted(self.done)[-200:])
        return len(changed)

    def _seed(self, s: Surface) -> None:
        """Start from the 3D scan's own (real) voxel colours, at a low weight that any photo outvotes."""
        rgb, ok = self.fusion.sample_colors(s.pts.reshape(-1, 3), reach=0.1)
        if ok.any():
            s.acc.reshape(-1, 3)[ok] += rgb[ok] * 0.02
            s.w.reshape(-1)[ok] += 0.02
            s.version += 1
        s.seeded = True

    def _paint_one(self, s: Surface, fr: dict, img: np.ndarray, walls: list) -> bool:
        pose, (fx, fy, cx, cy), (ws, hs) = fr["pose"], fr["K"], fr["size"]
        cam = pose[:3, 3]
        if np.min(np.linalg.norm(s.pts[:: max(1, s.nv // 8), :: max(1, s.nu // 8)] - cam, axis=2)) > MAX_RANGE:
            return False
        inv = np.linalg.inv(pose)
        P = s.pts.reshape(-1, 3)
        pc = P @ inv[:3, :3].T + inv[:3, 3]
        d = -pc[:, 2]
        ray = P - cam
        dist = np.linalg.norm(ray, axis=1)
        cosi = np.abs(ray @ s.normal) / np.maximum(dist, 1e-6)
        ok = (d > 0.3) & (dist < MAX_RANGE) & (cosi > 0.25)
        if not ok.any():
            return False
        u = fx * pc[:, 0] / np.maximum(d, 1e-6) + cx  # landscape sensor pixels
        v = -fy * pc[:, 1] / np.maximum(d, 1e-6) + cy
        x, y = hs - v, u  # upright portrait snapshot: x = h - v, y = u
        H, W = img.shape[:2]
        sx, sy = x * W / hs, y * H / ws
        ok &= (sx >= 0) & (sx < W - 1) & (sy >= 0) & (sy < H - 1)
        if not ok.any():
            return False
        idx = np.nonzero(ok)[0]
        idx = idx[~self._blocked(cam, P[idx], walls, s)]
        if not len(idx):
            return False
        col = img[sy[idx].astype(int), sx[idx].astype(int)]
        wt = (cosi[idx] ** 2 / np.maximum(dist[idx], 0.5) ** 2).astype(np.float32)
        acc, w = s.acc.reshape(-1, 3), s.w.reshape(-1)
        known = w[idx] > 0.5  # exposure / white balance against what this surface already looks like
        if known.sum() > 100:
            have = acc[idx[known]] / w[idx[known], None]
            gain = np.clip(np.median(have, axis=0) / np.maximum(np.median(col[known], axis=0), 1.0), 0.7, 1.45)
            col = np.clip(col * gain, 0, 255)
        np.add.at(acc, idx, col * wt[:, None])
        np.add.at(w, idx, wt)
        s.version += 1
        return True

    @staticmethod
    def _blocked(cam, pts: np.ndarray, walls: list, s: Surface) -> np.ndarray:
        """True where another wall stands between the camera and the texel (top-down 2D test)."""
        blocked = np.zeros(len(pts), bool)
        ox, oz = cam[0], cam[2]
        px, pz = pts[:, 0], pts[:, 2]
        for (ax, az), (bx, bz) in walls:
            if s.kind == "wall" and abs((ax - s.origin[0]) * s.normal[0] + (az - s.origin[2]) * s.normal[2]) < 0.05 \
                    and abs((bx - s.origin[0]) * s.normal[0] + (bz - s.origin[2]) * s.normal[2]) < 0.05:
                continue  # the wall itself (or a piece on the same line)
            rx, rz = px - ox, pz - oz
            ex, ez = bx - ax, bz - az
            den = rx * ez - rz * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ax - ox) * ez - (az - oz) * ex) / den
                q = ((ax - ox) * rz - (az - oz) * rx) / den
            blocked |= (np.abs(den) > 1e-9) & (t > 0.02) & (t < 0.98) & (q >= 0) & (q <= 1)
        return blocked

    # ---------------- output ----------------
    def messages(self, force: bool = False) -> list[bytes]:
        """0x06 messages for surfaces that changed since last sent (all of them with force=True)."""
        from PIL import Image

        out = []
        with self.lock:
            surfaces = list(self.surfaces.values())
        for s in surfaces:
            if not force and s.sent == s.version:
                continue
            rgb = push_pull(s.acc, s.w)
            img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)[::-1])  # image row 0 = top / far edge
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            out.append(MSG.pack(0x06, 1 if s.kind == "wall" else 2, s.id, *s.corners()) + buf.getvalue())
            if not force:
                s.sent = s.version
        return out

    def ids(self) -> set[int]:
        with self.lock:
            return {s.id for s in self.surfaces.values()}
