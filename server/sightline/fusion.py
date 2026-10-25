"""Live 3D surface fusion (TSDF) from every phone's depth frames.

Each depth frame updates a sparse truncated-signed-distance volume (4 cm voxels in 16^3 blocks,
allocated only near surfaces). Averaging many noisy monocular depth maps converges on the real
surfaces, which is what makes plain walls, tables and chairs appear. Dirty blocks are meshed with
marching cubes and streamed as compact binary meshes. From the same volume we derive walls
(tall vertical surfaces) and furniture (surfaces at seat / table height) for the map.
"""

from __future__ import annotations

import math
import struct
import threading
from collections import deque

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.measure import marching_cubes

VOXEL = 0.04  # metres
BLOCK = 16  # voxels per block edge (0.64 m)
TRUNC = 0.16  # truncation distance (m): generous, monocular depth is noisy
MAX_DEPTH = 5.0  # ignore depth beyond this (accuracy falls off)
MAX_BLOCKS = 60000  # ~250 MB at most; plenty for a large floor
MESH_MIN_WEIGHT = 0.45  # only mesh voxels seen consistently (several frames): clean surfaces, no speckle
MESH_REC = struct.Struct("<BhhhHH")  # 0x04 | block x,y,z | n_verts | n_faces
COLUMN = 0.1  # metres: 2D grid used to classify surfaces into walls / furniture
FURNITURE_BOOST = 2.5  # depth inside a YOLO furniture outline counts this much more: sharper furniture, sooner
COLOR_BAND = 0.06  # metres either side of the seen surface that take its colour
SNAP = 0.06  # metres: surface within this of a known wall / the floor is laid exactly onto it (flat, level)
HOLE_FILL = 2  # voxels: small holes next to confident surface are closed from weakly-seen voxels
LIDAR_TRUST = 4.0  # weight of LiDAR depth relative to the monocular AI estimate
FINE_VOXEL, FINE_RANGE, FINE_BLOCKS = 0.02, 1.6, 3000  # 2 cm detail for what phones saw up close (<= ~240 MB)
FURNITURE_MEMORY = 600  # furniture sightings kept to confirm fused boxes


class Fusion:
    def __init__(self, voxel: float | None = None, trunc: float | None = None, max_depth: float | None = None,
                 max_blocks: int | None = None, mesh_type: int = 0x04) -> None:
        self.voxel = voxel or VOXEL  # metres per voxel (blocks are always 16 voxels across)
        self.trunc = trunc or TRUNC
        self.max_depth = max_depth or MAX_DEPTH
        self.max_blocks = max_blocks or MAX_BLOCKS
        self.mesh_type = mesh_type  # 0x04 normal 4 cm meshes, 0x07 fine 2 cm meshes
        self.covered = None  # optional: points -> mask of places a finer layer already shows (we step aside)
        self.solid: set = set()  # blocks whose mesh has real content (a coarser layer leaves these to us)
        self.tsdf: dict[tuple[int, int, int], np.ndarray] = {}
        self.weight: dict[tuple[int, int, int], np.ndarray] = {}
        self.color: dict[tuple[int, int, int], np.ndarray] = {}
        self.dirty: set[tuple[int, int, int]] = set()
        self.lock = threading.Lock()
        self.queue: deque = deque(maxlen=8)  # newest frames win when we fall behind
        self.frames = 0
        self.meshes: dict[tuple[int, int, int], bytes] = {}  # latest mesh per block, for new viewers
        self.candidates: list[dict] = []  # wall gaps / pillar-like shapes for the laptop's scene AI
        # known flat surfaces (from the merged walls and floor), used to flatten the mesh. When textured
        # walls / floor are drawn (texture.py), the mesh triangles lying on them are left out.
        self.planes: dict = {"floor": None, "walls": [], "extent": None}
        self.drop_planar = False
        self.furniture_seen: deque = deque(maxlen=FURNITURE_MEMORY)
        self.doors_seen: deque = deque(maxlen=300)  # (x, z) floor positions where the phone's detector saw a door
        self.stairs_seen: deque = deque(maxlen=300)  # ... and stairs
        self.stairs: list[dict] = []  # stair flights found by the last semantics pass
        self.drops: list = []  # (x, z) floor edges that fall away, from the last semantics pass  # (x, z) floor positions YOLO saw furniture at
        bi = np.indices((BLOCK, BLOCK, BLOCK)).reshape(3, -1).T  # voxel offsets inside a block
        self._offsets = (bi + 0.5) * self.voxel

    # ---------------- integration ----------------
    def integrate(self, frame: dict) -> None:
        depth = frame["depth"].astype(np.float32) / 1000.0
        h, w = depth.shape
        fx, fy, cx, cy = frame["K"]
        pose = frame["pose"]
        valid = (depth > 0.25) & (depth < self.max_depth)
        if valid.sum() < 50:
            return
        boost = np.ones((h, w), np.float32)  # YOLO furniture outlines: trust depth there more
        for x0, y0, x1, y1 in frame.get("boxes", []):
            boost[y0:y1, x0:x1] = FURNITURE_BOOST
            patch = depth[y0:y1, x0:x1]
            ok = (patch > 0.25) & (patch < self.max_depth)
            if ok.sum() >= 4:  # where that piece of furniture is on the floor plan
                z = float(np.median(patch[ok]))
                u, v = (x0 + x1) / 2, (y0 + y1) / 2
                p3 = pose @ np.array([(u - cx) / fx * z, -(v - cy) / fy * z, -z, 1.0])
                self.furniture_seen.append((float(p3[0]), float(p3[2])))
        for kind, boxes in (("doors", frame.get("doors", [])), ("stairs", frame.get("stairs", []))):
            for x0, y0, x1, y1 in boxes:  # where a doorway / a flight of stairs is on the floor plan
                patch = depth[y0:y1, x0:x1]
                ok = (patch > 0.25) & (patch < self.max_depth)
                if ok.sum() >= 4:
                    z = float(np.median(patch[ok]))
                    u, v = (x0 + x1) / 2, (y0 + y1) / 2
                    p3 = pose @ np.array([(u - cx) / fx * z, -(v - cy) / fy * z, -z, 1.0])
                    (self.doors_seen if kind == "doors" else self.stairs_seen).append((float(p3[0]), float(p3[2])))
        # surface points -> which blocks to touch (plus a truncation-sized margin along the ray)
        v, u = np.nonzero(valid)
        z = depth[v, u]
        pc = np.stack([(u + 0.5 - cx) / fx * z, -(v + 0.5 - cy) / fy * z, -z, np.ones_like(z)])
        pts = (pose @ pc)[:3].T
        cam = pose[:3, 3]
        ray = pts - cam
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        keys = set()
        for t in (-self.trunc, 0.0, self.trunc):
            q = np.floor((pts + ray * t) / (self.voxel * BLOCK)).astype(np.int32)
            keys.update((int(a), int(b), int(c)) for a, b, c in np.unique(q, axis=0))
        with self.lock:
            keys = [k for k in keys if k in self.tsdf or len(self.tsdf) < self.max_blocks]
            if not keys:
                return
            for k in keys:
                if k not in self.tsdf:
                    self.tsdf[k] = np.ones(BLOCK ** 3, np.float32)
                    self.weight[k] = np.zeros(BLOCK ** 3, np.float32)
                    self.color[k] = np.zeros((BLOCK ** 3, 3), np.float32)
            origins = np.array(keys, np.float64) * self.voxel * BLOCK
            world = (origins[:, None, :] + self._offsets[None]).reshape(-1, 3)  # all voxel centres
            inv = np.linalg.inv(pose)
            pc = world @ inv[:3, :3].T + inv[:3, 3]
            zc = -pc[:, 2]
            front = zc > 0.1
            uu = np.full(zc.shape, -1, np.int32)
            vv = np.full(zc.shape, -1, np.int32)
            uu[front] = np.floor(fx * pc[front, 0] / zc[front] + cx).astype(np.int32)
            vv[front] = np.floor(-fy * pc[front, 1] / zc[front] + cy).astype(np.int32)
            inside = front & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            d = np.zeros(zc.shape, np.float32)
            d[inside] = depth[vv[inside], uu[inside]]
            sdf = d - zc
            upd = inside & (d > 0) & (sdf > -self.trunc)
            if not upd.any():
                return
            tsdf_new = np.clip(sdf / self.trunc, -1, 1)
            # weight: trust near measurements more (monocular error grows ~ with depth^2)
            wnew = np.where(upd, 1.0 / np.maximum(d, 0.5) ** 2, 0).astype(np.float32)
            if frame.get("lidar"):  # measured by a LiDAR sensor, not estimated: outvotes the AI depth
                wnew *= LIDAR_TRUST
            wnew[inside] *= boost[vv[inside], uu[inside]]
            col = np.zeros((len(zc), 3), np.float32)
            col[upd] = frame["rgb"][vv[upd], uu[upd]]
            # colour only voxels right at the seen surface (voxels far behind it would take the colour
            # of whatever is in front, which smears edges into a muddy average)
            paint = upd & (np.abs(sdf) < COLOR_BAND)
            # exposure / white balance: phones and frames see the same wall brighter or bluer; scale this
            # frame's colours to match what the volume already holds there, so blended colours stay true
            cur = np.concatenate([self.color[k] for k in keys])
            wcur = np.concatenate([self.weight[k] for k in keys])
            ref = paint & (wcur > 3) & (col.sum(axis=1) > 30) & (cur.sum(axis=1) > 30)
            if ref.sum() > 200:
                gain = np.clip(np.median(cur[ref], axis=0) / np.maximum(np.median(col[ref], axis=0), 1.0), 0.7, 1.45)
                col[paint] = np.clip(col[paint] * gain, 0, 255)
            n = BLOCK ** 3
            for i, k in enumerate(keys):
                sl = slice(i * n, (i + 1) * n)
                m = upd[sl]
                if not m.any():
                    continue
                W0, Wn = self.weight[k][m], wnew[sl][m]
                Wt = W0 + Wn
                self.tsdf[k][m] = (self.tsdf[k][m] * W0 + tsdf_new[sl][m] * Wn) / Wt
                pm = paint[sl][m]
                if pm.any():
                    cw = (Wn / Wt)[pm][:, None]
                    c = self.color[k][m]
                    c[pm] = c[pm] * (1 - cw) + col[sl][m][pm] * cw
                    self.color[k][m] = c
                self.weight[k][m] = np.minimum(Wt, 60.0)  # cap: the volume can still adapt if things move
                self.dirty.add(k)
            self.frames += 1

    # ---------------- meshing ----------------
    def _grid(self, k, field: str) -> np.ndarray | None:
        src = getattr(self, field).get(k)
        return None if src is None else src.reshape(BLOCK, BLOCK, BLOCK, *src.shape[1:])

    def mesh_block(self, k) -> bytes | None:
        """Marching cubes on one block plus a 1-voxel border from its neighbours -> 0x04 message."""
        with self.lock:
            if k not in self.tsdf:
                return None
            N = BLOCK + 1
            vol = np.ones((N, N, N), np.float32)
            wt = np.zeros((N, N, N), np.float32)
            col = np.zeros((N, N, N, 3), np.float32)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        nk = (k[0] + dx, k[1] + dy, k[2] + dz)
                        t = self._grid(nk, "tsdf")
                        if t is None:
                            continue
                        sx = slice(0, BLOCK) if dx == 0 else slice(0, 1)
                        sy = slice(0, BLOCK) if dy == 0 else slice(0, 1)
                        sz = slice(0, BLOCK) if dz == 0 else slice(0, 1)
                        ox, oy, oz = dx * BLOCK, dy * BLOCK, dz * BLOCK
                        dst = (slice(ox, ox + (BLOCK if dx == 0 else 1)), slice(oy, oy + (BLOCK if dy == 0 else 1)), slice(oz, oz + (BLOCK if dz == 0 else 1)))
                        vol[dst] = t[sx, sy, sz]
                        wt[dst] = self._grid(nk, "weight")[sx, sy, sz]
                        col[dst] = self._grid(nk, "color")[sx, sy, sz]
        known = wt > MESH_MIN_WEIGHT
        if known.any():  # close small holes: weakly-seen voxels right next to confident surface join in
            known = known | (binary_dilation(known, iterations=HOLE_FILL) & (wt > 0.02))
        if known.sum() >= 8:  # light smoothing of the distance field = calmer surfaces from noisy depth
            sm = gaussian_filter(np.where(known, vol, 1.0), sigma=0.85)
            vol = np.where(known, sm, vol)
        if known.sum() < 8 or vol[known].min() > 0 or vol[known].max() < 0:
            return self._pack(k, np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3)))
        try:
            verts, faces, normals, _ = marching_cubes(vol, level=0.0, mask=known, allow_degenerate=False)
        except (ValueError, RuntimeError):
            return self._pack(k, np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3)))
        colors = self._vertex_colors(col, wt, verts)
        verts = _taubin(verts, faces)  # smooth the staircase of 4 cm voxels; block borders stay put (no seams)
        world = (np.array(k) * BLOCK + verts + 0.5) * self.voxel
        world, snapped = self._flatten(world)
        if self.drop_planar and len(faces):  # textured walls / floor replace these triangles
            faces = faces[~snapped[faces].all(axis=1)]
        if self.covered is not None and len(faces):  # the fine 2 cm layer shows this spot already
            faces = faces[~self.covered(world[faces].mean(axis=1))]
        if len(faces) >= 20:
            self.solid.add(k)
        else:
            self.solid.discard(k)
        return self._pack(k, world, faces, colors, normals)

    def covers(self, pts: np.ndarray) -> np.ndarray:
        """True where this layer has a solid mesh block (used by the coarser layer to avoid double surfaces)."""
        if not self.solid or not len(pts):
            return np.zeros(len(pts), bool)
        bk = np.floor(pts / (self.voxel * BLOCK)).astype(np.int64)
        solid = self.solid
        return np.fromiter(((int(a), int(b), int(c)) in solid for a, b, c in bk), bool, len(bk))

    def sample_colors(self, pts: np.ndarray, reach: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Colour of the nearest painted voxel at each point (searching up to `reach` m along x, z, then y,
        since a wall or floor surface may sit a few cm off the exact plane). Returns (rgb N x 3, found)."""
        out = np.zeros((len(pts), 3), np.float32)
        found = np.zeros(len(pts), bool)
        offsets = [(None, 0.0)] + [(axis, d) for axis in (0, 2, 1) for r in np.arange(self.voxel, reach + 1e-9, self.voxel) for d in (r, -r)]
        with self.lock:
            for axis, dd in offsets:
                todo = np.nonzero(~found)[0]
                if not len(todo):
                    break
                q = pts[todo].copy()
                if axis is not None:
                    q[:, axis] += dd
                vi = np.floor(q / self.voxel).astype(np.int64)
                bk = np.floor_divide(vi, BLOCK)
                li = vi - bk * BLOCK
                flat = (li[:, 0] * BLOCK + li[:, 1]) * BLOCK + li[:, 2]
                kid = ((bk[:, 0] + 4096) * 8192 + (bk[:, 1] + 4096)) * 8192 + (bk[:, 2] + 4096)
                order = np.argsort(kid, kind="stable")
                uk, starts = np.unique(kid[order], return_index=True)
                for n, start in enumerate(starts):
                    sel = order[start: starts[n + 1] if n + 1 < len(starts) else len(order)]
                    key = tuple(int(v) for v in bk[sel[0]])
                    c = self.color.get(key)
                    if c is None:
                        continue
                    cc = c[flat[sel]]
                    good = (self.weight[key][flat[sel]] > 0.3) & (cc.sum(axis=1) > 0)
                    out[todo[sel[good]]] = cc[good]
                    found[todo[sel[good]]] = True
        return out, found

    def floor_estimate(self) -> float | None:
        """Floor height from the scan itself: the lowest strongly populated level of confident surface."""
        with self.lock:
            items = [(k, self.tsdf[k], self.weight[k]) for k in list(self.tsdf)[:4000]]
        ys = []
        for k, t, wgt in items:
            surf = (np.abs(t) < 0.3) & (wgt > 0.5)
            if surf.any():
                ys.append(k[1] * BLOCK * self.voxel + self._offsets[surf][:, 1])
        if not ys:
            return None
        y = np.concatenate(ys)
        if len(y) < 500:
            return None
        hist, edges = np.histogram(y, bins=np.arange(y.min(), y.min() + 1.0, 0.02))
        if not len(hist) or hist.max() < 100:
            return None
        i = int(np.argmax(hist >= 0.3 * hist.max()))  # lowest level that is a real surface, not stray points
        i = i + int(np.argmax(hist[i:i + 5]))
        return float((edges[i] + edges[i + 1]) / 2)

    def set_planes(self, floor_y: float | None, walls: list, extent=None) -> bool:
        """Known flat surfaces: the floor height and wall segments ((x1, z1), (x2, z2)). Returns True if they
        changed enough that existing meshes should be redone."""
        new = {"floor": None if floor_y is None else round(floor_y, 2),
               "walls": [((round(p[0], 2), round(p[1], 2)), (round(q[0], 2), round(q[1], 2))) for p, q in walls], "extent": extent}
        changed = new["floor"] != self.planes["floor"] or new["walls"] != self.planes["walls"]
        self.planes = new
        return changed

    def _flatten(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lay surface within SNAP of the floor or a known wall exactly onto it. Returns the moved vertices
        and which of them lie on a known plane."""
        snapped = np.zeros(len(world), bool)
        if not len(world):
            return world, snapped
        fy, ext = self.planes["floor"], self.planes["extent"]
        if fy is not None:
            m = np.abs(world[:, 1] - fy) < SNAP
            if ext is not None:
                m &= (world[:, 0] >= ext[0]) & (world[:, 0] <= ext[2]) & (world[:, 2] >= ext[1]) & (world[:, 2] <= ext[3])
            world[m, 1] = fy
            snapped |= m
        lo, hi = world.min(axis=0) - 0.1, world.max(axis=0) + 0.1
        for (ax, az), (bx, bz) in self.planes["walls"]:
            if max(ax, bx) < lo[0] or min(ax, bx) > hi[0] or max(az, bz) < lo[2] or min(az, bz) > hi[2]:
                continue
            L = math.hypot(bx - ax, bz - az)
            if L < 0.2:
                continue
            ux, uz = (bx - ax) / L, (bz - az) / L
            rx, rz = world[:, 0] - ax, world[:, 2] - az
            t, off = rx * ux + rz * uz, -rx * uz + rz * ux
            m = (np.abs(off) < SNAP) & (t > -0.05) & (t < L + 0.05) & ~snapped
            if fy is not None:
                m &= world[:, 1] > fy + 0.03
            world[m, 0] += off[m] * uz
            world[m, 2] -= off[m] * ux
            snapped |= m
        return world, snapped

    @staticmethod
    def _vertex_colors(col: np.ndarray, wt: np.ndarray, verts: np.ndarray) -> np.ndarray:
        """Trilinear colour at each vertex from the 8 surrounding voxels (only those that were actually
        painted), so colour changes smoothly across a surface instead of in 4 cm steps."""
        N = col.shape[0]
        v0 = np.clip(np.floor(verts).astype(int), 0, N - 2)
        f = np.clip(verts - v0, 0, 1)
        acc = np.zeros((len(verts), 3), np.float32)
        tot = np.zeros(len(verts), np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    i, j, k = v0[:, 0] + dx, v0[:, 1] + dy, v0[:, 2] + dz
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    c = col[i, j, k]
                    w = w * ((wt[i, j, k] > 0) & (c.sum(axis=1) > 0))
                    acc += c * w[:, None]
                    tot += w
        r = np.clip(np.round(verts).astype(int), 0, N - 1)
        near = col[r[:, 0], r[:, 1], r[:, 2]]
        return np.where(tot[:, None] > 1e-6, acc / np.maximum(tot, 1e-6)[:, None], near)

    def _pack(self, k, world: np.ndarray, faces: np.ndarray, colors: np.ndarray, normals: np.ndarray | None = None) -> bytes:
        if len(world) > 65535:  # can't happen for a 17^3 grid, but keep the format honest
            world, faces, colors, normals = world[:0], faces[:0], colors[:0], None
        # 0x04 | i16 block x,y,z | u16 n_verts | u16 n_faces | f16[n*3] vertex offsets from the block origin (m)
        #      | u8[n*3] rgb | u16[f*3] triangle indices | optional i8[n*3] normals (from the distance field:
        #      smooth shading that continues across blocks; viewers without it compute their own)
        head = MESH_REC.pack(self.mesh_type, *k, len(world), len(faces))
        origin = np.array(k) * BLOCK * self.voxel
        rel = (world - origin).astype("<f2")
        out = head + rel.tobytes() + np.clip(colors, 0, 255).astype(np.uint8).tobytes() + faces.astype("<u2").tobytes()
        if normals is not None and len(normals) == len(world) and len(world):
            n = -normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)  # outward (towards free space)
            out += np.clip(np.round(n * 127), -127, 127).astype(np.int8).tobytes()
        return out

    def take_dirty(self, limit: int = 60) -> list:
        with self.lock:
            out = list(self.dirty)[:limit]
            self.dirty.difference_update(out)
        return out

    def all_blocks(self) -> list:
        with self.lock:
            return list(self.tsdf.keys())

    # ---------------- map semantics: walls and furniture ----------------
    def surface_columns(self, floor_y: float) -> dict[tuple[int, int], tuple[float, float]]:
        """For each 10 cm floor cell: (lowest, highest) height above floor of confident surface voxels."""
        cols: dict[tuple[int, int], list[float]] = {}
        with self.lock:
            items = [(k, self.tsdf[k], self.weight[k]) for k in self.tsdf]
        for k, t, wgt in items:
            surf = (np.abs(t) < 0.35) & (wgt > 0.3)
            if not surf.any():
                continue
            pts = np.array(k) * BLOCK * self.voxel + self._offsets[surf]
            cx = np.floor(pts[:, 0] / COLUMN).astype(int)
            cz = np.floor(pts[:, 2] / COLUMN).astype(int)
            hy = pts[:, 1] - floor_y
            for a, b, y in zip(cx, cz, hy):
                c = cols.get((a, b))
                if c is None:
                    cols[(a, b)] = [y, y]
                else:
                    c[0] = min(c[0], y)
                    c[1] = max(c[1], y)
        return {k: (v[0], v[1]) for k, v in cols.items()}

    def semantics(self, floor_y: float, labels=None) -> tuple[list, list]:
        """Walls (segments) and furniture (boxes) inferred from surface height profiles.

        Also fills self.candidates: wall gaps and pillar-like shapes for the laptop's scene AI to look at.
        `labels(key)` returns what that AI decided ("window", "wall", "door", "column") or None.
        """
        labels = labels or (lambda key: None)
        cols = self.surface_columns(floor_y)
        # tall surfaces: walls, and blinds / curtains that stop short of the floor
        wall_cells = [(a, b) for (a, b), (lo, hi) in cols.items() if (hi - lo > 1.1 and hi > 1.4) or (hi - lo > 0.8 and hi > 1.6)]
        fit = fit_walls(wall_cells, floor_y)
        walls, candidates = list(fit["walls"]), []
        for g in fit["gaps"]:  # windows: a gap in a wall line with a sill below, or the scene AI says so
            mid = ((g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2)
            key = ("gap", round(mid[0] / 0.3), round(mid[1] / 0.3))
            said = labels(key)
            doorway = said == "door" or sum(1 for x, z in self.doors_seen if math.hypot(x - mid[0], z - mid[1]) < 0.7) >= 2
            if doorway:  # a door (scene AI, or the phone's detector twice): open for routes, drawn as a door
                walls.append((g[0], g[1], floor_y, floor_y + 2.1, "door"))
            elif said in ("window", "wall") or (said is None and _has_sill(g, cols)):
                walls.append((g[0], g[1], floor_y, floor_y + 2.4, "vertical"))
            candidates.append({"key": key, "what": "wall gap", "at": mid,
                               "points": [(g[0][0] + (g[1][0] - g[0][0]) * f, floor_y + h, g[0][1] + (g[1][1] - g[0][1]) * f)
                                          for f in (0.3, 0.5, 0.7) for h in (1.3, 1.7)]})
        for box, strong in find_pillars(fit["leftover"], cols):  # tall, narrow, free-standing: pillars
            x0, z0, x1, z1 = box
            mid = ((x0 + x1) / 2, (z0 + z1) / 2)
            if any(_seg_dist(mid, w[0], w[1]) < 0.45 for w in fit["walls"]):
                continue  # a leftover scrap at a wall's end or face, not a free-standing pillar
            key = ("pillar", round(mid[0] / 0.3), round(mid[1] / 0.3))
            said = labels(key)
            if (strong and said != "door") or said == "column":
                corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
                walls.extend((corners[i], corners[(i + 1) % 4], floor_y, floor_y + 2.4, "pillar") for i in range(4))
            candidates.append({"key": key, "what": "tall narrow shape", "at": mid,
                               "points": [(mid[0], floor_y + h, mid[1]) for h in (0.9, 1.4, 1.9)]})
        used = set(wall_cells)
        stairs = []
        for st in find_stairs(cols, used):  # a steady stepped slope (up or down from this floor)
            key = ("stairs", round(st["mid"][0] / 0.3), round(st["mid"][1] / 0.3))
            said = labels(key)
            (x0, z0), (x1, z1) = st["bbox"]
            seen = sum(1 for x, z in self.stairs_seen if x0 - 0.5 <= x <= x1 + 0.5 and z0 - 0.5 <= z <= z1 + 0.5)
            if st["strong"] or said == "stairs" or seen >= 2:
                stairs.append({k: st[k] for k in ("corners", "up", "rise", "down")})
                used |= st["cells"]  # stairs are walkable: never furniture / obstacles
            pts = sorted(st["cells"])[:: max(1, len(st["cells"]) // 6)][:6]
            candidates.append({"key": key, "what": "stepped slope", "at": st["mid"],
                               "points": [((a + 0.5) * COLUMN, floor_y + cols[(a, b)][1], (b + 0.5) * COLUMN) for a, b in pts]})
        self.stairs = stairs
        self.drops = find_drops(cols, used)
        self.candidates = candidates
        furn = classify_furniture({k: v for k, v in cols.items() if k not in used})
        seen = list(self.furniture_seen)
        for o in furn:  # confirmed = YOLO saw furniture inside (or right next to) this fused box
            o["confirmed"] = any(o["x0"] - 0.3 <= x <= o["x1"] + 0.3 and o["z0"] - 0.3 <= z <= o["z1"] + 0.3 for x, z in seen)
            o["large"] = is_large(o)
        return walls, furn


def _taubin(verts: np.ndarray, faces: np.ndarray, iters: int = 4, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin smoothing (shrink-free): removes the voxel staircase without shrinking the surface. Vertices on
    the block's border are fixed so neighbouring blocks still meet exactly."""
    if len(faces) == 0:
        return verts
    from scipy.sparse import coo_matrix
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    border = ((verts < 0.6) | (verts > BLOCK - 0.6)).any(axis=1)
    v = verts.astype(np.float64).copy()
    for _ in range(iters):
        for f in (lam, mu):
            delta = A @ v / deg[:, None] - v
            delta[border] = 0
            v += f * delta
    return v.astype(verts.dtype)


def fit_wall_lines(cells: list[tuple[int, int]], floor_y: float, min_len: float = 1.0, min_fill: float = 0.6) -> list:
    """Straight wall segments (p, q, y0, y1, 'vertical') from wall-like cells. See fit_walls."""
    return fit_walls(cells, floor_y, min_len, min_fill)["walls"]


def fit_walls(cells: list[tuple[int, int]], floor_y: float, min_len: float = 1.0, min_fill: float = 0.6) -> dict:
    """Greedy RANSAC: straight runs of wall-like cells -> wall segments (p, q, y0, y1, 'vertical').
    Each line is refit to all its cells (least squares), and a run only counts as a wall if it is at
    least min_len long and solid (min_fill of its length covered by wall cells): no stray lines.
    Also returns the gaps between consecutive runs of one line (doorways, windows: 0.45-3 m) and the
    wall-like cells that ended up in no wall (pillars and other tall, narrow things)."""
    out = {"walls": [], "gaps": [], "leftover": list(cells)}
    if len(cells) < 6:
        return out
    pts = (np.array(cells, float) + 0.5) * COLUMN
    remaining = np.ones(len(pts), bool)  # not yet claimed by a RANSAC line
    in_wall = np.zeros(len(pts), bool)  # part of an accepted wall run
    rng = np.random.default_rng(0)
    for _ in range(40):
        idx = np.nonzero(remaining)[0]
        if len(idx) < 6:
            break
        best = None
        for _ in range(120):
            i, j = rng.choice(idx, 2, replace=False)
            d = pts[j] - pts[i]
            L = np.hypot(*d)
            if L < 0.3:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            dist = np.abs((pts[idx] - pts[i]) @ nrm)
            inl = idx[dist < 0.08]
            if best is None or len(inl) > len(best[0]):
                best = (inl, pts[i], d / L)
        if best is None or len(best[0]) < 6:
            break
        inl, o, u = best
        o = pts[inl].mean(axis=0)  # refit: principal direction of all inliers, not two random cells
        u = np.linalg.svd(pts[inl] - o, full_matrices=False)[2][0]
        proj = (pts[inl] - o) @ u
        order = np.argsort(proj)
        t = proj[order]
        at = lambda v: (float(o[0] + u[0] * v), float(o[1] + u[1] * v))
        # split at gaps (doorways, windows, separate walls)
        runs, start, members = [], t[0], []
        for n, (a, b) in enumerate(zip(t, np.append(t[1:], np.inf))):
            members.append(inl[order[n]])
            if b - a > 0.45:
                if a - start >= min_len and len(members) * COLUMN >= min_fill * (a - start + COLUMN):
                    runs.append((start, a))
                    in_wall[members] = True
                start, members = b, []
        for s0, s1 in runs:
            out["walls"].append((at(s0), at(s1), floor_y, floor_y + 2.4, "vertical"))
        for (_, e0), (s1, _) in zip(runs, runs[1:]):
            if 0.45 < s1 - e0 <= 3.0:
                out["gaps"].append((at(e0), at(s1)))
        remaining[inl] = False
    out["leftover"] = [cells[i] for i in range(len(cells)) if not in_wall[i]]
    return out


def _has_sill(gap, cols: dict, near: float = 0.15) -> bool:
    """A window, not a doorway: along most of the gap there is something low (a sill / low wall,
    0.4-1.4 m high) right on the wall line. Doorways are open down to the floor."""
    (ax, az), (bx, bz) = gap
    steps = max(2, int(math.hypot(bx - ax, bz - az) / COLUMN))
    r = int(math.ceil(near / COLUMN))
    hits = 0
    for i in range(1, steps):
        cx = math.floor((ax + (bx - ax) * i / steps) / COLUMN)
        cz = math.floor((az + (bz - az) * i / steps) / COLUMN)
        if any(0.4 <= cols[(cx + dx, cz + dz)][1] <= 1.4 for dx in range(-r, r + 1) for dz in range(-r, r + 1) if (cx + dx, cz + dz) in cols):
            hits += 1
    return hits >= 0.5 * (steps - 1)


def _seg_dist(p, a, b) -> float:
    """Distance from point p to segment a-b (2D)."""
    ax, az, bx, bz = a[0], a[1], b[0], b[1]
    L2 = (bx - ax) ** 2 + (bz - az) ** 2
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - az) * (bz - az)) / L2))
    return math.hypot(p[0] - ax - t * (bx - ax), p[1] - az - t * (bz - az))


def find_drops(cols: dict, exclude: set, spacing: float = 0.8) -> list[tuple[float, float]]:
    """Floor edges where the surface falls away by more than DROP_DEPTH (not stairs): points along each edge."""
    floorish = {k for k, (lo, hi) in cols.items() if abs(lo) < 0.08}
    edge = {k for k, (lo, hi) in cols.items() if hi < -DROP_DEPTH and k not in exclude
            and any((k[0] + dx, k[1] + dz) in floorish for dx in (-1, 0, 1) for dz in (-1, 0, 1))}
    out, todo = [], set(edge)
    while todo:
        stack, comp = [todo.pop()], []
        while stack:
            a, b = stack.pop()
            comp.append((a, b))
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1), (a + 1, b + 1), (a - 1, b - 1), (a + 1, b - 1), (a - 1, b + 1)):
                if n in todo:
                    todo.discard(n)
                    stack.append(n)
        if len(comp) < 3:
            continue
        pts = sorted(((a + 0.5) * COLUMN, (b + 0.5) * COLUMN) for a, b in comp)
        last = None
        for p in pts:  # thin to one point every `spacing` metres along the edge
            if last is None or math.dist(p, last) >= spacing:
                out.append((round(p[0], 2), round(p[1], 2)))
                last = p
    return out


def find_stairs(cols: dict, exclude: set) -> list[dict]:
    """Flights of stairs (or ramps) from the surface heights: a connected patch whose top surfaces climb
    or drop steadily (19-45 degrees, at least 0.45 m of rise) and fit one sloped plane well. Works for
    stairs going up from this floor and, seen from the top, going down (heights below the floor)."""
    cand = {k: hi for k, (lo, hi) in cols.items() if k not in exclude and abs(hi) > 0.08 and -3.5 < hi < 2.6}
    todo, out = set(cand), []
    while todo:
        seed = todo.pop()
        comp, stack = [seed], [seed]
        while stack:
            a, b = stack.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1), (a + 1, b + 1), (a - 1, b - 1), (a + 1, b - 1), (a - 1, b + 1)):
                if n in todo and abs(cand[n] - cand[(a, b)]) < 0.25:  # one riser at a time (stairs: ~0.15-0.2 m)
                    todo.discard(n)
                    stack.append(n)
                    comp.append(n)
        if len(comp) < 40:
            continue
        xs = (np.array([c[0] for c in comp]) + 0.5) * COLUMN
        zs = (np.array([c[1] for c in comp]) + 0.5) * COLUMN
        hs = np.array([cand[c] for c in comp])
        rise = float(hs.max() - hs.min())
        if rise < 0.45:
            continue
        A = np.c_[xs, zs, np.ones(len(xs))]
        coef = np.linalg.lstsq(A, hs, rcond=None)[0]
        g = math.hypot(coef[0], coef[1])
        if not 0.35 <= g <= 1.0 or float(np.sqrt(np.mean((hs - A @ coef) ** 2))) > 0.12:
            continue
        ux, uz = coef[0] / g, coef[1] / g  # uphill direction
        t, s = xs * ux + zs * uz, -xs * uz + zs * ux
        corners = [(float(tt * ux - ss * uz), float(tt * uz + ss * ux)) for tt, ss in
                   ((t.min(), s.min()), (t.max(), s.min()), (t.max(), s.max()), (t.min(), s.max()))]
        out.append({"cells": set(comp), "corners": corners, "up": (float(ux), float(uz)), "rise": round(rise, 2),
                    "down": bool(hs.mean() < 0), "strong": rise >= 0.8 and len(comp) >= 80,
                    "mid": (float(xs.mean()), float(zs.mean())), "bbox": ((float(xs.min()), float(zs.min())), (float(xs.max()), float(zs.max())))})
    return out


def find_pillars(cells: list[tuple[int, int]], cols: dict) -> list[tuple[tuple[float, float, float, float], bool]]:
    """Tall, narrow, free-standing groups of wall-like cells that no wall took: pillars.
    Returns (x0, z0, x1, z1) boxes, each with `strong` = tall enough to trust without the scene AI."""
    todo, out = set(cells), []
    while todo:
        stack, comp = [todo.pop()], []
        while stack:
            a, b = stack.pop()
            comp.append((a, b))
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1), (a + 1, b + 1), (a - 1, b - 1), (a + 1, b - 1), (a - 1, b + 1)):
                if n in todo:
                    todo.discard(n)
                    stack.append(n)
        xs, zs = [c[0] for c in comp], [c[1] for c in comp]
        w, d = (max(xs) - min(xs) + 1) * COLUMN, (max(zs) - min(zs) + 1) * COLUMN
        if len(comp) < 3 or w > 1.0 or d > 1.0 or max(w, d) < 0.2:
            continue
        lo, hi = min(cols[c][0] for c in comp), max(cols[c][1] for c in comp)
        out.append(((min(xs) * COLUMN, min(zs) * COLUMN, (max(xs) + 1) * COLUMN, (max(zs) + 1) * COLUMN), hi > 1.9 and hi - lo > 1.5))
    return out


FURNITURE = [("low", 0.12, 0.32), ("seat", 0.32, 0.62), ("table", 0.62, 1.0), ("obstacle", 1.0, 1.9)]  # low: trip clutter
DROP_DEPTH = 0.3  # m below the floor next to floor-level surface = a drop (edge of a stairwell, pit, missing floor)
LARGE_AREA, LARGE_SIDE = 1.0, 1.6  # m^2 / m: only obstructions this big show on the map and block routes


def is_large(o: dict) -> bool:
    """A very large obstruction (big table, sofa, cabinet run): rooms stay hollow apart from these."""
    w, d = o["x1"] - o["x0"], o["z1"] - o["z0"]
    return bool(w * d >= LARGE_AREA or (max(w, d) >= LARGE_SIDE and w * d >= 0.4))


def classify_furniture(cols: dict[tuple[int, int], tuple[float, float]]) -> list[dict]:
    """Group neighbouring cells by top height into seats / tables / other obstacles."""
    out = []
    for kind, lo, hi in FURNITURE:
        cells = {k for k, (_, top) in cols.items() if lo <= top < hi}
        seen: set = set()
        for c in cells:
            if c in seen:
                continue
            stack, comp = [c], []
            seen.add(c)
            while stack:
                a, b = stack.pop()
                comp.append((a, b))
                for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1), (a + 1, b + 1), (a - 1, b - 1), (a + 1, b - 1), (a - 1, b + 1)):
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
            if len(comp) < 4:  # under ~0.04 m^2: noise
                continue
            xs = [a for a, _ in comp]
            zs = [b for _, b in comp]
            area = len(comp) * COLUMN * COLUMN
            if area > 6:  # a "table" the size of a room is really the floor or a misfit
                continue
            top = float(np.median([cols[x][1] for x in comp]))
            out.append({"kind": kind, "x0": min(xs) * COLUMN, "z0": min(zs) * COLUMN, "x1": (max(xs) + 1) * COLUMN,
                        "z1": (max(zs) + 1) * COLUMN, "top": round(top, 2), "area": round(area, 2)})
    return out
