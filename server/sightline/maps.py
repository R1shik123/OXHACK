"""Saved maps: the commander can save the shared map and load or delete it later.

Stored only on the laptop, in a small SQLite database next to the recordings. Each map is compressed
and encrypted with the same laptop-only key as recordings (Fernet), so the file is unreadable on its
own. A map holds geometry only: walls, AR planes, coloured points, 3D surfaces, coverage and pins.
No camera images, no names of responders, no trails. Nothing is ever stored on a phone.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .security import DATA

MAX_NAME = 40


class MapStore:
    def __init__(self, fernet: Fernet, path: Path | None = None) -> None:
        self.fernet = fernet
        self.path = path or DATA / "maps.db"
        new = not self.path.exists()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        if new:
            self.path.chmod(0o600)
        self.db.execute("CREATE TABLE IF NOT EXISTS maps (id INTEGER PRIMARY KEY, name TEXT, created REAL, bytes INTEGER, blob BLOB)")
        if "labels" not in {r[1] for r in self.db.execute("PRAGMA table_info(maps)")}:
            self.db.execute("ALTER TABLE maps ADD COLUMN labels BLOB")  # ground truth for evaluation (encrypted)
        self.db.commit()
        self.lock = threading.Lock()

    def save(self, name: str, payload: dict) -> dict:
        name = (name.strip() or time.strftime("Map %d %b %H:%M"))[:MAX_NAME]
        blob = self.fernet.encrypt(zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), 6))
        with self.lock:
            cur = self.db.execute("INSERT INTO maps (name, created, bytes, blob) VALUES (?, ?, ?, ?)", (name, time.time(), len(blob), blob))
            self.db.commit()
        return {"id": cur.lastrowid, "name": name}

    def list(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id, name, created, bytes, labels FROM maps ORDER BY created DESC").fetchall()
        out = []
        for i, n, c, b, lab in rows:
            labels = self._open(lab) if lab else None
            out.append({"id": i, "name": n, "created": c, "bytes": b, "area": (labels or {}).get("area"),
                        "labelled": {"rooms": len((labels or {}).get("rooms", [])), "risk": len((labels or {}).get("risk", []))}})
        return out

    def _open(self, blob: bytes) -> dict | None:
        try:
            return json.loads(zlib.decompress(self.fernet.decrypt(blob)))
        except (InvalidToken, zlib.error, ValueError):
            return None

    def labels(self, map_id: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT labels FROM maps WHERE id = ?", (map_id,)).fetchone()
        return self._open(row[0]) if row and row[0] else None

    def set_labels(self, map_id: int, labels: dict) -> bool:
        blob = self.fernet.encrypt(zlib.compress(json.dumps(labels, separators=(",", ":")).encode(), 6))
        with self.lock:
            cur = self.db.execute("UPDATE maps SET labels = ? WHERE id = ?", (blob, map_id))
            self.db.commit()
        return cur.rowcount > 0

    def load(self, map_id: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT blob FROM maps WHERE id = ?", (map_id,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(zlib.decompress(self.fernet.decrypt(row[0])))
        except (InvalidToken, zlib.error, ValueError):
            return None

    def delete(self, map_id: int) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM maps WHERE id = ?", (map_id,))
            self.db.commit()
            self.db.execute("VACUUM")  # really gone from the file, not just unlinked
        return cur.rowcount > 0


def capture(world) -> dict:
    """The current shared map as plain data (geometry only)."""
    with world.fusion.lock:
        meshes = [base64.b64encode(m).decode() for m in world.fusion.meshes.values() if m]
    with world.fine.lock:
        meshes += [base64.b64encode(m).decode() for m in world.fine.meshes.values() if m]
    return {
        "v": 1,
        "planes": [{**p, "owner": "saved"} for p in world.planes.values()],
        "depth_walls": [[list(p), list(q), y0, y1, k] for p, q, y0, y1, k in world.depth_walls],
        "objects": world.objects,
        "pins": [{k: v for k, v in pin.items()} for pin in world.pins.values()],
        "coverage": [[ix, iz, n] for (ix, iz), n in world.coverage.seen.items()],
        "voxels": [[*k, *c] for k, c in world.voxels.items()],
        "stairs": [{"corners": [list(c) for c in s["corners"]], "up": list(s["up"]), "rise": s["rise"], "down": s["down"]} for s in world.stairs],
        "drops": [list(d) for d in world.drops],
        "fall_spots": [list(f) for f in world.fall_spots],
        "visited": [list(v) for v in world.visited],
        "meshes": meshes,
    }


def restore(world, data: dict) -> None:
    """Replace the shared map with a saved one. Live phones stay connected and keep mapping on top."""
    keep = {pid: (p.name, p.connected) for pid, p in world.phones.items()}
    world.reset()
    for pid, (name, connected) in keep.items():
        world.phone(pid, name).connected = connected
    for p in data.get("planes", []):
        world.planes[p["id"]] = p
    world.depth_walls = [((p[0], p[1]), (q[0], q[1]), y0, y1, k) for p, q, y0, y1, k in data.get("depth_walls", [])]
    world.objects = data.get("objects", [])
    world.stairs = [{**s, "corners": [tuple(c) for c in s["corners"]], "up": tuple(s["up"])} for s in data.get("stairs", [])]
    world.fusion.stairs = list(world.stairs)
    world.drops = [tuple(d) for d in data.get("drops", [])]
    world.fusion.drops = list(world.drops)
    world.fall_spots = [tuple(f) for f in data.get("fall_spots", [])]
    world.visited = {tuple(v) for v in data.get("visited", [])}
    for pin in data.get("pins", []):
        world.pins[pin["id"]] = pin
    nums = [int(str(i)[3:]) for i in world.pins if str(i).startswith("pin") and str(i)[3:].isdigit()]
    if nums:
        import itertools
        world._pin_ids = itertools.count(max(nums) + 1)
    world.coverage.seen = {(ix, iz): n for ix, iz, n in data.get("coverage", [])}
    world.voxels = {(x, y, z): (r, g, b) for x, y, z, r, g, b in data.get("voxels", [])}
    for b64 in data.get("meshes", []):
        m = base64.b64decode(b64)
        if len(m) >= 7:
            k = tuple(int.from_bytes(m[i:i + 2], "little", signed=True) for i in (1, 3, 5))
            (world.fine if m[0] == 0x07 else world.fusion).meshes[k] = m
    world._merged = None
    world._walls_cache = None
    world.walls_dirty = True
    world.coverage.set_outline(world.merged_walls())
