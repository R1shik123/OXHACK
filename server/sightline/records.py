"""Opt-in, encrypted recordings on the laptop.

Two kinds, both started explicitly from the commander page:
  - "session": compact time series for ML training and replay (poses at 2 Hz, events, alerts,
    labels, coverage). Roughly 1-2 MB per hour. No camera frames, no point clouds.
  - "capture": a short room capture (depth frames, poses, a few camera frames) used once to tune
    wall / furniture / fusion thresholds, then deleted.
Each line is a Fernet token (AES-128-CBC + HMAC-SHA256) of one JSON record, so files can be
appended safely and are unreadable without data/record.key (mode 0600). Files older than
RETENTION_DAYS are deleted at startup and the folder is capped at MAX_BYTES (oldest go first).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .security import DATA, _write_private

RETENTION_DAYS = 7
MAX_BYTES = 200 * 1024 * 1024


class Recorder:
    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or DATA / "recordings"
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        key_path = self.folder.parent / "record.key"
        if not key_path.exists():
            _write_private(key_path, Fernet.generate_key())
        self.fernet = Fernet(key_path.read_bytes())
        self.lock = threading.Lock()
        self.active: dict[str, dict] = {}  # kind -> {"path", "fh", "started", "until", "meta", "bytes"}
        self.prune()

    # ---------------- writing ----------------
    def start(self, kind: str, meta: dict, seconds: float | None = None) -> str:
        self.stop(kind)
        name = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.slrec"
        path = self.folder / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        fh = os.fdopen(fd, "ab")
        with self.lock:
            self.active[kind] = {"path": path, "fh": fh, "started": time.time(), "until": time.time() + seconds if seconds else None,
                                 "meta": meta, "bytes": 0}
        self.write(kind, {"t": "meta", "kind": kind, **meta})
        return name

    def write(self, kind: str, record: dict) -> None:
        rec = self.active.get(kind)
        if rec is None:
            return
        if rec["until"] and time.time() > rec["until"]:
            self.stop(kind)
            return
        line = self.fernet.encrypt(json.dumps({"ts": round(time.time(), 3), **record}, separators=(",", ":")).encode()) + b"\n"
        with self.lock:
            if kind in self.active:
                rec["fh"].write(line)
                rec["bytes"] += len(line)

    def recording(self, kind: str) -> bool:
        rec = self.active.get(kind)
        if rec and rec["until"] and time.time() > rec["until"]:
            self.stop(kind)
            return False
        return rec is not None

    def stop(self, kind: str) -> None:
        with self.lock:
            rec = self.active.pop(kind, None)
        if rec:
            rec["fh"].close()

    def status(self) -> dict:
        now = time.time()
        return {k: {"name": v["path"].name, "seconds": round(now - v["started"]), "left": round(v["until"] - now) if v["until"] else None,
                    "bytes": v["bytes"]} for k, v in list(self.active.items())}

    # ---------------- housekeeping ----------------
    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.folder.glob("*.slrec")):
            st = p.stat()
            out.append({"name": p.name, "kind": p.name.split("-")[0], "bytes": st.st_size, "modified": st.st_mtime,
                        "active": any(v["path"] == p for v in self.active.values())})
        return out

    def delete(self, name: str) -> bool:
        p = (self.folder / name).resolve()
        if p.parent != self.folder.resolve() or p.suffix != ".slrec" or not p.exists():
            return False
        for kind, v in list(self.active.items()):
            if v["path"].resolve() == p:
                self.stop(kind)
        p.unlink()
        return True

    def prune(self) -> None:
        files = sorted(self.folder.glob("*.slrec"), key=lambda p: p.stat().st_mtime)
        cutoff = time.time() - RETENTION_DAYS * 86400
        for p in files:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        files = [p for p in files if p.exists()]
        total = sum(p.stat().st_size for p in files)
        while files and total > MAX_BYTES:
            p = files.pop(0)
            total -= p.stat().st_size
            p.unlink()

    # ---------------- reading (training, tuning) ----------------
    def read(self, name: str):
        p = (self.folder / name).resolve()
        if p.parent != self.folder.resolve():
            raise ValueError("outside the recordings folder")
        with open(p, "rb") as fh:
            for line in fh:
                try:
                    yield json.loads(self.fernet.decrypt(line.strip()))
                except (InvalidToken, ValueError):
                    continue
