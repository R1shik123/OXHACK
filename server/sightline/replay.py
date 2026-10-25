"""Replay a room capture through 3D fusion with different settings, and score each run.

    python -m sightline.replay                       # newest capture, standard variants
    python -m sightline.replay capture-2026....slrec --measure ../roomdata/measurements.json

The same recorded depth frames go through every variant, so differences come only from the settings
(e.g. furniture outlines on vs off). Scores need no ground truth; with a measurements file (tape
measure) the room size and furniture heights are also compared to reality.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from . import fusion as F
from .protocol import parse_depth_frame
from .records import Recorder

VARIANTS = {
    "standard": {},
    "no-furniture-outlines": {"FURNITURE_BOOST": 1.0},
    "strict-surfaces": {"MESH_MIN_WEIGHT": 0.8},
    "loose-surfaces": {"MESH_MIN_WEIGHT": 0.25},
    "tight-truncation": {"TRUNC": 0.10},
}


def load(rec: Recorder, name: str) -> tuple[dict, list[dict]]:
    meta, frames = {}, []
    for r in rec.read(name):
        if r.get("t") == "meta":
            meta = r
        elif r.get("t") == "depth":
            try:
                frames.append(parse_depth_frame(base64.b64decode(r["b64"])))
            except ValueError:
                continue
    return meta, frames


def run(frames: list[dict], floor_y: float, params: dict) -> dict:
    saved = {k: getattr(F, k) for k in params}
    for k, v in params.items():
        setattr(F, k, v)
    try:
        f = F.Fusion()
        first_table = None
        t0 = time.perf_counter()
        for i, fr in enumerate(frames):
            f.integrate(fr)
            if first_table is None and i % 10 == 9 and any(o["kind"] == "table" for o in f.semantics(floor_y)[1]):
                first_table = i + 1
        secs = time.perf_counter() - t0
        walls, objs = f.semantics(floor_y)
        meshes = [f.mesh_block(k) for k in list(f.tsdf)]
        tris = sum(int.from_bytes(m[9:11], "little") for m in meshes if m)
        weights = np.concatenate([w[np.abs(t) < 0.35] for t, w in zip(f.tsdf.values(), f.weight.values())]) if f.tsdf else np.zeros(0)
        speckle = float((weights < F.MESH_MIN_WEIGHT).mean()) if len(weights) else 0.0
        xs = [p for w in walls for p in (w[0][0], w[1][0])]
        zs = [p for w in walls for p in (w[0][1], w[1][1])]
        tables = [o["top"] for o in objs if o["kind"] == "table"]
        seats = [o["top"] for o in objs if o["kind"] == "seat"]
        return {
            "frames": len(frames), "ms_per_frame": round(1000 * secs / max(1, len(frames)), 1),
            "walls": len(walls), "wall_m": round(sum(math.dist(w[0], w[1]) for w in walls), 1),
            "room_x_m": round(max(xs) - min(xs), 2) if xs else None, "room_z_m": round(max(zs) - min(zs), 2) if zs else None,
            "tables": len(tables), "seats": len(seats), "confirmed": sum(1 for o in objs if o.get("confirmed")),
            "clutter": sum(1 for o in objs if o["kind"] == "obstacle" and not o.get("confirmed")),
            "table_top_m": round(statistics.median(tables), 2) if tables else None,
            "seat_top_m": round(statistics.median(seats), 2) if seats else None,
            "triangles": tris, "speckle_pct": round(100 * speckle, 1), "first_table_frame": first_table,
        }
    finally:
        for k, v in saved.items():
            setattr(F, k, v)


def compare(score: dict, truth: dict) -> dict:
    """Error vs tape measurements (only for the numbers you measured)."""
    out = {}
    room = sorted(v for v in (score["room_x_m"], score["room_z_m"]) if v)
    want = sorted(v for v in (truth.get("room_length_m"), truth.get("room_width_m")) if v)
    if room and want and len(room) == len(want):
        out["room_err_cm"] = round(100 * max(abs(a - b) for a, b in zip(room, want)))
    for k, t in (("table_top_m", "table_height_m"), ("seat_top_m", "seat_height_m")):
        if score.get(k) is not None and truth.get(t):
            out[k.replace("_m", "_err_cm")] = round(100 * abs(score[k] - truth[t]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", help="capture file name (default: newest capture)")
    ap.add_argument("--measure", help="JSON with room_length_m, room_width_m, table_height_m, seat_height_m")
    ap.add_argument("--only", nargs="*", help="variant names to run")
    a = ap.parse_args()
    rec = Recorder()
    caps = [r["name"] for r in rec.list() if r["kind"] == "capture"]
    if not caps:
        raise SystemExit("No captures yet: commander page -> Room capture (90 s).")
    name = a.capture or caps[-1]
    meta, frames = load(rec, name)
    if not frames:
        raise SystemExit(f"{name}: no depth frames in this capture.")
    floor = meta.get("floor_y", -1.4)
    truth = json.loads(Path(a.measure).read_text()) if a.measure else {}
    print(f"{name}: {len(frames)} depth frames, floor at {floor:.2f} m (marker = 0)\n")
    rows = []
    for v, params in VARIANTS.items():
        if a.only and v not in a.only:
            continue
        s = run(frames, floor, params)
        s.update(compare(s, truth))
        rows.append((v, s))
    keys = [k for k in rows[0][1]]
    print("variant".ljust(24) + "".join(k[:12].rjust(13) for k in keys))
    for v, s in rows:
        print(v.ljust(24) + "".join(str(s.get(k, "")).rjust(13) for k in keys))


if __name__ == "__main__":
    main()
