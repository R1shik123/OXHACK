"""Second opinion on objects the fused 3D map found: an image classifier on the laptop.

The phones send camera snapshots with the pose they were taken from (0x05). Every few seconds, for each
fused 3D object nobody has confirmed yet, we cut that object's region out of a real snapshot (projecting
its 3D box into the picture) and run Apple's Core ML MobileNetV2 (ImageNet, 1,000 everyday classes) on it.

Rules that keep it honest:
  - It only CONFIRMS shapes that fusion already measured; it never adds, moves or reshapes geometry and
    never draws anything (no generated imagery).
  - One guess proves nothing (a classifier always answers something), so an object is confirmed only when
    two different snapshots agree on the same class with probability >= MIN_PROB.
  - The class name stays in the laptop's memory and log; it is never sent to any screen.
Runs on the laptop (about 100 MB for the model), so the phones' 4 GB budget is untouched.
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("sightline")
MODEL = Path(__file__).resolve().parents[1] / "models" / "MobileNetV2.mlmodel"
MIN_PROB = 0.3  # per-view confidence needed for a vote
AGREE = 2  # views that must agree on the same class
MAX_RANGE = 4.5  # metres: farther objects are too small in a 360 px snapshot
CROP_MIN = 40  # pixels: smaller crops say nothing reliable


def project_box(obj: dict, floor_y: float, frame: dict) -> tuple[int, int, int, int] | None:
    """The object's 3D box -> a rectangle (x0, y0, x1, y1) in the portrait JPEG, or None if not in view."""
    pose, (fx, fy, cx, cy), (ws, hs) = frame["pose"], frame["K"], frame["size"]
    inv = np.linalg.inv(pose)
    xs, zs, ys = (obj["x0"], obj["x1"]), (obj["z0"], obj["z1"]), (floor_y, floor_y + max(0.1, obj["top"]))
    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                pc = inv @ np.array([x, y, z, 1.0])
                depth = -pc[2]
                if depth < 0.3 or depth > MAX_RANGE:
                    return None
                u, v = fx * pc[0] / depth + cx, -fy * pc[1] / depth + cy  # landscape sensor pixels
                pts.append((hs - v, u))  # the snapshot is rotated upright: portrait x = h - v, y = u
    px, py = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(px), max(px), min(py), max(py)
    if x1 < 0 or y1 < 0 or x0 > hs or y0 > ws:
        return None
    padx, pady = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    r = (max(0, int(x0 - padx)), max(0, int(y0 - pady)), min(int(hs), int(x1 + padx)), min(int(ws), int(y1 + pady)))
    return r if r[2] - r[0] >= CROP_MIN and r[3] - r[1] >= CROP_MIN else None


class ObjectConfirmer:
    def __init__(self, classify=None) -> None:
        self._classify = classify  # injectable for tests: (PIL image) -> (label, prob)
        self.scene = None  # the SceneLabeler: when its SegFormer-B2 is available it replaces MobileNetV2 here
        self._model = None
        self.frames: dict[str, dict] = {}  # latest posed snapshot per phone
        self.votes: dict[tuple[int, int], list[tuple[str, float, float]]] = {}  # object key -> (label, prob, frame ts)
        self.confirmed: dict[tuple[int, int], str] = {}  # object key -> class (internal only)
        self.lock = threading.Lock()

    @staticmethod
    def key(obj: dict) -> tuple[int, int]:
        """Objects move a little as fusion refines them: identify them by their centre on a 40 cm grid."""
        return (round((obj["x0"] + obj["x1"]) / 2 / 0.4), round((obj["z0"] + obj["z1"]) / 2 / 0.4))

    def add_frame(self, pid: str, frame: dict) -> None:
        with self.lock:
            self.frames[pid] = frame

    def classify(self, img) -> tuple[str, float]:
        if self._classify is not None:
            return self._classify(img)
        if self._model is None:
            import coremltools as ct  # Core ML runs on the Mac's Neural Engine / GPU

            self._model = ct.models.MLModel(str(MODEL))
        out = self._model.predict({"image": img.resize((224, 224))})
        label = out["classLabel"]
        return label, float(out["classLabelProbs"].get(label, 0.0))

    def furniture_share(self, fr: dict, box: tuple[int, int, int, int]) -> tuple[str, float]:
        """SegFormer-B2's view of the object's region: ("furniture", share of furniture pixels) or the
        region's most common class. Scene segmentation knows indoor furniture far better than ImageNet."""
        from .scene import FURNITURE, GRID

        seg = self.scene.segmentation(fr)
        if seg is None:
            return "unknown", 0.0
        ws, hs = fr["size"]  # portrait snapshot is hs wide, ws tall
        x0, y0, x1, y1 = box
        cx0, cx1 = int(x0 / hs * GRID), max(int(x0 / hs * GRID) + 1, int(x1 / hs * GRID))
        cy0, cy1 = int(y0 / ws * GRID), max(int(y0 / ws * GRID) + 1, int(y1 / ws * GRID))
        region = seg[cy0:cy1, cx0:cx1].ravel()
        if not len(region):
            return "unknown", 0.0
        share = float(np.isin(region, list(FURNITURE)).mean())
        return ("furniture", share) if share >= MIN_PROB else (f"ade{int(np.bincount(region).argmax())}", share)

    def is_confirmed(self, obj: dict) -> bool:
        k = self.key(obj)
        return any(abs(k[0] - a) <= 1 and abs(k[1] - b) <= 1 for a, b in self.confirmed)

    def run(self, objects: list[dict], floor_y: float, budget: int = 6) -> int:
        """Classify up to `budget` unconfirmed objects from the freshest snapshots. Returns new confirmations."""
        from PIL import Image

        with self.lock:
            frames = list(self.frames.values())
        if not frames:
            return 0
        new = 0
        for obj in [o for o in objects if not o.get("confirmed") and not self.is_confirmed(o)][:budget]:
            k = self.key(obj)
            for fr in sorted(frames, key=lambda f: -f["ts"]):
                box = project_box(obj, floor_y, fr)
                if box is None:
                    continue
                votes = self.votes.setdefault(k, [])
                if any(abs(ts - fr["ts"]) < 0.5 for _, _, ts in votes):
                    break  # this snapshot already voted for this object
                try:
                    if self._classify is None and self.scene is not None and self.scene.available():
                        label, prob = self.furniture_share(fr, box)
                    else:
                        img = Image.open(io.BytesIO(fr["jpeg"])).convert("RGB").crop(box)
                        label, prob = self.classify(img)
                except Exception as e:  # a bad snapshot must never stop the map
                    log.info("classifier skipped a snapshot: %s", e)
                    break
                if prob >= MIN_PROB:
                    votes.append((label, prob, fr["ts"]))
                    del votes[:-6]
                    same = [v for v in votes if v[0] == label]
                    if len({round(v[2], 1) for v in same}) >= AGREE:
                        self.confirmed[k] = label
                        new += 1
                        log.warning("classifier confirmed a %s-height object at (%.1f, %.1f) as '%s' (%d views, internal only)",
                                    obj["kind"], (obj["x0"] + obj["x1"]) / 2, (obj["z0"] + obj["z1"]) / 2, label.split(",")[0], len(same))
                break
        return new


def parse_posed_frame(body: bytes) -> dict:
    """0x05 payload: f32[16] camera->shared | f32 fx, fy, cx, cy | u16 w, h (scaled landscape sensor) | JPEG."""
    import struct

    head = struct.Struct("<16f4f2H")
    if len(body) < head.size + 4:
        raise ValueError("short posed frame")
    vals = head.unpack_from(body)
    pose = np.array(vals[:16], dtype=np.float64).reshape(4, 4).T
    fx, fy, cx, cy = vals[16:20]
    w, h = vals[20], vals[21]
    jpeg = body[head.size:]
    if not (np.all(np.isfinite(pose)) and math.isfinite(fx + fy + cx + cy)) or fx <= 0 or fy <= 0 or not (16 <= w <= 2048 and 16 <= h <= 2048):
        raise ValueError("bad camera")
    if abs(np.linalg.det(pose[:3, :3]) - 1) > 0.05 or np.abs(pose[:3, 3]).max() > 500:
        raise ValueError("bad pose")
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    return {"pose": pose, "K": (fx, fy, cx, cy), "size": (w, h), "jpeg": jpeg, "ts": time.time()}
