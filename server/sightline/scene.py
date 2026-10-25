"""What is in that gap? Scene segmentation on the laptop, for windows, blinds and pillars.

Glass is see-through, so depth (monocular AI or ARKit) passes through windows and leaves a gap in the
wall; pillars are too narrow to fit a wall line. Fusion therefore hands us *candidates*: gaps between
two pieces of the same wall line, and tall narrow shapes. We look at them in the phones' real camera
snapshots (0x05, with the pose they were taken from) using SegFormer-B2, a pre-trained scene
segmentation network (NVIDIA, trained on the ADE20K dataset: wall, windowpane, blind, curtain, door,
column ...), converted to Core ML and run on the Mac. Nothing is trained on our data.

Rules that keep it honest (same spirit as classify.py):
  - It only labels gaps and shapes that fusion measured; it never invents geometry or imagery.
  - A label needs two different snapshots that agree, and a clear majority of the votes.
  - Labels stay on the laptop: the maps only ever show walls (windows / blinds count as walls).
About 300 MB of RAM and ~60-80 ms per snapshot on an M1, at most a few snapshots every 3 s.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("sightline")
MODEL = Path(__file__).resolve().parents[1] / "models" / "SegFormerADE.mlpackage"
GRID = 128  # the model's output map: 128 x 128 labels for a 512 x 512 input
MAX_RANGE = 6.0  # metres: farther away, a 360 px snapshot shows too little
AGREE = 2  # snapshots that must agree
# ADE20K class ids -> what they mean for the map
# ADE20K ids that are furniture (the object confirmer in classify.py uses them)
FURNITURE = {7, 10, 15, 19, 23, 24, 30, 31, 33, 35, 44, 45, 55, 62, 64, 69, 75, 97, 99, 110}
CLASSES = {53: "stairs", 59: "stairs", 121: "stairs", 96: "stairs", 0: "wall", 8: "window", 147: "window", 18: "window", 63: "window", 14: "door", 58: "door", 42: "column"}


def project_point(p, frame: dict) -> tuple[float, float] | None:
    """World point -> (x, y) in the upright portrait snapshot, or None when behind / too far / off-picture."""
    pose, (fx, fy, cx, cy), (ws, hs) = frame["pose"], frame["K"], frame["size"]
    pc = np.linalg.inv(pose) @ np.array([p[0], p[1], p[2], 1.0])
    d = -pc[2]
    if d < 0.4 or d > MAX_RANGE:
        return None
    u, v = fx * pc[0] / d + cx, -fy * pc[1] / d + cy  # landscape sensor pixels
    x, y = hs - v, u  # the snapshot is rotated upright: portrait x = h - v, y = u
    return (x, y) if 0 <= x < hs and 0 <= y < ws else None


class SceneLabeler:
    def __init__(self, frames_from, segment=None) -> None:
        self.frames_from = frames_from  # the ObjectConfirmer: it already keeps each phone's latest snapshot
        self._segment = segment  # injectable for tests: (PIL image) -> 128x128 array of ADE20K ids
        self._model = None
        self._seen: dict[float, np.ndarray] = {}  # snapshot ts -> label map (small cache)
        self.votes: dict[tuple, list[tuple[float, str]]] = {}  # candidate key -> (snapshot ts, class)
        self.labels: dict[tuple, str] = {}  # decided: key -> "window" | "wall" | "door" | "column"
        self.lock = threading.Lock()

    def segment(self, img) -> np.ndarray:
        if self._segment is not None:
            return self._segment(img)
        if self._model is None:
            import coremltools as ct

            self._model = ct.models.MLModel(str(MODEL))
        return np.array(self._model.predict({"image": img.resize((512, 512))})["labels"]).reshape(GRID, GRID)

    def available(self) -> bool:
        return self._segment is not None or MODEL.exists()

    def segmentation(self, fr: dict) -> np.ndarray | None:
        """Label map (GRID x GRID ADE20K ids) for one snapshot, computed once and cached."""
        from PIL import Image

        seg = self._seen.get(fr["ts"])
        if seg is None:
            try:
                seg = self.segment(Image.open(io.BytesIO(fr["jpeg"])).convert("RGB"))
            except Exception as e:  # a bad snapshot must never stop the map
                log.info("scene segmentation skipped a snapshot: %s", e)
                return None
            with self.lock:
                self._seen[fr["ts"]] = seg
                for ts in sorted(self._seen)[:-12]:
                    del self._seen[ts]
        return seg

    def label(self, key: tuple) -> str | None:
        return self.labels.get(key)

    def run(self, candidates: list[dict], max_frames: int = 4) -> int:
        """Vote on fusion's candidates using the freshest snapshots. Returns how many got a new label."""
        from PIL import Image

        if not candidates or not MODEL.exists() and self._segment is None:
            return 0
        with self.frames_from.lock:
            frames = sorted(self.frames_from.frames.values(), key=lambda f: -f["ts"])[:max_frames]
        new = 0
        for fr in frames:
            seg = self.segmentation(fr)
            if seg is None:
                continue
            ws, hs = fr["size"]
            for c in candidates:
                hits = []
                for p in c["points"]:
                    xy = project_point(p, fr)
                    if xy is not None:
                        cls = CLASSES.get(int(seg[min(GRID - 1, int(xy[1] / ws * GRID)), min(GRID - 1, int(xy[0] / hs * GRID))]))
                        if cls:
                            hits.append(cls)
                if not hits:
                    continue
                cls = max(set(hits), key=hits.count)
                votes = self.votes.setdefault(c["key"], [])
                if any(abs(ts - fr["ts"]) < 0.3 for ts, _ in votes):
                    continue  # this snapshot already voted on this candidate
                votes.append((fr["ts"], cls))
                del votes[:-8]
                same = [v for v in votes if v[1] == cls]
                if len(same) >= AGREE and len(same) > 0.6 * len(votes) and self.labels.get(c["key"]) != cls:
                    self.labels[c["key"]] = cls
                    new += 1
                    log.warning("scene AI: %s at (%.1f, %.1f) looks like a %s (%d of %d snapshots, internal only)",
                                c["what"], c["at"][0], c["at"][1], cls, len(same), len(votes))
        return new
