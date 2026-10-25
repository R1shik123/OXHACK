"""Validation of everything a phone may send. Anything that doesn't parse is dropped.

Text messages are JSON (validated with pydantic). Binary messages start with a type byte:
  0x01  camera JPEG              (<= MAX_JPEG bytes)
  0x02  colored points, N x 16 B  little-endian float32 x,y,z (shared frame, metres),
                                  uint8 r,g,b, uint8 confidence (0-255)
  0x05  posed camera snapshot     f32[16] camera->shared | f32 fx,fy,cx,cy | u16 w,h | JPEG (see classify.py)
  0x03  depth frame               u8 version | u16 w | u16 h | f32[16] camera->shared (column-major)
                                  | f32 fx, fy, cx, cy | u16[w*h] depth mm | u8[w*h*3] rgb
                                  version 2 adds: u8 n | n x u16 x0, y0, x1, y1 = furniture outlines found by
                                  the phone's detector (YOLOv8n, Open Images), in depth-frame pixels; a door outline
                                  sets the top bit of x0, a stairs outline bit 14 (no class names: they are never needed)
                                  version 3 = version 2 layout, depth measured by a LiDAR sensor (trusted more)
"""

from __future__ import annotations

import math
import struct
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator

MAX_TEXT = 64 * 1024
MAX_JPEG = 300 * 1024
MAX_POINTS = 12000
WORLD_LIMIT = 500.0  # metres from the marker; beyond this is garbage or an attack
BIN_JPEG, BIN_POINTS, BIN_DEPTH, BIN_POSED_JPEG = 0x01, 0x02, 0x03, 0x05
MAX_DEPTH_PIXELS = 256 * 192
DEPTH_HEAD = struct.Struct("<BHH16f4f")
POINT = struct.Struct("<fffBBBB")

PlaneKind = Literal["wall", "door", "window", "floor", "ceiling", "table", "seat", "vertical", "horizontal"]
PinKind = Literal["hazard", "victim", "note", "rally"]


def _check_vec(v: list[float]) -> list[float]:
    if len(v) != 3 or not all(math.isfinite(x) and abs(x) <= WORLD_LIMIT for x in v):
        raise ValueError("bad vector")
    return [float(x) for x in v]


class Pose(BaseModel):
    t: Literal["pose"]
    pos: list[float]
    fwd: list[float]
    tracking: str | None = Field(default=None, max_length=32)
    wall_dist: float | None = Field(default=None, ge=0, le=100)
    act: str | None = Field(default=None, max_length=16)  # Apple Core Motion activity
    act_conf: int | None = Field(default=None, ge=0, le=2)

    _v = field_validator("pos", "fwd")(_check_vec)


class Status(BaseModel):
    t: Literal["status"]
    aligned: bool | None = None
    tracking: str | None = Field(default=None, max_length=32)
    battery: float | None = Field(default=None, ge=0, le=1)
    # stability telemetry from the phone
    mem_mb: int | None = Field(default=None, ge=-1, le=16384)
    thermal: int | None = Field(default=None, ge=0, le=3)  # 0 nominal, 1 fair, 2 serious, 3 critical
    ar_fps: float | None = Field(default=None, ge=0, le=240)
    depth_hz: float | None = Field(default=None, ge=0, le=60)
    mem_warnings: int | None = Field(default=None, ge=0, le=100000)
    memory_warning: bool | None = None


class Plane(BaseModel):
    t: Literal["plane"]
    id: str = Field(min_length=1, max_length=64)
    kind: PlaneKind
    corners: list[list[float]] = Field(min_length=3, max_length=8)

    @field_validator("corners")
    @classmethod
    def _corners(cls, v: list[list[float]]) -> list[list[float]]:
        return [_check_vec(c) for c in v]


class PlaneRemove(BaseModel):
    t: Literal["plane_remove"]
    id: str = Field(min_length=1, max_length=64)


class PlanesClear(BaseModel):
    t: Literal["planes_clear"]


class Pin(BaseModel):
    t: Literal["pin"]
    kind: PinKind
    label: str = Field(default="", max_length=40)
    pos: list[float]

    _v = field_validator("pos")(_check_vec)


class PinRemove(BaseModel):
    t: Literal["pin_remove"]
    id: str = Field(max_length=32)


class Xray(BaseModel):
    t: Literal["xray"]
    target: str | None = Field(default=None, max_length=32)


class Impact(BaseModel):
    """Accelerometer fall signature from the phone: free fall followed by an impact."""

    t: Literal["impact"]
    g: float = Field(ge=0, le=50)
    freefall_ms: int = Field(ge=0, le=5000)


class HazardResponse(BaseModel):
    """Responder's answer to "hazard ahead": keep away, or go on (which flags the commander)."""

    t: Literal["hazard"]
    id: str = Field(min_length=1, max_length=32)
    action: Literal["avoid", "proceed"]


class Nav(BaseModel):
    """Voice route the responder wants: to the command post, to the nearest exit, or none."""

    t: Literal["nav"]
    to: Literal["off", "home", "exit"]


class Ok(BaseModel):
    """Responder answered "Are you OK?"."""

    t: Literal["ok"]


PhoneMessage = TypeAdapter(Annotated[Union[Pose, Status, Plane, PlaneRemove, PlanesClear, Pin, PinRemove, Xray, Impact, Ok, HazardResponse, Nav], Field(discriminator="t")])


class AskMessage(BaseModel):
    role: Literal["user", "model"]
    text: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    """A question for the commander's AI assistant, with the conversation so far (oldest first)."""

    messages: list[AskMessage] = Field(min_length=1, max_length=24)

    @field_validator("messages")
    @classmethod
    def _last_is_question(cls, v: list[AskMessage]) -> list[AskMessage]:
        if v[-1].role != "user":
            raise ValueError("the last message must be the question")
        return v


def _check_xz(v: list[float]) -> list[float]:
    if len(v) != 2 or not all(math.isfinite(x) and abs(x) <= WORLD_LIMIT for x in v):
        raise ValueError("bad point")
    return [round(float(x), 3) for x in v]


class Labels(BaseModel):
    """Hand-made ground truth for one saved map: which area it is, its rooms and its fall-risk spots."""

    area: Literal["A", "B", "C"]
    rooms: list[list[list[float]]] = Field(default_factory=list, max_length=200)  # polygons of (x, z)
    risk: list[list[float]] = Field(default_factory=list, max_length=500)  # (x, z) points

    @field_validator("rooms")
    @classmethod
    def _rooms(cls, v):
        if any(not 3 <= len(poly) <= 40 for poly in v):
            raise ValueError("a room needs 3-40 corners")
        return [[_check_xz(p) for p in poly] for poly in v]

    @field_validator("risk")
    @classmethod
    def _risk(cls, v):
        return [_check_xz(p) for p in v]


class RouteUpdate(BaseModel):
    """The commander's waypoint route: stop order, who walks it, and whether it ends back at command."""

    order: list[str] = Field(default_factory=list, max_length=40)
    assignee: str | None = Field(default=None, max_length=32)
    home: bool = True


class PairRequest(BaseModel):
    code: str = Field(min_length=8, max_length=64)
    name: str = Field(min_length=1, max_length=32)
    platform: Literal["ios", "android", "sim"]


class AdminPin(BaseModel):
    t: Literal["pin"]
    kind: Literal["hazard", "victim", "note", "rally", "waypoint"]  # waypoints: commander only (see app.py)
    label: str = Field(default="", max_length=40)
    pos: list[float]

    _v = field_validator("pos")(_check_vec)


def parse_points(data: bytes) -> list[tuple[float, float, float, int, int, int]]:
    """Decode a 0x02 payload (without the type byte); drops non-finite or far-away points."""
    n = len(data) // POINT.size
    if n == 0 or n > MAX_POINTS or len(data) % POINT.size:
        raise ValueError("bad point payload")
    out = []
    for x, y, z, r, g, b, conf in POINT.iter_unpack(data):
        if conf >= 64 and math.isfinite(x + y + z) and abs(x) < WORLD_LIMIT and abs(y) < WORLD_LIMIT and abs(z) < WORLD_LIMIT:
            out.append((x, y, z, r, g, b))
    return out


def pack_points(points: list[tuple[float, float, float, int, int, int]]) -> bytes:
    """Encode points the way phones send them (used by the simulator and tests)."""
    return bytes([BIN_POINTS]) + b"".join(POINT.pack(x, y, z, r, g, b, 255) for x, y, z, r, g, b in points)


def parse_depth_frame(data: bytes) -> dict:
    """Decode and sanity-check a 0x03 payload (without the type byte). Raises ValueError if malformed."""
    import numpy as np

    if len(data) < DEPTH_HEAD.size:
        raise ValueError("short depth frame")
    head = DEPTH_HEAD.unpack_from(data)
    version, w, h = head[0], head[1], head[2]
    if version not in (1, 2, 3) or not (8 <= w <= 256 and 8 <= h <= 192) or w * h > MAX_DEPTH_PIXELS:
        raise ValueError("bad depth frame size")
    base = DEPTH_HEAD.size + w * h * 5
    boxes: list[tuple[int, int, int, int]] = []
    doors: list[tuple[int, int, int, int]] = []  # door outlines: x0 carries the top bit (0x8000)
    stairs: list[tuple[int, int, int, int]] = []  # stairs outlines: x0 carries bit 14 (0x4000)
    if version == 1:
        if len(data) != base:
            raise ValueError("bad depth frame length")
    else:
        if len(data) < base + 1:
            raise ValueError("bad depth frame length")
        n = data[base]
        if n > 32 or len(data) != base + 1 + 8 * n:
            raise ValueError("bad furniture box list")
        for i in range(n):
            x0, y0, x1, y1 = struct.unpack_from("<4H", data, base + 1 + 8 * i)
            door, stair = bool(x0 & 0x8000), bool(x0 & 0x4000)
            x0 &= 0x3FFF
            if x0 < x1 <= w and y0 < y1 <= h:
                (doors if door else stairs if stair else boxes).append((x0, y0, x1, y1))
    m = np.array(head[3:19], dtype=np.float64).reshape(4, 4).T  # columns -> row-major matrix
    fx, fy, cx, cy = head[19:23]
    if not (np.all(np.isfinite(m)) and math.isfinite(fx + fy + cx + cy)) or fx <= 0 or fy <= 0:
        raise ValueError("bad camera")
    if np.abs(m[:3, 3]).max() > WORLD_LIMIT or abs(np.linalg.det(m[:3, :3]) - 1) > 0.05:
        raise ValueError("bad pose")
    off = DEPTH_HEAD.size
    depth = np.frombuffer(data, dtype="<u2", count=w * h, offset=off).reshape(h, w)
    rgb = np.frombuffer(data, dtype=np.uint8, count=w * h * 3, offset=off + w * h * 2).reshape(h, w, 3)
    return {"w": w, "h": h, "pose": m, "K": (fx, fy, cx, cy), "depth": depth, "rgb": rgb, "boxes": boxes, "doors": doors, "stairs": stairs, "lidar": version == 3}


def pack_depth_frame(pose, K, depth, rgb, boxes=None) -> bytes:
    """Encode a depth frame the way phones send it (simulator + tests). pose: 4x4 camera->shared.
    boxes: optional furniture outlines (x0, y0, x1, y1) in depth-frame pixels -> version 2."""
    import numpy as np

    h, w = depth.shape
    head = DEPTH_HEAD.pack(2 if boxes is not None else 1, w, h, *np.asarray(pose, dtype=np.float32).T.reshape(-1), *K)
    out = bytes([BIN_DEPTH]) + head + np.asarray(depth, dtype="<u2").tobytes() + np.asarray(rgb, dtype=np.uint8).tobytes()
    if boxes is not None:
        out += bytes([len(boxes)]) + b"".join(struct.pack("<4H", *b) for b in boxes)
    return out
