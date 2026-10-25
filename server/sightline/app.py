"""Sightline server: two listeners sharing one in-memory world.

  Commander  http://127.0.0.1:8000   loopback only; needs the admin token printed at startup
  Phones     https://<lan-ip>:8443   TLS (pinned by the phones), paired devices only

Phones pair once (single-use QR code from the commander page) and then authenticate every
request and WebSocket with `Authorization: Bearer <device token>`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import os
import base64
import socket
import threading
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import httpx
import segno
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import protocol as P
from .classify import parse_posed_frame
from .motion import set_down
from .gemini import Gemini, configured as gemini_configured, situation
from . import voice as V
from .evaluate import load_tuned
from .maps import MapStore, capture, restore
from .records import Recorder
from .security import DATA, RateLimiter, Registry, hash_token
from .state import World

log = logging.getLogger("sightline")
ROOT = Path(__file__).resolve().parents[2]
WEB, MARKER = ROOT / "web", ROOT / "marker"
APK = ROOT / "android" / "app" / "build" / "outputs" / "apk" / "release" / "app-release.apk"
TICK = 0.1
PHONE_PORT, ADMIN_PORT, SCREEN_PORT = 8443, 8000, 8444  # SCREEN_PORT: commander page for other computers (TLS)
ADMIN_COOKIE = "sl_admin"

world = World()
registry = Registry()
phone_sockets: dict[str, WebSocket] = {}
viewer_sockets: set[WebSocket] = set()  # commander page(s)
map_sockets: dict[WebSocket, str] = {}  # phones' Map tab (read-only map + pins), socket -> device id
recorder = Recorder()
maps = MapStore(recorder.fernet)  # saved maps: laptop only, encrypted
gemini = Gemini()  # the commander's AI assistant (key stays on this laptop)
speech = V.Speech()  # ElevenLabs audio for phones' voice cues (key stays on this laptop)
load_tuned()  # detector settings tuned on the training area (if an evaluation has been run)
# Entrance marker in use: the printed 18 cm sheet, or the same image shown 15 cm wide on the commander's
# screen (a 13" laptop screen is under 18 cm tall). Phones are told the width so their distances stay true.
MARKER_SIZES = {"print": 0.18, "screen": 0.15}
marker = {"mode": "print"}


def screen_size() -> dict | None:
    """This Mac's main display: physical size (mm) and size in points, so the page can draw true centimetres."""
    try:
        import ctypes

        class CGSize(ctypes.Structure):
            _fields_ = [("w", ctypes.c_double), ("h", ctypes.c_double)]

        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayScreenSize.restype, cg.CGDisplayScreenSize.argtypes = CGSize, [ctypes.c_uint32]
        cg.CGDisplayPixelsWide.restype = cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
        cg.CGDisplayPixelsWide.argtypes = cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
        cg.CGDisplayCopyDisplayMode.restype, cg.CGDisplayCopyDisplayMode.argtypes = ctypes.c_void_p, [ctypes.c_uint32]
        cg.CGDisplayModeGetPixelWidth.restype = cg.CGDisplayModeGetWidth.restype = ctypes.c_size_t
        cg.CGDisplayModeGetPixelWidth.argtypes = cg.CGDisplayModeGetWidth.argtypes = cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]
        d = cg.CGMainDisplayID()
        mm = cg.CGDisplayScreenSize(d)
        if mm.w <= 0:
            return None
        mode = cg.CGDisplayCopyDisplayMode(d)
        backing = cg.CGDisplayModeGetPixelWidth(mode) / max(1, cg.CGDisplayModeGetWidth(mode)) if mode else 1.0
        if mode:
            cg.CGDisplayModeRelease(mode)
        # backing: device pixels per point (2 on Retina); the page divides by devicePixelRatio to undo browser zoom
        return {"mm_w": round(mm.w, 1), "mm_h": round(mm.h, 1), "pt_w": cg.CGDisplayPixelsWide(d), "pt_h": cg.CGDisplayPixelsHigh(d),
                "backing": round(backing, 3)}
    except (OSError, AttributeError):
        return None


SCREEN = screen_size()
capture_phones: set[str] = set()  # phones included in the running room capture
depth_limit = RateLimiter(rate=4, burst=6)
alert_stats = {"confirmed": 0, "false_alarm": 0}  # how often alerts were right (from "Confirm real" / "I'm OK")
world.alert_stats = alert_stats
text_limit = RateLimiter(rate=40, burst=80)
frame_limit = RateLimiter(rate=6, burst=10)
points_limit = RateLimiter(rate=6, burst=10)
pair_limit = RateLimiter(rate=10 / 60, burst=10)
ask_limit = RateLimiter(rate=12 / 60, burst=6)  # Ask Gemini: about one question every 5 s, short bursts
voice_limit = RateLimiter(rate=3, burst=12)  # voice cue audio per phone
CERT_FP = ""  # set by __main__ once the certificate exists


def lan_ip() -> str:
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
        return s.getsockname()[0]
    return "127.0.0.1"


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
}


def add_security_headers(app: FastAPI) -> None:
    @app.middleware("http")
    async def _headers(request: Request, call_next):
        resp = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        return resp


async def _send_json(ws: WebSocket, msg: dict) -> bool:
    try:
        await ws.send_text(json.dumps(msg, separators=(",", ":")))
        return True
    except Exception:
        return False


async def _send_bytes(ws: WebSocket, data: bytes) -> bool:
    try:
        await ws.send_bytes(data)
        return True
    except Exception:
        return False


def remove_pin(pin_id: str) -> None:
    world.pins.pop(pin_id, None)
    world.route_forget(pin_id)


def all_viewers() -> list[WebSocket]:
    return list(viewer_sockets) + list(map_sockets)


async def send_viewers(msg: dict | bytes) -> None:
    for ws in all_viewers():
        ok = await (_send_bytes(ws, msg) if isinstance(msg, bytes) else _send_json(ws, msg))
        if not ok:
            viewer_sockets.discard(ws)
            map_sockets.pop(ws, None)


def motion_tick() -> None:
    """Plain physical check on each track: a phone with no hand tremor at all is lying somewhere."""
    now = time.time()
    for p in world.phones.values():
        if not p.pos or not p.connected:
            continue
        p.phone_down = set_down(p.motion.features(now))
        if p.phone_down and not p.set_down_noted:
            p.set_down_noted = True
            world.event("set_down", f"{p.name}'s phone is lying still (not being held)", who=p.id)
        elif not p.phone_down:
            p.set_down_noted = False


THERMAL = ["normal", "warm", "hot", "critical"]


def note_health(p, m) -> None:
    """Stability telemetry: keep the latest readings, flag memory warnings and heat on the timeline."""
    h = p.health
    for k in ("mem_mb", "thermal", "ar_fps", "depth_hz", "mem_warnings"):
        v = getattr(m, k)
        if v is not None:
            h[k] = round(v, 1) if isinstance(v, float) else v
    if m.memory_warning:
        world.event("health", f"{p.name}'s phone is low on memory: 3D scanning paused for 15 s", who=p.id)
        log.warning("memory warning from %s (%s MB)", p.name, h.get("mem_mb"))
    t = h.get("thermal", 0)
    if t >= 2 and h.get("_last_thermal", 0) < 2:
        world.event("health", f"{p.name}'s phone is {THERMAL[t]}: scanning slowed down", who=p.id)
    h["_last_thermal"] = t
    if recorder.recording("session"):
        recorder.write("session", {"t": "health", "id": p.id, **{k: v for k, v in h.items() if not k.startswith("_")}})


def health_log() -> None:
    """Once a minute: one line per phone in the server log (a soak test reads like a chart)."""
    for p in world.phones.values():
        h = p.health
        if h:
            log.warning("health %-10s mem %4s MB  heat %-8s  AR %4s fps  depth %3s Hz  mem warnings %s", p.name, h.get("mem_mb", "?"),
                        THERMAL[h.get("thermal", 0)], h.get("ar_fps", "?"), h.get("depth_hz", "?"), h.get("mem_warnings", 0))


def note_label(label: str) -> None:
    alert_stats[label] = alert_stats.get(label, 0) + 1


def record_tick() -> None:
    """Opt-in session recording: compact 2 Hz state for ML training and replay."""
    if not recorder.recording("session"):
        return
    for p in world.phones.values():
        if p.pos:
            recorder.write("session", {"t": "pose", "id": p.id, "pos": [round(v, 3) for v in p.pos],
                                       "fwd": [round(v, 3) for v in (p.fwd or [0, 0, 0])], "alert": (p.vitals.alert or {}).get("kind"),
                                       "activity": p.activity, "phone_down": p.phone_down})


def voice_tick() -> None:
    """Work out each phone's spoken cues; start fetching ElevenLabs audio for new ones so it's ready when asked."""
    phones = [p for p in list(world.phones.values()) if p.connected and p.pos]
    if not phones:
        return
    from .predict import exit_doors
    walls = world.merged_walls()
    zones = world.risk_zones()
    exits = exit_doors(walls)
    now = time.time()
    for p in phones:
        p.voice.update(p, world, now, walls, zones, exits)
        fresh, p.voice.fresh = p.voice.fresh, []
        for cue in fresh:
            if not cue["local"]:
                speech.issue(cue["text"])
                if V.configured():
                    asyncio.get_running_loop().create_task(speech.get(cue["id"]))


async def broadcast_loop() -> None:
    last_walls = last_cov = last_mesh = last_sem = last_rec = last_ml = last_plan = last_voice = 0.0
    last_health = last_cls = last_scene = last_planes = last_paint = last_remesh = last_align = time.monotonic()
    sem_frames = -1
    milestones = {25, 50, 75, 90}
    while True:
        await asyncio.sleep(TICK)
        now = time.monotonic()
        planes = world.take_plane_delta()
        fine, coarse = world.take_point_deltas()
        if world.walls_dirty and now - last_walls > 1.0:
            world.walls_dirty, last_walls = False, now
            walls = world.walls_message()
            await send_viewers(walls)
            for ws in list(phone_sockets.values()):
                await _send_json(ws, walls)
        if now - last_cov > 1.0:
            last_cov = now
            cov = world.coverage.message(full=False)
            if cov:
                await send_viewers(cov)
                pct = cov.get("pct") or 0
                for m in sorted(milestones):
                    if pct >= m:
                        milestones.discard(m)
                        world.event("coverage", f"{m}% of the building searched")
        if now - last_align > 5.0:  # fine-align each phone to the reference phone's walls
            last_align = now
            for text in world.refine_alignment():
                world.event("align", text)
        if now - last_planes > 3.0:  # flat surfaces: flatten the mesh onto them, give them photo textures
            last_planes = now
            fy = world.floor_y() if any(p["kind"] == "floor" for p in world.planes.values()) else await asyncio.to_thread(world.fusion.floor_estimate)
            segs = [(p, q) for p, q, y0, y1, k in world.merged_walls() if k == "wall"]
            ext = None
            if segs:
                xs = [c for p, q in segs for c in (p[0], q[0])]
                zs = [c for p, q in segs for c in (p[1], q[1])]
                ext = (min(xs) - 0.3, min(zs) - 0.3, max(xs) + 0.3, max(zs) + 0.3)
            world.fine.set_planes(fy, segs, ext)
            if world.fusion.set_planes(fy, segs, ext) and now - last_remesh > 10:
                last_remesh = now
                for layer in (world.fusion, world.fine):
                    with layer.lock:
                        layer.dirty.update(layer.meshes.keys())  # redo meshes against the new planes
            before = world.textures.ids()
            world.textures.set_geometry(segs, fy, ext)
            if world.textures.ids() != before:
                await send_viewers({"t": "textures", "ids": sorted(world.textures.ids())})
        if now - last_paint > 2.0:
            last_paint = now
            try:
                if await asyncio.to_thread(world.textures.paint):
                    for m in await asyncio.to_thread(world.textures.messages):
                        await send_viewers(m)
            except Exception as e:  # looks only: never take the live map down
                log.warning("textures: %s", e)
        if now - last_mesh > 0.5:  # surface meshes of blocks fusion touched since last time
            last_mesh = now
            fine_dirty = world.fine.take_dirty(limit=60)
            if fine_dirty:  # fine 2 cm meshes first, so the 4 cm layer knows where to step aside
                before = set(world.fine.solid)
                meshes = await asyncio.to_thread(lambda: [(k, world.fine.mesh_block(k)) for k in fine_dirty])
                for k, m in meshes:
                    if m is not None:
                        world.fine.meshes[k] = m
                        await send_viewers(m)
                flipped = before ^ world.fine.solid
                if flipped:
                    with world.fusion.lock:
                        world.fusion.dirty.update(tuple(c // 2 for c in k) for k in flipped if tuple(c // 2 for c in k) in world.fusion.tsdf)
            dirty = world.fusion.take_dirty(limit=80)
            if dirty:
                meshes = await asyncio.to_thread(lambda: [(k, world.fusion.mesh_block(k)) for k in dirty])
                for k, m in meshes:
                    if m is not None:
                        world.fusion.meshes[k] = m
                        await send_viewers(m)
        if now - last_sem > 3.0 and world.fusion.frames != sem_frames:  # walls + furniture from fused depth
            last_sem, sem_frames = now, world.fusion.frames
            walls, objs = await asyncio.to_thread(world.fusion.semantics, world.floor_y(), world.scene.label)
            if world.set_semantics(walls, objs):
                msg = {"t": "objects", "objects": objs}
                await send_viewers(msg)
                for ws in list(phone_sockets.values()):
                    await _send_json(ws, msg)
        if world.new_events:
            evs, world.new_events = world.new_events, []
            for ev in evs:
                recorder.write("session", {"t": "event", **ev})
            await send_viewers({"t": "events", "events": evs})
        if now - last_rec > 0.5:
            last_rec = now
            record_tick()
        if now - last_scene > 3.0 and world.fusion.candidates:  # scene AI: windows / pillars in wall gaps
            last_scene = now
            try:
                if await asyncio.to_thread(world.scene.run, list(world.fusion.candidates)):
                    sem_frames = -1  # a new label: redo the walls on the next pass
            except Exception as e:  # advice layer: never take the live map down
                log.warning("scene AI: %s", e)
        if now - last_cls > 3.0 and world.objects:  # classifier second opinion on unconfirmed fused objects
            last_cls = now
            try:
                if await asyncio.to_thread(world.confirmer.run, world.objects, world.floor_y()):
                    objs = [dict(o) for o in world.objects]
                    if world.set_semantics(world.depth_walls, objs):
                        msg = {"t": "objects", "objects": world.objects}
                        await send_viewers(msg)
            except Exception as e:  # advice layer: never take the live map down
                log.warning("classifier: %s", e)
        if now - last_health > 60:
            last_health = now
            health_log()
        if now - last_ml > 0.5:
            last_ml = now
            motion_tick()
        if now - last_plan > 1.0:
            last_plan = now
            try:
                world.plan = await asyncio.to_thread(world.planner.plan, world)
            except Exception as e:  # planning is advice; never let it take the live map down
                log.warning("planner: %s", e)
        if now - last_voice > 0.5:
            last_voice = now
            try:
                voice_tick()
            except Exception as e:  # spoken guidance is advice; never let it take the live map down
                log.warning("voice: %s", e)
        state = world.viewer_state()
        state["recording"] = recorder.status()
        for ws in all_viewers():
            ok = await _send_json(ws, state)
            ok = ok and (planes is None or await _send_json(ws, planes))
            ok = ok and (fine is None or await _send_bytes(ws, fine))
            if not ok:
                viewer_sockets.discard(ws)
                map_sockets.pop(ws, None)
        for pid, ws in list(phone_sockets.items()):
            ok = await _send_json(ws, world.phone_state(pid))
            ok = ok and (planes is None or await _send_json(ws, planes))
            ok = ok and (coarse is None or await _send_bytes(ws, coarse))
            if not ok:
                phone_sockets.pop(pid, None)


def fusion_worker(stop: threading.Event) -> None:
    """Integrates queued depth frames off the event loop (numpy releases the GIL for the heavy parts)."""
    while not stop.is_set():
        try:
            frame = world.fusion.queue.popleft()
        except IndexError:
            stop.wait(0.02)
            continue
        try:
            world.fusion.integrate(frame)
            world.fine.integrate(frame)  # only the close-up part of the frame (within FINE_RANGE) lands here
        except Exception as e:  # a bad frame must never kill fusion
            log.warning("fusion: dropped frame (%s)", e)


_loop_task: asyncio.Task | None = None
_fusion_stop = threading.Event()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """Both apps share one broadcast loop; whichever starts first runs it."""
    global _loop_task, _fusion_stop
    if _loop_task is None or _loop_task.done():
        _loop_task = asyncio.create_task(broadcast_loop())
        _fusion_stop = threading.Event()
        threading.Thread(target=fusion_worker, args=(_fusion_stop,), daemon=True, name="fusion").start()
    yield
    if _loop_task is not None:
        _loop_task.cancel()
        _loop_task = None
        _fusion_stop.set()
        for kind in list(recorder.active):
            recorder.stop(kind)


def bearer(request_headers) -> str | None:
    h = request_headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else None


# ======================================================================================
# Phone listener (TLS, LAN)
# ======================================================================================
phone_app = FastAPI(title="Sightline phones", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
add_security_headers(phone_app)


@phone_app.get("/api/health")
def phone_health() -> dict:
    return {"ok": True, "service": "sightline"}


@phone_app.post("/api/pair")
async def pair(request: Request) -> JSONResponse:
    ip = request.client.host if request.client else "?"
    if not pair_limit.allow(ip):
        raise HTTPException(429, "too many pairing attempts")
    try:
        req = P.PairRequest.model_validate_json(await request.body())
    except ValidationError:
        raise HTTPException(400, "bad pairing request")
    result = registry.pair(req.code, req.name.strip(), req.platform)
    if result is None:
        raise HTTPException(403, "pairing code invalid or expired")
    dev, token = result
    phone = world.phone(dev.id, dev.name)
    phone.platform = dev.platform
    log.warning("paired %s (%s, %s) from %s", dev.name, dev.id, dev.platform, ip)
    world.event("paired", f"{dev.name} joined ({dev.platform})", who=dev.id)
    return JSONResponse({"device_id": dev.id, "token": token, "name": dev.name, "color": phone.color})


@phone_app.get("/download/sightline.apk")
def download_apk(c: str = "", request: Request = None) -> Response:
    """Android app download: only for someone holding a live (unused, unexpired) pairing code."""
    ip = request.client.host if request and request.client else "?"
    if not pair_limit.allow("apk:" + ip):
        raise HTTPException(429)
    if not registry.code_valid(c) or not APK.exists():
        raise HTTPException(404)
    return FileResponse(APK, media_type="application/vnd.android.package-archive", filename="sightline.apk")


@phone_app.get("/api/frame/{pid}.jpg")
def phone_frame(pid: str, request: Request) -> Response:
    if registry.authenticate(bearer(request.headers)) is None:
        raise HTTPException(401)
    return _frame(pid)


def _require_device(request: Request):
    dev = registry.authenticate(bearer(request.headers))
    if dev is None:
        raise HTTPException(401)
    return dev


@phone_app.get("/api/voice/{cid}.mp3")
async def phone_voice(cid: str, request: Request) -> Response:
    """ElevenLabs audio for a voice cue this laptop issued (by id: phones can't make it say anything else)."""
    dev = _require_device(request)
    if not voice_limit.allow(dev.id):
        raise HTTPException(429)
    if not re.fullmatch(r"[0-9a-f]{20}", cid):
        raise HTTPException(404)
    data = await speech.get(cid)
    if data is None:
        raise HTTPException(404)  # not set up / not issued / ElevenLabs failing: the phone uses its own voice
    return Response(data, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@phone_app.get("/map")
def phone_map(request: Request) -> FileResponse:
    """The commander's map page, for a phone's Map tab (loaded through the app's pinned connection)."""
    _require_device(request)
    return FileResponse(WEB / "index.html")


@phone_app.get("/static/{path:path}")
def phone_static(path: str, request: Request) -> FileResponse:
    _require_device(request)
    f = (WEB / path).resolve()
    if WEB.resolve() not in f.parents or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@phone_app.websocket("/ws/viewer")
async def ws_map(ws: WebSocket) -> None:
    """Read-only map stream for a phone's Map tab: same data as the commander, pins allowed, no admin actions."""
    dev = registry.authenticate(bearer(ws.headers))
    if dev is None or dev.id not in phone_sockets:  # only while that phone's main session is live
        await ws.close(code=4401)
        return
    await ws.accept()
    snap = world.snapshot()
    snap["me"] = dev.id
    await send_initial(ws, snap)
    map_sockets[ws] = dev.id
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > P.MAX_TEXT or not text_limit.allow(dev.id):
                continue
            m = json.loads(raw)
            if m.get("t") == "pin":
                pin = P.AdminPin.model_validate(m)
                if pin.kind != "waypoint" and len(world.pins) < 200:  # waypoints are the commander's only
                    world.add_pin(pin.kind, pin.label, pin.pos, dev.name)
            elif m.get("t") == "pin_remove":
                if world.pins.get(str(m.get("id")), {}).get("kind") != "waypoint":
                    remove_pin(str(m.get("id")))
    except (WebSocketDisconnect, RuntimeError, ValueError, ValidationError):
        pass
    finally:
        map_sockets.pop(ws, None)


def _frame(pid: str) -> Response:
    p = world.phones.get(pid)
    if not p or not p.frame:
        return JSONResponse({"error": "no frame"}, status_code=404)
    return Response(p.frame, media_type="image/jpeg", headers={"X-Frame-Age": f"{time.time() - p.frame_ts:.2f}"})


def handle_phone_text(pid: str, raw: str) -> None:
    m = P.PhoneMessage.validate_json(raw)
    p = world.phone(pid)
    p.last_seen = time.time()
    if isinstance(m, P.Pose):
        pos, fwd = world.correct_pose(pid, m.pos, m.fwd)  # per-phone fine alignment (align.py)
        world.set_pose(p, pos, fwd)
        hazards = [pin for pin in world.pins.values() if pin["kind"] == "hazard"]
        for text in p.hazard.update(m.pos, hazards, time.time()):
            world.event("hazard", f"{p.name} {text}", who=pid)
        p.wall_dist = m.wall_dist
        if m.act:
            p.activity, p.act_conf = m.act, m.act_conf or 0
        if m.tracking:
            p.tracking = m.tracking
    elif isinstance(m, P.HazardResponse):
        text = p.hazard.respond(m.id, m.action, time.time())
        if text:
            world.event("hazard", f"{p.name} {text}", who=pid)
    elif isinstance(m, P.Status):
        if m.aligned is not None:
            p.aligned = m.aligned
            if not m.aligned:
                p.pos = None
        p.tracking = m.tracking or p.tracking
        p.battery = m.battery if m.battery is not None else p.battery
        note_health(p, m)
    elif isinstance(m, P.Plane):
        if sum(1 for v in world.planes.values() if v["owner"] == pid) < 3000 or f"{pid}:{m.id}" in world.planes:
            world.upsert_plane(pid, m.id, m.kind, m.corners)  # corrected inside, raw kept
            if pid in capture_phones and recorder.recording("capture"):
                recorder.write("capture", {"t": "plane", "id": pid, "plane": m.model_dump()})
    elif isinstance(m, P.PlaneRemove):
        world.remove_plane(f"{pid}:{m.id}")
    elif isinstance(m, P.PlanesClear):
        for k in [k for k, v in world.planes.items() if v["owner"] == pid]:
            world.remove_plane(k)
    elif isinstance(m, P.Pin):
        if len(world.pins) < 200:
            world.add_pin(m.kind, m.label, m.pos, p.name)
    elif isinstance(m, P.PinRemove):
        remove_pin(m.id)
    elif isinstance(m, P.Xray):
        p.xray_target = m.target if m.target in world.phones else None
    elif isinstance(m, P.Impact):
        p.vitals.impact(m.g, time.time())
        world.event("impact", f"{p.name}: impact {m.g:.1f} g after {m.freefall_ms} ms free fall", who=pid)
    elif isinstance(m, P.Nav):
        if m.to != p.nav:
            p.nav = m.to
            p.voice.reset_nav()
            if m.to != "off":
                world.event("route", f"{p.name} asked for voice guidance to {V.DEST[m.to]}", who=pid)
    elif isinstance(m, P.Ok):
        if p.vitals.alert:  # a label for the ML model: this alert was a false alarm
            world.event("label", f"{p.name} answered \"I'm OK\"", who=pid, alert=p.vitals.alert["kind"], label="false_alarm")
            note_label("false_alarm")
        p.vitals.ok(time.time())


def handle_phone_bytes(pid: str, data: bytes) -> None:
    if not data:
        return
    kind, body = data[0], data[1:]
    p = world.phone(pid)
    capturing = pid in capture_phones and recorder.recording("capture")
    if kind == P.BIN_JPEG:
        if len(body) <= P.MAX_JPEG and body[:2] == b"\xff\xd8" and frame_limit.allow(pid):
            if capturing and time.time() - p.frame_ts > 0.9:  # ~1 camera frame per second is enough to tune with
                recorder.write("capture", {"t": "jpeg", "id": pid, "b64": base64.b64encode(body).decode()})
            p.frame, p.frame_ts = body, time.time()
    elif kind == P.BIN_POSED_JPEG:
        if len(body) <= P.MAX_JPEG + 84 and frame_limit.allow(pid):
            f = parse_posed_frame(body)
            f["pose"] = world.correct_matrix(pid, f["pose"])
            if capturing and time.time() - p.frame_ts > 0.9:
                recorder.write("capture", {"t": "posed_jpeg", "id": pid, "b64": base64.b64encode(body).decode()})
            p.frame, p.frame_ts = f["jpeg"], time.time()
            world.confirmer.add_frame(pid, f)
    elif kind == P.BIN_POINTS:
        if points_limit.allow(pid):
            world.add_points(P.parse_points(body), pid)
    elif kind == P.BIN_DEPTH:
        if depth_limit.allow(pid):
            frame = P.parse_depth_frame(body)
            frame["pose"] = world.correct_matrix(pid, frame["pose"])
            world.fusion.queue.append(frame)
            if capturing:
                recorder.write("capture", {"t": "depth", "id": pid, "b64": base64.b64encode(body).decode()})


@phone_app.websocket("/ws/phone")
async def ws_phone(ws: WebSocket) -> None:
    dev = registry.authenticate(bearer(ws.headers))
    if dev is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    pid = dev.id
    dev.connected_once = True
    p = world.phone(pid, dev.name)
    p.connected, p.platform = True, dev.platform
    old = phone_sockets.get(pid)
    phone_sockets[pid] = ws
    if old is not None:
        with contextlib.suppress(Exception):
            await old.close(code=4000)
    await _send_json(ws, {"t": "welcome", "id": pid, "name": p.name, "color": p.color, "marker_m": MARKER_SIZES[marker["mode"]]})
    await _send_json(ws, world.planes_snapshot())
    await _send_json(ws, world.walls_message())
    if world.coarse:
        await _send_bytes(ws, world.all_points(coarse=True))
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if registry.devices.get(pid) is None:  # revoked while connected
                await ws.close(code=4403)
                break
            try:
                if msg.get("bytes") is not None:
                    handle_phone_bytes(pid, msg["bytes"])
                elif msg.get("text") is not None:
                    if len(msg["text"]) > P.MAX_TEXT or not text_limit.allow(pid):
                        continue
                    handle_phone_text(pid, msg["text"])
            except (ValidationError, ValueError) as e:
                log.info("dropped bad message from %s: %s", pid, str(e)[:120])
    except WebSocketDisconnect:
        pass
    finally:
        if phone_sockets.get(pid) is ws:
            # Session-only: the app closed (or lost its link), so this pairing is over.
            phone_sockets.pop(pid, None)
            registry.revoke(pid)
            world.forget_phone(pid, keep_map=True)
            capture_phones.discard(pid)
            for mws in [w for w, d in map_sockets.items() if d == pid]:  # their Map tab goes too
                map_sockets.pop(mws, None)
                with contextlib.suppress(Exception):
                    await mws.close(code=4403)
            log.warning("session ended for %s (%s); token discarded", dev.name, pid)
            world.event("left", f"{dev.name} left (session ended)", who=pid)


# ======================================================================================
# Commander listener (loopback only, admin token)
# ======================================================================================
admin_app = FastAPI(title="Sightline command", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
add_security_headers(admin_app)
admin_app.mount("/static", StaticFiles(directory=WEB), name="static")
ALLOWED_ORIGINS = {f"http://127.0.0.1:{ADMIN_PORT}", f"http://localhost:{ADMIN_PORT}"}


def is_admin(request_or_ws) -> bool:
    return registry.is_admin(request_or_ws.cookies.get(ADMIN_COOKIE))


def require_admin(request: Request, mutating: bool = False) -> None:
    if not is_admin(request):
        raise HTTPException(401, "admin login required")
    if mutating and request.headers.get("x-sightline") != "1":  # CSRF: custom header forces same-origin
        raise HTTPException(403, "missing CSRF header")


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FindMyTeam · Command</title>
<link rel="stylesheet" href="/static/style.css"></head><body class="login"><div class="card">
<div class="brand"><span class="logo">FMT</span><h1>FindMyTeam<small>Command</small></h1></div><p>Open the <b>admin link</b> printed in the server terminal to sign in.</p>
<p class="dim">On another computer: on the server laptop, press <b>+ Add a computer</b> and open the one-time link it shows.</p></div></body></html>"""


def is_local(request) -> bool:
    """The request comes from this Mac itself (the main commander), not another computer on the Wi-Fi."""
    return bool(request.client) and request.client.host in ("127.0.0.1", "::1")


@admin_app.get("/")
def index(request: Request, admin: str | None = None, screen: str | None = None):
    if screen is not None:  # another computer opening a one-time link made on the Mac
        ip = request.client.host if request.client else "?"
        token = registry.redeem_screen(screen, ip) if pair_limit.allow("screen:" + ip) else None
        if token is None:
            return HTMLResponse(LOGIN_PAGE, status_code=401)
        world.event("screen", f"Another computer opened the commander page ({ip})")
        resp = RedirectResponse("/", status_code=303)  # drop the code from the address bar/history
        resp.set_cookie(ADMIN_COOKIE, token, httponly=True, secure=True, samesite="strict", max_age=12 * 3600)
        return resp
    if admin is not None:
        if not registry.is_admin(admin):
            return HTMLResponse(LOGIN_PAGE, status_code=401)
        resp = RedirectResponse("/", status_code=303)  # drop the token from the address bar/history
        resp.set_cookie(ADMIN_COOKIE, admin, httponly=True, samesite="strict", max_age=12 * 3600)
        return resp
    if not is_admin(request):
        return HTMLResponse(LOGIN_PAGE, status_code=401)
    return FileResponse(WEB / "index.html")


@admin_app.get("/marker.pdf")
def marker_pdf() -> FileResponse:
    return FileResponse(MARKER / "marker.pdf")


@admin_app.get("/api/admin/state")
def admin_state(request: Request) -> dict:
    require_admin(request)
    return world.snapshot()


@admin_app.get("/api/admin/devices")
def admin_devices(request: Request) -> list:
    require_admin(request)
    return [{"id": d.id, "name": d.name, "platform": d.platform, "created": d.created, "connected": d.id in phone_sockets} for d in registry.devices.values()]


@admin_app.post("/api/admin/pair")
def admin_pair(request: Request) -> dict:
    require_admin(request, mutating=True)
    code = registry.new_code()
    uri = "sightline://pair?" + urlencode({"h": lan_ip(), "p": PHONE_PORT, "c": code, "f": CERT_FP}, quote_via=quote)
    svg = segno.make(uri, error="l", boost_error=False).svg_inline(scale=6, border=2, dark="#000", light="#fff")
    apk = None
    if APK.exists():
        url = f"https://{lan_ip()}:{PHONE_PORT}/download/sightline.apk?c={code}"
        apk = {"url": url, "sha256": hashlib.sha256(APK.read_bytes()).hexdigest(),
               "svg": segno.make(url, error="l", boost_error=False).svg_inline(scale=4, border=2, dark="#000", light="#fff")}
    return {"uri": uri, "svg": svg, "expires_in": 600, "host": f"{lan_ip()}:{PHONE_PORT}", "fingerprint": CERT_FP, "apk": apk}


@admin_app.delete("/api/admin/device/{pid}")
async def admin_revoke(pid: str, request: Request) -> dict:
    require_admin(request, mutating=True)
    revoked = registry.revoke(pid)
    ws = phone_sockets.pop(pid, None)
    if ws is not None:
        with contextlib.suppress(Exception):
            await ws.close(code=4403)
    world.forget_phone(pid)
    return {"ok": revoked}


@admin_app.post("/api/admin/ack/{pid}")
def admin_ack(pid: str, request: Request) -> dict:
    """Commander has seen the alert: it stays visible (muted) until the responder moves or says OK."""
    require_admin(request, mutating=True)
    p = world.phones.get(pid)
    if p and p.vitals.alert and not p.vitals.alert["acked"]:
        p.vitals.alert["acked"] = True
        world.event("ack", f"Commander acknowledged {p.name}'s alert", who=pid, alert=p.vitals.alert["kind"])
    return {"ok": True}


@admin_app.post("/api/admin/confirm/{pid}")
def admin_confirm(pid: str, request: Request) -> dict:
    """Commander confirms the alert was real (a positive label for the ML model)."""
    require_admin(request, mutating=True)
    p = world.phones.get(pid)
    if p and p.vitals.alert:
        p.vitals.alert["acked"] = True
        world.event("label", f"Commander confirmed {p.name}'s alert was real", who=pid, alert=p.vitals.alert["kind"], label="confirmed")
        note_label("confirmed")
    return {"ok": True}


@admin_app.post("/api/admin/budget")
def set_budget(request: Request, minutes: float = 20) -> dict:
    """Air / time-on-task budget per responder, used for turnaround predictions."""
    require_admin(request, mutating=True)
    world.planner.budget_s = max(1, min(240, minutes)) * 60
    world.event("budget", f"Time budget set to {minutes:.0f} min per responder")
    return {"budget_s": world.planner.budget_s}


@admin_app.post("/api/admin/capture/start")
def capture_start(request: Request, secs: int = 90) -> dict:
    """Room capture for tuning: every connected phone's depth frames, planes and ~1 camera frame/s."""
    require_admin(request, mutating=True)
    capture_phones.clear()
    capture_phones.update(phone_sockets)
    if not capture_phones:
        raise HTTPException(409, "no phones connected")
    name = recorder.start("capture", {"phones": sorted(capture_phones), "floor_y": world.floor_y()}, seconds=max(10, min(secs, 180)))
    world.event("capture", f"Room capture started ({max(10, min(secs, 180))} s)")
    return {"name": name}


@admin_app.post("/api/admin/record/{action}")
def record_toggle(action: str, request: Request) -> dict:
    """Opt-in session recording (compact, encrypted) for ML training and replay."""
    require_admin(request, mutating=True)
    if action == "start":
        name = recorder.start("session", {"started": world.started})
        world.event("record", "Session recording started")
        return {"name": name}
    if action == "stop":
        recorder.stop("session")
        recorder.stop("capture")
        world.event("record", "Recording stopped")
        return {"ok": True}
    raise HTTPException(404)


@admin_app.get("/api/admin/records")
def records_list(request: Request) -> list:
    require_admin(request)
    return recorder.list()


@admin_app.delete("/api/admin/records/{name}")
def records_delete(name: str, request: Request) -> dict:
    require_admin(request, mutating=True)
    return {"ok": recorder.delete(name)}


# ---------- waypoint route (commander only; responders see it, the assignee is guided) ----------
@admin_app.post("/api/admin/route")
async def route_update(request: Request) -> dict:
    require_admin(request, mutating=True)
    try:
        r = P.RouteUpdate.model_validate_json(await request.body())
    except ValidationError:
        raise HTTPException(400, "bad route")
    world.route_set(r.order, r.assignee, r.home)
    return world.route


@admin_app.post("/api/admin/route/best")
async def route_best(request: Request) -> dict:
    require_admin(request, mutating=True)
    order = await asyncio.to_thread(world.planner.best_order, world)
    if order is None:
        raise HTTPException(409, "Need at least 2 waypoints and a mapped area to plan an order.")
    world.route_set(order, world.route["assignee"], world.route["home"])
    world.event("route", "Waypoint order optimised (shortest walk)")
    return world.route


@admin_app.post("/api/admin/route/clear")
def route_clear(request: Request) -> dict:
    require_admin(request, mutating=True)
    for pid in [k for k, v in world.pins.items() if v["kind"] == "waypoint"]:
        world.pins.pop(pid, None)
    world.route_set([], None, True)
    return {"ok": True}


# ---------- Ask Gemini (commander's AI assistant) ----------
@admin_app.get("/api/admin/ask")
def ask_status(request: Request) -> dict:
    require_admin(request)
    return {"configured": gemini_configured(), "model": gemini.model, "can_setup": is_local(request)}


@admin_app.post("/api/admin/ask/key")
async def ask_key(request: Request) -> dict:
    """Save the Gemini API key on this laptop (owner-only file). Only from the Mac itself; never echoed back."""
    require_admin(request, mutating=True)
    require_local(request)
    try:
        key = re.sub(r"\s+", "", str((await request.json()).get("key", "")))  # stray spaces / line breaks from copying
    except ValueError:
        key = ""
    key = key.strip("\"'`")
    if not re.fullmatch(r"[\x21-\x7e]{20,200}", key):
        raise HTTPException(400, "That doesn't look like an API key (it should be one long string with no spaces).")
    path = DATA / "gemini.key"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key)
    path.chmod(0o600)
    ok, message = await gemini.check()
    world.event("ai", "Gemini assistant set up on this laptop" if ok else "Gemini key saved, but Google didn't accept it")
    return {"configured": True, "ok": ok, "message": message}


@admin_app.post("/api/admin/ask")
async def ask(request: Request) -> dict:
    require_admin(request, mutating=True)
    if not ask_limit.allow(request.client.host if request.client else "?"):
        raise HTTPException(429, "Too many questions at once. Wait a few seconds.")
    try:
        req = P.AskRequest.model_validate_json(await request.body())
    except ValidationError:
        raise HTTPException(400, "bad question")
    if not gemini_configured():
        raise HTTPException(503, "Gemini isn't set up on this laptop yet (no API key).")
    try:
        answer = await gemini.ask([m.model_dump() for m in req.messages], situation(world))
    except RuntimeError as e:
        raise HTTPException(502, str(e) or "Gemini didn't answer.")
    except httpx.HTTPError:
        raise HTTPException(502, "Couldn't reach Gemini. Check this laptop's internet connection.")
    return {"answer": answer, "model": gemini.model}


# ---------- voice guidance on phones (ElevenLabs) ----------
@admin_app.get("/api/admin/voice")
def voice_status(request: Request) -> dict:
    require_admin(request)
    return {"configured": V.configured(), "voice": V.voice_id(), "model": speech.model, "can_setup": is_local(request)}


@admin_app.post("/api/admin/voice/key")
async def voice_key(request: Request) -> dict:
    """Save the ElevenLabs API key (and optionally a voice ID) on this laptop. Only from the Mac itself; never echoed."""
    require_admin(request, mutating=True)
    require_local(request)
    try:
        body = await request.json()
    except ValueError:
        body = {}
    key = re.sub(r"\s+", "", str(body.get("key", ""))).strip("\"'`")
    voice = re.sub(r"\s+", "", str(body.get("voice", "")))
    if not re.fullmatch(r"[\x21-\x7e]{20,200}", key):
        raise HTTPException(400, "That doesn't look like an API key (it should be one long string with no spaces).")
    if voice and not re.fullmatch(r"[A-Za-z0-9]{10,40}", voice):
        raise HTTPException(400, "A voice ID is letters and digits only (copy it from the voice's page on elevenlabs.io).")
    for name, value in (("elevenlabs.key", key), ("elevenlabs.voice", voice or None)):
        path = DATA / name
        if value is None:
            path.unlink(missing_ok=True)
            continue
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(value)
        path.chmod(0o600)
    speech.clear()
    ok, message = await speech.check()
    world.event("ai", "Voice guidance (ElevenLabs) set up on this laptop" if ok else "ElevenLabs key saved, but ElevenLabs didn't accept it")
    return {"configured": True, "ok": ok, "message": message}


# ---------- other computers showing the commander page ----------
def require_local(request: Request) -> None:
    if not is_local(request):
        raise HTTPException(403, "only on the server laptop")


@admin_app.post("/api/admin/screens")
def screens_add(request: Request) -> dict:
    require_admin(request, mutating=True)
    require_local(request)
    url = f"https://{lan_ip()}:{SCREEN_PORT}/?screen={registry.new_screen_code()}"
    svg = segno.make(url, error="l", boost_error=False).svg_inline(scale=5, border=2, dark="#000", light="#fff")
    return {"url": url, "svg": svg, "expires_in": 600, "fingerprint": CERT_FP}


@admin_app.get("/api/admin/screens")
def screens_list(request: Request) -> list:
    require_admin(request)
    require_local(request)
    return [{"id": s["id"], "ip": s["ip"], "created": s["created"]} for s in registry.screens.values()]


@admin_app.delete("/api/admin/screens/{screen_id}")
async def screens_revoke(screen_id: str, request: Request) -> dict:
    require_admin(request, mutating=True)
    require_local(request)
    key = next((k for k, s in registry.screens.items() if s["id"] == screen_id), None)
    ok = registry.revoke_screen(screen_id)
    for ws in list(viewer_sockets):  # its open page goes dark immediately
        tok = ws.cookies.get(ADMIN_COOKIE)
        if key and tok and hash_token(tok) == key:
            viewer_sockets.discard(ws)
            with contextlib.suppress(Exception):
                await ws.close(code=4403)
    if ok:
        world.event("screen", "A commander screen was revoked")
    return {"ok": ok}


# ---------- entrance marker (printed sheet or on this screen) ----------
@admin_app.get("/api/admin/marker")
def marker_get(request: Request) -> dict:
    require_admin(request)
    return {"mode": marker["mode"], "width_m": MARKER_SIZES[marker["mode"]], "screen": SCREEN, "local": is_local(request)}


@admin_app.post("/api/admin/marker")
async def marker_set(request: Request, mode: str = "print") -> dict:
    require_admin(request, mutating=True)
    if mode not in MARKER_SIZES:
        raise HTTPException(400, "mode must be print or screen")
    if mode != marker["mode"]:
        marker["mode"] = mode
        world.event("marker", "Entrance marker: " + ("on the commander's screen (15 cm)" if mode == "screen" else "printed sheet (18 cm)"))
        for ws in list(phone_sockets.values()):  # phones re-lock with the new size
            await _send_json(ws, {"t": "marker", "marker_m": MARKER_SIZES[mode]})
    return {"mode": mode, "width_m": MARKER_SIZES[mode], "screen": SCREEN, "local": is_local(request)}


# ---------- saved maps (laptop only, encrypted) ----------
@admin_app.get("/api/admin/maps")
def maps_list(request: Request) -> list:
    require_admin(request)
    return maps.list()


@admin_app.post("/api/admin/maps")
def maps_save(request: Request, name: str = "") -> dict:
    require_admin(request, mutating=True)
    saved = maps.save(name, capture(world))
    world.event("map", f"Map saved: {saved['name']}")
    return saved


@admin_app.post("/api/admin/maps/{map_id}/load")
async def maps_load(map_id: int, request: Request) -> dict:
    require_admin(request, mutating=True)
    data = maps.load(map_id)
    if data is None:
        raise HTTPException(404, "map not found")
    restore(world, data)
    name = next((m["name"] for m in maps.list() if m["id"] == map_id), "map")
    world.event("map", f"Map loaded: {name}")
    snap = world.snapshot()
    snap["lan_ip"] = lan_ip()
    snap["phone_port"] = PHONE_PORT
    for ws in list(viewer_sockets) + list(map_sockets):  # everyone redraws from the loaded map
        await send_initial(ws, snap)
    for ws in list(phone_sockets.values()):
        await _send_json(ws, world.planes_snapshot())
    return {"ok": True}


@admin_app.get("/api/admin/maps/{map_id}/labels")
def maps_labels_get(map_id: int, request: Request) -> dict:
    require_admin(request)
    return maps.labels(map_id) or {"area": None, "rooms": [], "risk": []}


@admin_app.put("/api/admin/maps/{map_id}/labels")
async def maps_labels_put(map_id: int, request: Request) -> dict:
    require_admin(request, mutating=True)
    try:
        labels = P.Labels.model_validate_json(await request.body())
    except ValidationError:
        raise HTTPException(400, "bad labels")
    if not maps.set_labels(map_id, labels.model_dump()):
        raise HTTPException(404, "map not found")
    return {"ok": True}


@admin_app.delete("/api/admin/maps/{map_id}")
def maps_delete(map_id: int, request: Request) -> dict:
    require_admin(request, mutating=True)
    return {"ok": maps.delete(map_id)}


@admin_app.post("/api/admin/reset")
def admin_reset(request: Request) -> dict:
    require_admin(request, mutating=True)
    world.reset()
    for pid, d in registry.devices.items():  # keep live sessions listed
        world.phone(pid, d.name).connected = pid in phone_sockets
    return {"ok": True}


@admin_app.get("/api/admin/frame/{pid}.jpg")
def admin_frame(pid: str, request: Request) -> Response:
    require_admin(request)
    return _frame(pid)


@admin_app.get("/api/admin/frame/{pid}.mjpg")
async def admin_frame_stream(pid: str, request: Request) -> StreamingResponse:
    require_admin(request)

    async def gen():
        last = 0.0
        while True:
            p = world.phones.get(pid)
            if p and p.frame and p.frame_ts != last:
                last = p.frame_ts
                yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(p.frame)).encode() + b"\r\n\r\n" + p.frame + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


async def send_initial(ws: WebSocket, snap: dict) -> None:
    """Everything a new viewer (commander or a phone's Map tab) needs to draw the full map."""
    await _send_json(ws, snap)
    await _send_json(ws, world.walls_message())
    await _send_json(ws, {"t": "objects", "objects": world.objects})
    await _send_json(ws, world.coverage.message(full=True))
    if world.voxels:
        await _send_bytes(ws, world.all_points(coarse=False))
    for m in list(world.fine.meshes.values()) + list(world.fusion.meshes.values()):
        await _send_bytes(ws, m)
    await _send_json(ws, {"t": "textures", "ids": sorted(world.textures.ids())})
    for m in await asyncio.to_thread(world.textures.messages, True):
        await _send_bytes(ws, m)


@admin_app.websocket("/ws/view")
async def ws_view(ws: WebSocket) -> None:
    if not is_admin(ws) or ws.headers.get("origin") not in ALLOWED_ORIGINS:
        await ws.close(code=4401)
        return
    await ws.accept()
    snap = world.snapshot()
    snap["lan_ip"] = lan_ip()
    snap["phone_port"] = PHONE_PORT
    await send_initial(ws, snap)
    viewer_sockets.add(ws)
    try:
        while True:
            m = json.loads(await ws.receive_text())
            if m.get("t") == "pin":
                pin = P.AdminPin.model_validate(m)
                world.add_pin(pin.kind, pin.label, pin.pos, "Command")
            elif m.get("t") == "pin_remove":
                remove_pin(str(m.get("id")))
    except (WebSocketDisconnect, RuntimeError, ValueError, ValidationError):
        pass
    finally:
        viewer_sockets.discard(ws)
