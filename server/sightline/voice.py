"""Voice guidance for responders' phones: what to say (from the live twin), and ElevenLabs speech for it.

The laptop decides what each phone hears, because it already has every route (A* on the occupancy map),
hazard pin, fall-risk zone and door. iPhone and Android therefore say exactly the same things. Cues ride
along in each phone's state message; the phone plays the laptop's ElevenLabs audio for a cue, or speaks it
with its own built-in voice when ElevenLabs isn't set up, is slow, or the cue is private.

What it says (most urgent first):
  0  a colleague's fall / no-movement alert, a hazard pin ahead, "turn back now"
  1  trip hazards ahead: stairs, floor drops, low clutter in the walkway, places someone fell
  2  navigation along the route the phone asked for (to command, to the nearest exit) or the commander's
     waypoint route: start, upcoming turns, "turn left / right / around", distance left, arrival
  3  doors (and exit doors) a few metres ahead

Security / privacy:
  - The ElevenLabs key stays on the laptop: ELEVENLABS_API_KEY or server/data/elevenlabs.key (owner-only).
    It is never sent to a phone or a browser, logged, or put in a URL.
  - Only fixed phrases built here go to ElevenLabs (directions, distances, kinds of hazard). Cues that contain
    a person's name or a pin label typed by the team are marked "local": phones speak them on the device.
  - Phones fetch audio by cue id, and only for cues the laptop issued: nobody can make the key say anything else.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import httpx

from .security import DATA

log = logging.getLogger("sightline")
API = "https://api.elevenlabs.io/v1"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George", one of ElevenLabs' default voices: calm, clear
# Tried in order until one answers; Flash is the low-latency one. ElevenLabs retires model names now and then.
MODELS = [m for m in (os.environ.get("ELEVENLABS_MODEL"), "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2") if m]
FORMAT = "mp3_44100_64"

ZONE_SAY_M = 3.0  # fall-risk zone this close ahead (edge distance) -> warn
ZONE_REARM_M = 5.0  # walked this far away again -> the next approach warns afresh
DOOR_SAY_M = 5.0
DOOR_REARM_M = 7.0
ARRIVE_M = 1.2
TURN_DEG = 35  # off the route by more than this -> "turn left / right"
CORNER_DEG = 40  # a bend in the route sharper than this gets "in N metres, turn ..."
TURN_EVERY = 4.0  # seconds between repeated turn instructions
KEEP = 8  # cues kept in the state message (phones skip the ones they've already played)


# ---- geometry on the floor plane (x, z); angles: positive = to my left (right-handed frame, Y up) ----------
def _norm(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _bearing(a, b) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


def side(angle: float | None) -> str:
    """Where something is, relative to where I'm facing."""
    if angle is None:
        return "nearby"
    d = math.degrees(abs(angle))
    lr = "left" if angle > 0 else "right"
    if d < 20:
        return "straight ahead"
    if d < 60:
        return f"ahead on your {lr}"
    if d < 120:
        return f"on your {lr}"
    return "behind you"


def metres(d: float) -> str:
    n = max(1, round(d))
    return "1 metre" if n == 1 else f"{n} metres"


def _resample(path: list, step: float = 0.25) -> list[tuple[float, float]]:
    """Evenly spaced points along a polyline (A* grid paths zig-zag; this plus wide windows smooths that)."""
    out = [tuple(path[0])]
    carry = 0.0
    for a, b in zip(path, path[1:]):
        seg = math.dist(a, b)
        t = step - carry
        while t <= seg:
            out.append((a[0] + (b[0] - a[0]) * t / seg, a[1] + (b[1] - a[1]) * t / seg))
            t += step
        carry = (carry + seg) % step
    if math.dist(out[-1], path[-1]) > 1e-6:
        out.append(tuple(path[-1]))
    return out


def next_corner(path: list, start: float = 1.5, window: float = 1.0, step: float = 0.25) -> tuple[float, float, tuple] | None:
    """First real bend along the route after `start` metres: (metres along, turn angle (+ left), where)."""
    pts = _resample(path, step)
    w = max(1, int(window / step))
    for i in range(max(w, int(start / step)), len(pts) - w):
        turn = _norm(_bearing(pts[i], pts[i + w]) - _bearing(pts[i - w], pts[i]))
        if abs(math.degrees(turn)) >= CORNER_DEG:
            return i * step, turn, pts[i]
    return None


def zone_edge(zone: dict, x: float, z: float) -> tuple[float, tuple[float, float]]:
    """(distance from me to the zone's edge, 0 inside; its centre)."""
    if "r" in zone:
        return max(0.0, math.hypot(x - zone["x"], z - zone["z"]) - zone["r"]), (zone["x"], zone["z"])
    from .risk import _seg_dist, inside
    poly = zone["poly"]
    c = (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    if inside(zone, x, z):
        return 0.0, c
    return min(_seg_dist((x, z), poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))), c


ZONE_WORDS = {
    "stairs": ("Stairs", "Take the handrail."),
    "drop": ("Floor drop", "Watch your step."),
    "clutter": ("Trip hazard", "Low obstacle on the floor."),
    "fall": ("Careful", "Someone fell near here."),
}
DEST = {"home": "the command post", "exit": "the nearest exit"}


@dataclass
class VoiceGuide:
    """Per phone: what has been said, so nothing repeats until it's relevant again."""

    seq: int = 0
    cues: deque = field(default_factory=lambda: deque(maxlen=KEEP))
    said: dict = field(default_factory=dict)  # key -> time last said
    armed: set = field(default_factory=set)  # zones / doors / corners announced on this approach
    nav_key: str | None = None  # what we're guiding to (a change -> announce the new route)
    bucket: int | None = None  # last "N metres to go" milestone (tens of metres)
    view: dict | None = None  # for the phone's arrow card: {"to", "m", "aim"}
    fresh: list = field(default_factory=list)  # cues issued since the server last looked (to prefetch audio)

    def say(self, key: str, text: str, prio: int, now: float, kind: str, local: bool = False, every: float | None = None) -> bool:
        last = self.said.get(key)
        if last is not None and (every is None or now - last < every):
            return False
        self.said[key] = now
        self.seq += 1
        cue = {"n": self.seq, "text": text, "prio": prio, "kind": kind, "local": local, "ts": round(now, 2)}
        if not local:
            cue["id"] = cue_id(text)
        self.cues.append(cue)
        self.fresh.append(cue)
        return True

    def forget(self, prefix: str) -> None:
        for k in [k for k in self.said if k.startswith(prefix)]:
            self.said.pop(k)

    def reset_nav(self) -> None:
        self.nav_key, self.bucket, self.view = None, None, None
        self.forget("nav:")
        self.armed = {k for k in self.armed if not k.startswith("corner:")}

    # ------------------------------------------------------------------
    def update(self, p, world, now: float, walls: list, zones: list, exits: list) -> None:
        """Called about twice a second for each aligned, connected phone."""
        if not p.pos:
            return
        x, z = p.pos[0], p.pos[2]
        heading = None
        if p.fwd and math.hypot(p.fwd[0], p.fwd[2]) > 0.3:  # camera roughly level: it points where I'm facing
            heading = math.atan2(p.fwd[0], p.fwd[2])
        else:  # phone pointing at the floor: use the walking direction
            k = world.planner.kalman.get(p.id)
            if k is not None and k.ready and k.speed > 0.3:
                heading = math.atan2(k.x[2], k.x[3])

        def rel(tx, tz):
            return math.hypot(tx - x, tz - z), (None if heading is None else _norm(math.atan2(tx - x, tz - z) - heading))

        info = (world.plan or {}).get("people", {}).get(p.id, {})

        # 0: colleagues down (names: spoken on the phone itself)
        from .geometry import walls_between
        for q in list(world.phones.values()):
            if q.id == p.id:
                continue
            a = q.vitals.public(now)
            key = f"alert:{q.id}"
            if not a or a["acked"] or not q.connected:
                self.forget(key)
                continue
            where = ""
            if q.pos:
                d, ang = rel(q.pos[0], q.pos[2])
                w = walls_between((x, z), (q.pos[0], q.pos[2]), (p.pos[1] + q.pos[1]) / 2, world.walls())
                where = f", {metres(d)} {side(ang)}" + (f", through {w} wall{'s' if w > 1 else ''}" if w else "")
            what = "may have fallen" if a["kind"] == "fall" else "has stopped moving"
            self.say(f"{key}:{a['kind']}", f"Alert. {q.name} {what}{where}.", 0, now, "alert", local=True, every=60)

        # 0: a hazard pin ahead (the label was typed by the team: local voice)
        hw = p.hazard.warn
        if hw:
            label = hw["label"] if hw["label"] and hw["label"] != "hazard" else "a marked hazard"
            self.say(f"hazard:{hw['id']}", f"Danger. {label}, {metres(hw['dist'])} ahead. Keep away.", 0, now, "alert",
                     local=True, every=45)

        # 0: time budget
        t = info.get("turnaround_s")
        if t is not None:
            if t <= 0:
                self.say("turn:now", "Turn back now. Head for the exit.", 0, now, "alert", every=90)
            elif t <= 120:
                self.say("turn:soon", "Two minutes until you must turn back.", 1, now, "alert")

        # 1: fall-risk zones ahead
        for zn in zones:
            edge, c = zone_edge(zn, x, z)
            key = f"zone:{zn['kind']}:{round(c[0] * 2)}:{round(c[1] * 2)}"
            if edge > ZONE_REARM_M:
                self.armed.discard(key)
                continue
            if key in self.armed or edge > ZONE_SAY_M:
                continue
            _, ang = rel(*c)
            if edge > 0.3 and ang is not None and abs(ang) > math.radians(70):
                continue  # beside or behind me: not in my way
            head, tail = ZONE_WORDS.get(zn["kind"], ("Careful", "Watch your step."))
            if zn["kind"] == "fall":
                text = f"Careful. Someone fell near here, {side(ang)}."
            elif edge <= 0.3:
                text = f"{head} right here. {tail}"
            else:
                text = f"{head} {side(ang)}, in {metres(edge)}. {tail}"
            self.armed.add(key)
            self.say(key, text, 1, now, "hazard", every=20)

        # 2: navigation
        self._navigate(p, world, info, now, x, z, heading, rel)

        # 3: doors ahead (exit doors called out as such)
        best = None
        for (ax, az), (bx, bz), _, _, kind in walls:
            if kind != "door":
                continue
            mx, mz = (ax + bx) / 2, (az + bz) / 2
            d, ang = rel(mx, mz)
            key = f"door:{round(mx * 2)}:{round(mz * 2)}"
            if d > DOOR_REARM_M:
                self.armed.discard(key)
            elif d <= DOOR_SAY_M and key not in self.armed and (ang is None or abs(ang) < math.radians(75)):
                if best is None or d < best[0]:
                    best = (d, ang, key, any(math.hypot(ex - mx, ez - mz) < 0.8 for ex, ez in exits))
        if best and now - self.said.get("door", 0) > 6:
            d, ang, key, is_exit = best
            self.armed.add(key)
            self.said["door"] = now
            self.say(key, f"{'Exit door' if is_exit else 'Door'} {side(ang)}, {metres(d)}.", 3, now, "door", every=30)

    def _navigate(self, p, world, info, now, x, z, heading, rel) -> None:
        guide = info.get("guide")
        route = (world.plan or {}).get("route") or {}
        path, dest, target = None, None, None
        if guide and not guide.get("done") and route.get("assignee") == p.id and route.get("legs"):
            leg = route["legs"][0]
            path, dest, target = (None if leg.get("blocked") else leg["path"]), f"waypoint {guide.get('next') or ''}".strip(), f"wp:{guide.get('next')}"
        elif guide and guide.get("done") and route.get("assignee") == p.id:
            self.say("nav:route-done", "Route complete. All waypoints reached.", 2, now, "nav")
            self.view = None
            return
        elif p.nav == "home":
            home = info.get("home") or {}
            path, dest, target = home.get("path"), DEST["home"], "home"
        elif p.nav == "exit":
            path, dest, target = info.get("exit_path"), DEST["exit"], "exit"
        if target is None:
            if self.nav_key is not None:
                self.reset_nav()
            return
        if target != self.nav_key:
            self.reset_nav()
            self.nav_key = target
        if not path or len(path) < 2:
            self.view = {"to": target, "m": None, "aim": None}
            self.say(f"nav:none:{target}", f"No known route to {dest} yet. Keep mapping.", 2, now, "nav", every=30)
            return
        near = min(range(len(path)), key=lambda i: math.dist(path[i], (x, z)))  # the plan is up to a second old
        path = path[near:]
        remaining = math.dist((x, z), path[0]) + sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        aim = next((pt for pt in path if math.hypot(pt[0] - x, pt[1] - z) >= 1.5), path[-1])
        self.view = {"to": target, "m": round(remaining, 1), "aim": [round(aim[0], 2), round(aim[1], 2)]}

        if remaining <= ARRIVE_M:
            if self.say(f"nav:arrived:{target}", f"You have arrived at {dest}.", 2, now, "nav"):
                self.forget("nav:start:")
            return
        if remaining > ARRIVE_M + 3:
            self.forget(f"nav:arrived:{target}")
        zone = (info.get("home") or {}).get("zone", 0) if target == "home" else 0
        care = " The route crosses a fall-risk area: take care." if zone == 1 else " The route passes a hazard." if zone == 2 else ""
        if self.say(f"nav:start:{target}", f"Guiding you to {dest}. {metres(remaining)}.{care}", 2, now, "nav"):
            self.bucket = int(remaining // 10)
            return

        # off the route: turn towards it
        if heading is not None:
            off = _norm(_bearing((x, z), aim) - heading)
            deg = math.degrees(abs(off))
            if deg > 135:
                self.say("nav:turn", "Turn around.", 2, now, "nav", every=TURN_EVERY)
                return
            if deg > TURN_DEG:
                self.say("nav:turn", f"Turn {'left' if off > 0 else 'right'}.", 2, now, "nav", every=TURN_EVERY)
                return
        # a bend coming up
        c = next_corner([[x, z]] + [list(pt) for pt in path])
        if c and 1.5 <= c[0] <= 6:
            key = f"corner:{round(c[2][0])}:{round(c[2][1])}"
            if key not in self.armed:
                self.armed.add(key)
                self.say(f"nav:{key}", f"In {metres(c[0])}, turn {'left' if c[1] > 0 else 'right'}.", 2, now, "nav")
                return
        # progress, every 10 metres
        b = int(remaining // 10)
        if self.bucket is not None and b < self.bucket and remaining > 5:
            self.say(f"nav:togo:{b}", f"{metres(remaining)} to {dest}.", 2, now, "nav")
        self.bucket = b if self.bucket is None else min(self.bucket, b)


# ---- ElevenLabs speech --------------------------------------------------------------------------------
def cue_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _key() -> str | None:
    k = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if k:
        return k
    try:
        return (DATA / "elevenlabs.key").read_text().strip() or None
    except OSError:
        return None


def configured() -> bool:
    return _key() is not None


def voice_id() -> str:
    v = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if v:
        return v
    try:
        return (DATA / "elevenlabs.voice").read_text().strip() or VOICE_ID
    except OSError:
        return VOICE_ID


class Speech:
    """Issued cue texts (id -> text) and their audio, cached in memory (the same phrases come up again and again)."""

    def __init__(self, max_audio: int = 400) -> None:
        self.texts: OrderedDict[str, str] = OrderedDict()
        self.audio: OrderedDict[str, bytes] = OrderedDict()
        self.pending: dict[str, asyncio.Future] = {}
        self.max_audio = max_audio
        self.model = MODELS[0]
        self.failed_until = 0.0  # after an error, phones use their own voice for a while instead of waiting

    def issue(self, text: str) -> str:
        cid = cue_id(text)
        self.texts[cid] = text
        self.texts.move_to_end(cid)
        while len(self.texts) > 2000:
            self.texts.popitem(last=False)
        return cid

    def clear(self) -> None:
        self.audio.clear()
        self.failed_until = 0.0

    async def get(self, cid: str) -> bytes | None:
        """mp3 for an issued cue; None if unknown, not set up, or ElevenLabs is failing."""
        if cid in self.audio:
            self.audio.move_to_end(cid)
            return self.audio[cid]
        text = self.texts.get(cid)
        if text is None or not configured() or time.time() < self.failed_until:
            return None
        if cid in self.pending:
            return await asyncio.shield(self.pending[cid])
        fut = asyncio.get_running_loop().create_future()
        self.pending[cid] = fut
        try:
            data = await self._synth(text)
        except Exception as e:  # never let speech take anything else down
            log.warning("elevenlabs: %s", str(e)[:200])
            self.failed_until = time.time() + 30
            data = None
        finally:
            self.pending.pop(cid, None)
        if data:
            self.audio[cid] = data
            while len(self.audio) > self.max_audio:
                self.audio.popitem(last=False)
        fut.set_result(data)
        return data

    async def _synth(self, text: str) -> bytes:
        key = _key()
        if key is None:
            raise RuntimeError("No key saved.")
        last = ""
        async with httpx.AsyncClient(timeout=15) as client:
            for model in [self.model] + [m for m in MODELS if m != self.model]:
                r = await client.post(f"{API}/text-to-speech/{voice_id()}", params={"output_format": FORMAT},
                                      headers={"xi-api-key": key, "accept": "audio/mpeg"},
                                      json={"text": text, "model_id": model,
                                            "voice_settings": {"stability": 0.6, "similarity_boost": 0.75}})
                if r.status_code == 200 and r.content:
                    self.model = model
                    return r.content
                detail = ""
                if r.headers.get("content-type", "").startswith("application/json"):
                    d = r.json().get("detail")
                    detail = d.get("message", "") if isinstance(d, dict) else str(d or "")
                if r.status_code == 401:
                    raise RuntimeError("ElevenLabs rejected this key. Copy it again from elevenlabs.io (Developers, API keys) "
                                       "and make sure it's allowed to use Text to Speech. " + detail[:160])
                if r.status_code == 404 and "voice" in detail.lower():
                    raise RuntimeError(f"ElevenLabs doesn't know voice {voice_id()!r}. Leave the voice box empty for the default. " + detail[:120])
                last = f"{r.status_code} {detail[:160]}"
                if r.status_code not in (400, 404, 422):  # quota / busy: another model won't help
                    break
        raise RuntimeError(f"ElevenLabs didn't answer ({last}). The key's monthly credits may be used up.")

    async def check(self) -> tuple[bool, str]:
        try:
            await self._synth("Voice guidance ready.")
            return True, f"Key works. Using {self.model}, voice {voice_id()}."
        except RuntimeError as e:
            return False, str(e)
        except httpx.HTTPError:
            return False, "Couldn't reach ElevenLabs. Check this laptop's internet connection."
