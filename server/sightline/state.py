"""In-memory world state. Everything here is already in the shared (marker) frame."""

from __future__ import annotations

import itertools
import math
import statistics
import struct
import time
from collections import deque
from dataclasses import dataclass, field

from .geometry import merge_walls, regularize_walls, stabilize_walls, straighten_walls, wall_segment, walls_between
from .classify import ObjectConfirmer
from .align import apply as align_apply, compose as align_compose, damped as align_damped, estimate as align_estimate, matrix as align_matrix
from .risk import fall_risk_zones
from .scene import SceneLabeler
from .texture import Textures
from .fusion import FINE_BLOCKS, FINE_RANGE, FINE_VOXEL, Fusion
from .motion import Tracker, set_down
from .predict import Planner
from .twin import Coverage, Vitals, HazardWatch
from .voice import VoiceGuide

STALE_AFTER = 3.0  # seconds without a pose before a phone is drawn as stale
TRAIL_STEP = 0.25  # metres between breadcrumbs
TRAIL_MAX = 800
OCCLUDING = {"wall", "vertical"}  # plane kinds that block line of sight
VOXEL = 0.05  # metres; fine point layer for the commander's 3D view
COARSE = 0.15  # metres; light layer sent to phones for their minimap
MAX_VOXELS = 600_000
MAX_COARSE = 120_000
VOXEL_REC = struct.Struct("<hhhBBB")  # voxel index x,y,z + colour, 9 bytes on the wire
CONFIRM = 2  # a voxel is published only once it has been observed this many times (kills speckle)
MAX_PENDING = 400_000
PALETTE = ["#ff5a5f", "#3fa7ff", "#ffb400", "#2ecc71", "#c77dff", "#ff8c42", "#00c2c7", "#f15bb5"]


@dataclass
class Phone:
    id: str
    name: str
    color: str
    connected: bool = True
    aligned: bool = False
    tracking: str = "unknown"
    battery: float | None = None
    pos: list[float] | None = None
    fwd: list[float] | None = None
    pose_ts: float = 0.0
    trail: list[list[float]] = field(default_factory=list)
    xray_target: str | None = None
    wall_dist: float | None = None
    platform: str = "ios"
    frame: bytes | None = None
    frame_ts: float = 0.0
    last_seen: float = field(default_factory=time.time)
    vitals: Vitals = field(default_factory=Vitals)
    hazard: HazardWatch = field(default_factory=HazardWatch)  # walking towards hazard pins
    voice: VoiceGuide = field(default_factory=VoiceGuide)  # spoken cues for this phone (voice.py)
    nav: str = "off"  # voice route the responder asked for: "off" | "home" | "exit"
    last_cov: float = 0.0
    activity: str | None = None  # Apple Core Motion's pre-trained activity classifier, from the phone
    act_conf: int = 0  # 0 low, 1 medium, 2 high
    motion: Tracker = field(default_factory=Tracker)
    phone_down: bool = False  # no hand tremor at all: phone lying somewhere
    health: dict = field(default_factory=dict)  # memory / heat / rates reported by the phone
    corr: tuple = (0.0, 0.0, 0.0)  # fine alignment to the reference phone: (yaw rad, dx m, dz m), see align.py
    set_down_noted: bool = False

    def public(self, now: float) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "connected": self.connected,
            "aligned": self.aligned,
            "tracking": self.tracking,
            "battery": self.battery,
            "pos": self.pos,
            "fwd": self.fwd,
            "age": round(now - self.pose_ts, 2) if self.pose_ts else None,
            "stale": (now - self.pose_ts) > STALE_AFTER if self.pose_ts else True,
            "xray": self.xray_target,
            "has_frame": self.frame is not None and now - self.frame_ts < 10,
            "wall_dist": self.wall_dist,
            "platform": self.platform,
            "alert": self.vitals.public(now),
            "hazard": self.hazard.public(),  # approaching a hazard despite the warning (commander sees it)
            "health": {k: v for k, v in self.health.items() if not k.startswith("_")} or None,
            "activity": {"state": self.activity, "conf": ["low", "medium", "high"][self.act_conf], "phone_down": self.phone_down}
            if self.activity else None,
        }


class World:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phones: dict[str, Phone] = {}
        self.planes: dict[str, dict] = {}
        self.raw_planes: dict[str, list] = {}  # plane corners as each phone sent them (before fine alignment)
        self.pins: dict[str, dict] = {}
        self.dirty_planes: set[str] = set()
        self.removed_planes: set[str] = set()
        self._pin_ids = itertools.count(1)
        self._walls_cache: list | None = None
        self.voxels: dict[tuple[int, int, int], tuple[int, int, int]] = {}  # confirmed, published
        self.coarse: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self.pending: dict[tuple[int, int, int], list[int]] = {}  # r,g,b sums + count, not yet confirmed
        self.pending_coarse: dict[tuple[int, int, int], list[int]] = {}
        self._merged: list | None = None
        self._prev_merged: list = []
        self.walls_dirty = True
        self.new_voxels: list[tuple[int, int, int, int, int, int]] = []
        self.new_coarse: list[tuple[int, int, int, int, int, int]] = []
        self.cleared = False  # tells clients to drop their point layers
        self.coverage = Coverage()
        self.alerts_changed = False
        self.fusion = Fusion()
        self.confirmer = ObjectConfirmer()  # laptop-side image classifier: a second opinion on fused objects
        self.scene = SceneLabeler(self.confirmer)  # laptop-side scene segmentation: windows, blinds, pillars, doors
        self.confirmer.scene = self.scene  # ... and the furniture double-check (better than ImageNet MobileNetV2)
        self.textures = Textures(self.confirmer, self.fusion)  # photo-textured walls + floor from the snapshots
        self.fusion.drop_planar = True  # the textured walls / floor replace the mesh triangles lying on them
        # 2 cm detail layer for anything a phone saw from close up (furniture, doorways); the 4 cm layer
        # leaves out its own triangles wherever the fine layer has surface
        self.fine = Fusion(voxel=FINE_VOXEL, trunc=0.06, max_depth=FINE_RANGE, max_blocks=FINE_BLOCKS, mesh_type=0x07)
        self.fine.drop_planar = True
        self.fusion.covered = self.fine.covers
        self.depth_walls: list = []  # walls inferred from fused depth (plain walls ARKit misses)
        self.objects: list[dict] = []  # furniture boxes inferred from fused depth (seat / table / obstacle height)
        self.stairs: list[dict] = []  # flights of stairs: {"corners": 4 x (x, z), "up": (dx, dz), "rise": m, "down": bool}
        self.route: dict = {"order": [], "assignee": None, "home": True, "reached": []}  # the commander's waypoint route
        self.drops: list = []  # (x, z) floor edges where the surface falls away
        self.fall_spots: list = []  # (x, z) where a fall alert went off this session
        self.visited: set = set()  # 40 cm cells someone has stood in (rooms "entered" vs only looked into)
        self._stairs_known: set = set()
        self.started = time.time()
        self.events: deque = deque(maxlen=500)  # session timeline (memory only)
        self.new_events: list[dict] = []
        self.planner = Planner()
        self.plan: dict = {"people": {}, "search": {}, "decisions": []}

    def event(self, kind: str, text: str, **data) -> dict:
        ev = {"ts": round(time.time(), 2), "kind": kind, "text": text, **data}
        self.events.append(ev)
        self.new_events.append(ev)
        return ev

    def floor_y(self) -> float:
        """Floor height in the shared frame: ARKit floor planes, else standing eye height minus 1.45 m."""
        ys = [p["corners"][0][1] for p in self.planes.values() if p["kind"] == "floor"]
        if ys:
            return statistics.median(ys)
        eyes = [p.pos[1] for p in self.phones.values() if p.pos]
        return (statistics.median(eyes) - 1.45) if eyes else -1.4

    # ---- phones -------------------------------------------------------
    def phone(self, pid: str, name: str | None = None) -> Phone:
        p = self.phones.get(pid)
        if p is None:
            color = PALETTE[len(self.phones) % len(PALETTE)]
            p = self.phones[pid] = Phone(id=pid, name=name or pid, color=color)
        elif name:
            p.name = name
        return p

    def set_pose(self, p: Phone, pos: list[float], fwd: list[float] | None) -> None:
        now = time.time()
        p.pos, p.fwd, p.pose_ts, p.last_seen = [float(v) for v in pos[:3]], fwd, now, now
        p.aligned = True
        p.motion.add(now, p.pos, fwd)
        self.planner.observe_pose(p.id, now, p.pos, walking=p.activity in ("walking", "running"))
        if p.vitals.update(p.pos, now):
            self.alerts_changed = True
            a = p.vitals.alert
            if a:
                self.event("alert", f"{p.name}: {'possible fall' if a['kind'] == 'fall' else 'not moving'}", who=p.id, alert=a["kind"])
                if a["kind"] == "fall":  # remembered as a fall-risk spot for everyone else (yellow zone)
                    self.fall_spots.append((round(p.pos[0], 2), round(p.pos[2], 2)))
                    self.walls_dirty = True
            else:
                self.event("alert_clear", f"{p.name} is moving again", who=p.id)
        if now - p.last_cov >= 0.2:  # coverage at 5 Hz is plenty
            p.last_cov = now
            self.coverage.observe(p.pos, fwd, self.walls())
        self.visited.add((math.floor(p.pos[0] / 0.4), math.floor(p.pos[2] / 0.4)))  # where someone has actually stood
        if not p.trail or math.dist(p.trail[-1], p.pos) >= TRAIL_STEP:
            p.trail.append([round(p.pos[0], 3), round(p.pos[1], 3), round(p.pos[2], 3)])
            del p.trail[:-TRAIL_MAX]

    def forget_phone(self, pid: str, keep_map: bool = False) -> None:
        """Drop a phone (marker, trail, frame). keep_map leaves the walls it mapped on the shared map."""
        self.phones.pop(pid, None)
        self.planner.forget(pid)
        if not keep_map:
            for k in [k for k, v in self.planes.items() if v["owner"] == pid]:
                self.remove_plane(k)

    # ---- planes -------------------------------------------------------
    def upsert_plane(self, owner: str, pid: str, kind: str, corners: list[list[float]]) -> None:
        key = f"{owner}:{pid}"
        raw = [[float(v) for v in c[:3]] for c in corners]
        self.raw_planes[key] = raw  # as the phone sent it; re-corrected if that phone's alignment improves
        corr = self.phones[owner].corr if owner in self.phones else (0.0, 0.0, 0.0)
        self.planes[key] = {"id": key, "owner": owner, "kind": kind, "corners": self._correct_corners(raw, corr)}
        self.dirty_planes.add(key)
        self.removed_planes.discard(key)
        self._walls_cache = None
        self._merged = None
        self.walls_dirty = True

    @staticmethod
    def _correct_corners(raw: list, corr: tuple) -> list:
        out = []
        for x, y, z in raw:
            cx, cz = align_apply(corr, x, z) if corr != (0.0, 0.0, 0.0) else (x, z)
            out.append([round(cx, 3), round(y, 3), round(cz, 3)])
        return out

    def refine_alignment(self) -> list[str]:
        """Nudge each phone's alignment towards the reference phone's walls. Returns timeline texts."""
        segs: dict[str, list] = {}
        for key, p in self.planes.items():
            if p["kind"] in ("wall", "vertical"):
                w = wall_segment(p["corners"])
                if w and math.dist(w[0], w[1]) >= 0.8:
                    segs.setdefault(p["owner"], []).append((w[0], w[1]))
        owners = [pid for pid in self.phones if len(segs.get(pid, [])) >= 3]
        if len(owners) < 2:
            return []
        ref, texts = owners[0], []
        for pid in owners[1:]:
            inc = align_estimate(segs[pid], segs[ref])
            if inc is None or (abs(inc[0]) < math.radians(0.15) and math.hypot(inc[1], inc[2]) < 0.02):
                continue
            step = align_damped(inc)
            ph = self.phones[pid]
            ph.corr = align_compose(step, ph.corr)
            for key, raw in self.raw_planes.items():
                if key.startswith(pid + ":") and key in self.planes:
                    self.planes[key]["corners"] = self._correct_corners(raw, ph.corr)
                    self.dirty_planes.add(key)
            self._walls_cache = None
            self._merged = None
            self.walls_dirty = True
            texts.append(f"Lined up {ph.name}'s map with {self.phones[ref].name}'s "
                         f"({math.degrees(ph.corr[0]):+.1f}°, {math.hypot(ph.corr[1], ph.corr[2]) * 100:.0f} cm)")
        return texts

    def correct_pose(self, pid: str, pos: list[float], fwd: list[float] | None):
        corr = self.phones[pid].corr if pid in self.phones else (0.0, 0.0, 0.0)
        if corr == (0.0, 0.0, 0.0):
            return pos, fwd
        x, z = align_apply(corr, pos[0], pos[2])
        if fwd is not None:
            fx, fz = align_apply((corr[0], 0.0, 0.0), fwd[0], fwd[2])
            fwd = [fx, fwd[1], fz]
        return [x, pos[1], z], fwd

    def correct_matrix(self, pid: str, pose):
        corr = self.phones[pid].corr if pid in self.phones else (0.0, 0.0, 0.0)
        return pose if corr == (0.0, 0.0, 0.0) else align_matrix(corr) @ pose

    def remove_plane(self, key: str) -> None:
        self.raw_planes.pop(key, None)
        if self.planes.pop(key, None) is not None:
            self.removed_planes.add(key)
            self.dirty_planes.discard(key)
            self._walls_cache = None
        self._merged = None
        self.walls_dirty = True

    def take_plane_delta(self) -> dict | None:
        if not self.dirty_planes and not self.removed_planes:
            return None
        msg = {"t": "planes", "upsert": [self.planes[k] for k in self.dirty_planes if k in self.planes], "remove": sorted(self.removed_planes)}
        self.dirty_planes.clear()
        self.removed_planes.clear()
        return msg

    def merged_walls(self) -> list:
        """Clean walls/doors/windows merged across all phones' plane pieces."""
        if self._merged is None:
            segs = []
            for p in self.planes.values():
                if p["kind"] in OCCLUDING or p["kind"] in ("door", "window"):
                    w = wall_segment(p["corners"])
                    # unclassified vertical planes are often furniture sides: only big, tall ones count
                    if w is not None and (p["kind"] != "vertical" or (math.dist(w[0], w[1]) >= 0.8 and w[3] - w[2] >= 1.0)):
                        segs.append((*w, p["kind"]))
            segs.extend(self.depth_walls)
            # pillars travel as their own kind so the clean-up keeps their short sides; on the map they are walls
            merged = [(p, q, y0, y1, "wall" if k == "pillar" else k)
                      for p, q, y0, y1, k in regularize_walls(merge_walls(segs, offset_tol=0.32))]  # clean floor plan (geometry.py)
            self._merged = stabilize_walls(merged, getattr(self, "_prev_merged", []))
            self._prev_merged = self._merged
        return self._merged

    def walls(self) -> list:
        """Occluding walls (merged) for line-of-sight counting."""
        if self._walls_cache is None:
            self._walls_cache = [(p, q, y0, y1) for p, q, y0, y1, k in self.merged_walls() if k == "wall"]
        return self._walls_cache

    def set_semantics(self, walls: list, objects: list[dict]) -> bool:
        """New walls/furniture from fusion; returns True if the furniture list changed."""
        for o in objects:  # confirmed by YOLO outlines (fusion) or by two agreeing classifier views
            o["confirmed"] = bool(o.get("confirmed")) or self.confirmer.is_confirmed(o)
        if walls != self.depth_walls:
            self.depth_walls = walls
            self._walls_cache = None
            self._merged = None
            self.walls_dirty = True
        if self.fusion.drops != self.drops:
            self.drops = list(self.fusion.drops)
            self.walls_dirty = True
        stairs = list(self.fusion.stairs)
        if stairs != self.stairs:
            for st in stairs:
                cx = sum(c[0] for c in st["corners"]) / 4
                cz = sum(c[1] for c in st["corners"]) / 4
                key = (round(cx / 0.5), round(cz / 0.5))
                if key not in self._stairs_known:
                    self._stairs_known.add(key)
                    self.event("stairs", f"Stairs found at ({cx:.1f}, {cz:.1f}) m, going {'down' if st['down'] else 'up'} about {st['rise']:.1f} m")
            self.stairs = stairs
            self.walls_dirty = True
        changed = objects != self.objects
        self.objects = objects
        if changed:
            self.walls_dirty = True  # clutter zones depend on the objects
        return changed

    def risk_zones(self) -> list[dict]:
        """Yellow fall-risk zones (stairs, drops, clutter in walkways, places someone fell)."""
        return fall_risk_zones(list(self.stairs), list(self.drops), list(self.objects), self.merged_walls(), list(self.fall_spots))

    def walls_message(self) -> dict:
        self.coverage.set_outline(self.merged_walls())
        return {"t": "walls", "walls": [[round(p[0], 3), round(p[1], 3), round(q[0], 3), round(q[1], 3), round(y0, 2), round(y1, 2), k]
                                        for p, q, y0, y1, k in self.merged_walls()],
                "stairs": [{"c": [[round(x, 2), round(z, 2)] for x, z in st["corners"]], "up": [round(v, 3) for v in st["up"]],
                            "rise": st["rise"], "down": st["down"]} for st in self.stairs],
                "risk": self.risk_zones()}

    # ---- 3D points --------------------------------------------------
    def add_points(self, pts: list[tuple[float, float, float, int, int, int]], pid: str | None = None) -> int:
        """Merge colored points into the voxel grids; returns how many fine voxels got confirmed."""
        corr = self.phones[pid].corr if pid in self.phones else (0.0, 0.0, 0.0)
        if corr != (0.0, 0.0, 0.0):
            pts = [(*((lambda c: (c[0], y, c[1]))(align_apply(corr, x, z))), r, g, b) for x, y, z, r, g, b in pts]
        added = 0
        for x, y, z, r, g, b in pts:
            k = (round(x / VOXEL), round(y / VOXEL), round(z / VOXEL))
            if abs(k[0]) > 32000 or abs(k[1]) > 32000 or abs(k[2]) > 32000:
                continue
            if self._observe(k, r, g, b, self.voxels, self.pending, self.new_voxels, MAX_VOXELS):
                added += 1
            c = (round(x / COARSE), round(y / COARSE), round(z / COARSE))
            self._observe(c, r, g, b, self.coarse, self.pending_coarse, self.new_coarse, MAX_COARSE)
        return added

    @staticmethod
    def _observe(k, r, g, b, confirmed, pending, new, cap) -> bool:
        if k in confirmed:
            return False
        acc = pending.get(k)
        if acc is None:
            if len(pending) < MAX_PENDING:
                pending[k] = [r, g, b, 1]
            return False
        acc[0] += r
        acc[1] += g
        acc[2] += b
        acc[3] += 1
        if acc[3] < CONFIRM or len(confirmed) >= cap:
            return False
        n = acc[3]
        col = (acc[0] // n, acc[1] // n, acc[2] // n)
        del pending[k]
        confirmed[k] = col
        new.append((*k, *col))
        return True

    @staticmethod
    def pack_voxels(layer: int, size: float, recs) -> bytes:
        """0x02 | layer u8 | voxel size f32 | N x (int16 x,y,z, uint8 r,g,b)"""
        head = struct.pack("<BBf", 0x02, layer, size)
        return head + b"".join(VOXEL_REC.pack(*r) for r in recs)

    def take_point_deltas(self) -> tuple[bytes | None, bytes | None]:
        fine = self.pack_voxels(0, VOXEL, self.new_voxels) if self.new_voxels else None
        coarse = self.pack_voxels(1, COARSE, self.new_coarse) if self.new_coarse else None
        self.new_voxels, self.new_coarse = [], []
        return fine, coarse

    def all_points(self, coarse: bool) -> bytes:
        layer = self.coarse if coarse else self.voxels
        return self.pack_voxels(int(coarse), COARSE if coarse else VOXEL, [(*k, *v) for k, v in layer.items()])

    # ---- pins ---------------------------------------------------------
    def add_pin(self, kind: str, label: str, pos: list[float], by: str) -> dict:
        pin = {"id": f"pin{next(self._pin_ids)}", "kind": kind, "label": label[:40], "pos": [float(v) for v in pos[:3]], "by": by, "ts": time.time()}
        self.pins[pin["id"]] = pin
        if kind == "waypoint":  # new waypoints go on the end of the route
            self.route["order"].append(pin["id"])
            self._number_waypoints()
        return pin

    # ---- waypoint route -----------------------------------------------
    def route_set(self, order: list[str], assignee: str | None, home: bool) -> None:
        wp = [k for k in order if self.pins.get(k, {}).get("kind") == "waypoint"]
        wp += [k for k, v in self.pins.items() if v["kind"] == "waypoint" and k not in wp]  # never lose a waypoint
        changed_who = assignee != self.route["assignee"]
        self.route = {"order": wp, "assignee": assignee if assignee in self.phones else None, "home": bool(home),
                      "reached": [] if changed_who else [k for k in self.route["reached"] if k in wp]}
        self._number_waypoints()
        if changed_who and self.route["assignee"]:
            self.event("route", f"Waypoint route assigned to {self.phones[assignee].name} ({len(wp)} stops)", who=assignee)

    def route_forget(self, pin_id: str) -> None:
        if pin_id in self.route["order"]:
            self.route["order"].remove(pin_id)
            if pin_id in self.route["reached"]:
                self.route["reached"].remove(pin_id)
            self._number_waypoints()

    def _number_waypoints(self) -> None:
        """Waypoint pins carry their stop number as the label (so every screen shows the same numbers)."""
        for n, k in enumerate(self.route["order"], 1):
            if k in self.pins:
                self.pins[k]["label"] = str(n)

    # ---- snapshots ----------------------------------------------------
    def viewer_state(self) -> dict:
        now = time.time()
        phones = []
        for p in self.phones.values():
            d = p.public(now)
            d["trail"] = p.trail
            phones.append(d)
        return {"t": "state", "ts": now, "phones": phones, "pins": list(self.pins.values()), "coverage_pct": self.coverage.percent(),
                "plan": self.plan, "budget_s": self.planner.budget_s, "alert_stats": getattr(self, "alert_stats", None)}

    def planes_snapshot(self) -> dict:
        return {"t": "planes", "upsert": list(self.planes.values()), "remove": [], "reset": True}

    def phone_state(self, me: str) -> dict:
        """What one phone needs: colleagues (with walls between us) and pins."""
        now = time.time()
        mine = self.phones.get(me)
        walls = self.walls()
        peers = []
        for p in self.phones.values():
            if p.id == me:
                continue
            d = p.public(now)
            if mine and mine.pos and p.pos:
                eye = (mine.pos[1] + p.pos[1]) / 2
                d["walls_between"] = walls_between((mine.pos[0], mine.pos[2]), (p.pos[0], p.pos[2]), eye, walls)
                d["distance"] = round(math.dist(mine.pos, p.pos), 2)
            peers.append(d)
        me_alert = mine.vitals.public(now) if mine else None
        mine_plan = self.plan["people"].get(me, {})
        todo = next((d["text"] for d in self.plan["decisions"] if d.get("who") == me), None)  # my top suggestion
        me_state = {"alert": me_alert, "suggestion": todo, "turnaround_s": mine_plan.get("turnaround_s"),
                    "exit_m": mine_plan.get("exit_m"), "exit_s": mine_plan.get("exit_s"), "guide": mine_plan.get("guide"),
                    "hazard": mine.hazard.warn if mine else None,
                    "voice": list(mine.voice.cues) if mine else [], "nav": mine.nav if mine else "off",
                    "nav_info": mine.voice.view if mine else None}
        return {"t": "state", "ts": now, "peers": peers, "pins": list(self.pins.values()), "me": me_state}

    def snapshot(self) -> dict:
        s = self.viewer_state()
        s["t"] = "snapshot"
        s["planes"] = list(self.planes.values())
        s["objects"] = self.objects
        s["fusion"] = {"frames": self.fusion.frames, "blocks": len(self.fusion.tsdf), "meshes": len(self.fusion.meshes),
                       "depth_walls": len(self.depth_walls)}
        s["events"] = list(self.events)[-100:]
        s["started"] = self.started
        return s
