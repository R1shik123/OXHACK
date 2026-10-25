import math

import numpy as np

from sightline.fusion import Fusion, fit_wall_lines
from sightline.protocol import pack_depth_frame, parse_depth_frame
from sightline.state import World

W, H = 96, 72
FX = FY = 80.0
CX, CY = W / 2, H / 2
FLOOR = -1.4  # marker-relative floor height


def look_pose(pos, yaw):
    """Camera at pos looking along (sin yaw, 0, cos yaw), level; columns = camera axes in the world."""
    f = np.array([math.sin(yaw), 0, math.cos(yaw)])
    z = -f
    y = np.array([0, 1.0, 0])
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, pos
    return m


def render(pose, noise, rng):
    """Depth of a scene: wall at z = 4 (x in -3..3), floor, and a 1 x 0.8 m table top at 0.75 m (z 2..2.8, x -0.5..0.5)."""
    depth = np.zeros((H, W), np.uint16)
    rgb = np.zeros((H, W, 3), np.uint8)
    o = pose[:3, 3]
    for v in range(H):
        for u in range(W):
            dc = np.array([(u + 0.5 - CX) / FX, -(v + 0.5 - CY) / FY, -1.0])
            d = pose[:3, :3] @ dc  # world ray (not unit; camera-z component is 1)
            best, col = None, None
            for t, c in (((4.0 - o[2]) / d[2] if d[2] > 1e-6 else None, (200, 200, 205)),  # wall
                         ((FLOOR - o[1]) / d[1] if d[1] < -1e-6 else None, (90, 90, 95)),  # floor
                         ((FLOOR + 0.75 - o[1]) / d[1] if d[1] < -1e-6 else None, (150, 95, 55))):  # table top
                if t is None or t <= 0:
                    continue
                p = o + d * t
                if c == (200, 200, 205) and not (-3 <= p[0] <= 3 and FLOOR <= p[1] <= FLOOR + 2.5):
                    continue
                if c == (150, 95, 55) and not (-0.5 <= p[0] <= 0.5 and 2.0 <= p[2] <= 2.8):
                    continue
                if c == (90, 90, 95) and not (-3 <= p[0] <= 3 and p[2] <= 4):
                    continue
                if best is None or t < best:
                    best, col = t, c
            if best is not None and best < 5:
                depth[v, u] = int(best * (1 + rng.normal(0, noise)) * 1000)  # t is camera-z depth here
                rgb[v, u] = col
    return depth, rgb


def test_fusion_builds_mesh_walls_and_table():
    rng = np.random.default_rng(1)
    f = Fusion()
    for i in range(14):  # walk sideways while looking at the wall, like a responder sweeping a room
        pose = look_pose([-1.5 + i * 0.22, 0.05, 0.2], math.radians(-12 + i * 2))
        depth, rgb = render(pose, 0.03, rng)
        frame = parse_depth_frame(pack_depth_frame(pose, (FX, FY, CX, CY), depth, rgb)[1:])
        f.integrate(frame)
    assert f.frames == 14 and len(f.tsdf) > 20
    meshes = [f.mesh_block(k) for k in f.take_dirty(limit=10_000)]
    nverts = sum(int.from_bytes(m[7:9], "little") for m in meshes if m)
    assert nverts > 500  # a real surface came out

    walls, furniture = f.semantics(FLOOR)
    assert any(abs(p[1] - 4.0) < 0.2 and abs(q[1] - 4.0) < 0.2 and abs(p[0] - q[0]) > 1.5 for p, q, *_ in walls), walls
    tables = [o for o in furniture if o["kind"] == "table"]
    assert tables, furniture
    t = max(tables, key=lambda o: o["area"])
    assert abs(t["top"] - 0.75) < 0.12 and t["x0"] > -0.8 and t["x1"] < 0.8 and t["z0"] > 1.7 and t["z1"] < 3.25  # grazing views smear horizontal tops ~0.3 m


def test_depth_frame_validation():
    pose = look_pose([0, 0, 0], 0)
    good = pack_depth_frame(pose, (FX, FY, CX, CY), np.zeros((H, W), np.uint16), np.zeros((H, W, 3), np.uint8))[1:]
    assert parse_depth_frame(good)["w"] == W
    for bad in (good[:-1], good[:20], b"\x02" + good[1:]):
        try:
            parse_depth_frame(bad)
            raise AssertionError("accepted a malformed frame")
        except ValueError:
            pass
    skewed = pose.copy()
    skewed[:3, :3] *= 3  # not a rotation
    try:
        parse_depth_frame(pack_depth_frame(skewed, (FX, FY, CX, CY), np.zeros((H, W), np.uint16), np.zeros((H, W, 3), np.uint8))[1:])
        raise AssertionError("accepted a non-rigid pose")
    except ValueError:
        pass


def test_wall_line_fit_splits_at_doorway():
    cells = [(x, 40) for x in range(0, 20)] + [(x, 40) for x in range(30, 50)]  # 2 m wall, 1 m gap, 2 m wall
    walls = fit_wall_lines(cells, FLOOR)
    lengths = sorted(abs(q[0] - p[0]) for p, q, *_ in walls)
    assert len(walls) == 2 and all(1.7 < L < 2.1 for L in lengths), walls


def test_furniture_outlines_boost_fusion_and_confirm_boxes():
    rng = np.random.default_rng(3)
    plain, boosted = Fusion(), Fusion()
    for i in range(6):
        pose = look_pose([-1.0 + i * 0.3, 0.05, 0.2], math.radians(-6 + i * 2))
        depth, rgb = render(pose, 0.03, rng)
        # the table top's pixels (brown) make the outline YOLO would draw
        ys, xs = np.nonzero((rgb == (150, 95, 55)).all(axis=2))
        box = [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)] if len(xs) else []
        plain.integrate(parse_depth_frame(pack_depth_frame(pose, (FX, FY, CX, CY), depth, rgb)[1:]))
        boosted.integrate(parse_depth_frame(pack_depth_frame(pose, (FX, FY, CX, CY), depth, rgb, boxes=box)[1:]))
    # the table top's voxels gather more confidence with the outlines -> it meshes sooner and cleaner
    def table_weight(f):
        tot = 0.0
        for k, wgt in f.weight.items():
            pts = np.array(k) * 16 * 0.04 + f._offsets
            m = (np.abs(pts[:, 1] - (FLOOR + 0.75)) < 0.08) & (np.abs(pts[:, 0]) < 0.5) & (pts[:, 2] > 2.0) & (pts[:, 2] < 2.8)
            tot += float(wgt[m].sum())
        return tot
    assert table_weight(boosted) > 1.8 * table_weight(plain)
    assert len(boosted.furniture_seen) >= 3
    _, furn = boosted.semantics(FLOOR)
    tables = [o for o in furn if o["kind"] == "table"]
    assert tables and all(o["confirmed"] for o in tables)
    assert all("label" not in o for o in furn)  # never any item names


def test_depth_frame_v2_rejects_bad_box_lists():
    pose = look_pose([0, 0, 0], 0)
    d, c = np.zeros((H, W), np.uint16), np.zeros((H, W, 3), np.uint8)
    good = pack_depth_frame(pose, (FX, FY, CX, CY), d, c, boxes=[(1, 2, 30, 40)])[1:]
    assert parse_depth_frame(good)["boxes"] == [(1, 2, 30, 40)]
    for bad in (good[:-1], good + b"\x00"):
        try:
            parse_depth_frame(bad)
            raise AssertionError("accepted a malformed box list")
        except ValueError:
            pass
    oob = parse_depth_frame(pack_depth_frame(pose, (FX, FY, CX, CY), d, c, boxes=[(10, 10, 999, 20)])[1:])
    assert oob["boxes"] == []  # an outline outside the image is ignored, not trusted


def test_classifier_confirms_only_on_agreement_and_never_labels_screens():
    from PIL import Image
    import io as _io
    from sightline.classify import ObjectConfirmer, parse_posed_frame, project_box
    import struct, time as _t

    # a phone 2 m in front of a 1 x 0.8 m table, looking at it; snapshot 360 x 480 portrait
    pose = look_pose([0.0, FLOOR + 1.45, 0.0], 0.0)
    pose[:3, :3] = pose[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(-0.5), -math.sin(-0.5)], [0, math.sin(-0.5), math.cos(-0.5)]])
    ws, hs = 480, 360
    K = (370.0, 370.0, ws / 2, hs / 2)
    buf = _io.BytesIO(); Image.new("RGB", (hs, ws), (90, 60, 40)).save(buf, "JPEG")
    body = struct.pack("<16f4f2H", *pose.astype(np.float32).T.reshape(-1), *K, ws, hs) + buf.getvalue()
    frame = parse_posed_frame(body)
    table = {"kind": "table", "x0": -0.5, "x1": 0.5, "z0": 1.6, "z1": 2.4, "top": 0.75}
    box = project_box(table, FLOOR, frame)
    assert box and 0 <= box[0] < box[2] <= hs and 0 <= box[1] < box[3] <= ws  # inside the picture
    assert project_box({**table, "z0": -2.4, "z1": -1.6}, FLOOR, frame) is None  # behind the camera

    answers = iter([("dining table", 0.6), ("dining table", 0.55), ("nematode", 0.9)])
    c = ObjectConfirmer(classify=lambda img: next(answers))
    c.add_frame("p", frame)
    assert c.run([table], FLOOR) == 0  # one view is not enough
    assert c.run([table], FLOOR) == 0  # same snapshot again: no second vote
    frame2 = parse_posed_frame(body); frame2["ts"] = frame["ts"] + 1
    c.add_frame("p", frame2)
    assert c.run([table], FLOOR) == 1 and c.is_confirmed(table)

    w = World()
    w.set_semantics([], [dict(table)])
    w.confirmer = c
    w.set_semantics([], [dict(table)])
    assert w.objects[0]["confirmed"] and "label" not in w.objects[0]  # confirmed, but no name ever leaves the laptop


def _wall_with_gap(sill: bool):
    """10 cm cells: a wall along z=0 from 0-1.5 m and 3-5 m; the 1.5 m gap has a sill (window) or not (door)."""
    cols = {}
    for i in list(range(0, 15)) + list(range(30, 50)):
        cols[(i, 0)] = (0.0, 2.3)
    if sill:
        for i in range(15, 30):
            cols[(i, 0)] = (0.0, 0.9)
    return cols


def test_window_gap_with_sill_is_bridged_but_doorway_stays_open():
    from sightline.fusion import _has_sill, fit_walls
    for sill in (True, False):
        cols = _wall_with_gap(sill)
        fit = fit_walls([k for k, (lo, hi) in cols.items() if hi > 1.4], FLOOR)
        assert len(fit["walls"]) == 2 and len(fit["gaps"]) == 1
        assert _has_sill(fit["gaps"][0], cols) is sill


def test_free_standing_pillar_is_found():
    from sightline.fusion import find_pillars
    cols = {(20 + a, 30 + b): (0.0, 2.5) for a in range(4) for b in range(4)}  # 40 x 40 cm, floor to 2.5 m
    cols.update({(60 + a, 30): (0.3, 1.0) for a in range(3)})  # a low thing: not a pillar
    out = find_pillars([k for k, (lo, hi) in cols.items() if hi - lo > 1.1], cols)
    assert len(out) == 1
    (x0, z0, x1, z1), strong = out[0]
    assert strong and abs(x1 - x0 - 0.4) < 1e-6 and abs(z1 - z0 - 0.4) < 1e-6


def test_scene_labeler_needs_two_agreeing_snapshots():
    import io
    import threading
    import numpy as np
    from PIL import Image
    from sightline.scene import SceneLabeler

    buf = io.BytesIO()
    Image.new("RGB", (360, 480), "gray").save(buf, "JPEG")
    pose = np.eye(4)  # camera at the origin looking down -z; portrait snapshot 360 x 480
    frame = lambda ts: {"pose": pose, "K": (400.0, 400.0, 240.0, 180.0), "size": (480, 360), "jpeg": buf.getvalue(), "ts": ts}

    class Frames:
        lock = threading.Lock()
        frames = {}

    seg = np.full((128, 128), 8)  # every pixel "windowpane"
    lab = SceneLabeler(Frames, segment=lambda img: seg)
    cand = [{"key": ("gap", 0, -10), "what": "wall gap", "at": (0, -3), "points": [(0.0, 0.0, -3.0)]}]
    Frames.frames = {"a": frame(1.0)}
    lab.run(cand)
    assert lab.label(("gap", 0, -10)) is None  # one snapshot proves nothing
    Frames.frames = {"a": frame(2.0)}
    assert lab.run(cand) == 1 and lab.label(("gap", 0, -10)) == "window"


def test_wall_texture_is_painted_from_a_posed_snapshot_and_holes_filled():
    import io
    import threading
    import numpy as np
    from PIL import Image
    from sightline.fusion import Fusion
    from sightline.texture import Textures, push_pull

    img = np.zeros((480, 360, 3), np.uint8)  # upright portrait snapshot
    img[:240] = (220, 30, 30)  # top half red  -> sensor u < cx -> world x < 0 for this camera
    img[240:] = (30, 30, 220)  # bottom half blue -> world x > 0
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")

    class Frames:
        lock = threading.Lock()
        frames = {"a": {"pose": np.eye(4), "K": (300.0, 300.0, 240.0, 180.0), "size": (480, 360), "jpeg": buf.getvalue(), "ts": 1.0}}

    tex = Textures(Frames, Fusion())
    tex.set_geometry([((-1.0, -3.0), (1.0, -3.0))], -1.4, None)  # a 2 m wall 3 m in front of the camera
    assert tex.paint() == 1
    s = next(iter(tex.surfaces.values()))
    rgb = push_pull(s.acc, s.w)
    row = rgb[int(1.4 * 40)]  # camera height
    assert row[5][0] > 150 and row[5][2] < 80  # left end of the wall: red
    assert row[-5][2] > 150 and row[-5][0] < 80  # right end: blue
    assert (rgb.sum(axis=2) > 0).all()  # unseen texels filled from seen neighbours, no black holes
    assert tex.messages() and not tex.messages()  # sent once, then only when something changes


def test_door_outlines_are_read_and_a_gap_with_doors_seen_becomes_a_door():
    import numpy as np
    from sightline.fusion import Fusion
    from sightline.protocol import pack_depth_frame, parse_depth_frame
    raw = pack_depth_frame(np.eye(4), (100.0, 100.0, 64.0, 48.0), np.full((96, 128), 2000), np.zeros((96, 128, 3)), boxes=[(10, 10, 30, 40)])
    raw = bytearray(raw)
    raw[-8:-6] = (10 | 0x8000).to_bytes(2, "little")  # the phone marks door outlines with the top bit
    fr = parse_depth_frame(bytes(raw[1:]))
    assert fr["doors"] == [(10, 10, 30, 40)] and fr["boxes"] == []
    f = Fusion()
    f.surface_columns = lambda floor_y: {**{(i, 0): (0.0, 2.3) for i in list(range(0, 15)) + list(range(30, 50))}}
    f.doors_seen.extend([(2.2, 0.05), (2.3, 0.0)])  # the detector saw a door in the gap twice
    walls, _ = f.semantics(-1.4)
    assert any(w[4] == "door" for w in walls)


def test_doors_in_outside_walls_are_exits():
    from sightline.predict import exit_doors
    walls = [((0, 0), (8, 0), -1.4, 1.0, "wall"), ((8, 0), (8, 6), -1.4, 1.0, "wall"), ((0, 6), (8, 6), -1.4, 1.0, "wall"),
             ((0, 0), (0, 6), -1.4, 1.0, "wall"),
             ((3, 6), (4, 6), -1.4, 0.7, "door"),  # in the outside wall: a way out
             ((4, 2.5), (4, 3.5), -1.4, 0.7, "door")]  # between two inside rooms: not an exit
    assert exit_doors(walls) == [(3.5, 6.0)]


def test_stairs_found_up_and_down_but_not_a_table():
    from sightline.fusion import find_stairs, is_large
    up = {(i, j): (0.0, 0.17 * (1 + i // 3)) for i in range(0, 24) for j in range(10)}  # 1 m wide, 17 cm risers every 30 cm
    out = find_stairs(up, set())
    assert len(out) == 1 and out[0]["strong"] and not out[0]["down"] and out[0]["up"][0] > 0.9  # climbs along +x
    down = {(i, j): (-0.17 * (1 + i // 3), -0.17 * (1 + i // 3)) for i in range(0, 24) for j in range(10)}
    out = find_stairs(down, set())
    assert len(out) == 1 and out[0]["down"]
    table = {(i, j): (0.74, 0.75) for i in range(15) for j in range(8)}
    assert find_stairs(table, set()) == []
    assert is_large({"x0": 0, "z0": 0, "x1": 1.6, "z1": 0.8}) and not is_large({"x0": 0, "z0": 0, "x1": 0.5, "z1": 0.5})
