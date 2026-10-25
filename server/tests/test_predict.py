import math

import numpy as np

from sightline.predict import Kalman, Planner, Router, search_eta
from sightline.state import World


def test_kalman_tracks_and_predicts_with_growing_uncertainty():
    k = Kalman()
    for i in range(30):  # walking +x at 1 m/s
        k.update(i * 0.1, (i * 0.1, 0.0))
    assert abs(k.speed - 1.0) < 0.15
    (x, z), r0 = k.predict(2.9)
    (x5, _), r5 = k.predict(2.9 + 5)
    assert 2.7 < x < 3.1 and r5 > r0 + 3  # keeps going ~3 s at most, then the circle grows
    assert x5 < x + 3.6


def test_router_goes_through_the_door_not_the_wall():
    r = Router()
    # wall along x = 0 from z = -5 to 5 with a door at z 2..3
    walls = [((0, -5), (0, 2), -1.4, 1.0, "wall"), ((0, 3), (0, 5), -1.4, 1.0, "wall"), ((0, 2), (0, 3), -1.4, 0.6, "door")]
    r.build(walls, [], [(-2, 0), (2, 0)])
    a, b = r.cell((-2, 0)), r.cell((2, 0))
    d, pred = r.from_sources([a])
    assert d[0, b] > 5.5  # straight would be 4 m; via the door ~ 2*sqrt(4+6.25) ~ 6.4 m
    path = r.path(pred[0], b)
    assert any(2.0 <= z <= 3.1 for x, z in path if abs(x) < 0.3)  # crosses x = 0 inside the doorway


def test_search_eta_from_saturating_curve():
    hist = [(t, 100 * (1 - math.exp(-0.01 * t))) for t in range(0, 120, 5)]
    eta = search_eta(hist, target=95)
    true_t = -math.log(0.05) / 0.01
    assert abs((hist[-1][0] + eta) - true_t) < 5


def test_planner_suggests_next_room_but_not_to_someone_down():
    w = World()
    # two rooms side by side (x -6..0 and 0..6), joined by a door in the middle wall
    w.depth_walls = [((-6, 0), (6, 0), -1.4, 1.0, "vertical"), ((-6, 0), (-6, 8), -1.4, 1.0, "vertical"),
                     ((6, 0), (6, 8), -1.4, 1.0, "vertical"), ((-6, 8), (6, 8), -1.4, 1.0, "vertical"),
                     ((0, 0), (0, 3.5), -1.4, 1.0, "vertical"), ((0, 4.5), (0, 8), -1.4, 1.0, "vertical"),
                     ((0, 3.5), (0, 4.5), -1.4, 0.7, "door")]
    w.coverage.set_outline(w.merged_walls())
    a, b = w.phone("a", "Alice"), w.phone("b", "Bob")
    for p, xz in ((a, (-3, 3)), (b, (-4, 6))):  # both in the west room
        p.pos, p.connected, p.pose_ts = [xz[0], 0.0, xz[1]], True, 1e12
    b.vitals.alert = {"kind": "fall", "since": 0, "acked": False, "pos": b.pos}
    out = Planner().plan(w)
    kinds = [d["kind"] for d in out["decisions"]]
    assert "assist" not in kinds  # the "converge" suggestion is switched off for now
    nxt = [d for d in out["decisions"] if d["kind"] == "next_room"]
    assert len(nxt) == 1 and nxt[0]["who"] == "a"  # Alice gets the east room; Bob is down
    assert len(out["search"]["rooms"]) == 2
    assert out["people"]["a"]["exit_m"] is not None and out["people"]["a"]["home"]["m"] > 0


def test_route_home_avoids_yellow_when_it_can_and_flags_it_when_it_cannot():
    from sightline.predict import Router
    walls = [((-4, 0), (4, 0), -1.4, 1.0, "wall"), ((-4, 10), (4, 10), -1.4, 1.0, "wall"),
             ((-4, 0), (-4, 10), -1.4, 1.0, "wall"), ((4, 0), (4, 10), -1.4, 1.0, "wall")]
    R = Router()
    clutter = [{"kind": "clutter", "x": 0.0, "z": 5.0, "r": 0.8}]  # in the middle of an 8 m wide room: easy to go round
    R.build(walls, [], [(0, 1), (0, 9)], clutter, [])
    dist, pred = R.from_sources([R.cell((0, 1))])
    path, length, worst = R.route(pred[0], R.cell((0, 9)))
    assert worst == 0 and length < 10  # went round it, only a little longer than 8 m
    wall_to_wall = [{"kind": "stairs", "poly": [[-4, 4], [4, 4], [4, 6], [-4, 6]]}]  # no way round
    R.build(walls, [], [(0, 1), (0, 9)], wall_to_wall, [])
    dist, pred = R.from_sources([R.cell((0, 1))])
    path, length, worst = R.route(pred[0], R.cell((0, 9)))
    assert worst == 1 and np.isfinite(dist[0, R.cell((0, 9))])  # crossed, and flagged


def test_victim_pin_gets_nearest_responder_by_route():
    w = World()
    w.depth_walls = [((-6, 0), (6, 0), -1.4, 1.0, "vertical"), ((-6, 10), (6, 10), -1.4, 1.0, "vertical")]
    a, b = w.phone("a", "Alice"), w.phone("b", "Bob")
    a.pos, a.connected, a.pose_ts = [0.0, 0.0, 2.0], True, 1e12
    b.pos, b.connected, b.pose_ts = [5.0, 0.0, 8.0], True, 1e12
    w.add_pin("victim", "trapped", [4.0, -1.2, 7.0], "Command")
    v = next(d for d in Planner().plan(w)["decisions"] if d["kind"] == "victim")
    assert v["who"] == "b" and v["prio"] == 1 and "trapped" in v["text"]


def test_evaluation_scores_rooms_and_risk_honestly():
    from sightline.evaluate import score_risk, score_rooms
    from sightline.rooms import segment_rooms
    walls = [((-6, 0), (6, 0), -1.4, 1.0, "wall"), ((-6, 0), (-6, 8), -1.4, 1.0, "wall"), ((6, 0), (6, 8), -1.4, 1.0, "wall"),
             ((-6, 8), (6, 8), -1.4, 1.0, "wall"), ((0, 0), (0, 3.5), -1.4, 1.0, "wall"), ((0, 4.5), (0, 8), -1.4, 1.0, "wall"),
             ((0, 3.5), (0, 4.5), -1.4, 0.7, "door")]
    rooms = segment_rooms(walls)
    truth = [[[-6, 0], [0, 0], [0, 8], [-6, 8]], [[0, 0], [6, 0], [6, 8], [0, 8]]]
    s = score_rooms(rooms, truth)
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["mean_iou"] > 0.8
    s = score_rooms(rooms, truth[:1] + [[[10, 10], [12, 10], [12, 12], [10, 12]]])  # one room the detector can't see
    assert s["recall"] == 0.5 and s["precision"] == 0.5
    zones = [{"kind": "clutter", "x": 1.0, "z": 1.0, "r": 0.5}, {"kind": "fall", "x": 5.0, "z": 5.0, "r": 1.0}]
    k = score_risk(zones, [[1.2, 1.1], [9.0, 9.0]])  # one spot found, one missed; one zone right, one wrong
    assert k["recall"] == 0.5 and k["precision"] == 0.5


def test_occupancy_fills_narrow_wall_gaps_but_never_seals_a_doorway():
    from sightline.occupancy import Occupancy
    room = [((-4, 0), (4, 0), -1.4, 1.0, "wall"), ((-4, 6), (4, 6), -1.4, 1.0, "wall"), ((-4, 0), (-4, 6), -1.4, 1.0, "wall")]
    narrow = room + [((4, 0), (4, 2.75), -1.4, 1.0, "wall"), ((4, 3.25), (4, 6), -1.4, 1.0, "wall")]  # 50 cm gap
    occ = Occupancy(narrow, [], {}, 0.25, set())
    c = occ.cell((4.0, 3.0))
    assert occ.gaps and occ.blocked[int((4.0 - occ.x0) / 0.2), int((3.0 - occ.z0) / 0.2)]  # filled: probably wall
    door = room + [((4, 0), (4, 2.55), -1.4, 1.0, "wall"), ((4, 3.45), (4, 6), -1.4, 1.0, "wall")]  # 90 cm: door-sized
    occ = Occupancy(door, [], {}, 0.25, set())
    assert not occ.blocked[int((4.0 - occ.x0) / 0.2), int((3.0 - occ.z0) / 0.2)]  # expensive, not sealed
    walked = Occupancy(narrow, [], {}, 0.25, {(10, 7), (9, 7)})  # someone stood in the gap (40 cm cells)
    assert not walked.blocked[int((4.0 - walked.x0) / 0.2), int((3.0 - walked.z0) / 0.2)]


def test_astar_prefers_measured_floor_over_unknown():
    from sightline.occupancy import Occupancy
    walls = [((-6, 0), (6, 0), -1.4, 1.0, "wall"), ((-6, 6), (6, 6), -1.4, 1.0, "wall")]
    seen = {(i, j): 5 for i in range(-24, 24) for j in range(0, 6)}  # a 1.5 m strip near the bottom wall was seen
    occ = Occupancy(walls, [], seen, 0.25, set())
    path = occ.astar(occ.cell((-5, 3)), occ.cell((5, 3)), occ.step_cost())
    zs = [occ.xz(c)[1] for c in path]
    assert path and min(zs) < 1.6  # detours through the measured strip rather than straight across unknown floor


def test_second_phone_is_fine_aligned_to_the_first():
    import math
    from sightline.align import apply
    w = World()
    a, b = w.phone("a", "Alice"), w.phone("b", "Bob")
    rooms = [((-5, 0), (5, 0)), ((5, 0), (5, 8)), ((5, 8), (-5, 8)), ((-5, 8), (-5, 0)), ((0, 0), (0, 3))]
    wrong = (math.radians(1.5), 0.12, -0.16)  # Bob's marker lock was 1.5 deg / 20 cm off
    for n, (p, q) in enumerate(rooms):
        quad = lambda p, q: [[p[0], -1.4, p[1]], [q[0], -1.4, q[1]], [q[0], 1.0, q[1]], [p[0], 1.0, p[1]]]
        w.upsert_plane("a", f"w{n}", "wall", quad(p, q))
        bp, bq = apply(wrong, *p), apply(wrong, *q)
        w.upsert_plane("b", f"w{n}", "wall", quad(bp, bq))
    for _ in range(12):
        w.refine_alignment()
    err = []
    for n, (p, q) in enumerate(rooms):
        c = w.planes[f"b:w{n}"]["corners"]
        err += [math.dist((c[0][0], c[0][2]), p), math.dist((c[1][0], c[1][2]), q)]
    assert max(err) < 0.04  # Bob's walls now sit on Alice's (within 4 cm)
    assert w.planes["a:w0"]["corners"][0][0] == -5  # the reference phone is never moved


def test_waypoint_route_best_order_progress_and_guidance():
    w = World()
    w.depth_walls = [((-8, 0), (8, 0), -1.4, 1.0, "vertical"), ((-8, 10), (8, 10), -1.4, 1.0, "vertical"),
                     ((-8, 0), (-8, 10), -1.4, 1.0, "vertical"), ((8, 0), (8, 10), -1.4, 1.0, "vertical")]
    a = w.phone("a", "Alice")
    a.pos, a.connected, a.pose_ts = [0.0, 0.0, 1.0], True, 1e12
    far, near, mid = w.add_pin("waypoint", "", [6, 0, 9], "Command"), w.add_pin("waypoint", "", [1, 0, 3], "Command"), \
        w.add_pin("waypoint", "", [3, 0, 6], "Command")
    w.route_set(w.route["order"], "a", False)
    pl = Planner()
    out = pl.plan(w)
    as_dropped = out["route"]["total_m"]
    order = pl.best_order(w)
    assert order == [near["id"], mid["id"], far["id"]]  # near -> mid -> far beats the order they were dropped in
    w.route_set(order, "a", False)
    out = pl.plan(w)
    assert out["route"]["total_m"] < as_dropped and [s["n"] for s in out["route"]["stops"]] == [1, 2, 3]
    g = out["people"]["a"]["guide"]
    assert g["next"] == 1 and g["total"] == 3 and g["m"] > 1
    a.pos = [1.2, 0.0, 3.1]  # Alice arrives at stop 1
    out = pl.plan(w)
    assert out["route"]["stops"][0]["reached"] and out["people"]["a"]["guide"]["next"] == 2
