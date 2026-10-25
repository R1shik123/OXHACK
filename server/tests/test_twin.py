import math

from sightline.state import World
from sightline.twin import COV_CELL, STILL_SECS, Coverage, Vitals

WALL = [((2.0, -5.0), (2.0, 5.0), 0.0, 2.5)]  # wall along x = 2


def cell(x, z):
    return (math.floor(x / COV_CELL), math.floor(z / COV_CELL))


def test_coverage_stops_at_walls():
    c = Coverage()
    c.observe([0, 0, 0], [1, 0, 0], WALL)  # looking at the wall from 2 m away
    assert cell(1.5, 0) in c.seen
    assert cell(3.0, 0) not in c.seen  # behind the wall: not searched
    c2 = Coverage()
    c2.observe([0, 0, 0], [1, 0, 0], [])  # no wall: sees up to 5 m
    assert cell(4.5, 0) in c2.seen


def test_coverage_ignores_looking_at_floor_and_reports_percent():
    c = Coverage()
    c.observe([0, 0, 0], [0.05, -1, 0], WALL)
    assert not c.seen
    c.set_outline([((0, 0), (2, 0), 0, 2, "wall"), ((0, 2), (2, 2), 0, 2, "wall")])
    assert c.percent() == 0.0
    c.observe([0.1, 0, 1], [1, 0, 0], [])
    assert 0 < c.percent() < 100
    msg = c.message(full=True)
    assert msg["t"] == "coverage" and msg["seen"] and msg["bbox"] == (0, 0, 2, 2)
    assert c.message(full=False)["seen"]  # a full snapshot for one viewer doesn't swallow the shared delta
    assert c.message(full=False) is None  # ...but once the delta went out, nothing is pending


def test_fall_detected_then_cleared_by_standing_up():
    v = Vitals()
    t = 0.0
    for _ in range(30):  # standing, walking a little
        t += 0.1
        assert not v.update([t * 0.1, 1.4, 0], t)
    for _ in range(3):  # camera drops to the floor within ~0.3 s
        t += 0.1
        v.update([0.3, 0.3, 0], t)
    for _ in range(10):
        t += 0.1
        v.update([0.3, 0.3, 0], t)
    assert v.alert and v.alert["kind"] == "fall"
    t += 0.1
    assert v.update([0.3, 1.4, 0], t)  # stood up (moved > 0.5 m): cleared
    assert v.alert is None


def test_still_alert_ok_and_ack_cycle():
    v = Vitals()
    t = 0.0
    while t < STILL_SECS + 1:
        t += 0.5
        v.update([1.0 + 0.01 * math.sin(t), 1.4, 2.0], t)  # hand tremor only
    assert v.alert and v.alert["kind"] == "still"
    assert v.public(t)["kind"] == "still"
    v.ok(t)  # "I'm OK" clears it and restarts the window
    assert v.alert is None
    t += 5
    v.update([1.0, 1.4, 2.0], t)
    assert v.alert is None  # not re-raised immediately


def test_walking_never_alerts():
    v = Vitals()
    for i in range(int((STILL_SECS + 10) / 0.1)):
        t = i * 0.1
        assert not v.update([t * 0.8, 1.4 + 0.03 * math.sin(t * 6), 0], t)


def test_impact_fall_from_kneeling():
    """Accelerometer free fall + impact with only a small height drop (kneeling -> floor) still counts."""
    v = Vitals()
    t = 0.0
    for _ in range(30):
        t += 0.1
        v.update([1.0, 0.0, 1.0], t)  # kneeling, still-ish
    v.impact(3.4, t)
    for _ in range(15):
        t += 0.1
        v.update([1.05, -0.35, 1.0], t)  # 35 cm lower, not moving
    assert v.alert and v.alert["kind"] == "fall" and v.alert["source"] == "impact"


def test_health_telemetry_is_kept_and_flags_heat():
    from sightline import app as A
    from sightline.protocol import Status
    A.world.reset()
    p = A.world.phone("x", "Xan")
    A.note_health(p, Status(t="status", mem_mb=812, thermal=2, ar_fps=58.5, depth_hz=2.1, mem_warnings=0))
    pub = p.public(0)["health"]
    assert pub["mem_mb"] == 812 and pub["thermal"] == 2 and "_last_thermal" not in pub
    assert any(e["kind"] == "health" and "hot" in e["text"] for e in A.world.events)
    A.note_health(p, Status(t="status", memory_warning=True))
    assert any("low on memory" in e["text"] for e in A.world.events)


def test_hazard_warns_on_approach_and_flags_commander():
    from sightline.twin import HazardWatch
    gas = [{"id": "pin1", "kind": "hazard", "label": "gas", "pos": [0.0, 0.0, 0.0]}]
    w, t = HazardWatch(), 100.0
    for z in (6.0, 5.0, 4.0, 3.5, 2.8):  # walking straight at it, one step a second
        assert w.update([0.0, 0.0, z], gas, t) == []
        t += 1.0
    assert w.warn and w.warn["label"] == "gas"  # phone asks: avoid or proceed?
    assert w.public() is None  # commander not bothered yet
    assert "chose to proceed" in w.respond("pin1", "proceed", t)
    assert w.public()[0]["why"] == "chose to proceed" and w.warn is None
    w.update([0.0, 0.0, 6.0], gas, t + 1)  # walked away: all clear, next approach warns again
    assert w.public() is None


def test_hazard_ignored_warning_flags_within_1_5_m():
    from sightline.twin import HazardWatch
    gas = [{"id": "pin1", "kind": "hazard", "label": "", "pos": [0.0, 0.0, 0.0]}]
    w, t = HazardWatch(), 0.0
    for z in (5.0, 4.0, 3.0, 2.0):
        w.update([0.0, 0.0, z], gas, t)
        t += 1.0
    w.respond("pin1", "avoid", t)
    assert w.update([0.0, 0.0, 1.2], gas, t + 1) and w.public()[0]["why"] == "kept going despite the warning"
    assert w.warn is None  # said "avoid": no more prompts for this one until they walk away
