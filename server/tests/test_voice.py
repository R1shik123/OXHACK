import math
import stat

import pytest
from fastapi.testclient import TestClient

from sightline import app as A
from sightline import voice as V
from sightline.security import Registry


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    A.world.reset()
    monkeypatch.setattr("sightline.security.DATA", tmp_path)
    monkeypatch.setattr(A, "DATA", tmp_path)
    monkeypatch.setattr(V, "DATA", tmp_path)  # never the laptop's real key, never the real ElevenLabs
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    A.registry = Registry()
    A.speech = V.Speech()
    for lim in (A.text_limit, A.pair_limit, A.voice_limit):
        lim.state.clear()
    yield


def phone(pos, fwd, pid="a", name="Alice"):
    p = A.world.phone(pid, name)
    p.pos, p.fwd, p.connected, p.pose_ts = list(pos), list(fwd), True, 1e12
    return p


def tick(p, now=1000.0, walls=(), zones=(), exits=()):
    p.voice.update(p, A.world, now, list(walls), list(zones), list(exits))
    return [c["text"] for c in p.voice.cues]


def test_sides_follow_the_shared_frame():
    # facing +Z (into the building), +X is on the left in a right-handed, Y-up frame
    p = phone([0, 1.4, 0], [0, 0, 1])
    door = ((1.5, 2.0), (2.5, 2.0), 0, 2, "door")
    assert tick(p, walls=[door]) == ["Door ahead on your left, 3 metres."]
    assert V.side(0.0) == "straight ahead" and V.side(-math.pi / 2) == "on your right" and V.side(math.pi) == "behind you"


def test_trip_hazard_ahead_once_per_approach_and_not_behind():
    p = phone([0, 1.4, 0], [0, 0, 1])
    clutter = {"kind": "clutter", "x": 0.0, "z": 2.5, "r": 0.5}
    behind = {"kind": "drop", "x": 0.0, "z": -2.0, "r": 0.5}
    assert tick(p, zones=[clutter, behind]) == ["Trip hazard straight ahead, in 2 metres. Low obstacle on the floor."]
    assert len(tick(p, now=1001, zones=[clutter])) == 1  # no repeat while still close
    p.pos = [0, 1.4, -6]
    tick(p, now=1030, zones=[clutter])  # walked away: re-armed
    p.pos = [0, 1.4, 0]
    assert len(tick(p, now=1040, zones=[clutter])) == 2


def test_stairs_warning_inside_zone():
    p = phone([0, 1.4, 0], [0, 0, 1])
    stairs = {"kind": "stairs", "poly": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}
    assert tick(p, zones=[stairs]) == ["Stairs right here. Take the handrail."]


def test_colleague_alert_and_hazard_label_are_spoken_on_the_phone_only():
    p = phone([0, 1.4, 0], [0, 0, 1])
    bob = phone([0, 1.4, 4], [0, 0, 1], "b", "Bob")
    bob.vitals.alert = {"kind": "fall", "since": 990.0, "acked": False}
    p.hazard.warn = {"id": "h1", "label": "gas leak", "dist": 2.4}
    tick(p)
    cues = list(p.voice.cues)
    assert [c["text"] for c in cues] == ["Alert. Bob may have fallen, 4 metres straight ahead.",
                                         "Danger. gas leak, 2 metres ahead. Keep away."]
    assert all(c["local"] and "id" not in c and c["prio"] == 0 for c in cues)  # names / labels never go to ElevenLabs


def test_navigation_start_turns_corner_and_arrival():
    p = phone([0, 1.4, 0], [0, 0, 1])
    p.nav = "home"
    # route: 4 m straight ahead (+Z), then 4 m towards -X (a right turn when facing +Z)
    path = [[0, z / 2] for z in range(0, 9)] + [[-x / 2, 4.0] for x in range(1, 9)]
    A.world.plan = {"people": {"a": {"home": {"path": path, "m": 8.0, "zone": 0}}}, "decisions": []}
    assert tick(p)[-1] == "Guiding you to the command post. 8 metres."
    assert tick(p, now=1001)[-1] == "In 4 metres, turn right."
    p.fwd = [0, 0, -1]  # facing away from the route
    assert tick(p, now=1002)[-1] == "Turn around."
    p.fwd = [1, 0, 0]  # facing +X: the route ahead (+Z) is on my right
    assert tick(p, now=1007)[-1] == "Turn right."
    assert p.voice.view["to"] == "home" and p.voice.view["m"] == 8.0
    p.pos = [-3.6, 1.4, 4.0]
    p.fwd = [-1, 0, 0]
    assert tick(p, now=1010)[-1] == "You have arrived at the command post."


def test_no_route_yet_and_nav_off_resets():
    p = phone([0, 1.4, 0], [0, 0, 1])
    p.nav = "exit"
    A.world.plan = {"people": {"a": {}}, "decisions": []}
    assert tick(p) == ["No known route to the nearest exit yet. Keep mapping."]
    p.nav = "off"
    tick(p, now=1001)
    assert p.voice.view is None and p.voice.nav_key is None


def test_next_corner_ignores_grid_zigzag():
    zigzag = [[0, 0]] + [[(i % 2) * 0.2, i * 0.2] for i in range(1, 40)]
    assert V.next_corner(zigzag) is None


def test_phone_can_set_nav_and_only_fetch_issued_cue_audio(monkeypatch):
    admin = TestClient(A.admin_app, base_url="http://127.0.0.1:8000")
    admin.cookies.set(A.ADMIN_COOKIE, A.registry.admin_token)
    code = admin.post("/api/admin/pair", headers={"X-Sightline": "1"}).json()["uri"].split("c=")[1].split("&")[0]
    ph = TestClient(A.phone_app)
    dev = ph.post("/api/pair", json={"code": code, "name": "Alice", "platform": "ios"}).json()
    h = {"Authorization": f"Bearer {dev['token']}"}
    A.handle_phone_text(dev["device_id"], '{"t": "nav", "to": "exit"}')
    assert A.world.phones[dev["device_id"]].nav == "exit"

    calls = []

    async def fake_synth(text):
        calls.append(text)
        return b"ID3fake-mp3"

    monkeypatch.setattr(A.speech, "_synth", fake_synth)
    cid = A.speech.issue("Turn left.")
    assert ph.get(f"/api/voice/{cid}.mp3", headers=h).status_code == 404  # no key: phone falls back to its own voice
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_" + "x" * 40)
    assert ph.get(f"/api/voice/{cid}.mp3").status_code == 401  # paired phones only
    r = ph.get(f"/api/voice/{cid}.mp3", headers=h)
    assert r.status_code == 200 and r.content == b"ID3fake-mp3" and r.headers["content-type"] == "audio/mpeg"
    ph.get(f"/api/voice/{cid}.mp3", headers=h)
    assert calls == ["Turn left."]  # cached
    assert ph.get(f"/api/voice/{V.cue_id('Say something else')}.mp3", headers=h).status_code == 404  # never issued
    assert ph.get("/api/voice/..%2Fkey.mp3", headers=h).status_code == 404


def test_elevenlabs_key_setup_only_from_the_mac_and_owner_only(tmp_path, monkeypatch):
    h = {"X-Sightline": "1"}
    mac = TestClient(A.admin_app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    mac.cookies.set(A.ADMIN_COOKIE, A.registry.admin_token)
    other = TestClient(A.admin_app, base_url="https://10.0.0.5:8444", client=("10.0.0.9", 50001))
    other.cookies.set(A.ADMIN_COOKIE, A.registry.admin_token)
    key = "sk_" + "y" * 45

    async def fake_check():
        return True, "Key works."

    monkeypatch.setattr(A.speech, "check", fake_check)
    assert not mac.get("/api/admin/voice").json()["configured"]
    assert other.post("/api/admin/voice/key", json={"key": key}, headers=h).status_code == 403
    assert mac.post("/api/admin/voice/key", json={"key": "nope"}, headers=h).status_code == 400
    assert mac.post("/api/admin/voice/key", json={"key": key, "voice": "bad voice!"}, headers=h).status_code == 400
    r = mac.post("/api/admin/voice/key", json={"key": f" {key}\n", "voice": "EXAVITQu4vr4xnSDxMaL"}, headers=h)
    assert r.status_code == 200 and r.json()["ok"] and key not in r.text
    f = tmp_path / "elevenlabs.key"
    assert f.read_text() == key and stat.S_IMODE(f.stat().st_mode) == 0o600
    st = mac.get("/api/admin/voice").json()
    assert st["configured"] and st["voice"] == "EXAVITQu4vr4xnSDxMaL" and key not in str(st)


def test_voice_tick_issues_cues_for_connected_phones(monkeypatch):
    p = phone([0, 1.4, 0], [0, 0, 1])
    monkeypatch.setattr(A.world, "merged_walls", lambda: [((-0.5, 3.0), (0.5, 3.0), 0, 2, "door")])
    A.voice_tick()
    cue = p.voice.cues[-1]
    assert cue["text"] == "Door straight ahead, 3 metres." and cue["kind"] == "door"
    assert A.speech.texts[cue["id"]] == cue["text"]  # issued: the phone may now fetch its audio
    state = A.world.phone_state("a")["me"]
    assert state["voice"][-1]["n"] == cue["n"] and state["nav"] == "off"
