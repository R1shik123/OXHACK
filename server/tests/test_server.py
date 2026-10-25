import struct

import pytest
from fastapi.testclient import TestClient

from sightline import app as A
from sightline.protocol import pack_points
from sightline.security import Registry, RateLimiter


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    A.world.reset()
    monkeypatch.setattr("sightline.security.DATA", tmp_path)
    A.registry = Registry()
    A.phone_sockets.clear()
    A.viewer_sockets.clear()
    for lim in (A.text_limit, A.frame_limit, A.points_limit, A.pair_limit):
        lim.state.clear()
    yield


def admin_client() -> TestClient:
    c = TestClient(A.admin_app, base_url="http://127.0.0.1:8000")
    r = c.get(f"/?admin={A.registry.admin_token}", follow_redirects=False)
    assert r.status_code == 303 and A.ADMIN_COOKIE in r.cookies
    c.cookies.set(A.ADMIN_COOKIE, A.registry.admin_token)
    return c


def pair(phone: TestClient, admin: TestClient, name: str) -> dict:
    code = admin.post("/api/admin/pair", headers={"X-Sightline": "1"}).json()["uri"].split("c=")[1].split("&")[0]
    r = phone.post("/api/pair", json={"code": code, "name": name, "platform": "sim"})
    assert r.status_code == 200, r.text
    return r.json()


def wait_for(ws, t, n=60):
    for _ in range(n):
        m = ws.receive()
        if m.get("text") and f'"t":"{t}"' in m["text"]:
            import json
            return json.loads(m["text"])
    raise AssertionError(f"no {t}")


def test_admin_requires_token():
    c = TestClient(A.admin_app, base_url="http://127.0.0.1:8000")
    assert c.get("/").status_code == 401
    assert c.get("/?admin=wrong").status_code == 401
    assert c.get("/api/admin/state").status_code == 401
    a = admin_client()
    assert a.get("/").status_code == 200
    assert a.post("/api/admin/reset").status_code == 403  # no CSRF header
    assert a.post("/api/admin/reset", headers={"X-Sightline": "1"}).status_code == 200


def test_pairing_is_single_use_and_tokens_authenticate(tmp_path):
    admin, phone = admin_client(), TestClient(A.phone_app)
    code = admin.post("/api/admin/pair", headers={"X-Sightline": "1"}).json()["uri"].split("c=")[1].split("&")[0]
    ok = phone.post("/api/pair", json={"code": code, "name": "Alice", "platform": "ios"})
    assert ok.status_code == 200
    again = phone.post("/api/pair", json={"code": code, "name": "Mallory", "platform": "ios"})
    assert again.status_code == 403
    assert phone.post("/api/pair", json={"code": "x" * 20, "name": "M", "platform": "ios"}).status_code == 403
    # nothing about devices is written to disk; only the token's hash is held in memory
    assert not (tmp_path / "devices.json").exists()
    assert all(ok.json()["token"] != d.token_hash for d in A.registry.devices.values())
    assert phone.get("/api/frame/x.jpg").status_code == 401
    with pytest.raises(Exception):
        with phone.websocket_connect("/ws/phone"):
            pass
    with pytest.raises(Exception):
        with phone.websocket_connect("/ws/phone", headers={"Authorization": "Bearer nope"}):
            pass


def test_pairing_rate_limited():
    phone = TestClient(A.phone_app)
    codes = [phone.post("/api/pair", json={"code": "y" * 20, "name": "M", "platform": "ios"}).status_code for _ in range(15)]
    assert 429 in codes


def test_view_socket_rejects_foreign_origin():
    a = admin_client()
    with pytest.raises(Exception):
        with a.websocket_connect("/ws/view", headers={"origin": "http://evil.example"}):
            pass


def test_end_to_end_secure():
    admin = admin_client()
    with TestClient(A.phone_app) as phone:  # runs the lifespan -> broadcast loop
        alice, bob = pair(phone, admin, "Alice"), pair(phone, admin, "Bob")
        ha = {"Authorization": f"Bearer {alice['token']}"}
        hb = {"Authorization": f"Bearer {bob['token']}"}
        with phone.websocket_connect("/ws/phone", headers=ha) as a, phone.websocket_connect("/ws/phone", headers=hb) as b:
            assert wait_for(a, "welcome")["id"] == alice["device_id"]
            wait_for(b, "welcome")
            a.send_json({"t": "pose", "pos": [-1, 1.4, 3], "fwd": [1, 0, 0], "wall_dist": 1.0})
            b.send_json({"t": "pose", "pos": [1, 1.4, 3], "fwd": [-1, 0, 0]})
            b.send_json({"t": "plane", "id": "w1", "kind": "wall", "corners": [[0, 0, 1], [0, 0, 5], [0, 2.5, 5], [0, 2.5, 1]]})
            b.send_json({"t": "plane", "id": "evil", "kind": "wall", "corners": [[1e9, 0, 0], [0, 0, 0], [0, 1, 0]]})  # rejected
            b.send_json({"t": "pose", "pos": ["x"], "fwd": [0, 0, 1]})  # rejected
            b.send_bytes(b"\x01\xff\xd8fakejpeg\xff\xd9")
            pts = [(0.5, 1.0, 2.0, 255, 0, 0), (0.51, 1.0, 2.0, 255, 0, 0), (2.0, 1.0, 2.0, 0, 255, 0), (5.0, 1.0, 5.0, 9, 9, 9)]
            b.send_bytes(pack_points(pts))
            b.send_bytes(pack_points(pts[:3]))  # seen twice -> confirmed; the lone (5,1,5) speck never is

            got_points = False
            for _ in range(80):
                m = a.receive()
                if m.get("bytes"):
                    kind, layer, size = struct.unpack("<BBf", m["bytes"][:6])
                    assert kind == 2 and layer == 1 and abs(size - 0.15) < 1e-6  # phones get the coarse layer
                    got_points = True
                elif m.get("text") and '"t":"state"' in m["text"] and got_points:
                    import json
                    st = json.loads(m["text"])
                    bob_state = next(p for p in st["peers"] if p["id"] == bob["device_id"])
                    if bob_state.get("walls_between") == 1:
                        break
            else:
                raise AssertionError("no state with walls_between / no coarse points")
            assert bob_state["name"] == "Bob"
            assert phone.get(f"/api/frame/{bob['device_id']}.jpg", headers=ha).content.startswith(b"\xff\xd8")
            assert "evil" not in str(A.world.planes)
            assert len(A.world.voxels) == 2  # nearby points merged into one 5 cm voxel; the single speck isn't published
            assert (100, 20, 100) not in A.world.voxels
            assert admin.get("/api/admin/state").json()["phones"]

            # Alice closes her app: her pairing ends and the token is useless, but her map stays
            a.close()
            import time as _t
            for _ in range(50):
                if alice["device_id"] not in A.registry.devices:
                    break
                _t.sleep(0.02)
            assert alice["device_id"] not in A.registry.devices
            assert phone.get(f"/api/frame/{bob['device_id']}.jpg", headers=ha).status_code == 401

            # revoke Bob: his token stops working immediately
            assert admin.delete(f"/api/admin/device/{bob['device_id']}", headers={"X-Sightline": "1"}).json()["ok"]
            assert phone.get(f"/api/frame/{alice['device_id']}.jpg", headers=hb).status_code == 401


def test_rate_limiter():
    r = RateLimiter(rate=1, burst=3)
    assert [r.allow("k") for _ in range(5)] == [True, True, True, False, False]


def test_unused_pairing_expires(monkeypatch):
    admin, phone = admin_client(), TestClient(A.phone_app)
    d = pair(phone, admin, "Ghost")
    monkeypatch.setattr("sightline.security.CONNECT_TTL", -1)  # pretend 60 s passed without connecting
    assert phone.get("/api/frame/x.jpg", headers={"Authorization": f"Bearer {d['token']}"}).status_code == 401
    assert d["device_id"] not in A.registry.devices


def test_merge_walls():
    from sightline.geometry import merge_walls
    pieces = [((0, 0), (2, 0), 0, 2.4, "wall"), ((1.8, 0.05), (4, 0.04), 0.1, 2.5, "vertical"), ((4.2, -0.02), (5, 0), 0, 2.3, "wall"),
              ((0, 3), (2, 3), 0, 2.4, "wall"), ((6, 0), (6.9, 0), 0, 2, "door")]
    merged = merge_walls(pieces)
    walls = [m for m in merged if m[4] == "wall"]
    assert len(walls) == 2  # one long wall along z=0 (0..5 m) + the parallel wall at z=3
    long = max(walls, key=lambda m: abs(m[1][0] - m[0][0]))
    assert abs(min(long[0][0], long[1][0]) - 0) < 0.05 and abs(max(long[0][0], long[1][0]) - 5) < 0.05
    assert any(m[4] == "door" for m in merged)


def test_phone_map_access_is_read_only():
    admin = admin_client()
    with TestClient(A.phone_app) as phone:
        d = pair(phone, admin, "Ann")
        h = {"Authorization": f"Bearer {d['token']}"}
        assert phone.get("/map").status_code == 401
        assert phone.get("/static/app.js").status_code == 401
        assert phone.get("/map", headers=h).status_code == 200
        assert phone.get("/static/view3d.js", headers=h).status_code == 200
        assert phone.get("/static/../sightline/app.py", headers=h).status_code == 404  # no path traversal
        with phone.websocket_connect("/ws/phone", headers=h) as main:
            wait_for(main, "welcome")
            with phone.websocket_connect("/ws/viewer", headers=h) as view:
                snap = view.receive_json()
                assert snap["t"] == "snapshot" and snap["me"] == d["device_id"]
                view.send_json({"t": "pin", "kind": "hazard", "label": "smoke", "pos": [1, 0, 2]})
                for _ in range(80):
                    m = view.receive()
                    if m.get("text") and '"label":"smoke"' in m["text"]:
                        break
                else:
                    raise AssertionError("pin from the phone map never appeared")
        # the admin listener's endpoints don't exist on the phone listener at all
        assert phone.post("/api/admin/reset", headers=h).status_code == 404


def test_straighten_walls_snaps_and_drops_noise():
    from sightline.geometry import straighten_walls
    walls = [
        ((0, 0), (5, 0.3), -1.4, 1.0, "wall"),  # 3.4 degrees off the room axis -> snapped straight
        ((0, 0), (0.2, 4), -1.4, 1.0, "wall"),
        ((1, 1), (2, 1.9), -1.4, 1.0, "wall"),  # short diagonal stray line -> dropped
        ((3, 2), (3.4, 2), -1.4, 1.0, "wall"),  # 0.4 m wall fragment -> dropped
        ((2, 0), (2.9, 0), -1.4, 1.0, "door"),  # doors stay even if short
    ]
    out = straighten_walls(walls)
    kinds = sorted(w[4] for w in out)
    assert kinds == ["door", "wall", "wall"]
    for p, q, *_ in out:
        dx, dz = abs(q[0] - p[0]), abs(q[1] - p[1])
        assert min(dx, dz) < 0.05 * max(dx, dz)  # every wall runs along the room's axes


def test_saved_maps_roundtrip_and_delete(tmp_path):
    from cryptography.fernet import Fernet
    from sightline.maps import MapStore
    A.maps = MapStore(Fernet(Fernet.generate_key()), tmp_path / "maps.db")
    A.world.upsert_plane("ph-x", "p1", "wall", [[0, -1.4, 0], [3, -1.4, 0], [3, 1, 0], [0, 1, 0]])
    A.world.add_pin("hazard", "gas", [1, 0, 2], "Command")
    A.world.coverage.seen[(1, 2)] = 3
    admin, h = admin_client(), {"X-Sightline": "1"}
    saved = admin.post("/api/admin/maps?name=Ground floor", headers=h).json()
    assert saved["name"] == "Ground floor"
    assert b"gas" not in (tmp_path / "maps.db").read_bytes()  # encrypted at rest
    A.world.reset()
    assert admin.post(f"/api/admin/maps/{saved['id']}/load", headers=h).json()["ok"]
    assert A.world.merged_walls() and A.world.coverage.seen == {(1, 2): 3}
    assert [p["label"] for p in A.world.pins.values()] == ["gas"]
    assert [m["name"] for m in admin.get("/api/admin/maps").json()] == ["Ground floor"]
    assert admin.delete(f"/api/admin/maps/{saved['id']}", headers=h).json()["ok"]
    assert admin.get("/api/admin/maps").json() == []


def test_straighten_keeps_real_angled_walls_and_pillars():
    import math
    from sightline.geometry import straighten_walls
    walls = [((0, 0), (6, 0), -1.4, 1.0, "wall"), ((0, 0), (0, 5), -1.4, 1.0, "wall"),
             ((3, 2), (3 + 1.8 * math.cos(0.52), 2 + 1.8 * math.sin(0.52)), -1.4, 1.0, "wall")]  # 1.8 m at 30 degrees
    walls += [((4, 3), (4.4, 3), -1.4, 1.0, "pillar"), ((4.4, 3), (4.4, 3.4), -1.4, 1.0, "pillar")]
    out = straighten_walls(walls)
    assert any(abs(math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 90 - 30) < 2 for p, q, *_ in out)  # bay / slanted wall kept
    assert sum(1 for w in out if w[4] == "pillar") == 2  # a pillar's short sides survive the clean-up


def test_other_computer_needs_one_time_link_and_can_be_revoked():
    mac = TestClient(A.admin_app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    mac.cookies.set(A.ADMIN_COOKIE, A.registry.admin_token)
    h = {"X-Sightline": "1"}
    other = TestClient(A.admin_app, base_url="https://10.0.0.5:8444", client=("10.0.0.9", 50001))
    assert other.get("/api/admin/state").status_code == 401  # on the Wi-Fi, but no link: nothing
    url = mac.post("/api/admin/screens", headers=h).json()["url"]
    code = url.split("screen=")[1]
    r = other.get(f"/?screen={code}", follow_redirects=False)
    assert r.status_code == 303 and "secure" in r.headers["set-cookie"].lower()
    other.cookies.set(A.ADMIN_COOKIE, r.cookies[A.ADMIN_COOKIE])
    assert other.get("/api/admin/state").status_code == 200
    third = TestClient(A.admin_app, base_url="https://10.0.0.5:8444", client=("10.0.0.10", 50002))
    assert third.get(f"/?screen={code}", follow_redirects=False).status_code == 401  # links work once
    assert other.post("/api/admin/screens", headers=h).status_code == 403  # only the Mac adds computers
    sid = mac.get("/api/admin/screens").json()[0]["id"]
    assert mac.delete(f"/api/admin/screens/{sid}", headers=h).json()["ok"]
    assert other.get("/api/admin/state").status_code == 401


def test_ask_gemini_uses_live_situation_and_keeps_key_on_laptop(monkeypatch, tmp_path):
    import sightline.gemini as G
    monkeypatch.setattr(G, "DATA", tmp_path)  # never the laptop's real key, never the real Gemini
    admin, h = admin_client(), {"X-Sightline": "1"}
    q = {"messages": [{"role": "user", "text": "Where is Alice?"}]}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert admin.post("/api/admin/ask", json=q, headers=h).status_code == 503  # no key yet: clear message
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    a = A.world.phone("a", "Alice")
    a.pos, a.connected, a.pose_ts = [2.0, 0.0, 5.0], True, 1e12
    seen = {}

    async def fake_ask(messages, context):
        seen["context"], seen["messages"] = context, messages
        return "Alice is 5.4 m from the entrance."

    monkeypatch.setattr(A.gemini, "ask", fake_ask)
    r = admin.post("/api/admin/ask", json=q, headers=h)
    assert r.status_code == 200 and "Alice" in r.json()["answer"]
    assert "Alice: at (2.0, 5.0)" in seen["context"] and seen["messages"][-1]["text"] == "Where is Alice?"
    assert "test-key-123" not in r.text and "test-key-123" not in admin.get("/api/admin/ask").text
    bad = {"messages": [{"role": "model", "text": "hi"}]}
    assert admin.post("/api/admin/ask", json=bad, headers=h).status_code == 400  # must end with a question


def test_gemini_key_setup_only_from_the_mac_and_owner_only(tmp_path, monkeypatch):
    import stat
    import sightline.gemini as G
    monkeypatch.setattr(A, "DATA", tmp_path)
    monkeypatch.setattr(G, "DATA", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    h = {"X-Sightline": "1"}
    mac = TestClient(A.admin_app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    mac.cookies.set(A.ADMIN_COOKIE, A.registry.admin_token)
    other = TestClient(A.admin_app, base_url="https://10.0.0.5:8444", client=("10.0.0.9", 50001))
    other.cookies.set(A.ADMIN_COOKIE, A.registry.admin_token)
    key = "AIza" + "x" * 35

    async def fake_check():
        return True, "Key works."

    monkeypatch.setattr(A.gemini, "check", fake_check)
    assert other.post("/api/admin/ask/key", json={"key": key}, headers=h).status_code == 403
    assert mac.post("/api/admin/ask/key", json={"key": "nope"}, headers=h).status_code == 400
    r = mac.post("/api/admin/ask/key", json={"key": f"  {key[:20]}\n{key[20:]} "}, headers=h)  # copied with stray whitespace
    assert r.status_code == 200 and r.json()["ok"] and key not in r.text
    f = tmp_path / "gemini.key"
    assert f.read_text() == key and stat.S_IMODE(f.stat().st_mode) == 0o600
    assert mac.get("/api/admin/ask").json()["configured"] and key not in mac.get("/api/admin/ask").text


def test_regularize_merges_faces_drops_crossings_and_closes_corners():
    import math
    from sightline.geometry import regularize_walls
    segs = [((0, 0), (8, 0), -1.4, 1.0, "wall"), ((0, 0.25), (8, 0.25), -1.4, 1.0, "wall"),  # two faces of one wall
            ((8.3, 0.4), (8.3, 6), -1.4, 1.0, "wall"),  # stops 40 cm short of the corner
            ((0, 6), (8, 6), -1.4, 1.0, "wall"), ((0, 0), (0, 6), -1.4, 1.0, "wall"),
            ((3, 2.6), (3, 4.0), -1.4, 1.0, "wall"), ((1.5, 3.3), (4.5, 3.3), -1.4, 1.0, "wall")]  # an X: the short one is noise
    out = [w for w in regularize_walls(segs) if w[4] == "wall"]
    horiz = [w for w in out if abs(w[0][1] - w[1][1]) < 0.02 and w[0][1] < 1]
    assert len(horiz) == 1 and abs(horiz[0][0][1] - 0.125) < 0.02  # one line, between the faces
    right = [w for w in out if abs(w[0][0] - w[1][0]) < 0.02 and w[0][0] > 7]
    assert right and min(right[0][0][1], right[0][1][1]) < 0.2  # extended to meet the bottom wall
    assert len(out) == 5  # the crossing pair lost its shorter line, nothing else


def test_phones_cannot_drop_or_remove_waypoints():
    from sightline import protocol as P
    import pytest as _pt
    with _pt.raises(Exception):
        P.PhoneMessage.validate_python({"t": "pin", "kind": "waypoint", "label": "", "pos": [0, 0, 1]})  # phone protocol
    assert P.AdminPin.model_validate({"t": "pin", "kind": "waypoint", "pos": [0, 0, 1]}).kind == "waypoint"  # commander
