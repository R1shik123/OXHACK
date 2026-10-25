// FindMyTeam commander view. World frame: x right, y up, z out of the marker wall into the building.
// Top-down map draws x to the right and z downward, so the entrance wall sits at the top.
// Served only on 127.0.0.1 with the admin cookie; mutating requests carry X-Sightline (CSRF guard).
import { View3D } from "./view3d.js";

const $ = (id) => document.getElementById(id);
// Inside the iPhone app's Map tab, data comes through the app's pinned connection (no network from this page).
const BRIDGE = window.webkit?.messageHandlers?.sightline;
if (BRIDGE) document.body.classList.add("phone");
const canvas = $("map");
const ctx = canvas.getContext("2d");
const view3d = new View3D($("map3d"));
const CELL = 0.2; // metres; 2D obstacle cells built from the 3D points

const S = {
  phones: new Map(), planes: new Map(), pins: [],
  selected: null, follow: false, trails: true, pinMode: null,
  cam: { x: 0, z: 4, scale: 40 }, autoFit: true, connected: false, lastMsg: 0,
  mode: "2d", cells: new Map(), floorY: -1.4, pointCount: 0, walls: [],
  objects: [], events: [], started: null, meshBlocks: 0, recording: {}, plan: { people: {}, search: {}, decisions: [] },
  cov: new Map(), covCell: 0.25, covBbox: null, covPct: null, showCov: true, knownPins: new Set(), alertKeys: new Set(),
};

const KIND_STYLE = {
  wall: { color: "#e6edf3", width: 4 },
  vertical: { color: "#8b98a6", width: 3 },
  door: { color: "#22c46f", width: 5, dash: [6, 4] },
  window: { color: "#3fa7ff", width: 4, dash: [2, 3] },
};
const FLOORISH = new Set(["floor", "horizontal", "table", "seat", "ceiling"]);
const PIN_STYLE = { rally: "#22c46f", hazard: "#ff3d47", victim: "#ffb400", note: "#c77dff" };

// ---------- networking ----------
function connect() {
  if (BRIDGE) {
    window.__sl = (m) => onMessage(m);
    window.__slb = (b64) => {
      const bin = atob(b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      onPoints(buf.buffer);
    };
    window.__slstatus = (ok) => setConn(ok);
    S.ws = { readyState: 1, send: (text) => BRIDGE.postMessage(text) };
    BRIDGE.postMessage(JSON.stringify({ t: "ready" }));
    return;
  }
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/view`);
  ws.binaryType = "arraybuffer";
  S.ws = ws;
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connect, 1000); };
  ws.onmessage = (e) => (typeof e.data === "string" ? onMessage(JSON.parse(e.data)) : onPoints(e.data));
}
function setConn(ok) {
  S.connected = ok;
  $("conn").textContent = ok ? "live" : "offline";
  $("conn").className = ok ? "" : "bad";
}
async function api(method, path) {
  const r = await fetch(path, { method, headers: { "X-Sightline": "1" }, credentials: "same-origin" });
  if (r.status === 401) location.reload(); // admin cookie expired
  return r.json();
}

// Photo textures for walls / floor (0x06): 3D view only; the 2D plan keeps the scan colours
function onTexture(buf) { view3d.addTexture(buf); }

// Points arrive as voxels: 0x02 | layer | size f32 | N x (int16 x,y,z, uint8 r,g,b)
function onPoints(buf) {
  S.lastMsg = performance.now();
  if (!S.connected) setConn(true); // data flowing again after a stall
  const type = new DataView(buf).getUint8(0);
  if (type === 0x04 || type === 0x07) { view3d.addMesh(buf); S.meshBlocks = view3d.meshes.size; return; }
  if (type === 0x06) { onTexture(buf); return; }
  view3d.addVoxels(buf);
  const dv = new DataView(buf);
  const size = dv.getFloat32(2, true);
  for (let o = 6; o + 9 <= buf.byteLength; o += 9) {
    S.pointCount++;
    const y = dv.getInt16(o + 2, true) * size;
    if (y < S.floorY + 0.15 || y > S.floorY + 1.9 || S.cells.size > 60000) continue; // keep obstacles, not floor/ceiling
    const key = Math.round((dv.getInt16(o, true) * size) / CELL) + "," + Math.round((dv.getInt16(o + 4, true) * size) / CELL);
    if (!S.cells.has(key)) S.cells.set(key, `rgb(${dv.getUint8(o + 6)},${dv.getUint8(o + 7)},${dv.getUint8(o + 8)})`);
  }
}

function send(m) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(m)); }

function onMessage(m) {
  S.lastMsg = performance.now();
  if (!S.connected) setConn(true); // data flowing again after a stall
  if (m.t === "snapshot") {
    S.planes.clear();
    view3d.clear();
    S.cells.clear();
    S.pointCount = 0;
    for (const p of m.planes) { S.planes.set(p.id, withSegment(p)); view3d.upsertPlane(p); }
    if (m.lan_ip) $("addr").textContent = `${m.lan_ip}:${m.phone_port}`;
    S.objects = m.objects || [];
    S.events = m.events || [];
    if (m.me && !S.me) { S.me = m.me; S.selected = m.me; S.follow = true; $("follow").classList.add("on"); }
    S.started = m.started;
    renderTimeline();
    updateFloor();
  }
  if (m.t === "snapshot" || m.t === "state") {
    const seen = new Set();
    for (const p of m.phones) { seen.add(p.id); S.phones.set(p.id, p); }
    for (const id of S.phones.keys()) if (!seen.has(id)) S.phones.delete(id);
    S.pins = m.pins;
    if (m.coverage_pct != null) S.covPct = m.coverage_pct;
    if (m.recording) { S.recording = m.recording; renderRecording(); }
    if (m.alert_stats) S.alertStats = m.alert_stats;
    if (m.plan) { S.plan = m.plan; renderDecisions(); view3d.setRoutes(S.plan, S.phones); renderRoute(); }
    if (m.budget_s && document.activeElement !== $("budget")) $("budget").value = Math.round(m.budget_s / 60);
    notifyNewPins();
    renderAlerts();
    view3d.updatePhones([...S.phones.values()], S.selected);
    view3d.updatePins(S.pins);
    renderList();
  } else if (m.t === "coverage") {
    if (m.reset) S.cov.clear();
    S.covCell = m.cell;
    S.covBbox = m.bbox;
    S.covPct = m.pct;
    for (const [ix, iz, n] of m.seen) S.cov.set(ix + "," + iz, n);
    view3d.setCoverage(S.cov, S.covCell, S.covBbox, S.floorY);
  } else if (m.t === "textures") {
    view3d.keepTextures(m.ids);
  } else if (m.t === "objects") {
    S.objects = m.objects; // only the very large ones are drawn (plain blocks, never labelled)
  } else if (m.t === "events") {
    S.events.push(...m.events);
    renderTimeline();
  } else if (m.t === "walls") {
    S.walls = m.walls;
    S.stairs = m.stairs || [];
    S.risk = m.risk || [];
    view3d.setZones(S.risk);
    view3d.setWalls(m.walls);
    view3d.setStairs(S.stairs, S.floorY);
  } else if (m.t === "planes") {
    for (const p of m.upsert) { S.planes.set(p.id, withSegment(p)); view3d.upsertPlane(p); }
    for (const id of m.remove) { S.planes.delete(id); view3d.removePlane(id); }
    updateFloor();
  }
}

// Floor height = median of the floor planes (marker is ~1.4 m above it by default).
function updateFloor() {
  const ys = [...S.planes.values()].filter((p) => p.kind === "floor").map((p) => p.corners[0][1]).sort((a, b) => a - b);
  if (ys.length) S.floorY = ys[Math.floor(ys.length / 2)];
  view3d.setFloor(S.floorY);
}

// Footprint of a vertical plane = its two most distant corners in xz (same as server/geometry.py).
function withSegment(p) {
  let best = -1;
  for (let i = 0; i < p.corners.length; i++)
    for (let j = i + 1; j < p.corners.length; j++) {
      const a = p.corners[i], b = p.corners[j];
      const d = (a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2;
      if (d > best) { best = d; p.seg = [a, b]; }
    }
  return p;
}

// ---------- view transform ----------
const W = () => canvas.clientWidth, H = () => canvas.clientHeight;
const toScreen = (x, z) => [W() / 2 + (x - S.cam.x) * S.cam.scale, H() / 2 + (z - S.cam.z) * S.cam.scale];
const toWorld = (sx, sy) => [S.cam.x + (sx - W() / 2) / S.cam.scale, S.cam.z + (sy - H() / 2) / S.cam.scale];

function fit() {
  if (!W() || !H()) return;
  const xs = [0], zs = [0];
  for (const p of S.phones.values()) if (p.pos) { xs.push(p.pos[0]); zs.push(p.pos[2]); }
  for (const p of S.planes.values()) for (const c of p.corners) { xs.push(c[0]); zs.push(c[2]); }
  const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
  S.cam.x = (x0 + x1) / 2; S.cam.z = (z0 + z1) / 2;
  S.cam.scale = Math.max(8, Math.min(120, Math.min(W() / (x1 - x0 + 4), H() / (z1 - z0 + 4))));
}

// ---------- drawing ----------
function draw() {
  requestAnimationFrame(draw);
  if (S.mode === "3d") { view3d.render(); return; }
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== W() * dpr || canvas.height !== H() * dpr) { canvas.width = W() * dpr; canvas.height = H() * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#070708"; ctx.fillRect(0, 0, W(), H());

  if (S.follow && S.selected) {
    const p = S.phones.get(S.selected);
    if (p && p.pos) { S.cam.x += (p.pos[0] - S.cam.x) * 0.15; S.cam.z += (p.pos[2] - S.cam.z) * 0.15; }
  }
  if (S.autoFit && !S.follow) fit();
  drawGrid();
  drawFloors();
  if (S.showCov) drawCoverage();
  drawLarge();
  drawStairs();
  drawZones();
  drawLabels();
  drawWalls();
  drawMarker();
  if (S.trails) for (const p of S.phones.values()) drawTrail(p);
  for (const p of S.phones.values()) drawXray(p);
  drawPlan();
  for (const pin of S.pins) drawPin(pin);
  for (const p of S.phones.values()) drawPhone(p);
  $("scale").textContent = `1 m = ${S.cam.scale.toFixed(0)} px · grid 1 m`;
  if (S.connected && performance.now() - S.lastMsg > 2000) setConn(false);
}

function drawGrid() {
  const [wx0, wz0] = toWorld(0, 0), [wx1, wz1] = toWorld(W(), H());
  const step = S.cam.scale < 15 ? 5 : 1;
  ctx.lineWidth = 1;
  for (let x = Math.floor(wx0 / step) * step; x <= wx1; x += step) {
    const [sx] = toScreen(x, 0);
    ctx.strokeStyle = x % 5 === 0 ? "#1d1d22" : "#121215";
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, H()); ctx.stroke();
  }
  for (let z = Math.floor(wz0 / step) * step; z <= wz1; z += step) {
    const [, sy] = toScreen(0, z);
    ctx.strokeStyle = z % 5 === 0 ? "#1d1d22" : "#121215";
    ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(W(), sy); ctx.stroke();
  }
}

function drawFloors() {
  ctx.fillStyle = "rgba(42, 52, 66, 0.55)";
  for (const p of S.planes.values()) {
    if (!FLOORISH.has(p.kind) || p.kind === "ceiling") continue;
    ctx.beginPath();
    p.corners.forEach((c, i) => { const [x, y] = toScreen(c[0], c[2]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.fill();
  }
}

// Search coverage: green = looked at (brighter = more looks), red = inside the building outline but never seen.
function drawCoverage() {
  const c = S.covCell, s = Math.max(1, c * S.cam.scale);
  if (S.covBbox) {
    const [x0, z0, x1, z1] = S.covBbox;
    ctx.fillStyle = "rgba(255, 70, 70, 0.16)";
    for (let ix = Math.floor(x0 / c); ix <= Math.floor(x1 / c); ix++)
      for (let iz = Math.floor(z0 / c); iz <= Math.floor(z1 / c); iz++) {
        if (S.cov.has(ix + "," + iz)) continue;
        const [x, y] = toScreen(ix * c, iz * c);
        ctx.fillRect(x, y, s + 0.5, s + 0.5);
      }
  }
  for (const [key, n] of S.cov) {
    const [ix, iz] = key.split(",");
    const [x, y] = toScreen(ix * c, iz * c);
    if (x < -s || y < -s || x > W() + s || y > H() + s) continue;
    ctx.fillStyle = `rgba(46, 204, 113, ${Math.min(0.42, 0.12 + n * 0.015)})`;
    ctx.fillRect(x, y, s + 0.5, s + 0.5);
  }
}

// Rooms stay hollow: only very large obstructions (big table, sofa, cabinet run) as plain grey blocks.
function drawLarge() {
  ctx.fillStyle = "rgba(120, 120, 130, 0.55)";
  ctx.strokeStyle = "rgba(190, 190, 200, 0.8)";
  ctx.lineWidth = 1.5;
  for (const o of S.objects || []) {
    if (!o.large) continue;
    const [x0, y0] = toScreen(o.x0, o.z0), [x1, y1] = toScreen(o.x1, o.z1);
    ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  }
}

// Stairs: amber outline, step lines across, arrow the way you go from this floor (up or down). No text.
function drawStairs() {
  for (const st of S.stairs || []) {
    const pts = st.c.map(([x, z]) => toScreen(x, z));
    ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
    ctx.fillStyle = "rgba(255, 176, 32, 0.16)"; ctx.fill();
    ctx.strokeStyle = "#ffb020"; ctx.lineWidth = 2; ctx.stroke();
    const [a, b, c, d] = st.c; // a-b and d-c run uphill
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]), n = Math.max(2, Math.round(len / 0.28));
    ctx.lineWidth = 1.2; ctx.beginPath();
    for (let i = 1; i < n; i++) {
      const t = i / n;
      const [px, py] = toScreen(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), [qx, qy] = toScreen(d[0] + (c[0] - d[0]) * t, d[1] + (c[1] - d[1]) * t);
      ctx.moveTo(px, py); ctx.lineTo(qx, qy);
    }
    ctx.stroke();
    const lo = [(a[0] + d[0]) / 2, (a[1] + d[1]) / 2], hi = [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2];
    const [sx, sy] = toScreen(...(st.down ? hi : lo)), [ex, ey] = toScreen(...(st.down ? lo : hi));
    const ang = Math.atan2(ey - sy, ex - sx);
    ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey);
    ctx.lineTo(ex - 9 * Math.cos(ang - 0.5), ey - 9 * Math.sin(ang - 0.5)); ctx.moveTo(ex, ey);
    ctx.lineTo(ex - 9 * Math.cos(ang + 0.5), ey - 9 * Math.sin(ang + 0.5)); ctx.stroke();
  }
}

function drawCells() {
  const s = Math.max(1, CELL * S.cam.scale);
  ctx.globalAlpha = 0.7;
  for (const [key, color] of S.cells) {
    const [cx, cz] = key.split(",");
    const [x, y] = toScreen(cx * CELL, cz * CELL);
    if (x < -s || y < -s || x > W() + s || y > H() + s) continue;
    ctx.fillStyle = color;
    ctx.fillRect(x - s / 2, y - s / 2, s, s);
  }
  ctx.globalAlpha = 1;
}

// Clean walls merged on the server from every phone's plane pieces: [x1, z1, x2, z2, y0, y1, kind]
function drawWalls() {
  ctx.lineCap = "round";
  for (const [x1, z1, x2, z2, , , kind] of S.walls) {
    const st = KIND_STYLE[kind];
    if (!st) continue;
    const [ax, ay] = toScreen(x1, z1), [bx, by] = toScreen(x2, z2);
    ctx.strokeStyle = st.color; ctx.lineWidth = st.width; ctx.setLineDash(st.dash || []);
    ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  ctx.setLineDash([]); ctx.globalAlpha = 1;
}

function drawMarker() {
  const [x, y] = toScreen(0, 0);
  const w = 0.18 * S.cam.scale;
  ctx.fillStyle = "#ff3d47";
  ctx.fillRect(x - Math.max(w, 6) / 2, y - 3, Math.max(w, 6), 6);
  ctx.strokeStyle = "#ff3d47"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 18); ctx.lineTo(x - 5, y + 12); ctx.moveTo(x, y + 18); ctx.lineTo(x + 5, y + 12); ctx.stroke();
  label("ENTRANCE MARKER", x, y - 10, "#ff3d47", "center");
}

// Trails: drop points closer than 20 cm, then two rounds of corner cutting (Chaikin) -> smooth, not zig-zag
function smoothPath(pts) {
  const out = [];
  for (const p of pts) if (!out.length || Math.hypot(p[0] - out[out.length - 1][0], p[2] - out[out.length - 1][2]) >= 0.2) out.push(p);
  let s = out;
  for (let k = 0; k < 2 && s.length > 2; k++) {
    const n = [s[0]];
    for (let i = 0; i < s.length - 1; i++) {
      const a = s[i], b = s[i + 1];
      n.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1], 0.75 * a[2] + 0.25 * b[2]],
             [0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1], 0.25 * a[2] + 0.75 * b[2]]);
    }
    n.push(s[s.length - 1]);
    s = n;
  }
  return s;
}

function drawTrail(p) {
  if (!p.trail || p.trail.length < 2) return;
  ctx.strokeStyle = p.color; ctx.globalAlpha = 0.35; ctx.lineWidth = 2; ctx.lineJoin = "round";
  ctx.beginPath();
  smoothPath(p.trail).forEach((c, i) => { const [x, y] = toScreen(c[0], c[2]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke(); ctx.globalAlpha = 1;
}

function drawXray(p) {
  const t = p.xray && S.phones.get(p.xray);
  if (!t || !p.pos || !t.pos) return;
  const [ax, ay] = toScreen(p.pos[0], p.pos[2]), [bx, by] = toScreen(t.pos[0], t.pos[2]);
  ctx.strokeStyle = "#c77dff"; ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
  ctx.lineDashOffset = -performance.now() / 40;
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  ctx.setLineDash([]); ctx.lineDashOffset = 0;
  if (Math.hypot(bx - ax, by - ay) > 90) label(`${p.name} 👁 ${t.name}`, ax + (bx - ax) * 0.6, ay + (by - ay) * 0.6 - 8, "#c77dff", "center");
}

function polyline(pts, color, dash, width = 2.5) {
  if (!pts || pts.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  ctx.beginPath();
  pts.forEach(([x, z], i) => { const [sx, sy] = toScreen(x, z); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
  ctx.stroke(); ctx.setLineDash([]);
}

// Fall-risk (yellow) and hazard (red) zones: soft areas, no text. Red rings sit around hazard pins.
function drawZones() {
  ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5;
  for (const z of S.risk || []) {
    ctx.beginPath();
    if (z.poly) z.poly.forEach(([x0, z0], i) => { const [x, y] = toScreen(x0, z0); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    else { const [x, y] = toScreen(z.x, z.z); ctx.arc(x, y, z.r * S.cam.scale, 0, 7); }
    ctx.closePath();
    ctx.fillStyle = "rgba(255, 204, 51, 0.13)"; ctx.fill();
    ctx.strokeStyle = "rgba(255, 204, 51, 0.7)"; ctx.stroke();
  }
  for (const pin of S.pins) {
    if (pin.kind !== "hazard") continue;
    const [x, y] = toScreen(pin.pos[0], pin.pos[2]);
    ctx.beginPath(); ctx.arc(x, y, 1.5 * S.cam.scale, 0, 7);
    ctx.fillStyle = "rgba(255, 61, 71, 0.12)"; ctx.fill();
    ctx.strokeStyle = "rgba(255, 61, 71, 0.75)"; ctx.stroke();
  }
  ctx.setLineDash([]);
}

// Everyone's safest way back to command (always on), suggested rooms, the routes there, predicted positions.
function drawRoute() {
  for (const leg of S.plan.route?.legs || []) {
    polyline(leg.path, leg.blocked ? "#ff3d47aa" : "#18a6c9dd", leg.blocked ? [2, 6] : leg.zone ? [8, 5] : [], 3.5);
  }
}

function drawPlan() {
  drawRoute();
  for (const [a, b] of S.plan.inferred_walls || []) { // filled in by inference (probably wall): faint, dashed
    const [ax, ay] = toScreen(a[0], a[1]), [bx, by] = toScreen(b[0], b[1]);
    ctx.setLineDash([3, 4]); ctx.lineWidth = 2.5; ctx.strokeStyle = "rgba(230, 230, 240, 0.4)";
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.setLineDash([]);
  }
  for (const [id, info] of Object.entries(S.plan.people || {})) { // way back to command, in the responder's colour
    const p = S.phones.get(id);
    if (!info.home?.path?.length || !p?.connected) continue;
    polyline(info.home.path, (p.color || "#fff") + "dd", info.home.zone ? [3, 5] : [], 2.5); // dashed = crosses a risk area
    if (info.home.zone === 2) { const [hx, hz] = info.home.path[Math.floor(info.home.path.length / 2)]; const [x, y] = toScreen(hx, hz); label("⚠", x, y, "#ff3d47", "center"); }
  }
  const rooms = Object.fromEntries((S.plan.search?.rooms || []).map((r) => [r.id, r]));
  for (const d of S.plan.decisions || []) {
    if (!d.path || !["next_room", "research", "victim"].includes(d.kind)) continue;
    const col = d.kind === "victim" ? "#ffb400" : S.phones.get(d.who)?.color || "#22c46f";
    polyline(d.path, col + "99", [7, 5]);
    const r = rooms[d.room];
    if (r) { // outline the suggested room: solid for a new room, dotted for a re-check
      ctx.beginPath(); r.poly.forEach(([x0, z0], i) => { const [x, y] = toScreen(x0, z0); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath();
      ctx.setLineDash(d.kind === "research" ? [2, 4] : []); ctx.lineWidth = 2; ctx.strokeStyle = col + "aa"; ctx.stroke(); ctx.setLineDash([]);
    }
  }
  for (const [id, info] of Object.entries(S.plan.people || {})) {
    if (!info.predicted) continue;
    const p = S.phones.get(id), [x, y] = toScreen(info.predicted.x, info.predicted.z);
    ctx.strokeStyle = (p?.color || "#fff") + "aa"; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(x, y, Math.max(6, info.predicted.r * S.cam.scale), 0, 7); ctx.stroke(); ctx.setLineDash([]);
    label(`${p?.name || "?"} (predicted)`, x, y + 4, "#8b98a6", "center");
  }
}

function drawPin(pin) {
  const [x, y] = toScreen(pin.pos[0], pin.pos[2]);
  if (pin.kind === "waypoint") { // numbered cyan stop; ticked off when reached
    const done = (S.plan.route?.stops || []).find((s) => s.id === pin.id)?.reached;
    ctx.fillStyle = done ? "#2a3a40" : "#18a6c9"; ctx.strokeStyle = "#070708"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 10, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = done ? "#9fd" : "#031216"; ctx.font = "800 11px -apple-system, system-ui, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(done ? "✓" : pin.label || "?", x, y + 4); ctx.textAlign = "left";
    return;
  }
  const c = PIN_STYLE[pin.kind] || "#fff";
  ctx.fillStyle = c; ctx.strokeStyle = "#070708"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y - 9); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 9); ctx.lineTo(x - 7, y); ctx.closePath();
  ctx.fill(); ctx.stroke();
  label(`${pin.kind.toUpperCase()}${pin.label ? ": " + pin.label : ""}`, x + 11, y + 4, c, "left");
}

function drawPhone(p) {
  if (!p.pos) return;
  const [x, y] = toScreen(p.pos[0], p.pos[2]);
  const dim = p.stale || !p.connected;
  const col = dim ? "#59636e" : p.color;
  if (p.fwd) {
    const yaw = Math.atan2(p.fwd[2], p.fwd[0]); // screen angle (z down = +y)
    const r = 3 * S.cam.scale, half = (30 * Math.PI) / 180;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, col + "88"); g.addColorStop(1, col + "00");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, yaw - half, yaw + half); ctx.closePath(); ctx.fill();
  }
  if (S.selected === p.id) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 13, 0, 7); ctx.stroke(); }
  if (p.alert) { // expanding red rings while a fall / no-movement alert is open
    const t = (performance.now() / 900) % 1;
    ctx.strokeStyle = `rgba(255, 59, 48, ${p.alert.acked ? 0.5 : 1 - t})`; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, 12 + (p.alert.acked ? 8 : 30 * t), 0, 7); ctx.stroke();
  }
  ctx.fillStyle = col; ctx.strokeStyle = "#070708"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.fill(); ctx.stroke();
  label(p.name + (dim ? (p.connected ? " (stale)" : " (offline)") : ""), x, y - 16, dim ? "#8b98a6" : "#fff", "center");
}

function label(text, x, y, color, align) {
  ctx.font = "600 12px -apple-system, system-ui, sans-serif";
  ctx.textAlign = align;
  ctx.lineWidth = 4; ctx.strokeStyle = "#070708"; ctx.strokeText(text, x, y);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}

// ---------- sidebar ----------
function renderList() {
  const ul = $("phones");
  const phones = [...S.phones.values()].sort((a, b) => a.name.localeCompare(b.name));
  $("count").textContent = phones.length ? `(${phones.length})` : "";
  ul.replaceChildren(...phones.map((p) => {
    const li = document.createElement("li");
    li.className = S.selected === p.id ? "sel" : "";
    li.dataset.id = p.id;
    let badge = ["live", "live"];
    if (!p.connected) badge = ["off", "offline"];
    else if (!p.aligned) badge = ["stale", "scan marker"];
    else if (p.stale) badge = ["stale", "stale"];
    if (p.xray && p.connected) badge = ["xray", "x-ray → " + (S.phones.get(p.xray)?.name || "?")];
    if (p.alert) badge = ["alert", alertText(p.alert)];
    const meta = [];
    if (p.pos) meta.push(`(${p.pos[0].toFixed(1)}, ${p.pos[2].toFixed(1)}) m`, `${dist(p.pos).toFixed(1)} m from entrance`);
    if (p.activity?.state) meta.push(`${p.activity.state} (${p.activity.conf})`);
    if (p.activity?.phone_down) meta.push("phone lying still");
    if (p.health?.mem_mb > 0) meta.push(`${p.health.mem_mb} MB`);
    if (p.health?.thermal >= 1) meta.push(["", "warm", "HOT", "CRITICAL HEAT"][p.health.thermal]);
    const pl = S.plan.people?.[p.id];
    if (pl?.exit_m != null) meta.push(`exit ${pl.exit_m} m`);
    if (pl?.turnaround_s != null) meta.push(pl.turnaround_s <= 0 ? "TURN BACK" : `turn back in ${mmss(pl.turnaround_s)}`);
    if (pl?.alone_s) meta.push(`alone ${mmss(pl.alone_s)}`);
    if (p.wall_dist != null) meta.push(`wall ${p.wall_dist.toFixed(1)} m`);
    if (p.tracking && p.tracking !== "normal" && p.tracking !== "unknown") meta.push(`tracking: ${p.tracking}`);
    if (p.platform) meta.push(p.platform);
    if (p.battery != null) meta.push(`🔋${Math.round(p.battery * 100)}%`);
    li.innerHTML = `<span class="dot"></span><span class="name"></span><span class="badge ${badge[0]}"></span><span class="meta"></span>`;
    li.querySelector(".dot").style.background = p.color; // CSP forbids inline style attributes
    li.querySelector(".name").textContent = p.name;
    li.querySelector(".badge").textContent = badge[1];
    li.querySelector(".meta").textContent = meta.join(" · ") || "no position yet";
    return li;
  }));
  const walls = S.walls.filter((w) => w[6] === "wall").length;
  const doors = S.walls.filter((w) => w[6] === "door").length;
  $("stats").textContent = `${walls} walls · ${doors} doors · ${S.pins.length} pins` +
    (S.covPct != null ? ` · ${S.covPct}% searched` : "");
  updateFeed();
}
const dist = (p) => Math.hypot(p[0], p[2]);

// Delegated: the list is rebuilt every tick, so a per-item click handler could miss its mouseup.
$("phones").addEventListener("pointerdown", (e) => { const li = e.target.closest("li"); if (li) select(li.dataset.id); });

function select(id) {
  S.selected = S.selected === id ? null : id;
  if (S.selected && S.mode === "3d") view3d.focus(S.phones.get(S.selected));
  renderList();
}

function updateFeed() {
  const p = S.selected && S.phones.get(S.selected);
  $("feedbox").hidden = !p;
  if (!p) { $("feed").removeAttribute("src"); $("feed").dataset.id = ""; return; }
  $("feedtitle").textContent = `Live view · ${p.name}`;
  if ($("feed").dataset.id !== p.id) {
    $("feed").dataset.id = p.id;
    $("feed").src = BRIDGE ? `/api/frame/${encodeURIComponent(p.id)}.jpg?t=${Date.now()}` : `/api/admin/frame/${encodeURIComponent(p.id)}.mjpg`;
  }
  $("feedmeta").textContent = p.has_frame ? "" : "No camera frames from this phone yet.";
}

// ---------- interaction ----------
let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.offsetX, y: e.offsetY, moved: false, cx: S.cam.x, cz: S.cam.z };
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; canvas.classList.add("dragging"); }
  if (drag.moved) { setAutoFit(false); S.cam.x = drag.cx - dx / S.cam.scale; S.cam.z = drag.cz - dy / S.cam.scale; S.follow = false; $("follow").classList.remove("on"); }
});
canvas.addEventListener("pointerup", (e) => {
  canvas.classList.remove("dragging");
  if (drag && !drag.moved) click(e.offsetX, e.offsetY, e.shiftKey);
  drag = null;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  setAutoFit(false);
  const [wx, wz] = toWorld(e.offsetX, e.offsetY);
  S.cam.scale = Math.max(4, Math.min(300, S.cam.scale * Math.exp(-e.deltaY * 0.0015)));
  const [nx, nz] = toWorld(e.offsetX, e.offsetY);
  S.cam.x += wx - nx; S.cam.z += wz - nz;
}, { passive: false });

function click(sx, sy, shift) {
  const [wx, wz] = toWorld(sx, sy);
  const near = (pos) => Math.hypot((pos[0] - wx) * S.cam.scale, (pos[2] - wz) * S.cam.scale) < 14;
  const pin = S.pins.find((p) => near(p.pos));
  if (pin && shift) { send({ t: "pin_remove", id: pin.id }); return; }
  if (LB.on) { labelClick(wx, wz, sx, sy); return; }
  if (S.pinMode === "waypoint") { // keep dropping waypoints until Esc / the button again
    send({ t: "pin", kind: "waypoint", label: "", pos: [wx, 0, wz] });
    return;
  }
  if (S.pinMode) {
    const lbl = prompt(`Label for ${S.pinMode} (optional)`, "") ?? null;
    if (lbl !== null) send({ t: "pin", kind: S.pinMode, label: lbl, pos: [wx, 0, wz] });
    setPinMode(null);
    return;
  }
  const phone = [...S.phones.values()].find((p) => p.pos && near(p.pos));
  if (phone) select(phone.id);
}

function setPinMode(kind) {
  S.pinMode = kind;
  document.querySelectorAll("[data-pin]").forEach((b) => b.classList.toggle("on", b.dataset.pin === kind));
  canvas.classList.toggle("pinning", !!kind);
  $("hint").style.display = kind ? "block" : "none";
  $("hint").textContent = !kind ? "" : kind === "waypoint" ? "Click stops in the order to visit them · Esc when done · Shift-click a stop to remove it"
    : `Click the map to drop a ${kind} · Esc to cancel · Shift-click a pin to delete`;
}

document.querySelectorAll("[data-pin]").forEach((b) => (b.onclick = () => setPinMode(S.pinMode === b.dataset.pin ? null : b.dataset.pin)));
function setAutoFit(on) { S.autoFit = on; $("fit").classList.toggle("on", on); }
$("fit").onclick = () => setAutoFit(true);
$("follow").onclick = () => { S.follow = !S.follow; $("follow").classList.toggle("on", S.follow); };
$("trails").onclick = () => { S.trails = !S.trails; view3d.showTrails = S.trails; $("trails").classList.toggle("on", S.trails); };
$("pts").onclick = () => { const on = !$("pts").classList.contains("on"); $("pts").classList.toggle("on", on); view3d.setPoints(on); };

// ---------- decisions ----------
const mmss = (s) => `${Math.floor(Math.max(0, s) / 60)}:${String(Math.round(Math.max(0, s) % 60)).padStart(2, "0")}`;
function renderDecisions() {
  const ds = S.plan.decisions || [];
  const items = ds.map((d) => {
    const li = document.createElement("li");
    li.className = "p" + d.prio;
    const dot = document.createElement("span"); dot.className = "pr";
    const t = document.createElement("span"); t.textContent = d.text;
    li.append(dot, t);
    return li;
  });
  if (!items.length) { const li = document.createElement("li"); li.className = "empty"; li.textContent = "No actions suggested right now."; items.push(li); }
  $("decisions").replaceChildren(...items);
  const se = S.plan.search || {};
  const st = S.alertStats;
  if (st && st.confirmed + st.false_alarm > 0) {
    const li = document.createElement("li"); li.className = "empty";
    li.textContent = `Alert accuracy so far: ${st.confirmed} confirmed, ${st.false_alarm} false alarms`;
    $("decisions").append(li);
  }
  $("eta").textContent = se.eta_s != null ? (se.eta_s === 0 ? "search done" : `search ~95% in ${mmss(se.eta_s)}`) : "";
}
$("budget").onchange = () => api("POST", `/api/admin/budget?minutes=${encodeURIComponent($("budget").value)}`);

// ---------- timeline ----------
const clock = (ts) => {
  if (!S.started) return "";
  const s = Math.max(0, Math.round(ts - S.started));
  return `T+${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
function renderTimeline() {
  const list = S.events.slice(-60).reverse();
  $("timeline").replaceChildren(...list.map((e) => {
    const li = document.createElement("li");
    li.className = "ev " + e.kind;
    const t = document.createElement("span"); t.className = "when"; t.textContent = clock(e.ts);
    const x = document.createElement("span"); x.textContent = e.text;
    li.append(t, x);
    return li;
  }));
}

// ---------- recording (opt-in) ----------
function renderRecording() {
  const r = S.recording || {};
  const parts = [];
  if (r.capture) parts.push(`Room capture: ${r.capture.left}s left · ${(r.capture.bytes / 1e6).toFixed(1)} MB`);
  if (r.session) parts.push(`Recording session · ${Math.floor(r.session.seconds / 60)} min · ${(r.session.bytes / 1e6).toFixed(2)} MB`);
  $("recstatus").textContent = parts.join(" · ") || "Not recording. Nothing is saved unless you start a recording.";
  $("recbtn").textContent = r.session ? "Stop recording" : "Record session";
  $("recbtn").classList.toggle("danger", !!r.session);
}
$("capbtn").onclick = async () => {
  if (!confirm("Record a 90 s room capture (depth, poses, ~1 photo/s) to this laptop, encrypted, for tuning? Walk slowly around the room while it runs.")) return;
  const r = await api("POST", "/api/admin/capture/start?secs=90");
  if (r.detail) alert(r.detail);
};
$("recbtn").onclick = () => api("POST", `/api/admin/record/${S.recording.session ? "stop" : "start"}`);
// Coverage: on by default on the 2D map, off by default in 3D (it would tint the real colours).
$("cov").onclick = () => {
  if (S.mode === "3d") { S.showCov3d = !S.showCov3d; view3d.showCoverage(S.showCov3d); }
  else S.showCov = !S.showCov;
  syncCovButton(); renderList();
};
const syncCovButton = () => $("cov").classList.toggle("on", S.mode === "3d" ? !!S.showCov3d : S.showCov);

// ---------- alerts (fall / no movement) ----------
const SOURCE = { impact: "accelerometer impact", height: "height drop", stillness: "no movement 30 s" };
const alertText = (a) => `${a.kind === "fall" ? "POSSIBLE FALL" : "NO MOVEMENT"} · ${Math.round(a.for)} s` +
  (a.source ? ` · ${SOURCE[a.source] || a.source}` : "") + (a.acked ? " · acknowledged" : "");

function renderAlerts() {
  const open = [...S.phones.values()].filter((p) => p.alert);
  let fresh = false;
  $("alerts").replaceChildren(...open.map((p) => {
    const key = p.id + p.alert.kind;
    if (!p.alert.acked && !S.alertKeys.has(key)) { S.alertKeys.add(key); fresh = true; }
    const div = document.createElement("div");
    div.className = "alert" + (p.alert.acked ? " acked" : "");
    const span = document.createElement("span");
    span.textContent = `⚠ ${p.name}: ${alertText(p.alert)}`;
    div.append(span);
    const show = document.createElement("button");
    show.textContent = "Show";
    show.onpointerdown = () => { S.selected = p.id; S.follow = true; $("follow").classList.add("on"); renderList(); };
    div.append(show);
    if (!BRIDGE) {
      const conf = document.createElement("button");
      conf.textContent = "Confirm real";
      conf.title = "Tell the model this alert was real (training label)";
      conf.onpointerdown = () => api("POST", `/api/admin/confirm/${encodeURIComponent(p.id)}`);
      div.append(conf);
    }
    if (!p.alert.acked && !BRIDGE) {
      const ack = document.createElement("button");
      ack.textContent = "Acknowledge";
      ack.onpointerdown = () => api("POST", `/api/admin/ack/${encodeURIComponent(p.id)}`);
      div.append(ack);
    }
    return div;
  }));
  for (const key of [...S.alertKeys]) if (!key.startsWith("hz") && !open.some((p) => p.id + p.alert.kind === key)) S.alertKeys.delete(key);
  // Responders walking towards a hazard pin after the warning (they chose to proceed, or kept going)
  let freshHazard = false;
  for (const p of S.phones.values()) for (const h of p.hazard || []) {
    const key = "hz" + p.id + h.pin;
    if (!S.alertKeys.has(key)) { S.alertKeys.add(key); freshHazard = true; }
    const div = document.createElement("div");
    div.className = "alert hazard";
    const span = document.createElement("span");
    span.textContent = `⚠ ${p.name} is approaching hazard “${h.label}”` + (h.dist != null ? ` · ${h.dist} m` : "") + ` · ${h.why}`;
    const show = document.createElement("button");
    show.textContent = "Show";
    show.onpointerdown = () => { S.selected = p.id; S.follow = true; $("follow").classList.add("on"); renderList(); };
    div.append(span, show);
    $("alerts").append(div);
  }
  for (const key of [...S.alertKeys]) if (key.startsWith("hz") && ![...S.phones.values()].some((p) => (p.hazard || []).some((h) => "hz" + p.id + h.pin === key))) S.alertKeys.delete(key);
  if (freshHazard) beep();
  // Commander: the alarm keeps sounding until every alert is acknowledged. Phones (Map tab): one beep.
  const ringing = !BRIDGE && open.some((p) => !p.alert.acked);
  setAlarm(ringing);
  if (fresh && BRIDGE) beep();
  if (ringing && audio?.state !== "running") {
    const hint = document.createElement("div");
    hint.className = "alert muted";
    hint.textContent = "🔇 Click anywhere on this page to turn on the alarm sound";
    $("alerts").append(hint);
  }
}

// Browsers only allow sound after the page has been clicked once, so any click unlocks it.
let audio = null, alarmTimer = null;
document.addEventListener("pointerdown", () => {
  try { audio = audio || new AudioContext(); audio.resume().then(renderAlerts); } catch { /* no audio */ }
}, { capture: true });

function beep() {
  try {
    audio = audio || new AudioContext();
    for (const [i, f] of [[0, 1050], [1, 1400], [2, 1050], [3, 1400]]) {
      const o = audio.createOscillator(), g = audio.createGain();
      o.type = "square"; o.frequency.value = f; o.connect(g); g.connect(audio.destination);
      const t = audio.currentTime + i * 0.2;
      g.gain.setValueAtTime(0.12, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.19);
      o.start(t); o.stop(t + 0.2);
    }
  } catch { /* audio blocked until the page is clicked once */ }
}

function setAlarm(on) {
  if (on && !alarmTimer) { beep(); alarmTimer = setInterval(beep, 1500); }
  else if (!on && alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; }
}

function notifyNewPins() {
  for (const pin of S.pins) {
    if (S.knownPins.has(pin.id)) continue;
    S.knownPins.add(pin.id);
    if (pin.kind === "victim" && pin.by !== "Command") toast(`${pin.by} marked a victim${pin.label ? ": " + pin.label : ""}`);
  }
}

let toastTimer = null;
function toast(text) {
  $("toast").textContent = text;
  $("toast").style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").style.display = "none"), 5000);
  beep();
}
$("reset").onclick = async () => {
  if (!confirm("Clear all walls, 3D points, trails and pins from the shared map? Paired phones stay paired.")) return;
  await api("POST", "/api/admin/reset");
  S.planes.clear(); S.cells.clear(); S.pointCount = 0; S.walls = []; S.cov.clear(); S.covBbox = null; S.covPct = null;
  view3d.clear(); view3d.setWalls([]); view3d.setCoverage(S.cov, S.covCell, null, S.floorY); setAutoFit(true);
};
$("mode").onclick = () => {
  S.mode = S.mode === "2d" ? "3d" : "2d";
  view3d.active = S.mode === "3d";
  $("mode").textContent = S.mode === "2d" ? "3D view" : "2D map";
  $("map").hidden = S.mode === "3d";
  $("map3d").hidden = S.mode === "2d";
  document.querySelectorAll(".only2d").forEach((b) => (b.hidden = S.mode === "3d"));
  syncCovButton();
};

// ---------- pairing & devices ----------
let pairTimer = null;
$("pair").onclick = async () => {
  const r = await api("POST", "/api/admin/pair");
  $("qr").innerHTML = r.svg; // server-generated SVG (segno), no user data in it
  $("pairinfo").textContent = `Scan with the "Scan QR code" button in the FindMyTeam app. Server ${r.host} · cert ${r.fingerprint.slice(0, 16)}…`;
  $("apkbox").hidden = !r.apk;
  if (r.apk) {
    $("apkqr").innerHTML = r.apk.svg; // server-generated SVG
    $("apkinfo").textContent = `Scan with the Android camera app to download (valid while this pairing code is). ` +
      `Your browser will warn about the laptop's self-signed certificate: continue, then check the file's SHA-256 is ${r.apk.sha256}`;
  }
  $("pairbox").hidden = false;
  let left = r.expires_in;
  clearInterval(pairTimer);
  pairTimer = setInterval(() => {
    left -= 1;
    $("paircount").textContent = left > 0 ? `Single use · expires in ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}` : "Expired: make a new code";
    if (left <= 0) { clearInterval(pairTimer); $("qr").replaceChildren(); }
  }, 1000);
  refreshDevices();
};
$("pairclose").onclick = () => { $("pairbox").hidden = true; $("qr").replaceChildren(); clearInterval(pairTimer); };

async function refreshDevices() {
  const devices = await api("GET", "/api/admin/devices");
  $("devices").replaceChildren(...devices.map((d) => {
    const li = document.createElement("li");
    li.className = "device";
    const name = document.createElement("span");
    name.textContent = `${d.name} · ${d.platform}${d.connected ? " · online" : ""}`;
    const b = document.createElement("button");
    b.textContent = "Revoke";
    b.className = "danger";
    b.onclick = async () => {
      if (!confirm(`Revoke ${d.name}? The phone is disconnected and must be paired again.`)) return;
      await api("DELETE", `/api/admin/device/${encodeURIComponent(d.id)}`);
      refreshDevices();
    };
    li.append(name, b);
    return li;
  }));
}
if (!BRIDGE) {
  setInterval(refreshDevices, 5000);
  refreshDevices();
} else {
  setInterval(() => { // no MJPEG through the app bridge: refresh the live view a few times a second
    const id = $("feed").dataset.id;
    if (id && !$("feedbox").hidden) $("feed").src = `/api/frame/${encodeURIComponent(id)}.jpg?t=${Date.now()}`;
  }, 400);
}
// ---------- entrance marker: printed sheet, or shown at a true 15 cm on this screen ----------
const MK = { mode: "print", screen: null, adjust: 1 };
try { MK.adjust = Number(localStorage.getItem("fmt.markerAdjust")) || 1; } catch { /* storage blocked */ }

function cssPxPerCm() {
  const sc = MK.screen;
  if (!sc) return 96 / 2.54; // unknown screen: CSS reference size
  // points per cm on this Mac, converted to CSS px under the current browser zoom
  return (sc.pt_w / (sc.mm_w / 10)) * ((sc.backing || 1) / (window.devicePixelRatio || 1)) * MK.adjust;
}

function renderMarker() {
  $("mk-print").classList.toggle("on", MK.mode === "print");
  $("mk-screen").classList.toggle("on", MK.mode === "screen");
  $("markerinfo").textContent = MK.mode === "screen"
    ? "Phones lock onto the marker shown on this screen at exactly 15 cm. Keep this laptop still while the team works."
    : "Phones lock onto the printed sheet (18 cm, from marker.pdf) taped by the entrance.";
  if (!$("markerview").hidden) layoutMarker();
}

function layoutMarker() {
  const pc = cssPxPerCm();
  $("mk-img").style.width = $("mk-img").style.height = `${15 * pc}px`;
  const r = $("mk-ruler");
  r.style.width = `${10 * pc}px`;
  r.replaceChildren(...Array.from({ length: 11 }, (_, i) => {
    const t = document.createElement("i");
    t.style.left = `${i * pc - 1}px`;
    t.style.height = i % 5 === 0 ? "18px" : "9px";
    return t;
  }));
}

async function setMarker(mode) {
  const r = await api("POST", `/api/admin/marker?mode=${mode}`);
  MK.mode = r.mode; MK.screen = r.screen;
  renderMarker();
}

async function showMarker() {
  if (MK.mode !== "screen") await setMarker("screen");
  $("markerview").hidden = false;
  layoutMarker();
  try { await document.documentElement.requestFullscreen(); } catch { /* fullscreen refused: still shown */ }
  layoutMarker();
}

function hideMarker() {
  $("markerview").hidden = true;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}

function nudgeMarker(f) {
  MK.adjust = Math.min(1.2, Math.max(0.8, MK.adjust * f));
  try { localStorage.setItem("fmt.markerAdjust", String(MK.adjust)); } catch { /* storage blocked */ }
  layoutMarker();
}

$("mk-print").onclick = () => setMarker("print");
$("mk-screen").onclick = () => setMarker("screen");
$("mk-show").onclick = showMarker;
$("mk-close").onclick = hideMarker;
$("mk-bigger").onclick = () => nudgeMarker(1.005);
$("mk-smaller").onclick = () => nudgeMarker(1 / 1.005);
window.addEventListener("resize", () => { if (!$("markerview").hidden) layoutMarker(); });

// ---------- waypoint route panel (commander edits; phones' Map tab shows it read-only) ----------
function routeOrder() { return (S.plan.route?.stops || []).map((s) => s.id); }
async function postRoute(order, assignee, home) {
  await fetch("/api/admin/route", { method: "POST", credentials: "same-origin",
    headers: { "X-Sightline": "1", "Content-Type": "application/json" }, body: JSON.stringify({ order, assignee: assignee || null, home }) });
}
function renderRoute() {
  const r = S.plan.route;
  $("routebox").hidden = !r;
  if (!r) return;
  const fmtT = (s) => (s >= 60 ? `${Math.round(s / 60)} min` : `${s} s`);
  $("routesum").textContent = `${r.stops.length} stops · ${r.total_m} m · ~${fmtT(r.eta_s)}${r.done ? " · done" : ""}`;
  const legs = r.legs.slice(); // legs run from the assignee (or command) through the stops not yet reached
  let li = 0;
  $("routelist").replaceChildren(...r.stops.map((s, i) => {
    const el = document.createElement("li");
    el.className = s.reached ? "done" : "";
    const num = document.createElement("span"); num.className = "num"; num.textContent = s.reached ? "✓" : s.n;
    const t = document.createElement("span");
    const leg = s.reached ? null : legs[li++];
    t.textContent = s.reached ? "reached" : leg?.blocked ? "no known way there yet" : `${leg?.m ?? "?"} m${leg?.zone === 2 ? " · passes a hazard" : leg?.zone === 1 ? " · fall-risk area" : ""}`;
    const ops = document.createElement("span"); ops.className = "ops adminonly";
    const mk = (txt, fn, title) => { const b = document.createElement("button"); b.textContent = txt; b.title = title; b.onclick = fn; return b; };
    const order = routeOrder();
    const move = (d) => { const o = order.slice(); const j = i + d; if (j < 0 || j >= o.length) return; [o[i], o[j]] = [o[j], o[i]]; postRoute(o, $("routewho").value, $("routehome").checked); };
    ops.append(mk("↑", () => move(-1), "Earlier"), mk("↓", () => move(1), "Later"), mk("✕", () => send({ t: "pin_remove", id: s.id }), "Remove"));
    el.append(num, t, ops);
    return el;
  }));
  const sel = $("routewho"), cur = r.assignee || "";
  const opts = [["", "Not assigned"], ...[...S.phones.values()].filter((p) => p.connected).map((p) => [p.id, `Guide ${p.name}`])];
  if (sel.options.length !== opts.length || [...sel.options].some((o, k) => o.value !== opts[k][0])) {
    sel.replaceChildren(...opts.map(([v, txt]) => { const o = document.createElement("option"); o.value = v; o.textContent = txt; return o; }));
  }
  if (document.activeElement !== sel) sel.value = cur;
  $("routehome").checked = r.home;
}
$("routewho").onchange = () => postRoute(routeOrder(), $("routewho").value, $("routehome").checked);
$("routehome").onchange = () => postRoute(routeOrder(), $("routewho").value, $("routehome").checked);
$("routebest").onclick = async () => {
  const r = await fetch("/api/admin/route/best", { method: "POST", credentials: "same-origin", headers: { "X-Sightline": "1" } });
  if (!r.ok) toast((await r.json().catch(() => ({}))).detail || "Couldn't plan an order");
};
$("routeclear").onclick = async () => { if (confirm("Remove all waypoints?")) await api("POST", "/api/admin/route/clear"); };

// ---------- label mode: hand-made ground truth (rooms, fall-risk spots) for evaluate.py ----------
const LB = { on: false, map: null, area: null, rooms: [], risk: [], tool: "room", draft: [] };
const LB_HINT = { room: "Room: click each corner of a room, then click the first corner again (or press Enter) to close it.",
  risk: "Fall-risk spot: click where someone could trip or fall (stair edges, drops, clutter in a walkway)." };

async function startLabelling(m) {
  if (!confirm(`Label “${m.name}”? It replaces the map on screen with this saved map.`)) return;
  await api("POST", `/api/admin/maps/${m.id}/load`);
  const l = await api("GET", `/api/admin/maps/${m.id}/labels`);
  Object.assign(LB, { on: true, map: m, area: l.area, rooms: l.rooms || [], risk: l.risk || [], tool: "room", draft: [] });
  if (S.mode === "3d") $("mode").click();
  $("lbname").textContent = `Labelling “${m.name}”`;
  $("labelbar").hidden = false;
  syncLabelBar();
}

function syncLabelBar() {
  document.querySelectorAll("#labelbar [data-area]").forEach((b) => b.classList.toggle("on", b.dataset.area === LB.area));
  document.querySelectorAll("#labelbar [data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === LB.tool));
  $("lbhint").textContent = (LB.area ? "" : "First choose this map's area (A, B or C). ") + LB_HINT[LB.tool] +
    ` · ${LB.rooms.length} rooms, ${LB.risk.length} spots`;
}

function labelClick(wx, wz, sx, sy) {
  if (LB.tool === "risk") { LB.risk.push([wx, wz]); syncLabelBar(); return; }
  if (LB.draft.length >= 3) {
    const [fx, fy] = toScreen(LB.draft[0][0], LB.draft[0][1]);
    if (Math.hypot(fx - sx, fy - sy) < 14) { closeRoom(); return; }
  }
  LB.draft.push([wx, wz]);
}
function closeRoom() { if (LB.draft.length >= 3) LB.rooms.push(LB.draft); LB.draft = []; syncLabelBar(); }

function drawLabels() {
  if (!LB.on) return;
  ctx.lineWidth = 2; ctx.strokeStyle = "#2aa7c9"; ctx.fillStyle = "rgba(42, 167, 201, 0.12)";
  for (const poly of [...LB.rooms, LB.draft]) {
    if (!poly.length) continue;
    ctx.beginPath(); poly.forEach(([x0, z0], i) => { const [x, y] = toScreen(x0, z0); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    if (poly !== LB.draft) { ctx.closePath(); ctx.fill(); }
    ctx.stroke();
  }
  for (const [x0, z0] of LB.draft) { const [x, y] = toScreen(x0, z0); ctx.fillStyle = "#2aa7c9"; ctx.fillRect(x - 3, y - 3, 6, 6); }
  ctx.strokeStyle = "#7fdcf5"; ctx.lineWidth = 2.5;
  for (const [x0, z0] of LB.risk) { const [x, y] = toScreen(x0, z0); ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke(); }
}

document.querySelectorAll("#labelbar [data-area]").forEach((b) => (b.onclick = () => { LB.area = b.dataset.area; syncLabelBar(); }));
document.querySelectorAll("#labelbar [data-tool]").forEach((b) => (b.onclick = () => { closeRoom(); LB.tool = b.dataset.tool; syncLabelBar(); }));
$("lbundo").onclick = () => { if (LB.draft.length) LB.draft.pop(); else if (LB.tool === "risk") LB.risk.pop(); else LB.rooms.pop(); syncLabelBar(); };
$("lbsave").onclick = async () => {
  closeRoom();
  if (!LB.area) { toast("Choose the area (A, B or C) first"); return; }
  const r = await fetch(`/api/admin/maps/${LB.map.id}/labels`, { method: "PUT", credentials: "same-origin",
    headers: { "X-Sightline": "1", "Content-Type": "application/json" }, body: JSON.stringify({ area: LB.area, rooms: LB.rooms, risk: LB.risk }) });
  toast(r.ok ? `Labels saved: ${LB.rooms.length} rooms, ${LB.risk.length} spots (area ${LB.area})` : "Couldn't save the labels");
  refreshMaps();
};
$("lbdone").onclick = () => { LB.on = false; LB.draft = []; $("labelbar").hidden = true; };

// ---------- saved maps (stored encrypted on this laptop) ----------
async function refreshMaps() {
  const list = await api("GET", "/api/admin/maps");
  if (!list.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No saved maps yet.";
    $("maplist").replaceChildren(li);
    return;
  }
  $("maplist").replaceChildren(...list.map((m) => {
    const li = document.createElement("li");
    li.className = "savedmap";
    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = m.name;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${new Date(m.created * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })} · ${(m.bytes / 1e6).toFixed(1)} MB`;
    info.append(name, meta);
    const load = document.createElement("button");
    load.textContent = "Load";
    load.onclick = async () => {
      if (!confirm(`Load “${m.name}”? It replaces the map on screen (save the current one first if you need it).`)) return;
      await api("POST", `/api/admin/maps/${m.id}/load`);
    };
    const lab = document.createElement("button");
    lab.textContent = "Label";
    lab.title = "Mark this map's real rooms and fall-risk spots (ground truth for testing the detectors)";
    lab.onclick = () => startLabelling(m);
    if (m.area) {
      const tag = document.createElement("div");
      tag.className = "lbl";
      tag.textContent = `Area ${m.area} · ${m.labelled.rooms} rooms, ${m.labelled.risk} risk spots labelled`;
      info.append(tag);
    }
    const del = document.createElement("button");
    del.textContent = "Delete";
    del.className = "danger";
    del.onclick = async () => {
      if (!confirm(`Delete “${m.name}” permanently?`)) return;
      await api("DELETE", `/api/admin/maps/${m.id}`);
      refreshMaps();
    };
    li.append(info, load, lab, del);
    return li;
  }));
}

$("mapsave").onclick = async () => {
  $("mapsave").disabled = true;
  try {
    const r = await api("POST", `/api/admin/maps?name=${encodeURIComponent($("mapname").value.trim())}`);
    $("mapname").value = "";
    toast(`Saved “${r.name}”`);
    refreshMaps();
  } finally { $("mapsave").disabled = false; }
};

// ---------- other computers (Chrome on a second laptop): one-time links, made on this Mac only ----------
async function refreshScreens() {
  const list = await api("GET", "/api/admin/screens");
  $("screens").replaceChildren(...list.map((sc) => {
    const li = document.createElement("li");
    li.className = "screen";
    const t = document.createElement("span");
    t.textContent = `Computer ${sc.ip} · since ${new Date(sc.created * 1000).toLocaleTimeString([], { timeStyle: "short" })}`;
    const b = document.createElement("button");
    b.textContent = "Revoke";
    b.className = "danger";
    b.onclick = async () => {
      if (!confirm(`Revoke ${sc.ip}? Its commander page goes dark straight away.`)) return;
      await api("DELETE", `/api/admin/screens/${encodeURIComponent(sc.id)}`);
      refreshScreens();
    };
    li.append(t, b);
    return li;
  }));
}

$("scradd").onclick = async () => {
  const r = await api("POST", "/api/admin/screens");
  $("scrbox").hidden = false;
  $("scrqr").innerHTML = r.svg; // server-generated QR (segno), no user input inside
  $("scrurl").textContent = r.url;
  $("scrinfo").textContent = `Open this link in Chrome on the other computer (works once, for 10 minutes).\n` +
    `Chrome will warn the connection is not private (the laptop makes its own certificate): check it is this laptop, then Advanced → Proceed.\n` +
    `Certificate SHA-256 starts ${r.fingerprint.slice(0, 16)}…`;
};

if (!BRIDGE) {
  refreshMaps();
  api("GET", "/api/admin/marker").then((r) => {
    MK.mode = r.mode; MK.screen = r.screen; renderMarker();
    if (!r.local) document.body.classList.add("remote"); // another computer: no screen marker, no adding screens
    else { refreshScreens(); setInterval(refreshScreens, 5000); refreshVoice(); }
  });
}

// ---------- voice guidance on phones: ElevenLabs key, saved on this Mac only, never shown again ----------
async function refreshVoice(note) {
  const st = await api("GET", "/api/admin/voice");
  $("vcstatus").textContent = note || (st.configured ? `ElevenLabs is set up (voice ${st.voice}).` : "Not set up: phones use their built-in voice.");
  $("vcform").hidden = st.configured;
  $("vcchange").hidden = !$("vcform").hidden;
}
$("vcchange").onclick = () => { $("vcform").hidden = false; $("vcchange").hidden = true; $("vckey").focus(); };
$("vcform").onsubmit = async (e) => {
  e.preventDefault();
  $("vcsave").disabled = true;
  $("vcstatus").textContent = "Checking the key with ElevenLabs…";
  try {
    const r = await fetch("/api/admin/voice/key", { method: "POST", credentials: "same-origin",
      headers: { "X-Sightline": "1", "Content-Type": "application/json" },
      body: JSON.stringify({ key: $("vckey").value, voice: $("vcvoice").value }) });
    const j = await r.json().catch(() => ({}));
    $("vckey").value = "";
    if (!r.ok || !j.ok) { $("vcstatus").textContent = j.detail || j.message || `Couldn't save the key (${r.status}).`; return; }
    refreshVoice(`ElevenLabs is set up. ${j.message}`);
  } catch { $("vcstatus").textContent = "Couldn't reach the FindMyTeam server."; }
  finally { $("vcsave").disabled = false; }
};

// ---------- Ask Gemini: the commander's AI assistant (conversation kept in this tab only) ----------
const ASK = { msgs: [], busy: false, ready: null };

function richText(el, text) { // tiny safe formatter: paragraphs, "- " bullets, **bold**; never innerHTML
  el.replaceChildren();
  let list = null;
  const inline = (parent, line) => line.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) { const b = document.createElement("b"); b.textContent = part.slice(2, -2); parent.append(b); }
    else if (part) parent.append(document.createTextNode(part));
  });
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const m = line.match(/^\s*(?:[-*\u2022]|\d+\.)\s+(.*)/);
    if (m) {
      if (!list) { list = document.createElement("ul"); el.append(list); }
      const li = document.createElement("li"); inline(li, m[1]); list.append(li);
    } else if (line.trim()) {
      list = null;
      const para = document.createElement("p"); inline(para, line.replace(/^#+\s*/, "")); el.append(para);
    } else list = null;
  }
}

function renderAsk(extra) {
  const log = $("asklog");
  const items = ASK.msgs.map((m) => {
    const div = document.createElement("div");
    div.className = "msg " + m.role;
    if (m.role === "model") richText(div, m.text); else div.textContent = m.text;
    return div;
  });
  if (!items.length && !extra) {
    const empty = document.createElement("div");
    empty.className = "askempty";
    empty.append($("askbtn").querySelector(".spark").cloneNode(true));
    const t = document.createElement("div");
    t.textContent = ASK.ready === false
      ? (ASK.canSetup ? "Set up Gemini once: paste your Gemini API key. It's saved only on this laptop and never shown again."
        : "Gemini isn't set up yet: add the API key on the server laptop's commander page.")
      : "Ask anything a rescuer radios in: where people are, exits, stairs, hazards, what's been searched, or general rescue guidance.";
    empty.append(t);
    if (ASK.ready === false && ASK.canSetup) {
      const f = document.createElement("form");
      f.className = "askkey";
      const inp = document.createElement("input");
      inp.type = "password"; inp.autocomplete = "off"; inp.placeholder = "Gemini API key";
      const b = document.createElement("button"); b.className = "primary"; b.textContent = "Save key";
      const err = document.createElement("div"); err.className = "dim small";
      f.append(inp, b, err);
      f.onsubmit = async (e) => {
        e.preventDefault();
        b.disabled = true; err.textContent = "Saving and checking the key with Google…";
        const res = await saveKey(inp.value);
        inp.value = ""; b.disabled = false;
        if (res.ok) { renderAsk(); $("askinput").focus(); } else err.textContent = res.message;
      };
      empty.append(f);
    }
    items.push(empty);
  }
  if (extra) items.push(extra);
  log.replaceChildren(...items);
  log.scrollTop = log.scrollHeight;
}

async function saveKey(key) {
  try {
    const r = await fetch("/api/admin/ask/key", { method: "POST", credentials: "same-origin",
      headers: { "X-Sightline": "1", "Content-Type": "application/json" }, body: JSON.stringify({ key }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) return { ok: false, message: j.detail || `Couldn't save the key (${r.status}).` };
    ASK.ready = !!j.ok;
    return { ok: !!j.ok, message: j.message || "" };
  } catch { return { ok: false, message: "Couldn't reach the FindMyTeam server." }; }
}

async function openAsk() {
  $("askview").hidden = false;
  if (ASK.ready === null) {
    try { const st = await api("GET", "/api/admin/ask"); ASK.ready = st.configured; ASK.canSetup = st.can_setup; } catch { ASK.ready = null; }
    $("askkeybtn").hidden = !ASK.canSetup;
  }
  renderAsk();
  $("askinput").focus();
}
function closeAsk() { $("askview").hidden = true; }

async function sendAsk(text) {
  text = text.trim();
  if (!text || ASK.busy) return;
  if (ASK.ready === false && ASK.canSetup && /^\S{30,200}$/.test(text)) { // a key pasted into the question box
    $("askinput").value = "";
    const res = await saveKey(text);
    const note = document.createElement("div");
    note.className = res.ok ? "msg model" : "msg err";
    note.textContent = res.ok ? `Gemini is set up. ${res.message}` : res.message;
    renderAsk(note);
    return;
  }
  ASK.busy = true; $("asksend").disabled = true;
  ASK.msgs.push({ role: "user", text });
  $("askinput").value = "";
  const typing = document.createElement("div");
  typing.className = "msg model typing";
  typing.textContent = "Gemini is thinking…";
  renderAsk(typing);
  try {
    const r = await fetch("/api/admin/ask", { method: "POST", credentials: "same-origin",
      headers: { "X-Sightline": "1", "Content-Type": "application/json" }, body: JSON.stringify({ messages: ASK.msgs.slice(-24) }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `Gemini couldn't answer (${r.status}).`);
    ASK.msgs.push({ role: "model", text: j.answer });
    renderAsk();
  } catch (e) {
    ASK.msgs.pop(); // keep the history valid: drop the unanswered question, show why
    const err = document.createElement("div");
    err.className = "msg err";
    err.textContent = `${e.message} Your question: "${text}"` +
      (ASK.canSetup && /credit|quota|billing|key|402|403|429/i.test(e.message) ? " Use Change key (top right) to try a different key." : "");
    renderAsk(err);
    $("askinput").value = text;
  } finally { ASK.busy = false; $("asksend").disabled = false; }
}

$("askbtn").onclick = openAsk;
$("askclose").onclick = closeAsk;
$("askclear").onclick = () => { ASK.msgs = []; renderAsk(); $("askinput").focus(); };
$("askkeybtn").onclick = () => { ASK.ready = false; ASK.msgs = []; renderAsk(); }; // shows the key box again (this Mac only)
$("askform").onsubmit = (e) => { e.preventDefault(); sendAsk($("askinput").value); };
$("askinput").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendAsk($("askinput").value); } };
document.querySelectorAll("#asksugg button").forEach((b) => (b.onclick = () => sendAsk(b.dataset.q)));

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("askview").hidden) { closeAsk(); return; }
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (LB.on && e.key === "Enter") { closeRoom(); return; }
  if (e.key === "a" && !BRIDGE) { e.preventDefault(); openAsk(); return; }
  if (e.key === "Escape") { setPinMode(null); hideMarker(); }
  if (e.key === "f") setAutoFit(true);
  if (e.key === "l") $("follow").click();
  if (e.key === "t") $("trails").click();
  if (e.key === "3") $("mode").click();
  if (e.key === "c") $("cov").click();
});

$("addr").textContent = location.host;
connect();
requestAnimationFrame(draw);
