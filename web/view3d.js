// 3D commander view: the shared AI-depth point cloud, ARKit/ARCore planes, responders and pins.
// Responders and pins are drawn without depth testing, so the commander sees everyone through walls.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const MAX_POINTS = 600000;
const KIND_COLOR = { wall: 0xe6edf3, vertical: 0x8b98a6, door: 0x2ecc71, window: 0x3fa7ff, floor: 0x2a3442, horizontal: 0x2a3442, table: 0x2a3442, seat: 0x2a3442, ceiling: 0x1b222b };
const PIN_COLOR = { rally: 0x2ecc71, hazard: 0xff3b30, victim: 0xffb400, note: 0xc77dff };
const WALL_KINDS = new Set(["wall", "vertical", "door", "window"]);

// Eye-dome lighting: darkens each point by how much its neighbours sit in front of it (log depth),
// which gives an unlit point cloud shape and crisp outlines. Rendered as a full-screen pass.
const EDL_FRAG = `
uniform sampler2D tColor; uniform sampler2D tDepth; uniform vec2 texel; uniform float cNear; uniform float cFar; uniform vec3 bg;
varying vec2 vUv;
float logDepth(vec2 uv) {
  float d = texture2D(tDepth, uv).x;
  if (d >= 1.0) return 1e9;
  float z = d * 2.0 - 1.0;
  return log2(2.0 * cNear * cFar / (cFar + cNear - z * (cFar - cNear)));
}
void main() {
  float L = logDepth(vUv);
  if (L > 1e8) { gl_FragColor = vec4(bg, 1.0); return; }
  float sum = 0.0, ao = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.785398;
    vec2 dir = vec2(cos(a), sin(a)) * texel;
    sum += max(0.0, L - logDepth(vUv + dir * 1.6));
    ao += clamp(L - logDepth(vUv + dir * 7.0), 0.0, 0.06);
  }
  // crisp outlines (eye-dome lighting) + a softer, wider darkening in corners and where things meet the floor
  float shade = exp(-sum * 28.0) * (1.0 - clamp(ao * 2.2, 0.0, 0.35));
  gl_FragColor = linearToOutputTexel(vec4(texture2D(tColor, vUv).rgb * shade, 1.0));
}`;

// IEEE half float -> number (mesh vertices arrive as float16 offsets from their block origin)
function half(h) {
  const e = (h >> 10) & 31, f = h & 1023, s = h & 32768 ? -1 : 1;
  if (e === 0) return s * 5.960464477539063e-8 * f;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}
// Camera colours arrive as sRGB bytes; three.js shades in linear light. Converting here (once) keeps the
// real colours: without it they were encoded twice and came out pale and washed out.
const SRGB_TO_LINEAR = new Float32Array(256).map((_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
});
const BLOCK_M = 16 * 0.04; // fused surface block size (m)
const blockId = (bx, by, bz) => ((bx + 512) * 1024 + (by + 512)) * 1024 + (bz + 512);

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

export class View3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.shadowMap.enabled = true; // soft shadows from a light overhead: furniture sits on the floor
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene(); // walls, people, pins, labels: drawn over the shaded cloud
    this.cloudScene = new THREE.Scene(); // points + grid: rendered to a target, then eye-dome lit
    this.bg = new THREE.Color(0x070708);
    // 4x multisampling on the off-screen pass: without it, edges in 3D come out jagged (stair-stepped)
    this.target = new THREE.WebGLRenderTarget(1, 1, { depthTexture: new THREE.DepthTexture(1, 1), samples: 4 });
    this.edl = new THREE.ShaderMaterial({
      uniforms: { tColor: { value: null }, tDepth: { value: null }, texel: { value: new THREE.Vector2() }, cNear: { value: 0.05 }, cFar: { value: 500 }, bg: { value: this.bg } },
      vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: EDL_FRAG,
      depthTest: false, depthWrite: false,
    });
    this.quadScene = new THREE.Scene();
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.edl));
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.wallGroup = new THREE.Group();
    // coverage heatmap: one texel per coverage cell, on a quad at floor height
    this.covCanvas = document.createElement("canvas");
    this.covTex = new THREE.CanvasTexture(this.covCanvas);
    this.covTex.magFilter = THREE.NearestFilter;
    this.covTex.minFilter = THREE.NearestFilter;
    this.covMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: this.covTex, transparent: true, depthWrite: false }));
    this.covMesh.rotation.x = -Math.PI / 2;
    this.covMesh.visible = false;
    this.covOn = false; // off in 3D unless asked for: it tints the real colours
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 500);
    this.camera.position.set(0, 14, -8);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, -1.2, 6);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9));

    // points: preallocated buffers, filled as voxels stream in
    this.pointCount = 0;
    this.positions = new Float32Array(MAX_POINTS * 3);
    this.colors = new Float32Array(MAX_POINTS * 3);
    const g = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage);
    const colAttr = new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("position", posAttr);
    g.setAttribute("color", colAttr);
    g.setDrawRange(0, 0);
    const pointMat = new THREE.PointsMaterial({ size: 0.05, vertexColors: true, sizeAttenuation: true });
    this.cloud = new THREE.Points(g, pointMat);
    this.cloud.frustumCulled = false;
    this.cloudScene.add(this.cloud);
    // the same points, but only those in places the smooth surfaces haven't covered yet (fills the gaps)
    this.gapIndex = new Uint32Array(MAX_POINTS);
    const gg = new THREE.BufferGeometry();
    gg.setAttribute("position", posAttr);
    gg.setAttribute("color", colAttr);
    gg.setIndex(new THREE.BufferAttribute(this.gapIndex, 1).setUsage(THREE.DynamicDrawUsage));
    gg.setDrawRange(0, 0);
    this.gaps = new THREE.Points(gg, pointMat);
    this.gaps.frustumCulled = false;
    this.gaps.visible = false;
    this.cloudScene.add(this.gaps);
    this.meshIds = new Set();
    this.gapsDirty = false;
    this.lastGaps = 0;

    this.grid = new THREE.GridHelper(60, 60, 0x26262b, 0x141417);
    this.grid.position.set(0, -1.4, 15);
    this.cloudScene.add(this.grid);
    this.scene.add(this.wallGroup);
    this.scene.add(this.covMesh);
    // fused surfaces (TSDF meshes) live with the points so they get eye-dome lighting and real depth
    this.surfaces = new THREE.Group();
    this.cloudScene.add(this.surfaces);
    // soft light that keeps the camera colours close to what the phone saw: a gentle sky/ground
    // gradient for shape, a weak key light for depth, nothing bright enough to bleach colours
    this.cloudScene.add(new THREE.HemisphereLight(0xffffff, 0x9a9a9a, 0.82));
    const sun = new THREE.DirectionalLight(0xffffff, 0.45);
    sun.position.set(4, 10, 6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0008;
    sun.shadow.normalBias = 0.02;
    sun.shadow.radius = 4;
    this.cloudScene.add(sun);
    this.cloudScene.add(sun.target);
    this.sun = sun;
    this.lastShadowFit = 0;
    this.meshes = new Map();
    this.texMeshes = new Map(); // photo-textured walls / floor
    this.surfaceMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
    this.walls = [];
    this.showPoints = false;
    const marker = new THREE.Mesh(new THREE.PlaneGeometry(0.18, 0.18), new THREE.MeshBasicMaterial({ color: 0xff3d47, side: THREE.DoubleSide }));
    this.scene.add(marker);
    this.scene.add(this.label("ENTRANCE MARKER", "#ff3d47", [0, 0.35, 0]));

    this.planeMeshes = new Map();
    this.avatars = new Map();
    this.pinMeshes = new Map();
    this.trails = new Map();
    this.showTrails = true;
    this.active = false;
  }

  // ---- data in ----
  addVoxels(buf) {
    // 0x02 | layer u8 | voxel size f32 | N x (int16 x,y,z, uint8 r,g,b)
    const dv = new DataView(buf);
    const size = dv.getFloat32(2, true);
    for (let o = 6; o + 9 <= buf.byteLength && this.pointCount < MAX_POINTS; o += 9) {
      const i = this.pointCount++ * 3;
      this.positions[i] = dv.getInt16(o, true) * size;
      this.positions[i + 1] = dv.getInt16(o + 2, true) * size;
      this.positions[i + 2] = dv.getInt16(o + 4, true) * size;
      this.colors[i] = SRGB_TO_LINEAR[dv.getUint8(o + 6)];
      this.colors[i + 1] = SRGB_TO_LINEAR[dv.getUint8(o + 7)];
      this.colors[i + 2] = SRGB_TO_LINEAR[dv.getUint8(o + 8)];
    }
    const g = this.cloud.geometry;
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    g.setDrawRange(0, this.pointCount);
    this.gapsDirty = true;
  }

  // Indices of points whose 0.64 m block has no fused surface yet. Rebuilt at most once a second.
  refreshGaps() {
    let n = 0;
    const P = this.positions, ids = this.meshIds;
    for (let i = 0; i < this.pointCount; i++) {
      const j = i * 3;
      if (!ids.has(blockId(Math.floor(P[j] / BLOCK_M), Math.floor(P[j + 1] / BLOCK_M), Math.floor(P[j + 2] / BLOCK_M)))) this.gapIndex[n++] = i;
    }
    const idx = this.gaps.geometry.index;
    idx.needsUpdate = true;
    this.gaps.geometry.setDrawRange(0, n);
    this.gapsDirty = false;
    this.lastGaps = performance.now();
  }

  updatePointVisibility() {
    this.cloud.visible = this.showPoints || this.meshes.size === 0;
    this.gaps.visible = !this.cloud.visible;
  }

  // 0x04 | i16 block x,y,z | u16 nv | u16 nf | f16[nv*3] offsets (m) | u8[nv*3] rgb | u16[nf*3] indices
  addMesh(buf) {
    const dv = new DataView(buf);
    const fine = dv.getUint8(0) === 0x07; // 0x07: 2 cm detail layer (0.32 m blocks), 0x04: 4 cm (0.64 m)
    const bx = dv.getInt16(1, true), by = dv.getInt16(3, true), bz = dv.getInt16(5, true);
    const nv = dv.getUint16(7, true), nf = dv.getUint16(9, true);
    const key = (fine ? "f" : "") + bx + "," + by + "," + bz;
    const old = this.meshes.get(key);
    if (old) { this.surfaces.remove(old); old.geometry.dispose(); this.meshes.delete(key); }
    const id = fine ? blockId(Math.floor(bx / 2), Math.floor(by / 2), Math.floor(bz / 2)) : blockId(bx, by, bz);
    if (!nv || !nf) { if (!fine) this.meshIds.delete(id); this.gapsDirty = true; return; }
    const block = 16 * (fine ? 0.02 : 0.04), ox = bx * block, oy = by * block, oz = bz * block;
    const pos = new Float32Array(nv * 3), col = new Float32Array(nv * 3), idx = new Uint16Array(nf * 3);
    let o = 11;
    for (let i = 0; i < nv * 3; i++, o += 2) pos[i] = half(dv.getUint16(o, true)) + (i % 3 === 0 ? ox : i % 3 === 1 ? oy : oz);
    const rgb = new Uint8Array(buf, o, nv * 3);
    for (let i = 0; i < nv * 3; i++) col[i] = SRGB_TO_LINEAR[rgb[i]];
    o += nv * 3;
    for (let i = 0; i < nf * 3; i++, o += 2) idx[i] = dv.getUint16(o, true);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    if (buf.byteLength >= o + nv * 3) { // smooth normals from the server's distance field: no faceting, no block seams
      g.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(buf.slice(o, o + nv * 3)), 3, true));
    } else g.computeVertexNormals();
    const m = new THREE.Mesh(g, this.surfaceMat);
    m.castShadow = m.receiveShadow = true;
    this.surfaces.add(m);
    const first = this.meshes.size === 0;
    this.meshes.set(key, m);
    this.meshIds.add(id);
    this.gapsDirty = true;
    this.updatePointVisibility();
    if (first) this.setWalls(this.walls); // walls switch to light outlines once real surfaces exist
  }

  setPoints(on) { this.showPoints = on; this.updatePointVisibility(); }

  // Photo-textured wall / floor: 0x06 | kind | u32 id | f32[12] corners (bl, br, tr, tl) | JPEG
  addTexture(buf) {
    const dv = new DataView(buf);
    const kind = dv.getUint8(1), id = dv.getUint32(2, true);
    const c = [];
    for (let i = 0; i < 12; i++) c.push(dv.getFloat32(6 + 4 * i, true));
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      let m = this.texMeshes.get(id);
      const tex = new THREE.Texture(img);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      tex.needsUpdate = true;
      if (!m) {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(c, 3));
        g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
        g.setIndex([0, 1, 2, 0, 2, 3]);
        g.computeVertexNormals();
        m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ map: tex, side: THREE.DoubleSide }));
        m.receiveShadow = true;
        m.castShadow = kind === 1;
        m.userData.kind = kind;
        this.surfaces.add(m);
        this.texMeshes.set(id, m);
      } else {
        m.geometry.attributes.position.array.set(c);
        m.geometry.attributes.position.needsUpdate = true;
        m.geometry.computeVertexNormals();
        m.material.map?.dispose();
        m.material.map = tex;
        m.material.needsUpdate = true;
      }
      if (kind === 1 && this.walls.length) this.setWalls(this.walls);
    };
    img.src = URL.createObjectURL(new Blob([new Uint8Array(buf, 54)], { type: "image/jpeg" }));
  }

  keepTextures(ids) {
    const keep = new Set(ids);
    for (const [id, m] of this.texMeshes) if (!keep.has(id)) { this.surfaces.remove(m); m.geometry.dispose(); m.material.map?.dispose(); m.material.dispose(); this.texMeshes.delete(id); }
  }

  clear() {
    for (const m of this.meshes.values()) { this.surfaces.remove(m); m.geometry.dispose(); }
    this.meshes.clear();
    this.meshIds.clear();
    this.keepTextures([]);
    this.pointCount = 0;
    this.cloud.geometry.setDrawRange(0, 0);
    this.gaps.geometry.setDrawRange(0, 0);
    this.updatePointVisibility();
    for (const m of this.planeMeshes.values()) this.scene.remove(m);
    this.planeMeshes.clear();
  }

  setFloor(y) { this.grid.position.y = y; }

  // Fall-risk zones: soft yellow patches on the floor (no text)
  setZones(zones) {
    if (!this.zoneGroup) { this.zoneGroup = new THREE.Group(); this.scene.add(this.zoneGroup); }
    for (const c of [...this.zoneGroup.children]) { c.geometry.dispose(); this.zoneGroup.remove(c); }
    const y = this.grid.position.y + 0.03;
    for (const z of zones) {
      const shape = new THREE.Shape();
      if (z.poly) z.poly.forEach(([x, zz], i) => (i ? shape.lineTo(x, -zz) : shape.moveTo(x, -zz)));
      else shape.absarc(z.x, -z.z, z.r, 0, Math.PI * 2, false);
      const g = new THREE.ShapeGeometry(shape, 24);
      g.rotateX(-Math.PI / 2); // shape x/y -> world x/-z ... then -zz above flips back to +z
      const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xffcc33, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide }));
      m.position.y = y;
      this.zoneGroup.add(m);
    }
  }

  // Everyone's way back to command: a line in their colour just above the floor
  setRoutes(plan, phones) {
    if (!this.routeGroup) { this.routeGroup = new THREE.Group(); this.scene.add(this.routeGroup); }
    for (const c of [...this.routeGroup.children]) { c.geometry.dispose(); this.routeGroup.remove(c); }
    const y = this.grid.position.y + 0.08;
    for (const [id, info] of Object.entries(plan?.people || {})) {
      const p = phones.get(id);
      if (!info.home?.path?.length || !p?.connected) continue;
      const pts = info.home.path.map(([x, z]) => new THREE.Vector3(x, y, z));
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        info.home.zone ? new THREE.LineDashedMaterial({ color: p.color, dashSize: 0.25, gapSize: 0.2 }) : new THREE.LineBasicMaterial({ color: p.color }));
      if (info.home.zone) line.computeLineDistances();
      this.routeGroup.add(line);
    }
  }

  // Stairs: amber outline + step lines on the floor plan, and a tinted ramp up (or down) the real slope
  setStairs(list, floorY) {
    if (!this.stairGroup) { this.stairGroup = new THREE.Group(); this.scene.add(this.stairGroup); }
    for (const c of [...this.stairGroup.children]) { c.geometry.dispose(); this.stairGroup.remove(c); }
    const fy = (floorY ?? this.grid.position.y) + 0.02, amber = 0xffb020;
    for (const st of list) {
      const [a, b, c, d] = st.c, rise = st.down ? -st.rise : st.rise;
      const lowY = st.down ? fy + rise : fy, highY = st.down ? fy : fy + rise; // a-d is the low edge, b-c the high edge
      const quad = new THREE.BufferGeometry();
      quad.setAttribute("position", new THREE.Float32BufferAttribute([a[0], lowY, a[1], b[0], highY, b[1], c[0], highY, c[1], d[0], lowY, d[1]], 3));
      quad.setIndex([0, 1, 2, 0, 2, 3]);
      this.stairGroup.add(new THREE.Mesh(quad, new THREE.MeshBasicMaterial({ color: amber, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false })));
      const pts = [];
      const n = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.28));
      for (let i = 0; i <= n; i++) {
        const t = i / n, y = lowY + (highY - lowY) * t;
        pts.push(new THREE.Vector3(a[0] + (b[0] - a[0]) * t, y, a[1] + (b[1] - a[1]) * t), new THREE.Vector3(d[0] + (c[0] - d[0]) * t, y, d[1] + (c[1] - d[1]) * t));
      }
      this.stairGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: amber })));
    }
  }

  // Keep the shadow camera tight around what has been mapped (sharper shadows than a fixed huge box)
  fitShadow() {
    this.lastShadowFit = performance.now();
    if (!this.surfaces.children.length) return;
    const box = new THREE.Box3().setFromObject(this.surfaces);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const r = Math.min(40, Math.max(size.x, size.z) / 2 + 1);
    this.sun.target.position.copy(c);
    this.sun.position.set(c.x + 4, c.y + 12, c.z + 6);
    const cam = this.sun.shadow.camera;
    cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r; cam.near = 0.5; cam.far = 40;
    cam.updateProjectionMatrix();
  }

  // Clean merged walls from the server: [x1, z1, x2, z2, y0, y1, kind]
  // Before any fused surface exists they are see-through slabs; afterwards the scan itself shows the
  // walls, so they shrink to a thin outline along the floor (doors and windows keep their colour).
  setWalls(walls) {
    this.walls = walls;
    for (const c of [...this.wallGroup.children]) { c.geometry.dispose(); this.wallGroup.remove(c); }
    if (this.meshes.size) {
      const fy = this.grid.position.y + 0.01;
      for (const [x1, z1, x2, z2, y0, y1, kind] of walls) {
        const pts = [new THREE.Vector3(x1, fy, z1), new THREE.Vector3(x2, fy, z2)];
        if (kind !== "wall") pts.push(new THREE.Vector3(x2, y1, z2), new THREE.Vector3(x1, y1, z1), pts[0]);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
          new THREE.LineBasicMaterial({ color: KIND_COLOR[kind] ?? 0xe6edf3, transparent: true, opacity: kind === "wall" ? 0.45 : 0.85 }));
        this.wallGroup.add(line);
      }
      return;
    }
    for (const [x1, z1, x2, z2, y0, y1, kind] of walls) {
      const len = Math.hypot(x2 - x1, z2 - z1);
      const h = Math.max(0.3, y1 - y0);
      const color = KIND_COLOR[kind] ?? 0xe6edf3;
      const slab = new THREE.Mesh(new THREE.BoxGeometry(len, h, 0.06),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: kind === "wall" ? 0.16 : 0.3, depthWrite: false }));
      slab.position.set((x1 + x2) / 2, (y0 + y1) / 2, (z1 + z2) / 2);
      slab.rotation.y = -Math.atan2(z2 - z1, x2 - x1);
      slab.add(new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 })));
      this.wallGroup.add(slab);
    }
  }

  setCoverage(cov, cell, bbox, floorY) {
    if (!bbox) { this.covMesh.visible = false; return; }
    const [x0, z0, x1, z1] = bbox;
    const ix0 = Math.floor(x0 / cell), iz0 = Math.floor(z0 / cell), nx = Math.floor(x1 / cell) - ix0 + 1, nz = Math.floor(z1 / cell) - iz0 + 1;
    if (nx <= 0 || nz <= 0 || nx * nz > 400000) return;
    this.covCanvas.width = nx; this.covCanvas.height = nz;
    const ctx = this.covCanvas.getContext("2d");
    const img = ctx.createImageData(nx, nz);
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const n = cov.get(ix0 + i + "," + (iz0 + j)) || 0, o = (j * nx + i) * 4;
      if (n) { img.data[o] = 46; img.data[o + 1] = 204; img.data[o + 2] = 113; img.data[o + 3] = Math.min(140, 40 + n * 6); }
      else { img.data[o] = 255; img.data[o + 1] = 70; img.data[o + 2] = 70; img.data[o + 3] = 55; }
    }
    ctx.putImageData(img, 0, 0);
    this.covTex.needsUpdate = true;
    this.covMesh.scale.set(nx * cell, nz * cell, 1);
    this.covMesh.position.set(ix0 * cell + (nx * cell) / 2, floorY + 0.02, iz0 * cell + (nz * cell) / 2);
    this.covMesh.visible = this.covOn;
  }

  showCoverage(on) { this.covOn = on; this.covMesh.visible = on && this.covCanvas.width > 0; }

  upsertPlane(p) {
    const old = this.planeMeshes.get(p.id);
    if (old) this.scene.remove(old);
    if (p.corners.length < 3 || WALL_KINDS.has(p.kind)) return; // walls come merged via setWalls
    const pos = [];
    for (let i = 1; i + 1 < p.corners.length; i++) pos.push(...p.corners[0], ...p.corners[i], ...p.corners[i + 1]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    const vertical = ["wall", "vertical", "door", "window"].includes(p.kind);
    // floor / table planes from the phone's AR: a faint tint, so they never hide the scanned colours
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
      color: KIND_COLOR[p.kind] ?? 0x59636e, transparent: true, opacity: vertical ? 0.22 : 0.12, side: THREE.DoubleSide, depthWrite: false,
    }));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: KIND_COLOR[p.kind] ?? 0x59636e, transparent: true, opacity: 0.25 }));
    m.add(edges);
    this.scene.add(m);
    this.planeMeshes.set(p.id, m);
  }

  removePlane(id) {
    const m = this.planeMeshes.get(id);
    if (m) { this.scene.remove(m); this.planeMeshes.delete(id); }
  }

  updatePhones(phones, selected) {
    const seen = new Set();
    for (const p of phones) {
      if (!p.pos) continue;
      seen.add(p.id);
      let a = this.avatars.get(p.id);
      if (!a) { a = this.makeAvatar(p); this.avatars.set(p.id, a); this.scene.add(a); }
      const dim = p.stale || !p.connected;
      const flash = p.alert && !p.alert.acked && Math.floor(performance.now() / 400) % 2 === 0;
      a.userData.body.material.color.set(flash ? 0xff3b30 : dim ? 0x59636e : p.color);
      a.userData.cone.material.color.set(dim ? 0x59636e : p.color);
      a.position.set(p.pos[0], p.pos[1], p.pos[2]);
      if (p.fwd) a.userData.cone.lookAt(p.pos[0] + p.fwd[0], p.pos[1] + p.fwd[1], p.pos[2] + p.fwd[2]);
      a.userData.ring.visible = p.id === selected;
      this.updateTrail(p);
    }
    for (const [id, a] of this.avatars) if (!seen.has(id)) { this.scene.remove(a); this.avatars.delete(id); }
  }

  makeAvatar(p) {
    const xray = (mat) => Object.assign(mat, { depthTest: false, transparent: true });
    const grp = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 1.0, 4, 12), xray(new THREE.MeshBasicMaterial({ color: p.color, opacity: 0.85 })));
    body.position.y = -0.75; // pose is the camera (eye height); body hangs below it
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.9, 2.5, 24, 1, true), xray(new THREE.MeshBasicMaterial({ color: p.color, opacity: 0.18, side: THREE.DoubleSide })));
    cone.geometry.rotateX(-Math.PI / 2);
    cone.geometry.translate(0, 0, 1.25);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.4, 0.03, 8, 32), xray(new THREE.MeshBasicMaterial({ color: 0xffffff })));
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -1.3;
    const label = this.label(p.name, "#ffffff", [0, 0.45, 0]);
    for (const o of [body, cone, ring, label]) { o.renderOrder = 10; grp.add(o); }
    grp.userData = { body, cone, ring };
    return grp;
  }

  updateTrail(p) {
    if (!p.trail || p.trail.length < 2) return;
    let line = this.trails.get(p.id);
    if (!line) {
      line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: p.color, transparent: true, opacity: 0.5 }));
      this.trails.set(p.id, line);
      this.scene.add(line);
    }
    if (line.userData.n !== p.trail.length) {
      line.geometry.setFromPoints(smoothPath(p.trail).map((c) => new THREE.Vector3(c[0], c[1] - 1.3, c[2])));
      line.userData.n = p.trail.length;
    }
    line.visible = this.showTrails;
  }

  updatePins(pins) {
    const ids = new Set(pins.map((p) => p.id));
    for (const pin of pins) {
      if (this.pinMeshes.has(pin.id)) continue;
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.2), new THREE.MeshBasicMaterial({ color: PIN_COLOR[pin.kind] ?? 0xffffff, depthTest: false, transparent: true }));
      m.renderOrder = 11;
      m.position.set(pin.pos[0], pin.pos[1] + 0.2, pin.pos[2]);
      const text = pin.kind.toUpperCase() + (pin.label ? ": " + pin.label : "");
      m.add(this.label(text, "#ffffff", [0, 0.45, 0]));
      this.scene.add(m);
      this.pinMeshes.set(pin.id, m);
    }
    for (const [id, m] of this.pinMeshes) if (!ids.has(id)) { this.scene.remove(m); this.pinMeshes.delete(id); }
  }

  label(text, color, offset) {
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d");
    ctx.font = "600 40px -apple-system, system-ui, sans-serif";
    c.width = Math.ceil(ctx.measureText(text).width) + 24;
    c.height = 56;
    ctx.font = "600 40px -apple-system, system-ui, sans-serif";
    ctx.fillStyle = "rgba(7,7,8,0.78)";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = color;
    ctx.fillText(text, 12, 42);
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }));
    s.scale.set(c.width / 160, c.height / 160, 1);
    s.position.set(...offset);
    s.renderOrder = 12;
    return s;
  }

  focus(p) {
    if (!p || !p.pos) return;
    const t = new THREE.Vector3(p.pos[0], p.pos[1] - 1, p.pos[2]);
    this.camera.position.add(t.clone().sub(this.controls.target));
    this.controls.target.copy(t);
  }

  render() {
    if (!this.active) return;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    const size = this.renderer.getSize(new THREE.Vector2());
    if (size.x !== w || size.y !== h) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      const pr = this.renderer.getPixelRatio();
      this.target.setSize(w * pr, h * pr);
      this.edl.uniforms.texel.value.set(1 / (w * pr), 1 / (h * pr));
    }
    this.controls.update();
    if (this.gapsDirty && this.gaps.visible && performance.now() - this.lastGaps > 1000) this.refreshGaps();
    if (performance.now() - this.lastShadowFit > 2000) this.fitShadow();
    // 1) points + grid into the target, 2) eye-dome lighting to screen, 3) everything else on top
    this.renderer.setRenderTarget(this.target);
    this.renderer.setClearColor(this.bg, 1);
    this.renderer.clear();
    this.renderer.render(this.cloudScene, this.camera);
    this.renderer.setRenderTarget(null);
    this.edl.uniforms.tColor.value = this.target.texture;
    this.edl.uniforms.tDepth.value = this.target.depthTexture;
    this.renderer.render(this.quadScene, this.quadCam);
    this.renderer.autoClear = false;
    this.renderer.clearDepth();
    this.renderer.render(this.scene, this.camera);
    this.renderer.autoClear = true;
  }
}
