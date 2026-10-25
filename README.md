# FindMyTeam (FMT)

*Formerly Sightline: code folders and the app's bundle ID keep the old name.*

Phones walking around a building share one live 3D map. The incident commander on a laptop sees every responder, their trail, the walls they've mapped and a coloured 3D point cloud of the rooms, in real time, in 2D or 3D. A responder can hold their phone up to a wall and see a colleague on the other side: a coloured ghost drawn through the wall, plus that colleague's live camera.

- **iPhone (13 and up, no LiDAR needed):**
  - ARKit world tracking with gravity alignment
  - plane detection with classification (wall, door, window, floor)
  - one detection image (the marker)
  - **AI depth**: Depth Anything V2, run with Core ML on the phone, calibrated to metres against ARKit's own measurements
  - no RoomPlan, no scene depth, no scene reconstruction
- **Android:** ARCore (marker, planes, **ARCore Depth API**) with the same protocol. It's built: a signed, hardened release APK.
- **Security first:**
  - pairing by single-use QR code, scanned in the app
  - TLS 1.3 pinned to the laptop's certificate
  - **session-only**: leaving the app turns the camera off, disconnects, and the laptop forgets the phone
  - nothing stored on phones
  - commander page only on the laptop
  - see [SECURITY.md](SECURITY.md)

```
server/   FastAPI + uvicorn, in-memory map; two listeners (commander on 127.0.0.1, phones over TLS)
web/      commander page: vanilla JS + canvas (2D) + three.js (3D, served locally), no build step
ios/      SwiftUI + ARKit + RealityKit + Core ML app (XcodeGen project.yml)
android/  Kotlin + ARCore + OkHttp app (Gradle), same protocol
sim/      sim_phone.py: fake phones that pair, pin TLS and stream like real ones
marker/   marker.png + marker.pdf (prints 18.0 cm wide on A4)
tools/    model conversion helpers (xcodegen, JDK, Android SDK and Gradle go here locally; not in git)
```

> **Setup notes:** AI model files aren't in the repo; see [MODELS.md](MODELS.md). For a plain-language tour of the whole system, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## How it works

1. **One shared frame.** Every phone scans the printed marker by the entrance. Only close, face-on sightings count, and the lock is the **median of 30**, shown on the phone as "Locked ±N mm". Whenever the marker comes back into view, the phone **corrects its drift** automatically. The marker's pose gives a frame with:
   - origin at the marker
   - Y up
   - Z pointing out of the wall into the building
   
   Each phone converts everything it sends into that frame, using `T = inverse(S)`, `S = [X Y Z p]`; see `ios/Sightline/SharedFrame.swift`, `android/.../SharedFrame.kt` and `server/sightline/frame.py`. The phone averages 20 sightings of the marker before locking.
2. **Map.** Phones stream:
   - their pose (10 Hz)
   - plane rectangles (walls, doors, windows, floors)
   - coloured 3D points (about 2 Hz)
   - small camera JPEGs (3 Hz)
   
   The server merges points into **5 cm voxels**, which removes duplicates when several people map the same room. A voxel is published only after it has been **seen twice**, which removes speckle. Wall pieces from every phone are **merged into clean straight walls**. The commander gets the fine layer, shaded with eye-dome lighting in 3D; phones get a light **15 cm** layer for their minimap.
3. **AI depth on iPhone.** Depth Anything V2 gives *relative* depth for each pixel. Each frame it's calibrated to metres with a robust fit (RANSAC) against depths ARKit already knows:
   - tracked feature points
   - hits on detected planes, which is what keeps plain or shiny walls working
   
   It then becomes about 5,000 coloured points per frame. The scan rate drops automatically when the phone heats up.
4. **Through walls.** The server counts the physical walls on the straight line between each pair of phones, merging copies of the same wall from different phones. Each phone shows colleagues in its cone of view as ghosts through the wall, with their live camera and "THROUGH 1 WALL". The commander's 3D view draws people with depth testing off, so they're always visible.

## Digital-twin features

- **Search coverage.** The server traces each responder's camera view (about 56° wide, 5 m deep), stopping at walls, so a room behind a wall doesn't count as searched.
  - The commander sees a heatmap: **green** for looked at (brighter means more looks), **red** for inside the building outline but never seen.
  - It also shows **"Searched N%"**.
  - Toggle with the **Coverage** button or C. It works in both 2D and 3D, and for every phone, since it's computed from poses on the server.
- **Fall / man-down alerts.** Raised from each responder's 10 Hz movement:
  - a **possible fall**: the camera drops more than 0.7 m within 2.5 s and stays down
  - **not moving**: less than 25 cm of movement for 30 s
  
  The commander gets a flashing banner, a pulsing ring on the map and an **alarm that repeats until Acknowledge** is pressed (browsers only play sound after the page has been clicked once; the banner says so if it hasn't). **Show** follows that responder. Colleagues' phones show a red banner with distance and walls between, and vibrate. The responder's own phone asks **"Are you OK?"** and sounds a loud **locator chirp** (through the silent switch on iPhone, on the alarm volume on Android) so the team can find them by ear. Chirp and alert stop when they tap "I'm OK" or move more than 0.5 m.
Both are **decision support**: they never replace a physical search.

## Search management, fall-risk zones and routes back to command
- **Rooms** are found from the walls (doors count as boundaries). A room is *searched* once someone stood in it and the cameras saw most of its floor, *glanced* if it was only seen (e.g. from the doorway).
- **Next room / re-check:** each free responder gets one suggestion: "next room: 14 m², 6 m away, not searched yet" or "re-check the room 4 m away: only 35% seen". Optimal assignment on route length. The suggested room is outlined on the map; the text also shows on the responder's phone.
- **Fall-risk zones (yellow, mild):** stairs (+ landings), floor drops, low clutter (12-35 cm) in walkways, and spots where a fall alert went off. No text, just soft yellow areas.
- **Routes back to command (always on):** every responder's safest route to the entrance marker, in their colour. Red hazard areas (1.5 m round a hazard pin) are crossed only if there is no other way; yellow areas are avoided whenever a reasonable detour exists. A route that must cross one is dashed, and the decisions list says so.
- "Searched N%" in the corner is gone; the converge suggestion is off.

## Clean map, uncertainty and A* routes
- **Occupancy with uncertainty** (`occupancy.py`): every 20 cm floor cell gets a probability of being blocked from measured evidence (walls, doors, large obstructions, floor the cameras saw, places people stood), plus an uncertainty.
- **Gap filling:** narrow wall gaps (< 70 cm) between lined-up wall pieces, with no seen opening, are filled as *probably wall* (faint dashed line on the 2D map); door-sized gaps (70 cm-1.2 m) only get expensive, never sealed. Unseen floor takes its neighbours' open-floor belief; uncertainty grows with distance from anything measured.
- **A\*** finds every route (back to command, to the next room, to the exit): cost = distance x (1 + 1.5 x uncertainty) x zone penalty (yellow fall-risk 6x, red hazard 250x).
- **Smoother registration:** both faces of a wall merge into one line in the middle; walls that move a few cm keep their place (no wobble); the laptop fine-aligns each phone to the first phone's walls (small, damped steps: "Lined up Bob's map with Alice's"); phones lock on from closer / more face-on and ease drift corrections in over two steps.
- **Smoother rendering:** 4x anti-aliasing in 3D, shrink-free (Taubin) surface smoothing with block borders fixed, smooth normals from the 3D distance field, smoothed trails.

## Testing the detectors on the real building (honest evaluation)
1. Walk two separate areas and **Save map** after each (a third, spare area C is optional).
2. In **Saved maps**, press **Label** on each: choose its area (**A** = training, **B** = test; **C** = optional spare), then click the real rooms' corners (**Room**) and the real trip / fall spots (**Fall-risk spot**). **Save labels**.
3. Run `cd server && ../.venv/bin/python -m sightline.evaluate`. It tunes the room and fall-risk settings on area A only, scores them once on area B, and writes `evaluation/report-<date>.md` (data used, precision / recall / F1, IoU, settings). Area C stays untouched until you add `--spare` once at the end.
4. Tuned settings go to `server/data/tuned.json`; restart the server to use them. Defaults are kept unless tuning is strictly better on A.

## Ask Gemini (commander's AI assistant)
- **Ask** button (sparkle, bottom left of the map, or press A) opens a full-screen chat. Type the questions rescuers radio in; quick buttons for "Where is everyone?", alerts, unsearched areas and a summary.
- **Setup, once:** open Ask on the server laptop and paste your Gemini API key (or set `GEMINI_API_KEY` before starting the server). It's stored in `server/data/gemini.key` (owner-only), never sent to a browser, logged or put in a URL. Only the laptop itself can set it.
- Each question goes to Google Gemini (tries `gemini-3.5-flash`, then `gemini-flash-latest`, the Flash-Lite models and Gemma 4; set `GEMINI_MODEL` to put your own first. A retired, over-quota or busy model is skipped automatically) with a text summary of the live incident: responder names, positions in metres from the entrance, alerts, hazards, exits, stairs, coverage, planner suggestions, last 15 timeline events. No camera images.
- The conversation lives only in that browser tab (New chat clears it). Up to about 12 questions a minute.
- The "converge" suggestion (route a colleague to someone who is down) is switched off for now; the fall / no-movement alarm is unchanged.

## Voice guidance on phones (ElevenLabs)
- **Guide** button (above the minimap on iPhone): **Guide me to → Command post / Nearest exit**, and **Voice → All guidance / Alerts and hazards only / Off**. A green arrow card shows the route's direction and metres left; × stops it. A waypoint route from the commander is guided the same way, automatically.
- **What it says**, most urgent first (an alert cuts off a door announcement):
  - **Alerts:** a colleague's possible fall / not moving (distance, side, walls between), a hazard pin ahead ("Danger. Gas leak, 2 metres ahead. Keep away."), "Turn back now".
  - **Trip hazards** a few metres ahead from the yellow fall-risk zones: stairs, floor drops, low clutter in the walkway, places someone fell. Once per approach; nothing about things beside or behind you.
  - **Navigation** along the A* route: "Guiding you to the nearest exit. 23 metres.", "In 4 metres, turn right.", "Turn left." / "Turn around." when you face away from the route, every 10 m "13 metres to go", "You have arrived".
  - **Doors** within 5 m ahead, exit doors called out as such.
- **How:** the laptop works out the cues (`server/sightline/voice.py`, twice a second) from the same routes, zones, pins and walls the commander sees, so iPhone and Android hear the same thing. Left / right comes from where the phone's camera points, or from the walking direction when the phone points at the floor.
- **Setup, once:** commander page → *Voice guidance (phones)* → paste your ElevenLabs API key (optionally a voice ID; default "George"). Or set `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` before starting the server. It's stored in `server/data/elevenlabs.key` (owner-only), set only from this Mac, never sent to a phone or browser. Model: `eleven_flash_v2_5` (override with `ELEVENLABS_MODEL`).
- **Privacy:** only fixed phrases go to ElevenLabs (directions, distances, kinds of hazard). Anything with a name or a team-typed pin label is spoken by the phone's own voice. Phones fetch audio only for cues the laptop issued; audio is cached in memory.
- **No key, no internet, or slow?** The phone speaks with its built-in voice instead (urgent cues wait at most 0.8 s for ElevenLabs). Voice is decision support, like everything else here.

## Doors and exits (AI)
- **Phones:** YOLOv8n trained on Open Images V7 (Ultralytics, 601 classes, Core ML, 6.8 MB) replaces YOLOv3-Tiny. It outlines ~20 furniture types (sharper 3D furniture, as before) and **doors**. Door outlines are placed in 3D using the depth frame. No class names leave the phone. Licence: AGPL-3.0 (fine for demos; commercial use needs an Ultralytics licence).
- **Laptop:** SegFormer-B2 (NVIDIA, ADE20K, 53 MB, ~45 ms per snapshot) replaces SegFormer-B0 for wall gaps / pillars / windows, and replaces MobileNetV2 as the furniture double-check (share of furniture pixels in the object's region).
- A wall gap becomes a **door** (green, open for routes) when the scene AI says door, or the phone's detector saw a door there twice.
- **Exits:** doors in the building's outside walls join the entrance and rally points as exits. Exit routes and turnaround times use the nearest one, and the timeline says "Exit door found".

## Windows, blinds, slanted walls and pillars
- **Windows:** glass is see-through, so depth leaves a gap in the wall. A gap in a wall line (0.45-3 m) becomes wall when there's a sill under it (something 0.4-1.4 m high on the wall line), or when the scene AI says window / blind / curtain / wall. Doorways (open to the floor, or labelled door) stay open for routing.
- **Blinds and curtains:** tall surfaces that stop short of the floor (0.8 m+ tall, reaching 1.6 m+) count as wall.
- **Slanted / angled walls:** walls that don't line up with the room are kept when they're at least 1.5 m long (shorter off-axis lines are treated as noise).
- **Pillars:** tall, narrow, free-standing shapes (up to 1 m across, floor to 1.9 m+) are drawn as a small square of wall, and count for routing and x-ray wall counting. Shorter ones need the scene AI to say "column".
- **Scene AI (laptop):** SegFormer-B0 (NVIDIA, pre-trained on ADE20K) converted to Core ML (`server/models/SegFormerADE.mlpackage`, 7 MB). It segments the phones' camera snapshots (about 20 ms each on an M1) and votes on each gap / pillar candidate; a label needs two agreeing snapshots. Labels stay on the laptop; the maps only show walls. Re-create it with `tools/convert-venv/bin/python tools/convert_segformer.py`.

## Hazard warnings
- Drop a hazard with the red **Hazard** button on a phone (it lands where you're pointing) or **+ Hazard** on the commander map.
- Anyone heading towards a hazard gets **"Hazard ahead"** at 3 m, with **Avoid** or **Proceed anyway**.
- The commander gets a red banner and a timeline entry if they tap Proceed, or come within 1.5 m anyway. Walking more than 4 m away resets it.

## Entrance marker on the laptop screen
- A 13" laptop screen is under 18 cm tall, so on screen the same marker is drawn at exactly **15 cm**. The page sizes it from this Mac's real screen size; check it against the 10 cm ruler shown and fine-tune with − / + if needed.
- Choose **Printed · 18 cm** or **Screen · 15 cm** under *Entrance marker*. Phones are told the size and lock on again by themselves.
- The marker now says FINDMYTEAM: reprint `marker/marker.pdf` if you use paper; old SIGHTLINE prints won't match.

## Commander page on another computer (Chrome)
- On the server laptop, press **+ Add a computer** (section *Other computers*). It shows a one-time link + QR code (works once, for 10 minutes), e.g. `https://<laptop-ip>:8444/?screen=…`.
- Open it in Chrome on the other computer (same Wi-Fi). Chrome warns the connection isn't private because the laptop makes its own certificate: check the fingerprint shown on the laptop, then **Advanced → Proceed**.
- That computer gets its own session (encrypted, memory only, ends when the server restarts) with the full commander page, except showing the screen marker and adding / revoking other computers, which stay on the laptop. **Revoke** cuts it off immediately.
- Without a link the Wi-Fi address answers nothing but the sign-in page.

## Saved maps
- **Save map** stores walls, AR planes, 3D surfaces, coloured points, coverage and pins in `server/data/maps.db`, encrypted with the laptop-only key. No camera images, names or trails.
- **Load** replaces the map on screen (live phones keep mapping on top). **Delete** removes it from the file for good.

## 3D surfaces, walls and furniture (live fusion)

- **Fused surfaces.** Phones send a small metric depth image (128×96, about 60 KB, twice a second) with their pose. The laptop fuses all of them into one TSDF volume (4 cm cubes, allocated only near surfaces) and streams solid, lit, coloured surface meshes to the commander and Map tabs. Averaging many noisy camera-only depth maps is what makes plain walls, tables and chairs appear. **Points** toggles the raw point view back.
- **Walls from depth.** Any surface taller than about 1.2 m becomes a wall, even where ARKit never found a plane (plain, textureless walls).
- **Furniture from height.** Surfaces at 0.3–0.6 m are seats, 0.6–1.0 m tables, 1.0–1.9 m other obstacles. They're drawn as labelled boxes, and routing walks around them.
- **Simulator result:** 3/3 tables and 3/3 chairs found, with 0–1 false objects.

## Pre-trained neural networks (nothing is trained on your data)

| Network | Where | Job |
|---|---|---|
| Depth Anything V2 (small) | iPhone | depth for every pixel from one camera |
| YOLOv3-Tiny | iPhone, about 1 Hz, paused when the phone is warm | outlines furniture (no names) so fusion trusts depth there 2.5× more: sharper furniture, sooner |
| Core Motion activity | iPhone (built in) | walking, running, stationary |
| MobileNetV2 (ImageNet) | **laptop**, Core ML, about 100 MB | second opinion on 3D objects: crops the object from real posed snapshots; **two agreeing views** confirm it |

**What the furniture models change:** furniture surfaces in the 3D scan come out sharper and sooner. Furniture is never drawn as boxes or labelled on any map; the laptop only uses it to route people around obstacles.

**What they never do:**
- **No labels:** item names never leave the laptop's memory and log.
- **Nothing invented:** no network adds or draws geometry. Every surface is measured depth.

Two physics checks sit beside them, with no network involved:
- **Accelerometer fall signature:** free fall of at least 80 ms, then a hit of more than 2.5 g.
- **"Phone lying still":** no hand tremor.

"I'm OK" and **Confirm real** record whether alerts were right.

**Stability:**
- **Telemetry:** every 5 s each phone reports its memory, heat level and frame rates. They appear in the responder list and as a once-a-minute line in the server log.
- **Memory warnings:** heavy work pauses for 15 s.
- **Heat:** YOLO stops when the phone is warm, and depth slows when it's hot.

## Predictions and decisions

- **Predictions:**
  - each responder's position while their phone is stale (Kalman filter, with a growing uncertainty circle)
  - learned walking pace
  - **exit route** over the walkable map (doors open, walls and furniture blocked)
  - **turnaround time** against the per-responder time budget you set on the commander page
  - **isolation** (no teammate within 10 m by route for over a minute)
  - **search ETA**, fitted from the coverage curve
- **Decisions**, ranked:
  1. nearest colleague **by route** to anyone down
  2. who must turn back
  3. who goes to a possible person
  4. isolation warnings
  5. an optimal assignment of free responders to the largest unsearched areas (Hungarian algorithm on route time)
  
  Routes are drawn on the map. Each phone shows its own suggestion and turnaround countdown.
- **Timeline:** joins, alerts, labels, sightings and search milestones, with mission time (T+mm:ss).

These are suggestions for the commander, not orders. Accuracy on real phones will be lower than on simulated movement (99.5%). Collect labels with recordings and the model improves.

## Recording (opt-in, encrypted)

Nothing is saved unless the commander starts it:
- **Record session:** compact time series (positions at 2 Hz, ML features at 1 Hz, events, labels), about 1–2 MB per hour.
- **Room capture (90 s):** depth frames, planes and about 1 camera frame per second from every connected phone. Used once to tune fusion, then delete it.
  Replay it through the 3D fusion with different settings (outlines on/off, surface strictness, truncation) and compare:
  `cd server && ../.venv/bin/python -m sightline.replay --measure ../roomdata/measurements.json`
  Fill in `roomdata/measurements.json` with tape measurements first and the table also shows errors in cm.

Files live in `server/data/recordings/`. Each line is encrypted (Fernet: AES with an HMAC) with `server/data/record.key` (mode 0600). They're deleted after 7 days and capped at 200 MB in total.

## Map tab on phones

The iPhone's **Map** button opens the commander's exact map: 2D and 3D, decisions, timeline, live camera views. It follows your own position. Phones can drop pins but can't pair, revoke, record or clear the map. The page and its data come through the app's own pinned, token-authenticated connection; the web view itself makes no network requests and stores nothing.

## Setup

### 1. Laptop: server and commander page

```bash
cd sightline
python3 -m venv .venv
.venv/bin/pip install -r server/requirements.txt
cd server && ../.venv/bin/python -m sightline
```

It prints a **Commander** link like `http://127.0.0.1:8000/?admin=…`. Open it on this Mac. The token is moved into a cookie, then removed from the address bar.
- The page only works on this Mac; other machines can't reach port 8000.
- Phones use `https://<laptop-ip>:8443`.
- The first start creates `server/data/` (TLS certificate and key only), readable only by your user. Delete it to rotate the certificate. Pairings are never stored.

Phones and laptop must be able to reach each other. Venue Wi-Fi often blocks this; the reliable option is the **iPhone's Personal Hotspot with the Mac joined to it**. Re-pair the phones after changing networks, because the QR code contains the laptop's IP.

### 2. Marker

Print `marker/marker.pdf` at **100% / Actual size** and check the 10 cm ruler. Tape it flat on a wall by the entrance, arrow up, about 1.4 m high.

### 3. iPhone app

On this Mac, with Xcode installed and the iPhone **plugged in once**:

```bash
./ios/build.sh install
```

Set your Apple Team ID (a free Personal Team works) in `ios/project.yml`, or pass it as `DEVELOPMENT_TEAM=<your team ID> ./ios/build.sh install`.

- **No wireless link:** Xcode's "Connect via network" would allow wireless installs, but it keeps a standing link between the Mac and the phone, so we don't use it (see SECURITY.md). Plug in to (re)install.
- **Free-account limits:** apps expire after 7 days (re-run `build.sh install`), and there's a maximum of 3 iPhones. Installing on other people's iPhones with no cable needs the paid Apple Developer Program and TestFlight.
- On the phone, the first time: Settings → General → VPN & Device Management → trust the developer. If asked, Settings → Privacy & Security → Developer Mode → On.

### 4. Pair a phone, every time the app opens

1. On the commander page, click **+ Add phone**. A QR code appears; it's single use and lasts 10 minutes.
2. In Sightline:
   - **iPhone:** tap **Scan QR code**.
   - **Android:** the camera is already scanning.
   
   Point it at the code.
3. Check the server address, enter your name, and tap **Pair**.
4. Allow camera and local-network access, then point at the marker until it locks.

Pairing is **session-only**: leaving the app, locking the phone or tapping **Disconnect** ends it, and the laptop forgets the phone (its walls and points stay on the map). The next time the app opens, scan a fresh code. **Revoke** on the commander page ends a session immediately.

### 5. Android app

Built already: `android/app/build/outputs/apk/release/app-release.apk`. To rebuild after changes:

```bash
cd android && JAVA_HOME=../tools/jdk/Contents/Home ./gradlew :app:assembleRelease
```

The toolchain (JDK 17, Android SDK 35, Gradle 8.9) lives in `tools/`. The signing key is `android/release.keystore`, with its passwords in `android/keystore.properties`, both readable only by you. **Back them up privately**: updates must be signed with the same key.

To install on an Android phone, with no cable and no USB debugging:
1. Commander page → **+ Add phone**. The panel also shows an **Android: first install** QR code.
2. Scan that with the phone's camera app. The browser warns about the laptop's self-signed certificate; continue, and check the SHA-256 shown on the commander page if you want.
3. Allow "install unknown apps" for the browser once, and install.
4. Open Sightline and scan the pairing QR.

It needs Google Play Services for AR (ARCore). Depth points need a phone with the ARCore Depth API, which most mid-range phones from 2020 onward have. The app refuses to run on rooted devices or if re-signed.

### 6. No phones? Simulate them

The simulator pairs and connects exactly like a phone: one-time code, pinned TLS, bearer token. It also produces AI-depth-style 3D points, including furniture.

```bash
SIGHTLINE_ADMIN_TOKEN=<token from the server's startup line> \
  .venv/bin/python sim/sim_phone.py --host 127.0.0.1                    # 3 phones touring
  # --phones 4 | --scenario xray | --check --duration 8 (self-test, exit 0 = pass)
```

### Tests

```bash
cd server && ../.venv/bin/python -m pytest -q          # security, protocol validation, voxels, end-to-end
swiftc -O ios/Sightline/SharedFrame.swift ios/SharedFrameTests/main.swift -o /tmp/sf && /tmp/sf
```

## Using it

**iPhone:**
- **Top bar:** Live/Offline (with the exact error when offline), surfaces found, tracking state.
- **Centre:** a crosshair with the **distance to the wall** it points at.
- **Bottom right:** the **minimap**: shared walls, obstacles from everyone's 3D scan, colleagues, pins, and you as an arrow.
- **Bottom left:** 3D-scan stats.
- **Point at a colleague, through walls:** their ghost plus a live camera card.
- **Buttons:**
  - ↻ re-aligns on the marker
  - **3D** turns AI scanning on or off
  - **X-ray** turns the x-ray card on or off
  - **Pin** drops a hazard, victim or note
  - ⚙ shows pairing details, the scanning toggle and Unpair

**Commander page:**
- **2D map:** walls, doors, windows, floor, obstacle cells from the 3D scan, view cones, trails, x-ray lines, pins.
- **3D view:** the coloured point cloud, wall planes, and people visible through walls. Drag to orbit, scroll to zoom, right-drag to pan.
- **Sidebar:**
  - **+ Add phone** and **Paired devices**, with Revoke
  - the responder list, showing distance from the entrance, distance to the wall in front, battery and platform
  - the selected responder's live camera
- **Shortcuts:** 3 switches between 2D and 3D, F auto-fits, L follows, T toggles trails, Esc cancels.

## Protocol (phones → `wss://<laptop>:8443/ws/phone`, `Authorization: Bearer <token>`)

| text message | fields |
|---|---|
| `pose` (10 Hz) | `pos`, `fwd` (shared frame), `tracking`, `wall_dist` |
| `status` | `aligned`, `tracking`, `battery` |
| `plane` / `plane_remove` / `planes_clear` | `id`, `kind` (`wall door window floor ceiling table seat vertical horizontal`), `corners` (3–8 × [x,y,z]) |
| `pin` / `pin_remove` | `kind` (`hazard victim note rally`), `label`, `pos` |
| `xray` | `target` (peer id or null) |
| `detect` | `kind` (`person`), `pos` (shared frame), `conf` (0–1) |
| `ok` | the responder answered "I'm OK" |
| *binary* `0x03` | depth frame: `u8 ver | u16 w | u16 h | f32[16] camera→shared | f32 fx,fy,cx,cy | u16 depth mm | u8 rgb` |

| binary message | layout |
|---|---|
| `0x01` + JPEG | camera thumbnail, ≤ 300 KB, 3 Hz |
| `0x02` + N × 16 B | points: float32 x,y,z + uint8 r,g,b,confidence, ≤ 12,000 per message |

From the server, a phone receives:
- `welcome`
- `state` (10 Hz: `peers` with `distance`, `walls_between` and `alert`, plus `pins` and `me.alert`)
- `walls` (merged walls, when they change)
- `planes` (`upsert`/`remove`/`reset`)
- binary voxels: `0x02 | layer | size f32 | N × (int16 x,y,z + uint8 r,g,b)`

Everything is validated: unknown fields, non-finite numbers and coordinates over 500 m are dropped. There are size limits and per-device rate limits.

HTTP: `POST /api/pair`, `GET /api/frame/<id>.jpg` (needs a token), `GET /download/sightline.apk?c=<live code>`.

## Limits worth knowing

- **Monocular depth is approximate:** expect 5–15% error. It's best within 4 m, and worse on mirrors and glass. Plane hits keep the calibration honest on plain walls.
- **ARKit and ARCore drift** a few cm per 10 m. Re-align at the marker now and then. Don't move the marker.
- **The map lives in memory:** restarting the server clears the map, but keeps pairings and the certificate.
- **Battery and heat:** AI scanning is heavy. The app slows the scan automatically when the phone is hot, and you can turn it off with **3D**.

## Demo script (≈3 minutes)

1. **Setup (before judges arrive).**
   - Server running, commander page full-screen in **3D view**, marker taped by the door.
   - Two paired phones (Alice and Bob) with the app open.
   - Hit **Clear map**.
2. **"Everyone shares one map, securely."** Show **+ Add phone**: a QR code that works once, with an encrypted connection pinned to this laptop. Alice and Bob each point at the marker; their avatars appear at the entrance.
3. **Live 3D mapping.** Both walk into different rooms, sweeping slowly across walls and furniture. On the laptop, the coloured 3D point cloud and walls build up in real time. Orbit the view: tables and chairs show up, not just walls.
4. **Search coverage.** Switch the coverage heatmap on: the red unsearched areas shrink as people sweep rooms, and "Searched N%" climbs.
5. **Commander awareness.**
   - Switch to 2D: obstacles, trails, and the distance to the wall in front of each responder.
   - Click Bob to show his live camera.
   - Drop a **Hazard**: it appears in Bob's AR view and on everyone's minimap.
6. **Man down.** A responder lowers their phone quickly to the floor and stays still. The laptop flashes "POSSIBLE FALL", colleagues' phones buzz with the distance and walls to them, the laptop alarm keeps sounding until acknowledged, and the responder's phone asks "Are you OK?" while chirping so the team can find them.
7. **The x-ray moment.** Alice holds her phone up to the wall between her and Bob. Bob's ghost appears *through* the wall, with his live camera and "THROUGH 1 WALL". On the laptop, a purple line marks "Alice 👁 Bob".
8. **Security beat.** Click **Revoke** on a phone: it drops off instantly and has to be paired again.
9. **Close.** "No LiDAR, no special hardware: a printed marker and the phones responders already carry, iPhone or Android."

If Wi-Fi or hardware fails, the backup is `sim/sim_phone.py --phones 4` (coverage and automatic person pins), then `--scenario mandown` (a fall alert), then `--scenario xray`. The laptop view looks identical.
