# FindMyTeam: how it works (team guide)

A plain-language guide to the whole system, so anyone on the team can explain it and answer questions.
For setup commands and technical detail, see [README.md](README.md). For the security design, see [SECURITY.md](SECURITY.md).

> **One-line pitch:** responders' ordinary phones map a building as they walk. Everyone, including the incident commander on a laptop outside, shares one live 3D map: where every responder is, what has been searched, what's dangerous, and the safest way out.

---

## 1. The big picture

There are three parts:

| Part | What it is | Runs on |
|---|---|---|
| **Phone app** | Camera + AR view for each responder. Maps the building, shows colleagues through walls, gives alerts and voice guidance. | iPhone (13 and newer, no LiDAR needed) and Android (ARCore) |
| **Server** | The "brain". Merges every phone's data into one map, finds routes, spots falls, plans the search. | The commander's laptop (Python) |
| **Commander page** | The live map in a web browser: 2D and 3D, alerts, timeline, AI assistant. | The laptop's browser (Chrome); other computers by invitation |

```
  Phone (Alice) ─┐                                  ┌─ Commander page (laptop browser)
  Phone (Bob)   ─┼── encrypted Wi-Fi link ──► SERVER ─┤
  Phone (Chen)  ─┘   (pose, walls, 3D points,        └─ Other computers (one-time link)
                      depth, camera thumbnails)
         ◄── shared map, colleagues, alerts, routes, voice cues ──
```

**Everything stays local.** Phones talk only to the laptop over the local Wi-Fi (or the iPhone's hotspot). Nothing goes to a cloud, except two optional extras: Gemini (the commander's questions) and ElevenLabs (the voice audio).

---

## 2. How a session goes, step by step

1. **Start the server** on the laptop. It prints a commander link; open it in Chrome.
2. **Tape the entrance marker** (a printed sheet, 18 cm wide) on a wall by the entrance, or show it on the laptop screen (15 cm).
3. **Pair each phone:** on the commander page click **+ Add phone**. A QR code appears; the phone scans it, the responder types their name and taps **Pair**.
4. **Each phone looks at the marker** until it says "Locked ±N mm". From then on, it knows exactly where it is relative to the entrance.
5. **Responders walk in.** Their phones map walls, doors, furniture and floor as they go. The commander watches the map build live.
6. **The server keeps analysing:** who is where, what's searched, who's in trouble, routes out. It sends advice to the commander and guidance to each phone.
7. **Leaving the app ends the session** for that phone: camera off, disconnected, and the laptop forgets the phone. The map it made stays.

---

## 3. The core idea: one shared map (the marker)

**The problem:** every phone's AR system (ARKit / ARCore) tracks movement in its *own* private coordinates, with an origin wherever the app started. Two phones' maps don't line up.

**The fix:** everyone scans the **same printed marker** at the entrance. The marker becomes the shared origin:
- the **origin (0, 0, 0)** is the marker itself
- **Y** points up
- **Z** points out of the wall, into the building
- positions everywhere are "metres from the entrance marker"

Each phone works out the marker's position and angle in its own world, then converts everything it sends into the shared frame (maths: `T = inverse(S)`, where `S` is the marker's pose).

**Making it accurate:**
- Only **close, face-on** sightings count.
- The phone takes the **median of 30 sightings** before locking, which throws out bad ones. It shows "Locked ±N mm".
- **Drift correction:** AR tracking drifts a few cm every 10 m. Whenever the marker comes into view again, the phone corrects itself, eased in over two steps so the map doesn't jump. **Realign** forces a fresh lock.
- **Fine alignment between phones:** the laptop also lines up each phone's walls with the first phone's walls in small, careful steps ("Lined up Bob's map with Alice's").

---

## 4. How the map is built

### What each phone sends
| Data | How often | Used for |
|---|---|---|
| **Pose** (position + where the camera points) | 10 times a second | position on the map, trails, fall detection, routes |
| **Planes** (flat surfaces from ARKit/ARCore: wall, door, window, floor) | when found / updated | walls and doors |
| **Coloured 3D points** | about 2 a second | the 3D point cloud |
| **Small depth images** (128×96) | twice a second | solid 3D surfaces |
| **Camera thumbnails** | 3 a second | live camera views, x-ray card, scene AI |

### Depth without LiDAR (iPhone 13)
The iPhone 13 has no depth sensor, so the app runs an AI model, **Depth Anything V2**, on the phone. It guesses depth for every pixel, but only *relatively* ("this is further than that"), not in metres.

To turn it into metres, every frame it's **calibrated against points whose real distance ARKit already knows**: tracked feature points, and hits on detected planes (which is what makes plain white walls work). A robust fit (RANSAC, which ignores outliers) gives the scale. The result is about 5,000 coloured 3D points per frame.

If the phone has LiDAR, it uses real measured depth instead, and trusts it more.

### Merging everyone's data on the laptop
- **Points → 5 cm cubes (voxels).** Points from several phones in the same 5 cm cube merge into one, which removes duplicates. A cube only appears once it has been **seen twice**, which removes speckle.
- **Depth → solid surfaces.** All depth images are fused into one 3D volume (a TSDF, 4 cm cells) and turned into lit, coloured surface meshes. Averaging many noisy depth maps is what makes plain walls, tables and chairs appear solid.
- **Walls get cleaned up.** Wall pieces from every phone are merged into clean straight walls; both faces of a wall become one line; walls that wobble by a few cm stay put. Any surface taller than about 1.2 m becomes a wall, even if ARKit never found a plane there.
- **Furniture from height:** 0.3–0.6 m seats, 0.6–1.0 m tables, 1.0–1.9 m other obstacles. Routes go around them.
- **Phones get a lighter copy** (15 cm cubes) for their minimap; the commander gets the fine one.

### Tricky things it handles
- **Windows:** glass leaves a gap in the depth. A gap with a sill under it, or that the scene AI calls a window / blind / curtain, is filled in as wall.
- **Doors:** a wall gap becomes a door (open for routes) when the scene AI says door, or the phone's detector saw a door there twice.
- **Blinds / curtains:** tall surfaces that stop short of the floor count as wall.
- **Slanted walls:** kept if at least 1.5 m long; shorter off-angle lines are treated as noise.
- **Pillars:** tall, narrow, free-standing shapes become a small square of wall.
- **Narrow gaps (< 70 cm)** between lined-up wall pieces are filled in as "probably wall" (faint dashed line). Door-sized gaps are never sealed, only made "expensive" for routes.

---

## 5. Features for responders (the phone app)

| Feature | What it does |
|---|---|
| **AR camera view** | The live camera with the map overlaid. A crosshair shows the **distance to the wall** you're pointing at. |
| **Top bar** | Live / Offline (with the reason), surfaces found, tracking quality. |
| **Minimap** | Shared walls, obstacles from everyone's scan, colleagues, pins, and you as an arrow. |
| **X-ray** | Point at a wall: colleagues on the other side appear as coloured ghosts with their **live camera** and "THROUGH 1 WALL". |
| **Compass** | Where colleagues are, with distance and walls between. |
| **Hazard button** | Drops a red hazard pin where you point. Everyone heading towards it is warned. |
| **Pin menu** | Drops a victim or note pin. |
| **Realign** | Lock on to the marker again. |
| **3D on/off** | Turns AI depth scanning on or off (saves battery). |
| **Map** | Opens the commander's full map (2D/3D) on the phone, following you. Read-only apart from pins. |
| **Guide** | Voice + arrow guidance to the **command post** or **nearest exit**, and voice settings (all / alerts only / off). |
| **Waypoint card** | When the commander assigns you a route: "Waypoint 2 of 4", distance and an arrow. |
| **Plan card** | Your suggested next task, turnaround countdown and distance to the exit. |
| **"Are you OK?"** | Appears if the system thinks you fell or stopped moving, with a loud locator chirp. |
| **Settings** | Pairing details, scanning toggle, Disconnect. |

---

## 6. Features for the commander (the web page)

| Feature | What it does |
|---|---|
| **2D map** | Walls, doors, windows, floor, obstacles, view cones, trails, x-ray lines, pins. |
| **3D view** | Coloured point cloud / solid surfaces, walls, people always visible through walls. Orbit, zoom, pan. |
| **Responder list** | Each person: distance from the entrance, distance to the wall ahead, battery, platform, status. Click for their **live camera**. |
| **Search coverage** (C) | Heatmap: green = looked at, red = inside the building but never seen. "Searched N%". |
| **Rooms** | Found automatically from the walls. *Searched* once someone stood in it and saw most of its floor; *glanced* if only seen from the doorway. |
| **Fall / man-down alerts** | Flashing banner, pulsing ring, alarm that repeats until **Acknowledge**. **Show** follows that person. |
| **Hazards** | **+ Hazard** on the map; red banner if someone goes near one anyway. |
| **Fall-risk zones** | Soft yellow areas: stairs, floor drops, low clutter in walkways, places someone fell. |
| **Routes back to command** | Always on: each responder's safest route to the entrance, in their colour. |
| **Waypoint routes** | Drop waypoints, assign them to a responder, **best order** (shortest walk). The phone guides them stop by stop. |
| **Decisions list** | Ranked suggestions (see section 8). |
| **Timeline** | Everything that happened, with mission time (T+mm:ss). |
| **Ask Gemini** (A) | AI assistant that answers questions about the live situation. |
| **Voice guidance setup** | Paste the ElevenLabs key once for natural phone voices. |
| **Entrance marker** | Choose printed (18 cm) or on-screen (15 cm); phones re-lock automatically. |
| **Saved maps** | Save, load, delete maps (encrypted). |
| **Recording** | Opt-in, encrypted session recording and 90 s room capture. |
| **Other computers** | Invite a second laptop with a one-time link; **Revoke** any time. |
| **Paired devices** | Every phone, with **Revoke**. |

---

## 7. Safety features in detail

### Fall and "not moving" alerts
- **Possible fall:** the phone drops more than **0.7 m within 2.5 s** and stays down. A second signal comes from the accelerometer: free fall of at least 80 ms, then a hit over 2.5 g.
- **Not moving:** less than **25 cm** of movement for **30 s**.
- **What happens:**
  - **Commander:** flashing banner, ring on the map, repeating alarm until Acknowledge. (Browsers only play sound after the page has been clicked once; the banner says so.)
  - **Colleagues' phones:** red banner with distance and walls between, vibration, and a voice alert.
  - **The responder's own phone:** "Are you OK?" plus a loud **locator chirp** through the silent switch, so the team can find them by ear.
  - It stops when they tap **I'm OK** or move more than 0.5 m.
- "I'm OK" and **Confirm real** are recorded as labels, to measure and improve accuracy.

### Hazard warnings
- Anyone heading towards a hazard pin gets **"Hazard ahead"** at **3 m**, with **Avoid** or **Proceed anyway**.
- The commander is flagged if they tap Proceed, or get within **1.5 m** anyway. Walking more than 4 m away resets it.

### Time budget (turnaround)
- The commander sets a time budget per responder (for example, air supply). The system works out how long the walk out takes, and when each person must **turn back**, with a safety margin.

### Isolation
- Warns when someone has had no teammate within 10 m (by route) for over a minute.

---

## 8. Pathfinding: how routes work

**Step 1, the grid.** The floor is divided into **20 cm squares**. For each square, the system estimates how likely it is to be blocked (walls, furniture, obstacles) and how **uncertain** that is. Floor nobody has seen yet is uncertain, more so the further it is from anything measured.

**Step 2, the cost.** Walking through a square costs:

```
cost = distance × (1 + 1.5 × uncertainty) × zone penalty
zone penalty: normal 1 · yellow fall-risk 6 · near a red hazard (1.5 m) 250
```

So routes prefer **short, well-mapped, safe** floor. They take a reasonable detour around a fall-risk area, and only go near a hazard if there's no other way at all.

**Step 3, the search.** **A\*** (a standard shortest-path algorithm) finds the cheapest route. **Dijkstra** is used to find the nearest of several exits at once.

**Destinations:**
- **Command post:** the entrance marker.
- **Exits:** the entrance, rally-point pins, and **doors in the building's outside walls** (found automatically; "Exit door found" appears on the timeline).
- **Waypoints:** pins the commander drops and assigns.

**It's recalculated every second** from where each person is standing, so routes follow people as they move and as the map grows. A route that has to cross a risky area is drawn dashed, and the decisions list says so.

### Planner predictions and decisions
- **Kalman filter** per responder: smooths movement, learns walking pace, and predicts where someone is if their phone goes quiet (with a growing uncertainty circle).
- **Exit time and turnaround time** from each person's route and pace.
- **Search ETA** from how fast coverage is growing.
- **Ranked decisions:**
  1. the nearest colleague **by route** to anyone who is down
  2. who must turn back
  3. who should check a possible person
  4. isolation warnings
  5. **next room** for each free responder: the best split of free people over unsearched rooms (Hungarian algorithm on route time), or "re-check" a room that was only partly seen.

These are **suggestions for the commander, never orders.**

---

## 9. Voice guidance (ElevenLabs)

**What responders hear**, most urgent first (an urgent message cuts off a less urgent one):
1. **Alerts:** "Alert. Bob may have fallen, 12 metres ahead on your left, through 1 wall." · "Danger. Gas leak, 2 metres ahead. Keep away." · "Turn back now."
2. **Trip hazards ahead:** "Stairs ahead on your right, in 2 metres. Take the handrail." · "Trip hazard straight ahead… low obstacle on the floor." Said once per approach; nothing about things beside or behind you.
3. **Navigation:** "Guiding you to the nearest exit. 23 metres." · "In 4 metres, turn right." · "Turn left." / "Turn around." · "13 metres to the command post." · "You have arrived."
4. **Doors** within 5 m ahead: "Door ahead on your left, 3 metres." / "Exit door…"

**How it works:**
- The **laptop decides what to say** (twice a second), using the same routes, zones, pins and doors as the map. So every phone says the same things in the same way.
- **Left / right** is worked out from where the phone's camera points, or from the walking direction if the phone points at the floor.
- **ElevenLabs** turns the phrases into a natural voice. The key stays on the laptop; audio is cached, since phrases repeat.
- **Fallback:** with no key, no internet, or a slow reply, the phone uses its own built-in voice. Urgent cues wait at most 0.8 s.
- **Privacy:** only fixed phrases go to ElevenLabs. Anything with a **name** or a **pin label** is spoken by the phone itself and never leaves the local network.
- Currently on **iPhone**; Android receives the cues but doesn't speak them yet.

---

## 10. The AI models (all pre-trained; nothing is trained on your data)

| Model | Where | Job |
|---|---|---|
| **Depth Anything V2** (small) | iPhone | Depth for every pixel from one camera |
| **YOLOv8n** (Open Images, 601 classes) | iPhone | Outlines furniture and **doors**, so the 3D scan is sharper there |
| **Core Motion activity** | iPhone (built in) | Walking / running / stationary |
| **SegFormer-B2** (ADE20K scenes) | Laptop | Decides if a wall gap is a door, window or pillar; double-checks furniture |
| **MobileNetV2** | Laptop | Backup furniture check if SegFormer isn't available |
| **Gemini** (Google, cloud) | Laptop → Google | Commander's question answering |
| **ElevenLabs** (cloud) | Laptop → ElevenLabs | Voice audio for phone guidance |

**What the AI never does:** it never invents geometry (every surface is measured depth), and object names never leave the laptop.

YOLOv8n's licence is AGPL-3.0: fine for demos; commercial use would need an Ultralytics licence.

---

## 11. Ask Gemini (commander's assistant)

- **Ask** button (or A) opens a chat. Type what rescuers radio in, or use quick buttons: "Where is everyone?", alerts, unsearched areas, summary.
- Each question goes to Gemini with a **text summary of the live incident**: names, positions in metres from the entrance, alerts, hazards, exits, stairs, coverage, planner suggestions, the last 15 timeline events. **No camera images.**
- It's told to be brief, never invent facts, and that the commander decides.
- If one Gemini model is retired or busy, it automatically tries the next.
- The conversation lives only in that browser tab. Up to about 12 questions a minute.

---

## 12. Security and privacy

| Measure | What it means |
|---|---|
| **Single-use QR pairing** | Each code works once and expires after 10 minutes. |
| **TLS 1.3, pinned** | Phone ↔ laptop traffic is encrypted, and the phone only trusts *this* laptop's certificate (from the QR code), so it can't be redirected to a fake server. |
| **Session-only** | Leaving the app, locking the phone or tapping Disconnect ends the session. The laptop forgets the phone; the next time needs a fresh QR code. |
| **Nothing stored on phones** | No map, images or data saved on the phone. |
| **Commander page on the laptop only** | It listens only on the laptop itself (127.0.0.1). Other computers need a one-time link and can be revoked. |
| **Revoke** | Cuts a phone or computer off immediately. |
| **Everything validated** | Bad or oversized messages, impossible coordinates (over 500 m) and floods are dropped and rate-limited. |
| **API keys stay on the laptop** | Gemini and ElevenLabs keys live in owner-only files, are set only from the laptop itself, and are never sent to a browser or phone. |
| **Encrypted saved data** | Saved maps and recordings are encrypted with a laptop-only key. Recordings are opt-in, deleted after 7 days, capped at 200 MB. |
| **No wireless install link** | The iPhone app is installed by cable, so there's no standing wireless link between the Mac and the phone. |

---

## 13. Honest evaluation

To measure the room and fall-risk detectors fairly:
1. Map two separate areas of a real building (A = training, B = test; C = optional spare).
2. Label the real rooms and trip spots by hand on the saved maps.
3. Settings are **tuned on area A only**, then **scored once on area B** (precision, recall, F1, overlap), with a written report. Area C stays untouched until the very end.
4. Tuned settings are only used if they're strictly better on A.

---

## 14. Limits (be upfront about these)

- **Camera-only depth is approximate:** about 5–15% error, best within 4 m, worse on mirrors and glass.
- **Drift:** a few cm per 10 m. Re-align at the marker now and then; **don't move the marker**.
- **Mapped areas only:** routes through unseen space are a best guess (that's why uncertainty raises the cost).
- **Single floor:** stairs are detected, but floors aren't linked yet.
- **Map lives in memory:** restarting the server clears it (save it first). Pairings don't survive a restart.
- **Networking:** venue Wi-Fi often blocks devices from talking to each other. Use the **iPhone's Personal Hotspot** with the laptop joined to it.
- **Battery and heat:** AI scanning is heavy; the app slows down automatically when hot, and 3D can be turned off.
- **Free Apple developer account:** the iPhone app expires after 7 days (reinstall), max 3 iPhones.
- **Smoke:** not tested in real smoke; camera-based tracking would struggle in thick smoke.
- **Decision support only:** it never replaces a physical search or the commander's judgement.

---

## 15. Likely questions and answers

**What problem does this solve?**
Inside a building, GPS doesn't work and the commander loses sight of responders. This gives everyone a shared live map: where people are, what's been searched, who's in trouble, and the safest way out.

**Why not just use GPS?**
GPS doesn't work reliably indoors and has no idea about floors, walls or rooms. We use the phone's AR tracking (camera + motion sensors), tied to a marker at the entrance.

**Do you need special hardware?**
No. A printed marker (or the laptop screen), a laptop, and ordinary phones. iPhone 13 and up (no LiDAR needed) or an ARCore Android phone.

**How do all the phones agree on positions?**
Every phone scans the same marker at the entrance, which becomes the shared origin. Each phone converts its positions into "metres from the marker". See section 3.

**How accurate is it?**
Marker lock is typically a few millimetres ("Locked ±N mm"). Tracking drifts a few cm per 10 m and corrects itself when the marker is seen again. Simulated positions came out within about 1–3 cm. Camera-only depth is roughly 5–15% off.

**How do you get 3D without LiDAR?**
An AI depth model on the phone (Depth Anything V2), calibrated to real metres every frame using points ARKit already knows. See section 4.

**What if two people map the same room?**
Their data merges: shared 5 cm cubes, merged walls, and fine alignment between phones. No double walls.

**How does the "see through walls" work?**
The server knows everyone's position and the walls. It counts the walls on the straight line between two people, and the phone draws the colleague as a ghost where they actually are, with their live camera. It's not seeing through walls physically; it's the shared map.

**How is a fall detected?**
From movement: the phone drops more than 0.7 m within 2.5 s and stays down, plus an accelerometer free-fall-then-impact signal. "Not moving" is under 25 cm in 30 s. See section 7.

**What about false alarms?**
The person can tap "I'm OK"; the commander can mark alerts real. Both are recorded as labels to measure accuracy. Alerts are advice, never automatic action.

**How do routes avoid danger?**
The floor is a grid of 20 cm squares; stairs, drops and clutter make squares 6× more "expensive", hazards 250×, unseen floor more too. A\* finds the cheapest route. See section 8.

**How does the phone know left from right for voice directions?**
It knows where the camera points (or which way you're walking) and compares that with the direction of the route 1–2 m ahead.

**Does it work without internet?**
Yes, everything core runs on the laptop over local Wi-Fi / hotspot. Only Gemini and ElevenLabs need internet; without it, phones use their built-in voice and the Ask button is unavailable.

**What data leaves the building?**
Nothing, unless you use the optional extras. Gemini gets a text summary (names, positions, alerts; no images). ElevenLabs gets only fixed phrases like "Turn left" (no names or labels).

**Could someone hack in or fake a phone?**
Pairing needs a one-time QR code from the commander; traffic is encrypted and pinned to the laptop; every message is validated and rate-limited; the commander can revoke anyone instantly. See section 12.

**What happens if the laptop dies?**
The phones lose the shared map (it lives on the laptop). Maps can be saved beforehand. This is a known limit; a backup laptop would be a next step.

**What if the Wi-Fi doesn't work?**
Use the iPhone's Personal Hotspot with the laptop joined to it. Phones re-pair after changing networks.

**Does it work on Android?**
Yes: same protocol, marker, planes, ARCore Depth API. Voice guidance is iPhone-only for now.

**Does it work in smoke or darkness?**
Not tested. Camera-based tracking needs to see something; thick smoke would be a problem. Future work: thermal cameras or other sensors.

**Multiple floors?**
Not yet. Stairs are detected and marked as fall-risk zones; linking floors is a next step.

**Is it trained on our data?**
No. All models are pre-trained. Our own tuning only adjusts a few settings (like zone sizes), tested fairly on a separate area.

**Who decides what responders do?**
The commander. Every route, alert and suggestion is decision support.

**What was the hardest part?**
Getting everyone into one accurate coordinate system, and getting real-metre depth from a single camera on plain walls.

**What's next?**
Voice on Android, multiple floors, field testing with real responders, and wireless installs (TestFlight).

---

## 16. Quick glossary

| Term | Meaning |
|---|---|
| **AR (ARKit / ARCore)** | Apple's / Google's phone tracking: works out where the phone is from the camera and motion sensors. |
| **Marker** | The printed pattern at the entrance; everyone's shared origin. |
| **Shared frame** | The common coordinate system: metres from the marker. |
| **Pose** | Position + direction of the phone. |
| **Plane** | A flat surface ARKit/ARCore found (wall, floor, door…). |
| **Voxel** | A small 3D cube (5 cm here) used to store and merge points. |
| **TSDF** | A way of fusing many depth images into one smooth 3D surface. |
| **RANSAC** | A fitting method that ignores bad data points. |
| **A\* / Dijkstra** | Shortest-path algorithms. |
| **Kalman filter** | Smooths noisy positions and predicts movement. |
| **Hungarian algorithm** | Best way to pair people with tasks (here: rooms). |
| **Coverage** | How much of the building's floor someone has actually looked at. |
| **Fall-risk zone (yellow)** | Stairs, drops, clutter, places someone fell: avoided when possible. |
| **Hazard (red)** | A danger pinned by the team: avoided unless there's no other way. |
| **TLS pinning** | The phone only trusts the laptop's exact certificate. |

---

## 17. Where things are in the code

| Folder / file | What's there |
|---|---|
| `server/sightline/app.py` | The server: phone + commander connections, endpoints, main loop |
| `server/sightline/state.py` | The live "world": phones, planes, walls, pins, map |
| `server/sightline/frame.py` | Shared-frame maths |
| `server/sightline/fusion.py` | 3D surface fusion, walls and furniture from depth |
| `server/sightline/geometry.py` | Wall merging and wall counting (x-ray) |
| `server/sightline/occupancy.py` | The 20 cm grid with uncertainty |
| `server/sightline/predict.py` | Routes, exits, Kalman, decisions, waypoint routes |
| `server/sightline/risk.py` | Fall-risk zones |
| `server/sightline/twin.py` | Coverage, fall / not-moving alerts, hazard warnings |
| `server/sightline/rooms.py` | Room detection |
| `server/sightline/voice.py` | Voice cues + ElevenLabs |
| `server/sightline/gemini.py` | Ask Gemini |
| `server/sightline/security.py` | Pairing, tokens, certificates |
| `web/` | Commander page (2D canvas + three.js 3D) |
| `ios/Sightline/` | iPhone app (AR, depth AI, network, voice, UI) |
| `android/` | Android app |
| `sim/sim_phone.py` | Fake phones for testing and demos without hardware |
| `marker/` | The printable marker |
