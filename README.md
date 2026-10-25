# FindMyTeam

## Inspiration

When firefighters enter a smoke-filled building, incident commanders lose visual contact immediately. GPS fails indoors, radios often garble, and traditional PASS alarms only work locally if someone is close enough to hear them. **FindMyTeam** enables ordinary smartphones to map indoor environments collectively in real time, giving commanders a live overview of responder locations, search coverage, and safe routes. It operates on existing hardware, starting with the non-LiDAR iPhone 13.

---

## What It Does

**FindMyTeam** synchronizes ordinary smartphones into a single, shared 3D operational map:

* **Unified Coordinate System:** Scans a printed entrance marker so all devices share a common metric frame.
* **Live Commander Dashboard:** Displays a 2D and 3D browser interface featuring responder paths, walls, doors, a colored point cloud, and fused surface meshes.
* **Through-Wall Visibility:** Renders colleagues on the opposite side of walls as colored ghosts with live camera feeds and obstruction indicators.
* **Search Coverage Heatmaps:** Tracks room-by-room progress (searched vs. unsearched) and assigns unsearched spaces to available responders.
* **Man-Down Alerts:** Detects falls and prolonged lack of movement, triggering local acoustic chirps and commander warnings.
* **Dynamic Safe Routing:** Computes real-time paths back to command and exits, avoiding hazards, drops, and stairs.
* **Voice Guidance:** Employs ElevenLabs for spatial audio cues (e.g., "Trip hazard ahead in 2 metres").
* **Gemini Integration:** Allows commanders to query tactical status using natural language.
* **Secure Architecture:** Features single-use QR pairing, TLS 1.3 pinning, and session-only connections with zero on-device persistence.

---

## How We Built It

**Stack:** Python FastAPI server (commander laptop), Vanilla JS + Three.js frontend, SwiftUI + ARKit + RealityKit + Core ML (iOS), and Kotlin + ARCore (Android).

### 1. Unified Coordinate System
Each phone tracks its local ARKit/ARCore world. By detecting a printed entrance marker, the system establishes a shared origin (Y up, Z pointing inward). 
* Pose matrix S is inverted (`T = S^(-1)`) to transform local coordinates into shared coordinates (`x_shared = T * x_local`).
* To eliminate drift, the system requires 30 stable, face-on sightings to calculate a median lock ("Locked ±N mm"), applying smooth corrections when the marker re-enters the field of view.

### 2. Depth Without LiDAR
For devices like the iPhone 13 lacking native LiDAR, the app runs **Depth Anything V2** via Core ML for relative depth (`d_hat`).
* Frame-by-frame calibration aligns relative depth to known ARKit feature points and plane ray-hits using RANSAC and least squares.
* The system fits linear (`z = a * d_hat + b`) and inverse (`1/z = a * d_hat + b`) models, selecting the one with the lowest relative error, with adaptive throttling to manage thermal load.

### 3. Multi-Phone Map Fusion
* **Voxel Filtering:** Points are mapped to 5 cm voxels, registering only after being observed at least twice to eliminate speckle.
* **TSDF Mesh Generation:** Low-resolution depth images (128x96) fuse into a Truncated Signed Distance Function (TSDF) volume (4 cm cells) to construct solid surface meshes.
* **Wall Alignment:** Individual wall segments are merged into centerlines, with device poses fine-aligned to the primary map.
* **Object Detection:** YOLOv8n identifies doors and furniture on-device, while SegFormer-B2 on the server classifies structural openings (doors, windows, pillars).

### 4. Pathfinding and Task Allocation
* **Grid Layout:** Floor plans are discretized into 20 cm cells with probabilistic occupancy and uncertainty weights.
* **A\* Routing:** Computes paths using a cost function balancing distance, zone uncertainty, and hazard proximity:
  `cost = d * (1 + 1.5 * u) * k_zone`
  where `k_zone` is 1 for clear areas, 6 for fall-risks, and 250 near hazard pins.
* **Coordination:** Employs Kalman filters for position prediction, Dijkstra for nearest-exit routing, and the Hungarian algorithm to assign unsearched rooms.

### 5. Safety Detection
* **Fall Detection:** Triggers if a phone drops over 0.7 meters within 2.5 seconds and remains stationary.
* **Immobility Detection:** Flags movement under 25 cm over 30 seconds.
* **Coverage Tracking:** Traces camera frustums (56° wide, 5 meters deep) against obstacles to ensure obscured areas are not falsely marked as searched.

### 6. Voice Guidance
* Computes turns based on heading deltas relative to route waypoints.
* **ElevenLabs Integration:** Uses Flash v2.5 for spatial audio cues, caching audio locally while routing sensitive identifiers (names, custom labels) through native text-to-speech to preserve privacy.

---

## Challenges We Ran Into

* **Coordinate Drift:** Initial marker sightings yielded high error over distance; resolved via median locking, face-on filtering, and wall-based fine alignment.
* **Monocular Depth Scale:** Calibrating relative depth on featureless white walls required integrating ray-hits from detected planes alongside feature points.
* **Map Consistency:** Merging multiple overlapping corridors required deduplicating wall faces and locking settled structures.
* **Thermal & Memory Management:** Running continuous inference, AR tracking, and streaming required thermal throttling and proactive memory monitoring.
* **Network Reliability:** Ad-hoc venue Wi-Fi proved unstable; deployed personal iPhone hotspots paired with laptop clients.
* **Audio Fatigue:** Refined pathfinding lookahead logic to prevent repetitive instruction loops.

---

## Accomplishments We're Proud Of

* Achieving a real-time, shared 3D mapping system using non-LiDAR consumer smartphones.
* Enabling real-time "through-wall" visualization of teammates via live video feeds.

---

## What We Learned

* **Geometry First:** Most mapping challenges were fundamentally geometric rather than machine learning problems.
* **Robust Statistics:** Simple statistical filtering (medians, RANSAC, observation thresholds) outperformed complex models.
* **Uncertainty Modeling:** Treating unmapped areas as uncertain rather than open yielded safer, more dependable routes.
* **Decision Support:** Designing the system to advise commanders—rather than automate decisions—maintained operational transparency.

---

## What's Next

* Expand voice guidance to Android platforms.
* Implement multi-floor tracking connected via stairwells.
