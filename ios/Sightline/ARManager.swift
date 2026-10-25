import ARKit
import Combine
import CoreImage
import ImageIO
import RealityKit
import UIKit

/// Owns the ARView/ARSession. Turns ARKit's local world into the shared frame and streams
/// pose, plane anchors, AI-depth points and camera thumbnails to the server.
///
/// iPhone 13 (no LiDAR): world tracking + gravity alignment + plane detection/classification
/// + one detection image. No scene depth, no scene reconstruction, no RoomPlan.
final class ARManager: NSObject, ObservableObject, ARSessionDelegate {
    enum Alignment: Equatable {
        case searching
        case locking(progress: Double)
        case aligned
    }

    static let lockSamples = 30 // good marker sightings (median) before we trust the frame
    static let reanchorSamples = 15 // sightings for a drift correction once aligned
    static let poseHz = 10.0
    static let frameHz = 3.0
    static let xrayCone: Float = 30 * .pi / 180

    @Published private(set) var alignment: Alignment = .searching
    @Published private(set) var trackingText = "Starting…"
    @Published private(set) var planeCount = 0
    @Published private(set) var xrayTarget: Peer?
    @Published private(set) var xrayScreenPoint: CGPoint?
    /// Relative bearing (radians, 0 = straight ahead, + = to the right) of every live colleague.
    @Published private(set) var bearings: [String: Float] = [:]
    @Published var xrayEnabled = true
    /// AI-depth 3D scanning (Depth Anything V2 on the Neural Engine).
    @Published var scanning = true
    @Published private(set) var scanText = ""
    /// Lock precision and last drift correction, shown under the top bar.
    @Published private(set) var calibText = ""
    /// Why the current marker sighting is being ignored (too far / too oblique), if it is.
    @Published private(set) var markerHint: String?
    @Published private(set) var wallDistance: Float?
    /// My position/heading in the shared frame, for the minimap.
    @Published private(set) var myPos: SIMD3<Float>?
    @Published private(set) var myHeading: Float?

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    let net: NetworkClient
    let sensors = MotionSensors()

    /// local (ARKit) -> shared, and shared -> local. nil until the marker has been seen.
    private(set) var toShared: simd_float4x4?
    private(set) var toLocal: simd_float4x4?

    private var samples: [simd_float4x4] = []
    private var reanchor: [simd_float4x4] = []
    private var lastReanchor: TimeInterval = 0
    private var lastPose: TimeInterval = 0
    private var lastFrame: TimeInterval = 0
    private var lastStatus: TimeInterval = 0
    private var lastPlaneSend: [UUID: TimeInterval] = [:]
    private var encoding = false
    private let encodeQueue = DispatchQueue(label: "sightline.jpeg", qos: .utility)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let scene: SceneOverlay
    private var updateSub: Cancellable?
    private var markerSub: AnyCancellable?
    private var sentXrayID: String?
    private let depth = DepthEstimator()
    private var depthBusy = false
    private var lastDepth: TimeInterval = 0
    private var lastWallCheck: TimeInterval = 0
    private var depthCenter: Float?
    private var pointsSent = 0
    private var depthRuns = 0
    private var memoryPauseUntil: TimeInterval = 0 // after an iOS memory warning, heavy work pauses briefly
    private var memoryWarnings = 0
    private var arFrames = 0
    private var memoryObserver: NSObjectProtocol?

    init(net: NetworkClient) {
        self.net = net
        scene = SceneOverlay(arView: arView)
        super.init()
        arView.session.delegate = self
        arView.renderOptions.formUnion([.disableMotionBlur, .disableDepthOfField, .disableHDR, .disableGroundingShadows, .disablePersonOcclusion])
        UIDevice.current.isBatteryMonitoringEnabled = true
        net.onConnect = { [weak self] in self?.resendEverything() }
        sensors.onImpact = { [weak self] g, freeFallMs in
            self?.net.send(["t": "impact", "g": g, "freefall_ms": freeFallMs])
        }
        updateSub = arView.scene.subscribe(to: SceneEvents.Update.self) { [weak self] _ in self?.updateXrayProjection() }
        // The commander switched between the printed marker and the one on the laptop screen: new size, lock on again.
        markerSub = net.$markerWidth.removeDuplicates().dropFirst().sink { [weak self] _ in
            DispatchQueue.main.async { self?.applyMarker() }
        }
        memoryObserver = NotificationCenter.default.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification,
                                                                object: nil, queue: .main) { [weak self] _ in
            guard let self else { return }
            self.memoryWarnings += 1
            self.memoryPauseUntil = CACurrentMediaTime() + 15 // let iOS reclaim memory instead of killing us
            self.scanText = "3D scan paused: low memory"
            self.net.send(["t": "status", "memory_warning": true])
        }
    }

    deinit {
        if let memoryObserver { NotificationCenter.default.removeObserver(memoryObserver) }
    }

    /// The app's own memory use (what iOS counts when deciding to kill it), in MB.
    static func memoryMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
    }

    // MARK: - session control

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            trackingText = "ARKit world tracking not supported on this device"
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        config.environmentTexturing = .none
        // Optional bonus: phones with a LiDAR sensor (Pro models) give measured depth. Phones without one
        // (e.g. iPhone 13) keep using the Depth Anything estimate; everything else works the same.
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        if let ref = markerReference() {
            config.detectionImages = [ref]
            config.maximumNumberOfTrackedImages = 1
        } else {
            trackingText = "marker.png missing from the app bundle"
        }
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        sensors.start()
        realign()
    }

    func pause() {
        arView.session.pause()
        sensors.stop()
    }

    /// Forget the marker lock: the next marker sighting defines the frame again.
    func realign() {
        toShared = nil
        toLocal = nil
        samples.removeAll()
        reanchor.removeAll()
        calibText = ""
        depth.reset()
        alignment = .searching
        scene.clear()
        net.send(["t": "planes_clear"])
        net.send(["t": "status", "aligned": false, "tracking": trackingText])
    }

    /// Same session, new marker size (keeps tracking; the shared-frame lock starts over).
    private func applyMarker() {
        guard let config = arView.session.configuration as? ARWorldTrackingConfiguration, let ref = markerReference() else { return }
        config.detectionImages = [ref]
        arView.session.run(config, options: [])
        realign()
    }

    func markerReference() -> ARReferenceImage? {
        guard let url = Bundle.main.url(forResource: "marker", withExtension: "png"),
              let image = UIImage(contentsOfFile: url.path)?.cgImage else { return nil }
        let ref = ARReferenceImage(image, orientation: .up, physicalWidth: CGFloat(net.markerWidth)) // metres
        ref.name = "entrance"
        return ref
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) { handle(anchors) }
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) { handle(anchors) }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for case let plane as ARPlaneAnchor in anchors {
            lastPlaneSend[plane.identifier] = nil
            net.send(["t": "plane_remove", "id": plane.identifier.uuidString])
        }
        planeCount = session.currentFrame?.anchors.filter { $0 is ARPlaneAnchor }.count ?? planeCount
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        switch camera.trackingState {
        case .normal: trackingText = "normal"
        case .notAvailable: trackingText = "not available"
        case .limited(let reason):
            switch reason {
            case .initializing: trackingText = "initializing"
            case .excessiveMotion: trackingText = "slow down"
            case .insufficientFeatures: trackingText = "low detail"
            case .relocalizing: trackingText = "relocalizing"
            @unknown default: trackingText = "limited"
            }
        }
        net.send(["t": "status", "aligned": toShared != nil, "tracking": trackingText])
    }

    func sessionWasInterrupted(_ session: ARSession) { trackingText = "interrupted" }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = frame.timestamp
        guard let toShared else { return }
        let cam = toShared * frame.camera.transform

        if now - lastPose >= 1 / Self.poseHz {
            lastPose = now
            let pos = cam.translation
            let fwd = simd_normalize(-SIMD3<Float>(cam.columns.2.x, cam.columns.2.y, cam.columns.2.z))
            net.send(["t": "pose", "pos": pos.array, "fwd": fwd.array, "tracking": trackingText,
                      "act": sensors.activity, "act_conf": sensors.confidence,
                      "wall_dist": wallDistance.map { $0 as Any } ?? NSNull()])
            myPos = pos
            myHeading = atan2(fwd.x, fwd.z)
            updatePeers(camShared: cam)
        }
        if now - lastWallCheck >= 0.2 {
            lastWallCheck = now
            updateWallDistance(frame)
        }
        scheduleDepth(frame, now: now, toShared: toShared)
        arFrames += 1
        if now - lastStatus >= 5 {
            let secs = now - lastStatus
            lastStatus = now
            let level = UIDevice.current.batteryLevel
            // Stability telemetry: memory, heat and work rates, so a crash or slowdown can be explained.
            net.send(["t": "status", "aligned": true, "tracking": trackingText, "battery": level < 0 ? NSNull() as Any : level as Any,
                      "mem_mb": Self.memoryMB(), "thermal": ProcessInfo.processInfo.thermalState.rawValue,
                      "ar_fps": secs < 30 ? Double(arFrames) / secs : 0, "depth_hz": secs < 30 ? Double(depthRuns) / secs : 0,
                      "mem_warnings": memoryWarnings])
            arFrames = 0
            depthRuns = 0
        }
        if now - lastFrame >= 1 / Self.frameHz, !encoding, net.connected {
            lastFrame = now
            encode(frame, toShared: toShared)
        }
    }

    // MARK: - distance to the wall in front

    /// Screen-centre distance: detected plane first (accurate), then ARKit's estimate, then AI depth.
    private func updateWallDistance(_ frame: ARFrame) {
        let center = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
        let camPos = frame.camera.transform.translation
        let hit = arView.raycast(from: center, allowing: .existingPlaneGeometry, alignment: .vertical).first
            ?? arView.raycast(from: center, allowing: .estimatedPlane, alignment: .any).first
        let d = hit.map { simd_distance($0.worldTransform.translation, camPos) } ?? depthCenter
        wallDistance = d.map { ($0 * 10).rounded() / 10 }
    }

    // MARK: - AI depth -> 3D points

    private func scheduleDepth(_ frame: ARFrame, now: TimeInterval, toShared: simd_float4x4) {
        let lidar = frame.smoothedSceneDepth
        guard scanning, lidar != nil || depth.available, !depthBusy, net.connected, case .normal = frame.camera.trackingState else { return }
        guard CACurrentMediaTime() > memoryPauseUntil else { return }
        let thermal = ProcessInfo.processInfo.thermalState
        // LiDAR needs no neural network, so it can run more often for the same heat
        let interval: TimeInterval = thermal == .critical ? .infinity : thermal == .serious ? 1.5 : (lidar != nil ? 0.25 : 0.45)
        guard now - lastDepth >= interval else {
            if thermal == .critical { scanText = "3D scan paused: phone too hot" }
            return
        }
        lastDepth = now
        let job = DepthEstimator.Job(image: frame.capturedImage, camera: frame.camera.transform, intrinsics: frame.camera.intrinsics,
                                     imageSize: frame.camera.imageResolution, anchors: lidar == nil ? depthAnchors(frame) : [], toShared: toShared,
                                     findFurniture: depthRuns % (lidar != nil ? 4 : 2) == 0 && thermal.rawValue < ProcessInfo.ThermalState.serious.rawValue,
                                     lidarDepth: lidar?.depthMap, lidarConfidence: lidar?.confidenceMap)
        depthBusy = true
        let started = CACurrentMediaTime()
        depth.queue.async { [weak self] in
            guard let self else { return }
            let out = job.lidarDepth != nil ? self.depth.runLiDAR(job) : self.depth.run(job)
            DispatchQueue.main.async {
                self.depthBusy = false
                guard let out else {
                    self.scanText = "3D scan: calibrating (aim at walls / textured areas)"
                    return
                }
                self.depthCenter = out.centerDepth
                if out.count > 0 { self.net.sendBinary(NetworkClient.binPoints, out.points) }
                self.net.sendBinary(NetworkClient.binDepth, out.depthFrame)
                self.depthRuns += 1
                self.pointsSent += out.count
                let ms = Int((CACurrentMediaTime() - started) * 1000)
                self.scanText = "3D scan\(job.lidarDepth != nil ? " (LiDAR)" : "") \(ms) ms · \(out.count) pts · \(self.pointsSent / 1000)k total"
            }
        }
    }

    /// Metric depths ARKit already knows, as (u, v, z) in normalized landscape image coordinates.
    private func depthAnchors(_ frame: ARFrame) -> [SIMD3<Float>] {
        let cam = frame.camera
        let inv = cam.transform.inverse
        let K = cam.intrinsics
        let w = Float(cam.imageResolution.width), h = Float(cam.imageResolution.height)
        func project(_ world: SIMD3<Float>) -> SIMD3<Float>? {
            let pc = inv.point(world)
            let z = -pc.z
            guard z > 0.2 else { return nil }
            let u = (K[0][0] * pc.x / z + K[2][0]) / w, v = (-K[1][1] * pc.y / z + K[2][1]) / h
            return SIMD3<Float>(u, v, z)
        }
        var out: [SIMD3<Float>] = []
        if let pts = frame.rawFeaturePoints?.points {
            for p in pts.prefix(400) { if let a = project(p) { out.append(a) } }
        }
        // plane hits matter on blank/shiny walls, where feature points are scarce
        for gy in 1...4 { for gx in 1...5 {
            let q = frame.raycastQuery(from: CGPoint(x: Double(gx) / 6, y: Double(gy) / 5), allowing: .existingPlaneGeometry, alignment: .any)
            if let hit = arView.session.raycast(q).first, let a = project(hit.worldTransform.translation) {
                out.append(a)
                out.append(a) // plane hits are more trustworthy than single feature points
            }
        } }
        return out
    }

    // MARK: - anchors

    private func handle(_ anchors: [ARAnchor]) {
        for anchor in anchors {
            if let image = anchor as? ARImageAnchor {
                handleMarker(image)
            } else if let plane = anchor as? ARPlaneAnchor {
                scene.showPlane(plane)
                if toShared != nil { sendPlane(plane, force: false) }
            }
        }
        planeCount = arView.session.currentFrame?.anchors.filter { $0 is ARPlaneAnchor }.count ?? planeCount
    }

    /// A sighting is only trusted when the camera is close and roughly face-on to the marker:
    /// far or oblique views give the noisiest image-anchor poses.
    private func goodSighting(_ image: ARImageAnchor) -> Bool {
        guard image.isTracked, let cam = arView.session.currentFrame?.camera.transform else { return false }
        let toCam = cam.translation - image.transform.translation
        let dist = simd_length(toCam)
        let normal = simd_normalize(SIMD3<Float>(image.transform.columns.1.x, image.transform.columns.1.y, image.transform.columns.1.z))
        let angle = acos(max(-1, min(1, simd_dot(normal, toCam / max(dist, 1e-3)))))
        // tighter than before (1.5 m / 50 deg): closer, more face-on sightings give a steadier lock
        if dist > 1.2 { markerHint = "Move closer to the marker"; return false }
        if angle > 40 * .pi / 180 { markerHint = "Face the marker more directly"; return false }
        markerHint = nil
        return true
    }

    private func handleMarker(_ image: ARImageAnchor) {
        guard goodSighting(image) else { return }
        if toShared == nil {
            samples.append(image.transform)
            if samples.count < Self.lockSamples {
                alignment = .locking(progress: Double(samples.count) / Double(Self.lockSamples))
                return
            }
            guard let (s, spread) = SharedFrame.sharedToLocal(median: samples) else { return }
            toLocal = s
            toShared = SharedFrame.rigidInverse(s)
            alignment = .aligned
            calibText = String(format: "Locked ±%.0f mm", spread * 1000)
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            resendEverything()
            return
        }
        // Already aligned and the marker is in view again: correct accumulated drift.
        let now = CACurrentMediaTime()
        guard now - lastReanchor > 3 else { return }
        reanchor.append(image.transform)
        guard reanchor.count >= Self.reanchorSamples, let old = toLocal,
              let (s, spread) = SharedFrame.sharedToLocal(median: reanchor) else { return }
        reanchor.removeAll()
        lastReanchor = now
        let moved = simd_distance(old.translation, s.translation)
        let turned = acos(max(-1, min(1, simd_dot(SIMD3<Float>(old.columns.2.x, 0, old.columns.2.z), SIMD3<Float>(s.columns.2.x, 0, s.columns.2.z)))))
        guard spread < 0.02, moved < 0.6, turned < 15 * .pi / 180 else { return } // implausible jump: ignore
        guard moved > 0.01 || turned > 0.003 else { return }
        // Ease the correction in: go halfway now, the next sighting finishes it. Small steps instead of one
        // visible jump of the whole map (and fewer doubled walls while the old pieces are re-sent).
        let fwd = { (m: simd_float4x4) in SIMD3<Float>(m.columns.2.x, 0, m.columns.2.z) }
        let eased = SharedFrame.sharedToLocal(origin: simd_mix(old.translation, s.translation, SIMD3<Float>(repeating: 0.5)),
                                              forward: simd_normalize(simd_mix(fwd(old), fwd(s), SIMD3<Float>(repeating: 0.5))))
        toLocal = eased
        toShared = SharedFrame.rigidInverse(eased)
        calibText = String(format: "Drift corrected %.0f cm", moved * 50)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        lastPlaneSend.removeAll()
        resendEverything() // walls re-sent in the corrected frame
    }

    private func resendEverything() {
        guard toShared != nil else {
            net.send(["t": "status", "aligned": false, "tracking": trackingText])
            return
        }
        net.send(["t": "status", "aligned": true, "tracking": trackingText])
        for case let plane as ARPlaneAnchor in arView.session.currentFrame?.anchors ?? [] {
            sendPlane(plane, force: true)
        }
    }

    private func sendPlane(_ plane: ARPlaneAnchor, force: Bool) {
        let now = CACurrentMediaTime()
        if !force, let last = lastPlaneSend[plane.identifier], now - last < 0.7 { return }
        guard let toShared else { return }
        lastPlaneSend[plane.identifier] = now
        let corners = Self.corners(of: plane).map { toShared.point($0).array }
        net.send(["t": "plane", "id": plane.identifier.uuidString, "kind": Self.kind(of: plane), "corners": corners])
    }

    /// The plane's bounding rectangle in ARKit world space (4 corners).
    static func corners(of plane: ARPlaneAnchor) -> [SIMD3<Float>] {
        let e = plane.planeExtent
        let w = e.width / 2, h = e.height / 2
        let rot = simd_float4x4(simd_quatf(angle: e.rotationOnYAxis, axis: [0, 1, 0]))
        let local = [SIMD3<Float>(-w, 0, -h), SIMD3<Float>(w, 0, -h), SIMD3<Float>(w, 0, h), SIMD3<Float>(-w, 0, h)]
        return local.map { plane.transform.point(plane.center + rot.direction($0)) }
    }

    static func kind(of plane: ARPlaneAnchor) -> String {
        switch plane.classification {
        case .wall: return "wall"
        case .floor: return "floor"
        case .ceiling: return "ceiling"
        case .door: return "door"
        case .window: return "window"
        case .table: return "table"
        case .seat: return "seat"
        default: return plane.alignment == .vertical ? "vertical" : "horizontal"
        }
    }

    // MARK: - colleagues / x-ray

    private func updatePeers(camShared: simd_float4x4) {
        let me = camShared.translation
        let fwd = -SIMD3<Float>(camShared.columns.2.x, camShared.columns.2.y, camShared.columns.2.z)
        let fwdFlat = simd_normalize(SIMD2<Float>(fwd.x, fwd.z) + SIMD2<Float>(1e-6, 0))
        var best: Peer?
        var bestAngle = Self.xrayCone
        var newBearings: [String: Float] = [:]
        for peer in net.peers where peer.connected && !peer.stale {
            guard let p = vec3(peer.pos) else { continue }
            let d = p - me
            let dist = simd_length(d)
            guard dist > 0.5 else { continue }
            let flat = simd_normalize(SIMD2<Float>(d.x, d.z) + SIMD2<Float>(1e-6, 0))
            // signed floor-plane angle, + = to my right (facing +Z with Y up, right is -X)
            newBearings[peer.id] = atan2(fwdFlat.x * flat.y - fwdFlat.y * flat.x, simd_dot(fwdFlat, flat))
            let angle = acos(max(-1, min(1, simd_dot(fwd, d / dist))))
            if xrayEnabled, dist < 25, angle < bestAngle {
                best = peer
                bestAngle = angle
            }
        }
        bearings = newBearings
        if best != xrayTarget { xrayTarget = best }
        if best?.id != sentXrayID {
            sentXrayID = best?.id
            net.send(["t": "xray", "target": best.map { $0.id as Any } ?? NSNull()])
        }
        if let toLocal { scene.update(peers: net.peers, pins: net.pins, toLocal: toLocal, highlight: best?.id) }
    }

    /// Screen position of the x-ray target's head, refreshed every rendered frame.
    private func updateXrayProjection() {
        guard let target = xrayTarget, let toLocal, let p = vec3(target.pos) else {
            if xrayScreenPoint != nil { xrayScreenPoint = nil }
            return
        }
        let point = arView.project(toLocal.point(p + SIMD3<Float>(0, 0.15, 0)))
        if point != xrayScreenPoint { xrayScreenPoint = point }
    }

    // MARK: - pins

    /// Drop a pin where the screen centre hits a plane (or 1.5 m ahead if nothing is hit).
    func dropPin(kind: String) {
        guard let toShared, let frame = arView.session.currentFrame else { return }
        let center = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
        var world: SIMD3<Float>
        if let hit = arView.raycast(from: center, allowing: .estimatedPlane, alignment: .any).first {
            world = hit.worldTransform.translation
        } else {
            let cam = frame.camera.transform
            world = cam.translation - 1.5 * SIMD3<Float>(cam.columns.2.x, cam.columns.2.y, cam.columns.2.z)
        }
        net.send(["t": "pin", "kind": kind, "label": "", "pos": toShared.point(world).array])
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }

    // MARK: - camera thumbnails

    /// Camera snapshot + where it was taken from (0x05): lets the laptop cut real objects out of real photos.
    /// Layout: f32[16] camera->shared | f32 fx, fy, cx, cy (landscape sensor, scaled) | u16 w, h (scaled sensor) | JPEG (portrait)
    private func encode(_ frame: ARFrame, toShared: simd_float4x4) {
        encoding = true
        let image = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right) // sensor is landscape; UI is portrait
        let pose = toShared * frame.camera.transform
        let K = frame.camera.intrinsics, res = frame.camera.imageResolution
        encodeQueue.async { [weak self] in
            guard let self else { return }
            let scale = 360 / image.extent.width
            let small = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            let options = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.45]
            guard let jpeg = self.ciContext.jpegRepresentation(of: small, colorSpace: CGColorSpaceCreateDeviceRGB(), options: options) else {
                DispatchQueue.main.async { self.encoding = false }
                return
            }
            var msg = Data(capacity: jpeg.count + 84)
            func f32(_ v: Float) { var x = v; withUnsafeBytes(of: &x) { msg.append(contentsOf: $0) } }
            func u16(_ v: UInt16) { var x = v; withUnsafeBytes(of: &x) { msg.append(contentsOf: $0) } }
            for c in 0..<4 { for r in 0..<4 { f32(pose[c][r]) } }
            let s = Float(scale)
            f32(K[0][0] * s); f32(K[1][1] * s); f32(K[2][0] * s); f32(K[2][1] * s)
            u16(UInt16(Float(res.width) * s)); u16(UInt16(Float(res.height) * s))
            msg.append(jpeg)
            DispatchQueue.main.async {
                self.encoding = false
                self.net.sendBinary(NetworkClient.binPosedJPEG, msg)
            }
        }
    }
}
