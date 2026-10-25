import ARKit
import RealityKit
import UIKit

/// RealityKit content: tinted detected planes, colleague "ghosts" and pins.
/// Without LiDAR there is no scene mesh, so nothing occludes virtual content:
/// a colleague behind a wall is drawn right through it, which is exactly the x-ray effect.
final class SceneOverlay {
    private let arView: ARView
    private let root = AnchorEntity(world: .zero)
    private var planes: [UUID: (anchor: AnchorEntity, model: ModelEntity)] = [:]
    private var ghosts: [String: ModelEntity] = [:]
    private var pinEntities: [String: ModelEntity] = [:]

    init(arView: ARView) {
        self.arView = arView
        arView.scene.addAnchor(root)
    }

    func clear() {
        for e in ghosts.values { e.removeFromParent() }
        for e in pinEntities.values { e.removeFromParent() }
        ghosts.removeAll()
        pinEntities.removeAll()
    }

    // MARK: planes (local feedback that the map is being built)

    func showPlane(_ plane: ARPlaneAnchor) {
        let e = plane.planeExtent
        let mesh = MeshResource.generatePlane(width: e.width, depth: e.height)
        if let existing = planes[plane.identifier] {
            existing.model.model?.mesh = mesh
            place(existing.model, in: plane)
            return
        }
        let model = ModelEntity(mesh: mesh, materials: [UnlitMaterial(color: Self.planeColor(ARManager.kind(of: plane)))])
        let anchor = AnchorEntity(anchor: plane)
        anchor.addChild(model)
        place(model, in: plane)
        arView.scene.addAnchor(anchor)
        planes[plane.identifier] = (anchor, model)
    }

    private func place(_ model: ModelEntity, in plane: ARPlaneAnchor) {
        model.position = plane.center
        model.orientation = simd_quatf(angle: plane.planeExtent.rotationOnYAxis, axis: [0, 1, 0])
    }

    static func planeColor(_ kind: String) -> UIColor {
        switch kind {
        case "wall", "vertical": return UIColor.white.withAlphaComponent(0.12)
        case "door": return UIColor.systemGreen.withAlphaComponent(0.3)
        case "window": return UIColor.systemBlue.withAlphaComponent(0.3)
        case "floor": return UIColor.systemTeal.withAlphaComponent(0.10)
        default: return UIColor.gray.withAlphaComponent(0.08)
        }
    }

    // MARK: colleagues and pins (shared frame -> local via toLocal)

    func update(peers: [Peer], pins: [Pin], toLocal: simd_float4x4, highlight: String?) {
        var seen = Set<String>()
        for peer in peers where peer.connected && !peer.stale {
            guard let p = vec3(peer.pos) else { continue }
            seen.insert(peer.id)
            let ghost = ghosts[peer.id] ?? makeGhost(color: peer.color, id: peer.id)
            // pos is the camera (eye); put the body's top just below it
            var t = Transform(matrix: matrix_identity_float4x4)
            t.translation = toLocal.point(p - SIMD3<Float>(0, 0.8, 0))
            t.scale = SIMD3<Float>(repeating: peer.id == highlight ? 1.15 : 1)
            ghost.move(to: t, relativeTo: nil, duration: 0.12)
        }
        for (id, e) in ghosts where !seen.contains(id) {
            e.removeFromParent()
            ghosts[id] = nil
        }

        let pinIDs = Set(pins.map(\.id))
        for pin in pins where pinEntities[pin.id] == nil {
            guard let p = vec3(pin.pos) else { continue }
            let e = pin.auto == true // automatic "possible person": a bigger orange sphere
                ? ModelEntity(mesh: .generateSphere(radius: 0.15), materials: [UnlitMaterial(color: UIColor.systemOrange.withAlphaComponent(0.85))])
                : ModelEntity(mesh: .generateBox(size: 0.18, cornerRadius: 0.03), materials: [UnlitMaterial(color: Self.pinColor(pin.kind))])
            e.position = toLocal.point(p + SIMD3<Float>(0, 0.1, 0))
            e.orientation = simd_quatf(angle: .pi / 4, axis: [0, 1, 0]) * simd_quatf(angle: .pi / 4, axis: [1, 0, 0])
            root.addChild(e)
            pinEntities[pin.id] = e
        }
        for (id, e) in pinEntities where !pinIDs.contains(id) {
            e.removeFromParent()
            pinEntities[id] = nil
        }
    }

    private func makeGhost(color: String, id: String) -> ModelEntity {
        let (r, g, b) = color.rgb
        let tint = UIColor(red: r, green: g, blue: b, alpha: 0.75)
        let body = ModelEntity(mesh: .generateBox(width: 0.42, height: 1.5, depth: 0.25, cornerRadius: 0.12), materials: [UnlitMaterial(color: tint)])
        let head = ModelEntity(mesh: .generateSphere(radius: 0.13), materials: [UnlitMaterial(color: tint)])
        head.position = [0, 0.95, 0]
        body.addChild(head)
        root.addChild(body)
        ghosts[id] = body
        return body
    }

    static func pinColor(_ kind: String) -> UIColor {
        switch kind {
        case "hazard": return .systemRed
        case "victim": return .systemOrange
        case "rally": return .systemGreen
        case "waypoint": return .systemCyan
        default: return .systemPurple
        }
    }
}
