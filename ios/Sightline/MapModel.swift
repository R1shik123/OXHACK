import SwiftUI

/// Everyone's shared map as this phone sees it: walls (from planes) and obstacle cells (from the
/// coarse 3D point layer). All in the shared frame, top-down x/z.
final class MapModel: ObservableObject {
    struct Wall: Hashable {
        let a: CGPoint
        let b: CGPoint
        let kind: String
    }

    @Published private(set) var walls: [String: Wall] = [:]
    @Published private(set) var cells: [Int64: Color] = [:]
    private var floorY: Float = -1.4
    static let cell: Float = 0.2
    static let maxCells = 40000

    func apply(upsert: [PlaneMsg], remove: [String], reset: Bool) {
        var w = reset ? [:] : walls
        for id in remove { w[id] = nil }
        for p in upsert {
            if p.kind == "floor", let y = p.corners.first?[safe: 1] { floorY = y }
            guard ["wall", "vertical", "door", "window"].contains(p.kind), let seg = Self.footprint(p.corners) else { continue }
            w[p.id] = Wall(a: seg.0, b: seg.1, kind: p.kind)
        }
        walls = w
    }

    /// Two most distant corners in x/z (same rule as the server and the web page).
    static func footprint(_ corners: [[Float]]) -> (CGPoint, CGPoint)? {
        let pts = corners.compactMap { $0.count >= 3 ? CGPoint(x: CGFloat($0[0]), y: CGFloat($0[2])) : nil }
        var best: (CGPoint, CGPoint)?
        var bestD: CGFloat = -1
        for i in pts.indices { for j in pts.indices where j > i {
            let d = hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y)
            if d > bestD { bestD = d; best = (pts[i], pts[j]) }
        } }
        return best
    }

    /// 0x02 | layer u8 | voxel size f32 | N x (int16 x,y,z, uint8 r,g,b)
    func addVoxels(_ data: Data) {
        guard data.count >= 6, data[data.startIndex] == 0x02 else { return }
        let bytes = [UInt8](data)
        let size = Float(bitPattern: UInt32(bytes[2]) | UInt32(bytes[3]) << 8 | UInt32(bytes[4]) << 16 | UInt32(bytes[5]) << 24)
        func i16(_ o: Int) -> Float { Float(Int16(bitPattern: UInt16(bytes[o]) | UInt16(bytes[o + 1]) << 8)) * size }
        var c = cells
        var o = 6
        while o + 9 <= bytes.count, c.count < Self.maxCells {
            let y = i16(o + 2)
            if y > floorY + 0.15, y < floorY + 1.9 { // obstacles only, not floor/ceiling
                let cx = Int64((i16(o) / Self.cell).rounded()), cz = Int64((i16(o + 4) / Self.cell).rounded())
                let key = cx << 32 | (cz & 0xffff_ffff)
                if c[key] == nil {
                    c[key] = Color(red: Double(bytes[o + 6]) / 255, green: Double(bytes[o + 7]) / 255, blue: Double(bytes[o + 8]) / 255)
                }
            }
            o += 9
        }
        cells = c
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}

/// Top-down minimap centred on me, entrance-side up. Shows walls, obstacles, colleagues, pins.
struct MinimapView: View {
    @ObservedObject var map: MapModel
    let me: SIMD3<Float>?
    let heading: Float? // radians, atan2(fwd.x, fwd.z)
    let myColor: String
    let peers: [Peer]
    let pins: [Pin]
    var radius: CGFloat = 7 // metres shown from centre to edge

    var body: some View {
        Canvas { ctx, size in
            let c = me.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.z)) } ?? .zero
            let s = min(size.width, size.height) / (2 * radius)
            func p(_ x: CGFloat, _ z: CGFloat) -> CGPoint { CGPoint(x: size.width / 2 + (x - c.x) * s, y: size.height / 2 + (z - c.y) * s) }

            ctx.fill(Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 14), with: .color(.black.opacity(0.6)))
            let cellPx = max(1, CGFloat(MapModel.cell) * s)
            for (key, color) in map.cells {
                let x = CGFloat(Float(key >> 32) * MapModel.cell), z = CGFloat(Float(Int32(truncatingIfNeeded: key)) * MapModel.cell)
                let q = p(x, z)
                guard abs(x - c.x) < radius, abs(z - c.y) < radius else { continue }
                ctx.fill(Path(CGRect(x: q.x - cellPx / 2, y: q.y - cellPx / 2, width: cellPx, height: cellPx)), with: .color(color.opacity(0.8)))
            }
            for w in map.walls.values {
                var path = Path()
                path.move(to: p(w.a.x, w.a.y))
                path.addLine(to: p(w.b.x, w.b.y))
                let col: Color = w.kind == "door" ? .green : w.kind == "window" ? .blue : .white
                ctx.stroke(path, with: .color(col), lineWidth: 2.5)
            }
            for pin in pins where pin.pos.count >= 3 {
                let q = p(CGFloat(pin.pos[0]), CGFloat(pin.pos[2]))
                ctx.fill(Path(ellipseIn: CGRect(x: q.x - 4, y: q.y - 4, width: 8, height: 8)), with: .color(pin.kind == "hazard" ? .red : pin.kind == "victim" ? .orange : pin.kind == "waypoint" ? .cyan : .green))
            }
            for peer in peers where peer.connected && !peer.stale {
                guard let pos = peer.pos, pos.count >= 3 else { continue }
                let q = p(CGFloat(pos[0]), CGFloat(pos[2]))
                ctx.fill(Path(ellipseIn: CGRect(x: q.x - 5, y: q.y - 5, width: 10, height: 10)), with: .color(Color(hex: peer.color)))
            }
            // me: arrow pointing along my heading
            let m = CGPoint(x: size.width / 2, y: size.height / 2)
            let h = CGFloat(heading ?? 0)
            let dir = CGPoint(x: sin(h), y: cos(h))
            let perp = CGPoint(x: -dir.y, y: dir.x)
            var arrow = Path()
            arrow.move(to: CGPoint(x: m.x + dir.x * 10, y: m.y + dir.y * 10))
            arrow.addLine(to: CGPoint(x: m.x - dir.x * 6 + perp.x * 6, y: m.y - dir.y * 6 + perp.y * 6))
            arrow.addLine(to: CGPoint(x: m.x - dir.x * 6 - perp.x * 6, y: m.y - dir.y * 6 - perp.y * 6))
            arrow.closeSubpath()
            ctx.fill(arrow, with: .color(Color(hex: myColor)))
            ctx.stroke(arrow, with: .color(.black), lineWidth: 1)
        }
        .frame(width: 150, height: 150)
        .allowsHitTesting(false)
    }
}
