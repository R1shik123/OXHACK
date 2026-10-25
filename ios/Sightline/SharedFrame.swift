import simd

/// The shared coordinate frame, anchored to the printed marker at the entrance.
///
/// Given the ARKit image-anchor transform M (in this phone's ARKit world):
///   p = M.columns.3.xyz                      origin
///   n = M.columns.1.xyz                      image normal (ARKit images lie in their local x-z plane)
///   if |horizontal n| < 0.3 use M.columns.2  (marker lying flat)
///   f = normalize(n.x, 0, n.z)
///   Y = (0,1,0), Z = f, X = cross(Y, Z), S = [X Y Z p], T = inverse(S)
/// Everything the phone sends is multiplied by T first; anything received is multiplied by S.
///
/// Pure simd so it can be checked on a Mac without Xcode: see ios/SharedFrameTests/main.swift.
/// The Python mirror lives in server/sightline/frame.py.
enum SharedFrame {
    static func markerBasis(_ m: simd_float4x4) -> (origin: SIMD3<Float>, forward: SIMD3<Float>) {
        let p = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        var n = SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z)
        if simd_length(SIMD2<Float>(n.x, n.z)) < 0.3 {
            n = SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z)
        }
        return (p, horizontal(n))
    }

    /// S = [X Y Z p]: maps shared-frame coordinates into this phone's ARKit world.
    static func sharedToLocal(origin p: SIMD3<Float>, forward f: SIMD3<Float>) -> simd_float4x4 {
        let y = SIMD3<Float>(0, 1, 0)
        let z = horizontal(f)
        let x = simd_cross(y, z)
        return simd_float4x4(SIMD4<Float>(x, 0), SIMD4<Float>(y, 0), SIMD4<Float>(z, 0), SIMD4<Float>(p, 1))
    }

    /// T = inverse(S) for a single marker observation.
    static func localToShared(marker m: simd_float4x4) -> simd_float4x4 {
        let b = markerBasis(m)
        return rigidInverse(sharedToLocal(origin: b.origin, forward: b.forward))
    }

    /// S from several marker observations (averages origin and forward to kill detection jitter).
    static func sharedToLocal(averaging samples: [simd_float4x4]) -> simd_float4x4? {
        guard !samples.isEmpty else { return nil }
        var p = SIMD3<Float>(repeating: 0)
        var f = SIMD3<Float>(repeating: 0)
        for m in samples {
            let b = markerBasis(m)
            p += b.origin
            f += b.forward
        }
        return sharedToLocal(origin: p / Float(samples.count), forward: f)
    }

    /// Robust S from many marker observations: component-wise median origin and median heading,
    /// so a few bad detections can't drag the frame. `spread` = median distance of the samples'
    /// origins from the result (metres): the lock's precision.
    static func sharedToLocal(median samples: [simd_float4x4]) -> (s: simd_float4x4, spread: Float)? {
        guard !samples.isEmpty else { return nil }
        let bases = samples.map(markerBasis)
        func median(_ v: [Float]) -> Float {
            let o = v.sorted()
            return o.count % 2 == 1 ? o[o.count / 2] : (o[o.count / 2 - 1] + o[o.count / 2]) / 2
        }
        let p = SIMD3<Float>(median(bases.map(\.origin.x)), median(bases.map(\.origin.y)), median(bases.map(\.origin.z)))
        // median heading, measured relative to the first sample so angles never wrap
        let ref = atan2(bases[0].forward.x, bases[0].forward.z)
        let yaw = ref + median(bases.map { b -> Float in
            var d = atan2(b.forward.x, b.forward.z) - ref
            while d > .pi { d -= 2 * .pi }
            while d < -.pi { d += 2 * .pi }
            return d
        })
        let spread = median(bases.map { simd_distance($0.origin, p) })
        return (sharedToLocal(origin: p, forward: SIMD3<Float>(sin(yaw), 0, cos(yaw))), spread)
    }

    /// Inverse of a rotation + translation matrix (cheaper and better conditioned than simd_inverse).
    static func rigidInverse(_ s: simd_float4x4) -> simd_float4x4 {
        let r = simd_float3x3(SIMD3<Float>(s.columns.0.x, s.columns.0.y, s.columns.0.z),
                              SIMD3<Float>(s.columns.1.x, s.columns.1.y, s.columns.1.z),
                              SIMD3<Float>(s.columns.2.x, s.columns.2.y, s.columns.2.z))
        let rt = r.transpose
        let t = -(rt * SIMD3<Float>(s.columns.3.x, s.columns.3.y, s.columns.3.z))
        return simd_float4x4(SIMD4<Float>(rt.columns.0, 0), SIMD4<Float>(rt.columns.1, 0), SIMD4<Float>(rt.columns.2, 0), SIMD4<Float>(t, 1))
    }

    private static func horizontal(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let h = SIMD3<Float>(v.x, 0, v.z)
        let l = simd_length(h)
        return l > 1e-6 ? h / l : SIMD3<Float>(0, 0, 1)
    }
}

extension simd_float4x4 {
    func point(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let r = self * SIMD4<Float>(v, 1)
        return SIMD3<Float>(r.x, r.y, r.z)
    }

    func direction(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let r = self * SIMD4<Float>(v, 0)
        return SIMD3<Float>(r.x, r.y, r.z)
    }

    var translation: SIMD3<Float> { SIMD3<Float>(columns.3.x, columns.3.y, columns.3.z) }
}

extension SIMD3 where Scalar == Float {
    var array: [Float] { [x, y, z] }
}

/// [x, y, z] from the server -> vector (nil if missing or short).
func vec3(_ a: [Float]?) -> SIMD3<Float>? {
    guard let a, a.count >= 3 else { return nil }
    return SIMD3<Float>(a[0], a[1], a[2])
}
