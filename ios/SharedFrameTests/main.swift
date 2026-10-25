// Checks SharedFrame.swift on a Mac, no Xcode needed:
//   swiftc -O ios/Sightline/SharedFrame.swift ios/SharedFrameTests/main.swift -o /tmp/sf && /tmp/sf
// Mirrors server/tests/test_frame.py so the phone and the simulator agree on the math.
import Foundation
import simd

var failures = 0
func check(_ ok: Bool, _ what: String) {
    print(ok ? "ok   \(what)" : "FAIL \(what)")
    if !ok { failures += 1 }
}
func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tol: Float = 1e-4) -> Bool { simd_length(a - b) < tol }

func yaw(_ theta: Float, _ t: SIMD3<Float> = .zero) -> simd_float4x4 {
    var m = simd_float4x4(simd_quatf(angle: theta, axis: [0, 1, 0]))
    m.columns.3 = SIMD4<Float>(t, 1)
    return m
}

// Marker upright on a wall, facing +Z: image x = +X, image normal (col 1) = +Z, image z = -Y.
let upright = simd_float4x4(SIMD4<Float>(1, 0, 0, 0), SIMD4<Float>(0, 0, 1, 0), SIMD4<Float>(0, -1, 0, 0), SIMD4<Float>(0, 0, 0, 1))

// 1. The marker itself maps to the origin, 1 m in front of it maps to +Z.
do {
    let m = yaw(0.7, [2, 1.3, -4]) * upright
    let t = SharedFrame.localToShared(marker: m)
    check(close(t.point([2, 1.3, -4]), .zero), "marker origin -> (0,0,0)")
    let n = SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z)
    check(close(t.point(SIMD3<Float>(2, 1.3, -4) + n), [0, 0, 1]), "1 m out of the wall -> (0,0,1)")
}

// 2. Five phones with different ARKit origins agree on the same physical point.
do {
    var rng = SystemRandomNumberGenerator()
    let worldMarker = yaw(0, [0, 1.4, 0]) * upright
    let worldPoint = SIMD3<Float>(3, 1.1, 5)
    var results: [SIMD3<Float>] = []
    for _ in 0..<5 {
        let localFromWorld = yaw(Float.random(in: -.pi ... .pi, using: &rng), [Float.random(in: -9...9, using: &rng), Float.random(in: -1...1, using: &rng), Float.random(in: -9...9, using: &rng)])
        let t = SharedFrame.localToShared(marker: localFromWorld * worldMarker)
        results.append(t.point(localFromWorld.point(worldPoint)))
    }
    check(results.allSatisfy { close($0, results[0], 1e-3) }, "5 random phone frames agree")
    check(close(results[0], [3, -0.3, 5], 1e-3), "shared coords = (3, -0.3, 5)")
}

// 3. A tilted marker still gives a gravity-aligned frame.
do {
    let tilt = simd_float4x4(SIMD4<Float>(1, 0, 0, 0), SIMD4<Float>(0, 0.3, 0.95, 0), SIMD4<Float>(0, -0.95, 0.3, 0), SIMD4<Float>(0, 0, 0, 1))
    check(close(SharedFrame.localToShared(marker: tilt).direction([0, 1, 0]), [0, 1, 0]), "tilted marker keeps Y up")
}

// 4. A marker lying flat falls back to column 2.
do {
    let flat = simd_float4x4(SIMD4<Float>(1, 0, 0, 0), SIMD4<Float>(0, 1, 0, 0), SIMD4<Float>(0, 0, 1, 0), SIMD4<Float>(0.5, 0, 0.5, 1))
    let b = SharedFrame.markerBasis(flat)
    check(close(b.forward, [0, 0, 1]) && close(b.origin, [0.5, 0, 0.5]), "flat marker uses column 2")
}

// 5. Averaging and the rigid inverse.
do {
    let m = yaw(1.1, [1, 2, 3]) * upright
    let s = SharedFrame.sharedToLocal(averaging: [m, yaw(0.004, .zero) * m, yaw(-0.004, .zero) * m])!
    let i = SharedFrame.rigidInverse(s) * s
    check(close(i.columns.0.xyz, [1, 0, 0]) && close(i.columns.3.xyz, .zero), "rigidInverse(S) * S = I")
    check(close(SharedFrame.localToShared(marker: m).point([5, 0, 5]), SharedFrame.rigidInverse(s).point([5, 0, 5]), 0.05), "averaged frame ~ single frame")
}

// 6. Median lock ignores a couple of wild detections; spread reports precision.
do {
    let m = yaw(0.4, [1, 1.4, 2]) * upright
    var samples = (0..<28).map { i in yaw(Float(i % 3 - 1) * 0.002, [Float(i % 5) * 0.001, 0, 0]) * m }
    samples.append(yaw(0.8, [0.5, 0, 0]) * m) // bad detections
    samples.append(yaw(-0.6, [0, 0.4, -0.3]) * m)
    let (s, spread) = SharedFrame.sharedToLocal(median: samples)!
    let single = SharedFrame.sharedToLocal(averaging: [m])!
    check(close(s.columns.3.xyz, single.columns.3.xyz, 0.02) && close(s.columns.2.xyz, single.columns.2.xyz, 0.01), "median lock rejects outliers")
    check(spread < 0.02, "spread reports a tight lock (\(spread * 1000) mm)")
}

extension SIMD4 where Scalar == Float { var xyz: SIMD3<Float> { SIMD3<Float>(x, y, z) } }

print(failures == 0 ? "PASS" : "FAIL (\(failures))")
exit(failures == 0 ? 0 : 1)
