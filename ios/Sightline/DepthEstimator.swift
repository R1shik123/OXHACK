import ARKit
import CoreImage
import CoreML
import Vision

/// Camera-only 3D: Depth Anything V2 (Core ML, Apple's conversion) gives relative depth per pixel;
/// we calibrate it to metres every frame against depths ARKit knows (feature points + plane hits),
/// then unproject into coloured points in the shared frame. Runs off the main thread, one frame at a time.
final class DepthEstimator {
    static let width = 518, height = 392 // model input/output (landscape, like the camera sensor)
    static let stride = 6
    static let frameW = 128, frameH = 96 // depth frame sent to the laptop for 3D fusion

    struct Job {
        let image: CVPixelBuffer
        let camera: simd_float4x4 // camera -> ARKit world
        let intrinsics: simd_float3x3 // for `imageSize`
        let imageSize: CGSize
        let anchors: [SIMD3<Float>] // (u, v) normalized image coords + true metric depth
        let toShared: simd_float4x4
        let findFurniture: Bool // YOLO runs every other frame (about 1 Hz), and not when the phone is warm
        /// Pro iPhones / iPads with a LiDAR sensor: ARKit's measured depth (metres) + confidence. When present
        /// it replaces the Depth Anything estimate entirely: centimetre-accurate, plain walls included.
        var lidarDepth: CVPixelBuffer? = nil
        var lidarConfidence: CVPixelBuffer? = nil
    }

    struct Output {
        let points: Data // N x 16 B: float32 x,y,z (shared frame), uint8 r,g,b,confidence
        let count: Int
        let centerDepth: Float?
        let fitError: Float
        /// Metric depth image for the laptop's surface fusion (see `depthFrame`).
        let depthFrame: Data
    }

    /// YOLOv8n (Open Images V7, 601 classes) classes that count as furniture or doors. Used only to outline
    /// them for sharper 3D fusion and to find doorways: no class name ever leaves the phone or appears anywhere.
    static let furniture: Set<String> = ["Chair", "Couch", "Sofa bed", "Studio couch", "Bed", "Bench", "Table", "Coffee table",
                                         "Kitchen & dining room table", "Desk", "Stool", "Cabinetry", "Cupboard", "Wardrobe",
                                         "Bookcase", "Shelf", "Chest of drawers", "Filing cabinet", "Bathroom cabinet", "Furniture"]
    static let doors: Set<String> = ["Door"]
    static let stairs: Set<String> = ["Stairs"]

    /// An outline in the raw landscape model image: a doorway, a flight of stairs, or a piece of furniture.
    struct Outline { let rect: CGRect; var door = false; var stairs = false }

    private let model: MLModel?
    private let yolo: VNCoreMLModel?
    private let ci = CIContext(options: [.cacheIntermediates: false])
    private var input: CVPixelBuffer?
    /// Calibration: inverse = true means 1/z = a*d + b, else z = a*d + b. Smoothed over frames.
    private var calib: (a: Float, b: Float, inverse: Bool)?
    private var calibTime: CFTimeInterval = 0
    let queue = DispatchQueue(label: "sightline.depth", qos: .userInitiated)

    init() {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        if let url = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlmodelc") {
            model = try? MLModel(contentsOf: url, configuration: cfg)
        } else {
            model = nil
        }
        if let url = Bundle.main.url(forResource: "YOLOv8nOIV7", withExtension: "mlmodelc"),
           let m = try? MLModel(contentsOf: url, configuration: cfg) {
            yolo = try? VNCoreMLModel(for: m)
        } else {
            yolo = nil
        }
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        CVPixelBufferCreate(nil, Self.width, Self.height, kCVPixelFormatType_32BGRA, attrs, &input)
    }

    var available: Bool { model != nil && input != nil }

    func reset() { calib = nil }

    /// LiDAR path: same outputs as `run`, from ARKit's measured depth instead of a neural estimate.
    /// Low-confidence pixels are dropped; the depth frame goes out at the sensor's full 256 x 192.
    func runLiDAR(_ job: Job) -> Output? {
        guard let input, let depthBuf = job.lidarDepth else { return nil }
        let W = Self.width, H = Self.height
        let sx = CGFloat(W) / job.imageSize.width, sy = CGFloat(H) / job.imageSize.height
        ci.render(CIImage(cvPixelBuffer: job.image).transformed(by: CGAffineTransform(scaleX: sx, y: sy)), to: input)
        CVPixelBufferLockBaseAddress(depthBuf, .readOnly)
        CVPixelBufferLockBaseAddress(input, .readOnly)
        if let c = job.lidarConfidence { CVPixelBufferLockBaseAddress(c, .readOnly) }
        defer {
            CVPixelBufferUnlockBaseAddress(depthBuf, .readOnly)
            CVPixelBufferUnlockBaseAddress(input, .readOnly)
            if let c = job.lidarConfidence { CVPixelBufferUnlockBaseAddress(c, .readOnly) }
        }
        guard let dBase = CVPixelBufferGetBaseAddress(depthBuf), let cBase = CVPixelBufferGetBaseAddress(input) else { return nil }
        let dRow = CVPixelBufferGetBytesPerRow(depthBuf), cRow = CVPixelBufferGetBytesPerRow(input)
        let dw = CVPixelBufferGetWidth(depthBuf), dh = CVPixelBufferGetHeight(depthBuf)
        let conf = job.lidarConfidence.flatMap { CVPixelBufferGetBaseAddress($0) }
        let confRow = job.lidarConfidence.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
        func metric(_ x: Int, _ y: Int) -> Float {
            let xx = min(max(x * dw / W, 0), dw - 1), yy = min(max(y * dh / H, 0), dh - 1)
            if let conf, conf.advanced(by: yy * confRow + xx).assumingMemoryBound(to: UInt8.self).pointee < UInt8(ARConfidenceLevel.medium.rawValue) {
                return -1 // the sensor itself isn't sure (edges, dark or shiny surfaces)
            }
            return dBase.advanced(by: yy * dRow).assumingMemoryBound(to: Float32.self)[xx]
        }
        return finish(job, sx: sx, sy: sy, metric: metric, fitError: 0, frameW: dw, frameH: dh, lidar: true) { x, y in
            let px = cBase.advanced(by: y * cRow + x * 4).assumingMemoryBound(to: UInt8.self)
            return (px[2], px[1], px[0])
        }
    }

    func run(_ job: Job) -> Output? {
        guard let model, let input else { return nil }
        let W = Self.width, H = Self.height
        let sx = CGFloat(W) / job.imageSize.width, sy = CGFloat(H) / job.imageSize.height
        let src = CIImage(cvPixelBuffer: job.image).transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        ci.render(src, to: input)

        guard let out = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])),
              let depthBuf = out.featureValue(for: "depth")?.imageBufferValue else { return nil }

        CVPixelBufferLockBaseAddress(depthBuf, .readOnly)
        CVPixelBufferLockBaseAddress(input, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthBuf, .readOnly)
            CVPixelBufferUnlockBaseAddress(input, .readOnly)
        }
        guard let dBase = CVPixelBufferGetBaseAddress(depthBuf), let cBase = CVPixelBufferGetBaseAddress(input) else { return nil }
        let dRow = CVPixelBufferGetBytesPerRow(depthBuf), cRow = CVPixelBufferGetBytesPerRow(input)
        let dw = CVPixelBufferGetWidth(depthBuf), dh = CVPixelBufferGetHeight(depthBuf)
        let fmt = CVPixelBufferGetPixelFormatType(depthBuf)
        func raw(_ x: Int, _ y: Int) -> Float {
            let xx = min(max(x * dw / W, 0), dw - 1), yy = min(max(y * dh / H, 0), dh - 1)
            let row = dBase.advanced(by: yy * dRow)
            if fmt == kCVPixelFormatType_OneComponent32Float {
                return row.assumingMemoryBound(to: Float.self)[xx]
            }
            return Float(Float16(bitPattern: row.assumingMemoryBound(to: UInt16.self)[xx]))
        }

        // --- calibrate relative output to metres -------------------------------------------
        var samples: [(d: Float, z: Float)] = []
        for a in job.anchors where a.x > 0 && a.x < 1 && a.y > 0 && a.y < 1 && a.z > 0.2 && a.z < 10 {
            samples.append((raw(Int(a.x * Float(W)), Int(a.y * Float(H))), a.z))
        }
        let fitted = Self.fit(samples)
        let now = CACurrentMediaTime()
        if let f = fitted {
            if let c = calib, c.inverse == f.inverse, now - calibTime < 1.5 {
                calib = (0.6 * c.a + 0.4 * f.a, 0.6 * c.b + 0.4 * f.b, f.inverse)
            } else {
                calib = (f.a, f.b, f.inverse)
            }
            calibTime = now
        }
        // A stale calibration (camera moved on, scene changed) would put points in the wrong place.
        guard let c = calib, now - calibTime < 1.0 else { return nil }
        func metric(_ d: Float) -> Float {
            let v = c.a * d + c.b
            return c.inverse ? (v > 1e-3 ? 1 / v : -1) : v
        }
        return finish(job, sx: sx, sy: sy, metric: { metric(raw($0, $1)) }, fitError: fitted?.err ?? -1,
                      frameW: Self.frameW, frameH: Self.frameH, lidar: false) { x, y in
            let px = cBase.advanced(by: y * cRow + x * 4).assumingMemoryBound(to: UInt8.self)
            return (px[2], px[1], px[0])
        }
    }

    /// Shared by both depth sources: coloured points for the live map and the depth frame for fusion.
    /// `metric(u, v)` is depth in metres at model-image pixel (u, v) of the W x H landscape image (<= 0 = unknown).
    private func finish(_ job: Job, sx: CGFloat, sy: CGFloat, metric: (Int, Int) -> Float, fitError: Float,
                        frameW: Int, frameH: Int, lidar: Bool, color: (Int, Int) -> (UInt8, UInt8, UInt8)) -> Output? {
        guard let input else { return nil }
        let W = Self.width, H = Self.height

        // --- unproject a grid of pixels ----------------------------------------------------
        let K = job.intrinsics
        let fx = K[0][0] * Float(sx), fy = K[1][1] * Float(sy), cx = K[2][0] * Float(sx), cy = K[2][1] * Float(sy)
        let M = job.toShared * job.camera
        var data = Data(capacity: (W / Self.stride) * (H / Self.stride) * 16)
        var n = 0
        let s = Self.stride
        for v in Swift.stride(from: s / 2, to: H - s, by: s) {
            for u in Swift.stride(from: s / 2, to: W - s, by: s) {
                let z = metric(u, v)
                guard z > 0.25, z < 7 else { continue }
                // drop "flying pixels" on depth edges
                let zr = metric(u + s, v), zd = metric(u, v + s)
                guard zr > 0, zd > 0, abs(zr - z) / z < 0.08, abs(zd - z) / z < 0.08 else { continue }
                let pc = SIMD3<Float>((Float(u) + 0.5 - cx) / fx * z, -(Float(v) + 0.5 - cy) / fy * z, -z)
                let p = M.point(pc)
                let px = color(u, v)
                let conf = UInt8(max(64, min(255, 255 - Int((z - 3) * (lidar ? 25 : 50)))))
                var rec = (p.x, p.y, p.z, px.0, px.1, px.2, conf)
                withUnsafeBytes(of: &rec.0) { data.append(contentsOf: $0) }
                withUnsafeBytes(of: &rec.1) { data.append(contentsOf: $0) }
                withUnsafeBytes(of: &rec.2) { data.append(contentsOf: $0) }
                data.append(contentsOf: [rec.3, rec.4, rec.5, rec.6])
                n += 1
            }
        }
        let center = metric(W / 2, H / 2)
        let outlines = job.findFurniture ? furnitureOutlines(in: input) : []
        let frame = depthFrame(M: M, fx: fx, fy: fy, cx: cx, cy: cy, outlines: outlines, w: frameW, h: frameH, lidar: lidar,
                               metric: metric, color: color)
        return Output(points: data, count: n, centerDepth: center > 0 ? center : nil, fitError: fitError, depthFrame: frame)
    }

    /// Depth frame wire format (after the 0x03 type byte), little-endian:
    ///   u8 version=1 | u16 w | u16 h | f32[16] camera->shared (column-major) | f32 fx, fy, cx, cy (for w x h)
    ///   | u16[w*h] depth in mm (0 = invalid) | u8[w*h*3] rgb
    /// Camera convention (ARKit): x right, y up, looking down -z; pixel v grows downwards.
    private func depthFrame(M: simd_float4x4, fx: Float, fy: Float, cx: Float, cy: Float, outlines: [Outline], w: Int, h: Int, lidar: Bool,
                            metric: (Int, Int) -> Float, color: (Int, Int) -> (UInt8, UInt8, UInt8)) -> Data {
        let W = Self.width, H = Self.height
        let sx = Float(w) / Float(W), sy = Float(h) / Float(H)
        var out = Data(capacity: 85 + w * h * 5)
        func f32(_ v: Float) { var x = v; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
        out.append(lidar ? 3 : 2) // version 2: furniture outlines follow the colour; 3: same, depth measured by LiDAR
        u16(UInt16(w)); u16(UInt16(h))
        for c in 0..<4 { for r in 0..<4 { f32(M[c][r]) } }
        f32(fx * sx); f32(fy * sy); f32(cx * sx); f32(cy * sy)
        var depth = [UInt16](repeating: 0, count: w * h)
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        var z = [Float](repeating: -1, count: w * h)
        for v in 0..<h { for u in 0..<w {
            z[v * w + u] = metric(min(W - 1, Int((Float(u) + 0.5) / sx)), min(H - 1, Int((Float(v) + 0.5) / sy)))
        } }
        for v in 0..<h { for u in 0..<w {
            let i = v * w + u, d = z[i]
            guard d > 0.25, d < 6.5 else { continue }
            // drop flying pixels: depth jumps of > 15% to a neighbour are object edges, not surfaces
            let r = u + 1 < w ? z[i + 1] : d, b = v + 1 < h ? z[i + w] : d
            guard r > 0, b > 0, abs(r - d) / d < 0.15, abs(b - d) / d < 0.15 else { continue }
            depth[i] = UInt16(min(65535, Int(d * 1000)))
            let c = color(min(W - 1, Int((Float(u) + 0.5) / sx)), min(H - 1, Int((Float(v) + 0.5) / sy)))
            rgb[i * 3] = c.0; rgb[i * 3 + 1] = c.1; rgb[i * 3 + 2] = c.2
        } }
        depth.withUnsafeBytes { out.append(contentsOf: $0) }
        out.append(contentsOf: rgb)
        // outlines, scaled from the model image (W x H) to the depth frame (w x h). Doors set the top bit of x0
        // (frames are at most 256 wide, so it's free); a laptop that doesn't know this simply ignores the box.
        let boxes = outlines.prefix(32).compactMap { o -> (UInt16, UInt16, UInt16, UInt16)? in
            let r = o.rect
            let x0 = max(0, Int(r.minX * CGFloat(sx))), y0 = max(0, Int(r.minY * CGFloat(sy)))
            let x1 = min(w, Int(ceil(r.maxX * CGFloat(sx)))), y1 = min(h, Int(ceil(r.maxY * CGFloat(sy))))
            let flag: UInt16 = o.door ? 0x8000 : o.stairs ? 0x4000 : 0 // doors: top bit, stairs: bit 14
            return x1 > x0 && y1 > y0 ? (UInt16(x0) | flag, UInt16(y0), UInt16(x1), UInt16(y1)) : nil
        }
        out.append(UInt8(boxes.count))
        for b in boxes { u16(b.0); u16(b.1); u16(b.2); u16(b.3) }
        return out
    }

    /// YOLOv8n (Open Images) on the model input (phone held upright, so Vision orientation .right). Returns
    /// furniture and door outlines as rectangles in the raw landscape model image (W x H pixels); class names
    /// are dropped here.
    private func furnitureOutlines(in buffer: CVPixelBuffer) -> [Outline] {
        guard let yolo else { return [] }
        let req = VNCoreMLRequest(model: yolo)
        req.imageCropAndScaleOption = .scaleFill
        guard (try? VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .right).perform([req])) != nil else { return [] }
        let W = CGFloat(Self.width), H = CGFloat(Self.height)
        var out: [Outline] = []
        for case let obs as VNRecognizedObjectObservation in req.results ?? [] {
            guard let top = obs.labels.first, top.confidence >= 0.35 else { continue }
            let door = Self.doors.contains(top.identifier), stairs = Self.stairs.contains(top.identifier)
            guard door || stairs || Self.furniture.contains(top.identifier) else { continue }
            let b = obs.boundingBox // oriented (portrait) image, normalized, origin bottom-left
            // oriented -> raw landscape: x_raw = (1 - v) * W, y_raw = H - u * H   (verified mapping, see tests)
            let xs = [(1 - b.minY) * W, (1 - b.maxY) * W], ys = [H - b.minX * H, H - b.maxX * H]
            out.append(Outline(rect: CGRect(x: xs.min()!, y: ys.min()!, width: abs(xs[0] - xs[1]), height: abs(ys[0] - ys[1])),
                               door: door, stairs: stairs))
        }
        return out
    }

    /// Robust fit (RANSAC on pairs, then least squares on the inliers) for both z = a*d + b and
    /// 1/z = a*d + b; keeps whichever explains ARKit's depths best, if it's plausible.
    static func fit(_ s: [(d: Float, z: Float)]) -> (a: Float, b: Float, inverse: Bool, err: Float)? {
        guard s.count >= 6 else { return nil }
        func zOf(_ a: Float, _ b: Float, _ d: Float, _ inverse: Bool) -> Float {
            let v = a * d + b
            return inverse ? (v > 1e-3 ? 1 / v : 1e3) : v
        }
        func relErr(_ a: Float, _ b: Float, _ p: (d: Float, z: Float), _ inverse: Bool) -> Float { abs(zOf(a, b, p.d, inverse) - p.z) / p.z }
        var best: (a: Float, b: Float, inverse: Bool, err: Float)?
        var rng = SystemRandomNumberGenerator()
        for inverse in [true, false] {
            let ys = s.map { inverse ? 1 / $0.z : $0.z }
            var bestInliers: [Int] = []
            for _ in 0..<80 {
                let i = Int.random(in: 0..<s.count, using: &rng), j = Int.random(in: 0..<s.count, using: &rng)
                guard i != j, abs(s[i].d - s[j].d) > 1e-6 else { continue }
                let a = (ys[i] - ys[j]) / (s[i].d - s[j].d), b = ys[i] - a * s[i].d
                let inl = s.indices.filter { relErr(a, b, s[$0], inverse) < 0.08 }
                if inl.count > bestInliers.count { bestInliers = inl }
            }
            guard bestInliers.count >= max(6, s.count * 2 / 5) else { continue } // need 40% agreement
            // least squares on the inliers
            let xs = bestInliers.map { s[$0].d }, yv = bestInliers.map { ys[$0] }
            let n = Float(xs.count), mx = xs.reduce(0, +) / n, my = yv.reduce(0, +) / n
            var sxx: Float = 0, sxy: Float = 0
            for k in xs.indices { sxx += (xs[k] - mx) * (xs[k] - mx); sxy += (xs[k] - mx) * (yv[k] - my) }
            guard sxx > 1e-9 else { continue }
            let a = sxy / sxx, b = my - a * mx
            let errs = bestInliers.map { relErr(a, b, s[$0], inverse) }.sorted()
            let med = errs[errs.count / 2]
            let support = Float(bestInliers.count) / Float(s.count)
            // prefer more support, then lower error
            let score = med / support
            if med < 0.15, best == nil || score < best!.err { best = (a, b, inverse, score) }
        }
        return best
    }
}

