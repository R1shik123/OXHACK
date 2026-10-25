import AVFoundation
import CoreMotion
import Foundation

/// Two motion signals for "is this responder OK?", both pre-built by Apple, both on-device:
///  1. Core Motion activity: Apple's pre-trained classifier on the motion coprocessor
///     (stationary / walking / running / automotive / cycling, with a confidence).
///  2. A fall signature in the accelerometer: free fall (|a| < 0.35 g for 80+ ms) followed within
///     1 s by an impact (|a| > 2.5 g). This is the classic physics-based fall detector.
/// Sensors run only while the responder screen is open (start/stop follow the app lifecycle).
final class MotionSensors: ObservableObject {
    @Published private(set) var activity = "unknown"
    @Published private(set) var confidence = 0 // 0 low, 1 medium, 2 high
    /// Called on the main queue when a free-fall + impact pattern is seen: (peak g, free-fall ms).
    var onImpact: ((Double, Int) -> Void)?

    private let activityManager = CMMotionActivityManager()
    private let motion = CMMotionManager()
    private let queue = OperationQueue()
    private var freeFallStart: TimeInterval?
    private var freeFallMs = 0
    private var freeFallEnded: TimeInterval?
    private var lastImpact: TimeInterval = 0

    func start() {
        if CMMotionActivityManager.isActivityAvailable() {
            activityManager.startActivityUpdates(to: .main) { [weak self] a in
                guard let self, let a else { return }
                self.activity = a.running ? "running" : a.walking ? "walking" : a.automotive ? "vehicle"
                    : a.cycling ? "cycling" : a.stationary ? "stationary" : "unknown"
                self.confidence = a.confidence.rawValue
            }
        }
        guard motion.isAccelerometerAvailable else { return }
        motion.accelerometerUpdateInterval = 1.0 / 50
        motion.startAccelerometerUpdates(to: queue) { [weak self] data, _ in
            guard let self, let a = data?.acceleration else { return }
            self.detect(sqrt(a.x * a.x + a.y * a.y + a.z * a.z), t: data!.timestamp)
        }
    }

    func stop() {
        activityManager.stopActivityUpdates()
        motion.stopAccelerometerUpdates()
    }

    private func detect(_ g: Double, t: TimeInterval) {
        if g < 0.35 { // falling: the phone (and person) is briefly weightless
            if freeFallStart == nil { freeFallStart = t }
            return
        }
        if let s = freeFallStart {
            freeFallMs = Int((t - s) * 1000)
            freeFallEnded = freeFallMs >= 80 ? t : nil
            freeFallStart = nil
        }
        if g > 2.5, let end = freeFallEnded, t - end < 1.0, t - lastImpact > 3 {
            lastImpact = t
            freeFallEnded = nil
            let ms = freeFallMs
            DispatchQueue.main.async { self.onImpact?(g, ms) }
        }
    }
}

/// Loud repeating chirp from a responder's phone while the laptop reports them as fallen or not moving,
/// so the commander and team can find them by ear (like a firefighter's PASS alarm). Plays even with
/// the silent switch on. Stops when they move, tap "I'm OK", or leave the app.
final class LocatorBeacon {
    static let shared = LocatorBeacon()
    // The audio engine is only built when an alarm actually fires, and always off the main thread:
    // setting it up while the screen opens could stall the app.
    private let queue = DispatchQueue(label: "sightline.beacon", qos: .userInitiated)
    private var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var on = false
    /// Read from the main thread by the voice guide, so it never reconfigures audio under the chirp.
    var isRunning: Bool { queue.sync { on } }

    func start() {
        queue.async { [self] in
            guard !on else { return }
            let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(.playback, mode: .default)
            try? session.setActive(true)
            if engine == nil {
                let e = AVAudioEngine(), p = AVAudioPlayerNode()
                e.attach(p)
                e.connect(p, to: e.mainMixerNode, format: format)
                engine = e
                player = p
            }
            guard let engine, let player, (try? engine.start()) != nil else { return }
            player.scheduleBuffer(Self.makeChirp(format), at: nil, options: .loops)
            player.play()
            on = true
        }
    }

    func stop() {
        queue.async { [self] in
            guard on else { return }
            player?.stop()
            engine?.stop()
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            on = false
        }
    }

    /// Three rising piercing tones (2.8-3.6 kHz, where hearing is most sensitive), then a pause; 1.2 s loop.
    private static func makeChirp(_ f: AVAudioFormat) -> AVAudioPCMBuffer {
        let sr = f.sampleRate
        let n = AVAudioFrameCount(sr * 1.2)
        let b = AVAudioPCMBuffer(pcmFormat: f, frameCapacity: n)!
        b.frameLength = n
        let d = b.floatChannelData![0]
        for i in 0..<Int(n) {
            let t = Double(i) / sr
            let k = Int(t / 0.16)
            let local = t - Double(k) * 0.16
            var v = 0.0
            if k < 3 && local < 0.13 {
                let env = min(1, local / 0.004, (0.13 - local) / 0.004)
                v = sin(2 * .pi * [2800.0, 3200.0, 3600.0][k] * t) * env
            }
            d[i] = Float(v * 0.95)
        }
        return b
    }
}
