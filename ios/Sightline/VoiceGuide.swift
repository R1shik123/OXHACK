import AVFoundation
import Foundation

/// Speaks the laptop's voice cues (trip hazards, doors, turns, alerts). The laptop decides what to say;
/// this plays its ElevenLabs audio when it arrives quickly, and the phone's own voice otherwise
/// (no key on the laptop, slow network, or a cue with a name / pin label, which never leaves the LAN).
/// Most urgent first: an alert cuts off a door announcement. Nothing is stored.
final class VoiceGuide: NSObject, ObservableObject, AVAudioPlayerDelegate, AVSpeechSynthesizerDelegate {
    enum Mode: String, CaseIterable, Identifiable {
        case all, alerts, off
        var id: String { rawValue }
        var title: String {
            switch self {
            case .all: "All guidance"
            case .alerts: "Alerts and hazards only"
            case .off: "Off"
            }
        }
    }

    @Published var mode: Mode {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: "voiceMode")
            if mode == .off { stop() }
        }
    }

    private struct Item {
        let cue: VoiceCue
        let received: Date
    }

    private let net: NetworkClient
    private var lastN = 0
    private var queue: [Item] = []
    private var current: Item?
    private var player: AVAudioPlayer?
    private let synth = AVSpeechSynthesizer()
    private var token = 0 // bumps on every interruption, so a slow audio download can't play late
    private var serverVoiceBackoff = Date.distantPast // after a failed download, use the phone's voice for a while

    init(net: NetworkClient) {
        self.net = net
        mode = Mode(rawValue: UserDefaults.standard.string(forKey: "voiceMode") ?? "") ?? .all
        super.init()
        synth.delegate = self
    }

    /// The cue list from the latest state message (the laptop keeps the last few; skip what we've played).
    func receive(_ cues: [VoiceCue]) {
        let newest = cues.map(\.n).max() ?? lastN
        if newest < lastN { lastN = 0 } // a new session on the laptop restarts the count
        // first message of a session: only what's current, not cues from before we connected
        let latestTs = cues.map(\.ts).max() ?? 0
        let fresh = cues.filter { $0.n > lastN && (lastN > 0 || latestTs - $0.ts < 5) }
        lastN = max(lastN, newest)
        guard mode != .off else { return }
        let wanted = fresh.filter { mode == .all || $0.prio <= 1 }
        guard !wanted.isEmpty else { return }
        let now = Date()
        queue.append(contentsOf: wanted.map { Item(cue: $0, received: now) })
        queue.sort { ($0.cue.prio, $0.cue.n) < ($1.cue.prio, $1.cue.n) }
        if let cur = current, let top = queue.first, top.cue.prio < cur.cue.prio {
            interrupt() // something more urgent than what's playing
        }
        playNext()
    }

    /// Session over / app in the background: silence and forget.
    func reset() {
        stop()
        lastN = 0
    }

    func stop() {
        queue.removeAll()
        interrupt()
        deactivate()
    }

    // MARK: - playback

    private func interrupt() {
        token += 1
        player?.stop()
        player = nil
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
        current = nil
    }

    private func playNext() {
        guard current == nil else { return }
        while let item = queue.first {
            queue.removeFirst()
            let age = Date().timeIntervalSince(item.received)
            // directions and doors go stale fast; a newer direction replaces an older one
            if item.cue.prio >= 2, age > 5 || queue.contains(where: { $0.cue.kind == item.cue.kind && $0.cue.n > item.cue.n }) { continue }
            current = item
            play(item)
            return
        }
        deactivate()
    }

    private func play(_ item: Item) {
        activate()
        token += 1
        let mine = token
        guard !item.cue.local, let id = item.cue.id, Date() > serverVoiceBackoff else { return speakLocally(item.cue.text) }
        // urgent cues can't wait for the network; others get a moment for the nicer voice
        net.fetch("api/voice/\(id).mp3", timeout: item.cue.prio == 0 ? 0.8 : 1.5) { [weak self] data, response in
            DispatchQueue.main.async {
                guard let self, self.token == mine else { return }
                if response?.statusCode == 200, let data, let p = try? AVAudioPlayer(data: data) {
                    p.delegate = self
                    self.player = p
                    if p.play() { return }
                }
                self.serverVoiceBackoff = Date().addingTimeInterval(response?.statusCode == 404 ? 60 : 20)
                self.speakLocally(item.cue.text)
            }
        }
    }

    private func speakLocally(_ text: String) {
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "en-GB") ?? AVSpeechSynthesisVoice(language: "en-US")
        u.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        synth.speak(u)
    }

    private func finished() {
        current = nil
        player = nil
        playNext()
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async { if player === self.player { self.finished() } }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.async { if self.player == nil, self.current != nil { self.finished() } }
    }

    // Spoken over other audio (music is ducked, not stopped) and through the silent switch, like the locator chirp.
    private func activate() {
        guard !LocatorBeacon.shared.isRunning else { return } // already playing through the silent switch; a change would stop the chirp
        let s = AVAudioSession.sharedInstance()
        try? s.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers, .interruptSpokenAudioAndMixWithOthers])
        try? s.setActive(true)
    }

    private func deactivate() {
        guard !LocatorBeacon.shared.isRunning else { return }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
