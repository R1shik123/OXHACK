import Foundation

/// Pinned WSS link to the Sightline server. Reconnects forever; all callbacks arrive on the main queue.
final class NetworkClient: ObservableObject {
    @Published private(set) var connected = false
    @Published private(set) var peers: [Peer] = []
    @Published private(set) var pins: [Pin] = []
    @Published private(set) var myColor = "#ffffff"
    @Published private(set) var lastError: String?
    /// Set when the server rejects our token (revoked / unknown): the UI sends the user back to pairing.
    @Published private(set) var revoked = false
    /// Why the laptop ended this session (shown on the pairing screen afterwards).
    private(set) var endReason: String?
    /// Server thinks *I* fell / stopped moving: the UI asks "Are you OK?".
    @Published private(set) var myAlert: AlertInfo?
    /// Width of the entrance marker in use: the printed sheet (0.18 m) or the commander's screen.
    @Published private(set) var markerWidth: Double = 0.18
    /// My suggestion, turnaround countdown and exit route, from the laptop's planner.
    @Published private(set) var me: MeState?

    /// Called (on main) every time the socket (re)connects, so the AR side can resend its map.
    var onConnect: (() -> Void)?
    let map = MapModel()

    private(set) var pairing: Pairing?
    private var session: URLSession?
    private var task: URLSessionWebSocketTask?
    private var generation = 0
    private var pingTimer: Timer?

    static let binJPEG: UInt8 = 0x01
    static let binPoints: UInt8 = 0x02
    static let binDepth: UInt8 = 0x03
    static let binPosedJPEG: UInt8 = 0x05

    func connect(_ pairing: Pairing) {
        disconnect()
        self.pairing = pairing
        revoked = false
        lastError = nil
        session = URLSession(configuration: .default, delegate: PinnedDelegate(fingerprint: pairing.fingerprint), delegateQueue: nil)
        open()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 4, repeats: true) { [weak self] _ in
            self?.task?.sendPing { error in
                if error != nil { DispatchQueue.main.async { self?.reconnect() } }
            }
        }
    }

    func disconnect() {
        generation += 1
        pingTimer?.invalidate()
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        session?.invalidateAndCancel()
        session = nil
        connected = false
    }

    func send(_ message: [String: Any]) {
        guard connected, let task, let data = try? JSONSerialization.data(withJSONObject: message),
              let text = String(data: data, encoding: .utf8) else { return }
        task.send(.string(text)) { _ in }
    }

    func sendBinary(_ kind: UInt8, _ payload: Data) {
        guard connected, let task else { return }
        var d = Data([kind])
        d.append(payload)
        task.send(.data(d)) { _ in }
    }

    /// Authenticated request for a colleague's latest camera frame (pinned session).
    func fetchFrame(of peerID: String, completion: @escaping (Data?) -> Void) {
        guard let pairing, let session else { return completion(nil) }
        var req = URLRequest(url: pairing.httpBase.appendingPathComponent("api/frame/\(peerID).jpg"))
        req.setValue("Bearer \(pairing.token)", forHTTPHeaderField: "Authorization")
        req.cachePolicy = .reloadIgnoringLocalCacheData
        req.timeoutInterval = 2
        session.dataTask(with: req) { data, response, _ in
            let ok = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async { completion(ok ? data : nil) }
        }.resume()
    }

    /// GET a path on the paired server over the pinned, authenticated session (Map tab assets, frames).
    func fetch(_ pathAndQuery: String, timeout: TimeInterval = 8, completion: @escaping (Data?, HTTPURLResponse?) -> Void) {
        guard let pairing, let session, let url = URL(string: pathAndQuery, relativeTo: pairing.httpBase) else { return completion(nil, nil) }
        var req = URLRequest(url: url)
        req.setValue("Bearer \(pairing.token)", forHTTPHeaderField: "Authorization")
        req.cachePolicy = .reloadIgnoringLocalCacheData
        req.timeoutInterval = timeout
        session.dataTask(with: req) { data, response, _ in completion(data, response as? HTTPURLResponse) }.resume()
    }

    /// Read-only map stream (same data as the commander) for the Map tab, on the pinned session.
    func makeViewerTask() -> URLSessionWebSocketTask? {
        guard let pairing, let session, connected else { return nil }
        var req = URLRequest(url: pairing.httpBase.appendingPathComponent("ws/viewer"))
        req.url = URL(string: req.url!.absoluteString.replacingOccurrences(of: "https://", with: "wss://"))
        req.setValue("Bearer \(pairing.token)", forHTTPHeaderField: "Authorization")
        let t = session.webSocketTask(with: req)
        t.maximumMessageSize = 32 << 20
        return t
    }

    // MARK: - private

    private func open() {
        guard let pairing, let session else { return }
        generation += 1
        let gen = generation
        var req = URLRequest(url: pairing.wsURL)
        req.setValue("Bearer \(pairing.token)", forHTTPHeaderField: "Authorization")
        let t = session.webSocketTask(with: req)
        t.maximumMessageSize = 8 << 20
        task = t
        t.resume()
        receive(t, gen)
    }

    private func reconnect() {
        guard pairing != nil, !revoked else { return }
        connected = false
        task?.cancel(with: .goingAway, reason: nil)
        let gen = generation
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            guard let self, self.generation == gen else { return }
            self.open()
        }
    }

    private func receive(_ t: URLSessionWebSocketTask, _ gen: Int) {
        t.receive { [weak self] result in
            DispatchQueue.main.async {
                guard let self, self.generation == gen else { return }
                switch result {
                case .failure(let error):
                    // 4401/4403 = token unknown or revoked; an HTTP 403 on upgrade looks the same to us
                    let code = t.closeCode.rawValue
                    let http = (t.response as? HTTPURLResponse)?.statusCode
                    if code == 4401 || code == 4403 || http == 403 || http == 401 {
                        // 4403 = the commander revoked us; otherwise the laptop already ended the session
                        // (the connection dropped, e.g. Wi-Fi, and session-only pairings don't survive that).
                        self.endReason = code == 4403
                            ? "The commander removed this phone from the session."
                            : "The connection to the laptop dropped, so the laptop ended the session (pairings never survive a disconnect)."
                        self.revoked = true
                        self.connected = false
                        self.lastError = self.endReason
                        return
                    }
                    let e = error as NSError
                    self.lastError = e.code == NSURLErrorCancelled
                        ? "Server certificate doesn't match the pairing. Not connecting."
                        : "\(e.localizedDescription) [\(e.code)]"
                    self.reconnect()
                case .success(let msg):
                    switch msg {
                    case .string(let text): self.handle(text)
                    case .data(let data): self.map.addVoxels(data)
                    @unknown default: break
                    }
                    self.receive(t, gen)
                }
            }
        }
    }

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    private func handle(_ text: String) {
        guard let data = text.data(using: .utf8), let m = try? Self.decoder.decode(ServerMessage.self, from: data) else { return }
        switch m.t {
        case "welcome":
            if let w = m.markerM, (0.05...0.5).contains(w) { markerWidth = w }
            connected = true
            lastError = nil
            if let c = m.color { myColor = c }
            onConnect?()
        case "state":
            if !connected { connected = true }
            if let p = m.peers, p != peers { peers = p }
            if let p = m.pins, p != pins { pins = p }
            if m.me?.alert != myAlert { myAlert = m.me?.alert }
            if m.me != me { me = m.me }
        case "marker":
            if let w = m.markerM, (0.05...0.5).contains(w) { markerWidth = w }
        case "planes":
            map.apply(upsert: m.upsert ?? [], remove: m.remove ?? [], reset: m.reset ?? false)
        default:
            break
        }
    }
}
