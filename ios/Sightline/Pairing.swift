import CryptoKit
import Foundation
import Security

/// What a phone learns from the pairing QR code, plus the token the server issues.
/// Held in memory only for this app session; never written to disk, Keychain or UserDefaults.
struct Pairing: Codable, Equatable {
    var host: String
    var port: Int
    var fingerprint: String // SHA-256 of the server certificate (DER), lowercase hex
    var deviceID: String
    var token: String
    var name: String

    var httpBase: URL { URL(string: "https://\(host):\(port)")! }
    var wsURL: URL { URL(string: "wss://\(host):\(port)/ws/phone")! }
}

/// Parsed `sightline://pair?h=...&p=...&c=...&f=...` link (from the system Camera scanning the QR).
struct PairLink: Equatable, Identifiable {
    let host: String
    let port: Int
    let code: String
    let fingerprint: String
    var id: String { code }

    init?(url: URL) {
        guard url.scheme == "sightline", url.host == "pair",
              let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems else { return nil }
        func q(_ k: String) -> String? { items.first { $0.name == k }?.value }
        guard let h = q("h"), let p = q("p").flatMap(Int.init), let c = q("c"), let f = q("f")?.lowercased(),
              (1...65535).contains(p), f.count == 64, f.allSatisfy(\.isHexDigit), c.count >= 8, c.count <= 64,
              h.count <= 253, h.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "." || $0 == "-" || $0 == ":" }) else { return nil }
        host = h
        port = p
        code = c
        fingerprint = f
    }
}

/// Accepts exactly one certificate: the one whose SHA-256 came in the QR code. No CA trust at all,
/// so someone on the venue Wi-Fi can't stand in for the laptop even with a "valid" certificate.
final class PinnedDelegate: NSObject, URLSessionDelegate, URLSessionTaskDelegate {
    let fingerprint: String

    init(fingerprint: String) {
        self.fingerprint = fingerprint
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func handle(_ challenge: URLAuthenticationChallenge, _ done: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let trust = challenge.protectionSpace.serverTrust,
              let chain = SecTrustCopyCertificateChain(trust) as? [SecCertificate], let leaf = chain.first else {
            done(.cancelAuthenticationChallenge, nil)
            return
        }
        let presented = Self.sha256Hex(SecCertificateCopyData(leaf) as Data)
        if presented == fingerprint {
            done(.useCredential, URLCredential(trust: trust))
        } else {
            done(.cancelAuthenticationChallenge, nil)
        }
    }

    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        handle(challenge, completionHandler)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        handle(challenge, completionHandler)
    }
}

enum PairingError: LocalizedError {
    case server(Int, String)
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .server(403, _): return "That QR code was already used or has expired. Make a new one on the laptop."
        case .server(429, _): return "Too many attempts. Wait a minute and try again."
        case .server(let code, let body): return "Server said \(code): \(body)"
        case .transport(let msg): return msg
        }
    }
}

/// Redeem the one-time code for a device token, over the pinned connection.
func redeem(_ link: PairLink, name: String, completion: @escaping (Result<Pairing, PairingError>) -> Void) {
    let delegate = PinnedDelegate(fingerprint: link.fingerprint)
    let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
    var req = URLRequest(url: URL(string: "https://\(link.host):\(link.port)/api/pair")!)
    req.httpMethod = "POST"
    req.timeoutInterval = 8
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = try? JSONSerialization.data(withJSONObject: ["code": link.code, "name": name, "platform": "ios"])
    session.dataTask(with: req) { data, response, error in
        defer { session.finishTasksAndInvalidate() }
        let result: Result<Pairing, PairingError>
        if let error {
            let e = error as NSError
            result = .failure(.transport(e.code == NSURLErrorCancelled
                ? "The server's certificate didn't match the QR code. Not pairing."
                : "\(e.localizedDescription) [\(e.code)]"))
        } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            result = .failure(.server(http.statusCode, String(data: data ?? Data(), encoding: .utf8)?.prefix(120).description ?? ""))
        } else if let data, let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = j["device_id"] as? String, let token = j["token"] as? String {
            result = .success(Pairing(host: link.host, port: link.port, fingerprint: link.fingerprint, deviceID: id, token: token,
                                      name: (j["name"] as? String) ?? name))
        } else {
            result = .failure(.transport("Unexpected reply from server"))
        }
        DispatchQueue.main.async { completion(result) }
    }.resume()
}
