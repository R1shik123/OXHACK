import Foundation

/// A colleague as the server reports it (positions are in the shared frame).
struct Peer: Decodable, Identifiable, Equatable {
    let id: String
    let name: String
    let color: String
    let connected: Bool
    let aligned: Bool
    let pos: [Float]?
    let fwd: [Float]?
    let stale: Bool
    let hasFrame: Bool
    let wallsBetween: Int?
    let distance: Float?
    let wallDist: Float?
    let platform: String?
    let alert: AlertInfo?
}

/// Fall / no-movement alert raised by the server from a responder's motion.
struct AlertInfo: Decodable, Equatable {
    let kind: String // "fall" | "still"
    let `for`: Float
    let acked: Bool

    var title: String { kind == "fall" ? "possible fall" : "not moving" }
}

struct Pin: Decodable, Identifiable, Equatable {
    let id: String
    let kind: String
    let label: String
    let pos: [Float]
    let by: String
    let auto: Bool?
    let conf: Float?
}

struct PlaneMsg: Decodable {
    let id: String
    let kind: String
    let corners: [[Float]]
}

struct ServerMessage: Decodable {
    let t: String
    let peers: [Peer]?
    let pins: [Pin]?
    let color: String?
    let upsert: [PlaneMsg]?
    let remove: [String]?
    let reset: Bool?
    let me: MeState?
    let markerM: Double?
}

struct MeState: Decodable, Equatable {
    let alert: AlertInfo?
    let suggestion: String?
    let turnaroundS: Float?
    let exitM: Float?
    let exitS: Float?
    let hazard: HazardWarn?
    let guide: GuideInfo?
    let voice: [VoiceCue]?
    let nav: String? // voice route I asked for: "off" | "home" | "exit"
    let navInfo: NavInfo?
}

/// Something the laptop wants this phone to say. `id` = ElevenLabs audio on the laptop; local = say it on the phone.
struct VoiceCue: Decodable, Equatable {
    let n: Int
    let text: String
    let prio: Int // 0 alert, 1 trip hazard, 2 navigation, 3 doors
    let kind: String // "alert" | "hazard" | "nav" | "door"
    let local: Bool
    let ts: Double
    let id: String?
}

/// The route I asked to be guided along (to command / the nearest exit): distance left and where to head.
struct NavInfo: Decodable, Equatable {
    let to: String
    let m: Float?
    let aim: [Float]?
}

/// The commander's waypoint route, when it's assigned to me: next stop, distance along the route, where to head.
struct GuideInfo: Decodable, Equatable {
    let next: Int?
    let total: Int?
    let m: Float?
    let aim: [Float]? // (x, z) a couple of metres ahead along the route, in the shared frame
    let done: Bool?
}

/// "Hazard ahead": I'm heading towards a hazard pin.
struct HazardWarn: Decodable, Equatable {
    let id: String
    let label: String
    let dist: Float
}

extension String {
    /// "#ff5a5f" -> (r, g, b) in 0...1
    var rgb: (Double, Double, Double) {
        let hex = trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        guard hex.count == 6, let v = UInt32(hex, radix: 16) else { return (1, 1, 1) }
        return (Double((v >> 16) & 0xff) / 255, Double((v >> 8) & 0xff) / 255, Double(v & 0xff) / 255)
    }
}
