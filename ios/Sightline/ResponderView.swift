import ARKit
import RealityKit
import SwiftUI

struct ARViewContainer: UIViewRepresentable {
    let arView: ARView
    func makeUIView(context: Context) -> ARView { arView }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

/// The responder's screen: camera + AR, alignment HUD, x-ray card, colleague compass, pin buttons.
struct ResponderView: View {
    let pairing: Pairing
    var unpair: (String) -> Void // ends the session; the reason is shown on the pairing screen

    @StateObject private var net: NetworkClient
    @StateObject private var ar: ARManager
    @StateObject private var voice: VoiceGuide
    @Environment(\.scenePhase) private var phase
    @State private var showSettings = false
    @State private var saidOK = false
    private let beacon = LocatorBeacon.shared
    @State private var showMap = false
    @State private var answeredHazard: String? // pin id answered on this phone (until the laptop clears it)

    private var name: String { pairing.name }

    init(pairing: Pairing, unpair: @escaping (String) -> Void) {
        self.pairing = pairing
        self.unpair = unpair
        let net = NetworkClient()
        _net = StateObject(wrappedValue: net)
        _ar = StateObject(wrappedValue: ARManager(net: net))
        _voice = StateObject(wrappedValue: VoiceGuide(net: net))
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                ARViewContainer(arView: ar.arView).ignoresSafeArea()

                if let target = ar.xrayTarget, let point = ar.xrayScreenPoint {
                    XRayCard(peer: target, net: net)
                        .position(cardPosition(for: point, in: geo.size))
                        .transition(.opacity)
                    Circle()
                        .stroke(Color(hex: target.color), lineWidth: 3)
                        .frame(width: 44, height: 44)
                        .position(point)
                }

                if ar.alignment == .aligned { crosshair }

                VStack(spacing: 10) {
                    topBar
                    alignmentBanner
                    if net.revoked { revokedBanner }
                    ForEach(net.peers.filter { $0.alert != nil && $0.connected }) { peer in colleagueAlert(peer) }
                    Spacer()
                    guideCard
                    navCard
                    planCard
                    HStack(alignment: .bottom) {
                        Text(ar.scanText).font(.caption2).foregroundStyle(.secondary)
                            .padding(6).background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
                        Spacer()
                        VStack(alignment: .trailing, spacing: 6) {
                            guideMenu
                            MinimapView(map: net.map, me: ar.myPos, heading: ar.myHeading, myColor: net.myColor, peers: net.peers, pins: net.pins)
                        }
                    }
                    compass
                    bottomBar
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)

                if let h = net.me?.hazard, h.id != answeredHazard, net.myAlert == nil { hazardAhead(h) }
                if let mine = net.myAlert, !saidOK { areYouOK(mine) }
            }
        }
        .onChange(of: net.peers.filter { $0.alert != nil && !($0.alert?.acked ?? true) }.map(\.id)) { old, new in
            if !Set(new).subtracting(old).isEmpty { UINotificationFeedbackGenerator().notificationOccurred(.warning) }
        }
        .onChange(of: net.me?.hazard?.id) { _, id in
            if id == nil { answeredHazard = nil } else if id != answeredHazard { UINotificationFeedbackGenerator().notificationOccurred(.warning) }
        }
        .onChange(of: net.me?.voice ?? []) { _, cues in voice.receive(cues) }
        .onChange(of: net.myAlert) { _, a in
            if a == nil { saidOK = false } else { UINotificationFeedbackGenerator().notificationOccurred(.error) }
            if a != nil && !saidOK { beacon.start() } else { beacon.stop() }
        }
        .animation(.easeInOut(duration: 0.2), value: ar.xrayTarget?.id)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            net.connect(pairing)
            ar.start()
        }
        .onDisappear {
            beacon.stop()
            voice.reset()
            ar.pause()
            net.disconnect()
        }
        .onChange(of: phase) { _, newPhase in
            if newPhase == .background {
                voice.reset()
                ar.pause() // camera off immediately
                net.disconnect()
            }
        }
        .onChange(of: net.revoked) { _, revoked in
            if revoked { unpair(net.endReason ?? "The laptop ended the session.") }
        }
        .sheet(isPresented: $showSettings) { settings }
        .fullScreenCover(isPresented: $showMap) { MapScreen(net: net) { showMap = false } }
        .preferredColorScheme(.dark)
    }

    /// Centre reticle with the distance to whatever wall/surface it points at.
    private var crosshair: some View {
        VStack(spacing: 4) {
            Image(systemName: "plus").font(.system(size: 26, weight: .light))
            if let d = ar.wallDistance {
                Text(String(format: "%.1f m", d)).font(.caption.monospacedDigit().bold())
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(.black.opacity(0.55), in: Capsule())
            }
        }
        .foregroundStyle(.white)
        .allowsHitTesting(false)
    }

    /// What the commander's planner suggests for me, and how long until I must turn back.
    /// Waypoint guidance from the commander's route: which stop, how far, and an arrow to follow.
    @ViewBuilder private var guideCard: some View {
        if let g = net.me?.guide {
            HStack(spacing: 12) {
                if g.done == true {
                    Image(systemName: "checkmark.circle.fill").font(.title).foregroundStyle(Brand.green)
                    Text("Route done: all \(g.total ?? 0) waypoints reached").font(.callout.bold())
                } else {
                    Image(systemName: "location.north.fill")
                        .font(.system(size: 30, weight: .bold))
                        .foregroundStyle(.cyan)
                        .rotationEffect(.radians(-Double(arrowAngle(g.aim) ?? 0)))
                        .animation(.easeOut(duration: 0.3), value: arrowAngle(g.aim) ?? 0)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Waypoint \(g.next ?? 0) of \(g.total ?? 0)").font(.headline)
                        Text(g.m.map { String(format: "%.0f m along the route", $0) } ?? "").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
            }
            .padding(10)
            .background(Color.cyan.opacity(0.16), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.cyan.opacity(0.5)))
        }
    }

    /// Voice route I asked for (to command / the nearest exit): an arrow along the route and the distance left.
    /// Hidden while the commander's waypoint route is guiding me (that one comes first).
    @ViewBuilder private var navCard: some View {
        if net.me?.guide == nil, let n = net.me?.navInfo {
            HStack(spacing: 12) {
                Image(systemName: "location.north.fill")
                    .font(.system(size: 30, weight: .bold))
                    .foregroundStyle(Brand.green)
                    .rotationEffect(.radians(-Double(arrowAngle(n.aim) ?? 0)))
                    .animation(.easeOut(duration: 0.3), value: arrowAngle(n.aim) ?? 0)
                    .opacity(n.aim == nil ? 0.3 : 1)
                VStack(alignment: .leading, spacing: 2) {
                    Text(n.to == "exit" ? "To the nearest exit" : "To the command post").font(.headline)
                    Text(n.m.map { String(format: "%.0f m along the route", $0) } ?? "No known route yet: keep mapping")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button { setNav("off") } label: { Image(systemName: "xmark.circle.fill").font(.title2) }
                    .accessibilityLabel("Stop guiding")
            }
            .padding(10)
            .background(Brand.green.opacity(0.16), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Brand.green.opacity(0.5)))
        }
    }

    /// Ask the laptop to guide me (by voice and arrow) along its safest route, and choose what the voice says.
    private var guideMenu: some View {
        Menu {
            Section("Guide me to") {
                Button { setNav("home") } label: { Label("Command post", systemImage: "flag.fill") }
                Button { setNav("exit") } label: { Label("Nearest exit", systemImage: "door.left.hand.open") }
                if (net.me?.nav ?? "off") != "off" {
                    Button(role: .destructive) { setNav("off") } label: { Label("Stop guiding", systemImage: "xmark") }
                }
            }
            Section("Voice") {
                Picker("Voice", selection: $voice.mode) {
                    ForEach(VoiceGuide.Mode.allCases) { Text($0.title).tag($0) }
                }
            }
        } label: {
            Label("Guide", systemImage: voice.mode == .off ? "speaker.slash.fill" : "speaker.wave.2.fill")
                .font(.caption.bold())
                .padding(.horizontal, 10).padding(.vertical, 7)
                .background(.black.opacity(0.55), in: Capsule())
                .foregroundStyle(.white)
        }
        .disabled(!net.connected)
        .opacity(net.connected ? 1 : 0.4)
        .accessibilityHint("Voice and arrow guidance to the command post or the nearest exit, and voice settings")
    }

    private func setNav(_ to: String) {
        net.send(["t": "nav", "to": to])
    }

    /// Turn needed from where I'm facing to the next point on the route (radians; positive = to my left).
    private func arrowAngle(_ aim: [Float]?) -> Float? {
        guard let aim, aim.count == 2, let pos = ar.myPos, let h = ar.myHeading else { return nil }
        let bearing = atan2(aim[0] - pos.x, aim[1] - pos.z)
        var d = bearing - h
        while d > .pi { d -= 2 * .pi }
        while d < -.pi { d += 2 * .pi }
        return d
    }

    @ViewBuilder private var planCard: some View {
        if let me = net.me, me.suggestion != nil || me.turnaroundS != nil {
            VStack(alignment: .leading, spacing: 4) {
                if let s = me.suggestion { Label(s, systemImage: "arrow.triangle.turn.up.right.circle.fill").font(.callout.bold()) }
                HStack(spacing: 12) {
                    if let t = me.turnaroundS {
                        Text(t <= 0 ? "TURN BACK NOW" : "Turn back in \(Int(t) / 60):\(String(format: "%02d", Int(t) % 60))")
                            .foregroundStyle(t <= 120 ? .red : .primary).bold()
                    }
                    if let m = me.exitM { Text(String(format: "Exit %.0f m", m)).foregroundStyle(.secondary) }
                }
                .font(.caption)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private func colleagueAlert(_ peer: Peer) -> some View {
        let a = peer.alert!
        var detail = peer.distance.map { String(format: "%.1f m away", $0) } ?? ""
        if let w = peer.wallsBetween, w > 0 { detail += " · \(w) wall\(w > 1 ? "s" : "")" }
        return HStack(spacing: 10) {
            Image(systemName: a.kind == "fall" ? "figure.fall" : "exclamationmark.triangle.fill").font(.title2)
            VStack(alignment: .leading, spacing: 2) {
                Text("\(peer.name): \(a.title)").bold()
                Text("\(Int(a.for)) s · \(detail)").font(.caption)
            }
            Spacer()
        }
        .padding(12)
        .foregroundStyle(.white)
        .background(Brand.red.opacity(a.acked ? 0.55 : 0.9), in: RoundedRectangle(cornerRadius: 12))
    }

    /// Walking towards a hazard pin: keep away, or go on (the commander is told).
    private func hazardAhead(_ h: HazardWarn) -> some View {
        VStack {
            Spacer()
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill").font(.title)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Hazard ahead").font(.title3.bold())
                        Text("\(h.label.isEmpty ? "Marked hazard" : h.label.capitalized) · \(String(format: "%.1f", h.dist)) m").font(.callout)
                    }
                }
                Text("You're heading towards a hazard marked by your team. Keep away if you can.").font(.callout).opacity(0.9)
                HStack(spacing: 10) {
                    Button {
                        answeredHazard = h.id
                        net.send(["t": "hazard", "id": h.id, "action": "avoid"])
                    } label: { Text("Avoid").bold().frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).tint(Brand.green).controlSize(.large)
                    Button {
                        answeredHazard = h.id
                        net.send(["t": "hazard", "id": h.id, "action": "proceed"])
                    } label: { Text("Proceed anyway").frame(maxWidth: .infinity) }
                    .buttonStyle(.bordered).tint(.white).controlSize(.large)
                }
                Text("Proceeding alerts the commander.").font(.caption).opacity(0.8)
            }
            .foregroundStyle(.white)
            .padding(18)
            .background(Brand.red.opacity(0.95), in: RoundedRectangle(cornerRadius: 18))
            .padding(.horizontal, 14)
            .padding(.bottom, 120)
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    /// Full-screen check when the server thinks this responder fell or stopped moving.
    private func areYouOK(_ a: AlertInfo) -> some View {
        VStack(spacing: 24) {
            Image(systemName: a.kind == "fall" ? "figure.fall" : "exclamationmark.triangle.fill").font(.system(size: 64))
            Text("Are you OK?").font(.largeTitle.bold())
            Text(a.kind == "fall" ? "It looks like you may have fallen." : "You haven't moved for a while.")
            Text("The commander has been alerted.").font(.callout).opacity(0.8)
            Label("Your phone is sounding a locator alarm so the team can find you.", systemImage: "speaker.wave.3.fill")
                .font(.callout).opacity(0.9).multilineTextAlignment(.center).padding(.horizontal, 30)
            Button {
                saidOK = true
                beacon.stop()
                net.send(["t": "ok"])
            } label: {
                Text("I'm OK").font(.title2.bold()).frame(maxWidth: .infinity).padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent).tint(.white).foregroundStyle(Brand.red).controlSize(.large)
            .padding(.horizontal, 40)
        }
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Brand.red.opacity(0.94))
        .ignoresSafeArea()
    }

    private var revokedBanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("This phone was removed by the commander.").bold()
            Button("Pair again") { unpair(net.endReason ?? "The laptop ended the session.") }.buttonStyle(.borderedProminent).tint(Brand.red)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var settings: some View {
        NavigationStack {
            Form {
                Section("Paired server") {
                    LabeledContent("Name", value: pairing.name)
                    LabeledContent("Server", value: "\(pairing.host):\(pairing.port)")
                    LabeledContent("Certificate", value: String(pairing.fingerprint.prefix(16)) + "…").font(.caption.monospaced())
                    LabeledContent("Device", value: pairing.deviceID)
                }
                Section {
                    Toggle("AI 3D scanning", isOn: $ar.scanning)
                } footer: {
                    Text("Runs a depth model on the phone to map furniture and plain walls. Uses more battery; pauses automatically if the phone gets hot.")
                }
                Section {
                    Button("Disconnect", role: .destructive) {
                        showSettings = false
                        net.disconnect()
                        unpair("You disconnected this phone.")
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showSettings = false } } }
        }
    }

    /// Keep the 200x290 card on screen, above the target if there is room.
    private func cardPosition(for p: CGPoint, in size: CGSize) -> CGPoint {
        let w: CGFloat = 200, h: CGFloat = 290
        let x = min(max(p.x, w / 2 + 10), size.width - w / 2 - 10)
        let above = p.y - h / 2 - 34
        let y = above > h / 2 + 60 ? above : min(p.y + h / 2 + 34, size.height - h / 2 - 140)
        return CGPoint(x: x, y: y)
    }

    private var topBar: some View {
        VStack(alignment: .leading, spacing: 6) {
        HStack {
            FMTLogo(size: 11)
            Label(net.connected ? "Live" : "Offline", systemImage: net.connected ? "dot.radiowaves.left.and.right" : "wifi.slash")
                .foregroundStyle(net.connected ? Brand.green : Brand.red)
            Spacer()
            Text(name).bold()
            Circle().fill(Color(hex: net.myColor)).frame(width: 10, height: 10)
            Spacer()
            Text("\(ar.planeCount) surfaces · \(ar.trackingText)")
                .foregroundStyle(.secondary)
        }
        if !ar.calibText.isEmpty { Text(ar.calibText).font(.caption2).foregroundStyle(.secondary) }
        if !net.connected {
            Text("wss://\(pairing.host):\(pairing.port) · pinned").font(.caption2.monospaced())
            Text(net.lastError ?? "connecting…").font(.caption2).foregroundStyle(.orange)
        }
        }
        .font(.caption)
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder private var alignmentBanner: some View {
        switch ar.alignment {
        case .searching:
            banner("viewfinder", "Point at the entrance marker", ar.markerHint ?? "Hold steady ~50 cm away, face-on, until it locks", .orange)
        case .locking(let progress):
            VStack(spacing: 6) {
                banner("scope", "Locking on marker…", ar.markerHint ?? "Keep it in view", .yellow)
                ProgressView(value: progress).tint(.yellow)
            }
        case .aligned:
            EmptyView()
        }
    }

    private func banner(_ icon: String, _ title: String, _ subtitle: String, _ color: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon).font(.title2).foregroundStyle(color)
            VStack(alignment: .leading) {
                Text(title).bold()
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    /// One arrow per colleague, pointing where they are relative to where the phone faces.
    private var compass: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(net.peers.filter { $0.connected && !$0.stale && ar.bearings[$0.id] != nil }) { peer in
                    HStack(spacing: 6) {
                        Image(systemName: "location.north.fill")
                            .rotationEffect(.radians(Double(ar.bearings[peer.id] ?? 0)))
                            .foregroundStyle(Color(hex: peer.color))
                        VStack(alignment: .leading, spacing: 0) {
                            Text(peer.name).font(.caption).bold()
                            Text(peerDetail(peer)).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
                }
            }
        }
    }

    private func peerDetail(_ p: Peer) -> String {
        var s = p.distance.map { String(format: "%.1f m", $0) } ?? "–"
        if let w = p.wallsBetween, w > 0 { s += " · \(w) wall\(w > 1 ? "s" : "")" }
        return s
    }

    /// One dock of equal tiles: an icon with its name underneath, so every button says what it does.
    /// Toggles (3D, X-ray) fill with colour while on.
    private var bottomBar: some View {
        HStack(spacing: 6) {
            Button { ar.realign() } label: { ToolTile(title: "Realign", icon: "scope") }
                .accessibilityHint("Line up on the entrance marker again")
            Button { ar.scanning.toggle() } label: {
                ToolTile(title: ar.scanning ? "3D on" : "3D off", icon: ar.scanning ? "cube.transparent.fill" : "cube.transparent",
                         tint: .teal, active: ar.scanning)
            }
            .accessibilityHint("Turns the 3D room scan on or off")
            Button { ar.xrayEnabled.toggle() } label: {
                ToolTile(title: "X-ray", icon: ar.xrayEnabled ? "eye.fill" : "eye.slash", tint: .purple, active: ar.xrayEnabled)
            }
            .accessibilityHint("See colleagues through walls")
            Button { ar.dropPin(kind: "hazard") } label: {
                ToolTile(title: "Hazard", icon: "exclamationmark.triangle.fill", tint: Brand.red, active: true)
            }
            .disabled(ar.alignment != .aligned)
            .opacity(ar.alignment == .aligned ? 1 : 0.4)
            .accessibilityHint("Marks a hazard where you are pointing; anyone walking towards it is warned")
            Menu {
                Button("Victim") { ar.dropPin(kind: "victim") }
                Button("Note") { ar.dropPin(kind: "note") }
            } label: { ToolTile(title: "Pin", icon: "mappin.and.ellipse") }
                .disabled(ar.alignment != .aligned)
                .opacity(ar.alignment == .aligned ? 1 : 0.4)
                .accessibilityHint("Mark a victim or a note where you are pointing")
            Button { showMap = true } label: { ToolTile(title: "Map", icon: "map.fill") }
                .disabled(!net.connected)
                .opacity(net.connected ? 1 : 0.4)
            Button { showSettings = true } label: { ToolTile(title: "Settings", icon: "gearshape.fill") }
        }
        .buttonStyle(.plain)
        .padding(6)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }
}

/// A dock button: icon on top, its name below (always one line).
private struct ToolTile: View {
    let title: String
    let icon: String
    var tint: Color = .white
    var active = false

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 20, weight: .semibold))
                .frame(height: 24)
            Text(title)
                .font(.caption2.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, minHeight: 58)
        .background(active ? tint.opacity(0.85) : Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
        .contentShape(RoundedRectangle(cornerRadius: 12))
    }
}

/// The colleague's live camera, anchored over where they are standing.
struct XRayCard: View {
    let peer: Peer
    let net: NetworkClient
    @StateObject private var loader = FrameLoader()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "eye.fill")
                Text(peer.name).bold()
                Spacer()
                if let d = peer.distance { Text(String(format: "%.1f m", d)) }
            }
            .font(.caption)
            .padding(8)
            .background(Color(hex: peer.color).opacity(0.85))

            ZStack {
                Color.black
                if let image = loader.image {
                    Image(uiImage: image).resizable().scaledToFill()
                } else {
                    ProgressView()
                }
            }
            .frame(width: 200, height: 220)
            .clipped()

            Text(wallText)
                .font(.caption2.bold())
                .frame(maxWidth: .infinity)
                .padding(6)
                .background(.black.opacity(0.8))
        }
        .frame(width: 200)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: peer.color), lineWidth: 2))
        .shadow(radius: 10)
        .onAppear { loader.start(peer.id, net) }
        .onChange(of: peer.id) { _, id in loader.start(id, net) }
        .onDisappear { loader.stop() }
    }

    private var wallText: String {
        guard let w = peer.wallsBetween else { return "LIVE VIEW" }
        return w == 0 ? "IN LINE OF SIGHT" : "THROUGH \(w) WALL\(w > 1 ? "S" : "")"
    }
}

/// Polls the colleague's latest frame about 5 times a second over the pinned, authenticated session.
final class FrameLoader: ObservableObject {
    @Published var image: UIImage?
    private var timer: Timer?
    private var inFlight = false

    func start(_ peerID: String, _ net: NetworkClient) {
        stop()
        image = nil
        timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self, weak net] _ in
            guard let self, let net, !self.inFlight else { return }
            self.inFlight = true
            net.fetchFrame(of: peerID) { data in
                self.inFlight = false
                if let data, let img = UIImage(data: data) { self.image = img }
            }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }
}

extension Color {
    init(hex: String) {
        let (r, g, b) = hex.rgb
        self.init(red: r, green: g, blue: b)
    }
}

/// FindMyTeam brand: black, signal red, safety green.
enum Brand {
    static let red = Color(red: 0.89, green: 0.13, blue: 0.17)
    static let green = Color(red: 0.13, green: 0.80, blue: 0.47)
    static let ink = Color(red: 0.035, green: 0.035, blue: 0.04)
}

/// The FMT mark: white letters on a red tile with a green "found" dot.
struct FMTLogo: View {
    var size: CGFloat = 18

    var body: some View {
        Text("FMT")
            .font(.system(size: size, weight: .black, design: .rounded))
            .tracking(size * 0.06)
            .foregroundStyle(.white)
            .padding(.horizontal, size * 0.45)
            .padding(.vertical, size * 0.18)
            .background(Brand.red, in: RoundedRectangle(cornerRadius: size * 0.32, style: .continuous))
            .overlay(alignment: .topTrailing) {
                Circle().fill(Brand.green).frame(width: size * 0.42, height: size * 0.42)
                    .overlay(Circle().stroke(Brand.ink, lineWidth: size * 0.08))
                    .offset(x: size * 0.16, y: -size * 0.16)
            }
            .accessibilityLabel("FindMyTeam")
    }
}
