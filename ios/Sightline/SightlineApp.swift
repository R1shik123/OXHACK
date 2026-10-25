import SwiftUI
import VisionKit

/// Session-only by design: the pairing lives in memory. When the app leaves the screen the camera
/// stops, the connection closes, the server discards the token, and nothing about it is stored here.
@main
struct SightlineApp: App {
    @State private var pairing: Pairing?
    @State private var pendingLink: PairLink?
    @State private var endReason: String? // why the last session ended (memory only)
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            ZStack {
                Group {
                    if let pairing {
                        ResponderView(pairing: pairing) { reason in
                            endReason = reason
                            self.pairing = nil
                        }
                        .id(pairing.deviceID)
                    } else {
                        PairingView(endReason: endReason) { url in pendingLink = PairLink(url: url) }
                    }
                }
                // Hide the camera view from the app-switcher snapshot.
                if phase != .active { Color.black.ignoresSafeArea() }
            }
            .sheet(item: $pendingLink) { link in
                ConfirmPairingView(link: link, replacing: pairing) { newPairing in
                    pairing = newPairing
                    endReason = nil
                    pendingLink = nil
                } cancel: { pendingLink = nil }
            }
            .onChange(of: phase) { _, newPhase in
                if newPhase == .background {
                    pendingLink = nil
                    if pairing != nil {
                        endReason = "FindMyTeam left the screen, so the session ended for security: camera off, connection closed, pairing forgotten."
                    }
                    pairing = nil // ResponderView disappears: AR session paused, socket closed
                }
            }
            .preferredColorScheme(.dark)
            .tint(Brand.red)
        }
    }
}

struct PairingView: View {
    var endReason: String?
    var onScan: (URL) -> Void
    @State private var scanning = false
    @State private var badCode = false

    var body: some View {
        VStack(spacing: 22) {
            if let endReason {
                VStack(alignment: .leading, spacing: 6) {
                    Label("Session ended", systemImage: "info.circle.fill").font(.headline)
                    Text(endReason).font(.callout)
                    Text("Scan a new QR code on the laptop to rejoin.").font(.callout).foregroundStyle(.secondary)
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.18), in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.orange.opacity(0.6)))
            }
            VStack(spacing: 10) {
                FMTLogo(size: 34)
                Text("FindMyTeam").font(.system(size: 30, weight: .heavy, design: .rounded))
                Text("Every responder, on one live map.").font(.subheadline).foregroundStyle(.secondary)
            }
            .padding(.bottom, 6)
            Text("Pair with the commander").font(.title3.bold())
            VStack(alignment: .leading, spacing: 12) {
                Label("On the laptop, open the FindMyTeam commander page and tap **+ Add phone**.", systemImage: "1.circle.fill")
                Label("Tap **Scan QR code** below and point at the code.", systemImage: "2.circle.fill")
                Label("Check the server address, enter your name, tap **Pair**.", systemImage: "3.circle.fill")
            }
            .font(.callout)
            Button {
                scanning = true
            } label: {
                Label("Scan QR code", systemImage: "qrcode.viewfinder").font(.headline).frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).tint(Brand.red).controlSize(.large)
            .disabled(!DataScannerViewController.isSupported)
            if badCode { Text("That isn't a FindMyTeam pairing code.").foregroundStyle(.orange).font(.footnote) }
            Text("Codes work once and expire after 10 minutes. The connection is encrypted and locked to that laptop's certificate, and it ends the moment you leave the app.")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .padding(28)
        .fullScreenCover(isPresented: $scanning) {
            ZStack(alignment: .topTrailing) {
                QRScanner { text in
                    scanning = false
                    if let url = URL(string: text), PairLink(url: url) != nil {
                        badCode = false
                        onScan(url)
                    } else {
                        badCode = true
                    }
                }
                .ignoresSafeArea()
                Button("Cancel") { scanning = false }
                    .buttonStyle(.borderedProminent).tint(.black.opacity(0.6)).padding()
            }
        }
    }
}

/// VisionKit QR scanner; reports the first QR payload it reads.
struct QRScanner: UIViewControllerRepresentable {
    var onCode: (String) -> Void

    func makeUIViewController(context: Context) -> DataScannerViewController {
        let vc = DataScannerViewController(recognizedDataTypes: [.barcode(symbologies: [.qr])], qualityLevel: .accurate,
                                           recognizesMultipleItems: false, isHighFrameRateTrackingEnabled: false, isHighlightingEnabled: true)
        vc.delegate = context.coordinator
        try? vc.startScanning()
        return vc
    }

    func updateUIViewController(_ vc: DataScannerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onCode) }

    final class Coordinator: NSObject, DataScannerViewControllerDelegate {
        let onCode: (String) -> Void
        var done = false
        init(_ onCode: @escaping (String) -> Void) { self.onCode = onCode }

        func dataScanner(_ scanner: DataScannerViewController, didAdd items: [RecognizedItem], allItems: [RecognizedItem]) {
            for case .barcode(let code) in items {
                guard !done, let text = code.payloadStringValue else { continue }
                done = true
                scanner.stopScanning()
                onCode(text)
            }
        }
    }
}

struct ConfirmPairingView: View {
    let link: PairLink
    let replacing: Pairing?
    var done: (Pairing) -> Void
    var cancel: () -> Void
    @State private var name = UserDefaults.standard.string(forKey: "responderName") ?? ""
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Address", value: "\(link.host):\(link.port)")
                    LabeledContent("Certificate", value: String(link.fingerprint.prefix(16)) + "…")
                        .font(.caption.monospaced())
                } header: {
                    Text("Server")
                } footer: {
                    Text("Only pair if you just scanned the code on your own commander's laptop.")
                }
                if replacing != nil {
                    Section { Text("This replaces the current pairing.").foregroundStyle(.orange) }
                }
                Section("Your name") {
                    TextField("e.g. Alice", text: $name).textInputAutocapitalization(.words)
                }
                if let error {
                    Section { Text(error).foregroundStyle(.red) }
                }
                Button(busy ? "Pairing…" : "Pair") { pair() }
                    .disabled(busy || name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .navigationTitle("Pair this phone")
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: cancel) } }
        }
    }

    private func pair() {
        busy = true
        error = nil
        let trimmed = String(name.trimmingCharacters(in: .whitespaces).prefix(32))
        UserDefaults.standard.set(trimmed, forKey: "responderName")
        redeem(link, name: trimmed) { result in
            busy = false
            switch result {
            case .success(let p): done(p)
            case .failure(let e): error = e.localizedDescription
            }
        }
    }
}
