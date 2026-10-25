import SwiftUI
import WebKit

/// The commander's map inside the app. The web view never touches the network itself:
///  - page + assets come through a private URL scheme served by `MapBridge` over the pinned session,
///  - live data comes from a read-only WebSocket the app opens (pinned, token-authenticated) and relays.
/// So the Map tab inherits the same certificate pinning, session-only token and teardown as the rest of the app.
final class MapBridge: NSObject, WKURLSchemeHandler, WKScriptMessageHandler {
    static let scheme = "sightline-map"
    private let net: NetworkClient
    weak var webView: WKWebView?
    private var socket: URLSessionWebSocketTask?
    private var live = Set<ObjectIdentifier>() // scheme tasks WebKit hasn't cancelled
    private var closed = false

    init(net: NetworkClient) { self.net = net }

    func close() {
        closed = true
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
    }

    // MARK: page + assets through the pinned session
    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url else { return }
        let id = ObjectIdentifier(task)
        live.insert(id)
        let path = url.path.isEmpty ? "/map" : url.path
        let allowed = path == "/map" || path.hasPrefix("/static/") || path.hasPrefix("/api/frame/")
        guard allowed else { return fail(task, id) }
        net.fetch(path + (url.query.map { "?" + $0 } ?? "")) { [weak self] data, http in
            DispatchQueue.main.async {
                guard let self, self.live.contains(id) else { return }
                guard let data, let http, http.statusCode == 200 else { return self.fail(task, id) }
                var headers = ["Content-Type": http.value(forHTTPHeaderField: "Content-Type") ?? "application/octet-stream",
                               "Cache-Control": "no-store"]
                if let csp = http.value(forHTTPHeaderField: "Content-Security-Policy") { headers["Content-Security-Policy"] = csp }
                let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!
                task.didReceive(resp)
                task.didReceive(data)
                task.didFinish()
                self.live.remove(id)
            }
        }
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {
        live.remove(ObjectIdentifier(task))
    }

    private func fail(_ task: WKURLSchemeTask, _ id: ObjectIdentifier) {
        guard live.contains(id) else { return }
        live.remove(id)
        task.didFailWithError(URLError(.resourceUnavailable))
    }

    // MARK: page <-> live map stream
    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let text = message.body as? String, text.count < 64 * 1024 else { return }
        if text.contains("\"t\":\"ready\"") {
            openSocket()
        } else if let socket {
            socket.send(.string(text)) { _ in } // pins only; the server ignores anything else
        }
    }

    private func openSocket() {
        guard socket == nil, !closed, let t = net.makeViewerTask() else { return }
        socket = t
        t.resume()
        receive(t)
        js("window.__slstatus && window.__slstatus(true)")
    }

    private func receive(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] result in
            DispatchQueue.main.async {
                guard let self, self.socket === t else { return }
                switch result {
                case .failure:
                    self.socket = nil
                    self.js("window.__slstatus && window.__slstatus(false)")
                    if !self.closed { DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { self.openSocket() } }
                case .success(.string(let text)):
                    self.js("window.__sl && window.__sl(\(text))") // JSON is a valid JS expression
                    self.receive(t)
                case .success(.data(let data)):
                    self.js("window.__slb && window.__slb('\(data.base64EncodedString())')")
                    self.receive(t)
                case .success:
                    self.receive(t)
                }
            }
        }
    }

    private func js(_ code: String) {
        webView?.evaluateJavaScript(code, completionHandler: nil)
    }
}

struct MapWebView: UIViewRepresentable {
    let bridge: MapBridge

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(bridge, forURLScheme: MapBridge.scheme)
        config.userContentController.add(bridge, name: "sightline")
        config.websiteDataStore = .nonPersistent() // nothing cached or stored on the phone
        let web = WKWebView(frame: .zero, configuration: config)
        web.isOpaque = false
        web.backgroundColor = .black
        web.scrollView.contentInsetAdjustmentBehavior = .never
        bridge.webView = web
        web.load(URLRequest(url: URL(string: "\(MapBridge.scheme)://app/map")!))
        return web
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    static func dismantleUIView(_ uiView: WKWebView, coordinator: ()) {
        uiView.configuration.userContentController.removeScriptMessageHandler(forName: "sightline")
        uiView.stopLoading()
    }
}

/// Full-screen Map tab with a close button. AR tracking keeps running underneath.
struct MapScreen: View {
    let net: NetworkClient
    var close: () -> Void
    @State private var bridge: MapBridge?

    var body: some View {
        // Header below the status bar, then the map: nothing of the map hides under the clock / battery.
        VStack(spacing: 0) {
            HStack {
                FMTLogo(size: 15)
                Text("Team map").font(.headline)
                Spacer()
                Button { close() } label: {
                    Label("Close", systemImage: "xmark").font(.subheadline.weight(.semibold))
                }
                .buttonStyle(.borderedProminent).tint(Color.white.opacity(0.14))
            }
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(Color.black)
            if let bridge { MapWebView(bridge: bridge).ignoresSafeArea(edges: .bottom) } else { Spacer() }
        }
        .background(Color.black)
        .onAppear { bridge = MapBridge(net: net) }
        .onDisappear { bridge?.close(); bridge = nil }
    }
}
