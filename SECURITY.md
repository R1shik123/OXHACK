# Sightline security model

## Threat model

Sightline runs on venue or emergency-scene networks you don't control. We assume anyone on that network can:
- sniff traffic
- run their own fake "server"
- try to connect as a fake phone
- try to overload the server

Camera feeds and responder positions are sensitive.

## Controls

| Risk | Control | Where |
|---|---|---|
| Eavesdropping | All phone traffic is TLS (WSS/HTTPS); TLS 1.3 in practice. No cleartext: iOS has no arbitrary-loads ATS exception; Android has `cleartextTrafficPermitted=false` | `server/sightline/__main__.py`, `ios/project.yml`, `android/.../network_security_config.xml` |
| Fake server / man-in-the-middle | **Certificate pinning:** the phone accepts only the certificate whose SHA-256 came in the pairing QR code. No certificate authority is trusted | `PinnedDelegate` (iOS), `Pinning.client` (Android), `check_pin` (sim) |
| Fake phones | Joining needs a **single-use, 10-minute pairing code** from the commander page. The server issues a 256-bit device token and keeps only its **SHA-256**, in memory | `server/sightline/security.py` |
| Lingering links after the app closes | **Session-only pairing.** Leaving the app (iOS: background; Android: `onPause`) stops the camera, closes the socket and drops the token from memory. The server discards the device the moment its socket closes, and a pairing that never connects expires after 60 s. Nothing about the laptop or the session is stored on the phone, and nothing about phones is stored on the laptop | `SightlineApp.swift`, `MainActivity.kt`, `app.py` `ws_phone` |
| Host laptop reading the phone | The server has no command it can send to a phone; phones only push pose, planes, points and small camera frames while the app is open. The apps request no storage, photos, contacts or location permissions | apps + protocol |
| Spoofing another phone | The server, not the phone, assigns device IDs; every socket and frame request is authenticated with the bearer token | `app.py` `ws_phone`, `phone_frame` |
| Lost or stolen phone | **Revoke** in the commander page: the socket closes with code 4403, the token is deleted, and the phone returns to pairing. Since pairings are session-only, a phone that's turned off or closed is already unpaired | `admin_revoke` |
| Commander page attacked over the network | The commander listener binds to **127.0.0.1** only, needs the **admin token** printed at startup (exchanged for an HttpOnly, SameSite=Strict cookie), and mutating calls need an `X-Sightline` header (CSRF). The 3D view's WebSocket checks the Origin | `admin_app` |
| Malformed or malicious data | Strict pydantic schemas with bounded strings, finite coordinates within 500 m, at most 3,000 planes per device and 200 pins; binary type and size checks (JPEG ≤ 300 KB, ≤ 12,000 points per message) | `server/sightline/protocol.py` |
| Flooding | Per-device token-bucket rate limits on text, frames and points; pairing attempts limited per IP; WebSocket message size caps (512 KB from phones, 64 KB for the commander) | `RateLimiter`, uvicorn `ws_max_size` |
| Secrets at rest | **None on the phones**: tokens live in memory for one app session (iOS keeps only your display name). Server: only the TLS key and certificate in `server/data/` (`0600` files, `0700` folder). Device tokens, the map, camera frames and points are **memory only** | `Pairing.swift`, `Pairing.kt`, `security.py` |
| Web injection | Strict CSP (`default-src 'self'`, no inline scripts or styles), `X-Frame-Options: DENY`, `nosniff`, no-referrer; three.js is served locally, not from a CDN; names are inserted as text, never HTML | `app.py` `SECURITY_HEADERS`, `web/` |
| Malicious pairing links | **No deep links at all**: both apps pair only by scanning the QR code with their built-in scanner (VisionKit on iOS; ZXing on Android, offline with no Google services). The payload is validated and **confirmed by the user**, who sees the server address and certificate fingerprint before anything is sent | `QRScanner`, `QrScanner.kt` |
| Info leaks | No server header and no `/docs` or OpenAPI on either listener; minimal health endpoint; logs never contain tokens | `app.py` |
| Android APK tampering | The download needs a live pairing code; the commander page shows the APK's SHA-256 to compare. The release build is signed with your own RSA-3072 key (v3 scheme), and at startup the app checks its own signing certificate against the fingerprint compiled in, refusing to run if it was re-signed | `download_apk`, `Integrity.kt` |
| Android runtime | Refuses to run if debuggable or rooted (su binaries, Magisk, test-keys). `FLAG_SECURE` (no screenshots, screen recording or recent-apps preview); `excludeFromRecents`; touches ignored while another app draws on top (tap-jacking). Only the launcher screen is exported; the AndroidX profile receiver and start-up initializers are stripped. `allowBackup=false`, data-extraction rules exclude everything. TLS 1.3 only (OkHttp `RESTRICTED_TLS`). R8 shrinking and obfuscation, and all `android.util.Log` calls removed from the release build. ARCore Cloud Anchors disabled | `android/` |
| iOS runtime | The camera view is blacked out in the app-switcher snapshot; no ATS exceptions for arbitrary loads; no URL schemes; no background modes | `ios/` |

## Things outside the app's control

- **Developer install links.** Installing from Xcode means the iPhone trusts this Mac. Android USB debugging gives the laptop a shell on the phone. These are OS-level links, not part of Sightline. To avoid them:
  - **Android:** install the APK by scanning the download QR. Never enable USB debugging, or turn it off after use.
  - **iPhone:** don't enable Xcode's "Connect via network". After installing, you can remove this Mac's trust with Settings → General → Transfer or Reset iPhone → Reset → **Reset Location & Privacy** (you'll need to trust it again to reinstall after 7 days).
- **Google Play Services for AR.** ARCore requires it, and it's a separate Google app with its own data practices. Sightline disables Cloud Anchors and sends nothing to Google itself.

## Known gaps (hackathon scope)

- **Admin token:** a new one is generated at each start unless `SIGHTLINE_ADMIN_TOKEN` is set. Anyone with a shell on the laptop can read it from the terminal.
- **Admin cookie across ports:** browsers scope cookies to the host, not the port, so another web service on `127.0.0.1` that you visit would also receive the admin cookie. It's HttpOnly and SameSite=Strict, and the token rotates each restart. Don't run untrusted local web services alongside the commander page.
- **Root detection is best-effort:** a determined attacker with root can hide it. Hardware attestation (Play Integrity / App Attest) would need Google or Apple servers, which conflicts with "no outside links".
- **Brief Wi-Fi drops end the session:** with session-only pairing there's no automatic reconnect; re-scan the QR.
- **Camera frames** are visible to every paired phone, which is how x-ray works. There are no per-role permissions yet.
- **No audit log** of who paired or revoked when, apart from the server console.
