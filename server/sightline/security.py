"""Security: TLS certificate, admin token, one-time pairing codes, device tokens, rate limits.

Threat model (venue Wi-Fi): anyone on the network may sniff, impersonate the server,
or connect as a fake phone. So:
- Phones talk only HTTPS/WSS to a self-signed certificate whose SHA-256 fingerprint they
  learned from the pairing QR code (certificate pinning, no CA involved).
- A phone joins only with a single-use, short-lived pairing code shown on the commander page;
  it then gets a 256-bit device token. The server keeps only SHA-256(token), in memory.
- Pairings are session-only: a device exists from pairing until its connection closes (or 60 s
  if it never connects). Nothing about phones is written to disk; closing the app ends the link.
- The commander page is served on 127.0.0.1 only and needs the admin token printed at startup.
- Every message is validated and rate limited per device; revoking a device closes its socket.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import ipaddress
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(os.environ.get("SIGHTLINE_DATA", Path(__file__).resolve().parents[1] / "data"))
PAIR_TTL = 600  # seconds a pairing code stays valid
CONNECT_TTL = 60  # seconds a fresh pairing may wait before its first connection


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def ensure_certificate(hosts: list[str]) -> tuple[Path, Path, str]:
    """Create (once) a self-signed EC certificate; return cert path, key path, SHA-256 fingerprint."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_p, key_p = DATA / "cert.pem", DATA / "key.pem"
    if not (cert_p.exists() and key_p.exists()):
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Sightline server")])
        sans: list[x509.GeneralName] = [x509.DNSName("localhost")]
        for h in hosts:
            try:
                sans.append(x509.IPAddress(ipaddress.ip_address(h)))
            except ValueError:
                sans.append(x509.DNSName(h))
        now = dt.datetime.now(dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        _write_private(key_p, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        _write_private(cert_p, cert.public_bytes(serialization.Encoding.PEM))
    cert = x509.load_pem_x509_certificate(cert_p.read_bytes())
    fp = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return cert_p, key_p, fp


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class Device:
    id: str
    name: str
    platform: str
    token_hash: str
    created: float = field(default_factory=time.time)
    connected_once: bool = False


class Registry:
    """Live, session-only device pairings (hashed tokens, memory only)."""

    def __init__(self) -> None:
        self.devices: dict[str, Device] = {}
        self.codes: dict[str, float] = {}  # pairing code -> expiry
        self.admin_token = os.environ.get("SIGHTLINE_ADMIN_TOKEN") or secrets.token_urlsafe(24)
        # Other computers showing the commander page (Chrome on a second laptop): one-time codes from
        # this Mac -> their own session token, hashed, memory only, revocable, gone on restart.
        self.screen_codes: dict[str, float] = {}
        self.screens: dict[str, dict] = {}  # token hash -> {"id", "ip", "created"}
        legacy = DATA / "devices.json"  # older builds persisted pairings; remove any leftovers
        if legacy.exists():
            legacy.unlink()

    def expire_unused(self) -> None:
        now = time.time()
        for d in [d for d in self.devices.values() if not d.connected_once and now - d.created > CONNECT_TTL]:
            del self.devices[d.id]

    def new_code(self) -> str:
        now = time.time()
        self.codes = {c: t for c, t in self.codes.items() if t > now}
        code = secrets.token_urlsafe(16)
        self.codes[code] = now + PAIR_TTL
        return code

    def code_valid(self, code: str) -> bool:
        exp = self.codes.get(code)
        return exp is not None and exp > time.time()

    def pair(self, code: str, name: str, platform: str) -> tuple[Device, str] | None:
        """Consume a pairing code (single use) and mint a device token."""
        exp = self.codes.pop(code, None)
        if exp is None or exp < time.time():
            return None
        token = secrets.token_urlsafe(32)
        dev = Device(id="ph-" + secrets.token_hex(4), name=name, platform=platform, token_hash=hash_token(token))
        self.devices[dev.id] = dev
        return dev, token

    def authenticate(self, token: str | None) -> Device | None:
        self.expire_unused()
        if not token:
            return None
        h = hash_token(token)
        for d in self.devices.values():
            if hmac.compare_digest(d.token_hash, h):
                return d
        return None

    def revoke(self, device_id: str) -> bool:
        return self.devices.pop(device_id, None) is not None

    def is_admin(self, token: str | None) -> bool:
        """The Mac's admin token, or the session of a commander screen added from the Mac."""
        if not token:
            return False
        return hmac.compare_digest(token.encode(), self.admin_token.encode()) or self.screen(token) is not None

    def screen(self, token: str | None) -> dict | None:
        if not token:
            return None
        h = hash_token(token)
        return next((s for k, s in self.screens.items() if hmac.compare_digest(k, h)), None)

    def new_screen_code(self) -> str:
        now = time.time()
        self.screen_codes = {c: t for c, t in self.screen_codes.items() if t > now}
        code = secrets.token_urlsafe(16)
        self.screen_codes[code] = now + PAIR_TTL
        return code

    def redeem_screen(self, code: str, ip: str) -> str | None:
        """Consume a one-time screen code; returns that computer's session token."""
        exp = self.screen_codes.pop(code, None)
        if exp is None or exp < time.time():
            return None
        token = secrets.token_urlsafe(32)
        self.screens[hash_token(token)] = {"id": "sc-" + secrets.token_hex(4), "ip": ip, "created": time.time()}
        return token

    def revoke_screen(self, screen_id: str) -> bool:
        for k, s in list(self.screens.items()):
            if s["id"] == screen_id:
                del self.screens[k]
                return True
        return False


class RateLimiter:
    """Token bucket per key: `rate` events/s, bursts up to `burst`."""

    def __init__(self, rate: float, burst: float) -> None:
        self.rate, self.burst = rate, burst
        self.state: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        tokens, last = self.state.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        ok = tokens >= cost
        self.state[key] = (tokens - cost if ok else tokens, now)
        return ok
