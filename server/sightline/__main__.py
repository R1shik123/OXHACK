"""Run both listeners:  python -m sightline   (from the server/ folder)"""

import argparse
import asyncio
import logging

import uvicorn

from . import app as A
from .security import ensure_certificate

ap = argparse.ArgumentParser(description="Sightline server")
ap.add_argument("--phone-port", type=int, default=A.PHONE_PORT)
ap.add_argument("--admin-port", type=int, default=A.ADMIN_PORT)
ap.add_argument("--screen-port", type=int, default=A.SCREEN_PORT, help="commander page for other computers (TLS, one-time links)")
args = ap.parse_args()
A.PHONE_PORT, A.ADMIN_PORT, A.SCREEN_PORT = args.phone_port, args.admin_port, args.screen_port
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

ip = A.lan_ip()
cert, key, A.CERT_FP = ensure_certificate([ip, "127.0.0.1"])
A.ALLOWED_ORIGINS = {f"http://127.0.0.1:{A.ADMIN_PORT}", f"http://localhost:{A.ADMIN_PORT}", f"https://{ip}:{A.SCREEN_PORT}"}


async def main() -> None:
    admin = uvicorn.Server(uvicorn.Config(A.admin_app, host="127.0.0.1", port=A.ADMIN_PORT, log_level="warning", ws_max_size=1 << 16))
    phone = uvicorn.Server(uvicorn.Config(
        A.phone_app, host="0.0.0.0", port=A.PHONE_PORT, log_level="warning",
        ssl_certfile=str(cert), ssl_keyfile=str(key), ws_max_size=512 * 1024, ws_ping_interval=10, ws_ping_timeout=20,
        server_header=False,
    ))
    # The same commander app, encrypted, for other computers on the Wi-Fi. Nothing opens without a
    # one-time link made on this Mac ("+ Add a computer"), so listening here exposes no data.
    screens = uvicorn.Server(uvicorn.Config(
        A.admin_app, host="0.0.0.0", port=A.SCREEN_PORT, log_level="warning", ws_max_size=1 << 16,
        ssl_certfile=str(cert), ssl_keyfile=str(key), server_header=False,
    ))
    print(f"""
  FindMyTeam running
  Commander (this Mac only):  http://127.0.0.1:{A.ADMIN_PORT}/?admin={A.registry.admin_token}
  Phones (TLS):               https://{ip}:{A.PHONE_PORT}   cert SHA-256 {A.CERT_FP[:16]}…
  Other computers (Chrome):   https://{ip}:{A.SCREEN_PORT}   one-time link via "+ Add a computer" on the commander page
  Pair a phone: open the commander page -> "Add phone" -> scan the QR code with the phone camera.
""", flush=True)
    await asyncio.gather(admin.serve(), phone.serve(), screens.serve())


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("FindMyTeam stopped.")
