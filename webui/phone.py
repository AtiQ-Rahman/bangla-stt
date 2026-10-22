"""Using the app from a phone on the same Wi-Fi: LAN address, password, HTTPS certificate.

Phone browsers only allow the microphone on HTTPS pages, so phone mode serves HTTPS with a
self-signed certificate made on first use (the phone warns about it once).
"""
import secrets
import socket
import subprocess
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parent.parent / "certs"
USER = "stt"


def lan_ip() -> str:
    """This PC's address on the local network (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.168.0.1", 9))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def certificate(ip: str) -> tuple[Path, Path]:
    """(cert, key) for this address, made with openssl if missing or for another address."""
    CERT_DIR.mkdir(exist_ok=True)
    cert, key, made_for = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "ip.txt"
    if not (cert.exists() and key.exists() and made_for.exists() and made_for.read_text() == ip):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
                        "-keyout", str(key), "-out", str(cert), "-subj", "/CN=Bangla STT",
                        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
                       check=True, capture_output=True)
        key.chmod(0o600)
        made_for.write_text(ip)
    return cert, key


def new_password() -> str:
    return secrets.token_urlsafe(6)
