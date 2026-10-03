"""Offline donor tokens for the localhost easter egg (stdlib only).

Tokens are Ed25519-signed JSON: `KD1.<b64url payload>.<b64url signature>`. The *private* key stays with
the project owner; the app ships only public keys, so a donor cannot mint tokens for others.
Ed25519 is implemented from RFC 8032 section 6 (pure Python, fine for a few verifications) and is
checked against the RFC's test vectors in the test-suite.

This is an easter egg, not DRM: the checks run on the user's machine against open source code, so
anyone who edits the code can bypass them. Do not use it to protect anything that matters.

CLI:
    python -m krystal_bot.donor keygen  <private_key_file>        # prints the public key (hex)
    python -m krystal_bot.donor issue   <private_key_file> <alias> [--days 365] [--tier supporter]
    python -m krystal_bot.donor verify  <token>
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------- Ed25519 (RFC 8032 s.6)
_p = 2 ** 255 - 19
_q = 2 ** 252 + 27742317777372353535851937790883648493


def _sha512(s: bytes) -> bytes:
    return hashlib.sha512(s).digest()


def _inv(x: int) -> int:
    return pow(x, _p - 2, _p)


_d = -121665 * _inv(121666) % _p
_I = pow(2, (_p - 1) // 4, _p)


def _recover_x(y: int, sign: int) -> Optional[int]:
    if y >= _p:
        return None
    x2 = (y * y - 1) * _inv(_d * y * y + 1)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_p + 3) // 8, _p)
    if (x * x - x2) % _p != 0:
        x = x * _I % _p
    if (x * x - x2) % _p != 0:
        return None
    if (x & 1) != sign:
        x = _p - x
    return x


_gy = 4 * _inv(5) % _p
_gx = _recover_x(_gy, 0)
_G = (_gx, _gy, 1, _gx * _gy % _p)


def _add(P: Tuple[int, int, int, int], Q: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    A, B = (P[1] - P[0]) * (Q[1] - Q[0]) % _p, (P[1] + P[0]) * (Q[1] + Q[0]) % _p
    C, D = 2 * P[3] * Q[3] * _d % _p, 2 * P[2] * Q[2] % _p
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _p, G * H % _p, F * G % _p, E * H % _p)


def _mul(s: int, P: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    Q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            Q = _add(Q, P)
        P = _add(P, P)
        s >>= 1
    return Q


def _eq(P: Tuple[int, int, int, int], Q: Tuple[int, int, int, int]) -> bool:
    return (P[0] * Q[2] - Q[0] * P[2]) % _p == 0 and (P[1] * Q[2] - Q[1] * P[2]) % _p == 0


def _compress(P: Tuple[int, int, int, int]) -> bytes:
    zi = _inv(P[2])
    x, y = P[0] * zi % _p, P[1] * zi % _p
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes) -> Optional[Tuple[int, int, int, int]]:
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _p)


def _secret_expand(secret: bytes) -> Tuple[int, bytes]:
    if len(secret) != 32:
        raise ValueError("secret key must be 32 bytes")
    h = _sha512(secret)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(secret: bytes) -> bytes:
    a, _ = _secret_expand(secret)
    return _compress(_mul(a, _G))


def sign(secret: bytes, msg: bytes) -> bytes:
    a, prefix = _secret_expand(secret)
    A = _compress(_mul(a, _G))
    r = int.from_bytes(_sha512(prefix + msg), "little") % _q
    Rs = _compress(_mul(r, _G))
    h = int.from_bytes(_sha512(Rs + A + msg), "little") % _q
    s = (r + h * a) % _q
    return Rs + int.to_bytes(s, 32, "little")


def verify(public: bytes, msg: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    A = _decompress(public)
    if A is None:
        return False
    Rs = signature[:32]
    R = _decompress(Rs)
    if R is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _q:
        return False
    h = int.from_bytes(_sha512(Rs + public + msg), "little") % _q
    return _eq(_mul(s, _G), _add(R, _mul(h, A)))


# ---------------------------------------------------------------------- tokens
def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(secret: bytes, alias: str, days: float = 365.0, tier: str = "supporter", now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    payload = {"v": 1, "sub": alias[:64], "tier": tier[:32], "iat": int(now), "exp": int(now + days * 86400), "nonce": os.urandom(8).hex()}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "KD1." + _b64e(body) + "." + _b64e(sign(secret, b"KD1." + body))


def check_token(token: str, public_keys: List[bytes], now: Optional[float] = None) -> Dict[str, Any]:
    """Never raises: returns {"valid": bool, "reason": str, "claims": {...}}."""
    now = time.time() if now is None else now
    try:
        if len(token) > 2048:
            return {"valid": False, "reason": "token too long"}
        tag, body_b64, sig_b64 = token.split(".")
        if tag != "KD1":
            return {"valid": False, "reason": "unknown token version"}
        body, sig = _b64d(body_b64), _b64d(sig_b64)
        if not any(verify(pk, b"KD1." + body, sig) for pk in public_keys):
            return {"valid": False, "reason": "bad signature or unknown issuer" if public_keys else "no issuer public key configured"}
        claims = json.loads(body.decode("utf-8"))
        if not isinstance(claims, dict) or not isinstance(claims.get("exp"), int):
            return {"valid": False, "reason": "malformed claims"}
        if claims["exp"] < now:
            return {"valid": False, "reason": "expired", "claims": claims}
        return {"valid": True, "reason": "ok", "claims": claims}
    except Exception:  # noqa: BLE001 - any parse failure is simply "not valid"
        return {"valid": False, "reason": "malformed token"}


def load_public_keys(path: str) -> List[bytes]:
    """Public keys live in a JSON list of hex strings, e.g. `config/donor_pubkeys.json`. Missing file = locked."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [bytes.fromhex(h) for h in raw if isinstance(h, str) and len(h) == 64]
    except (OSError, ValueError):
        return []


def _cli(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[0] == "keygen":
        if os.path.exists(argv[1]):
            print("refusing to overwrite existing key file", file=sys.stderr)
            return 2
        sk = os.urandom(32)
        with open(argv[1], "wb") as f:
            f.write(sk)
        print("public key:", public_key(sk).hex())
        print("Keep the private key file OUT of the repository.")
        return 0
    if len(argv) >= 3 and argv[0] == "issue":
        with open(argv[1], "rb") as f:
            sk = f.read(32)
        days = float(argv[argv.index("--days") + 1]) if "--days" in argv else 365.0
        tier = argv[argv.index("--tier") + 1] if "--tier" in argv else "supporter"
        print(issue_token(sk, argv[2], days, tier))
        return 0
    if len(argv) >= 2 and argv[0] == "verify":
        keys = load_public_keys(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "donor_pubkeys.json"))
        print(json.dumps(check_token(argv[1], keys), indent=2))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
