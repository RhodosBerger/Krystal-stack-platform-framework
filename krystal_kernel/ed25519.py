"""
RFC 8032 Compliant Ed25519 Pure-Python Cryptographic Engine
===========================================================
Module: krystal_kernel/ed25519.py
Milestone: V-IDENTITY-02 (Asymmetric Principal Identity & Non-Repudiation)

Zero-dependency implementation of Edwards-curve Digital Signature Algorithm (Ed25519)
over Curve25519 for sovereign cryptographic principal identity and non-repudiation.
"""

from __future__ import annotations

import hashlib
from typing import Tuple

# Curve25519 Constants (RFC 8032)
q = 2**255 - 19
l = 2**252 + 27742317777372353535851937790883648493


def _H(m: bytes) -> bytes:
    """SHA-512 cryptographic hash."""
    return hashlib.sha512(m).digest()


def _expmod(b: int, e: int, m: int) -> int:
    return pow(b, e, m)


def _inv(x: int) -> int:
    return pow(x, q - 2, q)


# Curve parameter d = -121665 * inv(121666) mod q
d = -121665 * _inv(121666) % q
# Square root of -1 in GF(q)
I = pow(2, (q - 1) // 4, q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(d * y * y + 1) % q
    x = pow(xx, (q + 3) // 8, q)
    if (x * x - xx) % q != 0:
        x = (x * I) % q
    if x % 2 != 0:
        x = q - x
    return x


# Base point B
By = 4 * _inv(5) % q
Bx = _xrecover(By)
B = (Bx, By)


def _edwards_add(P: Tuple[int, int], Q: Tuple[int, int]) -> Tuple[int, int]:
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + d * x1 * x2 * y1 * y2) % q
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - d * x1 * x2 * y1 * y2) % q
    return (x3, y3)


def _scalarmult(P: Tuple[int, int], e: int) -> Tuple[int, int]:
    if e == 0:
        return (0, 1)
    Q = _scalarmult(P, e // 2)
    Q = _edwards_add(Q, Q)
    if e & 1:
        Q = _edwards_add(Q, P)
    return Q


def _encodepoint(P: Tuple[int, int]) -> bytes:
    x, y = P
    bits = [(y >> i) & 1 for i in range(255)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(32))


def _decodepoint(s: bytes) -> Tuple[int, int]:
    y = sum(2**i * ((s[i // 8] >> (i % 8)) & 1) for i in range(255))
    x = _xrecover(y)
    if (x & 1) != ((s[31] >> 7) & 1):
        x = q - x
    P = (x, y)
    if not _isoncurve(P):
        raise ValueError("Decoded point is not on curve25519")
    return P


def _isoncurve(P: Tuple[int, int]) -> bool:
    x, y = P
    return (-x * x + y * y - 1 - d * x * x * y * y) % q == 0


def ed25519_public_key(secret_key_32: bytes) -> bytes:
    """Derives 32-byte Ed25519 public key from 32-byte private seed."""
    if len(secret_key_32) != 32:
        raise ValueError("Ed25519 private key seed must be exactly 32 bytes")
    h = _H(secret_key_32)
    a = 2**254 + sum(2**i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    A = _scalarmult(B, a)
    return _encodepoint(A)


def ed25519_sign(message: bytes, secret_key_32: bytes, public_key_32: bytes) -> bytes:
    """Computes 64-byte Ed25519 digital signature."""
    if len(secret_key_32) != 32 or len(public_key_32) != 32:
        raise ValueError("Keys must be 32 bytes")
    h = _H(secret_key_32)
    a = 2**254 + sum(2**i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    r = int.from_bytes(_H(h[32:] + message), "little") % l
    R = _scalarmult(B, r)
    k = int.from_bytes(_H(_encodepoint(R) + public_key_32 + message), "little") % l
    S = (r + k * a) % l
    return _encodepoint(R) + S.to_bytes(32, "little")


def ed25519_verify(signature_64: bytes, message: bytes, public_key_32: bytes) -> bool:
    """Verifies 64-byte Ed25519 digital signature against 32-byte public key."""
    if len(signature_64) != 64 or len(public_key_32) != 32:
        return False
    try:
        R = _decodepoint(signature_64[:32])
        A = _decodepoint(public_key_32)
        S = int.from_bytes(signature_64[32:], "little")
        if S >= l:
            return False
        k = int.from_bytes(_H(signature_64[:32] + public_key_32 + message), "little") % l
        v1 = _scalarmult(B, S)
        v2 = _edwards_add(R, _scalarmult(A, k))
        return v1 == v2
    except Exception:
        return False
