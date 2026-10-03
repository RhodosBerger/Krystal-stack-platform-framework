"""Named compute kernels. Every kernel is a pure function of a JSON-ish payload dict.

Named (not arbitrary-callable) dispatch keeps the process lane safe and picklable: workers only
execute functions registered here.
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable, Dict, List

REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def kernel(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


@kernel("echo")
def _echo(p):
    return p


@kernel("burn")
def _burn(p):
    """CPU-bound spin used by calibration and tests."""
    x = 0
    for i in range(int(p["n"])):
        x += i * i % 7
    return x


@kernel("sleep")
def _sleep(p):
    time.sleep(float(p["s"]))
    return p["s"]


@kernel("fail")
def _fail(p):
    raise RuntimeError(p.get("msg", "injected failure"))


@kernel("crash")
def _crash(p):
    """Kill the hosting process hard (used to test self-healing of the worker pool)."""
    import os
    os._exit(7)


EMBED_DIM = 64


def _ngrams(text: str, n_lo: int = 2, n_hi: int = 4):
    t = f" {text.lower()} "
    for n in range(n_lo, n_hi + 1):
        for i in range(len(t) - n + 1):
            yield t[i:i + n]


@kernel("hash_embed")
def hash_embed(p):
    """Feature-hashing char n-gram embedding, L2-normalised.

    Deterministic and real, but NOT semantic: it measures surface (character n-gram) similarity.
    """
    dim = int(p.get("dim", EMBED_DIM))
    out: List[List[float]] = []
    for text in p["texts"]:
        v = [0.0] * dim
        for g in _ngrams(str(text)):
            h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
            v[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([round(x / norm, 6) for x in v])
    return out


@kernel("text_stats")
def text_stats(p):
    """Cheap per-text statistics used by the reference chat model."""
    res = []
    for t in p["texts"]:
        t = str(t)
        words = t.split()
        res.append({"chars": len(t), "words": len(words), "unique_words": len(set(w.lower() for w in words))})
    return res
