"""OpenAI-compatible HTTP surface, as pure functions (testable without a socket).

    status, headers, body = handle(ctx, method, path, headers, body_bytes)

Implemented: GET /v1/models, POST /v1/embeddings, POST /v1/chat/completions (non-streaming),
GET /metrics (Prometheus text), GET /api/kernel/{status,patterns,events,params,hardware}.

Honest scope:
  * `krystal-hash-embed-64` is a deterministic feature-hashing embedding (surface similarity, not semantic).
  * `krystal-reference-chat` is NOT a language model; it returns text statistics. It exists so the whole
    pipeline (auth -> rate limit -> admission -> scheduler -> lane -> response) can be exercised and
    measured today. Plug a real model in by registering a backend.
  * `usage` token counts are whitespace-token approximations.
  * `stream=true` and `encoding_format` other than float/base64 are rejected explicitly.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import struct
import time
import urllib.parse
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backends import BackendRouter, ReferenceBackend
from .kernel import Backpressure, ComputeKernel, DeadlineMissed

EMBED_MODEL = "krystal-hash-embed-64"
CHAT_MODEL = "krystal-reference-chat"
MAX_INPUTS = 256
MAX_CHARS = 8192
Response = Tuple[int, Dict[str, str], bytes]


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: float, clock: Callable[[], float] = time.monotonic):
        self.rate, self.burst, self.clock = rate_per_s, burst, clock
        self._b: Dict[str, Tuple[float, float]] = {}

    def take(self, tenant: str, n: float = 1.0) -> Tuple[bool, float]:
        now = self.clock()
        tokens, last = self._b.get(tenant, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens >= n:
            self._b[tenant] = (tokens - n, now)
            return True, 0.0
        self._b[tenant] = (tokens, now)
        return False, (n - tokens) / max(self.rate, 1e-9)


class ApiContext:
    def __init__(self, kernel: ComputeKernel, router: Optional[BackendRouter] = None, keys: Optional[List[str]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.kernel = kernel
        self.router = router or BackendRouter([ReferenceBackend(kernel)], log=kernel.log, window=kernel.params.breaker_window,
                                              error_rate=kernel.params.breaker_error_rate, cooldown_s=kernel.params.breaker_cooldown_s)
        kernel.backends = self.router
        if keys is None:
            env = os.environ.get("KRYSTAL_API_KEYS", "")
            keys = [k.strip() for k in env.split(",") if k.strip()]
        self.keys = keys
        self.limiter = TokenBucket(kernel.params.rate_per_s, kernel.params.burst, clock)
        self.started = time.time()

    @property
    def max_body(self) -> int:
        return self.kernel.params.max_payload_bytes


def _json(status: int, obj: Any, extra: Optional[Dict[str, str]] = None) -> Response:
    h = {"Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return status, h, json.dumps(obj).encode("utf-8")


def _err(status: int, message: str, etype: str = "invalid_request_error", code: Optional[str] = None,
         param: Optional[str] = None, extra: Optional[Dict[str, str]] = None) -> Response:
    return _json(status, {"error": {"message": message, "type": etype, "param": param, "code": code}}, extra)


def _approx_tokens(texts: List[str]) -> int:
    return sum(len(t.split()) for t in texts)


def _authenticate(ctx: ApiContext, headers: Dict[str, str]) -> Tuple[bool, str]:
    if not ctx.keys:
        return True, "anonymous"
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        for k in ctx.keys:
            if hmac.compare_digest(tok.encode(), k.encode()):
                return True, "key:" + k[:4]
    return False, ""


def handle(ctx: ApiContext, method: str, path: str, headers: Dict[str, str], body: bytes = b"") -> Response:
    rid = "req_" + uuid.uuid4().hex[:16]
    headers = {k.lower(): v for k, v in headers.items()}
    status, h, out = _route(ctx, method.upper(), path, headers, body)
    h["X-Request-Id"] = rid
    h.setdefault("Cache-Control", "no-store")
    return status, h, out


def _route(ctx: ApiContext, method: str, path: str, headers: Dict[str, str], body: bytes) -> Response:
    u = urllib.parse.urlparse(path)
    route = u.path.rstrip("/") or "/"
    q = urllib.parse.parse_qs(u.query)
    ok, tenant = _authenticate(ctx, headers)
    if not ok:
        return _err(401, "Invalid or missing API key. Send 'Authorization: Bearer <key>'.", "authentication_error", "invalid_api_key",
                    extra={"WWW-Authenticate": "Bearer"})

    allowed, retry = ctx.limiter.take(tenant, 1.0)
    if not allowed:
        return _err(429, "Rate limit exceeded.", "rate_limit_error", "rate_limit_exceeded",
                    extra={"Retry-After": str(max(1, int(retry + 0.999)))})

    if method == "GET":
        if route == "/v1/models":
            return _json(200, {"object": "list", "data": [
                {"id": EMBED_MODEL, "object": "model", "created": int(ctx.started), "owned_by": "krystal-stack", "capabilities": ["embeddings"]},
                {"id": CHAT_MODEL, "object": "model", "created": int(ctx.started), "owned_by": "krystal-stack", "capabilities": ["chat"],
                 "note": "reference model, not an LLM"}]})
        if route == "/metrics":
            return 200, {"Content-Type": "text/plain; version=0.0.4"}, ctx.kernel.prometheus().encode("utf-8")
        if route == "/api/kernel/status":
            return _json(200, {"status": "OK", **ctx.kernel.stats(), "backends": ctx.router.status()})
        if route == "/api/kernel/patterns":
            pats = ctx.kernel.learner.patterns()
            return _json(200, {"status": "OK", "count": len(pats), "patterns": pats,
                               "routing_table": ctx.kernel.learner.bandit.table()})
        if route == "/api/kernel/events":
            n = int((q.get("n") or ["50"])[0])
            et = (q.get("type") or [None])[0]
            return _json(200, {"status": "OK", "counts": ctx.kernel.log.counts(), "events": ctx.kernel.log.tail(max(1, min(n, 500)), et)})
        if route == "/api/kernel/params":
            return _json(200, {"status": "OK", "params": ctx.kernel.params.explain()})
        if route == "/api/kernel/hardware":
            return _json(200, {"status": "OK", "hardware": ctx.kernel.profile.to_dict()})
        return _err(404, f"Unknown route {route}", "invalid_request_error", "not_found")

    if method == "POST":
        if "application/json" not in headers.get("content-type", "").lower():
            return _err(415, "Content-Type must be application/json.", code="unsupported_media_type")
        if len(body) > ctx.max_body:
            return _err(413, f"Body exceeds {ctx.max_body} bytes.", code="payload_too_large")
        try:
            data = json.loads(body.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("JSON object expected")
        except (ValueError, UnicodeDecodeError) as e:
            return _err(400, f"Invalid JSON body: {e}", code="invalid_json")
        try:
            if route == "/v1/embeddings":
                return _embeddings(ctx, data, tenant)
            if route == "/v1/chat/completions":
                return _chat(ctx, data, tenant)
        except Backpressure as bp:
            return _err(503, f"Server overloaded: {bp.reason}.", "server_error", "overloaded",
                        extra={"Retry-After": str(max(1, int(bp.retry_after_s + 0.999)))})
        except DeadlineMissed:
            return _err(504, "Request deadline passed while queued.", "server_error", "deadline_exceeded")
        except Exception as e:  # noqa: BLE001
            return _err(500, f"Internal error: {type(e).__name__}", "server_error", "internal_error")
        return _err(404, f"Unknown route {route}", code="not_found")
    return _err(405, f"Method {method} not allowed.", code="method_not_allowed")


def _embeddings(ctx: ApiContext, data: Dict[str, Any], tenant: str) -> Response:
    model = data.get("model", EMBED_MODEL)
    if model != EMBED_MODEL:
        return _err(404, f"Model '{model}' not found.", code="model_not_found", param="model")
    inp = data.get("input")
    texts = [inp] if isinstance(inp, str) else inp
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        return _err(400, "'input' must be a non-empty string or list of strings.", param="input", code="invalid_input")
    if len(texts) > MAX_INPUTS or any(len(t) > MAX_CHARS for t in texts):
        return _err(400, f"At most {MAX_INPUTS} inputs of {MAX_CHARS} characters.", param="input", code="input_too_large")
    if data.get("dimensions") not in (None, 64):
        return _err(400, "Only dimensions=64 is supported.", param="dimensions", code="unsupported_dimensions")
    fmt = data.get("encoding_format", "float")
    if fmt not in ("float", "base64"):
        return _err(400, "encoding_format must be 'float' or 'base64'.", param="encoding_format", code="invalid_encoding_format")
    extra = ctx.limiter.take(tenant, max(0.0, len(texts) / 16.0))  # batch size costs extra tokens
    if not extra[0]:
        return _err(429, "Rate limit exceeded for batch size.", "rate_limit_error", "rate_limit_exceeded",
                    extra={"Retry-After": str(max(1, int(extra[1] + 0.999)))})
    vecs, backend = ctx.router.embed(texts)
    items = []
    for i, v in enumerate(vecs):
        emb: Any = v if fmt == "float" else base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode("ascii")
        items.append({"object": "embedding", "index": i, "embedding": emb})
    n = _approx_tokens(texts)
    return _json(200, {"object": "list", "data": items, "model": EMBED_MODEL,
                       "usage": {"prompt_tokens": n, "total_tokens": n, "approximate": True}},
                 extra={"X-Krystal-Backend": backend})


def _chat(ctx: ApiContext, data: Dict[str, Any], tenant: str) -> Response:
    model = data.get("model", CHAT_MODEL)
    if model != CHAT_MODEL:
        return _err(404, f"Model '{model}' not found.", code="model_not_found", param="model")
    if data.get("stream"):
        return _err(400, "stream=true is not supported by this server.", param="stream", code="stream_not_supported")
    msgs = data.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("role") for m in msgs):
        return _err(400, "'messages' must be a non-empty list of {role, content} objects.", param="messages", code="invalid_messages")
    contents = [m["content"] for m in msgs]
    if any(len(c) > MAX_CHARS for c in contents):
        return _err(400, f"Each message is limited to {MAX_CHARS} characters.", param="messages", code="input_too_large")
    last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), contents[-1])
    stats = ctx.kernel.submit("text_stats", {"texts": [last_user]}, kind="chat", cost=1, deadline_ms=5000, tenant=tenant).result(timeout=10)[0]
    text = (f"[{CHAT_MODEL}] This server has no language model attached. Your last user message has {stats['words']} words "
            f"({stats['unique_words']} unique, {stats['chars']} characters).")
    pt = _approx_tokens(contents)
    ct = _approx_tokens([text])
    return _json(200, {
        "id": "chatcmpl-" + uuid.uuid4().hex[:20], "object": "chat.completion", "created": int(time.time()), "model": CHAT_MODEL,
        "system_fingerprint": "krystal-reference-0",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "approximate": True}})
