"""HTTP surface of the bot under `/api/bot/*` as a pure function (no sockets), like `krystal_kernel.openai_api`.

    status, headers, body = handle(svc, "POST", "/api/bot/message", headers, raw_body)

No wildcard CORS is ever emitted. Bodies are capped. Pairing approval is disabled unless the operator
sets `KRYSTAL_OPERATOR_KEY` and presents it in `X-Operator-Key`.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import urllib.parse
import uuid
from typing import Any, Dict, Optional, Tuple

from . import composer as CP
from . import morse as MO
from .bytecode import BytecodeError, Program
from .llm_gateway import GatewayDenied
from .service import BotService

Response = Tuple[int, Dict[str, str], bytes]
MAX_BODY = 256 * 1024
PREFIX = "/api/bot"


def _j(status: int, obj: Any, extra: Optional[Dict[str, str]] = None) -> Response:
    h = {"Content-Type": "application/json", "Cache-Control": "no-store"}
    if extra:
        h.update(extra)
    return status, h, json.dumps(obj).encode("utf-8")


def _e(status: int, message: str, code: str, extra: Optional[Dict[str, str]] = None) -> Response:
    return _j(status, {"error": {"message": message, "code": code}}, extra)


def handle(svc: BotService, method: str, path: str, headers: Dict[str, str], body: bytes = b"") -> Response:
    rid = uuid.uuid4().hex[:16]
    headers = {k.lower(): v for k, v in headers.items()}
    s, h, b = _route(svc, method.upper(), path, headers, body)
    h["X-Request-Id"] = rid
    return s, h, b


def _route(svc: BotService, method: str, path: str, headers: Dict[str, str], body: bytes) -> Response:
    u = urllib.parse.urlparse(path)
    route = u.path.rstrip("/")
    q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
    if not route.startswith(PREFIX):
        return _e(404, "not found", "not_found")
    ok, retry = svc.quotas.try_consume("api_requests", 1)
    if not ok:
        return _e(429, "API request quota exhausted.", "rate_limited", {"Retry-After": str(max(1, int(retry + 0.999)))})
    sub = route[len(PREFIX):] or "/"
    try:
        if method == "GET":
            return _get(svc, sub, q)
        if method == "POST":
            if "application/json" not in headers.get("content-type", "").lower():
                return _e(415, "Content-Type must be application/json.", "unsupported_media_type")
            if len(body) > MAX_BODY:
                return _e(413, f"Body exceeds {MAX_BODY} bytes.", "payload_too_large")
            try:
                data = json.loads(body.decode("utf-8") or "{}")
                if not isinstance(data, dict):
                    raise ValueError("object expected")
            except (ValueError, UnicodeDecodeError) as ex:
                return _e(400, f"Invalid JSON: {ex}", "invalid_json")
            return _post(svc, sub, data, headers)
        return _e(405, "method not allowed", "method_not_allowed")
    except GatewayDenied as ex:
        st = 429 if ex.code in ("rate_limited", "token_budget_exhausted", "busy") else 403 if ex.code == "policy_denied" else 502
        extra = {"Retry-After": str(max(1, int(min(ex.retry_after_s, 86400) + 0.999)))} if 0 < ex.retry_after_s < float("inf") else None
        return _e(st, str(ex), ex.code, extra)
    except PermissionError as ex:
        return _e(403, str(ex), "donor_only")
    except (BytecodeError, ValueError, binascii.Error) as ex:
        return _e(400, str(ex), "invalid_request")
    except Exception as ex:  # noqa: BLE001
        return _e(500, f"internal error: {type(ex).__name__}", "internal_error")


def _get(svc: BotService, sub: str, q: Dict[str, str]) -> Response:
    if sub in ("", "/", "/status"):
        k = svc.kernel
        return _j(200, {"status": "OK", "kernel_attached": k is not None, "kernel_health": k.health if k is not None else None,
                        "rag": svc.rag.stats(), "palette_entries": len(svc.palette), "pending": svc.pending.stats(),
                        "risk": svc.risk.status(), "llm_policy": svc.llm.policy(), "donor": svc.is_donor(),
                        "keras_configured": svc.keras is not None, "mesh_nodes": len(svc.mesh.peers()), "skills": svc.gateway.skills()})
    if sub == "/palette":
        return _j(200, {"status": "OK", "conversations": svc.palette.conversations()[:20],
                        "entries": svc.palette.tail(int(q.get("n", "50")), q.get("conversation"))})
    if sub == "/search":
        k = max(1, min(int(q.get("k", "5")), 20))
        return _j(200, {"status": "OK", "query": q.get("q", ""), "hits": svc.rag.search(q.get("q", ""), k, [x for x in q.get("kinds", "").split(",") if x] or None)})
    if sub == "/quota":
        return _j(200, {"status": "OK", "quotas": svc.quotas.snapshot()})
    if sub == "/risk":
        return _j(200, {"status": "OK", "model": svc.risk.status(), "pending": svc.pending.stats(), "recent_flags": svc.flags[-20:]})
    if sub == "/predict":
        return _j(200, {"status": "OK", **svc.predict_next(int(q.get("k", "3")))})
    if sub == "/llm/policy":
        return _j(200, {"status": "OK", "config": svc.llm.cfg.public(), "policy": svc.llm.policy(), "stats": svc.llm.stats, "audit": svc.audit_log[-20:]})
    if sub == "/mesh/peers":
        return _j(200, {"status": "OK", "peers": svc.mesh.peers(), "cycles": svc.mesh.cycles})
    if sub == "/donor":
        return _j(200, {"status": "OK", "active": svc.is_donor(), "alias": (svc.donor or {}).get("sub"), "issuer_keys_configured": len(svc.pubkeys)})
    if sub == "/palettes":
        return _j(200, {"status": "OK", "palettes": [{"id": p, "donor_only": p in CP.DONOR_PALETTES} for p in CP.PALETTE_IDS]})
    return _e(404, f"unknown route {sub}", "not_found")


def _post(svc: BotService, sub: str, d: Dict[str, Any], headers: Dict[str, str]) -> Response:
    if sub == "/message":
        text = d.get("text")
        if not isinstance(text, str) or not text.strip():
            return _e(400, "'text' must be a non-empty string.", "invalid_text")
        channel = d.get("channel", "extension")
        if channel not in ("extension", "web", "cli", "mesh"):
            return _e(400, "channel must be extension|web|cli|mesh", "invalid_channel")
        return _j(200, {"status": "OK", **svc.message(text[:4000], channel, d.get("conversation"), str(d.get("sender", "local"))[:64])})
    if sub == "/render":
        return _j(200, {"status": "OK", **svc.render(d.get("config"), d.get("format", "png"), bool(d.get("force", False)))})
    if sub == "/replicate":
        return _j(200, {"status": "OK", **svc.replicate(str(d.get("bytecode_b64", "")), float(d.get("rate", 0.08)), d.get("seed"), int(d.get("count", 1)))})
    if sub == "/morse":
        if "text" in d:
            return _j(200, {"status": "OK", "morse": MO.encode_text(str(d["text"])[:2000])})
        if "morse" in d:
            return _j(200, {"status": "OK", "text": MO.decode_text(str(d["morse"])[:20000])})
        if "bytecode_b64" in d:
            prog = Program.from_bytes(base64.b64decode(d["bytecode_b64"], validate=True))
            m = MO.encode_program(prog)
            return _j(200, {"status": "OK", "morse": m, "tensor_len": len(MO.morse_to_tensor(m, 20000)), "digest": prog.digest()})
        return _e(400, "provide text, morse or bytecode_b64", "invalid_request")
    if sub == "/risk/train":
        return _j(200, {"status": "OK", "report": svc.train_risk(str(d.get("prefer", "keras")))})
    if sub == "/reindex":
        out: Dict[str, Any] = {}
        if d.get("docs", True):
            out["docs"] = svc.reindex_docs()
        if d.get("events", True):
            out["events"] = svc.reindex_events()
        svc.rag.save()
        return _j(200, {"status": "OK", **out, "rag": svc.rag.stats()})
    if sub == "/donor/activate":
        r = svc.activate_donor(str(d.get("token", "")))
        return _j(200 if r["valid"] else 403, {"status": "OK" if r["valid"] else "DENIED", **r})
    if sub == "/mesh/register":
        if not svc.is_donor():
            raise PermissionError("the mesh is part of the donor easter egg")
        return _j(200, {"status": "OK", "node_id": svc.mesh.register(str(d.get("name", "node")), str(d.get("url", "")))})
    if sub == "/mesh/cycle":
        if not svc.is_donor():
            raise PermissionError("the mesh is part of the donor easter egg")
        return _j(200, {"status": "OK", **svc.mesh.cycle()})
    if sub == "/pair/approve":
        key = os.environ.get("KRYSTAL_OPERATOR_KEY", "")
        if not key:
            return _e(403, "pairing approval is disabled: set KRYSTAL_OPERATOR_KEY.", "approval_disabled")
        if not hmac.compare_digest(headers.get("x-operator-key", "").encode(), key.encode()):
            return _e(401, "bad operator key", "bad_operator_key")
        who = svc.gateway.pairing.approve(str(d.get("code", "")))
        return _j(200 if who else 404, {"status": "OK" if who else "NOT_FOUND", "paired": who})
    if sub == "/llm/document":
        kind, inp = d.get("kind"), d.get("input", "")
        if not isinstance(inp, str):
            return _e(400, "'input' must be a string (base64 for binary_log)", "invalid_input")
        if kind == "math":
            r = svc.llm.document_math(inp, str(d.get("context", "")))
        elif kind == "binary_log":
            r = svc.llm.document_binary_log(base64.b64decode(inp, validate=True)[:65536], str(d.get("janet_context", "")))
        elif kind == "janet":
            r = svc.llm.document_janet(inp)
        else:
            return _e(400, "kind must be math|binary_log|janet", "invalid_kind")
        return _j(200, {"status": "OK", "markdown": r["markdown"], "facts": r["facts"], "model": r["model"], "usage": r.get("usage")})
    return _e(404, f"unknown route {sub}", "not_found")
