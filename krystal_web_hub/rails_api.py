"""HTTP surface for configuration rails, mounted by ``krystal_web_hub/server.py``.

Routes (all JSON):

``GET  /api/rails/nodes``     node catalogue + limits
``GET  /api/rails/presets``   preset rails from ``krystal_kernel/rails/presets``
``POST /api/rails/validate``  body = rail document -> ``{valid, errors, order}``
``POST /api/rails/run``       body = rail document -> summarised run result

Hardening (repo rule ``engineering-verification-and-honesty``):

* No ``Access-Control-Allow-Origin`` header: the Config Rails Studio page is served
  from the same origin, so cross-origin pages cannot read results.
* POST requires ``Content-Type: application/json`` and, when a browser sends an
  ``Origin`` header, it must match ``Host``. The hub's generic ``do_OPTIONS`` answers
  every preflight with ``*``, so this check is what stops cross-site pages from
  *triggering* runs (real sources spin the CPU for up to 10 s per rail).
* The body is capped at :data:`MAX_BODY` and rejected *before* reading.
* At most one rail runs at a time (``429`` otherwise).
* Strict JSON out: ``nan``/``inf`` are converted to ``null``.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict
from urllib.parse import urlparse

from krystal_kernel.rails import (
    DEFAULT_REGISTRY,
    MAX_LINKS,
    MAX_NODES,
    MAX_RAIL_SOURCE_SECONDS,
    check_rail,
    jsonable,
    list_presets,
    run_rail,
    summarize_result,
)
from krystal_kernel.rails.library import MAX_SAMPLES

PREFIX = "/api/rails"
MAX_BODY = 64 * 1024
_RUN_LOCK = threading.Lock()


def is_rails_path(path: str) -> bool:
    route = urlparse(path).path.rstrip("/")
    return route == PREFIX or route.startswith(PREFIX + "/")


def _send(handler, status: int, payload: Dict[str, Any], close: bool = False) -> None:
    body = json.dumps(jsonable(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    # Deliberately no Access-Control-Allow-Origin (same-origin UI only).
    if close:
        handler.send_header("Connection", "close")
        handler.close_connection = True
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler, status: int, code: str, message: str, close: bool = False) -> None:
    _send(handler, status, {"error": {"code": code, "message": message}}, close=close)


def _same_origin(handler) -> bool:
    origin = handler.headers.get("Origin")
    if origin is None:
        return True  # non-browser clients (scripts, tests) send no Origin
    host = handler.headers.get("Host", "")
    return origin in (f"http://{host}", f"https://{host}")


def handle(handler, method: str) -> None:
    route = urlparse(handler.path).path.rstrip("/")
    if method == "GET":
        if route == PREFIX + "/nodes":
            _send(handler, 200, {
                "nodes": DEFAULT_REGISTRY.describe(),
                "limits": {"max_nodes": MAX_NODES, "max_links": MAX_LINKS,
                           "max_samples": MAX_SAMPLES,
                           "max_rail_source_seconds": MAX_RAIL_SOURCE_SECONDS,
                           "max_body_bytes": MAX_BODY},
            })
        elif route == PREFIX + "/presets":
            _send(handler, 200, {"presets": list_presets()})
        else:
            _error(handler, 404, "not_found", f"unknown rails route {route!r}")
        return

    if method != "POST":
        _error(handler, 405, "method_not_allowed", method, close=True)
        return
    if not _same_origin(handler):
        _error(handler, 403, "cross_origin", "Origin does not match Host.", close=True)
        return
    ctype = handler.headers.get("Content-Type", "")
    if not ctype.split(";")[0].strip().lower() == "application/json":
        _error(handler, 415, "unsupported_media_type", "Content-Type must be application/json.", close=True)
        return
    try:
        length = int(handler.headers.get("Content-Length", ""))
    except ValueError:
        length = -1
    if length < 0:
        _error(handler, 411, "length_required", "A valid Content-Length is required.", close=True)
        return
    if length > MAX_BODY:
        # Reject BEFORE reading so an oversized body is never buffered.
        _error(handler, 413, "payload_too_large", f"Body exceeds {MAX_BODY} bytes.", close=True)
        return
    raw = handler.rfile.read(length) if length else b""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        _error(handler, 400, "invalid_json", f"Body is not valid JSON: {exc}")
        return

    if route == PREFIX + "/validate":
        v, errors = check_rail(doc)
        _send(handler, 200, {"valid": not errors, "errors": errors,
                             "order": v.order if v else [],
                             "source_budget_s": v.source_seconds if v else None})
    elif route == PREFIX + "/run":
        if not _RUN_LOCK.acquire(blocking=False):
            _error(handler, 429, "busy", "Another rail is running; retry shortly.")
            return
        try:
            v, errors = check_rail(doc)
            if errors:
                _send(handler, 422, {"valid": False, "errors": errors})
                return
            result = run_rail(doc)
        finally:
            _RUN_LOCK.release()
        _send(handler, 200, {"valid": True, **summarize_result(result)})
    else:
        _error(handler, 404, "not_found", f"unknown rails route {route!r}")
