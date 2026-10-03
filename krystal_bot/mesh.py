"""Localhost request mesh: UUID-identified nodes that talk only over signed HTTP requests (stdlib only).

Not an internet service. Rules enforced in code:
* a node URL must be plain `http://` on a loopback address (127.0.0.0/8, ::1, localhost);
* redirects are never followed (a hostile node cannot bounce us elsewhere);
* every request and response carries an HMAC-SHA256 over method, path, timestamp, nonce and body hash
  under a per-mesh secret, with a replay cache and a clock-skew window;
* the node count is capped, and a node that fails three hops in a row is evicted.

A *cycle* sends a fresh UUID around the ring of nodes and records per-hop latency and each node's
self-reported telemetry, so the mesh is always built from requests and always circulates identifiers.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_NODES = 16
SKEW_S = 30.0
MAX_BODY = 64 * 1024


def is_loopback_url(url: str) -> bool:
    u = urllib.parse.urlparse(url)
    if u.scheme != "http" or not u.hostname or u.username or u.password:
        return False
    if u.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(u.hostname).is_loopback
    except ValueError:
        return False


def load_or_create_secret(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            s = f.read(32)
        if len(s) == 32:
            return s
    except OSError:
        pass
    s = os.urandom(32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(s)
    return s


class MeshAuth:
    def __init__(self, secret: bytes, clock: Callable[[], float] = time.time):
        if len(secret) < 16:
            raise ValueError("mesh secret too short")
        self.secret, self._clock = secret, clock
        self._seen: "OrderedDict[str, float]" = OrderedDict()
        self._lock = threading.Lock()

    def mac(self, kind: str, method: str, path: str, ts: str, nonce: str, body: bytes) -> str:
        msg = "\n".join([kind, method.upper(), path, ts, nonce, hashlib.sha256(body).hexdigest()]).encode("utf-8")
        return hmac.new(self.secret, msg, hashlib.sha256).hexdigest()

    def sign_request(self, method: str, path: str, body: bytes) -> Dict[str, str]:
        ts, nonce = f"{self._clock():.3f}", uuid.uuid4().hex
        return {"X-Krystal-Ts": ts, "X-Krystal-Nonce": nonce, "X-Krystal-Sig": self.mac("req", method, path, ts, nonce, body)}

    def verify_request(self, method: str, path: str, headers: Dict[str, str], body: bytes) -> Tuple[bool, str]:
        ts, nonce, sig = headers.get("X-Krystal-Ts", ""), headers.get("X-Krystal-Nonce", ""), headers.get("X-Krystal-Sig", "")
        try:
            skew = abs(self._clock() - float(ts))
        except ValueError:
            return False, "bad timestamp"
        if skew > SKEW_S:
            return False, "timestamp outside window"
        if not hmac.compare_digest(sig, self.mac("req", method, path, ts, nonce, body)):
            return False, "bad signature"
        with self._lock:
            now = self._clock()
            while self._seen and next(iter(self._seen.values())) < now - SKEW_S * 2:
                self._seen.popitem(last=False)
            if nonce in self._seen:
                return False, "replayed nonce"
            self._seen[nonce] = now
            while len(self._seen) > 4096:
                self._seen.popitem(last=False)
        return True, "ok"

    def sign_response(self, request_nonce: str, body: bytes) -> str:
        return self.mac("resp", "", "", "", request_nonce, body)

    def verify_response(self, request_nonce: str, body: bytes, sig: str) -> bool:
        return hmac.compare_digest(sig or "", self.sign_response(request_nonce, body))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a: Any, **k: Any) -> None:  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def signed_post(auth: MeshAuth, url: str, path: str, obj: Dict[str, Any], timeout: float = 2.0) -> Tuple[int, Dict[str, Any], bool]:
    """POST a signed JSON body to a loopback node. Returns (status, json, response_signature_valid)."""
    if not is_loopback_url(url):
        raise ValueError("mesh nodes must be plain-http loopback URLs")
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    hdr = auth.sign_request("POST", path, body)
    hdr["Content-Type"] = "application/json"
    req = urllib.request.Request(url.rstrip("/") + path, data=body, headers=hdr, method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as r:  # noqa: S310 - loopback http enforced above
            raw, status, rh = r.read(MAX_BODY), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        try:
            raw, status, rh = e.read(MAX_BODY), e.code, dict(e.headers or {})
        finally:
            e.close()
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        data = {}
    return status, data, auth.verify_response(hdr["X-Krystal-Nonce"], raw, rh.get("X-Krystal-Sig", ""))


class MeshNode:
    """A participant: owns a UUID and a tiny loopback HTTP server that answers `/mesh/hop`."""

    def __init__(self, auth: MeshAuth, name: str, telemetry: Optional[Callable[[], Dict[str, Any]]] = None):
        self.auth, self.name, self.telemetry = auth, name, telemetry or (lambda: {})
        self.node_id = uuid.uuid4().hex
        self._srv: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port = 0
        self.hops_served = 0
        node = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0") or 0)
                if n > MAX_BODY:
                    self.send_error(413)
                    return
                body = self.rfile.read(n)
                ok, why = node.auth.verify_request("POST", self.path, dict(self.headers), body)
                if not ok or self.path != "/mesh/hop":
                    out = json.dumps({"error": why if not ok else "not found"}).encode()
                    self.send_response(401 if not ok else 404)
                else:
                    try:
                        req = json.loads(body.decode("utf-8"))
                        out = json.dumps({"node_id": node.node_id, "name": node.name, "cycle_id": req.get("cycle_id"),
                                          "hop_index": req.get("hop_index"), "t": round(time.time(), 4), "telemetry": node.telemetry()}).encode()
                        node.hops_served += 1
                        self.send_response(200)
                    except (ValueError, TypeError):
                        out = b'{"error":"bad json"}'
                        self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.send_header("X-Krystal-Sig", node.auth.sign_response(self.headers.get("X-Krystal-Nonce", ""), out))
                self.end_headers()
                self.wfile.write(out)

        self._handler = H

    def start(self) -> int:
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._srv.daemon_threads = True
        self.port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True, name=f"mesh-{self.name}")
        self._thread.start()
        return self.port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            if self._thread:
                self._thread.join(timeout=3)
            self._srv = None


class Mesh:
    def __init__(self, auth: MeshAuth, on_cycle: Optional[Callable[[Dict[str, Any]], None]] = None, timeout: float = 2.0):
        self.auth, self.on_cycle, self.timeout = auth, on_cycle, timeout
        self._nodes: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.cycles = 0

    def register(self, name: str, url: str, node_id: Optional[str] = None) -> str:
        """Add a node. Without `node_id` the node is probed with a signed hop and its own UUID is adopted."""
        if not is_loopback_url(url):
            raise ValueError("node url must be plain-http loopback")
        if node_id is None:
            try:
                status, data, sig_ok = signed_post(self.auth, url, "/mesh/hop", {"cycle_id": "register", "hop_index": -1, "trace": []}, self.timeout)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"node did not answer a signed probe ({type(e).__name__})") from None
            nid_reported = data.get("node_id")
            if status != 200 or not sig_ok or not isinstance(nid_reported, str) or not 8 <= len(nid_reported) <= 64:
                raise ValueError("node failed the signed probe (wrong secret or not a mesh node)")
            node_id = nid_reported
        with self._lock:
            if len(self._nodes) >= MAX_NODES:
                raise ValueError(f"mesh is full ({MAX_NODES} nodes)")
            self._nodes[node_id] = {"node_id": node_id, "name": str(name)[:48], "url": url, "joined": time.time(), "failures": 0, "last_ok": None}
            return node_id

    def remove(self, node_id: str) -> bool:
        with self._lock:
            return self._nodes.pop(node_id, None) is not None

    def peers(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(v) for v in self._nodes.values()]

    def cycle(self) -> Dict[str, Any]:
        cid = uuid.uuid4().hex
        with self._lock:
            ring = list(self._nodes.values())
        t0, hops, trace = time.perf_counter(), [], []
        for i, n in enumerate(ring):
            h0 = time.perf_counter()
            rec: Dict[str, Any] = {"node_id": n["node_id"], "name": n["name"]}
            try:
                status, data, sig_ok = signed_post(self.auth, n["url"], "/mesh/hop", {"cycle_id": cid, "hop_index": i, "trace": trace[-8:]}, self.timeout)
                valid = status == 200 and sig_ok and data.get("cycle_id") == cid and data.get("node_id") == n["node_id"]
                rec.update(ok=valid, status=status, signature_valid=sig_ok, rtt_ms=round((time.perf_counter() - h0) * 1000, 3),
                           telemetry=data.get("telemetry") if valid else None)
                if not valid:
                    rec["error"] = "response failed validation"
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, rtt_ms=round((time.perf_counter() - h0) * 1000, 3), error=type(e).__name__)
            with self._lock:
                live = self._nodes.get(n["node_id"])
                if live is not None:
                    if rec["ok"]:
                        live["failures"], live["last_ok"] = 0, time.time()
                    else:
                        live["failures"] += 1
                        if live["failures"] >= 3:
                            del self._nodes[n["node_id"]]
                            rec["evicted"] = True
            trace.append(n["node_id"][:8])
            hops.append(rec)
        self.cycles += 1
        result = {"cycle_id": cid, "nodes": len(ring), "ok": bool(ring) and all(h["ok"] for h in hops),
                  "total_ms": round((time.perf_counter() - t0) * 1000, 3), "hops": hops}
        if self.on_cycle:
            try:
                self.on_cycle(result)
            except Exception:  # noqa: BLE001
                pass
        return result
