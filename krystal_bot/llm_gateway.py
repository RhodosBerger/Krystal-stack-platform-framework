"""Quota-managed gateway to any OpenAI-compatible chat-completions endpoint (stdlib only).

Policy (deny by default):

* disabled until `enabled: true` in the config;
* usable only in **debug mode** (env `KRYSTAL_DEBUG=1` or config `debug: true`) **or** on a
  **high-performance device** (logical cores / RAM above configurable thresholds);
* a non-private endpoint additionally needs `allow_remote: true` because prompt text leaves the machine;
* every call is charged against a requests/minute quota and a tokens/day quota *before* it is sent,
  with unused reservation refunded from the provider's `usage`;
* secrets are read from an environment variable and are never written to the config, logs or errors.

LLM output is untrusted and may be wrong, particularly for mathematics. Documentation helpers
therefore attach locally computed ground truth where it exists (safe arithmetic evaluation, binary
log structure statistics) and stamp every result as AI-generated and unverified.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .quota import QuotaManager

Transport = Callable[[str, Dict[str, str], bytes, float], Tuple[int, Dict[str, str], bytes]]


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434/v1"  # e.g. Ollama / llama.cpp / vLLM; any OpenAI-compatible server
    model: str = ""
    api_key_env: str = ""            # NAME of the env var holding the key, never the key itself
    allow_remote: bool = False
    debug: bool = False
    min_logical_cores: int = 16
    min_ram_gb: float = 32.0
    requests_per_min: int = 10
    tokens_per_day: int = 50_000
    max_prompt_chars: int = 12_000
    max_output_tokens: int = 700
    timeout_s: float = 60.0
    max_concurrency: int = 1
    breaker_failures: int = 3
    breaker_cooldown_s: float = 60.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "LLMConfig":
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.api_key_env = cfg.api_key_env if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cfg.api_key_env or "") else ""
        return cfg

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, path)

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        d["api_key_present"] = bool(self.api_key_env and os.environ.get(self.api_key_env))
        return d


class GatewayDenied(Exception):
    def __init__(self, code: str, message: str, retry_after_s: float = 0.0):
        super().__init__(message)
        self.code, self.retry_after_s = code, retry_after_s


# ---------------------------------------------------------------------- helpers
_SECRET_PATTERNS = [
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{12,}"), r"\1 [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "[REDACTED_KEY]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s]+"), r"C:\\Users\\[USER]"),
]


def redact(text: str) -> str:
    for pat, rep in _SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return text


def estimate_tokens(text: str) -> int:
    """Rough upper-ish estimate (~3.5 chars/token) used only for budgeting, never for billing."""
    return max(1, int(len(text) / 3.5) + 1)


def is_private_endpoint(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("localhost", "::1") or host.endswith(".local"):
        return True
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        a, b = int(parts[0]), int(parts[1])
        return a == 127 or a == 10 or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31)
    return False


def default_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Tuple[int, Dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - scheme validated by caller
            return r.status, dict(r.headers), r.read(2_000_000)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read(200_000)


# -- safe arithmetic (ground truth for math documentation) ----------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "exp": math.exp, "abs": abs,
          "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval_math(expr: str) -> float:
    """Evaluate a plain arithmetic expression without `eval`. Raises ValueError on anything else."""
    if len(expr) > 200:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"not an arithmetic expression: {e.msg}") from None

    def ev(n: ast.AST, depth: int = 0) -> float:
        if depth > 20:
            raise ValueError("expression too deep")
        if isinstance(n, ast.Expression):
            return ev(n.body, depth + 1)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand, depth + 1))
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left, depth + 1), ev(n.right, depth + 1)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1 and not n.keywords:
            return _FUNCS[n.func.id](ev(n.args[0], depth + 1))
        raise ValueError(f"unsupported element: {type(n).__name__}")

    try:
        v = ev(tree)
    except (ZeroDivisionError, OverflowError) as e:
        raise ValueError(type(e).__name__) from None
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("non-finite result")
    return v


# -- binary-log structure analysis (deterministic, local) -----------------------------
def hexdump(data: bytes, limit: int = 256, width: int = 16) -> str:
    rows = []
    for off in range(0, min(len(data), limit), width):
        chunk = data[off:off + width]
        rows.append(f"{off:08x}  {' '.join(f'{b:02x}' for b in chunk):<{width * 3}} |{''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)}|")
    return "\n".join(rows)


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    c = Counter(data)
    n = len(data)
    return -sum(v / n * math.log2(v / n) for v in c.values())


_MAGIC = [(b"\x89PNG", "png"), (b"PK\x03\x04", "zip"), (b"\x1f\x8b", "gzip"), (b"{", "json?"), (b"%PDF", "pdf"), (b"\x7fELF", "elf"), (b"MZ", "pe")]


def analyze_binary_log(data: bytes, max_record: int = 256) -> Dict[str, Any]:
    """Guess a fixed record size by byte-level autocorrelation and report cheap facts for the prompt."""
    n = len(data)
    sample = data[:65536]
    best, best_score = None, 0.0
    for size in range(2, min(max_record, len(sample) // 3) + 1):
        pairs = len(sample) - size
        same = sum(1 for i in range(pairs) if sample[i] == sample[i + size])
        score = same / pairs if pairs else 0.0
        if score > best_score + 1e-9:
            best, best_score = size, score
    magic = next((name for sig, name in _MAGIC if data.startswith(sig)), None)
    return {"bytes": n, "entropy_bits_per_byte": round(_entropy(sample), 3), "magic": magic,
            "printable_ratio": round(sum(1 for b in sample if 32 <= b < 127) / max(1, len(sample)), 3),
            "likely_record_size": best if best_score >= 0.30 else None, "record_size_confidence": round(best_score, 3),
            "looks_compressed_or_encrypted": _entropy(sample) > 7.5}


# ---------------------------------------------------------------------- gateway
class LLMGateway:
    def __init__(self, config: LLMConfig, quotas: Optional[QuotaManager] = None, transport: Optional[Transport] = None,
                 profile_fn: Optional[Callable[[], Dict[str, Any]]] = None, clock: Callable[[], float] = time.time,
                 audit: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.cfg = config
        self.quotas = quotas or QuotaManager(clock=clock)
        self.quotas.define("llm_requests", config.requests_per_min, 60.0)
        self.quotas.define("llm_tokens", config.tokens_per_day, 86400.0)
        self.transport = transport or default_transport
        self._profile_fn = profile_fn
        self._clock = clock
        self._audit = audit or (lambda t, f: None)
        self._sem = threading.BoundedSemaphore(max(1, config.max_concurrency))
        self._fail = 0
        self._open_until = 0.0
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "denied": 0, "errors": 0, "tokens_reported": 0}

    # -- policy
    def _hw(self) -> Dict[str, Any]:
        if self._profile_fn:
            try:
                return self._profile_fn()
            except Exception:  # noqa: BLE001
                pass
        return {"logical_cores": os.cpu_count() or 1, "ram_gb": 0.0}

    def high_performance_device(self) -> bool:
        hw = self._hw()
        return hw.get("logical_cores", 0) >= self.cfg.min_logical_cores and hw.get("ram_gb", 0.0) >= self.cfg.min_ram_gb

    def debug_mode(self) -> bool:
        return self.cfg.debug or os.environ.get("KRYSTAL_DEBUG", "") not in ("", "0", "false", "False")

    def policy(self) -> Dict[str, Any]:
        hw, dbg, perf = self._hw(), self.debug_mode(), self.high_performance_device()
        reasons = []
        if not self.cfg.enabled:
            reasons.append("disabled in config")
        if not (dbg or perf):
            reasons.append(f"neither debug mode nor high-performance device (need >= {self.cfg.min_logical_cores} logical cores and "
                           f">= {self.cfg.min_ram_gb} GB RAM; this host: {hw.get('logical_cores')} / {hw.get('ram_gb')} GB)")
        if not self.cfg.model:
            reasons.append("no model configured")
        if not is_private_endpoint(self.cfg.base_url) and not self.cfg.allow_remote:
            reasons.append("remote endpoint requires allow_remote=true")
        scheme = urllib.parse.urlparse(self.cfg.base_url).scheme
        if scheme not in ("http", "https"):
            reasons.append("base_url must be http(s)")
        return {"allowed": not reasons, "reasons": reasons, "debug_mode": dbg, "high_performance_device": perf,
                "endpoint_private": is_private_endpoint(self.cfg.base_url), "quotas": self.quotas.snapshot()}

    # -- core call
    def chat(self, messages: List[Dict[str, str]], purpose: str = "general", max_tokens: Optional[int] = None) -> Dict[str, Any]:
        pol = self.policy()
        if not pol["allowed"]:
            self.stats["denied"] += 1
            self._audit("llm_denied", {"purpose": purpose, "reasons": pol["reasons"]})
            raise GatewayDenied("policy_denied", "; ".join(pol["reasons"]))
        now = self._clock()
        with self._lock:
            if now < self._open_until:
                raise GatewayDenied("breaker_open", "LLM endpoint temporarily disabled after repeated failures", self._open_until - now)
        msgs = [{"role": m["role"], "content": redact(str(m["content"]))} for m in messages]
        prompt_chars = sum(len(m["content"]) for m in msgs)
        if prompt_chars > self.cfg.max_prompt_chars:
            raise GatewayDenied("prompt_too_large", f"prompt has {prompt_chars} chars; limit is {self.cfg.max_prompt_chars}")
        out_cap = min(max_tokens or self.cfg.max_output_tokens, self.cfg.max_output_tokens)
        reserve = sum(estimate_tokens(m["content"]) for m in msgs) + out_cap
        ok, retry = self.quotas.try_consume("llm_requests", 1)
        if not ok:
            self.stats["denied"] += 1
            raise GatewayDenied("rate_limited", "requests/minute quota exhausted", retry)
        ok, retry = self.quotas.try_consume("llm_tokens", reserve)
        if not ok:
            self.quotas.refund("llm_requests", 1)
            self.stats["denied"] += 1
            raise GatewayDenied("token_budget_exhausted", "daily token quota exhausted", retry)
        if not self._sem.acquire(timeout=1.0):
            self.quotas.refund("llm_requests", 1)
            self.quotas.refund("llm_tokens", reserve)
            raise GatewayDenied("busy", "another LLM call is in flight (max_concurrency)", 1.0)
        try:
            headers = {"Content-Type": "application/json"}
            key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""
            if key:
                headers["Authorization"] = "Bearer " + key
            body = json.dumps({"model": self.cfg.model, "messages": msgs, "max_tokens": out_cap, "temperature": 0.2, "stream": False}).encode("utf-8")
            url = self.cfg.base_url.rstrip("/") + "/chat/completions"
            self.stats["calls"] += 1
            try:
                status, _h, raw = self.transport(url, headers, body, self.cfg.timeout_s)
            except Exception as e:  # noqa: BLE001
                self._failure()
                self.quotas.refund("llm_tokens", reserve)
                raise GatewayDenied("transport_error", f"{type(e).__name__} contacting endpoint") from None
            if status != 200:
                self._failure()
                self.quotas.refund("llm_tokens", reserve)
                raise GatewayDenied("upstream_error", f"endpoint returned HTTP {status}")
            try:
                data = json.loads(raw.decode("utf-8"))
                text = data["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError):
                self._failure()
                self.quotas.refund("llm_tokens", reserve)
                raise GatewayDenied("bad_response", "endpoint did not return an OpenAI-style chat completion") from None
            with self._lock:
                self._fail = 0
            used = (data.get("usage") or {}).get("total_tokens")
            if isinstance(used, int) and used >= 0:
                self.quotas.refund("llm_tokens", max(0, reserve - used))
                self.stats["tokens_reported"] += used
            self._audit("llm_call", {"purpose": purpose, "model": self.cfg.model, "tokens": used, "prompt_chars": prompt_chars})
            return {"text": text, "model": data.get("model", self.cfg.model), "usage": data.get("usage"), "purpose": purpose}
        finally:
            self._sem.release()

    def _failure(self) -> None:
        self.stats["errors"] += 1
        with self._lock:
            self._fail += 1
            if self._fail >= self.cfg.breaker_failures:
                self._open_until = self._clock() + self.cfg.breaker_cooldown_s
                self._fail = 0

    # -- documentation helpers
    def _stamp(self, text: str, model: str, facts: Dict[str, Any]) -> str:
        head = f"> AI-generated by `{model}` via Krystal LLM gateway. **Unverified** - check against the code/data before relying on it."
        if facts:
            head += "\n> Locally computed facts (ground truth): `" + json.dumps(facts, sort_keys=True) + "`"
        return head + "\n\n" + text.strip() + "\n"

    def document_math(self, expression_or_code: str, context: str = "") -> Dict[str, Any]:
        facts: Dict[str, Any] = {}
        try:
            facts["numeric_value"] = safe_eval_math(expression_or_code.strip())
        except ValueError:
            pass
        sys_p = ("You document mathematical operations for engineers. Be precise and concise. Use Markdown with LaTeX. "
                 "State assumptions, domain/range, numeric stability concerns and a worked example. "
                 "If locally computed facts are provided, they are ground truth: never contradict them.")
        user = f"Document this operation:\n```\n{expression_or_code}\n```\n" + (f"Context: {context}\n" if context else "") + \
               (f"Ground truth: {json.dumps(facts)}\n" if facts else "")
        r = self.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}], "document_math")
        r["markdown"] = self._stamp(r["text"], r["model"], facts)
        r["facts"] = facts
        return r

    def document_binary_log(self, data: bytes, janet_context: str = "") -> Dict[str, Any]:
        facts = analyze_binary_log(data)
        dump = hexdump(data, 256)
        sys_p = ("You reverse-document binary log formats produced by Janet programs (marshal images, buffers, packed structs). "
                 "Propose a field layout as a table (offset, size, type, meaning, confidence). Mark every guess as a hypothesis. "
                 "Do not invent fields the data cannot support. Local analysis is ground truth.")
        user = f"Local analysis: {json.dumps(facts)}\nHexdump (first 256 bytes):\n{dump}\n" + (f"Janet context:\n{janet_context}\n" if janet_context else "")
        r = self.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}], "document_binary_log")
        r["markdown"] = self._stamp(r["text"], r["model"], facts)
        r["facts"] = facts
        return r

    def document_janet(self, source: str) -> Dict[str, Any]:
        sys_p = ("You write concise docstrings and a short Markdown explanation for Janet (the Lisp-like language) code. "
                 "Janet docstrings go right after the function name. Do not claim behaviour you cannot see in the code.")
        r = self.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": f"```janet\n{source}\n```"}], "document_janet")
        r["markdown"] = self._stamp(r["text"], r["model"], {})
        r["facts"] = {}
        return r
