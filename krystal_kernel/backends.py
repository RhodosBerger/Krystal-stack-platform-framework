"""Inference backends behind circuit breakers.

ReferenceBackend  always available; runs the `hash_embed` kernel through the ComputeKernel.
OpenVINOBackend   used only when the `openvino` package is importable; *unverified until selftest()
                  passes* on this host. It is intentionally generic (`infer_raw`) because a text
                  embedding needs a model-specific tokenizer that the caller must provide.
BackendRouter     tries backends in order, trips a breaker on repeated failure and falls back, so a
                  broken accelerator can never take the API down.
"""
from __future__ import annotations

import importlib.util
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .eventlog import EventLog


class CircuitBreaker:
    """closed -> open (error rate over a window) -> half_open (one probe after cooldown) -> closed."""

    def __init__(self, window: int = 20, error_rate: float = 0.5, cooldown_s: float = 10.0, min_calls: int = 5,
                 clock: Callable[[], float] = time.monotonic):
        self.window, self.error_rate, self.cooldown_s, self.min_calls, self._clock = window, error_rate, cooldown_s, min_calls, clock
        self.results: Deque[bool] = deque(maxlen=window)
        self.state = "closed"
        self.opened_at = 0.0
        self.trips = 0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open" and self._clock() - self.opened_at >= self.cooldown_s:
                self.state = "half_open"
                return True
            return self.state == "half_open"

    def record(self, ok: bool) -> Optional[str]:
        """Return a transition name ('opened'/'closed') if the state changed."""
        with self._lock:
            if self.state == "half_open":
                if ok:
                    self.state, self.results = "closed", deque(maxlen=self.window)
                    return "closed"
                self.state, self.opened_at = "open", self._clock()
                return "opened"
            self.results.append(ok)
            if self.state == "closed" and len(self.results) >= self.min_calls:
                fails = sum(1 for r in self.results if not r)
                if fails / len(self.results) >= self.error_rate:
                    self.state, self.opened_at = "open", self._clock()
                    self.trips += 1
                    return "opened"
        return None


class Backend:
    name = "backend"
    verified = False

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "verified": self.verified}


class ReferenceBackend(Backend):
    name = "reference-cpu"
    verified = True

    def __init__(self, kernel, deadline_ms: float = 5000.0):
        self.kernel, self.deadline_ms = kernel, deadline_ms

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        fut = self.kernel.submit("hash_embed", {"texts": list(texts)}, kind="embed", cost=len(texts), deadline_ms=self.deadline_ms)
        return fut.result(timeout=self.deadline_ms / 1000.0 + 5)


def openvino_available() -> bool:
    return importlib.util.find_spec("openvino") is not None


class OpenVINOBackend(Backend):
    """Thin, lazy OpenVINO wrapper. Not usable (and not tested) unless `openvino` is installed."""
    name = "openvino"

    def __init__(self, model_path: str, device: str = "AUTO", hint: str = "THROUGHPUT",
                 preprocess: Optional[Callable[[Sequence[str]], Dict[str, Any]]] = None,
                 postprocess: Optional[Callable[[Any], List[List[float]]]] = None):
        if not openvino_available():
            raise RuntimeError("openvino package is not installed")
        self.model_path, self.device, self.hint = model_path, device, hint
        self.preprocess, self.postprocess = preprocess, postprocess
        self._compiled = None
        self._lock = threading.Lock()
        self.verified = False

    def _compile(self):
        with self._lock:
            if self._compiled is None:
                import openvino as ov  # type: ignore
                core = ov.Core()
                self._compiled = core.compile_model(self.model_path, self.device, {"PERFORMANCE_HINT": self.hint})
        return self._compiled

    def infer_raw(self, feeds: Dict[str, Any]) -> Any:
        compiled = self._compile()
        return compiled(feeds)

    def selftest(self, feeds: Dict[str, Any]) -> bool:
        """Run one inference; marks the backend verified only if it returns without error."""
        try:
            self.infer_raw(feeds)
            self.verified = True
        except Exception:
            self.verified = False
        return self.verified

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not (self.preprocess and self.postprocess):
            raise NotImplementedError("provide preprocess/postprocess (tokenizer) for text embeddings")
        return self.postprocess(self.infer_raw(self.preprocess(texts)))


class BackendRouter:
    def __init__(self, backends: List[Backend], log: Optional[EventLog] = None, window: int = 20, error_rate: float = 0.5,
                 cooldown_s: float = 10.0, clock: Callable[[], float] = time.monotonic):
        self.backends = backends
        self.log = log
        self.breakers = {b.name: CircuitBreaker(window, error_rate, cooldown_s, clock=clock) for b in backends}
        self.last_used: Optional[str] = None
        self.fallbacks = 0

    def embed(self, texts: Sequence[str]) -> Tuple[List[List[float]], str]:
        last_err: Optional[Exception] = None
        for i, b in enumerate(self.backends):
            br = self.breakers[b.name]
            if not br.allow():
                continue
            try:
                vecs = b.embed(texts)
            except Exception as e:  # noqa: BLE001
                last_err = e
                tr = br.record(False)
                if self.log:
                    self.log.emit("backend_error", backend=b.name, error=f"{type(e).__name__}: {e}", breaker=br.state)
                    if tr == "opened":
                        self.log.emit("breaker_opened", backend=b.name)
                continue
            tr = br.record(True)
            if tr == "closed" and self.log:
                self.log.emit("breaker_closed", backend=b.name)
            if i > 0:
                self.fallbacks += 1
            self.last_used = b.name
            return vecs, b.name
        raise RuntimeError(f"all backends unavailable: {last_err}")

    def status(self) -> List[Dict[str, Any]]:
        return [{**b.info(), "breaker": self.breakers[b.name].state, "trips": self.breakers[b.name].trips} for b in self.backends]
