"""Execution lanes: where a named kernel actually runs.

ThreadLane   - in-process thread pool. Cheap dispatch, but the GIL serialises CPU-bound Python
               (measured: ~1.1x on 8 threads for a pure-Python spin loop).
ProcessLane  - N long-lived worker subprocesses speaking a length-prefixed pickle protocol.
               True parallelism; pays an IPC round-trip (measured ~40 us p50 on this host).
Both lanes carry an AIMD concurrency limiter so the *in-flight* count adapts to latency.
"""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .kernels import REGISTRY

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WorkerCrashed(RuntimeError):
    pass


class AIMD:
    """Additive-increase / multiplicative-decrease limiter driven by observed latency.

    limit += alpha/limit   when latency <= target   (~ +alpha per `limit` completions)
    limit *= beta          when latency >  target   (at most once per `cooldown_s`)
    """

    def __init__(self, lo: int, hi: int, init: Optional[int] = None, alpha: float = 1.0, beta: float = 0.7,
                 cooldown_s: float = 0.25, clock: Callable[[], float] = time.monotonic):
        self.lo, self.hi = max(1, lo), max(max(1, lo), hi)
        self.limit = float(init if init is not None else self.hi)
        self.alpha, self.beta, self.cooldown_s, self._clock = alpha, beta, cooldown_s, clock
        self._last_dec = -1e9
        self.decreases = 0
        self.increases = 0

    @property
    def value(self) -> int:
        return max(self.lo, min(self.hi, int(self.limit)))

    def on_signal(self, good: bool) -> None:
        """good -> additive increase; congestion -> multiplicative decrease (rate limited by cooldown)."""
        if good:
            self.limit = min(float(self.hi), self.limit + self.alpha / max(1.0, self.limit))
            self.increases += 1
        else:
            now = self._clock()
            if now - self._last_dec >= self.cooldown_s:
                self.limit = max(float(self.lo), self.limit * self.beta)
                self._last_dec = now
                self.decreases += 1

    def on_sample(self, latency_ms: float, target_ms: float) -> None:
        self.on_signal(latency_ms <= target_ms)

    def set_hi(self, hi: int) -> None:
        self.hi = max(self.lo, hi)
        self.limit = min(self.limit, float(self.hi))


class Lane:
    name = "lane"
    aimd: AIMD

    def __init__(self):
        self._inflight = 0
        self._lock = threading.Lock()
        self.completed = 0
        self.errors = 0
        self.restarts = 0

    # slot accounting is done by the kernel dispatcher through these two calls
    def has_slot(self) -> bool:
        with self._lock:
            return self._inflight < self.aimd.value

    def _enter(self):
        with self._lock:
            self._inflight += 1

    def _exit(self, ok: bool):
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self.completed += 1
            if not ok:
                self.errors += 1

    @property
    def inflight(self) -> int:
        return self._inflight

    def reset_inflight(self) -> None:
        """After a lane restart, abandoned futures never complete; free their slots."""
        with self._lock:
            self._inflight = 0

    def submit_raw(self, kernel: str, payload: Dict[str, Any]) -> Future:  # pragma: no cover - abstract
        raise NotImplementedError

    def warm(self) -> None:
        pass

    def ensure_alive(self) -> int:
        """Repair dead resources; return how many were repaired."""
        return 0

    def restart(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name, "inflight": self._inflight, "limit": self.aimd.value, "limit_max": self.aimd.hi,
                "completed": self.completed, "errors": self.errors, "restarts": self.restarts,
                "aimd_increases": self.aimd.increases, "aimd_decreases": self.aimd.decreases}


class ThreadLane(Lane):
    name = "thread"

    def __init__(self, workers: int = 4, aimd: Optional[AIMD] = None):
        super().__init__()
        self.workers = max(1, workers)
        self.aimd = aimd or AIMD(1, self.workers, self.workers)
        self._ex = ThreadPoolExecutor(self.workers, thread_name_prefix="kk-thread")

    def submit_raw(self, kernel: str, payload: Dict[str, Any]) -> Future:
        fn = REGISTRY[kernel]
        return self._ex.submit(fn, payload)

    def restart(self) -> None:
        old, self._ex = self._ex, ThreadPoolExecutor(self.workers, thread_name_prefix="kk-thread")
        old.shutdown(wait=False, cancel_futures=True)
        self.restarts += 1

    def close(self) -> None:
        self._ex.shutdown(wait=False, cancel_futures=True)


class _Worker:
    """One worker subprocess. Every (re)start is a new *generation*; a stale reader thread from a
    previous generation must never mark the new process dead or fail its requests."""

    def __init__(self, idx: int):
        self.idx = idx
        self.proc: Optional[subprocess.Popen] = None
        self.pending: Dict[int, Future] = {}
        self.started: Dict[int, float] = {}
        self.wlock = threading.Lock()
        self.plock = threading.Lock()
        self.ready = threading.Event()
        self.dead = False
        self.reader: Optional[threading.Thread] = None
        self.gen = 0
        self._seq = 0

    def _fail_pending(self, reason: str) -> None:
        with self.plock:
            pend, self.pending, self.started = self.pending, {}, {}
        for fut in pend.values():
            if not fut.done():
                fut.set_exception(WorkerCrashed(reason))

    def start(self) -> None:
        old = self.proc
        if old is not None:
            self._dispose(old)
        self._fail_pending(f"worker {self.idx} restarted")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.gen += 1
        self.dead = False
        self.ready.clear()
        proc = subprocess.Popen(
            [sys.executable, "-m", "krystal_kernel.worker"], cwd=REPO_ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags, bufsize=0)
        self.proc = proc
        self.reader = threading.Thread(target=self._read_loop, args=(proc, self.gen), daemon=True, name=f"kk-reader-{self.idx}")
        self.reader.start()

    @staticmethod
    def _dispose(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.kill()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.wait(timeout=2.0)
        except Exception:
            pass

    @staticmethod
    def _read_exact(proc: subprocess.Popen, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = proc.stdout.read(n - len(buf))  # type: ignore[union-attr]
            if not chunk:
                return None
            buf += chunk
        return buf

    def _read_loop(self, proc: subprocess.Popen, gen: int) -> None:
        try:
            while True:
                hdr = self._read_exact(proc, 4)
                if hdr is None:
                    break
                (n,) = struct.unpack(">I", hdr)
                if n == 0:
                    if gen == self.gen:
                        self.ready.set()
                    continue
                body = self._read_exact(proc, n)
                if body is None:
                    break
                req_id, ok, val = pickle.loads(body)
                with self.plock:
                    fut = self.pending.pop(req_id, None) if gen == self.gen else None
                    self.started.pop(req_id, None)
                if fut is not None and not fut.done():
                    if ok:
                        fut.set_result(val)
                    else:
                        fut.set_exception(RuntimeError(val))
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass
            if gen == self.gen:
                self.dead = True
                self.ready.set()
                self._fail_pending(f"worker {self.idx} exited")

    def submit(self, kernel: str, payload: Dict[str, Any]) -> Future:
        fut: Future = Future()
        proc = self.proc
        if self.dead or proc is None or proc.poll() is not None:
            fut.set_exception(WorkerCrashed(f"worker {self.idx} is dead"))
            return fut
        with self.plock:
            self._seq += 1
            rid = self._seq
            self.pending[rid] = fut
            self.started[rid] = time.monotonic()
        data = pickle.dumps((rid, kernel, payload), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            with self.wlock:
                proc.stdin.write(struct.pack(">I", len(data)) + data)  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
        except Exception as e:  # broken pipe etc.
            with self.plock:
                self.pending.pop(rid, None)
                self.started.pop(rid, None)
            if not fut.done():
                fut.set_exception(WorkerCrashed(f"worker {self.idx} write failed: {e}"))
        return fut

    def oldest_age(self) -> float:
        with self.plock:
            return (time.monotonic() - min(self.started.values())) if self.started else 0.0

    def load(self) -> int:
        return len(self.pending)

    def kill(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        self.dead = True

    def close(self) -> None:
        if self.proc is not None:
            self._dispose(self.proc)
        self.dead = True


class ProcessLane(Lane):
    name = "process"

    def __init__(self, workers: int = 2, aimd: Optional[AIMD] = None, hang_timeout_s: float = 30.0):
        super().__init__()
        self.workers_n = max(1, workers)
        self.aimd = aimd or AIMD(1, self.workers_n, self.workers_n)
        self.hang_timeout_s = hang_timeout_s
        self._workers: List[_Worker] = []
        self._spawn_lock = threading.Lock()
        for i in range(self.workers_n):
            w = _Worker(i)
            w.start()
            self._workers.append(w)

    def warm(self, timeout: float = 15.0) -> None:
        for w in self._workers:
            w.ready.wait(timeout)

    def submit_raw(self, kernel: str, payload: Dict[str, Any]) -> Future:
        alive = [w for w in self._workers if not w.dead]
        if not alive:
            f: Future = Future()
            f.set_exception(WorkerCrashed("no live workers"))
            return f
        w = min(alive, key=lambda x: x.load())
        return w.submit(kernel, payload)

    def ensure_alive(self) -> int:
        repaired = 0
        with self._spawn_lock:
            for w in self._workers:
                hung = w.oldest_age() > self.hang_timeout_s
                if hung:
                    w.kill()
                if w.dead or (w.proc is not None and w.proc.poll() is not None):
                    w.start()
                    repaired += 1
        self.restarts += repaired
        return repaired

    def dead_workers(self) -> int:
        return sum(1 for w in self._workers if w.dead or (w.proc is not None and w.proc.poll() is not None))

    def hung_workers(self) -> int:
        return sum(1 for w in self._workers if w.oldest_age() > self.hang_timeout_s)

    def restart(self) -> None:
        with self._spawn_lock:
            for w in self._workers:
                w.kill()
                w.start()
        self.restarts += 1

    def close(self) -> None:
        for w in self._workers:
            w.close()

    def snapshot(self) -> Dict[str, Any]:
        s = super().snapshot()
        s.update({"workers": self.workers_n, "dead_workers": self.dead_workers()})
        return s
