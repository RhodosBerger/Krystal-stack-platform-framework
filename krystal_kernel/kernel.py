"""ComputeKernel: EDF scheduler, admission control, back-pressure, lane routing and metrics.

Design rules that keep the kernel from hurting the program it serves:
  * bounded queue (Little's law sizing) - overload becomes an immediate, explicit rejection
  * early shedding - a request whose predicted wait already exceeds its deadline is refused
  * expired work is dropped, never executed
  * AIMD per lane driven by *service* time (queue wait would shrink capacity exactly when it is needed)
  * a crashed process worker retries once on the thread lane instead of failing the caller
"""
from __future__ import annotations

import bisect
import itertools
import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Any, Deque, Dict, List, Optional

from .eventlog import EventLog
from .hwprofile import HardwareProfile, REPO_ROOT, detect_profile
from .kernels import REGISTRY
from .lanes import AIMD, Lane, ProcessLane, ThreadLane, WorkerCrashed
from .learning import Learner, cost_bucket
from .params import KernelParams, derive_params


class Backpressure(Exception):
    def __init__(self, reason: str, retry_after_s: float):
        super().__init__(f"{reason} (retry after {retry_after_s:.2f}s)")
        self.reason, self.retry_after_s = reason, retry_after_s


class DeadlineMissed(Exception):
    pass


class _Item:
    __slots__ = ("seq", "kernel", "payload", "kind", "cost", "bucket", "deadline_ms", "deadline_abs", "allowed",
                 "outer", "t_submit", "advised", "retried", "tenant")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.retried = False
        self.advised = None

    def key(self):
        return (self.deadline_abs if self.deadline_abs is not None else float("inf"), self.seq)


class ComputeKernel:
    SCAN_WINDOW = 64
    SPILL_AFTER_S = 0.005

    def __init__(self, params: Optional[KernelParams] = None, profile: Optional[HardwareProfile] = None,
                 log: Optional[EventLog] = None, lanes: Optional[Dict[str, Lane]] = None,
                 persist_dir: Optional[str] = None, learner: Optional[Learner] = None):
        self.profile = profile or detect_profile()
        self.params = params or derive_params(self.profile)
        self.persist_dir = persist_dir
        self.log = log or EventLog(path=os.path.join(persist_dir, "kernel_events.jsonl") if persist_dir else None)
        if lanes is None:
            p = self.params
            lanes = {"thread": ThreadLane(p.thread_workers, AIMD(1, p.thread_workers, min(2, p.thread_workers)))}
            lanes["process"] = ProcessLane(p.process_workers, AIMD(1, p.process_workers, p.process_workers))
        self.lanes: Dict[str, Lane] = lanes
        self.learner = learner or Learner(self.log, self.params)
        if persist_dir:
            self.learner.load(os.path.join(persist_dir, "kernel_learned.json"))

        self._q: List[_Item] = []
        self._keys: List[Any] = []
        self._cv = threading.Condition()
        self._seq = itertools.count(1)
        self._stop = threading.Event()
        self.counters: Dict[str, int] = {"submitted": 0, "completed": 0, "rejected": 0, "expired": 0, "errors": 0, "retried": 0}
        self._svc_ewma = self.params.expected_service_ms
        self._lat: Dict[str, Deque[float]] = {}
        self._base: Dict[Any, Deque[float]] = {}
        self.last_completion = time.monotonic()
        self.health: Dict[str, Any] = {"state": "ok", "reasons": []}
        self.backends = None  # set by the API layer / supervisor if present
        self._dispatcher: Optional[threading.Thread] = None
        self.dispatcher_restarts = 0
        self._start_dispatcher()

    # ------------------------------------------------------------------ lifecycle
    def _start_dispatcher(self) -> None:
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True, name="kk-dispatch")
        self._dispatcher.start()

    def restart_dispatcher(self) -> None:
        self.dispatcher_restarts += 1
        self._start_dispatcher()  # the old loop exits via its generation check
        with self._cv:
            self._cv.notify_all()

    def dispatcher_alive(self) -> bool:
        return bool(self._dispatcher and self._dispatcher.is_alive())

    def warm(self) -> None:
        for lane in self.lanes.values():
            lane.warm()

    def shutdown(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self.persist_dir:
            try:
                self.learner.save(os.path.join(self.persist_dir, "kernel_learned.json"))
            except OSError:
                pass
        for lane in self.lanes.values():
            lane.close()

    # ------------------------------------------------------------------ submit
    @property
    def queue_len(self) -> int:
        return len(self._q)

    def _total_slots(self) -> int:
        return max(1, sum(l.aimd.value for l in self.lanes.values()))

    def estimated_wait_ms(self) -> float:
        backlog = len(self._q) + sum(l.inflight for l in self.lanes.values())
        return backlog * self._svc_ewma / self._total_slots()

    def submit(self, kernel: str, payload: Dict[str, Any], *, kind: Optional[str] = None, cost: float = 1.0,
               deadline_ms: Optional[float] = None, lanes: Optional[List[str]] = None, tenant: str = "default") -> Future:
        if kernel not in REGISTRY:
            raise KeyError(f"unknown kernel '{kernel}'")
        allowed = [l for l in (lanes or list(self.lanes)) if l in self.lanes]
        if not allowed:
            raise ValueError("no usable lane")
        kind = kind or kernel
        now = time.monotonic()
        with self._cv:
            if len(self._q) >= self.params.max_queue:
                self._reject(kind, "queue_full", tenant)
                raise Backpressure("queue_full", max(0.05, self.estimated_wait_ms() / 1000.0))
            wait_ms = self.estimated_wait_ms()
            if deadline_ms is not None and wait_ms > deadline_ms:
                self._reject(kind, "deadline_unreachable", tenant)
                raise Backpressure("deadline_unreachable", max(0.05, (wait_ms - deadline_ms) / 1000.0))
            item = _Item(seq=next(self._seq), kernel=kernel, payload=payload, kind=kind, cost=cost, bucket=cost_bucket(cost),
                         deadline_ms=deadline_ms, deadline_abs=(now + deadline_ms / 1000.0) if deadline_ms else None,
                         allowed=allowed, outer=Future(), t_submit=now, tenant=tenant)
            self._insert(item)
            self.counters["submitted"] += 1
            self._cv.notify()
        return item.outer

    def _insert(self, item: _Item) -> None:
        k = item.key()
        i = bisect.bisect_right(self._keys, k)
        self._keys.insert(i, k)
        self._q.insert(i, item)

    def _reject(self, kind: str, reason: str, tenant: str) -> None:
        self.counters["rejected"] += 1
        self.log.emit("task_rejected", kind=kind, reason=reason, tenant=tenant, queue=len(self._q))

    # ------------------------------------------------------------------ dispatch
    def _pick(self):
        now = time.monotonic()
        for idx in range(min(self.SCAN_WINDOW, len(self._q))):
            item = self._q[idx]
            free = [l for l in item.allowed if self.lanes[l].has_slot()]
            if not free:
                continue
            if item.advised is None:
                item.advised = self.learner.bandit.advise(item.kind, item.bucket, item.allowed)
            if item.advised in free:
                lane = item.advised
            elif now - item.t_submit >= self.SPILL_AFTER_S:
                lane = free[0]
            else:
                continue
            del self._q[idx]
            del self._keys[idx]
            self.lanes[lane]._enter()
            return item, lane
        return None

    def _dispatch_loop(self) -> None:
        me = threading.current_thread()
        while not self._stop.is_set() and self._dispatcher is me:
            with self._cv:
                picked = self._pick()
                if picked is None:
                    self._cv.wait(0.01)
                    continue
            self._launch(*picked)

    def _launch(self, item: _Item, lane_name: str) -> None:
        lane = self.lanes[lane_name]
        now = time.monotonic()
        if item.deadline_abs is not None and now > item.deadline_abs:
            lane._exit(True)
            lane.completed -= 1  # not real work
            self.counters["expired"] += 1
            self.log.emit("task_expired", kind=item.kind, lane=lane_name, waited_ms=round((now - item.t_submit) * 1000, 2))
            item.outer.set_exception(DeadlineMissed(f"{item.kind}: deadline passed while queued"))
            return
        t_start = time.monotonic()
        try:
            inner = lane.submit_raw(item.kernel, item.payload)
        except Exception as e:  # noqa: BLE001
            inner = Future()
            inner.set_exception(e)
        inner.add_done_callback(lambda f, it=item, ln=lane_name, ts=t_start: self._done(it, ln, f, ts))

    def _done(self, item: _Item, lane_name: str, fut: Future, t_start: float) -> None:
        lane = self.lanes[lane_name]
        t_end = time.monotonic()
        exc = fut.exception()
        ok = exc is None
        lane._exit(ok)
        service_ms = (t_end - t_start) * 1000.0
        if isinstance(exc, WorkerCrashed) and not item.retried and "thread" in self.lanes and lane_name != "thread":
            item.retried = True
            item.allowed = ["thread"]
            item.advised = "thread"
            self.counters["retried"] += 1
            self.log.emit("task_retry", kind=item.kind, from_lane=lane_name, to_lane="thread", error=str(exc))
            with self._cv:
                self._insert(item)
                self._cv.notify()
            return
        latency_ms = (t_end - item.t_submit) * 1000.0
        missed = item.deadline_ms is not None and latency_ms > item.deadline_ms
        self.counters["completed" if ok else "errors"] += 1
        self.last_completion = t_end
        self._svc_ewma = 0.9 * self._svc_ewma + 0.1 * service_ms
        self._lat.setdefault(item.kind, deque(maxlen=512)).append(latency_ms)
        # Concurrency feedback. Judge service time against the *uncontended baseline* of the same
        # (kind, size-bucket, lane): if running N at once inflates it beyond `inflation_limit`x, the lane
        # is oversubscribed (GIL convoy, thermal throttling, SMT contention) and AIMD backs off.
        if ok:
            bk = (item.kind, item.bucket, lane_name)
            hist = self._base.setdefault(bk, deque(maxlen=64))
            hist.append(service_ms)
            inflated = len(hist) >= 8 and service_ms > self.params.inflation_limit * max(min(hist), 0.05)
            lane.aimd.on_signal(not (inflated or service_ms > self.params.latency_target_ms))
        else:
            lane.aimd.on_signal(False)
        self.log.emit("task_done", kind=item.kind, bucket=item.bucket, lane=lane_name, ok=ok, missed=missed,
                      service_ms=round(service_ms, 3), queue_ms=round(latency_ms - service_ms, 3), latency_ms=round(latency_ms, 3),
                      deadline_ms=item.deadline_ms, tenant=item.tenant, retried=item.retried)
        if ok:
            item.outer.set_result(fut.result())
        else:
            item.outer.set_exception(exc)
        with self._cv:
            self._cv.notify()

    # ------------------------------------------------------------------ introspection
    def percentile(self, kind: str, q: float) -> Optional[float]:
        d = self._lat.get(kind)
        if not d:
            return None
        s = sorted(d)
        return round(s[min(len(s) - 1, int(q * len(s)))], 3)

    def stats(self) -> Dict[str, Any]:
        kinds = {k: {"n": len(v), "p50_ms": self.percentile(k, 0.5), "p95_ms": self.percentile(k, 0.95)} for k, v in self._lat.items()}
        return {
            "health": self.health, "queue_len": self.queue_len, "max_queue": self.params.max_queue,
            "estimated_wait_ms": round(self.estimated_wait_ms(), 2), "service_ewma_ms": round(self._svc_ewma, 3),
            "counters": dict(self.counters), "lanes": {n: l.snapshot() for n, l in self.lanes.items()},
            "kinds": kinds, "dispatcher_restarts": self.dispatcher_restarts,
            "learning": {"epsilon": round(self.learner.bandit.epsilon, 4), "decisions": dict(self.learner.bandit.decisions),
                         "rules": self.learner.bandit.rules, "patterns": len(self.learner.patterns())},
        }

    def prometheus(self) -> str:
        s = self.stats()
        out = ["# TYPE krystal_tasks_total counter"]
        for k, v in s["counters"].items():
            out.append(f'krystal_tasks_total{{status="{k}"}} {v}')
        out += ["# TYPE krystal_queue_depth gauge", f"krystal_queue_depth {s['queue_len']}",
                "# TYPE krystal_estimated_wait_ms gauge", f"krystal_estimated_wait_ms {s['estimated_wait_ms']}",
                "# TYPE krystal_lane_inflight gauge", "# TYPE krystal_lane_limit gauge", "# TYPE krystal_lane_restarts_total counter"]
        for n, l in s["lanes"].items():
            out += [f'krystal_lane_inflight{{lane="{n}"}} {l["inflight"]}', f'krystal_lane_limit{{lane="{n}"}} {l["limit"]}',
                    f'krystal_lane_restarts_total{{lane="{n}"}} {l["restarts"]}']
        out.append("# TYPE krystal_latency_ms gauge")
        for k, v in s["kinds"].items():
            for q, key in (("0.5", "p50_ms"), ("0.95", "p95_ms")):
                if v[key] is not None:
                    out.append(f'krystal_latency_ms{{kind="{k}",quantile="{q}"}} {v[key]}')
        state = {"ok": 0, "degraded": 1, "critical": 2}.get(self.health["state"], 2)
        out += ["# TYPE krystal_kernel_health gauge", f"krystal_kernel_health {state}"]
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------- singleton
_kernel: Optional[ComputeKernel] = None
_klock = threading.Lock()


def get_kernel() -> ComputeKernel:
    global _kernel
    with _klock:
        if _kernel is None:
            from .healing import Supervisor
            k = ComputeKernel(persist_dir=os.path.join(REPO_ROOT, "logs"))
            k.supervisor = Supervisor(k)  # type: ignore[attr-defined]
            k.supervisor.start()  # type: ignore[attr-defined]
            _kernel = k
        return _kernel


def shutdown_kernel() -> None:
    global _kernel
    with _klock:
        if _kernel is not None:
            sup = getattr(_kernel, "supervisor", None)
            if sup:
                sup.stop()
            _kernel.shutdown()
            _kernel = None
