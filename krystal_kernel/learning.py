"""Learning loop: event log -> pattern miner -> routing prior -> better events.

LaneBandit   epsilon-greedy routing per (kind, cost-bucket) over measured service time.
PatternMiner turns raw task events into named, evidence-backed patterns.
Learner      wires them to the event log. Mined `lane_crossover` patterns become *priors* for cost
             buckets that have not been explored yet - that is the amplification: knowledge learned
             at one size transfers to neighbouring sizes without paying exploration cost again.
"""
from __future__ import annotations

import json
import os
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .eventlog import EventLog
from .params import KernelParams


def cost_bucket(cost: float) -> int:
    """log2 bucket of an item-count style cost hint (0 for <=1)."""
    return max(0, int(max(1.0, float(cost))).bit_length() - 1)


class _Stat:
    __slots__ = ("n", "ewma_ms", "miss", "err", "sum_ms")

    def __init__(self):
        self.n = 0
        self.ewma_ms = 0.0
        self.miss = 0.0   # EWMA of miss indicator
        self.err = 0.0    # EWMA of error indicator
        self.sum_ms = 0.0

    def add(self, ms: float, ok: bool, missed: bool, a: float = 0.2):
        self.n += 1
        self.sum_ms += ms
        self.ewma_ms = ms if self.n == 1 else (1 - a) * self.ewma_ms + a * ms
        self.miss = (1 - a) * self.miss + a * (1.0 if missed else 0.0)
        self.err = (1 - a) * self.err + a * (0.0 if ok else 1.0)

    @property
    def mean_ms(self) -> float:
        return self.sum_ms / self.n if self.n else 0.0

    def score(self) -> float:
        # lower is better: latency, inflated by deadline misses, with a heavy penalty for errors
        return self.ewma_ms * (1.0 + 4.0 * self.miss) * (1.0 + 10.0 * self.err)


class LaneBandit:
    def __init__(self, params: KernelParams, rng: Optional[random.Random] = None, min_explore: int = 3):
        self.p = params
        self.rng = rng or random.Random(1337)
        self.min_explore = min_explore
        self.epsilon = params.epsilon
        self.stats: Dict[Tuple[str, int, str], _Stat] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}  # kind -> {"min_bucket": b, "lane": name, "below": other}
        self._lock = threading.Lock()
        self.decisions = {"explore": 0, "exploit": 0, "prior": 0}

    def update(self, kind: str, bucket: int, lane: str, service_ms: float, ok: bool, missed: bool) -> None:
        with self._lock:
            self.stats.setdefault((kind, bucket, lane), _Stat()).add(service_ms, ok, missed)

    def set_rule(self, kind: str, min_bucket: int, lane: str, below: str) -> None:
        with self._lock:
            self.rules[kind] = {"min_bucket": min_bucket, "lane": lane, "below": below}

    def advise(self, kind: str, bucket: int, allowed: List[str]) -> str:
        if len(allowed) == 1:
            return allowed[0]
        with self._lock:
            st = {l: self.stats.get((kind, bucket, l)) for l in allowed}
            cold = [l for l in allowed if st[l] is None or st[l].n < self.min_explore]
            rule = self.rules.get(kind)
            if cold:
                # a mined prior beats blind exploration for a cold bucket
                if rule and rule["lane"] in allowed and rule["below"] in allowed:
                    self.decisions["prior"] += 1
                    return rule["lane"] if bucket >= rule["min_bucket"] else rule["below"]
                self.decisions["explore"] += 1
                return min(cold, key=lambda l: (st[l].n if st[l] else 0, l))
            if self.rng.random() < self.epsilon:
                self.decisions["explore"] += 1
                self.epsilon = max(self.p.epsilon_floor, self.epsilon * self.p.epsilon_decay)
                return self.rng.choice(allowed)
            self.epsilon = max(self.p.epsilon_floor, self.epsilon * self.p.epsilon_decay)
            self.decisions["exploit"] += 1
            return min(allowed, key=lambda l: st[l].score())

    def table(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{"kind": k, "bucket": b, "lane": l, "n": s.n, "ewma_ms": round(s.ewma_ms, 3), "mean_ms": round(s.mean_ms, 3),
                     "miss_rate": round(s.miss, 3), "err_rate": round(s.err, 3)}
                    for (k, b, l), s in sorted(self.stats.items())]

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"epsilon": self.epsilon, "rules": self.rules,
                    "stats": [[k, b, l, s.n, s.ewma_ms, s.miss, s.err, s.sum_ms] for (k, b, l), s in self.stats.items()]}

    def load(self, d: Dict[str, Any]) -> None:
        with self._lock:
            self.epsilon = float(d.get("epsilon", self.epsilon))
            self.rules = dict(d.get("rules", {}))
            for k, b, l, n, ew, miss, err, sm in d.get("stats", []):
                s = _Stat()
                s.n, s.ewma_ms, s.miss, s.err, s.sum_ms = int(n), float(ew), float(miss), float(err), float(sm)
                self.stats[(k, int(b), l)] = s


class PatternMiner:
    def __init__(self, params: KernelParams):
        self.p = params
        self._lock = threading.Lock()
        self.agg: Dict[Tuple[str, int, str], _Stat] = {}
        self.queue_share: Dict[str, List[float]] = {}  # kind -> [sum_queue_ms, sum_total_ms]
        self.known: Dict[str, Dict[str, Any]] = {}

    def observe(self, ev: Dict[str, Any]) -> None:
        if ev.get("type") != "task_done":
            return
        with self._lock:
            self.agg.setdefault((ev["kind"], ev["bucket"], ev["lane"]), _Stat()).add(
                ev["service_ms"], ev["ok"], ev.get("missed", False))
            qs = self.queue_share.setdefault(ev["kind"], [0.0, 0.0])
            qs[0] += ev.get("queue_ms", 0.0)
            qs[1] += ev.get("latency_ms", 0.0)

    def mine(self) -> List[Dict[str, Any]]:
        """Return the *current* pattern set (stable ids, so callers can diff for novelty)."""
        ms = self.p.min_samples_for_pattern
        out: List[Dict[str, Any]] = []
        with self._lock:
            agg = dict(self.agg)
            qshare = {k: list(v) for k, v in self.queue_share.items()}
        kinds = sorted({k for (k, _, _) in agg})
        for kind in kinds:
            buckets = sorted({b for (k, b, _) in agg if k == kind})
            lanes = sorted({l for (k, _, l) in agg if k == kind})
            # 1) lane crossover: smallest bucket from which `process` is clearly faster than `thread`
            if "thread" in lanes and "process" in lanes:
                cross = None
                for b in buckets:
                    t, pr = agg.get((kind, b, "thread")), agg.get((kind, b, "process"))
                    if t and pr and t.n >= ms and pr.n >= ms and pr.mean_ms < 0.8 * t.mean_ms:
                        cross = b
                        break
                if cross is not None:
                    t, pr = agg[(kind, cross, "thread")], agg[(kind, cross, "process")]
                    out.append({"id": f"lane_crossover:{kind}", "kind": "lane_crossover", "subject": kind, "min_bucket": cross,
                                "min_cost": 2 ** cross, "faster_lane": "process", "below_lane": "thread",
                                "evidence": {"thread_mean_ms": round(t.mean_ms, 3), "process_mean_ms": round(pr.mean_ms, 3), "n": [t.n, pr.n]},
                                "rule": f"{kind}: cost >= {2 ** cross} items -> process lane; smaller -> thread lane"})
            # 2) lane dominance per bucket (only reported when no crossover rule explains it)
            for b in buckets:
                cands = [(l, agg[(kind, b, l)]) for l in lanes if (kind, b, l) in agg and agg[(kind, b, l)].n >= ms]
                if len(cands) >= 2:
                    cands.sort(key=lambda x: x[1].mean_ms)
                    best, worst = cands[0], cands[-1]
                    if best[1].mean_ms > 0 and worst[1].mean_ms / best[1].mean_ms >= 1.5:
                        out.append({"id": f"lane_dominance:{kind}:{b}", "kind": "lane_dominance", "subject": kind, "bucket": b,
                                    "best_lane": best[0], "ratio": round(worst[1].mean_ms / best[1].mean_ms, 2),
                                    "evidence": {l: round(s.mean_ms, 3) for l, s in cands},
                                    "rule": f"{kind}[cost~{2 ** b}]: {best[0]} is {worst[1].mean_ms / best[1].mean_ms:.1f}x faster than {worst[0]}"})
            # 3) deadline risk and 4) error hotspots
            for (k, b, l), s in agg.items():
                if k != kind or s.n < ms:
                    continue
                if s.miss > 0.10:
                    out.append({"id": f"deadline_risk:{kind}:{b}:{l}", "kind": "deadline_risk", "subject": kind, "bucket": b, "lane": l,
                                "miss_rate": round(s.miss, 3), "rule": f"{kind}[cost~{2 ** b}] on {l} misses deadlines ~{s.miss * 100:.0f}% of the time"})
                if s.err > 0.20:
                    out.append({"id": f"error_hotspot:{kind}:{l}", "kind": "error_hotspot", "subject": kind, "lane": l,
                                "err_rate": round(s.err, 3), "rule": f"{kind} on {l} fails ~{s.err * 100:.0f}% of requests"})
            # 5) capacity-bound: most of end-to-end latency is queueing, not service
            q = qshare.get(kind)
            n_total = sum(s.n for (k, _, _), s in agg.items() if k == kind)
            if q and q[1] > 0 and n_total >= ms and q[0] / q[1] > 0.5:
                out.append({"id": f"capacity_bound:{kind}", "kind": "capacity_bound", "subject": kind, "queue_share": round(q[0] / q[1], 3),
                            "rule": f"{kind}: {q[0] / q[1] * 100:.0f}% of latency is queue wait -> add lane capacity or shed load earlier"})
        return out


class Learner:
    """Subscribes to the event log; periodically mines patterns and feeds them back."""

    def __init__(self, log: EventLog, params: KernelParams, bandit: Optional[LaneBandit] = None, mine_every: int = 25):
        self.log, self.p = log, params
        self.bandit = bandit or LaneBandit(params)
        self.miner = PatternMiner(params)
        self.mine_every = mine_every
        self._n = 0
        self._announced: Dict[str, str] = {}
        self._lock = threading.Lock()
        log.subscribe(self._on_event)

    def _on_event(self, ev: Dict[str, Any]) -> None:
        if ev["type"] != "task_done":
            return
        self.bandit.update(ev["kind"], ev["bucket"], ev["lane"], ev["service_ms"], ev["ok"], ev.get("missed", False))
        self.miner.observe(ev)
        with self._lock:
            self._n += 1
            due = self._n % self.mine_every == 0
        if due:
            self.learn()

    def learn(self) -> List[Dict[str, Any]]:
        pats = self.miner.mine()
        for pat in pats:
            sig = json.dumps(pat.get("rule", ""), sort_keys=True)
            if self._announced.get(pat["id"]) == sig:
                continue
            self._announced[pat["id"]] = sig
            if pat["kind"] == "lane_crossover":
                self.bandit.set_rule(pat["subject"], pat["min_bucket"], pat["faster_lane"], pat["below_lane"])
            self.log.emit("pattern", pattern_id=pat["id"], pattern_kind=pat["kind"], rule=pat["rule"])
        return pats

    def patterns(self) -> List[Dict[str, Any]]:
        return self.miner.mine()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.bandit.to_dict(), f)

    def load(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.bandit.load(json.load(f))
            return True
        except (OSError, ValueError):
            return False
