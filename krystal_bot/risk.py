"""Learning from the kernel's own logs: deadline-miss / failure risk for pending work.

Features are taken only from what is known *at dispatch time* (size bucket, lane, deadline, time
already spent queued) - never from the service time, which would leak the label.

Two interchangeable models:
  * `StdlibLogistic` - always available, pure Python.
  * `KerasClient`    - a small Keras network run in the isolated `.venv-keras` worker.

A model is only *accepted* if it beats the trivial baseline on a held-out split and had enough
positive examples; otherwise the transparent heuristic (fraction of the deadline already consumed) is
used and the report says so. No synthetic data is ever generated.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

FEATURES = ["bucket", "log_deadline", "log_queue_ms", "queue_over_deadline", "lane_thread", "lane_process", "lane_other", "retried"]
MIN_POSITIVES = 30
MIN_AUC = 0.60


def featurize(ev: Dict[str, Any]) -> Optional[List[float]]:
    """Dispatch-time features of a `task_done` event, or None if the event is unusable."""
    if ev.get("type") != "task_done":
        return None
    dl, lat, svc = ev.get("deadline_ms"), ev.get("latency_ms"), ev.get("service_ms")
    q = ev.get("queue_ms")
    if q is None and isinstance(lat, (int, float)) and isinstance(svc, (int, float)):
        q = lat - svc
    if not isinstance(q, (int, float)):
        return None
    dl_f = float(dl) if isinstance(dl, (int, float)) and dl > 0 else 0.0
    lane = ev.get("lane")
    return [float(ev.get("bucket") or 0), math.log1p(dl_f), math.log1p(max(0.0, float(q))),
            min(5.0, max(0.0, float(q)) / dl_f) if dl_f else 0.0,
            1.0 if lane == "thread" else 0.0, 1.0 if lane == "process" else 0.0, 1.0 if lane not in ("thread", "process") else 0.0,
            1.0 if ev.get("retried") else 0.0]


def label(ev: Dict[str, Any]) -> int:
    return 1 if (not ev.get("ok", True)) or ev.get("missed") else 0


def build_dataset(events: Sequence[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
    xs, ys = [], []
    for ev in events:
        f = featurize(ev)
        if f is not None:
            xs.append(f)
            ys.append(label(ev))
    return xs, ys


def heuristic_risk(x: Sequence[float]) -> float:
    """Transparent fallback: how much of the deadline was already burnt waiting in the queue."""
    return min(1.0, max(0.0, x[3]))


def auc(y: Sequence[int], p: Sequence[float]) -> Optional[float]:
    pos = [s for s, t in zip(p, y) if t == 1]
    neg = [s for s, t in zip(p, y) if t == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


class StdlibLogistic:
    name = "stdlib-logistic"

    def __init__(self) -> None:
        self.w: List[float] = []
        self.b = 0.0
        self.mu: List[float] = []
        self.sd: List[float] = []

    def _norm(self, x: Sequence[float]) -> List[float]:
        return [(v - m) / s for v, m, s in zip(x, self.mu, self.sd)]

    def fit(self, xs: List[List[float]], ys: List[int], epochs: int = 200, lr: float = 0.1, seed: int = 7) -> None:
        n, d = len(xs), len(xs[0])
        self.mu = [sum(r[j] for r in xs) / n for j in range(d)]
        self.sd = [math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in xs) / n) or 1.0 for j in range(d)]
        data = [(self._norm(x), y) for x, y in zip(xs, ys)]
        pos = max(1, sum(ys))
        neg = max(1, n - pos)
        cw = {0: n / (2 * neg), 1: n / (2 * pos)}
        rnd = random.Random(seed)
        self.w, self.b = [0.0] * d, 0.0
        for ep in range(epochs):
            rnd.shuffle(data)
            step = lr / (1 + 0.02 * ep)
            for x, y in data:
                z = self.b + sum(w * v for w, v in zip(self.w, x))
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                g = (p - y) * cw[y]
                for j in range(d):
                    self.w[j] -= step * (g * x[j] + 1e-3 * self.w[j])
                self.b -= step * g

    def predict(self, xs: Sequence[Sequence[float]]) -> List[float]:
        out = []
        for x in xs:
            z = self.b + sum(w * v for w, v in zip(self.w, self._norm(x)))
            out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
        return out

    def to_json(self) -> Dict[str, Any]:
        return {"w": self.w, "b": self.b, "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "StdlibLogistic":
        m = cls()
        m.w, m.b, m.mu, m.sd = d["w"], d["b"], d["mu"], d["sd"]
        return m


class KerasClient:
    """Talks JSON lines to `krystal_bot.keras_worker` running in an isolated venv."""
    name = "keras"

    def __init__(self, python_exe: str, repo_root: str, model_path: str, threads: int = 2, timeout_s: float = 180.0):
        self.python_exe, self.repo_root, self.model_path = python_exe, repo_root, model_path
        self.threads, self.timeout_s = threads, timeout_s
        self._p: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.info: Dict[str, Any] = {}

    @staticmethod
    def find_python(repo_root: str) -> Optional[str]:
        cand = os.path.join(repo_root, ".venv-keras", "Scripts", "python.exe")
        if not os.path.exists(cand):
            cand = os.path.join(repo_root, ".venv-keras", "bin", "python")
        return cand if os.path.exists(cand) else None

    def _start(self) -> None:
        env = dict(os.environ, KERAS_BACKEND="torch", KRYSTAL_KERAS_THREADS=str(self.threads), PYTHONIOENCODING="utf-8")
        self._p = subprocess.Popen([self.python_exe, "-m", "krystal_bot.keras_worker"], cwd=self.repo_root, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        self.info = self._readline()
        if not self.info.get("ok"):
            raise RuntimeError("keras worker failed to start")

    def _readline(self) -> Dict[str, Any]:
        assert self._p and self._p.stdout
        result: Dict[str, Any] = {}

        def rd() -> None:
            try:
                result["line"] = self._p.stdout.readline()  # type: ignore[union-attr]
            except Exception as e:  # noqa: BLE001
                result["err"] = e

        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(self.timeout_s)
        if t.is_alive() or not result.get("line"):
            self.close()
            raise TimeoutError("keras worker did not answer")
        return json.loads(result["line"])

    def _call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._p is None or self._p.poll() is not None:
                self._start()
            assert self._p and self._p.stdin
            try:
                self._p.stdin.write(json.dumps(req) + "\n")
                self._p.stdin.flush()
                return self._readline()
            except (OSError, ValueError):
                self.close()
                raise

    def available(self) -> bool:
        try:
            return bool(self._call({"op": "ping"}).get("ok"))
        except Exception:  # noqa: BLE001
            return False

    def fit(self, xs: List[List[float]], ys: List[int], epochs: int = 60, seed: int = 7) -> Dict[str, Any]:
        r = self._call({"op": "train", "x": xs, "y": ys, "model_path": self.model_path, "epochs": epochs, "seed": seed})
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "train failed"))
        return r

    def predict(self, xs: Sequence[Sequence[float]]) -> List[float]:
        r = self._call({"op": "predict", "x": [list(x) for x in xs], "model_path": self.model_path})
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "predict failed"))
        return r["p"]

    def close(self) -> None:
        p, self._p = self._p, None
        if p is not None:
            try:
                if p.stdin:
                    p.stdin.close()
                p.terminate()
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
            for s in (p.stdout,):
                try:
                    if s:
                        s.close()
                except Exception:  # noqa: BLE001
                    pass


class RiskModel:
    """Facade: trains, validates, accepts/rejects, and predicts with the best available backend."""

    def __init__(self, state_path: str, keras: Optional[KerasClient] = None):
        self.state_path, self.keras = state_path, keras
        self.backend = "heuristic"
        self.accepted = False
        self.report: Dict[str, Any] = {"status": "untrained"}
        self._std: Optional[StdlibLogistic] = None
        self._load()

    def _load(self) -> None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, ValueError):
            return
        self.report, self.accepted, self.backend = st.get("report", self.report), bool(st.get("accepted")), st.get("backend", "heuristic")
        if st.get("stdlib"):
            self._std = StdlibLogistic.from_json(st["stdlib"])
        if self.backend == "keras" and not self.keras:
            self.backend, self.accepted = "heuristic", False  # model file exists but no worker to run it

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"report": self.report, "accepted": self.accepted, "backend": self.backend,
                       "stdlib": self._std.to_json() if self._std else None, "features": FEATURES}, f)
        os.replace(tmp, self.state_path)

    def fit(self, events: Sequence[Dict[str, Any]], prefer: str = "keras", seed: int = 7) -> Dict[str, Any]:
        xs, ys = build_dataset(events)
        n, npos = len(xs), sum(ys)
        rep: Dict[str, Any] = {"examples": n, "positives": npos, "features": FEATURES, "min_positives": MIN_POSITIVES, "min_auc": MIN_AUC}
        if n < 60 or npos < MIN_POSITIVES or (n - npos) < MIN_POSITIVES:
            rep.update(status="insufficient_data", accepted=False,
                       note=f"need >= {MIN_POSITIVES} positive and negative examples (have {npos}/{n - npos}); using heuristic")
            self.report, self.accepted, self.backend = rep, False, "heuristic"
            self._save()
            return rep
        rnd = random.Random(seed)
        idx = list(range(n))
        rnd.shuffle(idx)
        cut = int(0.8 * n)
        tr, ho = idx[:cut], idx[cut:]
        xs_tr, ys_tr = [xs[i] for i in tr], [ys[i] for i in tr]
        xs_ho, ys_ho = [xs[i] for i in ho], [ys[i] for i in ho]
        # Transparent baseline on the same holdout.
        base_auc = auc(ys_ho, [heuristic_risk(x) for x in xs_ho])
        std = StdlibLogistic()
        std.fit(xs_tr, ys_tr, seed=seed)
        std_auc = auc(ys_ho, std.predict(xs_ho))
        rep.update(holdout_n=len(ho), holdout_pos=sum(ys_ho), heuristic_auc=base_auc, stdlib_auc=std_auc)
        self._std = std
        chosen, chosen_auc = "stdlib-logistic", std_auc
        if prefer == "keras" and self.keras and self.keras.available():
            try:
                kr = self.keras.fit(xs, ys, seed=seed)  # worker makes its own 80/20 split with the same seed family
                rep["keras"] = {k: kr.get(k) for k in ("epochs_run", "loss", "holdout_auc", "holdout_n", "holdout_pos", "params")}
                k_auc = kr.get("holdout_auc")
                if k_auc is not None and (chosen_auc is None or k_auc >= chosen_auc):
                    chosen, chosen_auc = "keras", k_auc
            except Exception as e:  # noqa: BLE001
                rep["keras_error"] = f"{type(e).__name__}: {e}"
        elif prefer == "keras":
            rep["keras_error"] = "keras worker unavailable"
        beats = chosen_auc is not None and chosen_auc >= MIN_AUC and (base_auc is None or chosen_auc > base_auc + 0.01)
        rep.update(status="trained", chosen=chosen, chosen_auc=chosen_auc, accepted=bool(beats),
                   note=("accepted: beats heuristic on held-out data" if beats else
                         "rejected: does not beat the transparent heuristic on held-out data; heuristic stays in use"))
        self.report, self.accepted = rep, bool(beats)
        self.backend = chosen if beats else "heuristic"
        self._save()
        return rep

    def predict(self, xs: Sequence[Sequence[float]]) -> List[float]:
        if self.accepted:
            try:
                if self.backend == "keras" and self.keras:
                    return self.keras.predict(xs)
                if self.backend == "stdlib-logistic" and self._std:
                    return self._std.predict(xs)
            except Exception:  # noqa: BLE001
                pass  # a failing model must never block evaluation; fall through to the heuristic
        return [heuristic_risk(x) for x in xs]

    def status(self) -> Dict[str, Any]:
        return {"backend": self.backend, "accepted": self.accepted, "report": self.report,
                "keras_configured": self.keras is not None}
