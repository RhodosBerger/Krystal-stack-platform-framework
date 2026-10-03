"""BotService: wires RAG, palette, quotas, LLM gateway, risk model, pending evaluator, bytecode art,
Morse, donor gate, request mesh and the skill gateway into one object (stdlib only).

The service works standalone (no kernel) or attached to a `krystal_kernel.ComputeKernel`, in which
case it subscribes to the kernel's event log so every log record feeds RAG and the pending evaluator.
"""
from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import bytecode as BC
from . import composer as CP
from . import morse as MO
from .donor import check_token, load_public_keys
from .llm_gateway import GatewayDenied, LLMConfig, LLMGateway
from .mesh import Mesh, MeshAuth, load_or_create_secret
from .palette import Palette
from .pending import PendingEvaluator
from .quota import QuotaManager
from .rag import RagIndex, event_to_text, ingest_events, ingest_markdown_tree
from .risk import KerasClient, RiskModel
from .skills import Gateway, Pairing, Skill

HEALTH_RANK = {"ok": 0, "degraded": 1, "critical": 2}
HELP = """Commands:
/help                       this list
/status                     kernel health, quotas, model state
/search <text>              retrieve from docs and logs, with citations
/art <scene> [palette] [seed]   draw pixel art (scene: landscape | still_life)
/morse <text>               encode text to Morse
/risk                       risk-model status and held-out report
/predict                    predict the next kernel processes from the Morse-coded event stream
/quota                      LLM and evaluation quotas
/ask <question>             answer from retrieval; adds an LLM summary only if the LLM policy allows it
/pair                       request a pairing code (non-local senders)
/constellation              (donor easter egg) run a UUID cycle around the local mesh
Anything without a slash is treated as /search."""


class BotService:
    def __init__(self, root: str, data_dir: Optional[str] = None, kernel: Any = None, use_keras: bool = True,
                 llm_transport: Any = None, clock: Callable[[], float] = time.time):
        self.root = os.path.abspath(root)
        self.data = data_dir or os.path.join(self.root, "logs", "bot")
        self.cfg_dir = os.path.join(self.root, "config")
        os.makedirs(self.data, exist_ok=True)
        self.kernel, self._clock = kernel, clock
        self._lock = threading.RLock()
        self.rag = RagIndex(os.path.join(self.data, "rag.jsonl"))
        self.palette = Palette(os.path.join(self.data, "palette.jsonl"), on_entry=self._index_chat)
        self.quotas = QuotaManager(os.path.join(self.data, "quotas.json"), clock=clock)
        self.audit_log: List[Dict[str, Any]] = []
        self.llm = LLMGateway(LLMConfig.load(os.path.join(self.cfg_dir, "llm_gateway.json")), self.quotas, llm_transport,
                              self._hardware, clock, self._audit)
        py = KerasClient.find_python(self.root) if use_keras else None
        self.keras = KerasClient(py, self.root, os.path.join(self.data, "risk_model.keras")) if py else None
        self.risk = RiskModel(os.path.join(self.data, "risk_state.json"), self.keras)
        self.pending = PendingEvaluator(self.risk, self.quotas, on_flag=self._on_flag)
        self.predictor = MO.ProcessPredictor()
        self.mesh_auth = MeshAuth(load_or_create_secret(os.path.join(self.data, "mesh_secret.bin")), clock)
        self.mesh = Mesh(self.mesh_auth, on_cycle=self._on_cycle)
        self.pubkeys = load_public_keys(os.path.join(self.cfg_dir, "donor_pubkeys.json"))
        self.donor: Optional[Dict[str, Any]] = None
        self._load_donor()
        self.quotas.define("api_requests", 300, 60.0)
        self.quotas.define("renders", 60, 60.0)
        self.gateway = Gateway(Pairing(clock=clock), self.is_donor, self.llm.debug_mode)
        self._register_skills()
        self.flags: List[Dict[str, Any]] = []
        if kernel is not None:
            kernel.log.subscribe(self._on_event)

    # ------------------------------------------------------------------ plumbing
    def _hardware(self) -> Dict[str, Any]:
        if self.kernel is not None:
            return self.kernel.profile.to_dict()
        try:
            from krystal_kernel.hwprofile import detect_profile
            return detect_profile().to_dict()
        except Exception:  # noqa: BLE001
            return {"logical_cores": os.cpu_count() or 1, "ram_gb": 0.0}

    def _audit(self, kind: str, fields: Dict[str, Any]) -> None:
        self.audit_log.append({"t": round(self._clock(), 3), "type": kind, **fields})
        del self.audit_log[:-200]

    def _index_chat(self, entry: Dict[str, Any]) -> None:
        self.rag.add("palette", "chat", f"{entry['role']}: {entry['text']}", {"conversation": entry["conversation"], "channel": entry["channel"]}, ts=entry["t"])

    def _on_event(self, ev: Dict[str, Any]) -> None:
        self.pending.submit(ev)
        if ev.get("type") != "task_done":
            self.rag.add("kernel_events", "event", event_to_text(ev), {"seq": ev.get("seq"), "etype": ev.get("type")}, ts=ev.get("t"))

    def _on_flag(self, flag: Dict[str, Any]) -> None:
        ev = flag["event"]
        self.flags.append({"risk": flag["risk"], "backend": flag["backend"], "kind": ev.get("kind"), "lane": ev.get("lane"),
                           "queue_ms": ev.get("queue_ms"), "deadline_ms": ev.get("deadline_ms"), "actual_bad": flag["actual_bad"], "seq": ev.get("seq")})
        del self.flags[:-200]

    def _on_cycle(self, res: Dict[str, Any]) -> None:
        self.rag.add("mesh", "event", f"mesh_cycle id={res['cycle_id'][:8]} nodes={res['nodes']} ok={res['ok']} total_ms={res['total_ms']}")

    # ------------------------------------------------------------------ donor
    def _load_donor(self) -> None:
        try:
            with open(os.path.join(self.data, "donor.json"), "r", encoding="utf-8") as f:
                tok = json.load(f).get("token", "")
            r = check_token(tok, self.pubkeys, self._clock())
            self.donor = r["claims"] if r["valid"] else None   # re-verified on every start; a stale or edited file simply locks
        except (OSError, ValueError):
            self.donor = None

    def is_donor(self) -> bool:
        d = self.donor
        return bool(d) and d.get("exp", 0) >= self._clock()

    def activate_donor(self, token: str) -> Dict[str, Any]:
        r = check_token(token, self.pubkeys, self._clock())
        if r["valid"]:
            self.donor = r["claims"]
            tmp = os.path.join(self.data, "donor.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"token": token}, f)
            os.replace(tmp, os.path.join(self.data, "donor.json"))
            self.palette.add("system", f"Thank you, {r['claims'].get('sub', 'friend')}. The constellation is open.", "internal")
        return {"valid": r["valid"], "reason": r["reason"], "alias": (r.get("claims") or {}).get("sub") if r["valid"] else None}

    # ------------------------------------------------------------------ knowledge
    def reindex_docs(self) -> Dict[str, int]:
        return ingest_markdown_tree(self.rag, os.path.join(self.root, "docs"))

    def reindex_events(self) -> Dict[str, int]:
        path = os.path.join(self.root, "logs", "kernel_events.jsonl")
        try:
            from krystal_kernel.eventlog import EventLog
            evs = EventLog.read_file(path)
        except Exception:  # noqa: BLE001
            evs = []
        return {"events_read": len(evs), "indexed": ingest_events(self.rag, evs)}

    def event_history(self) -> List[Dict[str, Any]]:
        h = self.pending.history()
        if h:
            return h
        try:
            from krystal_kernel.eventlog import EventLog
            return list(EventLog.read_file(os.path.join(self.root, "logs", "kernel_events.jsonl")))
        except Exception:  # noqa: BLE001
            return []

    def train_risk(self, prefer: str = "keras") -> Dict[str, Any]:
        evs = [e for e in self.event_history() if e.get("type") == "task_done"]
        rep = self.risk.fit(evs, prefer=prefer)
        syms = MO.events_to_symbols(self.event_history())
        rep["process_predictor"] = self.predictor.evaluate(syms)
        self.predictor = MO.ProcessPredictor()
        self.predictor.fit(syms)
        return rep

    def predict_next(self, k: int = 3) -> Dict[str, Any]:
        syms = MO.events_to_symbols(self.event_history())
        ctx = syms[-8:]
        ev = self.predictor.evaluate(syms) if len(syms) >= 100 else {"status": "insufficient_data", "symbols": len(syms)}
        pred = self.predictor.predict(ctx, k) if self.predictor.trained_on else []
        return {"context": ctx, "context_morse": MO.encode_text(ctx), "evaluation": ev, "trusted": bool(ev.get("beats_baseline")),
                "predictions": [{"symbol": s, "p": p, "meaning": MO.SYMBOL_MEANING.get(s, "?")} for s, p in pred]}

    # ------------------------------------------------------------------ art
    def should_compute(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Decide *when* a render may run from the config's compute policy and live kernel signals."""
        comp = cfg["compute"]
        state = (self.kernel.health["state"] if self.kernel is not None else "ok")
        limit = comp.get("defer_if_health")
        if limit and HEALTH_RANK.get(state, 2) >= HEALTH_RANK[limit]:
            return {"run": False, "reason": f"deferred: kernel health is '{state}' (policy defers at '{limit}')"}
        trig = comp["trigger"]
        if trig == "idle" and self.kernel is not None and self.kernel.queue_len > max(1, self.kernel.params.max_queue // 10):
            return {"run": False, "reason": "deferred: kernel queue is not idle"}
        if trig == "on_pattern":
            want = comp.get("pattern")
            have = {p.get("kind") for p in self.kernel.learner.patterns()} if self.kernel is not None else set()
            if not want or want not in have:
                return {"run": False, "reason": f"deferred: waiting for pattern '{want}'"}
        return {"run": True, "reason": f"trigger '{trig}' satisfied; kernel health '{state}'"}

    def render(self, raw_cfg: Optional[Dict[str, Any]], fmt: str = "png", force: bool = False) -> Dict[str, Any]:
        cfg = BC.normalize_config(raw_cfg)
        if cfg["palette"] in CP.DONOR_PALETTES and not self.is_donor():
            raise PermissionError(f"palette '{cfg['palette']}' is part of the donor easter egg")
        ok, retry = self.quotas.try_consume("renders", 1)
        if not ok:
            raise GatewayDenied("rate_limited", "render quota exhausted", retry)
        decision = self.should_compute(cfg)
        if not decision["run"] and not force:
            return {"computed": False, "reason": decision["reason"], "config": cfg}
        t0 = time.perf_counter()
        prog = BC.compile_config(cfg)
        cv = BC.run(prog)
        ms = (time.perf_counter() - t0) * 1000
        data = prog.to_bytes()
        out: Dict[str, Any] = {"computed": True, "reason": decision["reason"], "config": cfg, "ops": len(prog), "bytecode_bytes": len(data),
                               "bytecode_b64": base64.b64encode(data).decode("ascii"), "digest": prog.digest(),
                               "fingerprint": cv.fingerprint(), "morse": MO.encode_program(prog), "render_ms": round(ms, 2),
                               "over_budget": ms > cfg["compute"]["budget_ms"], "size": [cv.w, cv.h]}
        if fmt == "png":
            out["png_b64"] = base64.b64encode(cv.to_png(cfg["scale"])).decode("ascii")
        elif fmt == "svg":
            out["svg"] = cv.to_svg(cfg["scale"])
        elif fmt == "ascii":
            out["ascii"] = cv.to_ascii()
        else:
            raise ValueError("format must be png, svg or ascii")
        return out

    def replicate(self, bytecode_b64: str, rate: float = 0.08, seed: Optional[int] = None, count: int = 1) -> Dict[str, Any]:
        prog = BC.Program.from_bytes(base64.b64decode(bytecode_b64, validate=True))
        rng = random.Random(seed)
        kids = []
        for _ in range(max(1, min(count, 8))):
            r = BC.replicate(prog, rng, max(0.0, min(rate, 0.5)))
            kid = r["child"]
            cv = BC.run(kid)
            kids.append({"digest": r["child_digest"], "parent": r["parent_digest"], "identical": r["identical"], "ops": len(kid),
                         "bytecode_b64": base64.b64encode(kid.to_bytes()).decode("ascii"), "morse_chars": len(MO.encode_program(kid)),
                         "png_b64": base64.b64encode(cv.to_png(4)).decode("ascii")})
        return {"parent": prog.digest(), "children": kids}

    # ------------------------------------------------------------------ messages
    def _register_skills(self) -> None:
        g = self.gateway
        g.register(Skill("help", "List commands", ["/help"]), lambda a, c: HELP)
        g.register(Skill("status", "System status", ["/status"]), lambda a, c: self._status_text())
        g.register(Skill("search", "Retrieve from docs and logs", ["/search"]), lambda a, c: self._search_text(a))
        g.register(Skill("ask", "Answer from retrieval, LLM optional", ["/ask"]), lambda a, c: self._ask_text(a))
        g.register(Skill("art", "Draw pixel art", ["/art"]), lambda a, c: self._art_text(a))
        g.register(Skill("morse", "Text to Morse", ["/morse"]), lambda a, c: MO.encode_text(a) or "(nothing to encode)")
        g.register(Skill("risk", "Risk model status", ["/risk"]), lambda a, c: json.dumps(self.risk.status()["report"], indent=2)[:1500])
        g.register(Skill("predict", "Predict next processes", ["/predict"]), lambda a, c: json.dumps(self.predict_next(), indent=2)[:1500])
        g.register(Skill("quota", "Show quotas", ["/quota"]), lambda a, c: json.dumps(self.quotas.snapshot(), indent=2))
        g.register(Skill("pair", "Request a pairing code", ["/pair"], channels=["extension", "web", "cli", "internal", "mesh"]),
                   lambda a, c: f"Pairing code {self.gateway.pairing.request(c['sender'])}: the local operator must approve it.")
        g.register(Skill("constellation", "Run a mesh UUID cycle", ["/constellation"], requires_donor=True), lambda a, c: self._mesh_text())

    def message(self, text: str, channel: str = "extension", conversation: Optional[str] = None, sender: str = "local") -> Dict[str, Any]:
        u = self.palette.add("user", text, channel, conversation)
        res = self.gateway.dispatch(text, channel, sender, fallback="search")
        b = self.palette.add("bot", res["reply"], channel, u["conversation"], parent=u["id"], meta={"skill": res.get("skill"), "ok": res["ok"]})
        return {"conversation": u["conversation"], "reply": res["reply"], "ok": res["ok"], "skill": res.get("skill"), "entry": b["id"]}

    def _status_text(self) -> str:
        k = self.kernel
        lines = [f"kernel: {'attached, health ' + k.health['state'] if k is not None else 'not attached'}",
                 f"rag: {self.rag.stats()['documents']} documents",
                 f"risk model: backend={self.risk.backend} accepted={self.risk.accepted}",
                 f"llm: {'allowed' if self.llm.policy()['allowed'] else 'off (' + '; '.join(self.llm.policy()['reasons'][:1]) + ')'}",
                 f"donor: {'yes' if self.is_donor() else 'no'}", f"pending evaluation: {self.pending.stats()['pending']}"]
        return "\n".join(lines)

    def _search_text(self, q: str) -> str:
        if not q:
            return "Give me something to search for."
        hits = self.rag.search(q, 4)
        if not hits:
            return "No matching documents or logs. (Index docs with POST /api/bot/reindex.)"
        return "\n".join(f"[{i + 1}] {h['citation']} (score {h['score']}): {h['text'][:220].strip()}" for i, h in enumerate(hits))

    def _ask_text(self, q: str) -> str:
        base = self._search_text(q)
        pol = self.llm.policy()
        if not pol["allowed"]:
            return base + "\n(LLM summary off: " + "; ".join(pol["reasons"][:2]) + ")"
        try:
            ctx = "\n".join(h["text"][:600] for h in self.rag.search(q, 4))
            r = self.llm.chat([{"role": "system", "content": "Answer only from the provided context. If it is insufficient, say so."},
                               {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}], "ask")
            return base + "\n\nLLM summary (unverified):\n" + r["text"].strip()
        except GatewayDenied as e:
            return base + f"\n(LLM summary skipped: {e})"

    def _art_text(self, arg: str) -> str:
        parts = arg.split()
        cfg: Dict[str, Any] = {"width": 48, "height": 24, "techniques": ["dither", "outline"]}
        if parts:
            cfg["scene"] = parts[0]
        if len(parts) > 1:
            cfg["palette"] = parts[1]
        if len(parts) > 2 and parts[2].isdigit():
            cfg["seed"] = int(parts[2])
        r = self.render(cfg, "ascii", force=True)
        return f"digest {r['digest']}  ({r['ops']} ops, {r['bytecode_bytes']} bytes, {r['render_ms']} ms)\n{r['ascii']}"

    def _mesh_text(self) -> str:
        peers = self.mesh.peers()
        if not peers:
            return "The mesh is empty. Start nodes and register them via POST /api/bot/mesh/register."
        r = self.mesh.cycle()
        return f"cycle {r['cycle_id'][:8]}: ok={r['ok']} nodes={r['nodes']} total={r['total_ms']} ms\n" + \
               "\n".join(f"  {h['name']}: {'ok' if h['ok'] else 'FAIL'} {h['rtt_ms']} ms" for h in r["hops"])

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.pending.start(2.0)

    def shutdown(self) -> None:
        self.pending.stop()
        if self.keras:
            self.keras.close()
        try:
            self.rag.save()
            self.quotas.save()
        except OSError:
            pass
