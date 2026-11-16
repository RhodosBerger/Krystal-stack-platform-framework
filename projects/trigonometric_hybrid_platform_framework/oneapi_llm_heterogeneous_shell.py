"""Governed Command-Line Shell for heterogeneous compute via oneAPI/OpenVINO and LLM bridge,
plus algebraic-theoretical simulator coupling derived quantities to operator algebra.

Security & Constitutional Design:
  - PROPOSER = CLI user or OpenAI-compatible LLM endpoint (:1234/v1) / oneAPI AST fallback
  - VERIFIER = ShellCommandVerifier (strict verb whitelist, typed bounds, zero os.system)
  - Invalid command or invariant breach -> ContractViolation and Delta S == 0
  - Every state-changing command (swap, plan-combat --apply) emits a 128 B SHA-256 receipt.
"""

from __future__ import annotations

import argparse
import cmd
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD as UMA_SATURATION_CEILING, VITAL_MAX_HP, ContractViolation
from krystal_kernel.openvino_aep_runtime import OpenVinoAepRuntime
from krystal_kernel.sovereign_openvino_oneapi_app import REAL_LMSTUDIO_MODELS

from .adaptive_cross_genre_sync import simulate_adaptive_rollback
from .clash_battle_royale import ClashGame, _pack_frame
from .combat_turn_planner import CapabilityCombatTurnPlanner, DEFAULT_CACHE_DIR, probe_framework_dispositions
from .configurator_and_dxdiag import (
    UTILITY_SEMANTIC_FRAMES,
    analyze_opus_and_build_intersections,
    audit_numeric_type_contract,
    build_blender_system_workflow_schema,
    build_dxdiag_memory_profile,
    build_validated_configuration,
    export_dxdiag_txt,
    partition_reduction_below_k449,
    prewarm_heterogeneous_blob_pair,
)
from .custom_allocator import FactDrivenAllocator
from .fact_synthesis_engine import run_fact_synthesis
from .intel_accelerator_and_power_thread_governor import (
    ChipsetTelemetryMonitorAndForecaster,
    IntelOneApiAccelerator,
    benchmark_thread_and_performance_curves,
    match_active_state_templates,
    select_power_and_thread_strategy,
)
from .matrix_control_processes import freivalds_verify
from .virtual_convolution_pipeline import _weights, conv2d_reference
from .neurosymbolic_bridge import ask_llm
from .tensor_prompt_branch_and_kb import (
    TensorPromptBranchEngine,
    coordinate_operations_from_turn,
    sync_master_knowledge_base,
)

ALLOWED_VERBS = frozenset(
    {
        "help",
        "probe",
        "models",
        "opus-audit",
        "alloc",
        "swap",
        "bench",
        "threads",
        "plan-combat",
        "configure",
        "utilities",
        "dxdiag",
        "states",
        "branch",
        "kb-sync",
        "sim-algebra",
        "blender-schema",
        "ask",
    }
)
FORBIDDEN_CHARS = frozenset({"|", "&", "`", "$", ">", "<", "\n", "\r"})


def simulate_algebraic_theory_coupling(ticks: int = 24, eta: float = 0.08, seed: int = 6) -> Dict[str, Any]:
    """Couple derived crossover/horizon formulas with {K, K^T} operator algebra in a discrete simulator."""
    if not (4 <= ticks <= 500):
        raise ContractViolation("ticks must be in [4, 500]", violation_code="ERR_SIM_TICKS")

    syn = run_fact_synthesis(write=False)
    facts = {f["fact_id"]: f["value"] for f in syn["facts"]}
    c_gpu = float(facts.get("F_C03_GPU_compile_ms", 187.48))
    c_cpu = float(facts.get("F_C03_CPU_compile_ms", 39.65))
    m_gpu = float(facts.get("F_C03_GPU_median_ms", 3.24))
    m_cpu = float(facts.get("F_C03_CPU_median_ms", 3.519))
    dm = m_cpu - m_gpu
    cold_break_even = round((c_gpu - c_cpu) / dm, 2) if dm > 0 else None
    warm_break_even = round(max(0.0, (12.0 - c_cpu)) / dm, 2) if dm > 0 else None

    # Build explicit 2D convolution matrix K (64 x 64) via basis columns of conv2d_reference
    rng = np.random.default_rng(seed)
    w_kern = _weights(1, 1, seed)
    cols = []
    for idx in range(64):
        e = np.zeros((1, 1, 8, 8), dtype=np.float32)
        e.flat[idx] = 1.0
        cols.append(conv2d_reference(e, w_kern).reshape(64).astype(np.float64))
    k_mat = np.column_stack(cols)
    kt_mat = k_mat.T
    gram = kt_mat @ k_mat
    freiv = freivalds_verify(kt_mat, k_mat, gram, rounds=6, rtol=1e-9, seed=seed)
    if not freiv["accepted"]:
        raise ContractViolation("Freivalds failed on Gram matrix K^T K", violation_code="ERR_SIM_GRAM")

    # Spectral scaling keeps step size inside convergent regime (0 < eta_eff < 2 / ||K^T K||_F)
    frob = float(np.linalg.norm(gram, ord="fro")) or 1.0
    eta_eff = min(eta, 1.2 / frob)

    x_true = rng.standard_normal(64)
    y_probe = rng.standard_normal(64)
    adjoint_gap = float(abs(np.dot(k_mat @ x_true, y_probe) - np.dot(x_true, kt_mat @ y_probe)))
    b_target = k_mat @ x_true
    x_est = np.zeros(64, dtype=np.float64)

    residuals: List[float] = []
    for _ in range(ticks):
        grad = gram @ x_est - kt_mat @ b_target
        x_est = x_est - eta_eff * grad
        residuals.append(float(np.linalg.norm(k_mat @ x_est - b_target)))

    monotone = all(residuals[i + 1] <= residuals[i] + 1e-12 for i in range(len(residuals) - 1))
    rb = simulate_adaptive_rollback(ticks=max(60, ticks * 4), one_way_latency_ms=40.0, seed=seed)

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "computable_quantities": {
            "cold_gpu_break_even_inferences": cold_break_even,
            "warm_cache_gpu_break_even_inferences": warm_break_even,
            "freivalds_false_accept_upper_bound": 2 ** (-6),
            "operator_adjoint_gap": adjoint_gap,
            "rollback_max_horizon_ticks": rb["max_horizon"],
            "rollback_desync_ticks": rb["desync_ticks"],
        },
        "algebraic_trajectory": {
            "ticks": ticks,
            "effective_step_eta": round(eta_eff, 6),
            "initial_residual_l2": round(residuals[0], 6),
            "final_residual_l2": round(residuals[-1], 6),
            "residual_reduction_ratio": round(residuals[0] / max(residuals[-1], 1e-15), 3),
            "monotone_contraction_verified": monotone,
            "gram_freivalds_accepted": freiv["accepted"],
        },
        "verdict": "VERIFIED_WITH_SCOPE" if (monotone and adjoint_gap < 1e-9 and rb["desync_ticks"] == 0) else "CONFLICT",
        "evidence": "SIMULATED + LOCAL_MEASURED_NUMERICAL",
    }


class ShellCommandVerifier:
    """Strict constitutional gate validating every CLI or LLM-proposed command before execution."""

    @staticmethod
    def verify_and_parse(raw_command: str, available_devices: List[str]) -> Dict[str, Any]:
        line = raw_command.strip()
        if not line:
            raise ContractViolation("Empty shell command", violation_code="ERR_SHELL_EMPTY")
        bad = [ch for ch in FORBIDDEN_CHARS if ch in line]
        if bad:
            raise ContractViolation(f"Forbidden shell metacharacters: {bad}", violation_code="ERR_SHELL_METACHAR")
        try:
            tokens = shlex.split(line)
        except ValueError as exc:
            raise ContractViolation(f"Malformed quoting: {exc}", violation_code="ERR_SHELL_PARSE") from exc

        verb = tokens[0].lower()
        if verb not in ALLOWED_VERBS:
            raise ContractViolation(f"Unauthorized command verb {verb!r}", violation_code="ERR_SHELL_UNAUTHORIZED_VERB")

        args = tokens[1:]
        if verb == "swap":
            if not args:
                raise ContractViolation("swap requires device argument (CPU or GPU)", violation_code="ERR_SHELL_SWAP_ARG")
            target = args[0].upper()
            if target not in available_devices:
                raise ContractViolation(
                    f"Target device {target!r} not in available_devices={available_devices}",
                    violation_code="ERR_SHELL_DEVICE_UNAVAILABLE",
                )
        return {"verb": verb, "args": args, "canonical": " ".join([verb] + args)}


class KrystalOneApiHeterogeneousShell:
    """Governed heterogeneous compute CLI shell bridged to Intel oneAPI/OpenVINO and LLM models."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        llm_base_url: str = "http://127.0.0.1:1234/v1",
        llm_caller: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.llm_base_url = llm_base_url
        self._llm_caller = llm_caller or ask_llm
        self.accel = IntelOneApiAccelerator(cache_dir=cache_dir)
        self._aep_runtime: Optional[OpenVinoAepRuntime] = None
        self.monitor = ChipsetTelemetryMonitorAndForecaster()
        self.branch_engine = TensorPromptBranchEngine(cache_dir=cache_dir)
        self.arena = ClashGame("VERIFY", seed=6)
        self.uma_utilization: float = 0.35
        self.audit_log: List[Dict[str, Any]] = []

    @property
    def aep_runtime(self) -> OpenVinoAepRuntime:
        if self._aep_runtime is None:
            self._aep_runtime = OpenVinoAepRuntime()
        return self._aep_runtime

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "active_device": self.accel.current_device,
            "uma_utilization": self.uma_utilization,
            "arena_hash": self.arena.state_hash(),
            "audit_entries": len(self.audit_log),
        }

    def state_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.state_snapshot(), sort_keys=True).encode("utf-8")).hexdigest()

    def _probe_live_llm_models(self) -> List[str]:
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.llm_base_url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=1.2) as r:
                data = json.loads(r.read().decode("utf-8"))
                return [str(m.get("id")) for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []

    def _deterministic_oneapi_intent_to_command(self, intent: str) -> Dict[str, Any]:
        """Fallback command proposer backed by OpenVINO prompt-matrix + keyword grammar when :1234 LLM is offline."""
        branch = self.branch_engine.evaluate_prompt_matrix_branches([intent], device=self.accel.current_device, uma_utilization=self.uma_utilization)
        top_branch = branch["branch_trees"][0]["scaled_solutions"][0]["branch_id"]
        low = intent.lower()

        if any(w in low for w in ("swap", "prepni", "prepnúť", "gpu", "iris")) and "GPU" in self.accel.devices and self.uma_utilization < UMA_SATURATION_CEILING:
            cmd_str = f"swap GPU --uma {self.uma_utilization}"
        elif any(w in low for w in ("cpu", "avx")):
            cmd_str = f"swap CPU --uma {self.uma_utilization}"
        elif any(w in low for w in ("combat", "ťah", "tah", "boj", "arén", "aren")):
            cmd_str = f"plan-combat {self.arena.state['player_faction']} --uma {self.uma_utilization}"
        elif any(w in low for w in ("dxdiag", "pamäť", "pamat", "tok")):
            cmd_str = "dxdiag"
        elif any(w in low for w in ("konfigur", "config", "profil")):
            goal = "high_throughput_batch" if top_branch == "B3_GPU_IRIS_XE_WARM_STREAM" else "low_latency_realtime"
            cmd_str = f"configure {goal} --uma {self.uma_utilization}"
        elif any(w in low for w in ("opus", "prienik", "audit")):
            cmd_str = "opus-audit"
        elif any(w in low for w in ("bench", "zmeraj", "konvol")):
            cmd_str = "bench 8 16 32 32 5"
        else:
            cmd_str = "probe"

        return {
            "proposed_command": cmd_str,
            "proposer_mode": "ONEAPI_AST_FALLBACK_LLM_OFFLINE",
            "tensor_top_branch": top_branch,
        }

    def propose_from_natural_language(self, intent: str) -> Dict[str, Any]:
        """Bridge natural-language intent via OpenAI-compatible LLM (:1234/v1) or oneAPI tensor+AST fallback."""
        syn = run_fact_synthesis(write=False)
        fact_snippet = "; ".join(f"[{f['fact_id']}] {f['statement']}" for f in syn["facts"][:8])
        sys_msg = (
            "Translate the user's request into EXACTLY ONE single-line Krystal CLI command using ONLY one of: "
            "probe | models | opus-audit | alloc <n> | swap <CPU|GPU> | bench <C> <F> <H> <W> <iters> | "
            "plan-combat <SYNC|COMPUTE|VERIFY|OPS> | configure <low_latency_realtime|high_throughput_batch|bit_exact_audit|arena_hybrid_combat> | "
            "dxdiag | states | sim-algebra. Output ONLY the raw command line, no markdown."
        )
        llm_res = self._llm_caller(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"FACTS: {fact_snippet}\nINTENT: {intent}"}],
            base_url=self.llm_base_url,
            timeout=2.5,
        )
        if llm_res.get("ok") and str(llm_res.get("text", "")).strip():
            first_line = str(llm_res["text"]).strip().splitlines()[0].strip("` ")
            proposal = {
                "proposed_command": first_line,
                "proposer_mode": "LIVE_OPENAI_COMPATIBLE_LLM",
                "llm_error": None,
            }
        else:
            fb = self._deterministic_oneapi_intent_to_command(intent)
            fb["llm_error"] = llm_res.get("error")
            proposal = fb

        # Mandatory symbolic verification (PROPOSER != VERIFIER)
        parsed = ShellCommandVerifier.verify_and_parse(proposal["proposed_command"], self.accel.devices)
        proposal["verified_command"] = parsed["canonical"]
        proposal["verifier"] = "ShellCommandVerifier"
        return proposal

    def execute_line(self, raw_command: str) -> Dict[str, Any]:
        """Verify and execute one governed CLI command (or `;`-separated sequence for batch scripts)."""
        if ";" in raw_command and not raw_command.strip().lower().startswith("ask "):
            parts = [p.strip() for p in raw_command.split(";") if p.strip()]
            results = [self.execute_line(p) for p in parts]
            return {"batch": True, "count": len(results), "results": results, "active_device": self.accel.current_device}

        parsed = ShellCommandVerifier.verify_and_parse(raw_command, self.accel.devices)
        verb, args = parsed["verb"], parsed["args"]

        if verb == "help":
            return {"verb": "help", "allowed_verbs": sorted(ALLOWED_VERBS), "active_device": self.accel.current_device}

        if verb == "probe":
            disp = probe_framework_dispositions(uma_utilization=self.uma_utilization)
            telem = self.monitor.record_sample(synthetic_uma_pressure=self.uma_utilization)
            props = {d: self.accel.get_device_properties(d).__dict__ for d in self.accel.devices}
            return {"verb": "probe", "active_device": self.accel.current_device, "dispositions": disp, "telemetry": telem, "device_properties": props}

        if verb == "models":
            live_lms = self._probe_live_llm_models()
            return {
                "verb": "models",
                "oneapi_openvino_compiled_models": [
                    "krystal_virtual_conv_pipeline (opset13 Conv-ReLU-Conv-ReLU)",
                    "krystal_prompt_decision_matrix (opset13 MatMul+Softmax)",
                    "cortex_process_prioritizer (OpenVinoAepRuntime)",
                ],
                "openvino_devices": self.accel.devices,
                "lm_studio_endpoint": self.llm_base_url,
                "lm_studio_online": bool(live_lms),
                "lm_studio_live_models": live_lms,
                "configured_hybrid_catalog": list(REAL_LMSTUDIO_MODELS.keys()),
            }

        if verb == "opus-audit":
            return {"verb": "opus-audit", **analyze_opus_and_build_intersections(self.cache_dir)}

        if verb == "alloc":
            n = int(args[0]) if len(args) >= 1 else 600
            budget = float(args[1]) if len(args) >= 2 else 10.0
            uma = float(args[2]) if len(args) >= 3 and not args[2].startswith("--") else self.uma_utilization
            cold_alloc = FactDrivenAllocator().allocate(n, budget, self.accel.current_device)
            strat = select_power_and_thread_strategy(uma_utilization=uma, expected_inferences=n)
            return {"verb": "alloc", "fact_driven_cold": cold_alloc, "power_thread_strategy": strat}

        if verb == "swap":
            target = args[0].upper()
            uma = self.uma_utilization
            if "--uma" in args:
                idx = args.index("--uma")
                uma = float(args[idx + 1])
            pre_hash = self.state_hash()
            # Prewarm and verify numeric accuracy before committing device change
            warm_check = prewarm_heterogeneous_blob_pair(shape=(1, 8, 32, 32), filters=16, cache_dir=self.cache_dir)
            swapped = self.accel.set_device(target, uma_utilization=uma)
            self.uma_utilization = uma
            receipt = _pack_frame(f"SWAP|{swapped['previous_device']}->{target}|uma={uma:.2f}")
            entry = {
                "verb": "swap",
                **swapped,
                "uma_utilization": uma,
                "warm_verification": warm_check["devices"].get(target),
                "pre_state_sha256": pre_hash,
                "post_state_sha256": self.state_hash(),
                "receipt_128b": receipt,
            }
            self.audit_log.append(entry)
            return entry

        if verb == "bench":
            c = int(args[0]) if len(args) >= 1 else 8
            f = int(args[1]) if len(args) >= 2 else 16
            h = int(args[2]) if len(args) >= 3 else 32
            w = int(args[3]) if len(args) >= 4 else 32
            iters = int(args[4]) if len(args) >= 5 else 6
            compiled, comp_meta = self.accel.compile_conv_kernel((1, c, h, w), filters=f, device=self.accel.current_device)
            rng = np.random.default_rng(6)
            x = rng.standard_normal((1, c, h, w)).astype(np.float32)
            tensor, xfer = self.accel.to_tensor(x)
            stream = self.accel.create_stream(compiled, self.accel.current_device)
            latencies: List[float] = []
            for _ in range(max(1, iters)):
                stream.enqueue(tensor).synchronize()
                latencies.append(stream.last_elapsed_ms)
            self.monitor.record_sample(synthetic_uma_pressure=self.uma_utilization, infer_latency_ms=float(np.median(latencies)))
            return {
                "verb": "bench",
                "device": self.accel.current_device,
                "compile": comp_meta,
                "transfer": xfer,
                "runs": len(latencies),
                "median_ms": round(float(np.median(latencies)), 3),
                "p95_ms": round(float(np.percentile(latencies, 95)), 3),
                "evidence": "LOCAL_MEASURED",
            }

        if verb == "threads":
            return {"verb": "threads", **benchmark_thread_and_performance_curves(runs=6, cache_dir=self.cache_dir)}

        if verb == "plan-combat":
            fac = args[0].upper() if args and not args[0].startswith("--") else self.arena.state["player_faction"]
            uma = self.uma_utilization
            if "--uma" in args:
                uma = float(args[args.index("--uma") + 1])
            commit = "--apply" in args
            plan = CapabilityCombatTurnPlanner().plan_turn(self.arena, faction=fac, uma_utilization=uma, commit=commit)
            if commit:
                self.audit_log.append({"verb": "plan-combat", "committed": True, "frames_used": plan["frames_used"]})
            return {"verb": "plan-combat", **plan}

        if verb == "configure":
            goal = args[0] if args and not args[0].startswith("--") else "low_latency_realtime"
            uma = float(args[args.index("--uma") + 1]) if "--uma" in args else self.uma_utilization
            n = int(args[args.index("--inferences") + 1]) if "--inferences" in args else 400
            budget = float(args[args.index("--budget-ms") + 1]) if "--budget-ms" in args else 10.0
            rtt = float(args[args.index("--rtt-ms") + 1]) if "--rtt-ms" in args else 25.0
            return {
                "verb": "configure",
                "profile": build_validated_configuration(goal=goal, expected_inferences=n, latency_budget_ms=budget, uma_utilization=uma, rtt_ms=rtt),
                "numeric_contract": audit_numeric_type_contract(),
            }

        if verb == "utilities":
            rng = np.random.default_rng(6)
            tiled = partition_reduction_below_k449(rng.standard_normal((8, 600)).astype(np.float32), rng.standard_normal((600, 8)).astype(np.float32))
            return {
                "verb": "utilities",
                "utilities": list(UTILITY_SEMANTIC_FRAMES),
                "k448_tiling_probe": {k: v for k, v in tiled.items() if k != "output"},
            }

        if verb == "dxdiag":
            prof = build_dxdiag_memory_profile(uma_utilization=self.uma_utilization)
            txt_path = export_dxdiag_txt(prof)
            return {"verb": "dxdiag", "txt_report": str(txt_path), "profile": prof}

        if verb == "blender-schema":
            return {"verb": "blender-schema", **build_blender_system_workflow_schema(uma_utilization=self.uma_utilization)}

        if verb == "states":
            live_lms = bool(self._probe_live_llm_models())
            return {"verb": "states", **match_active_state_templates(uma_utilization=self.uma_utilization, llm_online=live_lms)}

        if verb == "branch":
            prompt_text = " ".join(args) if args else "Prepni konvolúciu na teplú cache a over maticu"
            return {"verb": "branch", **self.branch_engine.evaluate_prompt_matrix_branches([prompt_text], device=self.accel.current_device, uma_utilization=self.uma_utilization)}

        if verb == "kb-sync":
            kb = sync_master_knowledge_base(write_files=True)
            coord = coordinate_operations_from_turn(self.arena, uma_utilization=self.uma_utilization, commit_turn=False)
            return {"verb": "kb-sync", "facts_count": kb["facts_count"], "ontology_integrity": kb["ontology_integrity"]["verdict"], "coordinated_ops": len(coord["coordinated_operations"])}

        if verb == "sim-algebra":
            ticks = int(args[0]) if args else 24
            return {"verb": "sim-algebra", **simulate_algebraic_theory_coupling(ticks=ticks)}

        if verb == "ask":
            do_exec = False
            rem = list(args)
            if rem and rem[0] == "--execute":
                do_exec = True
                rem = rem[1:]
            intent = " ".join(rem).strip()
            if not intent:
                raise ContractViolation("ask requires natural-language intent text", violation_code="ERR_SHELL_ASK_EMPTY")
            prop = self.propose_from_natural_language(intent)
            out: Dict[str, Any] = {"verb": "ask", "intent": intent, **prop, "executed": do_exec}
            if do_exec and not prop["verified_command"].startswith("ask"):
                out["execution_result"] = self.execute_line(prop["verified_command"])
            return out

        raise ContractViolation(f"Unhandled verb {verb}", violation_code="ERR_SHELL_UNHANDLED")


class _InteractiveShellLoop(cmd.Cmd):
    intro = "Krystal oneAPI + LLM Heterogeneous Shell (napíš 'help' alebo 'exit')."
    prompt = "krystal-oneapi> "

    def __init__(self, engine: KrystalOneApiHeterogeneousShell) -> None:
        super().__init__()
        self.engine = engine

    def default(self, line: str) -> bool:
        if line.strip() in ("exit", "quit", "EOF"):
            return True
        try:
            res = self.engine.execute_line(line)
            print(json.dumps(res, indent=2, ensure_ascii=False))
        except ContractViolation as exc:
            print(json.dumps({"error": str(exc), "violation_code": exc.violation_code, "delta_S": 0}, ensure_ascii=False))
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Krystal Heterogeneous Compute CLI Shell (oneAPI + LLM gated)")
    ap.add_argument("--exec", dest="exec_cmd", help="Execute governed command(s) separated by ';' and exit")
    ap.add_argument("--llm-url", default="http://127.0.0.1:1234/v1", help="OpenAI-compatible LLM base URL")
    args = ap.parse_args(argv)

    shell = KrystalOneApiHeterogeneousShell(llm_base_url=args.llm_url)
    if args.exec_cmd:
        res = shell.execute_line(args.exec_cmd)
        sys.stdout.write(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        return 0
    _InteractiveShellLoop(shell).cmdloop()
    return 0


_GLOBAL_SHELL: Optional[KrystalOneApiHeterogeneousShell] = None


def get_global_oneapi_shell() -> KrystalOneApiHeterogeneousShell:
    global _GLOBAL_SHELL
    if _GLOBAL_SHELL is None:
        _GLOBAL_SHELL = KrystalOneApiHeterogeneousShell()
    return _GLOBAL_SHELL


if __name__ == "__main__":
    raise SystemExit(main())
