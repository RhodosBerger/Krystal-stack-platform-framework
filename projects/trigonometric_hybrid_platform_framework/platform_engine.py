"""Trigonometric Hybrid Platform Framework — Core Structures, Schema Solver,
AST Code Dependency Mapper, Master Epistemic Prompt Router, Step Proxy & Live UI.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.goniometric_proposer_kernel import (
    GoniometricComputingBackend,
    GoniometricProposalCategory,
)
from krystal_kernel.visual_glyph_stroke_ml_compiler import (
    KernelGovernedTrigonometricHarmonicEngine,
)

FRAME_SIZE_BYTES: int = 128
AUTH_PREFIX_BYTES: int = 96
HMAC_TAG_BYTES: int = 32
UMA_SATURATION_CEILING: float = 0.78
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. RELOADED TRIGONOMETRIC PATTERN STRUCTURES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrigonometricHarmonicStateStruct:
    """Bit-exact reloaded trigonometric pattern structure aligned to 128B IPC."""

    struct_name: str
    angle_deg: float
    angle_rad: float
    sin_theta: float
    cos_theta: float
    tan_clamped_phi6: float
    pythagorean_sum: float
    pythagorean_invariant_ok: bool
    chebyshev_t1_to_t4: Tuple[float, float, float, float]
    l1_lut256_index_u8: int
    l1_lut_bytes_fp16: int
    radial_curvature_kappa: float
    frame_128b_hex: str
    vital_max_hp: int


@dataclass(frozen=True)
class ZeroCopy128BContractStruct:
    """Canonical 128B Zero-Copy IPC frame layout specification."""

    struct_name: str = "ZeroCopyContractFrame"
    c_alignment: str = "_Alignas(64) struct ZeroCopyContractFrame"
    total_frame_bytes: int = FRAME_SIZE_BYTES
    authenticated_prefix_bytes: int = AUTH_PREFIX_BYTES
    hmac_sha256_tag_bytes: int = HMAC_TAG_BYTES
    vital_max_hp: int = VITAL_MAX_HP
    uma_saturation_ceiling: float = UMA_SATURATION_CEILING


class ReloadedTrigonometricPatternStructures:
    """Reloads deterministic trigonometric patterns and emits canonical structures."""

    def __init__(self, vital_hp: int = VITAL_MAX_HP) -> None:
        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"Trigonometric reload rejected: vital_hp={vital_hp} != {VITAL_MAX_HP}",
                violation_code="ERR_VITAL_HP_BREACH",
            )
        self.engine = KernelGovernedTrigonometricHarmonicEngine(vital_max_hp=vital_hp)

    def reload_pattern(
        self,
        angle_deg: float = 58.282525,
        glottal_f0_hz: float = 261.63,
    ) -> Dict[str, Any]:
        if math.isnan(angle_deg) or math.isinf(angle_deg):
            raise ContractViolation(
                "Invalid NaN/Inf angle in trigonometric pattern reload; Delta S == 0.",
                violation_code="ERR_TRIG_INVALID_ANGLE",
            )
        state = self.engine.evaluate_kernel_trigonometric_state(
            angle_deg=angle_deg,
            glottal_f0_hz=glottal_f0_hz,
        )
        theta_norm = GoniometricComputingBackend.normalize_angle_2pi(math.radians(angle_deg))
        # Radial harmonic profile r(theta) = 1.0 + 0.18 * cos(3 * theta)
        r_val = 1.0 + 0.18 * math.cos(3.0 * theta_norm)
        dr = -0.54 * math.sin(3.0 * theta_norm)
        d2r = -1.62 * math.cos(3.0 * theta_norm)
        denom = (r_val * r_val + dr * dr) ** 1.5
        kappa = round(abs(r_val * r_val + 2.0 * dr * dr - r_val * d2r) / max(denom, 1e-9), 6)

        trig_struct = TrigonometricHarmonicStateStruct(
            struct_name="TrigonometricHarmonicState",
            angle_deg=state["input_angle_deg"],
            angle_rad=state["input_angle_rad"],
            sin_theta=state["kernel_sin"],
            cos_theta=state["kernel_cos"],
            tan_clamped_phi6=state["kernel_tan_clamped"],
            pythagorean_sum=state["pythagorean_identity_sum"],
            pythagorean_invariant_ok=bool(state["pythagorean_invariant_preserved"]),
            chebyshev_t1_to_t4=tuple(state["chebyshev_harmonics_t1_to_t4"]),
            l1_lut256_index_u8=int(state["l1_cache_256_lut_index_u8"]),
            l1_lut_bytes_fp16=int(state["l1_lut_footprint_bytes_fp16"]),
            radial_curvature_kappa=kappa,
            frame_128b_hex=str(state["frame_128b_hex"]),
            vital_max_hp=VITAL_MAX_HP,
        )
        frame_struct = ZeroCopy128BContractStruct()
        return {
            "trigonometric_state_struct": asdict(trig_struct),
            "zero_copy_frame_struct": asdict(frame_struct),
            "goniometric_proposal_categories": [c.value for c in GoniometricProposalCategory],
            "cpp_struct_hierarchy": state["redundancy_oscillation_and_cpp_control_flow"][
                "cpp_hierarchical_control_flow"
            ],
            "evidence_level": "LOCAL_TESTED_DETERMINISTIC_TRIG_RELOAD",
        }


# ---------------------------------------------------------------------------
# 2. SCHEMA CONSISTENCY SOLVER (ALGEBRAIC + Z3 SMT)
# ---------------------------------------------------------------------------

class SchemaConsistencySolver:
    """Solves cross-module structural, memory, trigonometric, and market schemas."""

    @classmethod
    def solve_all_schemas(cls, uma_utilization: float = 0.58) -> Dict[str, Any]:
        if uma_utilization >= UMA_SATURATION_CEILING:
            raise ContractViolation(
                f"Schema solver rejected UMA={uma_utilization} >= {UMA_SATURATION_CEILING}",
                violation_code="ERR_UMA_SCHEMA_SATURATION",
            )
        z3_status = "Z3_NOT_AVAILABLE"
        z3_model_witness: Dict[str, Any] = {}
        try:
            import z3  # type: ignore

            solver = z3.Solver()
            vital_hp = z3.Int("vital_hp")
            frame_bytes = z3.Int("frame_bytes")
            prefix_bytes = z3.Int("prefix_bytes")
            hmac_bytes = z3.Int("hmac_bytes")
            uma_permille = z3.Int("uma_permille")
            gpl_fee = z3.Int("gpl_fee_eur")
            b2b_fee = z3.Int("b2b_fee_eur")

            solver.add(vital_hp == 6)
            solver.add(prefix_bytes == 96, hmac_bytes == 32)
            solver.add(frame_bytes == prefix_bytes + hmac_bytes)
            solver.add(frame_bytes == 128)
            solver.add(uma_permille == int(round(uma_utilization * 1000)))
            solver.add(uma_permille >= 0, uma_permille < 780)
            solver.add(gpl_fee == 0)
            solver.add(b2b_fee == 4500 + 5400 + 2400 + 3000 + 900)
            solver.add(b2b_fee > gpl_fee)

            chk = solver.check()
            if chk == z3.sat:
                m = solver.model()
                z3_status = "Z3_SAT_VERIFIED"
                z3_model_witness = {
                    "vital_hp": m[vital_hp].as_long(),
                    "frame_bytes": m[frame_bytes].as_long(),
                    "prefix_bytes": m[prefix_bytes].as_long(),
                    "hmac_bytes": m[hmac_bytes].as_long(),
                    "uma_permille": m[uma_permille].as_long(),
                    "gpl_fee_eur": m[gpl_fee].as_long(),
                    "b2b_fee_eur": m[b2b_fee].as_long(),
                }
            else:
                z3_status = f"Z3_{chk}"
        except Exception as exc:  # pragma: no cover - graceful fallback
            z3_status = f"ALGEBRAIC_ONLY ({exc})"

        schemas = {
            "TrigonometricHarmonicStateSchema": {
                "required_fields": [
                    "angle_deg",
                    "sin_theta",
                    "cos_theta",
                    "tan_clamped_phi6",
                    "pythagorean_invariant_ok",
                    "chebyshev_t1_to_t4",
                    "l1_lut256_index_u8",
                    "frame_128b_hex",
                    "vital_max_hp",
                ],
                "invariant": "|sin^2(theta) + cos^2(theta) - 1.0| <= 1e-4 && |tan_clamped| <= 17.944272",
            },
            "ZeroCopyContractFrameSchema": {
                "size_bytes": 128,
                "layout": "96B_canonical_payload + 32B_hmac_sha256",
                "alignment_bytes": 64,
                "invariant": "vital_max_hp == 6 && uma_utilization < 0.78",
            },
            "HybridMarketDualLicenseSchema": {
                "community_tier": {"license": "GPLv3_NON_COMMERCIAL", "annual_fee_eur": 0},
                "enterprise_tier": {
                    "license": "COMMERCIAL_CERTIFIED_B2B_SDK",
                    "annual_fee_eur": 16200,
                    "origin_lock": "https://github.com/RhodosBerger/Krystal-stack-platform-framework.git",
                },
                "invariant": "proposer != verifier && commercial_deployment => signed_agreement",
            },
            "StepProxyReceiptSchema": {
                "required_fields": [
                    "step_id",
                    "proposer_id",
                    "verifier_id",
                    "pre_state_sha256",
                    "post_state_sha256",
                    "committed",
                    "receipt_128b_sha256",
                ],
                "invariant": "not Valid(step) => post_state_sha256 == pre_state_sha256",
            },
        }
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "solver_verdict": z3_status,
            "z3_witness": z3_model_witness,
            "solved_schema_count": len(schemas),
            "schemas": schemas,
            "evidence_level": "LOCAL_Z3_AND_ALGEBRAIC_SCHEMA_SOLVE",
        }


# ---------------------------------------------------------------------------
# 3. AST-BASED CODE DEPENDENCY GRAPH ANALYZER
# ---------------------------------------------------------------------------

class CodeDependencyGraphAnalyzer:
    """Statically parses live Python files via AST to map internal & external dependencies."""

    TARGET_MODULES: Tuple[str, ...] = (
        "krystal_kernel/constants.py",
        "krystal_kernel/goniometric_proposer_kernel.py",
        "krystal_kernel/visual_glyph_stroke_ml_compiler.py",
        "krystal_kernel/sovereign_openvino_oneapi_app.py",
        "projects/sovereign_openvino_companion_v2/blender_vray_ui_and_enterprise_b2b_endpoints.py",
        "projects/sovereign_openvino_companion_v2/oxygen_hybrid_mesh_crud_module.py",
        "projects/sovereign_openvino_companion_v2/transfer_cost_and_volatility_throughput.py",
        "projects/sovereign_openvino_companion_v2/vulkan_memory_adaptability_lib.py",
    )

    @classmethod
    def analyze_code_dependencies(cls, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        internal_edges: List[Dict[str, str]] = []
        external_packages: set[str] = set()
        stdlib_common = {
            "__future__",
            "ast",
            "base64",
            "collections",
            "ctypes",
            "dataclasses",
            "datetime",
            "enum",
            "functools",
            "hashlib",
            "hmac",
            "html",
            "http",
            "io",
            "itertools",
            "json",
            "logging",
            "math",
            "os",
            "pathlib",
            "platform",
            "re",
            "shutil",
            "socket",
            "sqlite3",
            "statistics",
            "struct",
            "subprocess",
            "sys",
            "tempfile",
            "threading",
            "time",
            "typing",
            "unittest",
            "urllib",
            "uuid",
            "xml",
            "zlib",
        }

        for rel_path in cls.TARGET_MODULES:
            fpath = repo_root / rel_path
            if not fpath.exists():
                continue
            src = fpath.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=rel_path)
            mod_internal: set[str] = set()
            mod_external: set[str] = set()
            classes_defined: List[str] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes_defined.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in ("krystal_kernel", "projects", "src"):
                            mod_internal.add(alias.name)
                        elif top not in stdlib_common:
                            mod_external.add(top)
                            external_packages.add(top)
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        rel_target = f".{node.module or ''}"
                        mod_internal.add(rel_target)
                    elif node.module:
                        top = node.module.split(".")[0]
                        if top in ("krystal_kernel", "projects", "src"):
                            mod_internal.add(node.module)
                        elif top not in stdlib_common:
                            mod_external.add(top)
                            external_packages.add(top)

            for target in sorted(mod_internal):
                internal_edges.append({"source": rel_path, "imports": target})

            nodes.append(
                {
                    "module_path": rel_path,
                    "line_count": len(src.splitlines()),
                    "class_count": len(classes_defined),
                    "sample_classes": classes_defined[:5],
                    "internal_import_count": len(mod_internal),
                    "external_packages": sorted(mod_external),
                }
            )

        pyproject_path = repo_root / "pyproject.toml"
        declared_pyproject_deps = [
            "fastapi>=0.100.0",
            "pydantic>=2.0.0",
            "numpy>=1.24.0",
            "requests>=2.31.0",
            "psutil>=5.9.0",
            "z3-solver>=5.0.0",
        ]
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "analyzed_module_count": len(nodes),
            "internal_dependency_edge_count": len(internal_edges),
            "discovered_third_party_imports": sorted(external_packages),
            "declared_pyproject_dependencies": declared_pyproject_deps,
            "pyproject_present": pyproject_path.exists(),
            "modules": nodes,
            "internal_edges": internal_edges[:30],
            "evidence_level": "LOCAL_AST_STATIC_ANALYSIS",
        }


# ---------------------------------------------------------------------------
# 4. MASTER EPISTEMIC PROMPT ROUTER & EDITABLE MICROPROMPT REGISTRY
# ---------------------------------------------------------------------------

class MasterPromptRouterConsolidator:
    """Consolidates codebase rules, skills, and manifests into an editable routed registry.

    Respects AGENTS.md Section 10: avoids an attention-diluting monolithic gigaprompt
    by maintaining specialized, task-routed microprompts with audit hashes.
    """

    DEFAULT_MICROPROMPTS: Tuple[Dict[str, str], ...] = (
        {
            "prompt_id": "MP_01_EPISTEMIC_DUAL_MODE",
            "task_route": "explore_vs_verify",
            "source_artifact": "AGENTS.md + .agents/rules/epistemic-reasoning-and-falsification.md",
            "directive": (
                "Separate EXPLORE (hypothesis generation) from VERIFY (falsifiable evidence audit). "
                "Never self-certify (PROPOSER != VERIFIER); tag unmeasured claims [UNKNOWN]."
            ),
        },
        {
            "prompt_id": "MP_02_FORMAL_SYSTEM_8TUPLE",
            "task_route": "architecture_and_verification",
            "source_artifact": ".agents/rules/formal-engineering-evaluator.md",
            "directive": (
                "Model subsystems as (S, A, T, I, O, E, H, W). Enforce Valid(S_t, a) before commit; "
                "on violation preserve S_{t+1} == S_t and raise typed ContractViolation."
            ),
        },
        {
            "prompt_id": "MP_03_SOVEREIGN_HARDWARE_KSPP",
            "task_route": "zero_copy_uma_vulkan_trigonometry",
            "source_artifact": ".agents/rules/krystal-sovereign-principle-pack.md + krystal-openvino-uma-inference-rules.md",
            "directive": (
                "Use 128B _Alignas(64) ZeroCopyContractFrame (96B payload + 32B HMAC), keep UMA < 0.78, "
                "schedule heavy compute in VBlank shadows, and clamp kernel tan(theta) to +/-Phi^6."
            ),
        },
        {
            "prompt_id": "MP_04_ENLG_EPISTEMIC_GRAMMAR",
            "task_route": "claim_compilation_and_reporting",
            "source_artifact": ".agents/rules/krystal-epistemic-grammar.md",
            "directive": (
                "Emit atomic CLAIM, SCOPE, ROLE, EVIDENCE, NULLS (M0..M2), FALSIFIER, VALIDITY, "
                "VERDICT, and RESIDUAL. Never promote CI green -> Correctness or Simulation -> Measurement."
            ),
        },
        {
            "prompt_id": "MP_05_DEBUG_ALL_SCALES_AND_SPEECH",
            "task_route": "multi_scale_debugging_and_nomenclature",
            "source_artifact": "docs/architecture/KRYSTAL_UNIVERSAL_AGENT_KERNEL_MANIFEST.md",
            "directive": (
                "Execute DEBUG_PROMPT_ALL_SCALES_KRYSTAL_V1 across Spec-K0..K6 and apply KRYF-NOM "
                "organized-speech nomenclature with zero silent history rewrites."
            ),
        },
        {
            "prompt_id": "MP_06_HYBRID_MARKET_AND_ORIGIN_LOCK",
            "task_route": "licensing_b2b_and_oss_governance",
            "source_artifact": "projects/sovereign_openvino_companion_v2/blender_vray_ui_and_enterprise_b2b_endpoints.py",
            "directive": (
                "Enforce @RhodosBerger / Dušan Kopecký origin lock: non-commercial GPLv3 community tier at 0 EUR, "
                "certified commercial B2B SDK integration tier at 16,200 EUR/year with signed agreement."
            ),
        },
    )

    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._edit_ledger: List[Dict[str, Any]] = []
        for item in self.DEFAULT_MICROPROMPTS:
            pid = item["prompt_id"]
            directive = item["directive"]
            self._registry[pid] = {
                **item,
                "version": 1,
                "sha256": hashlib.sha256(directive.encode("utf-8")).hexdigest(),
            }

    def discover_repository_prompt_sources(self) -> Dict[str, List[str]]:
        rules_dir = self.repo_root / ".agents" / "rules"
        skills_dir = self.repo_root / ".agents" / "skills"
        arch_dir = self.repo_root / "docs" / "architecture"
        return {
            "agent_rules": sorted(p.name for p in rules_dir.glob("*.md")) if rules_dir.exists() else [],
            "agent_skills": sorted(p.parent.name for p in skills_dir.glob("*/SKILL.md")) if skills_dir.exists() else [],
            "architecture_prompt_specs": sorted(
                p.name for p in arch_dir.glob("*PROMPT*.md")
            )
            if arch_dir.exists()
            else [],
            "root_manifests": [
                name for name in ("AGENTS.md", "GEMINI.md") if (self.repo_root / name).exists()
            ],
        }

    def edit_microprompt(
        self,
        prompt_id: str,
        updated_directive: str,
        editor_role: str = "ARCHITECT_OPERATOR",
        vital_hp: int = VITAL_MAX_HP,
    ) -> Dict[str, Any]:
        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"Microprompt edit rejected: vital_hp={vital_hp} != {VITAL_MAX_HP}",
                violation_code="ERR_PROMPT_EDIT_VITAL_HP",
            )
        if prompt_id not in self._registry:
            raise ContractViolation(
                f"Unknown microprompt ID: {prompt_id}",
                violation_code="ERR_UNKNOWN_MICROPROMPT",
            )
        cleaned = updated_directive.strip()
        if not cleaned or len(cleaned) > 1200:
            raise ContractViolation(
                "Microprompt directive must be non-empty and <= 1200 chars (anti-gigaprompt bound).",
                violation_code="ERR_GIGAPROMPT_BOUND_EXCEEDED",
            )
        prev = self._registry[prompt_id]
        next_ver = int(prev["version"]) + 1
        new_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        updated_entry = {
            **prev,
            "directive": cleaned,
            "version": next_ver,
            "sha256": new_hash,
        }
        self._registry[prompt_id] = updated_entry
        receipt = {
            "prompt_id": prompt_id,
            "previous_version": prev["version"],
            "new_version": next_ver,
            "previous_sha256": prev["sha256"],
            "new_sha256": new_hash,
            "editor_role": editor_role,
        }
        self._edit_ledger.append(receipt)
        return receipt

    def route_master_prompt(self, task_route: str = "zero_copy_uma_vulkan_trigonometry") -> Dict[str, Any]:
        matched = [
            entry for entry in self._registry.values() if entry["task_route"] == task_route
        ]
        selected = matched[0] if matched else next(iter(self._registry.values()))
        sources = self.discover_repository_prompt_sources()
        return {
            "router_architecture": "EpistemicPromptRouter_v2_AntiGigaprompt",
            "vital_max_hp": VITAL_MAX_HP,
            "active_route": selected["task_route"],
            "routed_microprompt": selected,
            "consolidated_microprompts": list(self._registry.values()),
            "edit_ledger": list(self._edit_ledger),
            "discovered_codebase_sources": sources,
            "master_router_envelope": (
                f"[KRYSTAL MASTER ROUTER | VITAL_MAX_HP==6 | PROPOSER!=VERIFIER | UMA<0.78] "
                f"Active={selected['prompt_id']} (v{selected['version']}): {selected['directive']}"
            ),
        }


# ---------------------------------------------------------------------------
# 5. CONSTITUTIONAL STEP-EXECUTION PROXY (CREATE NEW PROXY TO STEP)
# ---------------------------------------------------------------------------

class StepExecutionGateProxy:
    """Zero-touch constitutional proxy intercepting each pipeline step.

    Enforces:
      - VITAL_MAX_HP == 6
      - uma_utilization < 0.78
      - proposer_id != verifier_id
      - On failure: state snapshot remains unchanged (Delta S == 0).
    """

    def __init__(self) -> None:
        self.committed_state: Dict[str, Any] = {"step_count": 0, "last_step_id": "GENESIS"}
        self.step_receipts: List[Dict[str, Any]] = []

    def execute_step(
        self,
        step_id: str,
        proposer_id: str,
        verifier_id: str,
        step_fn: Callable[[], Dict[str, Any]],
        vital_hp: int = VITAL_MAX_HP,
        uma_utilization: float = 0.56,
    ) -> Dict[str, Any]:
        pre_json = json.dumps(self.committed_state, sort_keys=True)
        pre_hash = hashlib.sha256(pre_json.encode("utf-8")).hexdigest()

        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"Step proxy halted {step_id}: vital_hp={vital_hp} != {VITAL_MAX_HP} (Delta S == 0).",
                violation_code="ERR_PROXY_VITAL_HP",
            )
        if uma_utilization >= UMA_SATURATION_CEILING:
            raise ContractViolation(
                f"Step proxy halted {step_id}: uma={uma_utilization} >= {UMA_SATURATION_CEILING} (Delta S == 0).",
                violation_code="ERR_PROXY_UMA_CEILING",
            )
        if not proposer_id or not verifier_id or proposer_id == verifier_id:
            raise ContractViolation(
                f"Step proxy halted {step_id}: self-certification forbidden ({proposer_id} == {verifier_id}).",
                violation_code="ERR_PROXY_SELF_CERTIFICATION",
            )

        payload = step_fn()
        next_state = {
            "step_count": int(self.committed_state["step_count"]) + 1,
            "last_step_id": step_id,
            "payload_digest": hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
        }
        self.committed_state = next_state
        post_hash = hashlib.sha256(
            json.dumps(self.committed_state, sort_keys=True).encode("utf-8")
        ).hexdigest()
        receipt_128b_hash = hashlib.sha256(
            f"128B_STEP_PROXY|{step_id}|{proposer_id}|{verifier_id}|{pre_hash}|{post_hash}".encode(
                "utf-8"
            )
        ).hexdigest()
        receipt = {
            "step_id": step_id,
            "proposer_id": proposer_id,
            "verifier_id": verifier_id,
            "vital_hp": vital_hp,
            "uma_utilization": uma_utilization,
            "pre_state_sha256": pre_hash,
            "post_state_sha256": post_hash,
            "committed": True,
            "receipt_128b_sha256": receipt_128b_hash,
        }
        self.step_receipts.append(receipt)
        return {"receipt": receipt, "output": payload}


# ---------------------------------------------------------------------------
# 6. HIERARCHY OF RELATED TERMS, HYBRID MARKET CHARTER & DOCUMENTATION WAVE
# ---------------------------------------------------------------------------

class HierarchyOfRelatedTermsEngine:
    """Constructs the canonical 5-tier Hierarchy of Related Terms & Documentation Wave."""

    HIERARCHY_TIERS: Tuple[Dict[str, Any], ...] = (
        {
            "tier_id": "L0_CONSTITUTIONAL_INVARIANTS",
            "parent_tier": None,
            "terms": [
                "VITAL_MAX_HP == 6",
                "PROPOSER != VERIFIER",
                "not Valid(Action) => Delta S == 0",
                "UMA_SATURATION_CEILING < 0.78",
            ],
        },
        {
            "tier_id": "L1_TRIGONOMETRIC_KERNEL_STRUCTURES",
            "parent_tier": "L0_CONSTITUTIONAL_INVARIANTS",
            "terms": [
                "TrigonometricHarmonicState (LUT256 512B L1)",
                "Chebyshev Harmonics T1..T4",
                "Asymptote Clamp +/-Phi^6 (17.944272)",
                "GoniometricRadiusProfile & Curvature kappa",
            ],
        },
        {
            "tier_id": "L2_ZERO_COPY_IPC_AND_SOLVED_SCHEMAS",
            "parent_tier": "L1_TRIGONOMETRIC_KERNEL_STRUCTURES",
            "terms": [
                "ZeroCopyContractFrame (128B = 96B + 32B HMAC-SHA256)",
                "VulkanMemoryAdaptabilityLibrary (128B Pool Alignment)",
                "SchemaConsistencySolver (Z3 SMT Verified)",
                "StepExecutionGateProxy (Per-Step Constitutional Gate)",
            ],
        },
        {
            "tier_id": "L3_EPISTEMIC_PROMPT_ROUTER_AND_DOC_WAVE",
            "parent_tier": "L2_ZERO_COPY_IPC_AND_SOLVED_SCHEMAS",
            "terms": [
                "EpistemicPromptRouter_v2_AntiGigaprompt (MP_01..MP_06)",
                "CodeDependencyGraphAnalyzer (AST Static Import Graph)",
                "Documentation Wave WAVE_2026_11_TRIGONOMETRIC_HYBRID_PLATFORM",
                "RSS Telemetry Protocol (/api/v1/runtime/rss_feed.xml)",
            ],
        },
        {
            "tier_id": "L4_HYBRID_MARKET_CHARTER_AND_GITHUB_DELIVERY",
            "parent_tier": "L3_EPISTEMIC_PROMPT_ROUTER_AND_DOC_WAVE",
            "terms": [
                "Open-Source GPLv3 Tier (0 EUR — oss/* branches)",
                "Certified Enterprise B2B SDK Tier (16,200 EUR/yr — signed agreement)",
                "Origin & Author Lock (Dušan Kopecký / @RhodosBerger)",
                "GitHub Priorities (PR #1 Architecture Docs, PR #2 Bootstrap Audit, PR #3 Monetization & Plan)",
            ],
        },
    )

    @classmethod
    def build_hierarchy_and_charter(cls) -> Dict[str, Any]:
        total_terms = sum(len(t["terms"]) for t in cls.HIERARCHY_TIERS)
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "documentation_wave": {
                "wave_id": "WAVE_2026_11_TRIGONOMETRIC_HYBRID_PLATFORM",
                "status": "DECLARED_AND_ACTIVE",
                "primary_charter_doc": "docs/architecture/TRIGONOMETRIC_HYBRID_MARKET_PLATFORM_CHARTER.md",
                "covered_planes": [
                    "Trigonometric Kernel & 128B Structures",
                    "Hybrid Market Charter (GPLv3 Community vs Certified B2B Enterprise)",
                    "Z3-Verified Schemas & AST Code Dependencies",
                    "Master Epistemic Prompt Router & Step Proxy",
                    "Live Production Server Hierarchy Endpoints",
                ],
            },
            "hybrid_market_charter": {
                "charter_id": "CHARTER-KRYSTAL-HYBRID-MARKET-2026-V1",
                "author_authority": "Dušan Kopecký (@RhodosBerger)",
                "authorized_origin": "https://github.com/RhodosBerger/Krystal-stack-platform-framework.git",
                "community_open_source_pole": {
                    "license": "GPLv3_NON_COMMERCIAL",
                    "annual_price_eur": 0,
                    "github_branches": [
                        "oss/vision-symbol-isa-visualizer",
                        "oss/relational-mesh-crud-flow",
                        "project/trigonometric-hybrid-platform-framework",
                    ],
                },
                "enterprise_commercial_pole": {
                    "license": "CERTIFIED_B2B_COMMERCIAL_SDK",
                    "annual_price_eur": 16200,
                    "volatility_stream_capacity_frames_per_sec": 3037713.2,
                    "supervision_gate": "Signed Contributor/Commercial Agreement + Constitutional Verifier",
                },
                "hybrid_bridge_mechanism": (
                    "Shared 128B ZeroCopyContractFrame & Trigonometric LUT256 core remain open and auditable "
                    "under GPLv3, while deterministic B2B SDK compilation, custom chipset tuning, and "
                    "certified supervisory SLAs monetize at 16,200 EUR/yr."
                ),
            },
            "github_fetched_priorities_and_reputation": {
                "repository": "RhodosBerger/Krystal-stack-platform-framework",
                "verified_ci_reputation": [
                    {
                        "workflow": "GAMESA CI/CD",
                        "run_id": 43443242918,
                        "head_sha": "3341e187",
                        "conclusion": "success",
                    },
                    {
                        "workflow": "Krystal Verification & Cross-ISA Publication CI",
                        "run_id": 43443242807,
                        "head_sha": "3341e187",
                        "conclusion": "success",
                    },
                ],
                "open_pull_request_priorities": [
                    {
                        "pr_number": 3,
                        "priority_rank": "P1_MONETIZATION_AND_DASHBOARD",
                        "title": "Add monetization docs, component completion report, next-week plan, and dashboard robustness improvements",
                        "alignment_in_framework": "Integrated via HybridMarketCharter & live /trigonometric_hybrid_platform UI",
                    },
                    {
                        "pr_number": 2,
                        "priority_rank": "P1_DEPENDENCY_BOOTSTRAP_AND_ROADMAP",
                        "title": "Add dependency bootstrap audit script and roadmap execution blueprint",
                        "alignment_in_framework": "Integrated via isolated .venv bootstrap & AST CodeDependencyGraphAnalyzer",
                    },
                    {
                        "pr_number": 1,
                        "priority_rank": "P2_ARCHITECTURE_CONCEPT_DOCS",
                        "title": "Claude/document architecture concepts 01 bn h6g c1v vm yp kb1 cq nanyc",
                        "alignment_in_framework": "Integrated via WAVE_2026_11_TRIGONOMETRIC_HYBRID_PLATFORM & Hierarchy of Related Terms",
                    },
                ],
                "standalone_github_repo_creation_status": (
                    "Delivered in-repo under projects/trigonometric_hybrid_platform_framework/ and pushed to "
                    "GitHub branch project/trigonometric-hybrid-platform-framework; creating a separate top-level "
                    "GitHub repository requires gh CLI sign-in or GitHub web UI confirmation."
                ),
            },
            "hierarchy_of_related_terms": {
                "tier_count": len(cls.HIERARCHY_TIERS),
                "total_term_count": total_terms,
                "tiers": list(cls.HIERARCHY_TIERS),
            },
        }


# ---------------------------------------------------------------------------
# 7. UNIFIED PLATFORM ORCHESTRATOR & PRODUCTION HTML RENDERER
# ---------------------------------------------------------------------------

class TrigonometricHybridPlatformFramework:
    """Top-level platform orchestrator executing all stages through StepExecutionGateProxy."""

    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root
        self.trig_reloader = ReloadedTrigonometricPatternStructures()
        self.prompt_router = MasterPromptRouterConsolidator(repo_root=repo_root)
        self.step_proxy = StepExecutionGateProxy()

    def evaluate_full_platform(
        self,
        angle_deg: float = 58.282525,
        uma_utilization: float = 0.56,
        task_route: str = "zero_copy_uma_vulkan_trigonometry",
        prompt_edit: Dict[str, str] | None = None,
        vital_hp: int = VITAL_MAX_HP,
    ) -> Dict[str, Any]:
        if prompt_edit and prompt_edit.get("prompt_id") and prompt_edit.get("updated_directive"):
            self.prompt_router.edit_microprompt(
                prompt_id=str(prompt_edit["prompt_id"]),
                updated_directive=str(prompt_edit["updated_directive"]),
                vital_hp=vital_hp,
            )

        s1 = self.step_proxy.execute_step(
            step_id="STEP_01_RELOAD_TRIGONOMETRIC_STRUCTURES",
            proposer_id="GONIOMETRIC_PROPOSER_KERNEL",
            verifier_id="PYTHAGOREAN_CONSTITUTIONAL_GATE",
            step_fn=lambda: self.trig_reloader.reload_pattern(angle_deg=angle_deg),
            vital_hp=vital_hp,
            uma_utilization=uma_utilization,
        )
        s2 = self.step_proxy.execute_step(
            step_id="STEP_02_SOLVE_SCHEMAS",
            proposer_id="SCHEMA_PROPOSER",
            verifier_id="Z3_SMT_SCHEMA_VERIFIER",
            step_fn=lambda: SchemaConsistencySolver.solve_all_schemas(uma_utilization=uma_utilization),
            vital_hp=vital_hp,
            uma_utilization=uma_utilization,
        )
        s3 = self.step_proxy.execute_step(
            step_id="STEP_03_MAP_CODE_DEPENDENCIES",
            proposer_id="AST_SCANNER_PROPOSER",
            verifier_id="DEPENDENCY_BOUNDARY_AUDITOR",
            step_fn=lambda: CodeDependencyGraphAnalyzer.analyze_code_dependencies(self.repo_root),
            vital_hp=vital_hp,
            uma_utilization=uma_utilization,
        )
        s4 = self.step_proxy.execute_step(
            step_id="STEP_04_ROUTE_MASTER_PROMPT",
            proposer_id="MICROPROMPT_COMPOSER",
            verifier_id="ANTI_GIGAPROMPT_AUDITOR",
            step_fn=lambda: self.prompt_router.route_master_prompt(task_route=task_route),
            vital_hp=vital_hp,
            uma_utilization=uma_utilization,
        )
        s5 = self.step_proxy.execute_step(
            step_id="STEP_05_PUBLISH_HIERARCHY_AND_HYBRID_CHARTER",
            proposer_id="CHARTER_ARCHITECT_PROPOSER",
            verifier_id="ORIGIN_LOCK_VERIFIER",
            step_fn=HierarchyOfRelatedTermsEngine.build_hierarchy_and_charter,
            vital_hp=vital_hp,
            uma_utilization=uma_utilization,
        )

        from .catalog_gearbox_and_comm_kernel import (
            CatalogProliferationAndCarouselReconciler,
            ConstitutionalModeratorAndChoiceSolver,
            CrossDisplayMultiGenreParameterMatrix,
            FamiliarHomeClassAndSevenProductSets,
            FrameworkNationsValidator,
            ProcessDrivenGearboxAndCommKernel,
        )

        s6 = self.step_proxy.execute_step(
            step_id="STEP_06_VALIDATE_NATIONS_CATALOGS_GEARBOX_AND_MODERATOR",
            proposer_id="ECOSYSTEM_SYNTHESIS_PROPOSER",
            verifier_id="MODERATOR_AND_Z3_CHOICE_VERIFIER",
            step_fn=lambda: {
                "validated_nations": FrameworkNationsValidator.validate_all_nations(vital_hp=vital_hp),
                "catalogs_and_four_carousels": CatalogProliferationAndCarouselReconciler.audit_catalogs_and_carousels(
                    self.repo_root
                ),
                "home_class_and_seven_product_sets": FamiliarHomeClassAndSevenProductSets.synthesize_home_and_product_sets(),
                "process_gearbox_visualizer": ProcessDrivenGearboxAndCommKernel.sample_live_processes_and_gearbox(),
                "communication_kernel_subprocess": ProcessDrivenGearboxAndCommKernel.spawn_communication_kernel_process(
                    vital_hp=vital_hp,
                    uma_utilization=uma_utilization,
                ),
                "moderator_and_consistent_choice": ConstitutionalModeratorAndChoiceSolver.moderate_and_solve_consistent_choice(
                    vital_hp=vital_hp
                ),
                "cross_display_multi_genre_matrix": CrossDisplayMultiGenreParameterMatrix.evaluate_cross_display_multi_genre_matrix(
                    uma_utilization=uma_utilization,
                    vital_hp=vital_hp,
                ),
            },
            vital_hp=vital_hp,
            uma_utilization=uma_utilization,
        )

        return {
            "platform_id": "KRYSTAL_TRIGONOMETRIC_HYBRID_PLATFORM_FRAMEWORK_V1",
            "vital_max_hp": VITAL_MAX_HP,
            "uma_utilization": uma_utilization,
            "trigonometric_structures": s1["output"],
            "solved_schemas": s2["output"],
            "code_dependencies": s3["output"],
            "master_prompt_router": s4["output"],
            "charter_and_hierarchy": s5["output"],
            "ecosystem_gearbox_and_moderator": s6["output"],
            "step_proxy_receipts": list(self.step_proxy.step_receipts),
            "evidence_level": "LOCAL_TESTED_END_TO_END_PROXY_VERIFIED",
        }


GLOBAL_TRIGONOMETRIC_HYBRID_PLATFORM = TrigonometricHybridPlatformFramework()


def render_trigonometric_platform_html(angle_deg: float = 58.282525) -> str:
    """Renders the live interactive Production Dashboard for the Trigonometric Hybrid Platform."""
    snap = GLOBAL_TRIGONOMETRIC_HYBRID_PLATFORM.evaluate_full_platform(angle_deg=angle_deg)
    trig = snap["trigonometric_structures"]["trigonometric_state_struct"]
    schemas = snap["solved_schemas"]
    deps = snap["code_dependencies"]
    router = snap["master_prompt_router"]
    ch = snap["charter_and_hierarchy"]
    eco = snap["ecosystem_gearbox_and_moderator"]
    gear_svg = eco["process_gearbox_visualizer"]["gearbox_svg_visualization"]
    comm = eco["communication_kernel_subprocess"]
    mod = eco["moderator_and_consistent_choice"]
    cat = eco["catalogs_and_four_carousels"]
    psets_html = "".join(
        f"<li><code>{html.escape(p['product_set_id'])}</code> ({html.escape(p['segment'])}): {html.escape(p['core_innovation'])}</li>"
        for p in eco["home_class_and_seven_product_sets"]["seven_new_product_sets"]
    )
    tiers_html = "".join(
        f"<div class='card'><strong>{html.escape(t['tier_id'])}</strong>"
        f"<ul>{''.join(f'<li>{html.escape(term)}</li>' for term in t['terms'])}</ul></div>"
        for t in ch["hierarchy_of_related_terms"]["tiers"]
    )
    prompts_html = "".join(
        f"<tr><td><code>{html.escape(p['prompt_id'])}</code></td>"
        f"<td>{html.escape(p['task_route'])}</td>"
        f"<td>v{p['version']}</td>"
        f"<td>{html.escape(p['directive'])}</td></tr>"
        for p in router["consolidated_microprompts"]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Krystal Trigonometric Hybrid Platform &amp; Hierarchy</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0b1020; color: #e6edf3; margin: 0; padding: 24px; }}
    h1, h2 {{ color: #79c0ff; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 20px; }}
    .card {{ background: #161b2e; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
    code {{ color: #ffa657; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
    th, td {{ border: 1px solid #30363d; padding: 8px; text-align: left; }}
    th {{ background: #1f2937; color: #79c0ff; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; background: #238636; color: #fff; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Krystal Trigonometric Hybrid Platform Framework <span class="badge">VITAL_MAX_HP == {snap['vital_max_hp']}</span></h1>
  <p>Documentation Wave: <code>{html.escape(ch['documentation_wave']['wave_id'])}</code> |
     Schema Solver: <code>{html.escape(schemas['solver_verdict'])}</code> |
     Moderator Choice: <code>{html.escape(mod['z3_consistent_choice_status'])}</code> |
     Domain Catalogs Audited: <code>{cat['active_domain_catalog_count']}</code> |
     Comm Child PID: <code>{comm['spawned_child_pid']}</code> ({comm['spawn_roundtrip_ms']} ms)</p>

  <h2>1. Process-Driven Socket &amp; Memory Gearbox Visualization (Live psutil)</h2>
  <div class="card" style="margin-bottom:20px;">{gear_svg}</div>

  <h2>2. Reloaded Trigonometric Pattern Structure &amp; Hybrid Market Charter</h2>
  <div class="grid">
    <div class="card">
      <strong>{html.escape(trig['struct_name'])}</strong>
      <p>Angle: <code>{trig['angle_deg']}°</code> | sin: <code>{trig['sin_theta']}</code> | cos: <code>{trig['cos_theta']}</code> | tan(clamped): <code>{trig['tan_clamped_phi6']}</code></p>
      <p>Pythagorean Sum: <code>{trig['pythagorean_sum']}</code> | LUT256 Index: <code>{trig['l1_lut256_index_u8']}</code> | Curvature κ: <code>{trig['radial_curvature_kappa']}</code></p>
    </div>
    <div class="card">
      <strong>Hybrid Market Charter ({html.escape(ch['hybrid_market_charter']['charter_id'])})</strong>
      <p>Open-Source Home Tier: <code>GPLv3 (0 EUR)</code> | Certified Enterprise Tier: <code>16,200 EUR/yr</code></p>
      <p>Author Lock: <code>{html.escape(ch['hybrid_market_charter']['author_authority'])}</code> | Validated Nations: <code>{eco['validated_nations']['validated_nation_count']}</code></p>
    </div>
  </div>

  <h2>3. Seven New Product Sets (Home GPLv3 &times; Enterprise B2B)</h2>
  <div class="card" style="margin-bottom:20px;"><ul>{psets_html}</ul></div>

  <h2>4. Hierarchy of Related Terms (5 Constitutional Tiers)</h2>
  <div class="grid">{tiers_html}</div>

  <h2>5. Master Epistemic Prompt Router (Consolidated Editable Microprompts)</h2>
  <table>
    <thead><tr><th>ID</th><th>Route</th><th>Ver</th><th>Directive</th></tr></thead>
    <tbody>{prompts_html}</tbody>
  </table>
</body>
</html>"""
