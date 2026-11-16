"""Nomenclature propedeutics node: every term of the framework must bind to real code.

A term is accepted only if
  - its binding "module:qualname" imports and resolves to a callable/class,
  - its test file exists,
and the registry reports public symbols of the package that have no term (undocumented).
This keeps vocabulary from drifting away from what the code actually does.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Dict, List

PKG = "projects.trigonometric_hybrid_platform_framework"
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

TERMS: tuple = (
    ("Connection registry", "Typed nodes and edges of the sync architecture; only the gate may write state.",
     "adaptive_cross_genre_sync:validate_connection_graph", "tests/test_adaptive_cross_genre_sync.py"),
    ("Strictest-policy join", "When two genres touch one entity, the stricter sync mode wins; observers cannot write.",
     "adaptive_cross_genre_sync:join_sync_policy", "tests/test_adaptive_cross_genre_sync.py"),
    ("Adaptive horizon controller", "Rollback horizon = ceil((EWMA + 2 sigma_upper) / tick), immediate up, down after 12 ticks.",
     "adaptive_cross_genre_sync:AdaptiveHorizonController", "tests/test_adaptive_cross_genre_sync.py"),
    ("Rollback simulation", "Deterministic client/kernel replay with SHA-256 checksum on every tick.",
     "adaptive_cross_genre_sync:simulate_adaptive_rollback", "tests/test_adaptive_cross_genre_sync.py"),
    ("Card grid", "Design-manual card view of the sync architecture (Grid / List / Hex / Hex 3D).",
     "adaptive_sync_card_grid:render_adaptive_sync_card_grid_html", "tests/test_adaptive_sync_card_grid.py"),
    ("Simulation grid 3D", "Babylon.js replay of kernel state vs. client prediction on a 19-hex board.",
     "simulation_grid_3d:render_simulation_grid_3d_html", "tests/test_simulation_grid_3d_and_production.py"),
    ("Virtual convolution pipeline", "In-memory OpenVINO Conv-ReLU-Conv-ReLU checked against NumPy before timing.",
     "virtual_convolution_pipeline:run_virtual_convolution_pipeline", "tests/test_virtual_convolution_pipeline.py"),
    ("Summation-order fingerprint", "Compares a device against references that differ only in summation order and rounding.",
     "virtual_convolution_pipeline:diagnose_bit_identity", "tests/test_virtual_convolution_pipeline.py"),
    ("Contra modules", "Adjoint (transposed conv), CG inverse and ReLU partial inverse of the convolution.",
     "virtual_convolution_contra:run_contra_modules", "tests/test_virtual_convolution_contra.py"),
    ("Pattern-over-profiles sweep", "Runs the fingerprint over geometries/seeds; verdict per hypothesis, no averaging.",
     "pattern_profile_environment:run_pattern_sweep", "tests/test_pattern_profile_environment.py"),
    ("Consistency bot", "Executable claim checks published as JSON, Markdown and PNG.",
     "consistency_bot:run_bot", "tests/test_consistency_bot.py"),
    ("BLAS K threshold", "Smallest reduction length where NumPy BLAS leaves sequential-FMA order.",
     "consistency_bot:blas_sequential_fma_threshold", "tests/test_consistency_bot.py"),
    ("Fact synthesis", "Facts with provenance from reports, induced rules, bridges and a citing narrator.",
     "fact_synthesis_engine:run_fact_synthesis", "tests/test_fact_synthesis_engine.py"),
    ("Log perception", "Utilization share per stratum from computation ledgers, with hash-link check.",
     "fact_synthesis_engine:perceive_ledgers", "tests/test_fact_synthesis_engine.py"),
    ("Module distance", "Hops in the AST import graph plus Jaccard distance of neighbours.",
     "fact_synthesis_engine:fetch_distance_between_modules", "tests/test_fact_synthesis_engine.py"),
    ("Fact-driven allocator", "Picks the device with minimal compile + n * median cost under a p95 budget.",
     "custom_allocator:FactDrivenAllocator", "tests/test_custom_allocator.py"),
    ("Ontology integrity", "Unique fact IDs, existing sources, no rule citing a missing fact.",
     "custom_allocator:check_ontology_integrity", "tests/test_custom_allocator.py"),
    ("Neuro-symbolic bridge", "LLM proposes cited sentences; the fact base keeps only GROUNDED ones.",
     "neurosymbolic_bridge:verify_answer", "tests/test_neurosymbolic_bridge.py"),
    ("Capability combat planner", "Plans and shadow-verifies Clash turns from hardware/framework dispositions.",
     "combat_turn_planner:CapabilityCombatTurnPlanner", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Framework dispositions", "Probes OpenVINO devices, Win32 ISA flags, warm CACHE_DIR and card gates.",
     "combat_turn_planner:probe_framework_dispositions", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Intel oneAPI accelerator", "CUDA-inspired device, shared tensor and async stream layer over OpenVINO.",
     "intel_accelerator_and_power_thread_governor:IntelOneApiAccelerator", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Power and thread governor", "Selects thread count and governed PL1 envelope from UMA and horizon.",
     "intel_accelerator_and_power_thread_governor:select_power_and_thread_strategy", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Performance curve benchmark", "Measures compile_ms + n * median_ms curves across CPU threads and warm GPU.",
     "intel_accelerator_and_power_thread_governor:benchmark_thread_and_performance_curves", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Chipset forecaster", "EWMA + 2*sigma_upper telemetry monitor with 12-sample hysteresis dwell.",
     "intel_accelerator_and_power_thread_governor:ChipsetTelemetryMonitorAndForecaster", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("State-mapping templates", "Six structured templates mapping runtime states to governed transitions.",
     "intel_accelerator_and_power_thread_governor:match_active_state_templates", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Prompt-matrix branch engine", "OpenVINO opset13 MatMul+Softmax prompt router checked by Freivalds O(n^2).",
     "tensor_prompt_branch_and_kb:TensorPromptBranchEngine", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Turn operation coordinator", "Maps verified Clash turn steps to real framework operations.",
     "tensor_prompt_branch_and_kb:coordinate_operations_from_turn", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Master knowledge base", "Synthesizes facts, computing-theory axioms, branches and OpenAPI 3.1 paths.",
     "tensor_prompt_branch_and_kb:sync_master_knowledge_base", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Opus intersection audit", "Cross-checks Opus reports against live ISA, CACHE_DIR and K=449 laws.",
     "configurator_and_dxdiag:analyze_opus_and_build_intersections", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Goal-driven configurator", "Compiles SHA-256-signed profiles via semantic utilities U01..U06.",
     "configurator_and_dxdiag:build_validated_configuration", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Numeric type contract", "Audits exact int state invariants vs FP32 tensors vs FP64 oracle accumulators.",
     "configurator_and_dxdiag:audit_numeric_type_contract", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Sub-K449 tiled reduction", "Partitions matmul along K <= 448 below the OpenBLAS FMA boundary.",
     "configurator_and_dxdiag:partition_reduction_below_k449", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("DxDiag memory profile", "Four-panel silicon, 128B memory-flow and PATTERN_CANDIDATE diagnostic report.",
     "configurator_and_dxdiag:build_dxdiag_memory_profile", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Blender workflow schema", "Five-stage Blender <-> Krystal oneAPI schema with AST-verified bpy bridge.",
     "configurator_and_dxdiag:build_blender_system_workflow_schema", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("OneAPI heterogeneous shell", "Governed CLI shell bridging LLM intents and oneAPI accelerators.",
     "oneapi_llm_heterogeneous_shell:KrystalOneApiHeterogeneousShell", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Shell command verifier", "Whitelist and metacharacter gate ensuring PROPOSER != VERIFIER in CLI.",
     "oneapi_llm_heterogeneous_shell:ShellCommandVerifier", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("Algebraic-theory simulator", "Couples crossover/horizon formulas with {K, K^T} Gram contraction.",
     "oneapi_llm_heterogeneous_shell:simulate_algebraic_theory_coupling", "tests/test_oneapi_shell_combat_and_configurator.py"),
    ("x86 platform findings", "Five calibrated tests for cache tiers, SMT AVX-512, subnormal FTZ/DAZ, 64B stride and 128B frames.",
     "x86_platform_findings:run_x86_platform_findings_computation", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Hex combat bandwidth solver", "Closed-form hex combinatorics and wire/silicon combat bandwidth equations E1..E8.",
     "hex_combat_bandwidth_and_world_ai:compute_full_combat_bandwidth_solution", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("World and reactive AI chain", "Couples primary Clash actions to procedural hex effects and AI counter-moves.",
     "hex_combat_bandwidth_and_world_ai:resolve_action_with_world_and_ai_chain", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Spatial sector coordinators", "Seven hierarchical coordinators with procedural frustum/LOD culling and instancing.",
     "spatial_coordinators_and_vr_hologram_compositor:evaluate_spatial_coordinators_and_procedural_render", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("VR hologram ray compositor", "Animated 3D engine-profiling holograms with analytical ray tracing and WebXR view.",
     "spatial_coordinators_and_vr_hologram_compositor:build_and_trace_hologram_scene", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Sensory CV mind-map", "OpenVINO Sobel/Laplacian sensory pass fused with telemetry into a capability mind-map.",
     "spatial_coordinators_and_vr_hologram_compositor:synthesize_vision_sensory_mindmap", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Neuro-symbolic micro-LLM", "Locally trained causal micro-transformer compiled to OpenVINO opset13 under verify_answer.",
     "neurosymbolic_genesis_discipline_and_llm:NeuroSymbolicMicroLLM", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Arena tensor navigator", "Caches 37-hex transition matrix and diffuses movement potential in OpenVINO opset13.",
     "arena_cache_optical_sockets_and_consumption:BattleArenaCacheAndTensorNavigator", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Render consumption graph", "Tracks per-turn 128B and render costs and derives Before/After optimization savings.",
     "arena_cache_optical_sockets_and_consumption:TurnConsumptionAndRenderAuditGraph", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Optical binary codec", "Packs Fresnel/LOD optical descriptors into a SHA-256-verified .kopt little-endian binary.",
     "arena_cache_optical_sockets_and_consumption:synthesize_and_write_optical_binary", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Code-sample object library", "Compiles real AST metrics of bound framework functions into 3D meshes and SVG wireframes.",
     "object_library_texture_nodes_and_synthetic_network:build_renderable_object_library", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Preallocated texture node graph", "Five-node RGBA arena texture generator executing in a zero-reallocation bytearray pool.",
     "object_library_texture_nodes_and_synthetic_network:execute_arena_texture_node_graph", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Synthetic knowledge network", "Exports provenance-tagged training triples from AST and facts for model learning.",
     "object_library_texture_nodes_and_synthetic_network:build_synthetic_knowledge_training_network", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Adaptive conversation trainer", "High-intensity continual trainer with verifier beam pruning and episodic residual memory.",
     "adaptive_2b_spatial_linguistic_trainer:AdaptiveConversationTrainer", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Convolutional study conclusion", "Convolves cross-stratum precision signals into the formal study summary and SVG projection.",
     "adaptive_2b_spatial_linguistic_trainer:compile_convolutional_study_conclusion", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Operational field-line learner", "Integrates vector streamlines over operational ranges and trains a sequence predictor.",
     "playable_fieldlines_and_game_theory:generate_operational_field_lines_and_learn", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Syntactic function chain", "Chains verified arena, tensor, fieldline and socket steps with Antigravity telemetry.",
     "playable_fieldlines_and_game_theory:execute_syntactic_function_chain", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Standardized digital game theory", "Formal (S, A, T, I, O, E, H, W) specification and termination/budget theorems.",
     "playable_fieldlines_and_game_theory:evaluate_standardized_digital_game_theory", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("WSL hypervisor configurator", "Host-calibrated .wslconfig and OSS layer templates for Windows 11 / Ubuntu WSLg.",
     "wsl_hypervisor_mortar_and_geotile_adapter:configure_wsl_ubuntu_hypervisor_bridge", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Mortar CSS trajectory solver", "Parabolic ballistic arc from cached hex state with CSS keyframes and CEP reticle.",
     "wsl_hypervisor_mortar_and_geotile_adapter:compute_mortar_css_trajectories_and_reticles", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Geotile 3D arena replica", "Metric Web-Mercator slippy-tile to 37-hex 3D terrain replica converter.",
     "wsl_hypervisor_mortar_and_geotile_adapter:convert_geotile_location_to_3d_arena_replica", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Vulkan UMA shader isomorphism", "Bijective Hex37<->UMA slot map, physical-core LUTs and GLSL 450/SPIR-V templates.",
     "wsl_hypervisor_mortar_and_geotile_adapter:compile_vulkan_spirv_shader_sets_and_isomorphisms", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Satellite 3D mesh duplicator", "Trained OpenVINO MLP extruding satellite/DEM grids into trimesh-audited .obj meshes.",
     "cluster_pid_gemma_finetune_and_mesh_ml:train_and_duplicate_satellite_3d_mesh", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Cluster entity PID registry", "Collision-free 32-bit PIDs defining cluster ring/slot, procedures and combination caps.",
     "cluster_pid_gemma_finetune_and_mesh_ml:build_cluster_entity_pid_registry", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Intranet 128B packet bus", "Binary KPKT 128 B SHA-256 inter-module packets modulating presets and WSL2 quotas.",
     "cluster_pid_gemma_finetune_and_mesh_ml:run_neurosymbolic_datalink_and_hypervisor_exchange", "tests/test_x86_hex_vr_and_study_conclusion.py"),
    ("Gemma fine-tuning sets", "Gemma chat-formatted abstraction variants and four leak-free train/eval sets.",
     "cluster_pid_gemma_finetune_and_mesh_ml:build_gemma_abstraction_variants_and_finetune_sets", "tests/test_x86_hex_vr_and_study_conclusion.py"),
)


def _public_symbols() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for f in sorted(PACKAGE_DIR.glob("*.py")):
        if f.stem.startswith("_"):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        out[f.stem] = [n.name for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
    return out


def validate_nomenclature(terms: tuple = TERMS) -> Dict[str, Any]:
    rows, bound = [], set()
    for term, definition, binding, test in terms:
        mod_name, _, qual = binding.partition(":")
        status = "RESOLVED"
        try:
            obj = importlib.import_module(f"{PKG}.{mod_name}")
            for part in qual.split("."):
                obj = getattr(obj, part)
            if not callable(obj):
                status = "NOT_CALLABLE"
        except (ImportError, AttributeError) as exc:
            status = f"UNRESOLVED: {type(exc).__name__}"
        if status == "RESOLVED" and not (REPO_ROOT / test).exists():
            status = "NO_TEST_FILE"
        bound.add((mod_name, qual.split(".")[0]))
        rows.append({"term": term, "definition": definition, "binding": binding, "test": test, "status": status})
    undocumented = {m: [s for s in syms if (m, s) not in bound] for m, syms in _public_symbols().items()}
    undocumented = {m: s for m, s in undocumented.items() if s}
    unresolved = [r["term"] for r in rows if r["status"] != "RESOLVED"]
    return {"verdict": "VERIFIED_WITH_SCOPE" if not unresolved else "FALSIFIED",
            "terms": rows, "unresolved_terms": unresolved, "undocumented_public_symbols": undocumented}


def export_glossary(path: Path = REPO_ROOT / "reports" / "nomenclature_propedeutics.md") -> Dict[str, Any]:
    res = validate_nomenclature()
    lines = ["# Nomenclature propedeutics", "", f"Verdict: **{res['verdict']}**", "",
             "| Term | Meaning | Binds to | Test | Status |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['term']} | {r['definition']} | `{r['binding']}` | `{r['test']}` | {r['status']} |" for r in res["terms"]]
    n_undoc = sum(len(v) for v in res["undocumented_public_symbols"].values())
    lines += ["", f"Public symbols without a term: {n_undoc}", ""]
    lines += [f"- `{m}`: {', '.join(s)}" for m, s in res["undocumented_public_symbols"].items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res["path"] = str(path)
    return res


if __name__ == "__main__":
    r = export_glossary()
    print(r["verdict"], "terms:", len(r["terms"]), "unresolved:", r["unresolved_terms"],
          "undocumented:", sum(len(v) for v in r["undocumented_public_symbols"].values()), r["path"])
