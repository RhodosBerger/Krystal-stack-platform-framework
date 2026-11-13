"""CLI Runner & Paired A/B Verifier for `projects/sovereign_openvino_companion_v2`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from projects.sovereign_openvino_companion_v2.smart_tuner import SovereignOpenVinoCompanionV2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sovereign-openvino-companion-v2")
    parser.add_argument("--run-session", action="store_true")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--ab-benchmark", type=int, default=0)
    parser.add_argument("--candidate-nodes", action="store_true")
    parser.add_argument("--morph-sense", action="store_true")
    parser.add_argument("--compile-binary-ratios", action="store_true")
    parser.add_argument("--stack-flows", action="store_true")
    parser.add_argument("--doc-query", type=str, default="")
    parser.add_argument("--artifact-tree-query", action="store_true")
    parser.add_argument("--compose-constraints", action="store_true")
    parser.add_argument("--probabilistic-authority", action="store_true")
    parser.add_argument("--sprint-kg-orchestrate", action="store_true")
    parser.add_argument("--thermodynamic-vision", action="store_true")
    parser.add_argument("--autonomous-os-microservices", action="store_true")
    parser.add_argument("--epistemic-sql-scales", action="store_true")
    parser.add_argument("--reversible-binary-correlate", action="store_true")
    parser.add_argument("--spatial-cruise-scheduler", action="store_true")
    parser.add_argument("--realtime-inspector-governor", action="store_true")
    parser.add_argument("--ux-privilege-framework", action="store_true")
    parser.add_argument("--hexadim-log-morph", action="store_true")
    parser.add_argument("--epistemic-memory-transducer", action="store_true")
    parser.add_argument("--socket-thread-governance", action="store_true")
    parser.add_argument("--exploration-benchmarks", action="store_true")
    parser.add_argument("--tectonic-mesh-gearbox", action="store_true")
    parser.add_argument("--vulkan-endpoint-extension", action="store_true")
    parser.add_argument("--zero-copy-socket-observer", action="store_true")
    parser.add_argument("--conditioning-z", type=float, default=0.65)
    parser.add_argument("--scale-influence-z", type=float, default=0.50)
    parser.add_argument("--emit-bin-dir", type=str, default="")
    parser.add_argument("--morph-factor", type=float, default=0.618034)
    parser.add_argument("--conflict-fork", action="store_true")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--reset-ledger", action="store_true")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir) if args.log_dir else None
    bin_dir = Path(args.emit_bin_dir) if args.emit_bin_dir else None
    v2 = SovereignOpenVinoCompanionV2(log_dir=log_dir, reset_ledger=args.reset_ledger)
    out: Dict[str, Any] = {}
    any_explicit = (
        args.run_session
        or args.ab_benchmark > 0
        or args.candidate_nodes
        or args.morph_sense
        or args.compile_binary_ratios
        or args.stack_flows
        or bool(args.doc_query.strip())
        or args.artifact_tree_query
        or args.compose_constraints
        or args.probabilistic_authority
        or args.sprint_kg_orchestrate
        or args.thermodynamic_vision
        or args.autonomous_os_microservices
        or args.epistemic_sql_scales
        or args.reversible_binary_correlate
        or args.spatial_cruise_scheduler
        or args.realtime_inspector_governor
        or args.ux_privilege_framework
        or args.hexadim_log_morph
        or args.epistemic_memory_transducer
        or args.socket_thread_governance
        or args.exploration_benchmarks
        or args.tectonic_mesh_gearbox
        or args.vulkan_endpoint_extension
        or args.zero_copy_socket_observer
    )
    if args.run_session or not any_explicit:
        out["session"] = v2.run_smart_session(num_frames=args.frames)
    if args.ab_benchmark > 0:
        out["ab_benchmark"] = v2.run_v1_vs_v2_ab_benchmark(iterations=args.ab_benchmark)
    if args.candidate_nodes:
        out["candidate_nodes"] = v2.prototype_candidate_runtime_nodes()
    if args.morph_sense or args.compile_binary_ratios or args.stack_flows:
        full_syn = v2.run_sensory_binary_and_flow_synthesis(
            morph_factor=args.morph_factor,
            contradictory_evidence=args.conflict_fork,
            output_bin_dir=bin_dir,
        )
        if args.morph_sense:
            out["sensory_morphism"] = full_syn["sensory_morphism"]
        if args.compile_binary_ratios:
            out["binary_aspect_ratios"] = full_syn["binary_aspect_ratios"]
        if args.stack_flows:
            out["process_microservice_conclusion_audit"] = full_syn["process_microservice_conclusion_audit"]
            out["new_stack_flows"] = full_syn["new_stack_flows"]
            out["hypervisor_matrix_and_knowledge"] = full_syn["hypervisor_matrix_and_knowledge"]
            out["survival_governor_and_motivation"] = full_syn["survival_governor_and_motivation"]
        out["chain_valid"] = full_syn["chain_valid"]
        out["log_audit"] = full_syn["log_audit"]
    if args.doc_query.strip():
        from projects.sovereign_openvino_companion_v2.markdown_kg_module import (
            MarkdownBehavioralKnowledgeGraphModule,
        )

        kg_mod = MarkdownBehavioralKnowledgeGraphModule(logger=v2.logger)
        out["markdown_knowledge_graph"] = kg_mod.derive_knowledge_graph_and_outcomes(
            query=args.doc_query.strip(), companion_v2=v2
        )
        chain_ok, audit = v2.logger.verify_chain_integrity()
        out["chain_valid"] = chain_ok
        out["log_audit"] = audit
    if args.artifact_tree_query:
        from projects.sovereign_openvino_companion_v2.markdown_kg_module import (
            MarkdownBehavioralKnowledgeGraphModule,
        )

        out["beneficial_tree_artifacts"] = MarkdownBehavioralKnowledgeGraphModule().query_beneficial_artifacts_in_tree()
    if args.compose_constraints or args.probabilistic_authority:
        cap_out = v2.run_constraint_composer_and_probabilistic_authority()
        out["constraint_authority_and_calibration"] = cap_out
        out["chain_valid"] = cap_out["chain_valid"]
        out["log_audit"] = cap_out["log_audit"]
    if (
        args.sprint_kg_orchestrate
        or args.thermodynamic_vision
        or args.autonomous_os_microservices
        or args.epistemic_sql_scales
    ):
        sprint_out = v2.run_morph_vision_and_sprint_kg_orchestration(
            query=args.doc_query.strip() or "Compare thermal survival and foveated vision routing",
            conditioning_z=args.conditioning_z,
            scale_influence_z=args.scale_influence_z,
        )
        out["morph_vision_sprint_kg"] = sprint_out
        out["chain_valid"] = sprint_out["chain_valid"]
        out["log_audit"] = sprint_out["log_audit"]
    if (
        args.reversible_binary_correlate
        or args.spatial_cruise_scheduler
        or args.realtime_inspector_governor
        or args.ux_privilege_framework
        or args.hexadim_log_morph
        or args.epistemic_memory_transducer
        or args.socket_thread_governance
        or args.exploration_benchmarks
        or args.tectonic_mesh_gearbox
        or args.vulkan_endpoint_extension
        or args.zero_copy_socket_observer
    ):
        rev_out = v2.run_reversible_correlation_and_spatial_cruise()
        out["reversible_correlation_and_cruise"] = rev_out
        out["chain_valid"] = rev_out["chain_valid"]
        out["log_audit"] = rev_out["log_audit"]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
