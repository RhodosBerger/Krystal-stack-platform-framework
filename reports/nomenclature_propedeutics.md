# Nomenclature propedeutics

Verdict: **VERIFIED_WITH_SCOPE**

| Term | Meaning | Binds to | Test | Status |
| --- | --- | --- | --- | --- |
| Connection registry | Typed nodes and edges of the sync architecture; only the gate may write state. | `adaptive_cross_genre_sync:validate_connection_graph` | `tests/test_adaptive_cross_genre_sync.py` | RESOLVED |
| Strictest-policy join | When two genres touch one entity, the stricter sync mode wins; observers cannot write. | `adaptive_cross_genre_sync:join_sync_policy` | `tests/test_adaptive_cross_genre_sync.py` | RESOLVED |
| Adaptive horizon controller | Rollback horizon = ceil((EWMA + 2 sigma_upper) / tick), immediate up, down after 12 ticks. | `adaptive_cross_genre_sync:AdaptiveHorizonController` | `tests/test_adaptive_cross_genre_sync.py` | RESOLVED |
| Rollback simulation | Deterministic client/kernel replay with SHA-256 checksum on every tick. | `adaptive_cross_genre_sync:simulate_adaptive_rollback` | `tests/test_adaptive_cross_genre_sync.py` | RESOLVED |
| Card grid | Design-manual card view of the sync architecture (Grid / List / Hex / Hex 3D). | `adaptive_sync_card_grid:render_adaptive_sync_card_grid_html` | `tests/test_adaptive_sync_card_grid.py` | RESOLVED |
| Simulation grid 3D | Babylon.js replay of kernel state vs. client prediction on a 19-hex board. | `simulation_grid_3d:render_simulation_grid_3d_html` | `tests/test_simulation_grid_3d_and_production.py` | RESOLVED |
| Virtual convolution pipeline | In-memory OpenVINO Conv-ReLU-Conv-ReLU checked against NumPy before timing. | `virtual_convolution_pipeline:run_virtual_convolution_pipeline` | `tests/test_virtual_convolution_pipeline.py` | RESOLVED |
| Summation-order fingerprint | Compares a device against references that differ only in summation order and rounding. | `virtual_convolution_pipeline:diagnose_bit_identity` | `tests/test_virtual_convolution_pipeline.py` | RESOLVED |
| Contra modules | Adjoint (transposed conv), CG inverse and ReLU partial inverse of the convolution. | `virtual_convolution_contra:run_contra_modules` | `tests/test_virtual_convolution_contra.py` | RESOLVED |
| Pattern-over-profiles sweep | Runs the fingerprint over geometries/seeds; verdict per hypothesis, no averaging. | `pattern_profile_environment:run_pattern_sweep` | `tests/test_pattern_profile_environment.py` | RESOLVED |
| Consistency bot | Executable claim checks published as JSON, Markdown and PNG. | `consistency_bot:run_bot` | `tests/test_consistency_bot.py` | RESOLVED |
| BLAS K threshold | Smallest reduction length where NumPy BLAS leaves sequential-FMA order. | `consistency_bot:blas_sequential_fma_threshold` | `tests/test_consistency_bot.py` | RESOLVED |
| Fact synthesis | Facts with provenance from reports, induced rules, bridges and a citing narrator. | `fact_synthesis_engine:run_fact_synthesis` | `tests/test_fact_synthesis_engine.py` | RESOLVED |
| Log perception | Utilization share per stratum from computation ledgers, with hash-link check. | `fact_synthesis_engine:perceive_ledgers` | `tests/test_fact_synthesis_engine.py` | RESOLVED |
| Module distance | Hops in the AST import graph plus Jaccard distance of neighbours. | `fact_synthesis_engine:fetch_distance_between_modules` | `tests/test_fact_synthesis_engine.py` | RESOLVED |
| Fact-driven allocator | Picks the device with minimal compile + n * median cost under a p95 budget. | `custom_allocator:FactDrivenAllocator` | `tests/test_custom_allocator.py` | RESOLVED |
| Ontology integrity | Unique fact IDs, existing sources, no rule citing a missing fact. | `custom_allocator:check_ontology_integrity` | `tests/test_custom_allocator.py` | RESOLVED |
| Neuro-symbolic bridge | LLM proposes cited sentences; the fact base keeps only GROUNDED ones. | `neurosymbolic_bridge:verify_answer` | `tests/test_neurosymbolic_bridge.py` | RESOLVED |
| Capability combat planner | Plans and shadow-verifies Clash turns from hardware/framework dispositions. | `combat_turn_planner:CapabilityCombatTurnPlanner` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Framework dispositions | Probes OpenVINO devices, Win32 ISA flags, warm CACHE_DIR and card gates. | `combat_turn_planner:probe_framework_dispositions` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Intel oneAPI accelerator | CUDA-inspired device, shared tensor and async stream layer over OpenVINO. | `intel_accelerator_and_power_thread_governor:IntelOneApiAccelerator` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Power and thread governor | Selects thread count and governed PL1 envelope from UMA and horizon. | `intel_accelerator_and_power_thread_governor:select_power_and_thread_strategy` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Performance curve benchmark | Measures compile_ms + n * median_ms curves across CPU threads and warm GPU. | `intel_accelerator_and_power_thread_governor:benchmark_thread_and_performance_curves` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Chipset forecaster | EWMA + 2*sigma_upper telemetry monitor with 12-sample hysteresis dwell. | `intel_accelerator_and_power_thread_governor:ChipsetTelemetryMonitorAndForecaster` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| State-mapping templates | Six structured templates mapping runtime states to governed transitions. | `intel_accelerator_and_power_thread_governor:match_active_state_templates` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Prompt-matrix branch engine | OpenVINO opset13 MatMul+Softmax prompt router checked by Freivalds O(n^2). | `tensor_prompt_branch_and_kb:TensorPromptBranchEngine` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Turn operation coordinator | Maps verified Clash turn steps to real framework operations. | `tensor_prompt_branch_and_kb:coordinate_operations_from_turn` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Master knowledge base | Synthesizes facts, computing-theory axioms, branches and OpenAPI 3.1 paths. | `tensor_prompt_branch_and_kb:sync_master_knowledge_base` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Opus intersection audit | Cross-checks Opus reports against live ISA, CACHE_DIR and K=449 laws. | `configurator_and_dxdiag:analyze_opus_and_build_intersections` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Goal-driven configurator | Compiles SHA-256-signed profiles via semantic utilities U01..U06. | `configurator_and_dxdiag:build_validated_configuration` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Numeric type contract | Audits exact int state invariants vs FP32 tensors vs FP64 oracle accumulators. | `configurator_and_dxdiag:audit_numeric_type_contract` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Sub-K449 tiled reduction | Partitions matmul along K <= 448 below the OpenBLAS FMA boundary. | `configurator_and_dxdiag:partition_reduction_below_k449` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| DxDiag memory profile | Four-panel silicon, 128B memory-flow and PATTERN_CANDIDATE diagnostic report. | `configurator_and_dxdiag:build_dxdiag_memory_profile` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Blender workflow schema | Five-stage Blender <-> Krystal oneAPI schema with AST-verified bpy bridge. | `configurator_and_dxdiag:build_blender_system_workflow_schema` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| OneAPI heterogeneous shell | Governed CLI shell bridging LLM intents and oneAPI accelerators. | `oneapi_llm_heterogeneous_shell:KrystalOneApiHeterogeneousShell` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Shell command verifier | Whitelist and metacharacter gate ensuring PROPOSER != VERIFIER in CLI. | `oneapi_llm_heterogeneous_shell:ShellCommandVerifier` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |
| Algebraic-theory simulator | Couples crossover/horizon formulas with {K, K^T} Gram contraction. | `oneapi_llm_heterogeneous_shell:simulate_algebraic_theory_coupling` | `tests/test_oneapi_shell_combat_and_configurator.py` | RESOLVED |

Public symbols without a term: 62

- `adaptive_cross_genre_sync`: SyncMode, SyncNode, SyncConnection, describe_adaptive_sync_architecture
- `adaptive_sync_card_grid`: build_card_dataset, export_standalone_html
- `catalog_gearbox_and_comm_kernel`: FrameworkNationsValidator, CatalogProliferationAndCarouselReconciler, FamiliarHomeClassAndSevenProductSets, ProcessDrivenGearboxAndCommKernel, ConstitutionalModeratorAndChoiceSolver, CrossDisplayMultiGenreParameterMatrix
- `clash_arena_page`: render_clash_arena_html
- `clash_battle_royale`: hex_distance, board_cells, ClashGame, ClashRegistry
- `combat_turn_planner`: get_cached_hw_profile, probe_isa_flags
- `configurator_and_dxdiag`: prewarm_heterogeneous_blob_pair, extract_pattern_candidates_from_diagnostics, export_dxdiag_txt, render_dxdiag_profile_html
- `consistency_bot`: BotCheck
- `fact_synthesis_engine`: synthesize_facts, induce_rules, module_distance_matrix, narrate
- `intel_accelerator_and_power_thread_governor`: IntelDeviceProperties, IntelStream, get_shared_ov_core
- `matrix_control_processes`: freivalds_verify, run_operator_mashup
- `neurosymbolic_bridge`: build_grounded_prompt, ask_llm, build_compendium
- `nomenclature_registry`: validate_nomenclature, export_glossary
- `oneapi_llm_heterogeneous_shell`: main, get_global_oneapi_shell
- `pattern_profile_environment`: ProfileSpec
- `platform_engine`: TrigonometricHarmonicStateStruct, ZeroCopy128BContractStruct, ReloadedTrigonometricPatternStructures, SchemaConsistencySolver, CodeDependencyGraphAnalyzer, MasterPromptRouterConsolidator, StepExecutionGateProxy, HierarchyOfRelatedTermsEngine, TrigonometricHybridPlatformFramework, render_trigonometric_platform_html
- `simulation_grid_3d`: ensure_babylon_vendor, hex_spiral_axial, build_trace_dataset, export_standalone_html
- `tensor_prompt_branch_and_kb`: encode_prompt_to_feature_row, build_openapi_31_strategy_spec
- `virtual_convolution_contra`: conv2d_transpose_reference, openvino_transpose_conv
- `virtual_convolution_pipeline`: conv2d_reference, reference_pipeline, build_openvino_model
