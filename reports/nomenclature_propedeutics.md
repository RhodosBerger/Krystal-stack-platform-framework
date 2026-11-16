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
| x86 platform findings | Five calibrated tests for cache tiers, SMT AVX-512, subnormal FTZ/DAZ, 64B stride and 128B frames. | `x86_platform_findings:run_x86_platform_findings_computation` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Hex combat bandwidth solver | Closed-form hex combinatorics and wire/silicon combat bandwidth equations E1..E8. | `hex_combat_bandwidth_and_world_ai:compute_full_combat_bandwidth_solution` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| World and reactive AI chain | Couples primary Clash actions to procedural hex effects and AI counter-moves. | `hex_combat_bandwidth_and_world_ai:resolve_action_with_world_and_ai_chain` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Spatial sector coordinators | Seven hierarchical coordinators with procedural frustum/LOD culling and instancing. | `spatial_coordinators_and_vr_hologram_compositor:evaluate_spatial_coordinators_and_procedural_render` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| VR hologram ray compositor | Animated 3D engine-profiling holograms with analytical ray tracing and WebXR view. | `spatial_coordinators_and_vr_hologram_compositor:build_and_trace_hologram_scene` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Sensory CV mind-map | OpenVINO Sobel/Laplacian sensory pass fused with telemetry into a capability mind-map. | `spatial_coordinators_and_vr_hologram_compositor:synthesize_vision_sensory_mindmap` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Neuro-symbolic micro-LLM | Locally trained causal micro-transformer compiled to OpenVINO opset13 under verify_answer. | `neurosymbolic_genesis_discipline_and_llm:NeuroSymbolicMicroLLM` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Arena tensor navigator | Caches 37-hex transition matrix and diffuses movement potential in OpenVINO opset13. | `arena_cache_optical_sockets_and_consumption:BattleArenaCacheAndTensorNavigator` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Render consumption graph | Tracks per-turn 128B and render costs and derives Before/After optimization savings. | `arena_cache_optical_sockets_and_consumption:TurnConsumptionAndRenderAuditGraph` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Optical binary codec | Packs Fresnel/LOD optical descriptors into a SHA-256-verified .kopt little-endian binary. | `arena_cache_optical_sockets_and_consumption:synthesize_and_write_optical_binary` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Code-sample object library | Compiles real AST metrics of bound framework functions into 3D meshes and SVG wireframes. | `object_library_texture_nodes_and_synthetic_network:build_renderable_object_library` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Preallocated texture node graph | Five-node RGBA arena texture generator executing in a zero-reallocation bytearray pool. | `object_library_texture_nodes_and_synthetic_network:execute_arena_texture_node_graph` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Synthetic knowledge network | Exports provenance-tagged training triples from AST and facts for model learning. | `object_library_texture_nodes_and_synthetic_network:build_synthetic_knowledge_training_network` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Adaptive conversation trainer | High-intensity continual trainer with verifier beam pruning and episodic residual memory. | `adaptive_2b_spatial_linguistic_trainer:AdaptiveConversationTrainer` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Convolutional study conclusion | Convolves cross-stratum precision signals into the formal study summary and SVG projection. | `adaptive_2b_spatial_linguistic_trainer:compile_convolutional_study_conclusion` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Operational field-line learner | Integrates vector streamlines over operational ranges and trains a sequence predictor. | `playable_fieldlines_and_game_theory:generate_operational_field_lines_and_learn` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Syntactic function chain | Chains verified arena, tensor, fieldline and socket steps with Antigravity telemetry. | `playable_fieldlines_and_game_theory:execute_syntactic_function_chain` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Standardized digital game theory | Formal (S, A, T, I, O, E, H, W) specification and termination/budget theorems. | `playable_fieldlines_and_game_theory:evaluate_standardized_digital_game_theory` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| WSL hypervisor configurator | Host-calibrated .wslconfig and OSS layer templates for Windows 11 / Ubuntu WSLg. | `wsl_hypervisor_mortar_and_geotile_adapter:configure_wsl_ubuntu_hypervisor_bridge` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Mortar CSS trajectory solver | Parabolic ballistic arc from cached hex state with CSS keyframes and CEP reticle. | `wsl_hypervisor_mortar_and_geotile_adapter:compute_mortar_css_trajectories_and_reticles` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Geotile 3D arena replica | Metric Web-Mercator slippy-tile to 37-hex 3D terrain replica converter. | `wsl_hypervisor_mortar_and_geotile_adapter:convert_geotile_location_to_3d_arena_replica` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Vulkan UMA shader isomorphism | Bijective Hex37<->UMA slot map, physical-core LUTs and GLSL 450/SPIR-V templates. | `wsl_hypervisor_mortar_and_geotile_adapter:compile_vulkan_spirv_shader_sets_and_isomorphisms` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Satellite 3D mesh duplicator | Trained OpenVINO MLP extruding satellite/DEM grids into trimesh-audited .obj meshes. | `cluster_pid_gemma_finetune_and_mesh_ml:train_and_duplicate_satellite_3d_mesh` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Cluster entity PID registry | Collision-free 32-bit PIDs defining cluster ring/slot, procedures and combination caps. | `cluster_pid_gemma_finetune_and_mesh_ml:build_cluster_entity_pid_registry` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Intranet 128B packet bus | Binary KPKT 128 B SHA-256 inter-module packets modulating presets and WSL2 quotas. | `cluster_pid_gemma_finetune_and_mesh_ml:run_neurosymbolic_datalink_and_hypervisor_exchange` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |
| Gemma fine-tuning sets | Gemma chat-formatted abstraction variants and four leak-free train/eval sets. | `cluster_pid_gemma_finetune_and_mesh_ml:build_gemma_abstraction_variants_and_finetune_sets` | `tests/test_x86_hex_vr_and_study_conclusion.py` | RESOLVED |

Public symbols without a term: 102

- `adaptive_2b_spatial_linguistic_trainer`: calibrate_new_shapes_and_model_catalog, build_spatial_and_linguistic_2b_kb, EpisodicResidualMemoryBank, compute_four_core_project_summary_and_roadmap
- `adaptive_cross_genre_sync`: SyncMode, SyncNode, SyncConnection, describe_adaptive_sync_architecture
- `adaptive_sync_card_grid`: build_card_dataset, export_standalone_html
- `arena_cache_optical_sockets_and_consumption`: evaluate_compact_model_interaction_scaffolding, run_multiplayer_entity_script_round, read_optical_binary, build_option_geometry_holograms, render_wireframe_sockets_and_holograms_html
- `catalog_gearbox_and_comm_kernel`: FrameworkNationsValidator, CatalogProliferationAndCarouselReconciler, FamiliarHomeClassAndSevenProductSets, ProcessDrivenGearboxAndCommKernel, ConstitutionalModeratorAndChoiceSolver, CrossDisplayMultiGenreParameterMatrix
- `clash_arena_page`: render_clash_arena_html
- `clash_battle_royale`: hex_distance, board_cells, ClashGame, ClashRegistry
- `cluster_pid_gemma_finetune_and_mesh_ml`: pack_intranet_128b_packet, unpack_intranet_128b_packet, build_godot_scene_and_goniometric_behavioral_kernel
- `combat_turn_planner`: get_cached_hw_profile, probe_isa_flags
- `configurator_and_dxdiag`: prewarm_heterogeneous_blob_pair, extract_pattern_candidates_from_diagnostics, export_dxdiag_txt, render_dxdiag_profile_html
- `consistency_bot`: BotCheck
- `fact_synthesis_engine`: synthesize_facts, induce_rules, module_distance_matrix, narrate
- `hex_combat_bandwidth_and_world_ai`: hex_cell_count, hex_undirected_edge_count, verify_hex_combinatorial_identities, measure_clash_engine_action_throughput, instantiate_combat_preset, evaluate_all_faction_duel_combinations, generate_world_hex_influences
- `intel_accelerator_and_power_thread_governor`: IntelDeviceProperties, IntelStream, get_shared_ov_core
- `matrix_control_processes`: freivalds_verify, run_operator_mashup
- `neurosymbolic_bridge`: build_grounded_prompt, ask_llm, build_compendium
- `neurosymbolic_genesis_discipline_and_llm`: build_henc_discipline_and_train_demo
- `nomenclature_registry`: validate_nomenclature, export_glossary
- `object_library_texture_nodes_and_synthetic_network`: render_object_from_code_sample, PreallocatedPhaseRenderPool, compile_carbon_environment_script
- `oneapi_llm_heterogeneous_shell`: main, get_global_oneapi_shell
- `pattern_profile_environment`: ProfileSpec
- `platform_engine`: TrigonometricHarmonicStateStruct, ZeroCopy128BContractStruct, ReloadedTrigonometricPatternStructures, SchemaConsistencySolver, CodeDependencyGraphAnalyzer, MasterPromptRouterConsolidator, StepExecutionGateProxy, HierarchyOfRelatedTermsEngine, TrigonometricHybridPlatformFramework, render_trigonometric_platform_html
- `playable_fieldlines_and_game_theory`: render_playable_syntactic_studio_html, solve_algebraic_tensor_word_problems, build_llm_engine_tool_integrations, fetch_staged_processes_and_rig_skeleton_animations, render_lightweight_bot_perception_hud, run_engine_node_benchmark_and_recommend_preset
- `simulation_grid_3d`: ensure_babylon_vendor, hex_spiral_axial, build_trace_dataset, export_standalone_html
- `spatial_coordinators_and_vr_hologram_compositor`: compile_cycle_driven_ui_schema, render_vr_hologram_profiler_html
- `tensor_prompt_branch_and_kb`: encode_prompt_to_feature_row, build_openapi_31_strategy_spec
- `virtual_convolution_contra`: conv2d_transpose_reference, openvino_transpose_conv
- `virtual_convolution_pipeline`: conv2d_reference, reference_pipeline, build_openvino_model
- `wsl_hypervisor_mortar_and_geotile_adapter`: define_os_call_visual_semantics, evaluate_decision_tensor_ui_and_2b_bridge, adapt_integrations_to_four_geospatial_tiles, wsl_hypervisor_render_dispatch
- `x86_platform_findings`: measure_x01_cache_bandwidth_hierarchy, measure_x02_physical_vs_smt_avx512_scaling, measure_x03_subnormal_assist_vs_mxcsr_ftz, measure_x04_cacheline_stride_and_alignment, measure_x05_128b_binary_frame_vs_json
