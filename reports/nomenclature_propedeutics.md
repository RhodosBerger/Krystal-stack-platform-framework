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

Public symbols without a term: 44

- `adaptive_cross_genre_sync`: SyncMode, SyncNode, SyncConnection, describe_adaptive_sync_architecture
- `adaptive_sync_card_grid`: build_card_dataset, export_standalone_html
- `catalog_gearbox_and_comm_kernel`: FrameworkNationsValidator, CatalogProliferationAndCarouselReconciler, FamiliarHomeClassAndSevenProductSets, ProcessDrivenGearboxAndCommKernel, ConstitutionalModeratorAndChoiceSolver, CrossDisplayMultiGenreParameterMatrix
- `consistency_bot`: BotCheck
- `fact_synthesis_engine`: synthesize_facts, induce_rules, module_distance_matrix, narrate
- `matrix_control_processes`: freivalds_verify, run_operator_mashup
- `neurosymbolic_bridge`: build_grounded_prompt, ask_llm, build_compendium
- `nomenclature_registry`: validate_nomenclature, export_glossary
- `pattern_profile_environment`: ProfileSpec
- `platform_engine`: TrigonometricHarmonicStateStruct, ZeroCopy128BContractStruct, ReloadedTrigonometricPatternStructures, SchemaConsistencySolver, CodeDependencyGraphAnalyzer, MasterPromptRouterConsolidator, StepExecutionGateProxy, HierarchyOfRelatedTermsEngine, TrigonometricHybridPlatformFramework, render_trigonometric_platform_html
- `simulation_grid_3d`: ensure_babylon_vendor, hex_spiral_axial, build_trace_dataset, export_standalone_html
- `virtual_convolution_contra`: conv2d_transpose_reference, openvino_transpose_conv
- `virtual_convolution_pipeline`: conv2d_reference, reference_pipeline, build_openvino_model
