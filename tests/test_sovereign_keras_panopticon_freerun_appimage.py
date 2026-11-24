r"""Dedikovaný verifikačný a falzifikačný testovací súbor (`python -O -m pytest`) pre celú novú
architektúru Krystal Stack:
  1. Protokol Render<->LLM, ML fonémový anglický našepkávač, kompilátor questov a propedeutický cache,
  2. Sonic 3D Pixel-Art / Voxel ASCII engine + Terminálový prehliadač (`KrystalTerminalBrowser`),
  3. Panoptikum hry, `30 %` obrazovky UMA grafy, C11 `state_before/state_after`, Anno->Cyber60s protokol,
     Linux GDM/COSMIC/Niri režimy okien, Godot 4 WebGL2 a 2D Castlevania-štýl labyrint v našom spektre,
  4. Veo brainstorming skripty, Mortar zbraňový systém, algebraické zameriavače so šípkami a 4-stupňový pre-cache,
  5. Štatistické modely + neurónová sieť v Kerase (`Keras 3.15.1`), nízko-RAM systémové presety,
     pozadové streamovanie mapy počas kolových NPC kartových bojov s ovládacími slučkami a Ubuntu `.AppImage`.

Zákaz holého `assert` — všetky invarianty vynucujeme cez `ContractViolation` kompatibilný s `python -O`!
"""

from __future__ import annotations

from pathlib import Path

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation
from projects.trigonometric_hybrid_platform_framework.keras_lowram_npc_streaming_and_ubuntu_appimage import (
    MANDATORY_CONTROL_CYCLE_STEPS,
    StatisticalAndKerasArchitectureTrainer,
    ZeroReloadCombatMapStreamerAndControlLoop,
    build_ubuntu_appimage_installer_package,
    select_low_ram_system_preset,
)
from projects.trigonometric_hybrid_platform_framework.last_tribe_cyber60s_vps_cloud_and_blender_bench import (
    execute_last_tribe_arc_vps_and_blender_suite,
)
from projects.trigonometric_hybrid_platform_framework.panopticon_anno_cyber60s_linux_gdm_godot_suite import (
    CacheAccelerationAnd30PctUmaBenchmark,
    execute_panopticon_anno_cyber60s_linux_gdm_godot_suite,
)
from projects.trigonometric_hybrid_platform_framework.phoneme_dialogue_quest_propedeutic_and_nanobanana import (
    StandardizedRenderLlmProtocol,
    execute_phoneme_quest_propedeutic_sonic_and_nanobanana_suite,
)
from projects.trigonometric_hybrid_platform_framework.visual_compositor_timeline_damage_and_microservices import (
    execute_compositor_timeline_damage_and_microservices_suite,
)
from scripts.generate_veo_mortar_brainstorming_previews import (
    MortarWeaponAndAlgebraicReticleSimulator,
    generate_veo_brainstorming_scripts_and_previews,
)
from scripts.krystal_terminal_browser_and_projections import KrystalTerminalBrowser, TERMINAL_PROJECTION_MODES


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractViolation(msg, violation_code="ERR_TEST_ASSERTION_FAILED")


def test_render_llm_phoneme_quest_sonic_and_terminal_browser() -> None:
    res = execute_phoneme_quest_propedeutic_sonic_and_nanobanana_suite()
    _require(res["vital_max_hp"] == VITAL_MAX_HP == 6, "VITAL_MAX_HP musí byť 6")
    _require(res["quest_compilation"]["quest_node"]["compilation_status"] == "COMPILED_IN_SHADOW_WINDOW", "Quest sa musí skompilovať počas pohybu pri UMA < 0.78")
    _require(res["propedeutic_map_preset"]["propedeutic_coverage_ratio"] >= 0.60, "Propedeutický preset musí pokrývať >= 60 % čankov")
    _require(res["nanobanana_projection_previews"]["previews_count"] == 4, "Musia vzniknúť 4 kontextové PNG náhľady")

    # Falzifikátor: porušenie VITAL_MAX_HP v protokole musí byť odmietnuté
    rejected = False
    try:
        StandardizedRenderLlmProtocol.build_protocol_packet(vital_hp=9)
    except ContractViolation:
        rejected = True
    _require(rejected, "Porušenie VITAL_MAX_HP != 6 musí vyhodiť ContractViolation")

    # Overenie všetkých 5 projekcií Terminálového Prehliadača
    tb = KrystalTerminalBrowser()
    for mode in TERMINAL_PROJECTION_MODES:
        view = tb.render_terminal_viewport(projection_mode=mode)
        _require("VITAL_MAX_HP == 6/6" in view["plain_terminal_screen"], f"Chýba kotva HP=6 v {mode}")


def test_panopticon_30pct_uma_c_scaler_linux_godot_and_castlevania_labyrinth() -> None:
    suite = execute_panopticon_anno_cyber60s_linux_gdm_godot_suite()
    _require(suite["vital_max_hp"] == 6, "VITAL_MAX_HP musí byť 6")
    _require(
        suite["cache_acceleration_30pct_uma_and_fps"]["uma_graph_screen_allocation_pct"] == 30.0,
        "Graf UMA zbernice musí mať vyhradených presne 30 % obrazovky",
    )
    _require(
        suite["cache_acceleration_30pct_uma_and_fps"]["uma_bus_telemetry_series"]["max_uma"] < UMA_SATURATION_THRESHOLD,
        "UMA zbernica nesmie prekročiť strop 0.78",
    )
    _require(suite["anno_modding_and_c_state_scaler"]["c_terminology_state_scaler"]["committed"] == 1, "C11 state_after musí byť potvrdený")
    _require(len(suite["linux_gdm_wsl_godot_and_propedeutics"]["narrative_propedeutics_monitoring_pipeline"]) == 5, "5 metrík propedeutiky príbehu")
    _require(suite["castlevania_krystal_spectrum_labyrinth"]["total_viewport_share_pct"] == 100.0, "4 násobené viewporty musia dávať 100 %")
    _require(suite["in_engine_predicted_previews"]["previews_count"] == 4, "4 komplexné PNG náhľady v grafike enginu")


def test_veo_mortar_algebraic_reticle_and_progressive_precache() -> None:
    veo = generate_veo_brainstorming_scripts_and_previews()
    _require(veo["vital_max_hp"] == 6, "VITAL_MAX_HP == 6")
    _require(veo["mortar_simulation"]["algebraic_solution"]["mortar_high_angle_deg"] > 45.0, "Mortar musí mať hornú balistickú eleváciu > 45°")
    _require(len(veo["mortar_simulation"]["vector_process_arrows"]) >= 3, "Musia byť prítomné vektorové šípky procesu")
    _require(veo["progressive_precache"]["fits_within_148fps_6_7ms_budget"] is True, "4 pre-cache vrstvy sa musia zmestiť do rozpočtu 148.8 FPS")
    _require(len(veo["generated_previews"]) == 2, "2 kľúčové PNG náhľady pre Veo")


def test_statistical_and_keras_models_lowram_npc_streaming_and_ubuntu_appimage() -> None:
    # 1. Tréning a validácia štatistických modelov + neurónovej siete v Kerase (Keras 3.15.1)
    ml = StatisticalAndKerasArchitectureTrainer().train_and_export_architecture_models(n_samples=240)
    _require(ml["dataset_splits"]["leakage_free"] is True, "Žiadny únik dát medzi train/val/test")
    _require(ml["keras_architecture_neural_net"]["test_metrics"]["rmse"] < 0.16, "Keras sieť musí dosiahnuť nízku chybu RMSE < 0.16")
    _require(Path(ml["int8_quantized_weights_path"]).exists(), "INT8 kvantizované váhy musia existovať na disku")

    # 2. Nízko-RAM systémové presety
    preset = select_low_ram_system_preset(available_free_ram_mb=384, domain_hint="terminal")
    _require(preset["selected_preset"]["target_ram_ceiling_mb"] <= 96, "Terminálový preset musí držať RAM strop <= 96 MB")

    # 3. Pozadové streamovanie mapy počas kolového NPC boja + falzifikátor preskočenia ovládacej slučky
    combat = ZeroReloadCombatMapStreamerAndControlLoop()
    wrong_order_rejected = False
    try:
        combat.execute_player_control_step("CTRL_LOOP_04_COMMIT_ROUND_AND_SWAP_MAP_SLICE")
    except ContractViolation:
        wrong_order_rejected = True
    _require(wrong_order_rejected, "Preskočenie krokov ovládacej slučky musí byť zamietnuté (Delta S = 0)")

    last_state = {}
    for step_id in MANDATORY_CONTROL_CYCLE_STEPS:
        last_state = combat.execute_player_control_step(
            step_id=step_id,
            npc_intent="NPC_ARMORED_GOLEM",
            chosen_combo_id="COMBO_TYPE_C_TOXIC_CATALYST_STACK",
            uma_utilization=0.39,
        )
    _require(last_state["combo_differentiated_correctly"] is True, "Správne odlíšené kombo musí byť potvrdené")
    _require(last_state["can_continue_to_next_combat_phase"] is True, "Po 4 krokoch môže boj pokračovať")
    _require(
        last_state["background_map_streaming"]["zero_full_map_reload_ready"] is True,
        "Sektor mapy musí byť na 100 % nastreamovaný na pozadí bez reloadu celej mapy",
    )

    # 4. Kompilácia a kontrola The Last Tribe, Kompozítora a Ubuntu .AppImage balíčka
    lt = execute_last_tribe_arc_vps_and_blender_suite()
    comp = execute_compositor_timeline_damage_and_microservices_suite()
    appimg = build_ubuntu_appimage_installer_package()
    _require(lt["vital_max_hp"] == 6 and comp["vital_max_hp"] == 6, "Invariant HP == 6")
    _require(Path(appimg["appimage_path"]).exists() and appimg["appimage_size_bytes"] > 1024, "Ubuntu .AppImage musí byť zostavený")
