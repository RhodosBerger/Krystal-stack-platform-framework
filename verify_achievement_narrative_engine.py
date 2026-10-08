#!/usr/bin/env python3
"""
Verification Script for Achievement Narrative Engine & Intel/llama.cpp Telemetry Governor.
Validates:
  1. Canonical achievement definitions and economic reward tiers.
  2. Story chapter authoring (Chronicle Codex) with context injection.
  3. Intel OpenVINO vs llama.cpp hardware telemetry harvesting (Watts, Joules/tok, TTFT).
  4. Closed-loop telemetry backpressure and throttling.
"""

import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from krystal_web_hub.economic_engine.achievement_narrative_engine import (
    AchievementNarrativeEngine,
    AchievementCategory,
    AchievementTier,
    LLMRuntimeBackend,
    TelemetryGovernor,
    CANONICAL_ACHIEVEMENTS
)

def test_achievement_registry():
    print("[TEST 1] Validating Canonical Achievement Registry...")
    engine = AchievementNarrativeEngine()
    achievements = engine.list_achievements()
    assert len(achievements) >= 6, f"Expected at least 6 achievements, got {len(achievements)}"
    
    categories = {a["category"] for a in achievements}
    assert AchievementCategory.TACTICAL_COMBAT.value in categories
    assert AchievementCategory.URBAN_ARCHITECTURE.value in categories
    assert AchievementCategory.OPTIC_TELEMETRY.value in categories
    assert AchievementCategory.ECONOMIC_COMMERCE.value in categories
    
    print(f"  -> Registry OK: {len(achievements)} achievements across {len(categories)} categories.")

def test_unlock_and_narrative_generation():
    print("[TEST 2] Validating Achievement Unlock & Narrative Chronicle Authoring...")
    engine = AchievementNarrativeEngine(preferred_backend=LLMRuntimeBackend.INTEL_OPENVINO_GENAI)
    
    # Unlock Urban Architecture Achievement
    res = engine.unlock_achievement("praha_old_town_surveyor", {
        "city": "Praha - Staré Město",
        "tribe": "Kryštálový Kmeň"
    })
    
    assert res["status"] == "UNLOCKED"
    assert res["achievement"]["unlocked"] == True
    assert res["economic_impact"]["credits_granted"] == 300
    assert engine.total_credits_minted == 300
    
    story = res["story_chapter"]
    assert "Týnskeho chrámu" in story["narrative_text"]
    assert story["city_context"] == "Praha - Staré Město"
    assert story["tribe_affected"] == "Kryštálový Kmeň"
    
    print(f"  -> Unlock OK: Chapter '{story['chapter_title']}', Credits Minted: {engine.total_credits_minted}")

def test_intel_vs_llamacpp_telemetry():
    print("[TEST 3] Validating Intel OpenVINO vs llama.cpp Telemetry Harvesting...")
    
    # 1. Intel OpenVINO GenAI Telemetry (NPU/iGPU)
    gov_intel = TelemetryGovernor(backend=LLMRuntimeBackend.INTEL_OPENVINO_GENAI)
    t_intel = gov_intel.harvest_telemetry(prompt_len=120, generated_tokens=64, elapsed_s=0.25)
    
    assert t_intel.power_watts < 10.0, f"Intel NPU power should be low (~4-7W), got {t_intel.power_watts}W"
    assert t_intel.target_hardware == "NPU+IGPU"
    assert t_intel.time_to_first_token_ms < 500.0
    
    # 2. llama.cpp Telemetry (CPU / Vulkan)
    gov_llama = TelemetryGovernor(backend=LLMRuntimeBackend.LLAMA_CPP_GGUF)
    t_llama = gov_llama.harvest_telemetry(prompt_len=120, generated_tokens=64, elapsed_s=0.35)
    
    assert t_llama.power_watts > 12.0, f"llama.cpp CPU/GPU power should be higher (~15-25W), got {t_llama.power_watts}W"
    assert t_llama.target_hardware == "CPU_VULKAN"
    
    print(f"  -> Telemetry OK:")
    print(f"     • Intel OpenVINO NPU: {t_intel.power_watts}W, {t_intel.energy_joules_per_tok} J/tok, Mem: {t_intel.memory_footprint_mb}MB")
    print(f"     • llama.cpp Vulkan:   {t_llama.power_watts}W, {t_llama.energy_joules_per_tok} J/tok, Mem: {t_llama.memory_footprint_mb}MB")

def test_closed_loop_backpressure():
    print("[TEST 4] Validating Closed-Loop Telemetry Backpressure Throttling...")
    gov = TelemetryGovernor(backend=LLMRuntimeBackend.INTEL_OPENVINO_GENAI)
    gov.max_allowed_power_watts = 5.0 # Artificially restrict power ceiling
    
    t_snap = gov.harvest_telemetry(prompt_len=200, generated_tokens=64, elapsed_s=0.5)
    assert t_snap.is_throttled == True, "Governor should trip throttling when power exceeds ceiling"
    
    print(f"  -> Backpressure OK: Successfully triggered throttling (is_throttled={t_snap.is_throttled}).")

if __name__ == "__main__":
    print("=" * 65)
    print("KRYSTAL-STACK: VERIFYING ACHIEVEMENT NARRATIVE & TELEMETRY ENGINE")
    print("=" * 65)
    test_achievement_registry()
    test_unlock_and_narrative_generation()
    test_intel_vs_llamacpp_telemetry()
    test_closed_loop_backpressure()
    print("=" * 65)
    print("ALL TESTS PASSED WITH 100% INTEGRITY!")
    print("=" * 65)
