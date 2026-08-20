# Architecture Mapping: Dev-Conditional to GAMESA Production

## Layer Correspondence

| Dev-Conditional Layer | GAMESA Production Component | Location |
|-----------------------|----------------------------|----------|
| `HardwareLevel` | Thread Boost Layer | `autonomous_game_engine/thread_boost_layer.{h,c}` |
| `SignalLevel` | RPG Craft System + IPC | `autonomous_game_engine/rpg_craft_system.h` |
| `LearningLevel` | Guardian Decision Policy | `gamesa/guardian/decision_policy.py` |
| `PredictionLevel` | Process Predictor | `gamesa/guardian/process_predictor.py` |
| `EmergenceLevel` | Hybrid Event Pipeline | `gamesa/guardian/hybrid_event_pipeline.py` |
| `GenerationLevel` | OpenVINO/TPU Presets | `gamesa/guardian/` |

## Integration Points

### Python Guardian <-> C Core
```
unified_system.py  ->  guardian_hooks.py  ->  thread_boost_layer.c
                                          ->  rpg_craft_system.h
```

### Python Guardian <-> Rust Bot
```
unified_system.py  ->  kernel_bridge.py  ->  rust-bot/src/orchestration.rs
                                         ->  rust-bot/src/types.rs
```

## Key Contracts

- **Telemetry**: `TelemetrySnapshot` matches Rust `Telemetry` struct
- **Directives**: `DirectiveDecision` maps to C `boost_zone_t` actions
- **Presets**: `UnifiedPreset` aligns with RPG craft `preset_t`

## Running Both Systems

```bash
# Dev-Conditional (prototyping)
python -m src.python.cli run --cycles 100

# GAMESA Production
python -m gamesa.guardian.hybrid_event_pipeline
```

## Migration Path

1. Prototype in Dev-Conditional
2. Port to guardian modules
3. Wire to C/Rust via existing bridges
4. Test with full stack
