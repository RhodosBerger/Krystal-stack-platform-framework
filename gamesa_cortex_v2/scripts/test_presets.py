import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gamesa_cortex_v2.src.core.npu_coordinator import NPUCoordinator
from gamesa_cortex_v2.src.core.presets import PresetManager

def test_presets():
    print("Initializing NPU Coordinator...", flush=True)
    coordinator = NPUCoordinator()
    
    print("\n[Test] Initial State Check:", flush=True)
    print(f"Workers: {coordinator.executor._max_workers}", flush=True)
    print(f"Budget Cap: {coordinator.economics.budget_cap}", flush=True)
    print(f"OpenVINO Hint: {coordinator.openvino.current_hint}", flush=True)
    
    presets_to_test = ["IDLE_ECO", "HIGH_PERFORMANCE", "OPENVINO_INFERENCE"]
    
    for preset_name in presets_to_test:
        print(f"\n[Test] Applying Preset: {preset_name}", flush=True)
        coordinator.apply_preset(preset_name)
        
        target = PresetManager.get_preset(preset_name)
        
        # Verify
        current_workers = coordinator.executor._max_workers
        current_cap = coordinator.economics.budget_cap
        current_hint = coordinator.openvino.current_hint
        
        print(f"  > Workers: {current_workers} (Expected: {target.max_workers})", flush=True)
        print(f"  > Budget:  {current_cap} (Expected: {target.budget_cap})", flush=True)
        print(f"  > Hint:    {current_hint} (Expected: {target.openvino_hint})", flush=True)
        
        if current_workers == target.max_workers and current_hint == target.openvino_hint:
             print("  [PASS] Configuration updated successfully.", flush=True)
        else:
             print("  [FAIL] Configuration mismatch.", flush=True)
             
    print("\nShutting down...", flush=True)
    coordinator.shutdown()

if __name__ == "__main__":
    test_presets()
