import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

print("Testing imports...", flush=True)

try:
    from gamesa_cortex_v2.src.core.config import GamesaConfig
    print(f"Config loaded. Max Workers: {GamesaConfig.MAX_WORKERS}", flush=True)
except Exception as e:
    print(f"Config import failed: {e}", flush=True)

try:
    from gamesa_cortex_v2.src.core.utils import PreciseTimer
    t = PreciseTimer()
    print(f"Timer loaded. Elapsed: {t.elapsed_ms()}", flush=True)
except Exception as e:
    print(f"Timer import failed: {e}", flush=True)

try:
    from gamesa_cortex_v2.src.core.power_governor import PowerGovernor
    p = PowerGovernor()
    print(f"PowerGovernor loaded. Mode: {p.current_mode}", flush=True)
except Exception as e:
    print(f"PowerGovernor import failed: {e}", flush=True)

try:
    from gamesa_cortex_v2.src.core.economic_governor import EconomicGovernor
    e = EconomicGovernor()
    print(f"EconomicGovernor loaded. Budget: {e.budget_credits}", flush=True)
except Exception as e:
    print(f"EconomicGovernor import failed: {e}", flush=True)

try:
    from gamesa_cortex_v2.src.core.npu_coordinator import NPUCoordinator
    n = NPUCoordinator()
    print(f"NPUCoordinator loaded.", flush=True)
except Exception as e:
    print(f"NPUCoordinator import failed: {e}", flush=True)
