import sys
import os

from krystal_kernel.slovak_llm_adapter import SlovakLLMAdapter
from krystal_kernel.cascade.cascade_planner import CascadePlanner

def main():
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    adapter = SlovakLLMAdapter()
    devs = adapter.get_available_devices()
    ram_mb = adapter.get_physical_memory_available_mb()
    rss_mb = adapter.get_process_rss_mb()
    print(f"DEVICES: {devs}")
    print(f"AVAILABLE_RAM_MB: {ram_mb}")
    print(f"PROCESS_RSS_MB: {rss_mb}")
    
    snap = CascadePlanner.capture_environment_snapshot()
    print(f"SNAPSHOT: {snap}")
    
    cp = CascadePlanner()
    strat_short = cp.select_phase_aware_strategy(prompt_tokens=16, est_gen_tokens=16, available_devices=devs)
    print(f"STRAT_SHORT: {strat_short['selected_strategy']} | {strat_short['rationale']}")
    
    strat_long = cp.select_phase_aware_strategy(prompt_tokens=128, est_gen_tokens=32, available_devices=devs)
    print(f"STRAT_LONG: {strat_long['selected_strategy']} | {strat_long['rationale']}")

if __name__ == "__main__":
    main()
