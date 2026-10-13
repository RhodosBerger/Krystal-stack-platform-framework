#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: LINUX HARDWARE ALLOCATION MONITOR CLI UTILITY
==============================================================================
Script: scripts/krystal_hw_monitor.py
Description:
    Stand-alone Linux terminal executable for inspecting and profiling
    hardware resource allocations across all 8 specialized domains.

Usage:
    ./scripts/krystal_hw_monitor.py             # Single snapshot
    ./scripts/krystal_hw_monitor.py -w          # Real-time interactive watch mode
    ./scripts/krystal_hw_monitor.py --json      # Raw JSON output for automation
    ./scripts/krystal_hw_monitor.py -c memory   # Inspect only Memory/UMA
    ./scripts/krystal_hw_monitor.py -c npu      # Inspect only NPU & Model Cascades

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.linux_hw_monitor_utility import main

if __name__ == "__main__":
    main()
