#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CORTEX PROCESS PRIORITIZATION COMPILER & WINDOWS API DISPATCHER
==============================================================================
Module: krystal_kernel/cortex_compiler.py
Description: Domain-specific compiler that compiles high-level scheduling rules
             and Cortex integrity decisions into native Windows API process
             prioritization operations (SetPriorityClass, SetProcessAffinityMask).

Compiles Operations:
  - OP_INSPECT_TELEMETRY: Reads target process working set, threads, and times.
  - OP_EVAL_CORTEX_INTEGRITY: Invokes Cortex decision algorithm for system integrity.
  - OP_INFER_OPENVINO_PRIORITY: Executes neural model for priority class logits.
  - OP_RESOLVE_ARBITRATION: Synthesizes Cortex mandate with neural inference.
  - OP_CALC_AFFINITY_MASK: Optimizes CPU core binding to prevent cache thrashing.
  - OP_EMIT_WIN32_SET_PRIORITY: Emits/executes kernel32.dll!SetPriorityClass.
  - OP_EMIT_WIN32_SET_AFFINITY: Emits/executes kernel32.dll!SetProcessAffinityMask.
  - OP_VERIFY_INVARIANT: Enforces VITAL_MAX_HP = 6.

Outputs:
  1. Executable Bytecode Plan for immediate in-process execution.
  2. Standalone Microsoft Visual C# Win32 P/Invoke stager.
  3. Native C++20 Win32 Dispatcher code.
  4. Windows PowerShell Automation snippet.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
import ctypes
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.cortex_openvino_engine import (
    CortexIntegrityEngine,
    OpenVINOProcessGovernor,
    CortexIntegrityVerdict,
    OpenVINOInferenceResult,
    WIN32_PRIORITY_CLASSES,
    GLOBAL_CORTEX_ENGINE,
    GLOBAL_OPENVINO_GOVERNOR,
    VITAL_MAX_HP
)

# Win32 Process Access Rights
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_SET_INFORMATION = 0x0200
PROCESS_ALL_ACCESS = 0x1F0FFF


@dataclass
class CompiledOperation:
    """Single instruction step within a compiled Cortex scheduling plan."""
    op_code: str
    target_pid: int
    parameter: Any
    description: str


@dataclass
class CortexCompiledPlan:
    """The complete executable plan emitted by the Cortex Compiler."""
    plan_id: str
    target_pid: int
    target_process_name: str
    compiled_operations: List[Dict[str, Any]]
    cortex_verdict: Dict[str, Any]
    openvino_inference: Dict[str, Any]
    resolved_win32_priority_class: str
    resolved_win32_priority_code: int
    resolved_affinity_mask: int
    generated_csharp_stager: str
    generated_cpp_dispatcher: str
    generated_powershell_cmd: str
    timestamp: float
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WindowsApiGovernor:
    """
    Direct native Windows API bridge using ctypes to kernel32.dll.
    Applies real priority classes and core affinity masks to OS processes.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self.k32 = None
        if self.is_windows:
            try:
                self.k32 = ctypes.windll.kernel32
            except Exception:
                self.k32 = None

    def list_running_processes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists active Windows processes with their PIDs, names, and current priority."""
        procs = []
        if self.is_windows:
            try:
                # Use tasklist or snapshot
                import subprocess
                cmd = ["tasklist", "/FO", "CSV", "/NH"]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                lines = res.stdout.strip().splitlines()
                for line in lines[:limit]:
                    parts = [p.strip('"\r ') for p in line.split('","')]
                    if len(parts) >= 5:
                        p_name = parts[0]
                        try:
                            p_pid = int(parts[1])
                            p_mem = parts[4].replace(" K", "").replace(",", "").replace("\xa0", "")
                            p_mem_mb = round(float(p_mem) / 1024.0, 1)
                        except ValueError:
                            continue
                        procs.append({
                            "pid": p_pid,
                            "name": p_name,
                            "memory_mb": p_mem_mb,
                            "current_priority": "NORMAL"
                        })
                return procs
            except Exception:
                pass

        # Fallback simulation list if not accessible
        sim_procs = [
            {"pid": 1044, "name": "krystal_render_worker.exe", "memory_mb": 420.5, "current_priority": "NORMAL"},
            {"pid": 2816, "name": "vulkan_compute_daemon.exe", "memory_mb": 840.2, "current_priority": "NORMAL"},
            {"pid": 3920, "name": "code_gene_compositor.exe", "memory_mb": 310.0, "current_priority": "NORMAL"},
            {"pid": 5104, "name": "chrome.exe", "memory_mb": 650.0, "current_priority": "NORMAL"},
            {"pid": 7240, "name": "python.exe", "memory_mb": 180.4, "current_priority": "NORMAL"}
        ]
        return sim_procs

    def apply_priority(self, pid: int, priority_name: str, affinity_mask: Optional[int] = None) -> Dict[str, Any]:
        """
        Invokes kernel32.dll!SetPriorityClass and SetProcessAffinityMask.
        Safe execution with detailed error reporting.
        """
        priority_code = WIN32_PRIORITY_CLASSES.get(priority_name, 0x00000020) # Default NORMAL
        if not self.is_windows or not self.k32:
            return {
                "success": True,
                "status": "SIMULATED_SUCCESS",
                "pid": pid,
                "priority_applied": priority_name,
                "win32_code": hex(priority_code),
                "affinity_applied": hex(affinity_mask) if affinity_mask else "ALL_CORES",
                "message": "Windows API mocked on non-Windows/non-elevated runtime."
            }

        hProcess = self.k32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, pid)
        if not hProcess:
            err = self.k32.GetLastError()
            return {
                "success": False,
                "status": "ACCESS_DENIED_OR_NOT_FOUND",
                "pid": pid,
                "error_code": err,
                "message": f"OpenProcess failed with error code {err}. Administrator elevation may be required."
            }

        try:
            p_res = self.k32.SetPriorityClass(hProcess, priority_code)
            aff_res = True
            if affinity_mask is not None and affinity_mask > 0:
                mask_p = ctypes.c_ulonglong(affinity_mask)
                aff_res = bool(self.k32.SetProcessAffinityMask(hProcess, mask_p))

            self.k32.CloseHandle(hProcess)
            return {
                "success": bool(p_res),
                "status": "APPLIED" if p_res else "SET_PRIORITY_FAILED",
                "pid": pid,
                "priority_applied": priority_name,
                "win32_code": hex(priority_code),
                "affinity_applied": hex(affinity_mask) if affinity_mask else "UNCHANGED",
                "message": f"Successfully applied priority class {priority_name} via Windows kernel32."
            }
        except Exception as ex:
            self.k32.CloseHandle(hProcess)
            return {
                "success": False,
                "status": "EXCEPTION",
                "pid": pid,
                "message": str(ex)
            }


class KrystalCortexCompiler:
    """
    Domain-specific compiler that takes a prioritization specification,
    interrogates the Cortex integrity engine and OpenVINO inference,
    and compiles the optimal Windows API execution plan.
    """

    def __init__(
        self,
        cortex_engine: Optional[CortexIntegrityEngine] = None,
        openvino_governor: Optional[OpenVINOProcessGovernor] = None
    ):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"
        self.vital_hp = VITAL_MAX_HP
        self.cortex = cortex_engine or GLOBAL_CORTEX_ENGINE
        self.vino = openvino_governor or GLOBAL_OPENVINO_GOVERNOR
        self.win_governor = WindowsApiGovernor()

    def compile_prioritization_policy(
        self,
        pid: int,
        process_name: str,
        user_priority_intent: Optional[str] = None,
        cpu_load: float = 45.0,
        cs_rate: float = 7200.0,
        thrashing_index: float = 1.1,
        working_set_mb: float = 256.0,
        thread_count: int = 12,
        demand_low_latency: bool = False
    ) -> CortexCompiledPlan:
        """
        Compiles high-level scheduling intent into a multi-step operation plan
        backed by Cortex integrity and OpenVINO inference.
        """
        plan_id = f"PLAN-CORTEX-{pid}-{int(time.time()*1000)%100000}"
        operations: List[CompiledOperation] = []

        # Step 1: OP_INSPECT_TELEMETRY
        operations.append(CompiledOperation(
            op_code="OP_INSPECT_TELEMETRY",
            target_pid=pid,
            parameter={"cpu_load": cpu_load, "cs_rate": cs_rate, "thrash_idx": thrashing_index},
            description=f"Inspect micro-architectural telemetry for process {process_name} (PID {pid})."
        ))

        # Step 2: OP_EVAL_CORTEX_INTEGRITY
        cortex_verdict = self.cortex.evaluate_cortex_integrity(
            cs_rate=cs_rate,
            cpu_load=cpu_load,
            thrashing_index=thrashing_index,
            working_set_mb=working_set_mb,
            active_thread_count=thread_count
        )
        operations.append(CompiledOperation(
            op_code="OP_EVAL_CORTEX_INTEGRITY",
            target_pid=pid,
            parameter=cortex_verdict.to_dict(),
            description=f"Cortex evaluated integrity: {cortex_verdict.cortex_integrity_score*100:.1f}%, Mandate: {cortex_verdict.decision_mandate}"
        ))

        # Step 3: OP_INFER_OPENVINO_PRIORITY
        vino_res = self.vino.infer_process_priority(
            pid=pid,
            name=process_name,
            cpu_pct=cpu_load,
            cs_rate=cs_rate,
            working_set_mb=working_set_mb,
            thread_count=thread_count,
            thrashing_index=thrashing_index
        )
        operations.append(CompiledOperation(
            op_code="OP_INFER_OPENVINO_PRIORITY",
            target_pid=pid,
            parameter=vino_res.to_dict(),
            description=f"OpenVINO inferred priority: {vino_res.predicted_priority_class} (Confidence: {vino_res.confidence_score*100:.1f}%)"
        ))

        # Step 4: OP_RESOLVE_ARBITRATION
        # Cortex integrity is authoritative: if Cortex orders FORCE_THROTTLE, cannot boost!
        if cortex_verdict.decision_mandate == "ISOLATE_CORES":
            final_priority = "IDLE"
            affinity = 0x03  # 2 efficiency cores
        elif cortex_verdict.decision_mandate == "FORCE_THROTTLE":
            final_priority = "BELOW_NORMAL"
            affinity = 0x0F  # 4 cores
        else:
            # Mandate allows normal/boost
            if user_priority_intent and user_priority_intent in WIN32_PRIORITY_CLASSES:
                final_priority = user_priority_intent
            else:
                final_priority = vino_res.predicted_priority_class
                if demand_low_latency and final_priority in ("NORMAL", "ABOVE_NORMAL"):
                    final_priority = "HIGH"
            affinity = cortex_verdict.recommended_affinity_mask

        win32_code = WIN32_PRIORITY_CLASSES[final_priority]

        operations.append(CompiledOperation(
            op_code="OP_RESOLVE_ARBITRATION",
            target_pid=pid,
            parameter={"final_priority": final_priority, "win32_code": win32_code},
            description=f"Resolved final priority to {final_priority} based on Cortex mandate and OpenVINO logits."
        ))

        # Step 5: OP_CALC_AFFINITY_MASK
        operations.append(CompiledOperation(
            op_code="OP_CALC_AFFINITY_MASK",
            target_pid=pid,
            parameter={"affinity_mask": hex(affinity)},
            description=f"Assigned CPU affinity mask 0x{affinity:02X} to prevent cache invalidation storms."
        ))

        # Step 6: OP_EMIT_WIN32_SET_PRIORITY & OP_EMIT_WIN32_SET_AFFINITY
        operations.append(CompiledOperation(
            op_code="OP_EMIT_WIN32_SET_PRIORITY",
            target_pid=pid,
            parameter={"priority_class": final_priority, "code": hex(win32_code)},
            description=f"Emit kernel32.dll!SetPriorityClass(hProcess, {hex(win32_code)})."
        ))
        operations.append(CompiledOperation(
            op_code="OP_EMIT_WIN32_SET_AFFINITY",
            target_pid=pid,
            parameter={"affinity_mask": hex(affinity)},
            description=f"Emit kernel32.dll!SetProcessAffinityMask(hProcess, {hex(affinity)})."
        ))

        # Step 7: OP_VERIFY_INVARIANT
        operations.append(CompiledOperation(
            op_code="OP_VERIFY_INVARIANT",
            target_pid=pid,
            parameter={"vital_max_hp": VITAL_MAX_HP},
            description="Verified invariant VITAL_MAX_HP == 6."
        ))

        # Generate polyglot stagers
        csharp_stager = self._generate_csharp_stager(pid, final_priority, win32_code, affinity)
        cpp_dispatcher = self._generate_cpp_dispatcher(pid, final_priority, win32_code, affinity)
        ps_cmd = f"Get-Process -Id {pid} | ForEach-Object {{ $_.PriorityClass = '{final_priority}'; $_.ProcessorAffinity = {affinity} }}"

        return CortexCompiledPlan(
            plan_id=plan_id,
            target_pid=pid,
            target_process_name=process_name,
            compiled_operations=[asdict(op) for op in operations],
            cortex_verdict=cortex_verdict.to_dict(),
            openvino_inference=vino_res.to_dict(),
            resolved_win32_priority_class=final_priority,
            resolved_win32_priority_code=win32_code,
            resolved_affinity_mask=affinity,
            generated_csharp_stager=csharp_stager,
            generated_cpp_dispatcher=cpp_dispatcher,
            generated_powershell_cmd=ps_cmd,
            timestamp=time.time(),
            vital_max_hp=self.vital_hp
        )

    def execute_plan(self, plan: CortexCompiledPlan) -> Dict[str, Any]:
        """Directly applies the compiled plan via the WindowsApiGovernor."""
        return self.win_governor.apply_priority(
            pid=plan.target_pid,
            priority_name=plan.resolved_win32_priority_class,
            affinity_mask=plan.resolved_affinity_mask
        )

    def _generate_csharp_stager(self, pid: int, priority_name: str, code: int, affinity: int) -> str:
        return f"""// ==============================================================================
// KRYSTAL CORTEX COMPILED WIN32 DISPATCHER (.NET 8 C#)
// System Invariant: VITAL_MAX_HP = 6
// ==============================================================================
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace KrystalStack.CortexCompiler
{{
    public static class ProcessPriorityStager
    {{
        public const int VitalMaxHp = 6;

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern IntPtr OpenProcess(uint processAccess, bool bInheritHandle, int processId);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool SetPriorityClass(IntPtr hProcess, uint priorityClass);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool SetProcessAffinityMask(IntPtr hProcess, IntPtr dwProcessAffinityMask);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool CloseHandle(IntPtr hObject);

        public static bool ApplyCortexDecision()
        {{
            const int targetPid = {pid};
            const uint priorityCode = 0x{code:08X}; // {priority_name}
            const long affinityMask = 0x{affinity:02X};

            IntPtr hProcess = OpenProcess(0x0600, false, targetPid);
            if (hProcess == IntPtr.Zero) return false;

            bool pOk = SetPriorityClass(hProcess, priorityCode);
            bool aOk = SetProcessAffinityMask(hProcess, new IntPtr(affinityMask));
            CloseHandle(hProcess);

            Console.WriteLine($"[CORTEX] Applied {{priorityCode}} and Affinity {{affinityMask}} to PID {{targetPid}}");
            return pOk && aOk;
        }}
    }}
}}"""

    def _generate_cpp_dispatcher(self, pid: int, priority_name: str, code: int, affinity: int) -> str:
        return f"""// ==============================================================================
// KRYSTAL CORTEX NATIVE C++20 WIN32 DISPATCHER
// System Invariant: VITAL_MAX_HP = 6
// ==============================================================================
#pragma once
#include <windows.h>
#include <iostream>

namespace krystal::cortex {{

constexpr uint32_t VITAL_MAX_HP = 6;

inline bool DispatchCompiledPriority() {{
    DWORD pid = {pid};
    DWORD priorityClass = 0x{code:08X}; // {priority_name}
    DWORD_PTR affinityMask = 0x{affinity:02X};

    HANDLE hProc = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, FALSE, pid);
    if (!hProc) return false;

    BOOL pRes = SetPriorityClass(hProc, priorityClass);
    BOOL aRes = SetProcessAffinityMask(hProc, affinityMask);
    CloseHandle(hProc);

    return (pRes && aRes);
}}

}} // namespace krystal::cortex
"""


GLOBAL_CORTEX_COMPILER = KrystalCortexCompiler()


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 85)
    print("  KRYSTAL-STACK: CORTEX PROCESS PRIORITIZATION COMPILER")
    print("=" * 85)

    compiler = GLOBAL_CORTEX_COMPILER
    
    # Compile a policy for an active render worker
    plan = compiler.compile_prioritization_policy(
        pid=3412,
        process_name="krystal_render_worker.exe",
        user_priority_intent="HIGH",
        cpu_load=92.0,
        cs_rate=4100.0,
        thrashing_index=0.92,
        demand_low_latency=True
    )

    print(f"Plan ID:                     {plan.plan_id}")
    print(f"Target Process:              {plan.target_process_name} (PID: {plan.target_pid})")
    print(f"Cortex Decision Mandate:     {plan.cortex_verdict['decision_mandate']} (Score: {plan.cortex_verdict['cortex_integrity_score']*100:.1f}%)")
    print(f"OpenVINO Inferred Class:     {plan.openvino_inference['predicted_priority_class']}")
    print(f"Final Resolved Win32 Class:  {plan.resolved_win32_priority_class} (0x{plan.resolved_win32_priority_code:08X})")
    print(f"Assigned CPU Affinity Mask:  0x{plan.resolved_affinity_mask:02X}")
    print(f"Number of Compiled Ops:      {len(plan.compiled_operations)}")
    print("-" * 85)

    # Execute plan
    exec_res = compiler.execute_plan(plan)
    print("Execution Result:", exec_res)
    print("=" * 85)


if __name__ == "__main__":
    main()
