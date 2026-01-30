#!/usr/bin/env python3
"""
PROCESS SCHEDULER: The "CPU" of the Factory.
Executes the Manufacturing Instruction Set.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SCHEDULER] - %(message)s')
logger = logging.getLogger(__name__)

class OpCode(Enum):
    ROUGH = "OP_ROUGH"    # Standard Roughing
    FINISH = "OP_FINISH"  # Precision Finishing
    PROBE = "OP_PROBE"    # Measurement
    COOL = "OP_COOL"      # Thermal Pause
    TOOL = "OP_TOOL"      # Tool Change

@dataclass
class ProcessInstruction:
    op_code: OpCode
    target: str # e.g., "POCKET_1", "SURFACE_TOP"
    params: Dict # Extra args like {depth: 10, tool: "T01"}

class ManufacturingCPU:
    def __init__(self):
        # Registers
        self.reg_heat = 0.0 # 0.0 - 1.0 (Thermal Load)
        self.reg_vib = 0.0  # Vibration History
        self.reg_tool = "NONE"
        
        self.pipeline = [] # Instruction Queue

    def load_program(self, instructions: List[ProcessInstruction]):
        self.pipeline = instructions
        logger.info(f"Loaded Program: {len(instructions)} Ops.")

    def run(self):
        """
        The Fetch-Decode-Execute Cycle.
        """
        pc = 0 # Program Counter
        while pc < len(self.pipeline):
            instr = self.pipeline[pc]
            logger.info(f"[FETCH] PC:{pc} | {instr.op_code.value} -> {instr.target}")
            
            # --- HAZARD DETECTION (Lookahead) ---
            if self._check_thermal_hazard(instr):
                logger.warning(">>> HAZARD: Thermal Limit Predicted! Inserting STALL (OP_COOL).")
                self._execute(ProcessInstruction(OpCode.COOL, "AUTO_INSERT", {"duration": 2}))
                # PC does not increment, we retry the instruction after cooling
                continue 

            # --- EXECUTE ---
            self._execute(instr)
            pc += 1
            
        logger.info("[HLT] Program Complete.")

    def _check_thermal_hazard(self, instr: ProcessInstruction) -> bool:
        """
        Returns True if executing this instruction would overheat the part
        for a subsequent Precision Op.
        """
        # Rule: Cannot RUN FINISH if Heat > 0.5
        if instr.op_code == OpCode.FINISH and self.reg_heat > 0.5:
            return True
        return False

    def _execute(self, instr: ProcessInstruction):
        """
         The ALU (Arithmetic Logic Unit).
        """
        op = instr.op_code
        
        if op == OpCode.ROUGH:
            # ROUGHING Generates Heat
            self.reg_heat += 0.3
            logger.info(f"   >>> ALU: Removing Material ({instr.target}). Heat rises to {self.reg_heat:.1f}")
            
        elif op == OpCode.FINISH:
            # FINISHING Requires Stability, generates little heat
            self.reg_heat += 0.05
            logger.info(f"   >>> ALU: Precision Pass ({instr.target}). Surface ok.")
            
        elif op == OpCode.COOL:
            # COOLING reduces Heat
            duration = instr.params.get("duration", 1)
            self.reg_heat = max(0.0, self.reg_heat - (0.3 * duration))
            logger.info(f"   >>> ALU: Cooling for {duration}s. Heat drops to {self.reg_heat:.1f}")
            
        elif op == OpCode.TOOL:
            new_tool = instr.params.get("id", "CTX")
            logger.info(f"   >>> ALU: Tool Change {self.reg_tool} -> {new_tool}")
            self.reg_tool = new_tool
            
        elif op == OpCode.PROBE:
            logger.info(f"   >>> ALU: Probing {instr.target}. Updating Registers.")

# Usage Example
if __name__ == "__main__":
    cpu = ManufacturingCPU()
    
    # Example Program: Aggressive Roughing -> Precision Finish
    # This should trigger a Thermal Hazard
    program = [
        ProcessInstruction(OpCode.TOOL, "LOAD_TOOL", {"id": "T01"}),
        ProcessInstruction(OpCode.ROUGH, "POCKET_A", {}), # Heat += 0.3 -> 0.3
        ProcessInstruction(OpCode.ROUGH, "POCKET_B", {}), # Heat += 0.3 -> 0.6
        ProcessInstruction(OpCode.TOOL, "LOAD_TOOL", {"id": "T02"}),
        ProcessInstruction(OpCode.FINISH, "SURFACE_A", {}) # Heat is 0.6! Should Trigger Hazard.
    ]
    
    cpu.load_program(program)
    cpu.run()
