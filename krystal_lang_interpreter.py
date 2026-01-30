import sys
import json
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field

# --- Dátový Model Vzťahov ---

@dataclass
class Node:
    id: str
    type: str # SENSOR, ACTOR, MEMORY, LOGIC
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    source: str
    target: str
    operator: str # ->, =>, ::, <>
    metadata: Dict[str, Any] = field(default_factory=dict)

class RelationGraph:
    """Pamäťová databáza vzťahov."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.relations: List[Relation] = []

    def add_node(self, node_id, node_type="GENERIC", params=None):
        self.nodes[node_id] = Node(node_id, node_type, params or {})

    def add_relation(self, source, operator, target, meta=None):
        # Auto-create nodes if not exist
        if source not in self.nodes: self.add_node(source)
        if target not in self.nodes: self.add_node(target)
        
        self.relations.append(Relation(source, target, operator, meta or {}))
        print(f"[GRAPH] Vytvorený vzťah: [{source}] {operator} [{target}]")

    def find_problems(self):
        """Telemetrická analýza vzťahov."""
        issues = []
        for rel in self.relations:
            # Simulácia kontroly: Ak je vo vzťahu "LATENCY" > 100, je to problém
            latency = rel.metadata.get("latency", 0)
            if latency > 100:
                issues.append(f"CRITICAL LATENCY on link {rel.source}->{rel.target} ({latency}ms)")
        return issues

# --- Interpreter Jazyka ---

class KrystalInterpreter:
    def __init__(self):
        self.graph = RelationGraph()
        # Regex pre parsovanie: [SOURCE] OP [TARGET] @ {JSON}
        self.pattern = re.compile(r"\[([\w_]+)\]\s*([\->=:<]+)\s*\[([\w_]+)\]\s*(?:@\s*(\{.*\}))?")

    def execute(self, command: str):
        command = command.strip()
        if not command: return

        # 1. Príkaz pre Status
        if command == "status":
            print(f"Nodes: {len(self.graph.nodes)} | Relations: {len(self.graph.relations)}")
            return

        if command == "analyze":
            problems = self.graph.find_problems()
            if problems:
                for p in problems: print(f"[!] {p}")
            else:
                print("[OK] System Coherence Nominal.")
            return

        # 2. Parsovanie Vzťahu
        match = self.pattern.match(command)
        if match:
            src, op, tgt, meta_str = match.groups()
            meta = {}
            if meta_str:
                try:
                    meta = json.loads(meta_str)
                except:
                    print(f"[ERR] Invalid JSON params: {meta_str}")
            
            self.graph.add_relation(src, op, tgt, meta)
        else:
            print(f"[ERR] Syntax Error: {command}")
            print("Usage: [SOURCE] -> [TARGET] @ {params}")

    def run_shell(self):
        print("KRYSTAL-LANG Interactive Shell v1.0")
        print("Paradigm: Coding Scheme Relations")
        print("Type 'exit' to quit.")
        
        while True:
            try:
                cmd = input("kry> ")
                if cmd in ["exit", "quit"]: break
                self.execute(cmd)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[EXCEPTION] {e}")

# --- CLI Entry Point ---

if __name__ == "__main__":
    interpreter = KrystalInterpreter()
    
    # Ak sú argumenty z príkazového riadku (Bash mód)
    if len(sys.argv) > 1:
        # Spojí argumenty do jedného stringu (napr. kry [CPU]->[RAM])
        full_cmd = " ".join(sys.argv[1:])
        interpreter.execute(full_cmd)
    else:
        # Inak spustí interaktívny mód
        interpreter.run_shell()
