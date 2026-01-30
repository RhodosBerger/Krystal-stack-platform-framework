import threading
import time
import json
import random
from typing import Dict, Any

# Importujeme našu chrbtovú kosť
from hyper_state_bus import HyperStateBus, StateMessage

class YggdrasilTree:
    """
    Udržiava Živý Strom (Live Tree) stavu celého systému.
    """
    def __init__(self):
        self.tree = {
            "timestamp": 0,
            "HARDWARE": {},
            "COGNITIVE": {},
            "TOPOLOGY": {}
        }
        self.lock = threading.Lock()

    def update_branch(self, branch: str, leaf: str, data: Any):
        with self.lock:
            self.tree["timestamp"] = time.time()
            if branch not in self.tree:
                self.tree[branch] = {}
            self.tree[branch][leaf] = data

    def get_snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.tree)) # Deep copy

class StrategosCommander:
    """
    Generál, ktorý sa pozerá na Yggdrasil Strom a vydáva rozkazy Kernelu.
    """
    def __init__(self, bus: HyperStateBus):
        self.bus = bus

    def analyze_and_command(self, tree_snapshot: Dict):
        """
        Logika: IF (Tree Pattern) THEN (Kernel Command)
        """
        hardware = tree_snapshot.get("HARDWARE", {})
        cognitive = tree_snapshot.get("COGNITIVE", {})
        
        # 1. Stratégia: GAMMA LOCKING
        # Ak je mozog v GAMMA stave a pamäť je pod tlakom -> ZAMKNI RAM
        openvino_state = cognitive.get("OpenVINO", {}).get("state", "ALPHA")
        ram_pressure = hardware.get("Kernel", {}).get("ram_pressure", 0)
        
        if openvino_state == "GAMMA" and ram_pressure > 60:
            self._issue_kernel_command("VIRTUAL_LOCK", {"target": "MAIN_PROCESS", "priority": "CRITICAL"})

        # 2. Stratégia: THERMAL AFFINITY
        # Ak je teplota vysoká -> PRESUŇ VLÁKNA
        cpu_temp = hardware.get("Sensors", {}).get("temp", 0)
        if cpu_temp > 85:
            self._issue_kernel_command("SHIFT_AFFINITY", {"target": "COOL_CORES_ONLY"})

    def _issue_kernel_command(self, cmd_type, params):
        print(f"[STRATEGOS] >>> Vydávam Kernel Rozkaz: {cmd_type} | Params: {params}")
        # V realite by tu bolo volanie ctypes.windll.kernel32...
        self.bus.publish("STRATEGOS", "KERNEL_OP", {"cmd": cmd_type, "params": params}, priority=2)

class SystemOrchestrator:
    """
    Hlavný Integrátor. Spája Bus, Strom a Stratéga.
    """
    def __init__(self):
        self.bus = HyperStateBus()
        self.yggdrasil = YggdrasilTree()
        self.commander = StrategosCommander(self.bus)
        self.running = True
        
        # Prihlásenie na odber všetkého (*)
        self.bus.subscribe("*", self.on_any_message)

    def on_any_message(self, msg: StateMessage):
        # 1. Update Stromu na základe témy správy
        if msg.source_id == "OPENVINO":
            self.yggdrasil.update_branch("COGNITIVE", "OpenVINO", msg.payload)
        elif msg.source_id == "KERNEL_SENSE":
            self.yggdrasil.update_branch("HARDWARE", "Kernel", msg.payload)
        elif msg.source_id == "GRID_CONTROLLER":
            self.yggdrasil.update_branch("TOPOLOGY", "HexGrid", msg.payload)

    def run_loop(self):
        print("--- YGGDRASIL ORCHESTRATOR ONLINE ---")
        
        # Simulácia externých komponentov posielajúcich dáta
        threading.Thread(target=self._simulate_components, daemon=True).start()
        
        try:
            while self.running:
                # 1. Získaj aktuálny obraz systému
                snapshot = self.yggdrasil.get_snapshot()
                
                # 2. Vizualizácia Stromu (JSON dump)
                # print(f"\n[YGGDRASIL] Current State:\n{json.dumps(snapshot, indent=2)}")
                
                # 3. Strategické Rozhodnutie
                self.commander.analyze_and_command(snapshot)
                
                time.sleep(1.0) # 1Hz Tick
        except KeyboardInterrupt:
            print("Shutting down.")

    def _simulate_components(self):
        """Simuluje dáta prichádzajúce z iných modulov cez Bus"""
        while self.running:
            # OpenVINO mení stavy
            state = random.choice(["ALPHA", "BETA", "GAMMA"])
            self.bus.publish("OPENVINO", "COGNITIVE_STATE", {"state": state})
            
            # Kernel hlási tlak
            pressure = random.randint(40, 90)
            self.bus.publish("KERNEL_SENSE", "MEMORY_STATUS", {"ram_pressure": pressure})
            
            time.sleep(0.5)

if __name__ == "__main__":
    orch = SystemOrchestrator()
    orch.run_loop()
