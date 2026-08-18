import time
import random
import json
import uuid
from typing import Dict, List, Any
from dataclasses import dataclass

# --- 1. HISTORICAL MODEL (PAMÄŤ MINULÝCH ROZHODNUTÍ) ---

class HistoricalModel:
    """
    Databáza skúseností. Učí sa, ktoré rozhodnutia boli dobré.
    """
    def __init__(self):
        # Format: { "PROCESS_SIGNATURE": { "trust_score": 0.5, "past_crimes": 0 } }
        self.reputation_db = {}
        
    def get_reputation(self, signature: str) -> float:
        if signature not in self.reputation_db:
            # Nový proces začína s neutrálnou dôverou
            self.reputation_db[signature] = {"trust_score": 0.5, "past_crimes": 0}
        return self.reputation_db[signature]["trust_score"]

    def update_reputation(self, signature: str, delta: float, reason: str):
        if signature not in self.reputation_db: self.get_reputation(signature)
        
        entry = self.reputation_db[signature]
        entry["trust_score"] = max(0.0, min(1.0, entry["trust_score"] + delta))
        
        if delta < 0:
            entry["past_crimes"] += 1
            
        print(f"[MODEL] Updated '{signature}': Trust={entry['trust_score']:.2f} | Reason: {reason}")

# --- 2. LOG PARSER & VECTORIZER (ZMYSLY) ---

class LogVector:
    """Reprezentuje log ako matematický objekt pre OpenVINO."""
    def __init__(self, raw_text: str, source: str):
        self.raw = raw_text
        self.source = source
        self.risk_vector = self._vectorize()
        
    def _vectorize(self) -> float:
        """Premení text na číslo rizika (0.0 - 1.0)"""
        text = self.raw.lower()
        if "error" in text or "fail" in text or "timeout" in text:
            return 0.9 # Vysoké riziko
        if "warning" in text or "retry" in text:
            return 0.5 # Stredné riziko
        if "success" in text or "optimized" in text:
            return 0.0 # Žiadne riziko (Prínos)
        return 0.1 # Neznáme

class NeuralLogInspector:
    """
    Hlavný Inšpektor. Číta logy, konzultuje Model a riadi Pooly.
    """
    def __init__(self):
        self.history = HistoricalModel()
        
        # Algoritmické Pooly
        self.pools = {
            "ELITE": [],      # High Priority, L1 Cache
            "STANDARD": [],   # Normal Priority
            "PROBATION": [],  # Monitored
            "QUARANTINE": []  # Restricted
        }
        
    def ingest_log_stream(self, process_id: str, log_lines: List[str]):
        """
        Spracuje dávku logov pre daný proces.
        """
        print(f"\n--- INSPECTING PROCESS: {process_id} ---")
        
        # 1. Parsing & Vectorization
        total_risk = 0.0
        contribution_points = 0
        
        for line in log_lines:
            vector = LogVector(line, process_id)
            total_risk += vector.risk_vector
            
            # Heuristika pre body prínosu
            if vector.risk_vector == 0.0:
                contribution_points += 10
            elif vector.risk_vector > 0.5:
                contribution_points -= 20
                
        avg_risk = total_risk / len(log_lines) if log_lines else 0
        
        # 2. OpenVINO Decision (Simulované)
        # "Sudca" rozhoduje na základe rizika a reputácie
        current_reputation = self.history.get_reputation(process_id)
        
        decision_score = (current_reputation * 0.7) + ((1.0 - avg_risk) * 0.3)
        
        # 3. Feedback Loop (Učenie)
        # Ak mal proces vysoké riziko, zníž reputáciu
        if avg_risk > 0.4:
            self.history.update_reputation(process_id, -0.1, "High Risk Logs Detected")
        else:
            self.history.update_reputation(process_id, 0.05, "Stable Operation")

        # 4. Pool Assignment (Exekúcia)
        self._assign_pool(process_id, decision_score)
        
    def _assign_pool(self, process_id: str, score: float):
        # Odstránenie zo starých poolov
        for p in self.pools.values():
            if process_id in p: p.remove(process_id)
            
        target_pool = "STANDARD"
        
        if score > 0.85:
            target_pool = "ELITE"
        elif score < 0.3:
            target_pool = "QUARANTINE"
        elif score < 0.6:
            target_pool = "PROBATION"
            
        self.pools[target_pool].append(process_id)
        print(f"[JUDGEMENT] Process '{process_id}' assigned to [{target_pool}] Pool. (Score: {score:.2f})")

    def print_system_state(self):
        print("\n=== SYSTEM POOL STATUS ===")
        for name, processes in self.pools.items():
            print(f"[{name}]: {processes}")
        print("==========================")

# --- SIMULÁCIA ---

if __name__ == "__main__":
    inspector = NeuralLogInspector()
    
    # Scenár 1: Stabilný Render Engine
    logs_render = [
        "Initializing shader cache... SUCCESS",
        "Geometry load time: 12ms (Optimized)",
        "Vulkan context created."
    ]
    
    # Scenár 2: Problémový Sieťový Modul
    logs_network = [
        "Connection attempt 1... TIMEOUT",
        "Retrying packet 404... WARNING",
        "Socket error: Buffer overflow."
    ]
    
    # Scenár 3: Nový, neznámy plugin
    logs_plugin = [
        "Plugin loaded.",
        "Waiting for input..."
    ]
    
    # Cyklus Inšpekcie
    inspector.ingest_log_stream("Render_Engine_v4", logs_render)
    inspector.ingest_log_stream("Net_Module_X", logs_network)
    inspector.ingest_log_stream("Unknown_Plugin", logs_plugin)
    
    # Výpis stavu
    inspector.print_system_state()
    
    # Druhé kolo (Evolúcia) - Sieťový modul pokračuje v chybách
    print("\n--- ROUND 2 (Evolution) ---")
    logs_network_2 = ["CRITICAL FAILURE: Port Closed."]
    inspector.ingest_log_stream("Net_Module_X", logs_network_2)
    
    inspector.print_system_state()
