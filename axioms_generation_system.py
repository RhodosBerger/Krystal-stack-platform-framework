import json
import random
import time
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ObservedEvent:
    timestamp: float
    trigger: str      # e.g., "CPU_SPIKE"
    consequence: str  # e.g., "FPS_DROP"
    latency: float    # Čas medzi triggerom a následkom

@dataclass
class Axiom:
    name: str
    condition: str
    preventive_action: str
    confidence: float
    origin: str = "AUTONOMOUS_DEDUCTION"

class AxiomsGenerator:
    """
    Tento systém vykonáva 'Filozofickú Prácu' stroja.
    Pozerá sa na minulosť (Indukcia) a píše zákony pre budúcnosť (Dedukcia).
    """
    def __init__(self):
        self.memory_stream: List[ObservedEvent] = []
        self.known_axioms: List[Axiom] = []
        
    def observe(self, trigger: str, consequence: str, latency_ms: float):
        """Krok 1: Zber Dát (Empíria)"""
        event = ObservedEvent(time.time(), trigger, consequence, latency_ms)
        self.memory_stream.append(event)
        # Udržuj len krátkodobú pamäť pre rýchlu analýzu
        if len(self.memory_stream) > 100:
            self.memory_stream.pop(0)

    def induce_patterns(self):
        """Krok 2: Indukcia (Hľadanie Vzorov)"""
        print("\n--- INDUKČNÁ FÁZA ---")
        correlations = {}
        
        for event in self.memory_stream:
            key = f"{event.trigger}->{event.consequence}"
            if key not in correlations:
                correlations[key] = []
            correlations[key].append(event.latency)
            
            # Analýza korelácie
        for key, latencies in correlations.items():
            count = len(latencies)
            avg_lat = sum(latencies) / count
            
            if count >= 3: # Ak sa to stalo aspoň 3x, je to Vzor.
                print(f"  > Pozorovanie: '{key}' sa stalo {count}x (Avg Latency: {avg_lat:.1f}ms)")
                self.deduce_axiom(key, avg_lat, count)

    def deduce_axiom(self, pattern_key: str, avg_latency: float, strength: int):
        """Krok 3: Dedukcia (Tvorba Pravidla)"""
        trigger, consequence = pattern_key.split("->")
        
        # Logika Dedukcie:
        # Ak A spôsobuje B (negatívne), musíme urobiť C skôr, než nastane B.
        
        action = "MONITOR"
        axiom_name = f"LAW_OF_{trigger}_{random.randint(100,999)}"
        
        if consequence == "FPS_DROP":
            action = "PRE_RENDER_FRAMES" # Kompenzácia
        elif consequence == "THERMAL_THROTTLE":
            action = "PRE_EMPTIVE_FAN_BOOST" # Prevencia
        elif consequence == "IO_WAIT":
            action = "RAM_CACHE_INJECTION" # Zrýchlenie
            
        new_axiom = Axiom(
            name=axiom_name,
            condition=f"IF DETECTED({trigger})",
            preventive_action=f"EXECUTE({action}) WITHIN {avg_latency * 0.8:.1f}ms",
            confidence=min(0.99, strength * 0.2)
        )
        
        # Check if exists
        if not any(a.name.startswith(f"LAW_OF_{trigger}") for a in self.known_axioms):
            self.known_axioms.append(new_axiom)
            print(f"  >>> DEDUKCIA: Nový Axióm Vytvorený: {new_axiom.name}")
            print(f"      Pravidlo: {new_axiom.condition} -> {new_axiom.preventive_action}")

    def save_knowledge(self):
        """Uloží 'Múdrosť' do JSONu"""
        data = [vars(a) for a in self.known_axioms]
        with open("runtime_axioms.json", "w") as f:
            json.dump(data, f, indent=2)
        print("\n[SYSTEM] Axiómy zapísané do 'runtime_axioms.json'.")

# --- Simulácia Autonómneho Učenia ---
if __name__ == "__main__":
    brain = AxiomsGenerator()
    
    print("Simulujem prevádzku systému...")
    
    # 1. Simulácia udalostí (Tréning)
    # Systém si všimne, že High Load spôsobuje Thermal Throttle
    for _ in range(5):
        brain.observe("HIGH_CPU_LOAD", "THERMAL_THROTTLE", random.uniform(1500, 2000))
        time.sleep(0.1)
        
    # Systém si všimne, že Disk Access spôsobuje FPS Drop
    for _ in range(4):
        brain.observe("DISK_ACCESS_HEAVY", "FPS_DROP", random.uniform(50, 100))
        time.sleep(0.1)

    # 2. Spustenie Myšlienkového Procesu
    brain.induce_patterns()
    
    # 3. Uloženie
    brain.save_knowledge()
