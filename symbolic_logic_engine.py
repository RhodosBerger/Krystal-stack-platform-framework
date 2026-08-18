import time
import random
from typing import List, Dict, Callable

class SymbolicLogicEngine:
    """
    ENGINE: VETVENIE LOGIKY POMOCOU SYMBOLIKY
    Tento systém nepoužíva 'if cpu > 80', ale 'if Ω in sequence'.
    Symboly sú inštrukcie.
    """
    
    def __init__(self):
        # 1. Definícia Glyfov (Symbolov) a ich "Váhy"
        self.glyphs = {
            "ORACLE": "Ψ",    # Predikcia / Budúcnosť
            "FORCE": "⚡",     # Výkon / Napätie
            "RESIST": "Ω",    # Odpor / Inhibícia
            "VOID": "::",     # Prázdno / Pripravenosť
            "CHAOS": "≈",     # Tok / Zmena
            "WALL": "║"       # Blokáda
        }
        
        # 2. Mapovanie Vzorov na Funkcie (The Logic Map)
        self.logic_branches = {
            "Ψ⚡": self._branch_prophetic_boost,  # Predikcia + Sila = Prediktívny Boost
            "Ω║": self._branch_hard_brake,       # Odpor + Stena = Núdzová Brzda
            "≈::": self._branch_fluid_optimization, # Tok + Prázdno = Defragmentácia
            "⚡Ω": self._branch_force_collapse    # Sila + Odpor = Prehriatie (Riziko)
        }

    def transmute_telemetry(self, telemetry: Dict) -> str:
        """
        Krok 1: Premena surových dát na Magickú Sekvenciu (Glyfy).
        """
        sequence = ""
        
        # A. Analýza Predikcie
        if telemetry.get('predicted_load', 0) > telemetry.get('current_load', 0):
            sequence += self.glyphs["ORACLE"]  # Pridaj Ψ
            
        # B. Analýza Napätia/Výkonu
        if telemetry.get('cpu_load', 0) > 80:
            sequence += self.glyphs["FORCE"]   # Pridaj ⚡
        elif telemetry.get('cpu_load', 0) < 20:
            sequence += self.glyphs["VOID"]    # Pridaj ::
            
        # C. Analýza Odporu (Teplota)
        if telemetry.get('temp', 0) > 75:
            sequence += self.glyphs["RESIST"]  # Pridaj Ω
            if telemetry.get('temp', 0) > 90:
                sequence += self.glyphs["WALL"] # Pridaj ║ (Kritické)
        else:
            sequence += self.glyphs["CHAOS"]   # Pridaj ≈ (Normálny tok)
            
        return sequence

    def execute_sequence(self, sequence: str):
        """
        Krok 2: Vykonanie vetvy na základe symbolov.
        Hľadá "Magické Slová" v sekvencii.
        """
        print(f"\n[ENGINE] Spracovávam sekvenciu: [{sequence}]")
        action_taken = False
        
        # Prechádzame známe vzory (Patterns)
        for pattern, handler_func in self.logic_branches.items():
            if pattern in sequence:
                print(f"  > Zhoda Vzoru '{pattern}' -> Aktivujem vetvu: {handler_func.__name__}")
                handler_func() # Spusti logiku
                action_taken = True
                
        if not action_taken:
            self._branch_idle_monitoring()

    # --- Logické Vetvy (Handler Functions) ---

    def _branch_prophetic_boost(self):
        """Vetva: Ψ⚡"""
        print("    >>> ACTION: PRE-FETCHING ASSETS (Oracle vidí záťaž)")
        print("    >>> ACTION: INJECTING MICRO-VOLTAGE (Priprava na náraz)")

    def _branch_hard_brake(self):
        """Vetva: Ω║"""
        print("    >>> CRITICAL: THERMAL LIMIT REACHED")
        print("    >>> ACTION: HALT ALL THREADS (Inhibícia)")

    def _branch_fluid_optimization(self):
        """Vetva: ≈::"""
        print("    >>> ACTION: RE-ALIGNING HEX GRID (Systém je v kľude)")
        print("    >>> ACTION: GARBAGE COLLECTION")

    def _branch_force_collapse(self):
        """Vetva: ⚡Ω"""
        print("    >>> WARNING: HIGH POWER vs HIGH RESISTANCE")
        print("    >>> ACTION: FAN SPEED 100%")

    def _branch_idle_monitoring(self):
        print("    >>> Status: Nominal. Waiting for sigils.")

# --- Simulácia Beh ---

if __name__ == "__main__":
    engine = SymbolicLogicEngine()
    
    # Scenár 1: Hra sa načítava (Vysoká predikcia, Nízka aktuálna záťaž)
    telemetry_game_load = {
        "current_load": 20,
        "predicted_load": 90, # Oracle vidí budúcnosť
        "temp": 40
    }
    
    # Scenár 2: Prehrievanie
    telemetry_overheat = {
        "current_load": 95,
        "predicted_load": 95,
        "temp": 92 # Kritické
    }
    
    print("--- SCENÁR 1: The Prophet ---")
    seq1 = engine.transmute_telemetry(telemetry_game_load)
    engine.execute_sequence(seq1) # Mala by nájsť Ψ⚡ alebo podobné
    
    time.sleep(1)
    
    print("--- SCENÁR 2: The Crash ---")
    seq2 = engine.transmute_telemetry(telemetry_overheat)
    engine.execute_sequence(seq2) # Mala by nájsť Ω║
