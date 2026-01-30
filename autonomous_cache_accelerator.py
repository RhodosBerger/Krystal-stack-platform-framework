import time
import random
import threading
from typing import Dict

class SmartCache:
    """
    Simulovaná RAM Cache (L2/L3 Buffer).
    """
    def __init__(self):
        self.storage = {}
        
    def add(self, filename, content):
        print(f"[CACHE] Presúvam '{filename}' do RAM (Zero-Latency Zone)...")
        self.storage[filename] = content
        
    def has(self, filename):
        return filename in self.storage

class AutonomousAccelerator:
    """
    Modul, ktorý sleduje I/O operácie a aplikuje Axiómy v praxi.
    """
    def __init__(self):
        self.access_log: Dict[str, int] = {}
        self.cache = SmartCache()
        self.threshold = 3 # Koľkokrát musí byť súbor žiadaný, aby sa cachoval
        
    def request_file(self, filename: str) -> str:
        """Simulácia požiadavky na súbor (napr. textúra v hre)"""
        
        # 1. Kontrola Cache (Rýchla Cesta)
        if self.cache.has(filename):
            return f"DATA_FROM_RAM_SPEED_OF_LIGHT ({filename})"
        
        # 2. Pomalá Cesta (Disk)
        time.sleep(0.1) # Simulácia latencie disku
        
        # 3. Indukcia (Učenie)
        self.access_log[filename] = self.access_log.get(filename, 0) + 1
        print(f"[DISK] Čítam '{filename}' (Hit: {self.access_log[filename]})")
        
        # 4. Dedukcia (Rozhodnutie)
        if self.access_log[filename] >= self.threshold:
            self._deduce_and_optimize(filename)
            
        return f"DATA_FROM_DISK_SLOW ({filename})"

    def _deduce_and_optimize(self, filename):
        """
        Axióm: Ak Hit > Threshold -> Presuň do RAM.
        """
        print(f"  >>> DEDUKCIA: Súbor '{filename}' je 'HOT'. Aktivujem akceleráciu.")
        # Simulácia obsahu
        content = f"BINARY_BLOB_{random.randint(0,999)}"
        self.cache.add(filename, content)

# --- Simulácia ---
if __name__ == "__main__":
    accel = AutonomousAccelerator()
    
    files = ["shader_base.vsh", "texture_sky.png", "config.ini"]
    
    print("--- FÁZA 1: Prvé prístupy (Učenie) ---")
    for _ in range(3):
        f = random.choice(files)
        print(accel.request_file(f))
        time.sleep(0.2)
        
    print("\n--- FÁZA 2: Opakované prístupy (Optimalizácia) ---")
    # Vynútime opakovaný prístup k jednému súboru
    target = "texture_sky.png"
    for _ in range(4):
        print(accel.request_file(target))
        
    print("\n--- FÁZA 3: Finálny Test (Zero Latency) ---")
    # Teraz by to malo ísť z RAM okamžite
    print(accel.request_file(target))
