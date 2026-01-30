import ctypes
import time
import random
import threading
import sys
from ctypes import wintypes

# --- 1. WINDOWS API WRAPPER (ZMYSLOVÉ ORGÁNY) ---

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]

class WindowsSense:
    """
    Trieda, ktorá 'cíti' stav operačného systému cez Kernel32.
    """
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        
    def get_memory_pressure(self):
        """Vráti percentuálne zaťaženie pamäte (0-100)"""
        mem_status = MEMORYSTATUSEX()
        mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        self.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
        return mem_status.dwMemoryLoad

    def set_thread_priority(self, level="NORMAL"):
        """
        Zmení prioritu aktuálneho vlákna (Motorická reakcia).
        Levely: LOW, NORMAL, HIGH, REALTIME
        """
        handle = self.kernel32.GetCurrentThread()
        
        # Windows Constants
        THREAD_PRIORITY_LOWEST = -2
        THREAD_PRIORITY_NORMAL = 0
        THREAD_PRIORITY_HIGHEST = 2
        
        priority = THREAD_PRIORITY_NORMAL
        if level == "LOW": priority = THREAD_PRIORITY_LOWEST
        elif level == "HIGH": priority = THREAD_PRIORITY_HIGHEST
        
        success = self.kernel32.SetThreadPriority(handle, priority)
        return success

# --- 2. OPENVINO BRAIN SIMULATOR (MOZOG) ---

class OpenVinoCortex:
    """
    Simulácia neurónovej siete, ktorá spracováva dáta z WindowsSense.
    V produkcii by tu bol: from openvino.runtime import Core
    """
    def __init__(self):
        self.state = "ALPHA" # Východzí stav (Meditácia)
        self.history_tensor = [] 
        
    def infer_cognitive_state(self, sensory_input: list) -> str:
        """
        Vstup: Vektor [RAM_Load, CPU_Noise, User_Input]
        Výstup: Kognitívny Stav (ALPHA, BETA, GAMMA)
        """
        # 1. Normalizácia (Preprocessing)
        ram_load = sensory_input[0] / 100.0
        cpu_noise = sensory_input[1]
        
        # 2. Inferencia (Simulácia váh neurónovej siete)
        # Gamma Neuron Activation Function:
        gamma_score = (ram_load * 0.7) + (cpu_noise * 0.3)
        
        prev_state = self.state
        
        # 3. Rozhodovanie (Softmax)
        if gamma_score > 0.8:
            self.state = "GAMMA" # Hyper-Focus
        elif gamma_score > 0.4:
            self.state = "BETA"  # Active
        else:
            self.state = "ALPHA" # Idle
            
        return self.state, gamma_score

# --- 3. KOGNITÍVNY PROCESOR (INTEGRÁCIA) ---

class CognitiveOS:
    def __init__(self):
        self.senses = WindowsSense()
        self.brain = OpenVinoCortex()
        self.running = True
        
        # Symbolika pre výstup
        self.symbols = {
            "ALPHA": "≈ (Flow)",
            "BETA": ":: (Grid)",
            "GAMMA": "⚡ (Force)"
        }

    def run_thought_cycle(self):
        print("--- KOGNITÍVNY OS INICIALIZOVANÝ ---")
        print("Prepojenie: Kernel32 <-> OpenVINO <-> Python Logic\n")
        
        try:
            while self.running:
                # 1. ZMYSLY (SENSE)
                mem_load = self.senses.get_memory_pressure()
                cpu_noise = random.random() # Simulácia CPU šumu
                
                # 2. MYSLENIE (THINK - OpenVINO)
                input_tensor = [mem_load, cpu_noise]
                state, confidence = self.brain.infer_cognitive_state(input_tensor)
                
                # 3. AKCIA (ACT - Windows API)
                action_log = ""
                if state == "GAMMA":
                    # Ak sme v GAMMA, zvýš prioritu vlákna
                    self.senses.set_thread_priority("HIGH")
                    action_log = "INJECTING HIGH PRIORITY"
                elif state == "ALPHA":
                    # Ak sme v ALPHA, uvoľni zdroje
                    self.senses.set_thread_priority("LOW")
                    action_log = "RELEASING RESOURCES"
                else:
                    self.senses.set_thread_priority("NORMAL")
                    action_log = "MAINTAINING COHERENCE"

                # 4. VIZUALIZÁCIA (REBUS)
                symbol = self.symbols[state]
                bar_len = int(confidence * 10)
                visual_bar = "█" * bar_len + "░" * (10 - bar_len)
                
                print(f"SENSE[RAM:{mem_load}%] -> BRAIN[{symbol}] {visual_bar} -> ACT[{action_log}]")
                
                time.sleep(0.5) # 2Hz kognitívny cyklus
                
        except KeyboardInterrupt:
            print("\n--- ODPOJENIE VEDOMIA ---")

if __name__ == "__main__":
    # Spustenie Kognitívneho OS
    # (Vyžaduje Windows pre ctypes volania, na Linuxe použije mock)
    if sys.platform != 'win32':
        print("VAROVANIE: Detekovaný non-Windows systém. Spúšťam v simulačnom režime.")
        # Mock pre WindowsSense na Linuxe
        WindowsSense.get_memory_pressure = lambda self: random.randint(20, 95)
        WindowsSense.set_thread_priority = lambda self, l: True

    os_mind = CognitiveOS()
    os_mind.run_thought_cycle()
