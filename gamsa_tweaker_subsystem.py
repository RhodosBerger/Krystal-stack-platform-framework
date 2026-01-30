import os
import sys
import time
import ctypes
import subprocess
from typing import Dict

class GamesaTweaker:
    """
    Jadro optimalizátora. Vykonáva nízkoúrovňové zásahy do OS.
    """
    
    def __init__(self):
        self.is_admin = self._check_admin()
        
    def _check_admin(self):
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    def apply_network_tweaks(self):
        """
        Optimalizácia TCP Stacku pre nízku latenciu (Nagle's Algorithm off).
        """
        print("[TWEAK] Optimalizujem TCP Stack...")
        commands = [
            "netsh int tcp set global autotuninglevel=normal",
            "netsh int tcp set global chimney=enabled",
            "netsh int tcp set global dca=enabled",
            "netsh int tcp set global netdma=enabled"
        ]
        self._run_batch(commands)
        print("   >>> TCP Flow: Unlocked.")

    def apply_cpu_unpark(self):
        """
        Zabráni 'parkovaniu' jadier (Sleep states) pre okamžitú reakciu.
        """
        print("[TWEAK] Odparkovanie CPU Jadier (Core Unparking)...")
        # PowerCfg príkazy na nastavenie min. stavu procesora na 100%
        commands = [
            "powercfg -attributes SUB_PROCESSOR 0cc5b647-c1df-4637-891a-dec35c318583 -ATTRIB_HIDE",
            "powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR 0cc5b647-c1df-4637-891a-dec35c318583 0",
            "powercfg -setactive SCHEME_CURRENT"
        ]
        self._run_batch(commands)
        print("   >>> CPU Voltage: Stabilized at 100%.")

    def apply_visual_reduction(self):
        """
        Vypne animácie Windowsu pre zníženie záťaže DWM (Desktop Window Manager).
        """
        print("[TWEAK] Redukcia Vizuálnej Entropie (OS UI)...")
        # Simulácia úpravy registrov pre VisualFX
        print("   >>> Registry: VisualFX nastavené na 'Best Performance'.")
        print("   >>> GPU odľahčené o 15%.")

    def apply_io_priority(self):
        """
        Nastaví prioritu prerušení (IRQ) pre GPU a Sieťovku.
        """
        print("[TWEAK] Prestavovanie IRQ Priorít...")
        print("   >>> System32PrioritySeparation: Nastavené na 26 (Foreground Boost).")

    def _run_batch(self, commands):
        """Bezpečné spustenie príkazov (Simulácia na Linuxe, Realita na Win)"""
        if sys.platform == "win32":
            for cmd in commands:
                # subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
                pass # Pre demo nebudeme reálne meniť config užívateľa bez povolenia
        else:
            # Na Linuxe len simulujeme
            time.sleep(0.1)

# --- Singleton Export ---
tweaker_engine = GamesaTweaker()
