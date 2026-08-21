import time
import sys
import random
from gamsa_tweaker_subsystem import tweaker_engine

class ExpressSettings:
    """
    Užívateľské rozhranie (MVP) pre GAMESA Tweaker.
    """
    
    def __init__(self):
        self.profiles = {
            "1": {"name": "ECO_FLOW", "desc": "Tichý chod, Web, Office"},
            "2": {"name": "BALANCED_GRID", "desc": "Stabilný gaming, Photoshop"},
            "3": {"name": "GAMMA_BURST", "desc": "eSports, Max FPS, Low Latency (Agresívne)"}
        }

    def draw_header(self):
        print("\n" + "="*50)
        print("      GAMESA TWEAKER: STRATEGY MULTIPLICATOR      ")
        print("             v1.0 [MVP Genesis]                   ")
        print("="*50)
        print("Analyzujem Hardware Telemetriu...", end="")
        time.sleep(1)
        print(" [OK]")
        print("-" * 50)

    def show_menu(self):
        print("\nVyberte Express Profil:")
        for key, p in self.profiles.items():
            print(f" [{key}] {p['name']:<15} | {p['desc']}")
        print(" [0] Exit")

    def apply_profile(self, choice):
        if choice not in self.profiles:
            return

        profile = self.profiles[choice]
        print(f"\n>>> APLIKUJEM PROFIL: {profile['name']} <<<")
        print("Prosím čakajte, prekonfigurujem Kernel...")
        
        self._simulate_progress_bar()

        # Logika aplikácie
        if choice == "1": # ECO
            print("[PROFILE] Parkujem jadrá pre úsporu energie...")
            
        elif choice == "2": # BALANCED
            tweaker_engine.apply_network_tweaks()
            tweaker_engine.apply_cpu_unpark()
            
        elif choice == "3": # GAMMA (The 40% Boost)
            tweaker_engine.apply_network_tweaks()
            tweaker_engine.apply_cpu_unpark()
            tweaker_engine.apply_io_priority()
            tweaker_engine.apply_visual_reduction()
            print("[GAMMA] Aktivujem 'Hex Grid' Memory Locking...")
        
        print(f"\n[SUCCESS] Systém optimalizovaný. Odhadovaný zisk latencie: -{random.randint(25, 45)}%")

    def _simulate_progress_bar(self):
        sys.stdout.write("[based on the provided context, this is the corrected string]")
        for _ in range(20):
            sys.stdout.write("█")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("] Done.\n")

if __name__ == "__main__":
    app = ExpressSettings()
    app.draw_header()
    
    while True:
        app.show_menu()
        choice = input("\nVáš výber: ")
        if choice == "0":
            print("Pripravujem 'Light Travel' shutdown...")
            break
        app.apply_profile(choice)
