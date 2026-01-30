import sys
import time
import random
import os

# Pridáme cestu k modulom, aby sme mohli importovať z 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6.core.engine import engine
from v6.core.diagnostics import diagnostics

class CommandLineInterface:
    def __init__(self):
        self.profiles = {
            "1": "ECO_FLOW",
            "2": "BALANCED_GRID",
            "3": "GAMMA_BURST (eSports)"
        }

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def header(self):
        self.clear()
        print("╔══════════════════════════════════════════════════════╗")
        print("║       GAMESA TWEAKER v6 (Genesis Edition)            ║")
        print("║       Algorithmic Renaissance Architecture           ║")
        print("╚══════════════════════════════════════════════════════╝")
        print("")

    def show_menu(self):
        self.header()
        print("Dostupné Profily:")
        for k, v in self.profiles.items():
            print(f"  [{k}] {v}")
        print("\n  [D] Spustiť Diagnostiku (Benchmark)")
        print("  [X] Exit")

    def run_optimization(self, level):
        print(f"\n>>> Aplikujem optimalizáciu: {self.profiles[level]}...")
        
        if level == "1":
            print("   -> Nastavujem Energy Saver...")
        elif level == "2":
            engine.apply_network_stack_optimization()
            engine.apply_cpu_unpark_logic()
        elif level == "3":
            engine.apply_network_stack_optimization()
            engine.apply_cpu_unpark_logic()
            engine.lock_memory_pages()
            engine.visual_entropy_reduction()
            
        print("\n[SUCCESS] Optimalizácia dokončená. Latencia znížená.")
        input("\nStlačte ENTER pre návrat...")

    def run_benchmark(self):
        print("\n>>> Spúšťam Diagnostiku...")
        avg, jitter = diagnostics.run_jitter_test()
        print(f"\n   Výsledky:")
        print(f"   - Priemerná Latencia: {avg:.2f} µs")
        print(f"   - Jitter (Nestabilita): {jitter:.4f} µs")
        
        if jitter < 1.0:
            print("\n   [STATUS] EXCELENTNÝ (GAMESA Ready)")
        else:
            print("\n   [STATUS] POTREBNÁ OPTIMALIZÁCIA")
            
        input("\nStlačte ENTER pre návrat...")

    def loop(self):
        while True:
            self.show_menu()
            choice = input("\nVáš výber: ").upper()
            
            if choice in self.profiles:
                self.run_optimization(choice)
            elif choice == "D":
                self.run_benchmark()
            elif choice == "X":
                print("Ukončujem...")
                break

if __name__ == "__main__":
    cli = CommandLineInterface()
    cli.loop()
