import math
import random
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class MemoryCell:
    data: Any
    access_count: int
    last_access: float
    gravity_pull: List[str] # ID iných buniek, ku ktorým je táto bunka priťahovaná

class HolocubeMemoryEngine:
    """
    Motor Virtuálnej 3D Pamäte.
    Spravuje dáta v priestore (X, Y, Z) a simuluje fyziku gravitácie dát.
    """
    def __init__(self, size=16):
        self.size = size # 16x16x16 Grid
        self.grid: Dict[Tuple[int, int, int], MemoryCell] = {}
        self.id_map: Dict[str, Tuple[int, int, int]] = {} # Rýchle vyhľadanie podľa ID

    def materialize(self, data_id: str, data: Any, vector: Tuple[int, int, int]):
        """
        Zapíše dáta na konkrétnu 3D súradnicu. (Materializácia)
        """
        if not self._is_valid(vector):
            print(f"[HOLOCUBE] Error: Vector {vector} out of bounds.")
            return

        cell = MemoryCell(data, access_count=0, last_access=time.time(), gravity_pull=[])
        self.grid[vector] = cell
        self.id_map[data_id] = vector
        print(f"[HOLOCUBE] Materialized '{data_id}' at {vector}")

    def raycast_read(self, start: Tuple[int, int, int], direction: Tuple[int, int, int], depth: int):
        """
        Prečíta sériu dát v línii (napr. cez časovú os Z).
        """
        results = []
        current = list(start)
        
        print(f"[HOLOCUBE] Casting Ray from {start} vector {direction}...")
        
        for _ in range(depth):
            pos = tuple(current)
            if pos in self.grid:
                cell = self.grid[pos]
                cell.access_count += 1
                cell.last_access = time.time()
                results.append(cell.data)
            else:
                results.append(None) # Void
            
            # Posun v 3D
            current[0] += direction[0]
            current[1] += direction[1]
            current[2] += direction[2]
            
        return results

    def apply_gravity_optimization(self):
        """
        Simuluje 'Pritiahnutie' často používaných dát k sebe.
        Toto je ekvivalent Defragmentácie, ale v 3D.
        """
        print("\n[HOLOCUBE] --- Applying Memory Gravity ---")
        moved_count = 0
        
        # Jednoduchá logika: Presuň "Hot" (High Access) dáta smerom k stredu (Y=5, Cache Tier)
        # a k sebe navzájom.
        
        snapshot = list(self.grid.items())
        for pos, cell in snapshot:
            if cell.access_count > 5:
                # Bunka je "Horúca". Skúsme ju posunúť na Y=10 (Vyššia priorita/Tier)
                target_y = 10
                if pos[1] < target_y:
                    new_pos = (pos[0], pos[1] + 1, pos[2])
                    if new_pos not in self.grid and self._is_valid(new_pos):
                        # Move
                        del self.grid[pos]
                        self.grid[new_pos] = cell
                        # Update ID Map (Reverse lookup would need iteration, skipping for demo)
                        moved_count += 1
                        print(f"  > Levitation: Moved HOT cell from {pos} to {new_pos}")

        print(f"[HOLOCUBE] Gravity Cycle Complete. Re-aligned {moved_count} memory blocks.\n")

    def visualize_slice(self, z_index: int):
        """
        Vykreslí 2D rez kockou v ASCII (pre Z=konštanta).
        """
        print(f"--- SLICE Z={z_index} (Time/Depth) ---")
        for y in range(self.size - 1, -1, -1): # Top to Bottom
            row_str = ""
            for x in range(self.size):
                if (x, y, z_index) in self.grid:
                    cell = self.grid[(x, y, z_index)]
                    # Zobraziť 'H' ak je Hot, 'C' ak je Cold
                    char = "H" if cell.access_count > 5 else "O"
                else:
                    char = "."
                row_str += f" {char} "
            print(f"Y={y:02d} |{row_str}|")

    def _is_valid(self, v):
        return 0 <= v[0] < self.size and 0 <= v[1] < self.size and 0 <= v[2] < self.size

# --- Simulácia ---

if __name__ == "__main__":
    holo = HolocubeMemoryEngine(size=12)
    
    # 1. Naplnenie dátami (Objekty v scéne)
    # X=Semantika, Y=Priorita, Z=Čas
    holo.materialize("Texture_Wall", "BINARY_TEX", (5, 2, 0))
    holo.materialize("Physics_Wall", "BINARY_PHY", (5, 2, 1)) # Fyzika je 'budúcnosť' kolízie
    
    # 2. Raycast (Načítanie objektu cez čas)
    # Čítame X=5, Y=2, naprieč Z (0 -> 1)
    data_stream = holo.raycast_read((5,2,0), (0,0,1), depth=3)
    print(f"Raycast Result: {data_stream}")
    
    # 3. Simulácia používania (Zohriatie buniek)
    print("Simulating Heavy Access...")
    for _ in range(10):
        holo.raycast_read((5,2,0), (0,0,1), depth=2)
        
    # 4. Aplikácia Gravitácie (Optimalizácia)
    holo.visualize_slice(0) # Pred
    holo.apply_gravity_optimization()
    holo.visualize_slice(0) # Po (Bunka by sa mala posunúť hore na Y osi)
