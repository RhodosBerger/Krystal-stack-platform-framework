import math
import time
from typing import Dict, Tuple, Any

class HexTopology:
    """
    LAYER I: Topologická Pamäť.
    Implementuje 3D Grid a Pamäťovú Gravitáciu.
    """
    def __init__(self):
        self.grid: Dict[Tuple[int, int, int], Any] = {}
        
    def write(self, x, y, z, data):
        """Uloží dáta na súradnicu."""
        self.grid[(x, y, z)] = {"data": data, "heat": 0, "last_access": time.time()}
        
    def read_ray(self, start_x, start_y, depth):
        """
        Číta dáta cez časovú os (Z).
        Vracia zoznam dát [Z, Z+1, Z+2...]
        """
        result = []
        for z in range(depth):
            coords = (start_x, start_y, z)
            if coords in self.grid:
                self.grid[coords]["heat"] += 1 # Zohrej bunku
                self.grid[coords]["last_access"] = time.time()
                result.append(self.grid[coords]["data"])
            else:
                result.append(None)
        return result

    def apply_gravity(self):
        """
        Optimalizačný algoritmus: Presúva 'Horúce' dáta bližšie k centru (0,0,0).
        """
        snapshot = list(self.grid.items())
        moves = 0
        for coords, cell in snapshot:
            if cell["heat"] > 5:
                # Bunka je horúca, skús ju posunúť bližšie k 0,0,0
                x, y, z = coords
                new_x = max(0, x - 1)
                new_y = max(0, y - 1)
                
                if (new_x, new_y, z) not in self.grid:
                    del self.grid[coords]
                    self.grid[(new_x, new_y, z)] = cell
                    moves += 1
        return moves
