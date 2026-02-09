#!/usr/bin/env python3
"""
Quadratic Graph Scanner ("The Scanner").
Visualizes the 'Mantinels' and the current Operating Point.
Now grounded in Theory 4: Quadratic Mantinel (Speed = f(Curvature^2)).
"""

import math
import logging
from parameter_standard import Mantinel, MantinelType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SCANNER] - %(message)s')
logger = logging.getLogger(__name__)

class QuadraticScanner:
    def __init__(self, limit_formula: str = None):
        """
        Initialize with a boundary formula or default to Quadratic Mantinel.
        """
        self.formula = limit_formula or "(rpm * feed) < 6000000"
        self.c2_continuity = 1.0 # Default Curvature Continuity

    def scan_space(self, width=40, height=20, max_x=10000, max_y=5000):
        """
        Generates an ASCII heatmap of the Safe Zone.
        """
        grid = []
        x_step = max_x / width
        y_step = max_y / height
        
        # Build Grid (Top to Bottom)
        for row in range(height):
            line = ""
            y_val = max_y - (row * y_step)
            for col in range(width):
                x_val = col * x_step
                
                # Check Mantinel
                is_safe = self._check_safe(x_val, y_val)
                line += "." if is_safe else "#"
            grid.append(line)
        
        return grid

    def _check_safe(self, x, y):
        # Maps x->rpm, y->feed for the eval context
        try:
            # 1. Formula Check
            base_safe = eval(self.formula, {"__builtins__": {}}, {"rpm": x, "feed": y})
            
            # 2. Theory 4: Quadratic Mantinel Guardrail
            # Speed (Feed) = f(Curvature^2). As curvature increases (radius decreases), max speed drops quadratically.
            # curvature = 1/radius. For simplicity, we assume radius is provided or mocked.
            mock_min_radius = 5.0 # mm
            curvature = 1.0 / mock_min_radius
            max_feed_mantinel = 2000 / (curvature ** 2) 
            
            return base_safe and (y <= max_feed_mantinel)
        except:
            return False

    def get_max_safe_speed(self, radius: float) -> float:
        """
        Theory 4: Explicit Speed = f(Curvature^2) calculation.
        """
        curvature = 1.0 / max(0.1, radius)
        # Empirical constant for the VMC-ALPHA machine
        K = 2500.0 
        return K / (curvature ** 2)

    def visualize(self, current_rpm, current_feed):
        """
        Prints the graph with the current point marked.
        """
        w, h = 40, 20
        max_x, max_y = 10000, 2000
        
        print(f"\n[GRAPH SCANNER] Formula: {self.formula}")
        print(f"Current Point: RPM={current_rpm}, Feed={current_feed}")
        
        grid = self.scan_space(w, h, max_x, max_y)
        
        # Calculate Point Position
        px = int((current_rpm / max_x) * w)
        py = int(((max_y - current_feed) / max_y) * h)
        
        # Draw
        print(f"Feed {max_y} ^")
        for r, line in enumerate(grid):
            # Overlay Point
            if r == py:
                line_list = list(line)
                if 0 <= px < len(line_list):
                    line_list[px] = "O" # The Operating Point
                line = "".join(line_list)
            
            print(f"          | {line}")
        print(f"      0   +{'':->40}> RPM {max_x}\n")

# Usage
if __name__ == "__main__":
    # Limit: RPM * Feed < 10,000,000
    scanner = QuadraticScanner("rpm * feed < 6000000")
    
    # Safe Point
    scanner.visualize(3000, 1000) # 3M < 6M (Safe)
    
    # Unsafe Point
    scanner.visualize(8000, 1500) # 12M > 6M (Unsafe - Marked on # region)
