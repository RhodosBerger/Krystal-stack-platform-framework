import tkinter as tk
import time
import threading
import random
import math
import sys
import os

# Ensure we can import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.diagnostics import diagnostics

class HolographicHUD(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # 1. Window Configuration (Transparent & Topmost)
        self.title("GAMESA HUD")
        self.geometry("300x120+50+50") # Top-Left corner
        self.overrideredirect(True) # Remove borders
        self.attributes("-topmost", True) # Always on top
        self.attributes("-alpha", 0.7) # Transparency
        self.configure(bg="black")
        
        # Make it click-through (Windows specific)
        try:
            self.make_click_through()
        except Exception as e:
            print(f"Click-through overlay not supported: {e}")

        # 2. Canvas for Drawing
        self.canvas = tk.Canvas(self, width=300, height=120, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        self.running = True
        self.start_monitoring()

    def make_click_through(self):
        """
        Uses Windows API to make the window transparent to mouse clicks.
        """
        import ctypes
        from ctypes import wintypes
        
        hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
        gwl_exstyle = -20
        ws_ex_layer = 0x80000
        ws_ex_transparent = 0x20
        
        old_style = ctypes.windll.user32.GetWindowLongA(hwnd, gwl_exstyle)
        ctypes.windll.user32.SetWindowLongA(hwnd, gwl_exstyle, old_style | ws_ex_layer | ws_ex_transparent)

    def draw_hex_pulse(self, load_level):
        """Draws a sci-fi hex pulse that reacts to system load"""
        self.canvas.delete("all")
        
        # Color Logic: Cyan (Low) -> Orange (Med) -> Red (High)
        color = "#66fcf1"
        if load_level > 50: color = "#ffcc00"
        if load_level > 80: color = "#ff0000"
        
        # Draw Hexagon
        cx, cy = 50, 60
        size = 30 + (math.sin(time.time() * 5) * 2) # Pulsing size
        
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            x = cx + size * math.cos(angle_rad)
            y = cy + size * math.sin(angle_rad)
            points.extend([x, y])
            
        self.canvas.create_polygon(points, outline=color, fill="", width=2)
        
        # Inner Text
        self.canvas.create_text(cx, cy, text=f"{int(load_level)}%", fill="white", font=("Consolas", 10, "bold"))
        self.canvas.create_text(cx, cy+40, text="GRID LOAD", fill="#888", font=("Arial", 8))

        # Jitter Graph
        self.draw_mini_graph(120, 30)

    def draw_mini_graph(self, x_offset, y_offset):
        # Simulated Jitter Line
        points = []
        for i in range(20):
            x = x_offset + (i * 8)
            noise = random.randint(0, 15)
            y = y_offset + 30 - noise
            points.extend([x, y])
            
        self.canvas.create_line(points, fill="#00ff00", width=1)
        self.canvas.create_text(x_offset, y_offset-10, text="LATENCY STREAM", anchor="nw", fill="#00ff00", font=("Arial", 8))

    def start_monitoring(self):
        def loop():
            t = 0
            while self.running:
                # Simulated Load
                load = 40 + math.sin(t) * 20 + random.randint(0, 10)
                
                self.after(0, lambda: self.draw_hex_pulse(load))
                
                t += 0.1
                time.sleep(0.05) # 20 FPS Update
                
        threading.Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    app = HolographicHUD()
    app.mainloop()
