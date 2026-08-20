import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import os
import random
import math

# Pridanie cesty k v6 modulom
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import engine
from core.diagnostics import diagnostics

# --- KONFIGURÁCIA DIZAJNU (CYBERPUNK THEME) ---
THEME = {
    "bg_main": "#0b0c10",      # Hlboká čierna/modrá
    "bg_sec": "#1f2833",       # Tmavá šedá
    "accent": "#66fcf1",       # Neon Cyan
    "text": "#c5c6c7",         # Svetlá šedá
    "success": "#00ff00",      # Matrix Green
    "warning": "#ffcc00",
    "danger": "#ff0000",
    "font_main": ("Segoe UI", 10),
    "font_mono": ("Consolas", 10),
    "font_header": ("Orbitron", 16, "bold") # Fallback to Arial if missing
}

class GamesaControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GAMESA CONTROL CENTER v6")
        self.geometry("900x600")
        self.configure(bg=THEME["bg_main"])
        self.resizable(False, False)

        # Stav aplikácie
        self.active_profile = "NONE"
        self.is_running = True
        
        self.setup_ui()
        
        # Spustenie monitorovacieho vlákna (Live Stats)
        self.monitor_thread = threading.Thread(target=self.live_monitor_loop, daemon=True)
        self.monitor_thread.start()

    def setup_ui(self):
        # 1. HEADER
        header_frame = tk.Frame(self, bg=THEME["bg_sec"], height=60)
        header_frame.pack(fill=tk.X)
        
        tk.Label(header_frame, text="GAMESA // STRATEGY MULTIPLICATOR", 
                 bg=THEME["bg_sec"], fg=THEME["accent"], font=THEME["font_header"]).pack(pady=15)

        # 2. MAIN CONTAINER (Sidebar + Content)
        container = tk.Frame(self, bg=THEME["bg_main"])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # --- SIDEBAR (Ovládanie) ---
        sidebar = tk.Frame(container, bg=THEME["bg_main"], width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        self.create_profile_btn(sidebar, "ECO FLOW", "1", "#00aa00")
        self.create_profile_btn(sidebar, "BALANCED GRID", "2", "#ffaa00")
        self.create_profile_btn(sidebar, "GAMMA BURST", "3", "#ff0000")
        
        tk.Frame(sidebar, height=30, bg=THEME["bg_main"]).pack() # Spacer
        
        btn_diag = tk.Button(sidebar, text="RUN DIAGNOSTICS", bg=THEME["bg_sec"], fg=THEME["accent"],
                             font=THEME["font_mono"], relief=tk.FLAT, command=self.run_diagnostics)
        btn_diag.pack(fill=tk.X, ipady=10)

        # --- CONTENT (Vizualizácia) ---
        content = tk.Frame(container, bg=THEME["bg_main"])
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Live Graphs (Simulated using Canvas)
        self.stats_canvas = tk.Canvas(content, bg="#000000", height=200, highlightthickness=1, highlightbackground=THEME["accent"])
        self.stats_canvas.pack(fill=tk.X, pady=(0, 20))
        self.overlay_text = self.stats_canvas.create_text(10, 10, text="SYSTEM INITIALIZING...", anchor="nw", fill=THEME["accent"], font=THEME["font_mono"])

        # Console Log
        tk.Label(content, text="KERNEL LOG STREAM:", bg=THEME["bg_main"], fg=THEME["text"], font=THEME["font_mono"]).pack(anchor="w")
        self.log_box = tk.Text(content, bg="#000000", fg="#00ff00", font=THEME["font_mono"], height=10, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def create_profile_btn(self, parent, name, level, color_strip):
        frame = tk.Frame(parent, bg=THEME["bg_sec"], pady=1)
        frame.pack(fill=tk.X, pady=5)
        
        btn = tk.Button(frame, text=name, bg=THEME["bg_sec"], fg="white", font=THEME["font_main"],
                        relief=tk.FLAT, anchor="w", padx=20,
                        command=lambda: self.apply_profile(level))
        btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, ipady=8)
        
        # Color Strip Indicator
        tk.Frame(frame, bg=color_strip, width=5).pack(side=tk.LEFT, fill=tk.Y)

    def log(self, message):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"> {message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def apply_profile(self, level):
        self.log(f"Initiating Profile Change -> Level {level}...")
        
        def worker():
            time.sleep(0.5)
            if level == "1":
                self.log("Applying Energy Saver policies...")
            elif level == "2":
                engine.apply_network_stack_optimization()
                self.log("TCP Stack Optimized.")
                engine.apply_cpu_unpark_logic()
                self.log("Cores Unparked.")
            elif level == "3":
                self.log("!!! WARNING: GAMMA BURST ENGAGED !!!")
                engine.apply_network_stack_optimization()
                engine.apply_cpu_unpark_logic()
                engine.lock_memory_pages()
                self.log("Memory Pages Locked (Hex Grid).")
                engine.visual_entropy_reduction()
                self.log("Visual Entropy Reduced.")
            
            self.active_profile = level
            self.log("OPTIMIZATION COMPLETE.")
            
        threading.Thread(target=worker).start()

    def run_diagnostics(self):
        self.log("Running Jitter/Latency Benchmark...")
        
        def worker():
            avg, jitter = diagnostics.run_jitter_test(duration=3)
            self.log(f"BENCHMARK RESULT:")
            self.log(f"  Avg Latency: {avg:.2f} µs")
            self.log(f"  Jitter:      {jitter:.4f} µs")
            
            if jitter < 1.0:
                self.log("STATUS: EXCELLENT (Gamesa Optimized)")
            else:
                self.log("STATUS: NEEDS OPTIMIZATION")
                
        threading.Thread(target=worker).start()

    def live_monitor_loop(self):
        """Simuluje živé metriky systému (Sine waves pre efekt)"""
        t = 0
        while self.is_running:
            # Generovanie fake dát (v realite psutil)
            cpu_load = 20 + math.sin(t) * 10 + random.randint(0, 5)
            ram_load = 40 + math.cos(t*0.5) * 5
            
            # Kreslenie grafu (Bar Chart style)
            self.stats_canvas.delete("bars")
            
            w = self.stats_canvas.winfo_width()
            bar_w = w * (cpu_load / 100)
            
            # CPU Bar
            color = THEME["accent"] if self.active_profile != "3" else THEME["danger"]
            self.stats_canvas.create_rectangle(10, 50, 10 + bar_w, 80, fill=color, tags="bars")
            self.stats_canvas.create_text(20, 65, text=f"CPU LOAD: {int(cpu_load)}%", anchor="w", fill="white", font=THEME["font_mono"], tags="bars")

            # RAM Bar
            bar_w_ram = w * (ram_load / 100)
            self.stats_canvas.create_rectangle(10, 100, 10 + bar_w_ram, 130, fill="#00ff00", tags="bars")
            self.stats_canvas.create_text(20, 115, text=f"HEX GRID USAGE: {int(ram_load)}%", anchor="w", fill="black", font=THEME["font_mono"], tags="bars")
            
            # Status Text
            status_txt = f"PROFILE: {self.get_profile_name()} | CORE: ONLINE"
            self.stats_canvas.itemconfig(self.overlay_text, text=status_txt)
            
            t += 0.1
            time.sleep(0.1)

    def get_profile_name(self):
        if self.active_profile == "1": return "ECO FLOW"
        if self.active_profile == "2": return "BALANCED"
        if self.active_profile == "3": return "GAMMA BURST"
        return "STANDARD"

if __name__ == "__main__":
    app = GamesaControlPanel()
    app.mainloop()
