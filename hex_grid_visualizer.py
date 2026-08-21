import tkinter as tk
from tkinter import ttk
import random
import threading
import time
import math

class HexGridVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GAMESA: Hex-Grid Operator & Inhibition Simulator")
        self.root.geometry("1000x700")
        self.root.configure(bg="#0f0f15") # Dark Sci-Fi background

        # --- System State ---
        self.voltage = 1.0  # Normalized 0.0 to 1.5
        self.temperature = 40.0 # Celsius
        self.logic_load = 0.0 # 0.0 to 1.0
        self.is_inhibited_voltage = False
        self.is_inhibited_theory = False
        self.dopamine_level = 0
        
        # Hex Grid Data (Simulating 3D addresses flattened)
        self.cells = {} # {id: {'state': 'IDLE', 'addr': '0x7F...', 'val': 0}}
        
        self.create_ui()
        self.start_guardian_thread()

    def create_ui(self):
        # 1. Sidebar (Controls)
        sidebar = tk.Frame(self.root, bg="#1a1a24", width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.add_label(sidebar, "TELEMETRY INPUTS", "#00ffcc", 14)
        
        # Voltage Slider
        self.add_label(sidebar, "Voltage (VRM)", "#ffffff", 10)
        self.volt_slider = tk.Scale(sidebar, from_=0.8, to=1.5, resolution=0.01, orient=tk.HORIZONTAL, bg="#1a1a24", fg="white", command=self.update_telemetry)
        self.volt_slider.set(1.0)
        self.volt_slider.pack(fill=tk.X, padx=10)

        # Temp Slider
        self.add_label(sidebar, "Core Temp (°C)", "#ffffff", 10)
        self.temp_slider = tk.Scale(sidebar, from_=30, to=100, orient=tk.HORIZONTAL, bg="#1a1a24", fg="white", command=self.update_telemetry)
        self.temp_slider.set(45)
        self.temp_slider.pack(fill=tk.X, padx=10)
        
        # Load Slider
        self.add_label(sidebar, "Logic Load (Tasks)", "#ffffff", 10)
        self.load_slider = tk.Scale(sidebar, from_=0, to=100, orient=tk.HORIZONTAL, bg="#1a1a24", fg="white", command=self.update_telemetry)
        self.load_slider.set(20)
        self.load_slider.pack(fill=tk.X, padx=10)

        tk.Frame(sidebar, height=20, bg="#1a1a24").pack() # Spacer

        self.add_label(sidebar, "GUARDIAN STATUS", "#ff00ff", 14)
        
        self.lbl_inhibit_volt = tk.Label(sidebar, text="VOLTAGE INHIBITION: OFF", bg="#1a1a24", fg="#00ff00", font=("Consolas", 10))
        self.lbl_inhibit_volt.pack(pady=5)
        
        self.lbl_inhibit_theory = tk.Label(sidebar, text="THEORETICAL INHIBITION: OFF", bg="#1a1a24", fg="#00ff00", font=("Consolas", 10))
        self.lbl_inhibit_theory.pack(pady=5)

        self.lbl_dopamine = tk.Label(sidebar, text="DOPAMINE: 0", bg="#1a1a24", fg="#ffff00", font=("Consolas", 12, "bold"))
        self.lbl_dopamine.pack(pady=20)
        
        self.log_box = tk.Text(sidebar, height=15, width=30, bg="black", fg="#00ff00", font=("Consolas", 8))
        self.log_box.pack(padx=5, pady=5)
        self.log("System Initialized.")

        # 2. Main Area (The Grid)
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.draw_hex_grid()
        self.canvas.bind("<Button-1>", self.on_grid_click)

    def add_label(self, parent, text, color, size):
        tk.Label(parent, text=text, bg="#1a1a24", fg=color, font=("Arial", size, "bold")).pack(pady=(10, 2), anchor="w", padx=10)

    def draw_hex_grid(self):
        """Draws a visual representation of the Hexadecimal Topology"""
        cols = 10
        rows = 12
        size = 25
        x_offset = 40
        y_offset = 40

        for r in range(rows):
            for c in range(cols):
                x = x_offset + c * size * 1.5
                y = y_offset + r * size * math.sqrt(3) + (size * math.sqrt(3) / 2 if c % 2 else 0)
                
                # Hex Address Simulation
                hex_addr = f"0x7FFF{r:X}{c:X}"
                
                # Draw Hexagon
                points = self.get_hex_points(x, y, size)
                tag = f"cell_{r}_{c}"
                item_id = self.canvas.create_polygon(points, outline="#333333", fill="#111111", width=2, tags=(tag, "hex"))
                
                self.cells[tag] = {
                    'id': item_id,
                    'addr': hex_addr,
                    'state': 'IDLE',
                    'coords': (r, c)
                }

    def get_hex_points(self, x, y, size):
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append(x + size * math.cos(angle_rad))
            points.append(y + size * math.sin(angle_rad))
        return points

    def update_telemetry(self, _=None):
        self.voltage = self.volt_slider.get()
        self.temperature = self.temp_slider.get()
        self.logic_load = self.load_slider.get()

    def guardian_logic_cycle(self):
        """The 'Prefrontal Cortex' loop"""
        while True:
            try:
                # 1. Analyze Telemetry
                # Voltage Inhibition (Hardware Safety)
                if self.voltage > 1.4 or self.temperature > 85:
                    self.is_inhibited_voltage = True
                    self.root.after(0, lambda: self.lbl_inhibit_volt.config(text="VOLTAGE INHIBITION: ACTIVE", fg="red"))
                else:
                    self.is_inhibited_voltage = False
                    self.root.after(0, lambda: self.lbl_inhibit_volt.config(text="VOLTAGE INHIBITION: OFF", fg="#00ff00"))

                # Theoretical Inhibition (Software Optimization)
                # "If logic load is high, inhibit new allocations to preserve bandwidth"
                if self.logic_load > 80:
                    self.is_inhibited_theory = True
                    self.root.after(0, lambda: self.lbl_inhibit_theory.config(text="THEORETICAL INHIBITION: ACTIVE", fg="orange"))
                else:
                    self.is_inhibited_theory = False
                    self.root.after(0, lambda: self.lbl_inhibit_theory.config(text="THEORETICAL INHIBITION: OFF", fg="#00ff00"))

                # 2. Update Grid Visuals based on State
                self.update_grid_visuals()
                
            except Exception as e:
                print(f"Guardian Error: {e}")
            
            time.sleep(0.5) # 2Hz Cycle

    def update_grid_visuals(self):
        for tag, data in self.cells.items():
            color = "#111111" # Default IDLE
            
            if data['state'] == 'ACTIVE':
                color = "#00ccff" # Blue (Processing)
            
            # Inhibition Overrides
            if self.is_inhibited_voltage:
                color = "#330000" # Dark Red (Hardware Lock)
            elif self.is_inhibited_theory and data['state'] == 'IDLE':
                color = "#332200" # Dark Orange (Reserved/Inhibited)

            self.root.after(0, lambda t=tag, c=color: self.canvas.itemconfig(self.cells[t]['id'], fill=c))

    def on_grid_click(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(item)
        
        target_tag = None
        for t in tags:
            if t.startswith("cell_"):
                target_tag = t
                break
        
        if target_tag:
            self.process_request(target_tag)

    def process_request(self, tag):
        """Simulates a computing request to a specific Hex Address"""
        cell = self.cells[tag]
        addr = cell['addr']
        
        # Check Inhibition Rules
        if self.is_inhibited_voltage:
            self.log(f"REQ {addr} -> DENIED (VOLT LIMIT)")
            self.flash_cell(tag, "red")
            return

        if self.is_inhibited_theory:
            # Theoretical inhibition allows ONLY critical tasks (simulated by random chance here)
            if random.random() > 0.8:
                self.log(f"REQ {addr} -> OVERRIDE (CRITICAL)")
            else:
                self.log(f"REQ {addr} -> DENIED (LOGIC PREDICTION)")
                self.flash_cell(tag, "orange")
                return

        # Success - "Light Travel" Data Flow
        self.log(f"REQ {addr} -> SUCCESS (FLOW STATE)")
        self.cells[tag]['state'] = 'ACTIVE'
        self.flash_cell(tag, "#00ffcc")
        self.dopamine_level += 10
        self.lbl_dopamine.config(text=f"DOPAMINE: {self.dopamine_level}")
        
        # Reset after task done
        self.root.after(1000, lambda: self.reset_cell(tag))

    def flash_cell(self, tag, color):
        self.canvas.itemconfig(self.cells[tag]['id'], fill=color)

    def reset_cell(self, tag):
        self.cells[tag]['state'] = 'IDLE'

    def log(self, message):
        self.log_box.insert(tk.END, f"> {message}\n")
        self.log_box.see(tk.END)

    def start_guardian_thread(self):
        t = threading.Thread(target=self.guardian_logic_cycle)
        t.daemon = True
        t.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = HexGridVisualizer(root)
    root.mainloop()
