import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import random
import math
from collections import deque

# Import Engines
try:
    from equation_grid_engine import MathematicalGridEngine
    from predictive_mirror_framework import TemporalPredictor, ResourcePreloader
except ImportError:
    MathematicalGridEngine = None
    TemporalPredictor = None

class PredictiveDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KrystalStack: Predictive Mirror Interface")
        self.geometry("1200x800")
        self.configure(bg="#0b0c10") # Deep Sci-Fi Blue/Black

        # Components
        self.engine = MathematicalGridEngine(50, 50) if MathematicalGridEngine else None
        self.predictor = TemporalPredictor(history_size=30) if TemporalPredictor else None
        self.preloader = ResourcePreloader() if TemporalPredictor else None
        
        # Simulation State
        self.sim_time = 0.0
        self.load_history = deque(maxlen=100) # For drawing history graph
        
        self.create_ui()
        self.start_simulation_loop()

    def create_ui(self):
        # 1. Top Header
        header = tk.Frame(self, bg="#1f2833", height=60)
        header.pack(fill=tk.X)
        tk.Label(header, text="PREDICTIVE MIRROR // TEMPORAL PARALLAX", 
                 fg="#66fcf1", bg="#1f2833", font=("Orbitron", 16, "bold")).pack(pady=15)

        # 2. Main Content Split
        content = tk.Frame(self, bg="#0b0c10")
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel: The Grid (Present Reality)
        self.left_panel = tk.Frame(content, bg="#0b0c10", width=500)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(self.left_panel, text="PRESENT STATE [T=0]", fg="white", bg="#0b0c10").pack(anchor="w")
        self.canvas_grid = tk.Canvas(self.left_panel, bg="black", height=400, highlightthickness=1, highlightbackground="#45a29e")
        self.canvas_grid.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right Panel: The Prediction (Future Graph)
        self.right_panel = tk.Frame(content, bg="#0b0c10", width=600)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        tk.Label(self.right_panel, text="TEMPORAL HORIZON [T+10]", fg="#45a29e", bg="#0b0c10").pack(anchor="w")
        self.canvas_pred = tk.Canvas(self.right_panel, bg="#101015", height=400, highlightthickness=1, highlightbackground="#45a29e")
        self.canvas_pred.pack(fill=tk.BOTH, expand=True, pady=5)

        # 3. Bottom Panel: Metrics & Actions
        bottom = tk.Frame(self, bg="#1f2833", height=150)
        bottom.pack(fill=tk.X)
        
        self.lbl_action = tk.Label(bottom, text="SYSTEM ACTION: IDLE", fg="#c5c6c7", bg="#1f2833", font=("Consolas", 12))
        self.lbl_action.pack(pady=10)
        
        self.lbl_stats = tk.Label(bottom, text="CONFIDENCE: 100% | SHADOW BUFFER: EMPTY", fg="#66fcf1", bg="#1f2833", font=("Arial", 10))
        self.lbl_stats.pack(pady=5)

    def draw_prediction_graph(self, predictions):
        """
        Draws the history (Solid line) and Prediction (Dashed Cone).
        """
        self.canvas_pred.delete("all")
        w = self.canvas_pred.winfo_width()
        h = self.canvas_pred.winfo_height()
        if w < 10 or h < 10: return

        # Draw History (Past)
        if len(self.load_history) > 1:
            points = []
            scale_x = (w * 0.6) / 100 # History takes 60% of width
            for i, val in enumerate(self.load_history):
                x = i * scale_x
                y = h - (val * (h/100)) # Normalize 0-100 load to height
                points.extend([x, y])
            self.canvas_pred.create_line(points, fill="#66fcf1", width=2, smooth=True)

        # Draw Prediction (Future)
        if predictions:
            start_x = (len(self.load_history) - 1) * ((w * 0.6) / 100)
            start_y = h - (self.load_history[-1] * (h/100))
            
            scale_x_future = (w * 0.4) / len(predictions) # Future takes 40%
            
            last_x, last_y = start_x, start_y
            
            for i, pt in enumerate(predictions):
                # Calculate coords
                x = start_x + ((i + 1) * scale_x_future)
                y = h - (pt.value * (h/100))
                
                # Color based on confidence (Green=High, Red=Low)
                conf_hex = int(pt.confidence * 255)
                color = f"#{255-conf_hex:02x}{conf_hex:02x}00"
                
                # Draw Line Segment
                self.canvas_pred.create_line(last_x, last_y, x, y, fill=color, width=2, dash=(4, 2))
                
                # Draw "Cone of Uncertainty" (Vertical bars)
                uncertainty = (1.0 - pt.confidence) * 50
                self.canvas_pred.create_line(x, y - uncertainty, x, y + uncertainty, fill="#222222", width=1)
                
                last_x, last_y = x, y

            # Draw "Threshold" Line (e.g., 80% Load)
            self.canvas_pred.create_line(0, h*0.2, w, h*0.2, fill="red", dash=(2, 4))
            self.canvas_pred.create_text(10, h*0.2 - 10, text="OVERLOAD THRESHOLD", fill="red", anchor="w")

    def update_grid(self, load_val):
        """Updates the visual grid based on load"""
        self.canvas_grid.delete("all")
        if not self.engine: return
        
        # If load is high, the "Sphere" pulses faster
        t = self.sim_time
        radius = 15 + math.sin(t) * (load_val * 0.2)
        
        self.engine.objects = []
        self.engine.add_equation_sphere("Load_Orb", 25, 25, radius)
        
        frame, _ = self.engine.compute_frame_with_parallax({})
        
        # Render Frame (Simplified for performance in this view)
        w = self.canvas_grid.winfo_width()
        h = self.canvas_grid.winfo_height()
        cw, ch = w/50, h/50
        
        for y, row in enumerate(frame):
            for x, val in enumerate(row):
                if val > 0.1:
                    intensity = int(val * 255)
                    # Color shifts based on Load: Blue (Low) -> Purple (High)
                    r = int((load_val/100) * 255)
                    g = 0
                    b = 255 - r
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    self.canvas_grid.create_rectangle(x*cw, y*ch, (x+1)*cw, (y+1)*ch, fill=color, outline="")

    def start_simulation_loop(self):
        def loop():
            while True:
                self.sim_time += 0.2
                
                # 1. Generate Simulated Load (Sine wave + Random noise)
                # Simulates a user playing a game with varying intensity
                base_load = 50 + math.sin(self.sim_time * 0.5) * 30 
                noise = random.uniform(-10, 10)
                current_load = max(0, min(100, base_load + noise))
                
                self.load_history.append(current_load)
                
                # 2. Feed Predictor
                if self.predictor:
                    self.predictor.input_data(current_load)
                    predictions = self.predictor.predict_horizon(steps=15)
                    
                    # 3. Check Preloader Logic
                    actions = self.preloader.check_and_preload(predictions)
                    
                    # 4. Update UI
                    self.after(0, lambda: self.draw_prediction_graph(predictions))
                    self.after(0, lambda: self.update_ui_labels(actions, predictions))
                
                self.after(0, lambda: self.update_grid(current_load))
                
                time.sleep(0.05) # 20 FPS
        
        t = threading.Thread(target=loop)
        t.daemon = True
        t.start()

    def update_ui_labels(self, actions, predictions):
        if actions:
            self.lbl_action.config(text=f"SYSTEM ACTION: {', '.join(actions)}", fg="#ff00ff")
        else:
            self.lbl_action.config(text="SYSTEM ACTION: MONITORING", fg="#c5c6c7")
            
        if predictions:
            avg_conf = sum(p.confidence for p in predictions) / len(predictions)
            self.lbl_stats.config(text=f"CONFIDENCE: {avg_conf:.0%} | SHADOW BUFFER: {'ACTIVE' if actions else 'IDLE'}")

if __name__ == "__main__":
    app = PredictiveDashboard()
    app.mainloop()