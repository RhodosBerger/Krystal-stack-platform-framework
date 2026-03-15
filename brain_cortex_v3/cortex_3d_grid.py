#!/usr/bin/env python3
"""
Brain Cortex V3 — Neuro-Visual 3D Engine
--------------------------------------------------
Obsahuje:
1. 3D Tenzor s ročnou iteráciou (Vulkan, RAM, Thermal).
2. ASCII DLSS (Dynamic Local Subpixel Scaling) - Trilineárna interpolácia
   hrubého gridu do vysokého rozlíšenia pre ušetrenie CPU.
3. DOPAMINE REWARD SYSTEM - Bot dostáva odmeny za riešenie hotspotov,
   čo mení jeho vizuálnu auru (farbu) a rýchlosť.
"""

import sys
import math
import time
import random

# Rozlíšenie plátna v subpixeloch
WIDTH = 140
HEIGHT = 110

class AsciiDLSS:
    """ASCII Deep Learning Super Sampling - Interpolácia hrubého tenzoru"""
    @staticmethod
    def upscale(tensor_in, scale_factor=2):
        si_x = len(tensor_in)
        si_y = len(tensor_in[0])
        si_z = len(tensor_in[0][0])
        
        so_x, so_y, so_z = si_x * scale_factor, si_y * scale_factor, si_z * scale_factor
        tensor_out = [[[0.0 for _ in range(so_z)] for _ in range(so_y)] for _ in range(so_x)]
        
        for x in range(so_x):
            for y in range(so_y):
                for z in range(so_z):
                    # Zjednodušená trilineárna interpolácia (Nearest-neighbor smooth)
                    ix = x / scale_factor
                    iy = y / scale_factor
                    iz = z / scale_factor
                    
                    x0, y0, z0 = int(ix), int(iy), int(iz)
                    
                    # Anti-aliasing / smoothing factor
                    fx, fy, fz = ix - x0, iy - y0, iz - z0
                    
                    val = tensor_in[x0][y0][z0]
                    # Pridáme mikro-šum reprezentujúci "AI upscaling" detail
                    detail_noise = random.uniform(0, 0.05) if val > 0.1 else 0
                    
                    tensor_out[x][y][z] = min(1.0, val + detail_noise)
                    
        return tensor_out

class CortexBotAgent:
    """Agent s vlastnou motiváciou (Dopamín) a skóre"""
    def __init__(self):
        self.dopamine = 0.5    # 0.0 (Depresia/Fail) -> 1.0 (Euforia/Success)
        self.score = 0
        self.last_action = "OBSERVING"
        self.aura_color = (139, 92, 246) # Purple base
        
    def evaluate_grid(self, tensor):
        """Vyhodnotí sieť a nájde hotspoty"""
        max_heat = 0.0
        for x in tensor:
            for y in x:
                for z in y:
                    if z > max_heat: max_heat = z
                    
        # Motivácia: Ak stúpa teplo, musí zasiahnuť
        if max_heat > 0.8:
            action = self.take_action("Thermal Spike Detected -> INJECT_COOLING")
            return "COOLING"
        elif max_heat < 0.3 and self.dopamine > 0.6:
            action = self.take_action("Low Load + High Dopamine -> BOOST_GPU_FREQ")
            return "BOOST"
        else:
            self.dopamine = max(0.1, self.dopamine - 0.01) # Nuda jemne znižuje dopamín
            return "IDLE"

    def take_action(self, reason):
        self.last_action = reason
        # Úspešná akcia dáva dopamín (Score)
        reward = random.uniform(0.1, 0.3)
        self.dopamine = min(1.0, self.dopamine + reward)
        self.score += int(reward * 100)
        return True

class NeuroEngine3D:
    def __init__(self, base_size=5):
        self.base_size = base_size
        self.tensor = [[[0.0 for _ in range(base_size)] for _ in range(base_size)] for _ in range(base_size)]
        
        self.braille_base = 0x2800
        self.pixel_map = [
            [0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]
        ]
        self.agent = CortexBotAgent()
        self.dlss = AsciiDLSS()
        self.target_heat = 0.5

    def update_hardware_sim(self, time_t):
        """Simulácia hrubého (Base) hardvérového stavu"""
        action = self.agent.evaluate_grid(self.tensor)
        
        # Reakcia na akciu bota
        if action == "COOLING":
            self.target_heat = max(0.2, self.target_heat - 0.4)
        elif action == "BOOST":
            self.target_heat = min(1.0, self.target_heat + 0.3)
        else:
            # Náhodné fluktuácie z OS
            self.target_heat += random.uniform(-0.1, 0.1)
            self.target_heat = max(0.1, min(0.9, self.target_heat))

        # Naplnenie base tenzora
        for x in range(self.base_size):
            for y in range(self.base_size):
                for z in range(self.base_size):
                    dist = math.sqrt((x-self.base_size/2)**2 + (y-self.base_size/2)**2 + (z-self.base_size/2)**2)
                    wave = math.sin(dist - time_t * (1.0 + self.agent.dopamine)) # Rýchlejšie pulzovanie pri vysokom dopamíne
                    val = max(0, wave * self.target_heat)
                    self.tensor[x][y][z] = val

    def project_3d_to_2d(self, x, y, z, angle_z, angle_x):
        x_r = x * math.cos(angle_z) - y * math.sin(angle_z)
        y_r = x * math.sin(angle_z) + y * math.cos(angle_z)
        y_r2 = y_r * math.cos(angle_x) - z * math.sin(angle_x)
        z_r2 = y_r * math.sin(angle_x) + z * math.cos(angle_x)
        
        scale = 5.0
        sx = int(WIDTH / 2 + x_r * scale)
        sy = int(HEIGHT / 2 + (y_r2 - z_r2 * 1.5) * scale * 0.5) 
        return sx, sy, z_r2

    def render_frame(self, time_t):
        # 1. ASCII DLSS Upscaling (5x5x5 -> 10x10x10 pre vysokú vizuálnu čistotu)
        dlss_tensor = self.dlss.upscale(self.tensor, scale_factor=2)
        up_size = len(dlss_tensor)
        
        sub_pixels = {}
        
        # Kamera sa točí podľa toho, akú má bot motiváciu (Dopamine level)
        cam_yaw = time_t * (0.2 + self.agent.dopamine * 0.8)
        cam_pitch = 0.6
        
        for z in range(up_size):
            for y in range(up_size):
                for x in range(up_size):
                    val = dlss_tensor[x][y][z]
                    if val > 0.05:
                        sx, sy, depth = self.project_3d_to_2d(
                            x - up_size/2, y - up_size/2, z - up_size/2, 
                            cam_yaw, cam_pitch
                        )
                        
                        intensity = val * (0.5 + (z / up_size) * 0.5)
                        
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0]:
                                px, py = sx + dx, sy + dy
                                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                                    curr = sub_pixels.get((px, py), 0)
                                    sub_pixels[(px, py)] = max(curr, intensity)

        out = "\033[H\033[2J"
        out += "\033[38;2;16;185;129m\033[1m" + " NVIDIA ASCII DLSS + DOPAMINE AGENT ENGINE ".center(WIDTH//2, "═") + "\033[0m\n\n"
        
        char_rows = HEIGHT // 4
        char_cols = WIDTH // 2
        
        for cy in range(char_rows):
            row_str = "    "
            for cx in range(char_cols):
                char_val = 0
                max_int = 0
                for dy in range(4):
                    for dx in range(2):
                        px, py = cx * 2 + dx, cy * 4 + dy
                        if (px, py) in sub_pixels:
                            char_val |= self.pixel_map[dy][dx]
                            if sub_pixels[(px, py)] > max_int: max_int = sub_pixels[(px, py)]
                
                if char_val == 0:
                    row_str += " "
                else:
                    # Dopamine ovplyvňuje farby
                    if self.agent.dopamine > 0.8:
                        # Euphoric Gold / White
                        r, g, b = int(255*max_int), int(215*max_int), int(0*max_int)
                        if max_int > 0.8: r, g, b = 255, 255, 255
                    else:
                        if max_int < 0.3:  r, g, b = 6, 182, 212
                        elif max_int < 0.6: r, g, b = 59, 130, 246
                        elif max_int < 0.8: r, g, b = 139, 92, 246
                        else:               r, g, b = 239, 68, 68
                        
                    ch = chr(self.braille_base + char_val)
                    row_str += f"\033[38;2;{r};{g};{b}m{ch}\033[0m"
            out += row_str + "\n"
            
        # UI pre Agentov Dopamín System
        dp_bar_len = 20
        dp_fill = int(self.agent.dopamine * dp_bar_len)
        dp_bar = "█" * dp_fill + "░" * (dp_bar_len - dp_fill)
        dp_color = "\033[38;2;250;204;21m" if self.agent.dopamine > 0.7 else "\033[38;2;167;139;250m"
        
        ui = f" BASE GRID: {self.base_size}^3 | DLSS UPSCALED: {up_size}^3 | PX: {WIDTH}x{HEIGHT}\n"
        ui += f" BOT SCORE: \033[1m{self.agent.score:06d}\033[0m | DOPAMINE: {dp_color}{dp_bar} ({self.agent.dopamine*100:.0f}%)\033[0m\n"
        ui += f" LAST ACTION: \033[38;2;239;68;68m{self.agent.last_action}\033[0m"
        
        out += "\n" + ui.center(WIDTH//2) + "\n"
        sys.stdout.write(out)
        sys.stdout.flush()

if __name__ == "__main__":
    print("\033[?25l")
    engine = NeuroEngine3D(base_size=6)
    try:
        t = 0
        while True:
            engine.update_hardware_sim(t)
            engine.render_frame(t)
            t += 0.1
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\033[?25h")
        sys.exit(0)
