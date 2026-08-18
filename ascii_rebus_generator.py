import time
import random
import sys
import threading

class AsciiRebusLogger:
    def __init__(self):
        self.running = True
        # Symbolika pre "Deje"
        self.symbols = {
            "PREDICTION": "Ψ",    # Grécke Psí (Psyché/Myseľ/Predikcia)
            "REALITY": "R",       # Realita
            "SHADOW": "░",        # Tieňová pamäť
            "BARRIER": "║",       # Inhibícia/Hranica
            "FLOW": "≈",          # Light Travel/Tok
            "SPIKE": "⚡",         # Voltage Injection
            "LOCK": "Ω",          # Ohm/Odpor/Zámok
            "GRID": "::"          # Hex Mriežka
        }
        
    def generate_timeline_line(self):
        """
        Generuje jeden riadok 'príbehu' (Log).
        Vľavo je minulosť, v strede prítomnosť, vpravo tieňová budúcnosť.
        """
        # 1. Stav Systému (Náhodná simulácia)
        load = random.random()
        prediction_confidence = random.random()
        
        # 2. Vytvorenie Rébusu
        
        # Časť 1: Minulosť (Memory Grid)
        # Ak bola záťaž vysoká, mriežka je "hustá" (##), inak riedka (::)
        past_block = "[####]" if random.random() > 0.7 else "[::::]"
        
        # Časť 2: Bariéra (Guardian)
        # Ak je predikcia neistá, Guardian stavia bariéru
        barrier = self.symbols["BARRIER"] if prediction_confidence < 0.4 else self.symbols["FLOW"]
        
        # Časť 3: Jadro (Procesor)
        # Ak prichádza SPIKE, zobrazí blesk
        core_action = self.symbols["SPIKE"] if load > 0.8 else self.symbols["REALITY"]
        
        # Časť 4: Budúcnosť (Shadow Buffer)
        # Psí (Predikcia) vidí do tieňa (░)
        future_sight = f"{self.symbols['PREDICTION']}->{self.symbols['SHADOW']}"
        
        # 3. Zloženie vety
        # Formát: [PAMÄŤ] --(TOK)--> [JADRO] --(VIDENIE)--> [BUDÚCNOSŤ]
        log_line = f"{past_block} {barrier*3} [{core_action}] {barrier*3} {future_sight}"
        
        # Pridanie "Vibrácie" (Hex dáta)
        hex_noise = f"0x{random.randint(0, 255):02X}"
        
        return f"{time.strftime('%H:%M:%S')} | {hex_noise} | {log_line}"

    def run(self):
        print("--- INICIALIZÁCIA KRYPTOGRAFICKÉHO LOGU ---")
        print("Legenda: Ψ=Predikcia, ░=Tieň, ⚡=Excitácia, Ω=Inhibícia\n")
        time.sleep(1)
        
        try:
            while self.running:
                line = self.generate_timeline_line()
                # Efekt písacieho stroja
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
                
                # Rýchlosť logovania závisí od "záťaže"
                time.sleep(random.uniform(0.1, 0.5))
        except KeyboardInterrupt:
            print("\n--- LOG UKONČENÝ ---")

if __name__ == "__main__":
    logger = AsciiRebusLogger()
    logger.run()
