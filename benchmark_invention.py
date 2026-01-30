import time
import math
import statistics

class LatencyBenchmark:
    """
    Nástroj na dôkaz výkonu. Meria 'Jitter' (nepravidelnosť) v systémovej slučke.
    Menej je lepšie.
    """
    
    def run_stress_test(self, duration_sec=3):
        print(f"\n--- SPUSTENIE LATENCY BENCHMARKU ({duration_sec}s) ---")
        print("Meriam mikrosekundové odchýlky (Jitter)...")
        
        deltas = []
        start_time = time.perf_counter()
        
        # Tight Loop
        while (time.perf_counter() - start_time) < duration_sec:
            t1 = time.perf_counter()
            # Simulácia mikrozáťaže (CPU operácia)
            _ = math.sin(t1) * math.sqrt(t1)
            t2 = time.perf_counter()
            deltas.append((t2 - t1) * 1_000_000) # Prevod na mikrosekundy
            
        avg_lat = statistics.mean(deltas)
        jitter = statistics.stdev(deltas)
        
        print("-" * 40)
        print(f"Priemerná Latencia Slučky: {avg_lat:.4f} µs")
        print(f"Jitter (Nestabilita):      {jitter:.4f} µs")
        print("-" * 40)
        
        if jitter < 0.5:
            print("VÝSLEDOK: EXCELENTNÝ (GAMESA Optimized)")
        elif jitter < 2.0:
            print("VÝSLEDOK: DOBRÝ (Standard)")
        else:
            print("VÝSLEDOK: ZLÝ (Potrebná optimalizácia)")

if __name__ == "__main__":
    bench = LatencyBenchmark()
    bench.run_stress_test()