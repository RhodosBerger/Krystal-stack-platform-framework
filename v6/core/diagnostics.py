import time
import math
import statistics

class LatencyDiagnostics:
    """
    Diagnostický nástroj pre v6.
    Meria Jitter a Latenciu v reálnom čase.
    """
    
    @staticmethod
    def run_jitter_test(duration=2):
        print(f"\n   [DIAG] Spúšťam Jitter Test ({duration}s)...")
        deltas = []
        start = time.perf_counter()
        
        while (time.perf_counter() - start) < duration:
            t1 = time.perf_counter()
            _ = math.sin(t1) # Dummy load
            t2 = time.perf_counter()
            deltas.append((t2 - t1) * 1_000_000)
            
        avg = statistics.mean(deltas)
        jitter = statistics.stdev(deltas)
        
        return avg, jitter

diagnostics = LatencyDiagnostics()
