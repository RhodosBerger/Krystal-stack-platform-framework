import threading
import time
import queue
import uuid
from typing import Dict, List, Callable, Any
from dataclasses import dataclass

# --- Dátové Štruktúry ---

@dataclass
class StateMessage:
    """Univerzálna správa o stave."""
    source_id: str
    topic: str          # napr. "CPU_LOAD", "THERMAL_WARN", "NEW_AXIOM"
    payload: Any        # Hodnota (číslo, dict, objekt)
    timestamp: float
    priority: int       # 0=Low, 1=High, 2=CRITICAL

# --- Jadro HyperStateBus ---

class HyperStateBus:
    """
    Centrálny Nervový Systém (CNS).
    Umožňuje bleskovú výmenu informácií medzi Guardianom, Gridom a Inventormi.
    """
    _instance = None # Singleton Pattern

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HyperStateBus, cls).__new__(cls)
            cls._instance._init_bus()
        return cls._instance

    def _init_bus(self):
        self.subscribers: Dict[str, List[Callable[[StateMessage], None]]] = {}
        self.message_queue = queue.PriorityQueue() # Prioritná fronta
        self.running = True
        self.history_log = [] # Black Box (Krátkodobá pamäť)
        
        # Spustenie distribučného vlákna
        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatcher_thread.start()
        print("[BUS] HyperStateBus Online. Ready for synaptic connections.")

    def subscribe(self, topic: str, callback: Callable[[StateMessage], None]):
        """Komponent sa prihlási na odber témy."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        print(f"[BUS] New Subscriber for topic: '{topic}'")

    def publish(self, source: str, topic: str, payload: Any, priority: int = 1):
        """Komponent vysiela správu do éteru."""
        msg = StateMessage(
            source_id=source,
            topic=topic,
            payload=payload,
            timestamp=time.time(),
            priority=priority
        )
        # Prioritná fronta radí podľa najnižšieho čísla, tak to otočíme (2=Highest -> -2)
        # Aby kritické správy išli prvé
        self.message_queue.put((-priority, msg)) 

    def _dispatch_loop(self):
        """Nekonečná slučka, ktorá rozosiela správy."""
        while self.running:
            try:
                # Blokujúci get, čaká na správu
                _, msg = self.message_queue.get(timeout=1.0)
                
                # 1. Logovanie do Black Boxu
                self.history_log.append(msg)
                if len(self.history_log) > 1000: self.history_log.pop(0)

                # 2. Distribúcia Subscriberom
                if msg.topic in self.subscribers:
                    for callback in self.subscribers[msg.topic]:
                        # Spustíme callback v try/except, aby jedna chyba nezabila bus
                        try:
                            callback(msg)
                        except Exception as e:
                            print(f"[BUS ERROR] Callback failed: {e}")
                
                # 3. Wildcard Subscribers (počúvajú všetko "*")
                if "*" in self.subscribers:
                    for callback in self.subscribers["*"]:
                        callback(msg)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[BUS FATAL] Dispatcher error: {e}")

# --- Príklad Komponentov (Simulácia) ---

class MockGuardian:
    def __init__(self, bus: HyperStateBus):
        self.bus = bus
        self.bus.subscribe("THERMAL_CRITICAL", self.on_thermal_alarm)
        self.bus.subscribe("SYSTEM_LOAD", self.on_load_change)

    def on_thermal_alarm(self, msg: StateMessage):
        print(f"!!! GUARDIAN ACTIVATED !!! Source: {msg.source_id} | Value: {msg.payload}°C")
        print("    -> ACTION: INHIBIT_VOLTAGE()")

    def on_load_change(self, msg: StateMessage):
        # Guardian len monitoruje, ak je load nízky
        if msg.payload < 20:
            print(f"Guardian: Load is low ({msg.payload}%). Allowing Optimization.")

class MockSensor:
    def __init__(self, bus: HyperStateBus):
        self.bus = bus
        
    def simulate_spike(self):
        print("\n[SENSOR] Detecting Heat Spike...")
        self.bus.publish("SENSOR_CPU", "THERMAL_CRITICAL", 95.5, priority=2)

    def simulate_normal(self):
        print("\n[SENSOR] Normal operation...")
        self.bus.publish("SENSOR_CPU", "SYSTEM_LOAD", 15.0, priority=0)

# --- Spustenie ---

if __name__ == "__main__":
    # 1. Init Bus
    bus = HyperStateBus()
    
    # 2. Init Komponenty
    guardian = MockGuardian(bus)
    sensor = MockSensor(bus)
    
    # 3. Simulácia Deja
    time.sleep(0.5)
    sensor.simulate_normal()
    
    time.sleep(1)
    sensor.simulate_spike() # Kritická správa! 
    
    time.sleep(1)
    print("\n[BUS] History dump (Last 2 messages):")
    for m in bus.history_log[-2:]:
        print(f"  - [{time.ctime(m.timestamp)}] {m.topic}: {m.payload}")
