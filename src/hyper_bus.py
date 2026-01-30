import queue
import threading
import time
from typing import Dict, List, Callable, Any
from dataclasses import dataclass

@dataclass
class Message:
    topic: str
    payload: Any
    priority: int = 1 # 1=Normal, 2=High, 3=Critical

class HyperStateBus:
    """
    LAYER II: Centrálny Nervový Systém.
    Zabezpečuje, že 'Oko' (Optic) vie povedať 'Mozgu' (Cortex), čo sa deje.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.queue = queue.PriorityQueue()
        self.running = True
        
        # Start Dispatcher
        threading.Thread(target=self._dispatch_loop, daemon=True).start()

    def subscribe(self, topic: str, handler: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)

    def publish(self, topic: str, payload: Any, priority: int = 1):
        # PriorityQueue sorts lowest first, so we negate priority (-3 is processed before -1)
        self.queue.put((-priority, Message(topic, payload, priority)))

    def _dispatch_loop(self):
        while self.running:
            try:
                _, msg = self.queue.get(timeout=0.5)
                if msg.topic in self.subscribers:
                    for handler in self.subscribers[msg.topic]:
                        try:
                            handler(msg.payload)
                        except Exception as e:
                            print(f"[BUS ERROR] Handler failed: {e}")
            except queue.Empty:
                continue
