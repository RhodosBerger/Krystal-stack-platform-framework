import requests
import time
import random
import threading

API_URL = "http://localhost:8000/api/swarm/broadcast"
NODES = 50

def simulate_node(node_id):
    print(f"📡 Node {node_id} online")
    while True:
        try:
            # Simulate dynamic machine behavior
            load = random.uniform(10, 100)
            rpm = random.randint(5000, 12000)
            vibration = random.uniform(0.01, 0.5)
            
            payload = {
                "status": "ONLINE",
                "load": load,
                "rpm": rpm,
                "vibration": vibration,
                "action": "SWARM_SYNC" if load < 80 else "TASK_STEALING_ACTIVE",
                "neuro_state": {
                    "dopamine": random.uniform(40, 90),
                    "cortisol": random.uniform(10, 80),
                    "serotonin": random.uniform(30, 70)
                }
            }
            
            requests.post(f"{API_URL}?machine_id=NODE-{node_id}", json=payload, timeout=1)
            
        except Exception as e:
            # print(f"Node {node_id} error: {e}")
            pass
            
        time.sleep(random.uniform(1.0, 3.0))

if __name__ == "__main__":
    print(f"🚀 Launching Global Swarm Simulation ({NODES} nodes)...")
    threads = []
    for i in range(1, NODES + 1):
        t = threading.Thread(target=simulate_node, args=(f"{i:03d}",))
        t.daemon = True
        t.start()
        threads.append(t)
        time.sleep(0.1) # Stagger start
        
    print("🌟 Swarm is now BROADCASTING. Check React/Vue dashboards for visualization.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Swarm...")
