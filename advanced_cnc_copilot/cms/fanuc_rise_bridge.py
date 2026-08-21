#!/usr/bin/env python3
"""
Fanuc Rise Bridge (Mock).
Simulates the "Wave" metrics from a Fanuc Controller via FOCAS.
"""

import asyncio
import random
import logging
from typing import Dict

# Import Message Bus if available
try:
    from message_bus import global_bus
except ImportError:
    global_bus = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [FANUC_RISE] - %(message)s')
logger = logging.getLogger(__name__)

class FanucRiseBridge:
    def __init__(self):
        self.connected = False
        self.stream_active = False

    async def connect(self, ip: str = "127.0.0.1"):
        logger.info(f"Connecting to Fanuc FOCAS at {ip}...")
        await asyncio.sleep(0.5)
        self.connected = True
        logger.info("Connected to Controller.")

    async def stream_wave_metrics(self):
        """
        Simulates high-speed sampling of Servo Load and Current.
        """
        self.stream_active = True
        logger.info("Starting Wave Metric Stream (Servo Load, Current)...")
        
        while self.stream_active:
            # Simulate Data
            # Normal operation with occasional "Spikes"
            
            # 1. Servo Load (Following Error)
            # Normal: 0.01 - 0.05. Dragging: > 0.1
            servo_load = random.uniform(0.01, 0.06)
            if random.random() > 0.95: 
                servo_load += 0.1 # SPIKE
            
            # 2. Spindle Current (Harmonics)
            # Base 10A + Noise.
            current = 10.0 + random.normalvariate(0, 0.5)
            
            metrics = {
                "servo_load_error": servo_load,
                "spindle_current_amp": current,
                "status": "ACTIVE"
            }
            
            # Publish if Bus exists
            if global_bus:
                await global_bus.publish("FANUC_WAVE_METRICS", metrics, sender_id="FANUC_RISE")
            else:
                # Local debug
                # only print spikes to avoid spam
                if servo_load > 0.1:
                    logger.warning(f"DETECTED WAVE SPIKE: Load={servo_load:.3f}")
            
            await asyncio.sleep(0.5) # 2Hz sample rate for mock

    def stop(self):
        self.stream_active = False

# Usage
if __name__ == "__main__":
    async def test():
        bridge = FanucRiseBridge()
        await bridge.connect()
        # Run for 3 seconds
        task = asyncio.create_task(bridge.stream_wave_metrics())
        await asyncio.sleep(3)
        bridge.stop()
        await task

    asyncio.run(test())
