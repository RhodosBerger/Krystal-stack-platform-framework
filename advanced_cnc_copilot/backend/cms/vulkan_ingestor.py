#!/usr/bin/env python3
"""
Vulkan Log Ingestor
Reads `grid_update` events from `openvino_oneapi_system` runtime logs.
Connects the discrete OVO performance system with the main CNC Copilot Strategic Engine.
"""

import os
import json
import time
import asyncio
import logging
from typing import Optional, Dict, Any


from .message_bus import global_bus


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VULKAN_INGESTOR")

# Configuration
OVO_LOG_REL_PATH = "../../../../../openvino_oneapi_system/logs/runtime_log.jsonl"
LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), OVO_LOG_REL_PATH))

# Global State for external consumers (e.g., AuditorAgent)
grid_state = {
    "saturation": 0.0,
    "last_update": 0.0
}

class VulkanLogIngestor:
    """
    Tails the OVO runtime log and emits VULKAN_GRID_UPDATE events.
    """

    def __init__(self):
        self.running = False
        self.file_pos = 0
        self.bus = global_bus
        logger.info(f"Vulkan Ingestor Initialized. Watching: {LOG_FILE_PATH}")

    async def start(self):
        """Starts the ingestion loop."""
        self.running = True
        logger.info("🌀 Vulkan Ingestion Loop Started")
        
        # Initial seek to end of file to avoid re-playing old history
        if os.path.exists(LOG_FILE_PATH):
            self.file_pos = os.path.getsize(LOG_FILE_PATH)
        
        while self.running:
            await self._read_new_lines()
            await asyncio.sleep(1.0) # Poll every second

    async def stop(self):
        self.running = False
        logger.info("Vulkan Ingestion Loop Stopped")

    async def _read_new_lines(self):
        if not os.path.exists(LOG_FILE_PATH):
            # File might not be created yet if OVO isn't running
            return

        try:
            current_size = os.path.getsize(LOG_FILE_PATH)
            if current_size < self.file_pos:
                # File truncated/rotated
                self.file_pos = 0
            
            if current_size > self.file_pos:
                with open(LOG_FILE_PATH, 'r') as f:
                    f.seek(self.file_pos)
                    lines = f.readlines()
                    self.file_pos = f.tell()
                    
                    for line in lines:
                        await self._process_log_line(line)
        except Exception as e:
            logger.error(f"Error reading log file: {e}")

    async def _process_log_line(self, line: str):
        try:
            entry = json.loads(line)
            event_type = entry.get("event")
            data = entry.get("data", {})

            if event_type == "grid_update":
                # Significant Vulkan/Grid Event
                await self._publish_grid_update(data)
            
            elif event_type == "inference":
                # Inference performance stats
                pass 

        except json.JSONDecodeError:
            pass

    async def _publish_grid_update(self, data: Dict[str, Any]):
        """
        Publishes the grid state to the nervous system.
        """
        capacity = data.get("capacity", 0)
        used = data.get("used_cells", 0)
        saturation = used / capacity if capacity > 0 else 0.0

        payload = {
            "source": "VULKAN_GRID",
            "saturation": saturation,
            "raw_data": data
        }
        
        # Publish to the bus
        # Strategic Engine (Auditor) can subscribe to 'VULKAN_GRID_UPDATE'
        await self.bus.publish("VULKAN_GRID_UPDATE", payload)
        
        if saturation > 0.9:
            logger.warning(f"⚠️ GRID SATURATION CRITICAL: {saturation:.2%}")
            await self.bus.publish("SYSTEM_ALERT", {"level": "CRITICAL", "msg": "Vulkan Grid Memory Full"})
            
        # Update global memory state
        grid_state["saturation"] = saturation
        grid_state["last_update"] = time.time()


if __name__ == "__main__":
    # Test Runner
    ingestor = VulkanLogIngestor()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(ingestor.start())
    except KeyboardInterrupt:
        loop.run_until_complete(ingestor.stop())
