#!/usr/bin/env python3
"""
Signal Repeater 📡
The "Black Box" of the Swarm.
Responsibility: Listen to *everything* and archive it.
"""
import logging
import json
import os
try:
    from cms.message_bus import global_bus, Message
except ImportError:
    from message_bus import global_bus, Message

logger = logging.getLogger("SIGNAL_REPEATER")

class SignalRepeater:
    def __init__(self):
        self.bus = global_bus
        self.log_file = "swarm_blackbox.jsonl"
        logger.info(f"Signal Repeater Online. Archiving to {self.log_file}")

    async def start(self):
        # We subscribe to specific critical channels for now, 
        # as wildcard support might be tricky in our local bus shim unless we implement it.
        # But MessagePubSub does support "*" in Redis. 
        # For local dispatch, let's subscribe to major channels.
        for channel in ["DRAFT_PLAN", "VALIDATION_RESULT", "THERMAL_REPORT", "VOTE_BIOCHEMIST"]:
            self.bus.subscribe(channel, self._archive_event)

    async def _archive_event(self, msg: Message):
        """Persist event to permanent storage (File/DB)"""
        # import aiofiles # Not available in env
        
        entry = {
             "timestamp": msg.timestamp,
             "channel": msg.channel,
             "sender": msg.sender_id,
             "payload": msg.payload
        }
        
        # Simple File Append (Simulating DB Insert)
        # Using sync open for robustness in MVP script
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        logger.info(f"Archived Event: [{msg.channel}] from {msg.sender_id}")

if __name__ == "__main__":
    import asyncio
    async def main():
         repeater = SignalRepeater()
         await repeater.start()
         while True: await asyncio.sleep(1)
    asyncio.run(main())
