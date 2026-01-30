#!/usr/bin/env python3
"""
The Nervous System: Asynchronous Message Bus
Allows the 'Shadow Council' members to communicate without blocking.
"""

import asyncio
import logging
import uuid
from typing import Callable, Dict, List, Any, Awaitable
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BUS] - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """A standard message envelope."""
    channel: str
    payload: Any
    sender_id: str
    message_id: str
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = datetime.now().timestamp()

class MessageBus:
    """
    Publish/Subscribe Event Bus.
    """
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Message], Awaitable[None]]]] = {}
        self.bus_id = f"BUS_{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"Message Bus Initialized: {self.bus_id}")

    def subscribe(self, channel: str, callback: Callable[[Message], Awaitable[None]]):
        """Register a callback for a specific channel."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)
        logger.info(f"New Subscriber on channel '{channel}'")

    async def publish(self, channel: str, payload: Any, sender_id: str):
        """Publish a message to all subscribers of a channel."""
        if channel not in self._subscribers:
            # logger.debug(f"No subscribers for channel '{channel}'")
            return

        msg = Message(
            channel=channel,
            payload=payload,
            sender_id=sender_id,
            message_id=f"MSG_{uuid.uuid4().hex[:8].upper()}"
        )

        logger.info(f"Event: [{channel}] from {sender_id}")
        
        # Notify all subscribers concurrently
        tasks = []
        for callback in self._subscribers[channel]:
            tasks.append(callback(msg))
        
        if tasks:
            await asyncio.gather(*tasks)

# Global Bus Instance
global_bus = MessageBus()
