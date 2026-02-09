#!/usr/bin/env python3
"""
The Nervous System: Asynchronous Message Bus
Allows the 'Shadow Council' members to communicate without blocking.
Now grounded in Phase 1: Nervous System Upgrade (Redis Integration).
"""

import asyncio
import logging
import uuid
import json
from typing import Callable, Dict, List, Any, Awaitable, Optional
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
        if not self.timestamp:
            self.timestamp = datetime.now().timestamp()

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, data_str: str):
        data = json.loads(data_str)
        return cls(**data)

class MessageBus:
    """
    Publish/Subscribe Event Bus using Redis for inter-process communication
    with local fallback for development.
    """
    def __init__(self, redis_host='localhost', redis_port=6379):
        self._subscribers: Dict[str, List[Callable[[Message], Awaitable[None]]]] = {}
        self.bus_id = f"BUS_{uuid.uuid4().hex[:8].upper()}"
        self.redis = None
        self.pubsub_task = None
        
        # Connect to Redis Nervous System
        try:
            import redis.asyncio as redis
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            logger.info(f"✅ Nervous System Uplink (Redis) Established: {self.bus_id}")
        except Exception as e:
            logger.warning(f"⚠️ Redis Uplink Failed ({e}). Running in Local Context Only.")

    def subscribe(self, channel: str, callback: Callable[[Message], Awaitable[None]]):
        """Register a callback for a specific channel."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
            
            # If using Redis, we need a permanent listener for this new channel
            if self.redis:
                if not self.pubsub_task:
                    self.pubsub_task = asyncio.create_task(self._redis_listener())
                
        self._subscribers[channel].append(callback)
        logger.info(f"New Subscriber on channel '{channel}'")

    async def _redis_listener(self):
        """Background task that pulls messages from Redis and routes to local callbacks."""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.psubscribe("*") # Listen to all channels
        except Exception as e:
            logger.warning(f"⚠️ Redis Uplink Failed in Listener ({e}). Disabling Redis.")
            self.redis = None
            return
        
        logger.info("📡 Global Nervous System Listener Active.")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    channel = message["channel"]
                    data = message["data"]
                    
                    try:
                        msg = Message.from_json(data)
                        # Avoid echoing our own messages if sent via Redis
                        if msg.sender_id == self.bus_id:
                            continue
                            
                        await self._dispatch_local(channel, msg)
                    except Exception as e:
                        logger.error(f"Error decoding Redis message: {e}")
        except asyncio.CancelledError:
            await pubsub.punsubscribe("*")
            await pubsub.close()

    async def _dispatch_local(self, channel: str, msg: Message):
        """Dispatches a message to local subscribers."""
        if channel in self._subscribers:
            tasks = []
            for callback in self._subscribers[channel]:
                 try:
                     tasks.append(callback(msg))
                 except Exception as e:
                     logger.error(f"Error in subscriber {callback}: {e}")
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def publish(self, channel: str, payload: Any, sender_id: Optional[str] = None):
        """Publish a message to all subscribers (Local and Redis)."""
        sender = sender_id or self.bus_id
        msg = Message(
            channel=channel,
            payload=payload,
            sender_id=sender,
            message_id=f"MSG_{uuid.uuid4().hex[:8].upper()}"
        )

        logger.info(f"Event: [{channel}] from {sender}")
        
        # 1. Dispatch locally first for speed
        await self._dispatch_local(channel, msg)
        
        # 2. Transmit to the rest of the Nervous System (Redis)
        if self.redis:
            try:
                await self.redis.publish(channel, msg.to_json())
            except Exception as e:
                logger.error(f"Failed to publish to Redis: {e}")

# Global Bus Instance
global_bus = MessageBus()
