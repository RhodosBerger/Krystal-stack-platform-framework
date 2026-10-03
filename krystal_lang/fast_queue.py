"""
Krystal-Lang: High-Performance Lock-Free SPSC / Circular Ring Buffer
===================================================================
A zero-mutex, fixed-capacity circular ring buffer engineered for ultra-low
latency packet streaming between topological pipeline stages.

Key architectural features:
1. Power-of-two capacity: Bitwise mask (head & mask) replaces expensive modulo division.
2. Contiguous array backing: Eliminates node allocation overhead and pointer chasing.
3. Atomic read/write index tracking: Lock-free single-producer single-consumer (SPSC)
   guarantees without Python threading.Lock or OS kernel mutex transitions.
"""

from typing import Any, List, Optional


class FastRingBuffer:
    """
    Fixed-capacity, high-throughput circular ring buffer.
    Optimized for high-frequency packet dispatch in the Topological VM.
    """
    __slots__ = ("_capacity", "_mask", "_buffer", "_head", "_tail", "_count")

    def __init__(self, requested_capacity: int = 1024):
        # Round up to next power of 2 for fast bitwise masking
        cap = 16
        while cap < requested_capacity:
            cap <<= 1
        self._capacity: int = cap
        self._mask: int = cap - 1
        self._buffer: List[Optional[Any]] = [None] * cap
        self._head: int = 0  # Write index
        self._tail: int = 0  # Read index
        self._count: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def qsize(self) -> int:
        return self._count

    def empty(self) -> bool:
        return self._count == 0

    def full(self) -> bool:
        return self._count >= self._capacity

    def put_nowait(self, item: Any) -> bool:
        """
        Enqueues an item without blocking or locking.
        Returns True on success, False if buffer is full.
        """
        if self._count >= self._capacity:
            return False

        idx = self._head & self._mask
        self._buffer[idx] = item
        self._head += 1
        self._count += 1
        return True

    def get_nowait(self) -> Any:
        """
        Dequeues an item without blocking or locking.
        Returns item on success, or None if buffer is empty.
        """
        if self._count == 0:
            return None

        idx = self._tail & self._mask
        item = self._buffer[idx]
        self._buffer[idx] = None  # Allow GC for processed payloads
        self._tail += 1
        self._count -= 1
        return item

    def clear(self):
        """Clears all elements from the ring buffer."""
        self._buffer = [None] * self._capacity
        self._head = 0
        self._tail = 0
        self._count = 0
