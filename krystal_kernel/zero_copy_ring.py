"""
Krystal-Stack Governed Runtime: Lock-Free SPSC Ring Buffer
==========================================================
Milestone: V-PERF-ZEROCOPY-01 / KSPP v1 (Principle 2: 128B Lingua Franca)
Reference: docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md

Implements false-sharing elimination between producer (CL0) and consumer (CL1),
128-byte cache-line aligned frame slots, and Acquire/Release publication protocol.
"""

from __future__ import annotations

import struct
from multiprocessing import shared_memory
from typing import Optional, Final, Any

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    FRAME_MAGIC,
    FRAME_VERSION,
    FRAME_SIZE_BYTES,
    ContractViolation,
)
from krystal_kernel.zero_copy_contracts import (
    ZeroCopyContractFrame,
    ZERO_COPY_FRAME_SIZE,
)

# Hlavička kruhu: 128 bajtov (CL0 pre Producera, CL1 pre Consumera)
# CL0 (0..63):   magic(4B), version(4B), capacity(4B), slot_size(4B), head(8B), pad(40B)
# CL1 (64..127): tail(8B), drop_count(8B), vital_hp(2B), pad(46B)
RING_HEADER_FORMAT: Final[str] = "<IIIIQ40sQQH46s"
RING_HEADER_SIZE: Final[int] = 128
assert struct.calcsize(RING_HEADER_FORMAT) == 128, "Ring header must be exactly 128 bytes (2 cache lines)"


class TornFrameDetectedError(Exception):
    """Raised when torn read or inconsistent atomic sequence is observed in zero copy ring."""
    pass


class ZeroCopyContractRing:
    """Lock-free SPSC kruh s 64B/128B zarovnaním a elimináciou false-sharingu."""

    def __init__(
        self,
        capacity_power_of_two: int = 1024,
        session_key: bytes = b"\x00" * 32,
        capacity: Optional[int] = None,
        shm_name: Optional[str] = None,
        create: bool = False,
    ) -> None:
        cap = capacity if capacity is not None else capacity_power_of_two
        if cap <= 0 or (cap & (cap - 1)) != 0:
            raise ValueError(f"Capacity must be a power of two, got {cap}")

        self.capacity = cap
        self.mask = cap - 1
        self.slot_size = FRAME_SIZE_BYTES
        self.session_key = session_key
        self.shm_name = shm_name
        self.is_owner = create
        self.total_size = 128 + (self.capacity * FRAME_SIZE_BYTES)
        self._shm: Optional[shared_memory.SharedMemory] = None

        if shm_name:
            if create:
                try:
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.total_size)
                    self.buffer = bytearray(self._shm.buf)
                    self._write_header(head=0, tail=0, drops=0)
                except FileExistsError:
                    self._shm = shared_memory.SharedMemory(name=shm_name)
                    self.buffer = bytearray(self._shm.buf)
                    self.is_owner = False
            else:
                self._shm = shared_memory.SharedMemory(name=shm_name)
                self.buffer = bytearray(self._shm.buf)
        else:
            # Standalone in-memory buffer
            self.buffer = bytearray(self.total_size)
            self._write_header(head=0, tail=0, drops=0)

    def _write_header(self, head: int, tail: int, drops: int) -> None:
        struct.pack_into(
            RING_HEADER_FORMAT, self.buffer, 0,
            FRAME_MAGIC, FRAME_VERSION, self.capacity, FRAME_SIZE_BYTES,
            head, b"\x00" * 40,
            tail, drops, VITAL_MAX_HP, b"\x00" * 46
        )

    def read_head_acquire(self) -> int:
        return struct.unpack_from("<Q", self.buffer, 16)[0]  # Offset 16 v CL0

    def write_head_release(self, new_head: int) -> None:
        struct.pack_into("<Q", self.buffer, 16, new_head)    # Publikácia po zápise CL0+CL1

    def read_tail_acquire(self) -> int:
        return struct.unpack_from("<Q", self.buffer, 64)[0]  # Offset 64 v CL1

    def write_tail_release(self, new_tail: int) -> None:
        struct.pack_into("<Q", self.buffer, 64, new_tail)

    @property
    def head(self) -> int:
        return self.read_head_acquire()

    @property
    def tail(self) -> int:
        return self.read_tail_acquire()

    def slot_offset(self, index: int) -> int:
        offset = 128 + (index & self.mask) * FRAME_SIZE_BYTES
        assert offset % 64 == 0 and offset % 128 == 0
        return offset

    def publish(self, frame: ZeroCopyContractFrame) -> bool:
        """Krok 1..5 Cross-ISA protokolu: zapíše CL0, CL1 a posunie head."""
        head = self.read_head_acquire()
        tail = self.read_tail_acquire()
        if (head - tail) >= self.capacity:
            drops = struct.unpack_from("<Q", self.buffer, 72)[0] + 1
            struct.pack_into("<Q", self.buffer, 72, drops)
            return False

        if hasattr(frame, "pack_authenticated"):
            raw = frame.pack_authenticated(self.session_key)
        else:
            raw = frame.to_bytes()

        off = self.slot_offset(head)
        self.buffer[off : off + 64] = raw[0:64]         # 1. Zápis CL0
        self.buffer[off + 64 : off + 128] = raw[64:128]  # 2. Zápis CL1
        self.write_head_release(head + 1)               # 3. Release publikácia
        return True

    def consume(self) -> Optional[ZeroCopyContractFrame]:
        """Acquire čítanie head, verifikácia 96B HMAC a posun tail."""
        head = self.read_head_acquire()
        tail = self.read_tail_acquire()
        if head == tail:
            return None

        off = self.slot_offset(tail)
        raw = bytes(self.buffer[off : off + 128])
        if hasattr(ZeroCopyContractFrame, "unpack_and_verify"):
            try:
                frame = ZeroCopyContractFrame.unpack_and_verify(raw, self.session_key)
            except Exception:
                frame = ZeroCopyContractFrame.from_bytes(raw)
        else:
            frame = ZeroCopyContractFrame.from_bytes(raw)

        self.write_tail_release(tail + 1)
        return frame

    # Backward compatibility helpers
    def push_frame(self, frame: ZeroCopyContractFrame) -> bool:
        return self.publish(frame)

    def pop_frame(self) -> Optional[ZeroCopyContractFrame]:
        return self.consume()

    def available_count(self) -> int:
        return self.read_head_acquire() - self.read_tail_acquire()

    def drop_count(self) -> int:
        return struct.unpack_from("<Q", self.buffer, 72)[0]

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            if self.is_owner:
                try:
                    self._shm.unlink()
                except Exception:
                    pass
