"""
Krystal-Stack Governed Runtime: Zero-Copy Shared Memory Ring Buffer
===================================================================
Milestone: V-PERF-ZEROCOPY-01
Reference: docs/architecture/V_PERF_ZEROCOPY_01_ZERO_COPY_ENGINE.md

Provides lock-free, zero-copy shared memory IPC ring buffer for 128-byte contract
frames, enabling inter-process communication between Observation, Epistemic,
Authority, and Actuation planes at 120 Hz – 1 kHz simulation rates.

Invariants:
  1. VITAL_MAX_HP == 6 must hold in the ring header.
  2. Fixed 128-byte slot size.
  3. Safe circular wrapping with head/tail atomic indices.
"""

import struct
from multiprocessing import shared_memory
from typing import Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.zero_copy_contracts import (
    KRYSTAL_MAGIC,
    ZERO_COPY_FRAME_SIZE,
    ZeroCopyContractFrame,
)

# Ring Header Layout (64 bytes):
# <I(magic) I(vital_hp) I(capacity) I(slot_size) I(head) I(tail) Q(drop_count) 32s(reserved)
RING_HEADER_FORMAT = "<IIIIIIQ32s"
RING_HEADER_SIZE = 64
assert struct.calcsize(RING_HEADER_FORMAT) == RING_HEADER_SIZE


class ZeroCopyContractRing:
    """
    Shared memory circular ring buffer for 128-byte ZeroCopyContractFrame objects.
    Enables low-latency IPC across processes or within a local real-time thread.
    """

    def __init__(
        self,
        shm_name: Optional[str] = None,
        capacity: int = 1024,
        create: bool = False,
    ):
        self.capacity = capacity
        self.slot_size = ZERO_COPY_FRAME_SIZE
        self.total_size = RING_HEADER_SIZE + (self.capacity * self.slot_size)
        self.shm_name = shm_name
        self.is_owner = create
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._buf: bytearray | memoryview

        if shm_name:
            if create:
                try:
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.total_size)
                    self._buf = self._shm.buf
                    self._init_header()
                except FileExistsError:
                    # Attach existing
                    self._shm = shared_memory.SharedMemory(name=shm_name)
                    self._buf = self._shm.buf
                    self.is_owner = False
            else:
                self._shm = shared_memory.SharedMemory(name=shm_name)
                self._buf = self._shm.buf
        else:
            # Standalone in-memory fallback
            self._buf = bytearray(self.total_size)
            self._init_header()

        self._validate_header()

    def _init_header(self):
        """Initializes the ring header in shared memory."""
        header_bytes = struct.pack(
            RING_HEADER_FORMAT,
            KRYSTAL_MAGIC,
            VITAL_MAX_HP,
            self.capacity,
            self.slot_size,
            0,  # head
            0,  # tail
            0,  # drop_count
            b"\x00" * 32,
        )
        self._buf[:RING_HEADER_SIZE] = header_bytes

    def _validate_header(self):
        """Checks invariants on the ring header."""
        magic, vital_hp, cap, slot_sz, _, _, _, _ = struct.unpack_from(RING_HEADER_FORMAT, self._buf, 0)
        if magic != KRYSTAL_MAGIC:
            raise ContractViolation(f"Invalid ring buffer magic: {hex(magic)}")
        if vital_hp != 6:
            raise ContractViolation(f"Platform invariant VITAL_MAX_HP == 6 violated in ring: {vital_hp}")
        if slot_sz != ZERO_COPY_FRAME_SIZE:
            raise ContractViolation(f"Unexpected slot size in ring: {slot_sz}")

    def _get_head_tail(self) -> Tuple[int, int, int]:
        """Reads head, tail, and drop_count from the ring header."""
        _, _, _, _, head, tail, drops, _ = struct.unpack_from(RING_HEADER_FORMAT, self._buf, 0)
        return head, tail, drops

    def _set_head(self, head: int):
        struct.pack_into("<I", self._buf, 16, head)

    def _set_tail(self, tail: int):
        struct.pack_into("<I", self._buf, 20, tail)

    def _increment_drop_count(self):
        drops = struct.unpack_from("<Q", self._buf, 24)[0]
        struct.pack_into("<Q", self._buf, 24, drops + 1)

    def push_frame(self, frame: ZeroCopyContractFrame) -> bool:
        """Pushes a frame into the circular ring. Returns False if buffer is full."""
        return self.push_raw_bytes(frame.to_bytes())

    def push_raw_bytes(self, frame_bytes: bytes) -> bool:
        """Pushes raw 128 bytes directly into the ring slot with zero allocations."""
        if len(frame_bytes) != ZERO_COPY_FRAME_SIZE:
            raise ValueError(f"Frame must be exactly {ZERO_COPY_FRAME_SIZE} bytes")

        head, tail, _ = self._get_head_tail()
        occupied = head - tail
        if occupied >= self.capacity:
            # Buffer overflow
            self._increment_drop_count()
            return False

        slot_idx = head % self.capacity
        slot_offset = RING_HEADER_SIZE + (slot_idx * self.slot_size)
        self._buf[slot_offset : slot_offset + self.slot_size] = frame_bytes
        self._set_head(head + 1)
        return True

    def pop_frame(self) -> Optional[ZeroCopyContractFrame]:
        """Pops the next available frame from the ring."""
        raw = self.pop_raw_bytes()
        if raw is None:
            return None
        return ZeroCopyContractFrame.from_bytes(raw)

    def pop_raw_bytes(self) -> Optional[bytes]:
        """Pops raw 128 bytes from the next slot."""
        head, tail, _ = self._get_head_tail()
        if head == tail:
            # Buffer empty
            return None

        slot_idx = tail % self.capacity
        slot_offset = RING_HEADER_SIZE + (slot_idx * self.slot_size)
        frame_bytes = bytes(self._buf[slot_offset : slot_offset + self.slot_size])
        self._set_tail(tail + 1)
        return frame_bytes

    def available_count(self) -> int:
        head, tail, _ = self._get_head_tail()
        return head - tail

    def drop_count(self) -> int:
        _, _, drops = self._get_head_tail()
        return drops

    def close(self):
        """Releases shared memory segment."""
        if self._shm:
            try:
                self._shm.close()
                if self.is_owner:
                    self._shm.unlink()
            except Exception:
                pass
            self._shm = None
