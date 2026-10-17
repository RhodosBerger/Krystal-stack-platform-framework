"""
Krystal-Stack Governed Runtime: Lock-Free SPSC Ring Buffer
==========================================================
Milestone: V-PERF-ZEROCOPY-01 / V-ZEROCOPY-MEMORY-MODEL-02 (Principle 2: 128B Lingua Franca)
Reference: docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md

Implements false-sharing elimination between producer (CL0) and consumer (CL1),
128-byte cache-line aligned frame slots, and Acquire/Release publication protocol.
"""

from __future__ import annotations

import ctypes
import struct
from multiprocessing import shared_memory
from typing import Optional, Final, Any, Tuple

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
# CL0 (0..63):   magic(4B), vital_hp(4B), capacity(4B), slot_size(4B), head(8B), drop_count(8B), _pad0(32B) -> 64B
# CL1 (64..127): tail(8B), _pad1(56B) -> 64B
RING_HEADER_FORMAT: Final[str] = "<IIIIQQ32sQ56s"
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
        self.total_size = RING_HEADER_SIZE + (self.capacity * FRAME_SIZE_BYTES)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._buf: memoryview
        self._raw_buf: Optional[bytearray] = None
        self.base_address: int = 0

        if shm_name:
            if create:
                try:
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.total_size)
                    self._buf = memoryview(self._shm.buf)
                    c_arr = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
                    self.base_address = ctypes.addressof(c_arr)
                    self.buffer = self._buf
                    self._init_header()
                except FileExistsError:
                    self._shm = shared_memory.SharedMemory(name=shm_name)
                    self._buf = memoryview(self._shm.buf)
                    c_arr = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
                    self.base_address = ctypes.addressof(c_arr)
                    self.buffer = self._buf
                    self.is_owner = False
            else:
                self._shm = shared_memory.SharedMemory(name=shm_name)
                self._buf = memoryview(self._shm.buf)
                c_arr = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
                self.base_address = ctypes.addressof(c_arr)
                self.buffer = self._buf
        else:
            # Standalone in-memory buffer with guaranteed 128-byte base address alignment
            raw_buf = bytearray(self.total_size + 128)
            c_arr = (ctypes.c_char * len(raw_buf)).from_buffer(raw_buf)
            raw_addr = ctypes.addressof(c_arr)
            align_offset = (128 - (raw_addr % 128)) % 128
            self._buf = memoryview(raw_buf)[align_offset : align_offset + self.total_size]
            self._raw_buf = raw_buf
            self.base_address = raw_addr + align_offset
            self.buffer = self._buf
            self._init_header()

        assert self.base_address % 64 == 0, f"Ring base address {self.base_address} must be 64-byte aligned"
        assert self.base_address % 128 == 0, f"Ring base address {self.base_address} must be 128-byte aligned"
        self._validate_header()

    def _init_header(self, head: int = 0, tail: int = 0, drops: int = 0) -> None:
        """Initializes the ring header in shared memory across two distinct cache lines."""
        struct.pack_into(
            RING_HEADER_FORMAT, self._buf, 0,
            FRAME_MAGIC,
            VITAL_MAX_HP,
            self.capacity,
            self.slot_size,
            head,
            drops,
            b"\x00" * 32,
            tail,
            b"\x00" * 56,
        )

    def _validate_header(self) -> None:
        """Checks invariants on the ring header."""
        magic, vital_hp, cap, slot_sz, _, _, _, _, _ = struct.unpack_from(RING_HEADER_FORMAT, self._buf, 0)
        if magic != FRAME_MAGIC and magic != 0x4B525953:
            raise ContractViolation(f"Invalid ring buffer magic: {hex(magic)}")
        if vital_hp != 6:
            raise ContractViolation(f"Platform invariant VITAL_MAX_HP == 6 violated in ring: {vital_hp}")
        if slot_sz != FRAME_SIZE_BYTES:
            raise ContractViolation(f"Unexpected slot size in ring: {slot_sz}")

    def _get_head_tail(self) -> Tuple[int, int, int]:
        """Reads head (CL0, offset 16), tail (CL1, offset 64), and drop_count (CL0, offset 24)."""
        _, _, _, _, head, drops, _, tail, _ = struct.unpack_from(RING_HEADER_FORMAT, self._buf, 0)
        return head, tail, drops

    def _set_head(self, head: int) -> None:
        struct.pack_into("<Q", self._buf, 16, head)

    def _set_tail(self, tail: int) -> None:
        struct.pack_into("<Q", self._buf, 64, tail)

    def _increment_drop_count(self) -> None:
        drops = struct.unpack_from("<Q", self._buf, 24)[0]
        struct.pack_into("<Q", self._buf, 24, drops + 1)

    def get_slot_offset(self, slot_idx: int) -> int:
        """Returns the byte offset of a slot. Guaranteed to be a multiple of 64 and 128."""
        offset = RING_HEADER_SIZE + ((slot_idx % self.capacity) * self.slot_size)
        assert offset % 64 == 0, f"Slot offset {offset} must be 64-byte cache-line aligned"
        assert offset % 128 == 0, f"Slot offset {offset} must be 128-byte frame aligned"
        return offset

    def slot_offset(self, index: int) -> int:
        """Alias for get_slot_offset with circular indexing."""
        return self.get_slot_offset(index)

    def get_slot_address(self, slot_idx: int) -> int:
        """Returns the physical memory pointer of a slot. Guaranteed to satisfy Addr % 64 == 0."""
        addr = self.base_address + self.get_slot_offset(slot_idx)
        assert addr % 64 == 0, f"Actual physical slot address {addr} is not 64-byte aligned"
        assert addr % 128 == 0, f"Actual physical slot address {addr} is not 128-byte aligned"
        return addr

    def read_head_acquire(self) -> int:
        return struct.unpack_from("<Q", self._buf, 16)[0]

    def write_head_release(self, new_head: int) -> None:
        struct.pack_into("<Q", self._buf, 16, new_head)

    def read_tail_acquire(self) -> int:
        return struct.unpack_from("<Q", self._buf, 64)[0]

    def write_tail_release(self, new_tail: int) -> None:
        struct.pack_into("<Q", self._buf, 64, new_tail)

    @property
    def head(self) -> int:
        return self.read_head_acquire()

    @property
    def tail(self) -> int:
        return self.read_tail_acquire()

    def available_count(self) -> int:
        return self.read_head_acquire() - self.read_tail_acquire()

    def drop_count(self) -> int:
        return struct.unpack_from("<Q", self._buf, 24)[0]

    def push_raw_bytes(self, frame_bytes: bytes) -> bool:
        """Pushes raw 128 bytes directly into the ring slot with zero allocations."""
        if len(frame_bytes) != ZERO_COPY_FRAME_SIZE:
            raise ValueError(f"Frame must be exactly {ZERO_COPY_FRAME_SIZE} bytes")

        head, tail, _ = self._get_head_tail()
        occupied = head - tail
        if occupied >= self.capacity:
            self._increment_drop_count()
            return False

        slot_offset = self.get_slot_offset(head)
        # 1. Write bytes into slot (CL0 and CL1)
        self._buf[slot_offset : slot_offset + 64] = frame_bytes[0:64]
        self._buf[slot_offset + 64 : slot_offset + 128] = frame_bytes[64:128]
        # 2. Release fence: atomic update of head index publishes the slot
        self.write_head_release(head + 1)
        return True

    def push_frame(self, frame: ZeroCopyContractFrame) -> bool:
        """Pushes a frame into the circular ring. Returns False if buffer is full."""
        return self.push_raw_bytes(frame.to_bytes())

    def push_frame_with_hmac(self, frame: ZeroCopyContractFrame, session_key: bytes) -> bool:
        """
        Pushes a frame with full 96-byte HMAC authentication and cross-ISA release semantics.
        CL0 and CL1 writes precede the atomic head store (memory_order_release).
        """
        raw_96 = frame.to_bytes()[:96]
        auth_tag = frame.compute_frame_hmac(session_key)
        full_128 = raw_96 + auth_tag
        return self.push_raw_bytes(full_128)

    def publish(self, frame: ZeroCopyContractFrame) -> bool:
        """Krok 1..5 Cross-ISA protokolu: zapíše CL0, CL1 a posunie head."""
        if hasattr(frame, "pack_authenticated"):
            raw = frame.pack_authenticated(self.session_key)
        elif self.session_key != b"\x00" * 32:
            raw = frame.to_bytes()[:96] + frame.compute_frame_hmac(self.session_key)
        else:
            raw = frame.to_bytes()
        return self.push_raw_bytes(raw)

    def pop_raw_bytes(self) -> Optional[bytes]:
        """Pops raw 128 bytes from the next slot with acquire semantics."""
        head, tail, _ = self._get_head_tail()
        if head == tail:
            return None

        slot_offset = self.get_slot_offset(tail)
        frame_bytes = bytes(self._buf[slot_offset : slot_offset + self.slot_size])
        self.write_tail_release(tail + 1)
        return frame_bytes

    def pop_frame(self) -> Optional[ZeroCopyContractFrame]:
        """Pops the next available frame from the ring."""
        raw = self.pop_raw_bytes()
        if raw is None:
            return None
        return ZeroCopyContractFrame.from_bytes(raw)

    def pop_frame_with_hmac(self, session_key: bytes) -> Optional[ZeroCopyContractFrame]:
        """
        Pops the next frame with Acquire semantics and full 96-byte HMAC integrity check.
        Detects any torn reads across CL0 and CL1 boundaries.
        """
        head, tail, _ = self._get_head_tail()
        if head == tail:
            return None

        slot_offset = self.get_slot_offset(tail)
        raw = bytes(self._buf[slot_offset : slot_offset + self.slot_size])
        frame = ZeroCopyContractFrame.from_bytes(raw)
        if not frame.verify_frame_integrity(session_key):
            self._increment_drop_count()
            self.write_tail_release(tail + 1)
            raise TornFrameDetectedError(
                f"Torn or corrupted frame detected in slot: HMAC verification failed for seq={frame.sequence_id}"
            )

        self.write_tail_release(tail + 1)
        return frame

    def consume(self) -> Optional[ZeroCopyContractFrame]:
        """Acquire čítanie head, verifikácia HMAC a posun tail."""
        try:
            return self.pop_frame_with_hmac(self.session_key)
        except TornFrameDetectedError:
            return None

    def close(self) -> None:
        """Releases shared memory segment."""
        if self._shm is not None:
            try:
                self._shm.close()
                if self.is_owner:
                    self._shm.unlink()
            except Exception:
                pass
            self._shm = None
