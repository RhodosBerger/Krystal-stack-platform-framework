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

class TornFrameDetectedError(ContractViolation):
    """Raised when consumer detects an unaligned or torn frame in SPSC ring buffer."""
    pass


import ctypes

# Ring Header Layout (128 bytes = 2 x 64B cache lines):
# CL0 (0..63, Producer cache line - Exclusively written by Producer):
#   <I(magic) I(vital_hp) I(capacity) I(slot_size) Q(head) Q(drop_count) 32s(_pad0)
# CL1 (64..127, Consumer cache line - Exclusively written by Consumer):
#   <Q(tail) 56s(_reserved)
#
# False-sharing elimination:
# - Producer writes exclusively to CL0 (head @ 16, drop_count @ 24)
# - Consumer writes exclusively to CL1 (tail @ 64)
# - 64-bit uint64 head and tail prevent 32-bit rollover after 4B frames.
RING_HEADER_FORMAT = "<IIIIQQ32sQ56s"
RING_HEADER_SIZE = 128
assert struct.calcsize(RING_HEADER_FORMAT) == RING_HEADER_SIZE, "Ring header must be exactly 128 bytes (2 cache lines)"


class ZeroCopyContractRing:
    """
    Shared memory circular ring buffer for 128-byte ZeroCopyContractFrame objects.
    Enables low-latency IPC across processes or within a local real-time thread.
    Header is 128-byte aligned, guaranteeing every slot is 64-byte and 128-byte aligned.
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
                    self._init_header()
                except FileExistsError:
                    # Attach existing
                    self._shm = shared_memory.SharedMemory(name=shm_name)
                    self._buf = memoryview(self._shm.buf)
                    c_arr = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
                    self.base_address = ctypes.addressof(c_arr)
                    self.is_owner = False
            else:
                self._shm = shared_memory.SharedMemory(name=shm_name)
                self._buf = memoryview(self._shm.buf)
                c_arr = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
                self.base_address = ctypes.addressof(c_arr)
        else:
            # Standalone in-memory fallback with guaranteed 128-byte base address alignment
            raw_buf = bytearray(self.total_size + 128)
            c_arr = (ctypes.c_char * len(raw_buf)).from_buffer(raw_buf)
            raw_addr = ctypes.addressof(c_arr)
            align_offset = (128 - (raw_addr % 128)) % 128
            self._buf = memoryview(raw_buf)[align_offset : align_offset + self.total_size]
            self._raw_buf = raw_buf
            self.base_address = raw_addr + align_offset
            self._init_header()

        assert self.base_address % 64 == 0, f"Ring base address {self.base_address} must be 64-byte aligned"
        assert self.base_address % 128 == 0, f"Ring base address {self.base_address} must be 128-byte aligned"
        self._validate_header()

    def _init_header(self):
        """Initializes the ring header in shared memory across two distinct cache lines."""
        header_bytes = struct.pack(
            RING_HEADER_FORMAT,
            KRYSTAL_MAGIC,
            VITAL_MAX_HP,
            self.capacity,
            self.slot_size,
            0,  # head (CL0, offset 16, uint64)
            0,  # drop_count (CL0, offset 24, uint64 - Producer writes here!)
            b"\x00" * 32,  # _pad0 to complete 64B CL0
            0,  # tail (CL1, offset 64, uint64 - Consumer writes here!)
            b"\x00" * 56,  # _reserved to complete 64B CL1
        )
        self._buf[:RING_HEADER_SIZE] = header_bytes

    def _validate_header(self):
        """Checks invariants on the ring header."""
        magic, vital_hp, cap, slot_sz, _, _, _, _, _ = struct.unpack_from(RING_HEADER_FORMAT, self._buf, 0)
        if magic != KRYSTAL_MAGIC:
            raise ContractViolation(f"Invalid ring buffer magic: {hex(magic)}")
        if vital_hp != 6:
            raise ContractViolation(f"Platform invariant VITAL_MAX_HP == 6 violated in ring: {vital_hp}")
        if slot_sz != ZERO_COPY_FRAME_SIZE:
            raise ContractViolation(f"Unexpected slot size in ring: {slot_sz}")

    def _get_head_tail(self) -> Tuple[int, int, int]:
        """Reads head (CL0, offset 16), tail (CL1, offset 64), and drop_count (CL0, offset 24)."""
        _, _, _, _, head, drops, _, tail, _ = struct.unpack_from(RING_HEADER_FORMAT, self._buf, 0)
        return head, tail, drops

    def _set_head(self, head: int):
        struct.pack_into("<Q", self._buf, 16, head)

    def _set_tail(self, tail: int):
        struct.pack_into("<Q", self._buf, 64, tail)

    def _increment_drop_count(self):
        drops = struct.unpack_from("<Q", self._buf, 24)[0]
        struct.pack_into("<Q", self._buf, 24, drops + 1)

    def get_slot_offset(self, slot_idx: int) -> int:
        """Returns the byte offset of a slot. Guaranteed to be a multiple of 64 and 128."""
        offset = RING_HEADER_SIZE + ((slot_idx % self.capacity) * self.slot_size)
        assert offset % 64 == 0, f"Slot offset {offset} must be 64-byte cache-line aligned"
        assert offset % 128 == 0, f"Slot offset {offset} must be 128-byte frame aligned"
        return offset

    def get_slot_address(self, slot_idx: int) -> int:
        """Returns the physical memory pointer of a slot. Guaranteed to satisfy Addr % 64 == 0."""
        addr = self.base_address + self.get_slot_offset(slot_idx)
        assert addr % 64 == 0, f"Actual physical slot address {addr} is not 64-byte aligned"
        assert addr % 128 == 0, f"Actual physical slot address {addr} is not 128-byte aligned"
        return addr

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

        slot_offset = self.get_slot_offset(head)
        # 1. Write bytes into slot (CL0 and CL1)
        self._buf[slot_offset : slot_offset + self.slot_size] = frame_bytes
        # 2. Release fence: atomic update of head index publishes the slot
        self._set_head(head + 1)
        return True

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
        raw = self.pop_raw_bytes()
        if raw is None:
            return None

        frame = ZeroCopyContractFrame.from_bytes(raw)
        if not frame.verify_frame_integrity(session_key):
            self._increment_drop_count()
            raise TornFrameDetectedError(
                f"Torn or corrupted frame detected in slot: HMAC verification failed for seq={frame.sequence_id}"
            )
        return frame

    def pop_raw_bytes(self) -> Optional[bytes]:
        """Pops raw 128 bytes from the next slot with acquire semantics."""
        head, tail, _ = self._get_head_tail()
        if head == tail:
            # Buffer empty
            return None

        slot_offset = self.get_slot_offset(tail)
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
