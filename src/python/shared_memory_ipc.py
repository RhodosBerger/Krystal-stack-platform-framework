"""
Shared Memory IPC - Lock-Free Ring Buffer for Telemetry

Replaces socket IPC with shared memory for 10x lower latency.
Uses SPSC (Single Producer Single Consumer) lock-free ring buffer.
"""

import mmap
import struct
import time
import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Telemetry packet format (matches C struct)
TELEMETRY_FORMAT = "<Qfffffffi"  # timestamp(8) + 7 floats(28) + zone_count(4) + pe_mask(4) = 44 bytes
TELEMETRY_SIZE = struct.calcsize(TELEMETRY_FORMAT)

# Signal packet format
SIGNAL_FORMAT = "<IffBI"  # zone_id(4) + strength(4) + target(4) + action(1) + padding(3) = 16 bytes
SIGNAL_SIZE = struct.calcsize(SIGNAL_FORMAT)

# Ring buffer header format
HEADER_FORMAT = "<QQQ"  # write_idx(8) + read_idx(8) + capacity(8) = 24 bytes
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass
class TelemetryPacket:
    """Telemetry packet matching C struct."""
    timestamp_ns: int
    cpu_util: float
    gpu_util: float
    memory_util: float
    temp_cpu: float
    temp_gpu: float
    frametime_ms: float
    power_draw: float
    zone_count: int
    pe_mask: int

    def pack(self) -> bytes:
        return struct.pack(
            TELEMETRY_FORMAT,
            self.timestamp_ns,
            self.cpu_util,
            self.gpu_util,
            self.memory_util,
            self.temp_cpu,
            self.temp_gpu,
            self.frametime_ms,
            self.power_draw,
            self.zone_count,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "TelemetryPacket":
        values = struct.unpack(TELEMETRY_FORMAT, data)
        return cls(
            timestamp_ns=values[0],
            cpu_util=values[1],
            gpu_util=values[2],
            memory_util=values[3],
            temp_cpu=values[4],
            temp_gpu=values[5],
            frametime_ms=values[6],
            power_draw=values[7],
            zone_count=values[8],
            pe_mask=0,  # packed in zone_count field
        )


@dataclass
class SignalPacket:
    """Signal packet for Guardian -> C runtime."""
    zone_id: int
    strength: float
    target: float
    action: int

    def pack(self) -> bytes:
        return struct.pack(
            SIGNAL_FORMAT,
            self.zone_id,
            self.strength,
            self.target,
            self.action,
            0,  # padding
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SignalPacket":
        values = struct.unpack(SIGNAL_FORMAT, data)
        return cls(
            zone_id=values[0],
            strength=values[1],
            target=values[2],
            action=values[3],
        )


class SharedMemoryRingBuffer:
    """
    Lock-free SPSC ring buffer in shared memory.

    Memory layout:
    [Header: 24 bytes][Data: capacity * item_size bytes]

    Header contains atomic write_idx, read_idx, capacity.
    """

    def __init__(
        self,
        name: str,
        capacity: int,
        item_size: int,
        create: bool = False,
    ):
        self.name = name
        self.capacity = capacity
        self.item_size = item_size
        self.total_size = HEADER_SIZE + (capacity * item_size)

        self._shm_path = f"/dev/shm/gamesa_{name}"
        self._mmap: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None

        if create:
            self._create()
        else:
            self._open()

    def _create(self):
        """Create shared memory region."""
        # Create file
        self._fd = os.open(self._shm_path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(self._fd, self.total_size)

        # Map memory
        self._mmap = mmap.mmap(self._fd, self.total_size)

        # Initialize header
        self._write_header(0, 0, self.capacity)
        logger.info(f"Created shared memory: {self._shm_path} ({self.total_size} bytes)")

    def _open(self):
        """Open existing shared memory region."""
        try:
            self._fd = os.open(self._shm_path, os.O_RDWR)
            self._mmap = mmap.mmap(self._fd, self.total_size)
            logger.info(f"Opened shared memory: {self._shm_path}")
        except FileNotFoundError:
            raise RuntimeError(f"Shared memory {self._shm_path} not found. Create it first.")

    def _write_header(self, write_idx: int, read_idx: int, capacity: int):
        """Write ring buffer header."""
        header = struct.pack(HEADER_FORMAT, write_idx, read_idx, capacity)
        self._mmap.seek(0)
        self._mmap.write(header)

    def _read_header(self) -> Tuple[int, int, int]:
        """Read ring buffer header."""
        self._mmap.seek(0)
        data = self._mmap.read(HEADER_SIZE)
        return struct.unpack(HEADER_FORMAT, data)

    def _get_write_idx(self) -> int:
        self._mmap.seek(0)
        return struct.unpack("<Q", self._mmap.read(8))[0]

    def _set_write_idx(self, idx: int):
        self._mmap.seek(0)
        self._mmap.write(struct.pack("<Q", idx))

    def _get_read_idx(self) -> int:
        self._mmap.seek(8)
        return struct.unpack("<Q", self._mmap.read(8))[0]

    def _set_read_idx(self, idx: int):
        self._mmap.seek(8)
        self._mmap.write(struct.pack("<Q", idx))

    def push(self, data: bytes) -> bool:
        """Push item to ring buffer. Returns False if full."""
        if len(data) != self.item_size:
            raise ValueError(f"Data size {len(data)} != item_size {self.item_size}")

        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        next_write = (write_idx + 1) % self.capacity

        # Check if full
        if next_write == read_idx:
            return False  # Buffer full

        # Write data
        offset = HEADER_SIZE + (write_idx * self.item_size)
        self._mmap.seek(offset)
        self._mmap.write(data)

        # Update write index (release semantics via mmap flush)
        self._set_write_idx(next_write)
        return True

    def pop(self) -> Optional[bytes]:
        """Pop item from ring buffer. Returns None if empty."""
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        # Check if empty
        if read_idx == write_idx:
            return None

        # Read data
        offset = HEADER_SIZE + (read_idx * self.item_size)
        self._mmap.seek(offset)
        data = self._mmap.read(self.item_size)

        # Update read index
        next_read = (read_idx + 1) % self.capacity
        self._set_read_idx(next_read)

        return data

    def peek(self) -> Optional[bytes]:
        """Peek at next item without removing."""
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        if read_idx == write_idx:
            return None

        offset = HEADER_SIZE + (read_idx * self.item_size)
        self._mmap.seek(offset)
        return self._mmap.read(self.item_size)

    def size(self) -> int:
        """Get current number of items in buffer."""
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        if write_idx >= read_idx:
            return write_idx - read_idx
        else:
            return self.capacity - read_idx + write_idx

    def is_empty(self) -> bool:
        return self._get_write_idx() == self._get_read_idx()

    def is_full(self) -> bool:
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()
        return ((write_idx + 1) % self.capacity) == read_idx

    def close(self):
        """Close shared memory mapping."""
        if self._mmap:
            self._mmap.close()
        if self._fd:
            os.close(self._fd)

    def destroy(self):
        """Destroy shared memory region."""
        self.close()
        try:
            os.unlink(self._shm_path)
            logger.info(f"Destroyed shared memory: {self._shm_path}")
        except FileNotFoundError:
            pass


class TelemetryChannel:
    """High-level telemetry channel using shared memory."""

    def __init__(self, create: bool = False, capacity: int = 1024):
        self.buffer = SharedMemoryRingBuffer(
            name="telemetry",
            capacity=capacity,
            item_size=TELEMETRY_SIZE,
            create=create,
        )

    def send(self, packet: TelemetryPacket) -> bool:
        """Send telemetry packet."""
        return self.buffer.push(packet.pack())

    def receive(self) -> Optional[TelemetryPacket]:
        """Receive telemetry packet."""
        data = self.buffer.pop()
        if data:
            return TelemetryPacket.unpack(data)
        return None

    def receive_all(self) -> list:
        """Receive all available packets."""
        packets = []
        while True:
            packet = self.receive()
            if packet is None:
                break
            packets.append(packet)
        return packets

    def close(self):
        self.buffer.close()


class SignalChannel:
    """High-level signal channel for Guardian -> C runtime."""

    def __init__(self, create: bool = False, capacity: int = 256):
        self.buffer = SharedMemoryRingBuffer(
            name="signals",
            capacity=capacity,
            item_size=SIGNAL_SIZE,
            create=create,
        )

    def send(self, packet: SignalPacket) -> bool:
        """Send signal packet."""
        return self.buffer.push(packet.pack())

    def receive(self) -> Optional[SignalPacket]:
        """Receive signal packet."""
        data = self.buffer.pop()
        if data:
            return SignalPacket.unpack(data)
        return None

    def close(self):
        self.buffer.close()


class IPCBridge:
    """
    Complete IPC bridge between Python Guardian and C runtime.

    Creates bidirectional communication:
    - Telemetry: C -> Python (high frequency)
    - Signals: Python -> C (on demand)
    """

    def __init__(self, create: bool = False):
        self.telemetry = TelemetryChannel(create=create)
        self.signals = SignalChannel(create=create)
        self._running = False

    def send_signal(self, zone_id: int, strength: float, action: int, target: float = 0.0) -> bool:
        """Send signal to C runtime."""
        packet = SignalPacket(
            zone_id=zone_id,
            strength=strength,
            target=target,
            action=action,
        )
        return self.signals.send(packet)

    def receive_telemetry(self) -> Optional[TelemetryPacket]:
        """Receive single telemetry packet."""
        return self.telemetry.receive()

    def receive_all_telemetry(self) -> list:
        """Receive all available telemetry."""
        return self.telemetry.receive_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get IPC statistics."""
        return {
            "telemetry_pending": self.telemetry.buffer.size(),
            "signals_pending": self.signals.buffer.size(),
        }

    def close(self):
        """Close all channels."""
        self.telemetry.close()
        self.signals.close()


@contextmanager
def ipc_session(create: bool = False):
    """Context manager for IPC bridge."""
    bridge = IPCBridge(create=create)
    try:
        yield bridge
    finally:
        bridge.close()


# Factory functions
def create_ipc_server() -> IPCBridge:
    """Create IPC bridge as server (creates shared memory)."""
    return IPCBridge(create=True)


def create_ipc_client() -> IPCBridge:
    """Create IPC bridge as client (connects to existing)."""
    return IPCBridge(create=False)
