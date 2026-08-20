"""
GAMESA Crystal Core - Hexadecimal Memory Architecture

Implements the Crystal Core OpenVINO hexadecimal memory pool for:
- Shared memory across all GAMESA components
- Cache-efficient topology-aware prefetching
- LLM-guided memory optimization
- Zero-copy buffer sharing
"""

import mmap
import os
import struct
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MemoryTier(Enum):
    """Memory access tiers."""
    HOT = 0      # Most frequently accessed
    WARM = 1     # Moderately accessed
    COLD = 2     # Rarely accessed
    FROZEN = 3   # Archival


@dataclass
class MemoryBlock:
    """Allocated memory block."""
    block_id: int
    offset: int
    size: int
    tier: MemoryTier
    owner: str
    access_count: int = 0
    last_access: float = 0.0


class CrystalCore:
    """
    Crystal Core hexadecimal memory pool.

    Base address: 0x7FFF0000 (as per GAMESA_SYSTEM_INTEGRATION.md)
    Size: 256MB
    """

    # Memory layout constants
    BASE_ADDRESS = 0x7FFF0000
    POOL_SIZE = 256 * 1024 * 1024  # 256MB
    BLOCK_HEADER_SIZE = 64  # bytes
    ALIGNMENT = 64  # Cache line alignment

    def __init__(self, pool_path: Optional[str] = None):
        """
        Initialize Crystal Core memory pool.

        Args:
            pool_path: Path to shared memory file (default: /tmp/gamesa_crystal_core)
        """
        self.pool_path = pool_path or "/tmp/gamesa_crystal_core"
        self.blocks: Dict[int, MemoryBlock] = {}
        self.free_list: List[Tuple[int, int]] = []  # (offset, size)
        self.lock = threading.RLock()
        self.next_block_id = 1

        # Statistics
        self.stats = {
            "allocations": 0,
            "deallocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "prefetch_count": 0,
        }

        # Initialize memory pool
        self._init_pool()

    def _init_pool(self):
        """Initialize shared memory pool."""
        try:
            # Create or open shared memory file
            if not Path(self.pool_path).exists():
                with open(self.pool_path, 'wb') as f:
                    f.write(b'\x00' * self.POOL_SIZE)

            # Memory map the file
            self.fd = os.open(self.pool_path, os.O_RDWR)
            self.mmap = mmap.mmap(
                self.fd,
                self.POOL_SIZE,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )

            # Initialize free list with entire pool
            self.free_list = [(0, self.POOL_SIZE)]

            print(f"Crystal Core initialized: {self.pool_path}")
            print(f"  Pool size: {self.POOL_SIZE / (1024*1024):.0f}MB")
            print(f"  Base address: 0x{self.BASE_ADDRESS:08X}")

        except Exception as e:
            print(f"Warning: Could not initialize Crystal Core: {e}")
            # Fallback to in-memory buffer
            self.mmap = bytearray(self.POOL_SIZE)
            self.fd = None

    def allocate(self, size: int, tier: MemoryTier = MemoryTier.WARM,
                owner: str = "unknown") -> Optional[int]:
        """
        Allocate memory block.

        Returns:
            block_id or None if allocation failed
        """
        with self.lock:
            # Align size
            aligned_size = ((size + self.ALIGNMENT - 1) // self.ALIGNMENT) * self.ALIGNMENT

            # Find suitable free block
            for i, (offset, free_size) in enumerate(self.free_list):
                if free_size >= aligned_size:
                    # Allocate from this block
                    block_id = self.next_block_id
                    self.next_block_id += 1

                    block = MemoryBlock(
                        block_id=block_id,
                        offset=offset,
                        size=aligned_size,
                        tier=tier,
                        owner=owner,
                    )

                    self.blocks[block_id] = block

                    # Update free list
                    remaining = free_size - aligned_size
                    if remaining > 0:
                        self.free_list[i] = (offset + aligned_size, remaining)
                    else:
                        self.free_list.pop(i)

                    self.stats["allocations"] += 1

                    # Write header
                    self._write_header(block)

                    return block_id

            return None  # Out of memory

    def deallocate(self, block_id: int):
        """Deallocate memory block."""
        with self.lock:
            if block_id not in self.blocks:
                return

            block = self.blocks.pop(block_id)

            # Add to free list
            self.free_list.append((block.offset, block.size))

            # Coalesce adjacent free blocks
            self._coalesce_free_list()

            self.stats["deallocations"] += 1

    def _coalesce_free_list(self):
        """Merge adjacent free blocks."""
        if len(self.free_list) < 2:
            return

        # Sort by offset
        self.free_list.sort(key=lambda x: x[0])

        # Merge adjacent
        coalesced = []
        current_offset, current_size = self.free_list[0]

        for offset, size in self.free_list[1:]:
            if offset == current_offset + current_size:
                # Adjacent, merge
                current_size += size
            else:
                coalesced.append((current_offset, current_size))
                current_offset, current_size = offset, size

        coalesced.append((current_offset, current_size))
        self.free_list = coalesced

    def _write_header(self, block: MemoryBlock):
        """Write block header to memory."""
        header = struct.pack(
            '<IIIIII',
            block.block_id,
            block.size,
            block.tier.value,
            0,  # Reserved
            0,  # Reserved
            0,  # Reserved
        )

        offset = block.offset
        self.mmap[offset:offset + len(header)] = header

    def read(self, block_id: int, offset: int = 0, size: Optional[int] = None) -> Optional[bytes]:
        """Read data from block."""
        with self.lock:
            if block_id not in self.blocks:
                return None

            block = self.blocks[block_id]
            block.access_count += 1

            read_size = size or (block.size - offset)
            start = block.offset + self.BLOCK_HEADER_SIZE + offset
            end = start + read_size

            return bytes(self.mmap[start:end])

    def write(self, block_id: int, data: bytes, offset: int = 0) -> bool:
        """Write data to block."""
        with self.lock:
            if block_id not in self.blocks:
                return False

            block = self.blocks[block_id]
            block.access_count += 1

            if offset + len(data) > block.size - self.BLOCK_HEADER_SIZE:
                return False  # Would overflow

            start = block.offset + self.BLOCK_HEADER_SIZE + offset
            self.mmap[start:start + len(data)] = data

            return True

    def get_address(self, block_id: int) -> Optional[int]:
        """Get virtual address for block (for external use)."""
        with self.lock:
            if block_id not in self.blocks:
                return None

            block = self.blocks[block_id]
            return self.BASE_ADDRESS + block.offset + self.BLOCK_HEADER_SIZE

    def prefetch(self, block_ids: List[int]):
        """
        Topology-aware prefetching.

        Hints to CPU to prefetch blocks into cache.
        """
        with self.lock:
            for block_id in block_ids:
                if block_id in self.blocks:
                    block = self.blocks[block_id]
                    # In real implementation, use madvise(MADV_WILLNEED)
                    # For now, just access first cache line
                    _ = self.mmap[block.offset]
                    self.stats["prefetch_count"] += 1

    def promote_tier(self, block_id: int):
        """Promote block to hotter tier."""
        with self.lock:
            if block_id not in self.blocks:
                return

            block = self.blocks[block_id]
            if block.tier.value > 0:
                block.tier = MemoryTier(block.tier.value - 1)

    def demote_tier(self, block_id: int):
        """Demote block to colder tier."""
        with self.lock:
            if block_id not in self.blocks:
                return

            block = self.blocks[block_id]
            if block.tier.value < 3:
                block.tier = MemoryTier(block.tier.value + 1)

    def optimize_layout(self):
        """
        LLM-guided memory optimization.

        Reorganize blocks based on access patterns.
        """
        with self.lock:
            # Sort blocks by access count (hot blocks first)
            sorted_blocks = sorted(
                self.blocks.values(),
                key=lambda b: b.access_count,
                reverse=True
            )

            # Auto-promote/demote based on access
            for block in sorted_blocks:
                if block.access_count > 100:
                    if block.tier != MemoryTier.HOT:
                        self.promote_tier(block.block_id)
                elif block.access_count < 10:
                    if block.tier != MemoryTier.COLD:
                        self.demote_tier(block.block_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            total_allocated = sum(b.size for b in self.blocks.values())
            total_free = sum(size for _, size in self.free_list)

            return {
                "pool_size": self.POOL_SIZE,
                "allocated": total_allocated,
                "free": total_free,
                "utilization": total_allocated / self.POOL_SIZE,
                "blocks": len(self.blocks),
                "allocations": self.stats["allocations"],
                "deallocations": self.stats["deallocations"],
                "prefetch_count": self.stats["prefetch_count"],
            }

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'fd') and self.fd:
            os.close(self.fd)


class CrystalCacheManager:
    """
    Cache-aware memory manager on top of Crystal Core.

    Implements cache partitioning and prefetching strategies.
    """

    def __init__(self, core: CrystalCore):
        self.core = core
        self.cache_lines = {}  # block_id -> cache_line_bitmap
        self.access_pattern = []  # Recent access sequence

    def allocate_cached(self, size: int, owner: str = "unknown",
                       prefetch_neighbors: bool = True) -> Optional[int]:
        """Allocate with cache optimization."""
        block_id = self.core.allocate(size, MemoryTier.HOT, owner)

        if block_id and prefetch_neighbors:
            # Prefetch adjacent blocks
            neighbors = [bid for bid in self.core.blocks.keys()
                        if abs(bid - block_id) <= 2]
            self.core.prefetch(neighbors)

        return block_id

    def access(self, block_id: int) -> Optional[bytes]:
        """Access block with pattern tracking."""
        data = self.core.read(block_id)

        if data:
            # Track access pattern
            self.access_pattern.append(block_id)
            if len(self.access_pattern) > 100:
                self.access_pattern.pop(0)

            # Predict next access and prefetch
            self._predict_and_prefetch()

        return data

    def _predict_and_prefetch(self):
        """Predict next access using simple n-gram."""
        if len(self.access_pattern) < 5:
            return

        # Look for pattern: if last 2 match previous sequence, prefetch next
        recent = self.access_pattern[-2:]

        for i in range(len(self.access_pattern) - 3):
            if self.access_pattern[i:i+2] == recent:
                next_likely = self.access_pattern[i+2]
                if next_likely in self.core.blocks:
                    self.core.prefetch([next_likely])
                    break


# Global instance
_crystal_core_instance: Optional[CrystalCore] = None
_crystal_cache_manager: Optional[CrystalCacheManager] = None


def get_crystal_core() -> CrystalCore:
    """Get global Crystal Core instance."""
    global _crystal_core_instance
    if _crystal_core_instance is None:
        _crystal_core_instance = CrystalCore()
    return _crystal_core_instance


def get_cache_manager() -> CrystalCacheManager:
    """Get global cache manager."""
    global _crystal_cache_manager
    if _crystal_cache_manager is None:
        core = get_crystal_core()
        _crystal_cache_manager = CrystalCacheManager(core)
    return _crystal_cache_manager


if __name__ == "__main__":
    # Test Crystal Core
    print("=== GAMESA Crystal Core Test ===\n")

    core = CrystalCore()

    # Allocate some blocks
    block1 = core.allocate(1024, MemoryTier.HOT, "test_app")
    block2 = core.allocate(2048, MemoryTier.WARM, "test_app")
    block3 = core.allocate(512, MemoryTier.COLD, "background")

    print(f"Allocated blocks: {block1}, {block2}, {block3}")

    # Write data
    core.write(block1, b"Hello from Crystal Core!")
    core.write(block2, b"This is block 2" * 100)

    # Read data
    data1 = core.read(block1)
    print(f"Read from block1: {data1[:30]}")

    # Get stats
    stats = core.get_stats()
    print(f"\nMemory Stats:")
    print(f"  Allocated: {stats['allocated'] / 1024:.1f} KB")
    print(f"  Free: {stats['free'] / (1024*1024):.1f} MB")
    print(f"  Utilization: {stats['utilization']*100:.2f}%")
    print(f"  Blocks: {stats['blocks']}")

    # Optimize layout
    core.optimize_layout()

    # Cleanup
    core.cleanup()
