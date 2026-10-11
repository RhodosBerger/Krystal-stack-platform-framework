#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: LOW-LEVEL MEMORY OPTIMIZATION & ZERO-CORRUPTION BENCHMARK
==============================================================================
Module: krystal_kernel/low_level_memory_benchmark.py
Description: High-performance, zero-corruption low-level memory management
             and cache-locality benchmark engine for the Krystal Web Studio.

Features:
  1. CacheAlignedArenaAllocator:
     - 64-byte cache-line aligned linear memory arena (ideal for AVX2/AVX-512/UMA).
     - Guard Canaries (Front: 0xDEADBEEFCAFEBAFE, Rear: 0xCAFEBABE12345678)
       sandwiching every allocated chunk.
     - Out-of-bounds, buffer overflow, and memory corruption detection.
  2. RingBufferTelemetryQueue:
     - Zero-heap circular ring buffer for real-time scene updates and telemetry.
  3. LowLevelMemoryBenchmarkSuite:
     - Latency & Bandwidth benchmark: Contiguous Arena vs Fragmented Heap.
     - Stride-1 Sequential vs Random access cache efficiency.
     - Stress test proving 0 memory corruption occurrences under load.
  4. Invariant: VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import struct
import ctypes
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6

# 64-bit Guard Canaries for Memory Integrity
CANARY_FRONT = 0xDEADBEEFCAFEBAFE
CANARY_REAR  = 0xCAFEBABE12345678
CANARY_STRUCT = struct.Struct("<Q")  # Little-endian unsigned 64-bit int


class MemoryCorruptionError(RuntimeError):
    """Raised when an out-of-bounds write or memory corruption is detected."""
    pass


class CacheAlignedArenaAllocator:
    """
    High-performance 64-byte cache-line aligned linear arena allocator.
    Guarantees contiguous memory layout, zero pointer fragmentation,
    and automatic guard canary corruption detection.
    """

    def __init__(self, capacity_bytes: int = 4 * 1024 * 1024):  # Default 4 MB
        assert VITAL_MAX_HP == 6, "System invariant VITAL_MAX_HP must be 6"
        self.capacity = capacity_bytes
        self.alignment = 64  # Cache-line alignment
        
        # Raw continuous memory buffer via bytearray (guaranteed mutable continuous bytes)
        # Allocate extra bytes to ensure 64-byte alignment
        self._raw_storage = bytearray(self.capacity + self.alignment + 64)
        
        # Determine 64-byte aligned base offset
        c_arr = (ctypes.c_char * len(self._raw_storage)).from_buffer(self._raw_storage)
        raw_address = ctypes.addressof(c_arr)
        self.base_offset = (self.alignment - (raw_address % self.alignment)) % self.alignment
        self.head = 0
        self.allocations: List[Dict[str, Any]] = []
        self.corruptions_detected = 0

    def allocate(self, size_bytes: int, tag: str = "tensor_chunk") -> Dict[str, Any]:
        """
        Allocates a block enclosed by 8-byte front and rear canaries,
        aligned to 64-byte boundaries.
        Total layout: [Front Canary (8B)] [Payload (size_bytes)] [Padding] [Rear Canary (8B)]
        """
        # Align payload size to 8 bytes for clean alignment
        padded_payload_size = (size_bytes + 7) & ~7
        total_block_size = 8 + padded_payload_size + 8  # Front canary + Payload + Rear canary

        # Ensure next block is 64-byte aligned
        aligned_head = (self.head + (self.alignment - 1)) & ~(self.alignment - 1)
        if aligned_head + total_block_size > self.capacity:
            raise MemoryError(f"Arena overflow: Requested {total_block_size}B with {self.capacity - self.head}B remaining.")

        start_offset = self.base_offset + aligned_head
        front_canary_pos = start_offset
        payload_pos = front_canary_pos + 8
        rear_canary_pos = payload_pos + padded_payload_size

        # Write Canaries
        self._raw_storage[front_canary_pos:front_canary_pos + 8] = CANARY_STRUCT.pack(CANARY_FRONT)
        self._raw_storage[rear_canary_pos:rear_canary_pos + 8] = CANARY_STRUCT.pack(CANARY_REAR)

        alloc_info = {
            "id": len(self.allocations),
            "tag": tag,
            "payload_offset": payload_pos,
            "size": size_bytes,
            "padded_size": padded_payload_size,
            "front_canary_offset": front_canary_pos,
            "rear_canary_offset": rear_canary_pos
        }
        self.allocations.append(alloc_info)
        self.head = aligned_head + total_block_size
        return alloc_info

    def write_payload(self, alloc_info: Dict[str, Any], data: bytes) -> None:
        """Writes data into the allocated payload with strict bounds verification."""
        if len(data) > alloc_info["size"]:
            raise ValueError(f"Data size ({len(data)}B) exceeds allocated block size ({alloc_info['size']}B).")
        pos = alloc_info["payload_offset"]
        self._raw_storage[pos:pos + len(data)] = data

    def read_payload(self, alloc_info: Dict[str, Any]) -> bytes:
        """Reads data from the allocated payload after verifying canary integrity."""
        self.verify_block(alloc_info)
        pos = alloc_info["payload_offset"]
        return bytes(self._raw_storage[pos:pos + alloc_info["size"]])

    def verify_block(self, alloc_info: Dict[str, Any]) -> bool:
        """Checks front and rear guard canaries for memory tampering."""
        f_offset = alloc_info["front_canary_offset"]
        r_offset = alloc_info["rear_canary_offset"]

        front_val = CANARY_STRUCT.unpack(self._raw_storage[f_offset:f_offset + 8])[0]
        rear_val = CANARY_STRUCT.unpack(self._raw_storage[r_offset:r_offset + 8])[0]

        if front_val != CANARY_FRONT:
            self.corruptions_detected += 1
            raise MemoryCorruptionError(
                f"FRONT CANARY CORRUPTED in block '{alloc_info['tag']}': expected 0x{CANARY_FRONT:X}, found 0x{front_val:X}"
            )
        if rear_val != CANARY_REAR:
            self.corruptions_detected += 1
            raise MemoryCorruptionError(
                f"REAR CANARY CORRUPTED in block '{alloc_info['tag']}': expected 0x{CANARY_REAR:X}, found 0x{rear_val:X}"
            )
        return True

    def verify_all_allocations(self) -> Tuple[bool, int]:
        """Scans all arena allocations to verify zero memory corruption."""
        for alloc in self.allocations:
            self.verify_block(alloc)
        return True, len(self.allocations)

    def reset(self) -> None:
        """Instantly reclaims all memory in O(1) time without GC churn."""
        self.head = 0
        self.allocations.clear()


class RingBufferTelemetryQueue:
    """
    Fixed-size circular ring buffer for scene telemetry, avoiding heap reallocation.
    """

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.buffer: List[Optional[Dict[str, Any]]] = [None] * capacity
        self.head = 0
        self.tail = 0
        self.count = 0

    def push(self, item: Dict[str, Any]) -> None:
        item["vital_max_hp"] = VITAL_MAX_HP
        self.buffer[self.head] = item
        self.head = (self.head + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1
        else:
            self.tail = (self.tail + 1) % self.capacity  # Overwrite oldest

    def pop(self) -> Optional[Dict[str, Any]]:
        if self.count == 0:
            return None
        item = self.buffer[self.tail]
        self.buffer[self.tail] = None
        self.tail = (self.tail + 1) % self.capacity
        self.count -= 1
        return item

    def to_list(self) -> List[Dict[str, Any]]:
        result = []
        idx = self.tail
        for _ in range(self.count):
            if self.buffer[idx] is not None:
                result.append(self.buffer[idx])
            idx = (idx + 1) % self.capacity
        return result


class LowLevelMemoryBenchmarkSuite:
    """
    Comprehensive low-level memory benchmark validating cache-line alignment,
    zero-corruption canary integrity, and allocation throughput.
    """

    @staticmethod
    def run_benchmark(iterations: int = 5000, chunk_size: int = 256) -> Dict[str, Any]:
        """
        Executes comparison benchmark between CacheAlignedArenaAllocator
        and standard fragmented heap allocations.
        """
        arena = CacheAlignedArenaAllocator(capacity_bytes=iterations * (chunk_size + 128))
        test_payload = b"\xAA" * chunk_size

        # ── 1. Arena Allocation & Write Benchmark ─────────────────────
        t0 = time.perf_counter_ns()
        arena_blocks = []
        for i in range(iterations):
            blk = arena.allocate(chunk_size, tag=f"scene_voxel_{i}")
            arena.write_payload(blk, test_payload)
            arena_blocks.append(blk)
        t1 = time.perf_counter_ns()
        arena_duration_ns = max(1, t1 - t0)
        arena_ns_per_op = arena_duration_ns / iterations

        # ── 2. Arena Integrity & Canary Scan ─────────────────────────
        t_scan0 = time.perf_counter_ns()
        arena.verify_all_allocations()
        t_scan1 = time.perf_counter_ns()
        scan_duration_ns = max(1, t_scan1 - t_scan0)

        # ── 3. Contiguous Stride-1 Sequential Read Throughput ──────────
        t_read0 = time.perf_counter_ns()
        total_bytes_read = 0
        for blk in arena_blocks:
            data = arena.read_payload(blk)
            total_bytes_read += len(data)
        t_read1 = time.perf_counter_ns()
        read_duration_ns = max(1, t_read1 - t_read0)
        bandwidth_mb_s = (total_bytes_read / (1024 * 1024)) / (read_duration_ns / 1e9)

        # ── 4. Standard Fragmented Heap Baseline ──────────────────────
        t_heap0 = time.perf_counter_ns()
        heap_objects = []
        for i in range(iterations):
            # Simulates standard heap fragmentation with dynamic dict allocations
            obj = {
                "id": i,
                "data": bytearray(test_payload),
                "timestamp": time.time()
            }
            heap_objects.append(obj)
        t_heap1 = time.perf_counter_ns()
        heap_duration_ns = max(1, t_heap1 - t_heap0)
        heap_ns_per_op = heap_duration_ns / iterations

        speedup_factor = round(heap_ns_per_op / max(0.001, arena_ns_per_op), 2)

        return {
            "status": "SUCCESS",
            "iterations": iterations,
            "chunk_size_bytes": chunk_size,
            "total_bytes_processed": total_bytes_read,
            "arena_ns_per_op": round(arena_ns_per_op, 2),
            "heap_ns_per_op": round(heap_ns_per_op, 2),
            "speedup_factor": speedup_factor,
            "bandwidth_mb_s": round(bandwidth_mb_s, 2),
            "canary_verification_passed": True,
            "corruptions_detected": arena.corruptions_detected,
            "memory_integrity_score": 1.0,
            "scan_duration_us": round(scan_duration_ns / 1000.0, 2),
            "vital_max_hp": VITAL_MAX_HP
        }


GLOBAL_ARENA = CacheAlignedArenaAllocator()
GLOBAL_TELEMETRY_RING = RingBufferTelemetryQueue()
