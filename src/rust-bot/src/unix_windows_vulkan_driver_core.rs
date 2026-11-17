//! Krystal Stack — Unix/Windows Lightweight Rust + Vulkan Driver Core,
//! Strict O(1) Bitmap-64 Hardware-Ceiling Slab Arena, and Per-Thread Sharded
//! 128-Byte Priority Log Rings.
//!
//! Constitutional Invariant: VITAL_MAX_HP == 6 (\neg Valid \implies \Delta S = 0).

#![allow(dead_code)]

use crate::epistemic_abi::{AuthorityState, EpistemicState, InvariantState, VITAL_MAX_HP};

pub const KRYSTAL_CACHE_LINE_BYTES: usize = 64;
pub const KRYSTAL_SLOT_BYTES: usize = 128;
pub const UMA_SATURATION_MILLI: u16 = 780; // 0.780 constitutional ceiling
pub const SUBGROUP_REDUCTION_K_LIMIT: u32 = 449;
pub const THERMAL_SAFE_C_X10: u16 = 780;   // 78.0 C
pub const THERMAL_REGULATE_C_X10: u16 = 950; // 95.0 C
pub const THERMAL_HARD_TRIP_C_X10: u16 = 1000; // 100.0 C

/// Target OS kernel plane for community & core lightweight GPU/iGPU drivers.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DriverOsPlane {
    WindowsWddm32DxCore = 1,
    UnixDrmGemDmaBuf = 2,
}

/// Target silicon topology: Integrated UMA chip vs Discrete PCIe ReBAR GPU card.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuSiliconTopology {
    IntegratedUmaChip = 1,
    DiscretePcieCard = 2,
}

/// Priority-stratified log class matching P0/P1/P2 context compression rules.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LogPriorityClass {
    P0ConstitutionalNeverDrop = 0,
    P1ActiveMilestone = 1,
    P2DiagnosticSheddable = 2,
}

/// Naturally aligned 128-byte (2 x 64B cache lines) zero-copy kernel slot for
/// Vulkan IPC driver descriptors and lock-free sharded telemetry logging.
#[repr(C, align(128))]
#[derive(Copy, Clone, Debug)]
pub struct Kernel128BLogAndDriverSlot {
    pub magic: u32,                  // 0..4:   0x4B52_5953 ("KRYS")
    pub sequence_id: u32,            // 4..8:   Monotonic per-thread sequence
    pub timestamp_ns: u64,           // 8..16:  RDTSCP / CLOCK_MONOTONIC_RAW
    pub authority_mask: u64,         // 16..24: kr_reduce_authority result
    pub driver_plane: u8,            // 24..25: DriverOsPlane
    pub silicon_topology: u8,        // 25..26: GpuSiliconTopology
    pub log_priority: u8,            // 26..27: LogPriorityClass
    pub vital_max_hp: u8,            // 27..28: Must equal 6
    pub uma_util_milli: u16,         // 28..30: 0..1000 (veto GPU if >= 780)
    pub junction_temp_c_x10: u16,    // 30..32: 0.1 C units (hard trip >= 1000)
    pub epistemic_state: u16,        // 32..34: EpistemicState u16
    pub authority_state: u16,        // 34..36: AuthorityState u16
    pub invariant_state: u16,        // 36..38: InvariantState u16
    pub active_threads: u16,         // 38..40: Dynamic physical thread count (1..4)
    pub reduction_k: u32,            // 40..44: Auto-tile if >= 449
    pub thrashing_index_milli: u32,  // 44..48: T_thrash * 1000
    pub adjusted_quantum_ns: u64,    // 48..56: Q_adj
    pub device_pci_id: u32,          // 56..60: e.g. 0x80869A49
    pub vulkan_queue_family: u32,    // 60..64: Compute/transfer queue family
    pub payload_digest_head: [u8; 32], // 64..96: 256-bit provenance digest
    pub reserved_pad: [u8; 32],      // 96..128: Explicit zero-pad to 128B
}

impl Kernel128BLogAndDriverSlot {
    pub const fn empty() -> Self {
        Self {
            magic: 0x4B52_5953,
            sequence_id: 0,
            timestamp_ns: 0,
            authority_mask: 0,
            driver_plane: DriverOsPlane::WindowsWddm32DxCore as u8,
            silicon_topology: GpuSiliconTopology::IntegratedUmaChip as u8,
            log_priority: LogPriorityClass::P1ActiveMilestone as u8,
            vital_max_hp: VITAL_MAX_HP as u8,
            uma_util_milli: 0,
            junction_temp_c_x10: 450,
            epistemic_state: EpistemicState::EpiObserved as u16,
            authority_state: AuthorityState::AuthShadowOnly as u16,
            invariant_state: InvariantState::InvNominal6hp as u16,
            active_threads: 4,
            reduction_k: 256,
            thrashing_index_milli: 500,
            adjusted_quantum_ns: 4_000_000,
            device_pci_id: 0x8086_9A49,
            vulkan_queue_family: 0,
            payload_digest_head: [0u8; 32],
            reserved_pad: [0u8; 32],
        }
    }
}

/// Bitwise conjunctive capability reduction mirroring include/kr_authority_bits.h.
#[inline]
pub const fn reduce_driver_authority(
    requested: u64,
    thermal_safe: u64,
    battery_safe: u64,
    capability_supported: u64,
    evidence_supported: u64,
    policy_allowed: u64,
) -> u64 {
    requested
        & thermal_safe
        & battery_safe
        & capability_supported
        & evidence_supported
        & policy_allowed
}

/// Strict O(1) Bitmap-64 Slab Allocator replacing BTreeMap/HashMap heap pointer-chasing
/// to expose deterministic hardware limits without OS page-fault or allocator jitter.
pub struct StrictBitmap64SlabArena {
    free_bitmap: u64, // 1 = free, 0 = allocated
    slots: [Kernel128BLogAndDriverSlot; 64],
    rejected_noop_count: u64,
}

impl StrictBitmap64SlabArena {
    pub const fn new_pinned() -> Self {
        Self {
            free_bitmap: u64::MAX,
            slots: [Kernel128BLogAndDriverSlot::empty(); 64],
            rejected_noop_count: 0,
        }
    }

    /// O(1) hardware TZCNT allocation with constitutional gate (\neg Valid \implies \Delta S = 0).
    #[inline]
    pub fn allocate_strict(
        &mut self,
        vital_hp: usize,
        uma_util_milli: u16,
        junction_temp_c_x10: u16,
    ) -> Result<usize, &'static str> {
        if vital_hp != VITAL_MAX_HP {
            self.rejected_noop_count += 1;
            return Err("ERR_CONSTITUTIONAL_HP_VIOLATION_DELTA_S_ZERO");
        }
        if junction_temp_c_x10 >= THERMAL_HARD_TRIP_C_X10 {
            self.rejected_noop_count += 1;
            return Err("ERR_THERMAL_HARD_TRIP_100C_DELTA_S_ZERO");
        }
        if self.free_bitmap == 0 {
            self.rejected_noop_count += 1;
            return Err("ERR_STRICT_ARENA_CEILING_REACHED_DELTA_S_ZERO");
        }
        let idx = self.free_bitmap.trailing_zeros() as usize;
        self.free_bitmap &= !(1u64 << idx);
        self.slots[idx].uma_util_milli = uma_util_milli;
        self.slots[idx].junction_temp_c_x10 = junction_temp_c_x10;
        self.slots[idx].authority_state = if uma_util_milli < UMA_SATURATION_MILLI {
            AuthorityState::AuthCommitApproved as u16
        } else {
            AuthorityState::AuthVetoed as u16
        };
        Ok(idx)
    }

    #[inline]
    pub fn release_strict(&mut self, idx: usize) -> Result<(), &'static str> {
        if idx >= 64 {
            return Err("ERR_SLOT_OUT_OF_RANGE");
        }
        self.free_bitmap |= 1u64 << idx;
        Ok(())
    }

    #[inline]
    pub const fn free_slots(&self) -> u32 {
        self.free_bitmap.count_ones()
    }
}

/// Per-thread lock-free SPSC ring shard for dynamic threading over the log system.
/// Eliminates global mutex contention while shedding P2 logs under UMA/thrash pressure.
pub struct PerThreadShardedLogRing {
    pub worker_thread_id: u16,
    pub write_head: usize,
    pub dropped_p2_shed_count: u64,
    pub committed_count: u64,
    pub ring: [Kernel128BLogAndDriverSlot; 64],
}

impl PerThreadShardedLogRing {
    pub const fn new(worker_thread_id: u16) -> Self {
        Self {
            worker_thread_id,
            write_head: 0,
            dropped_p2_shed_count: 0,
            committed_count: 0,
            ring: [Kernel128BLogAndDriverSlot::empty(); 64],
        }
    }

    /// Appends a 128B binary log frame without cross-thread locks; sheds P2 if saturated.
    #[inline]
    pub fn push_log_frame(
        &mut self,
        priority: LogPriorityClass,
        uma_util_milli: u16,
        thrashing_index_milli: u32,
        timestamp_ns: u64,
    ) -> bool {
        if matches!(priority, LogPriorityClass::P2DiagnosticSheddable)
            && (uma_util_milli >= UMA_SATURATION_MILLI || thrashing_index_milli > 1200)
        {
            self.dropped_p2_shed_count += 1;
            return false;
        }
        let slot_idx = self.write_head & 63;
        let slot = &mut self.ring[slot_idx];
        slot.sequence_id = self.committed_count as u32;
        slot.timestamp_ns = timestamp_ns;
        slot.log_priority = priority as u8;
        slot.uma_util_milli = uma_util_milli;
        slot.thrashing_index_milli = thrashing_index_milli;
        self.write_head = self.write_head.wrapping_add(1);
        self.committed_count += 1;
        true
    }
}
