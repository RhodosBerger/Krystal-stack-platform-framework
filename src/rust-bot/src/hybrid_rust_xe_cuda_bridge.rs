//! Krystal Stack — Explicit Hybrid Rust Integration Structures & CUDA-Equivalent
//! Control Primitives for Intel Xe Stream Processors (Subslices, EUs, SIMD8/16/32, SLM).
//!
//! Constitutional Invariant: VITAL_MAX_HP == 6 (\neg Valid \implies \Delta S = 0).

#![allow(dead_code)]

use crate::epistemic_abi::{AuthorityState, EpistemicState, InvariantState, VITAL_MAX_HP};

pub const XE_IRIS_SUBSLICES: u32 = 6;
pub const XE_EUS_PER_SUBSLICE: u32 = 16;
pub const XE_TOTAL_EUS: u32 = XE_IRIS_SUBSLICES * XE_EUS_PER_SUBSLICE; // 96 EUs
pub const XE_FP32_LANES_PER_EU: u32 = 8;
pub const XE_TOTAL_FP32_STREAM_PROCESSORS: u32 = XE_TOTAL_EUS * XE_FP32_LANES_PER_EU; // 768 SPs
pub const XE_HW_THREADS_PER_EU: u32 = 7;
pub const XE_MAX_HW_THREADS: u32 = XE_TOTAL_EUS * XE_HW_THREADS_PER_EU; // 672 HW threads
pub const XE_SLM_BYTES_PER_SUBSLICE: u32 = 65_536; // 64 KB Shared Local Memory (= CUDA __shared__)
pub const XE_STANDARD_GRF_COUNT: u16 = 128; // 128 x 32B = 4 KB GRF per thread

/// CUDA `dim3` equivalent for Intel Xe Dispatch Grid & Workgroup sizing.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct XeDim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl XeDim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn volume(&self) -> u64 {
        (self.x as u64) * (self.y as u64) * (self.z as u64)
    }
}

/// Intel EU Subgroup SIMD width (equivalent to tunable CUDA Warp width 8 / 16 / 32).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum XeSimdSubgroupWidth {
    Simd8LowRegisterPressure = 8,
    Simd16BalancedDefault = 16,
    Simd32HighThroughput = 32,
}

/// Ownership tag explicitly declaring how Rust integrates with C11 MMIO, Linux DRM/DMA-BUF,
/// Intel Xe Stream Processors, and Python/XERJ orchestration.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HybridIntegrationOwner {
    RustMemorySafeCore = 1,
    C11LinuxWindowsMmioAbi = 2,
    IntelXeStreamProcessor = 3,
    OpenVinoOneApiVulkan = 4,
    PythonXerjOrchestrator = 5,
}

/// 128-byte cache-line-aligned Hybrid Rust <-> Intel Xe Stream Launch Descriptor
/// (`kernel<<<grid, block, slm_bytes, xe_stream>>>` in zero-copy shared memory).
#[repr(C, align(128))]
#[derive(Copy, Clone, Debug)]
pub struct HybridRustXeLaunchDescriptor128B {
    pub magic: u32,                   // 0..4:   0x5845_4355 ("XECU")
    pub stream_id: u16,               // 4..6:   XeStream queue index
    pub simd_width: u8,               // 6..7:   8, 16, or 32
    pub vital_max_hp: u8,             // 7..8:   Must equal 6
    pub grid_dim: XeDim3,             // 8..20:  Workgroup grid
    pub block_dim: XeDim3,            // 20..32: Local threads per workgroup
    pub slm_shared_bytes: u32,        // 32..36: Shared Local Memory (<= 65536 B)
    pub grf_registers_per_thread: u16,// 36..38: <= 128 standard, <= 256 large-GRF
    pub owner_plane: u8,              // 38..39: HybridIntegrationOwner
    pub linux_dmabuf_exported: u8,    // 39..40: 1 if exported via DRM/DMA-BUF fd
    pub usm_input_offset: u64,        // 40..48: 64B-aligned zero-copy source
    pub usm_output_offset: u64,       // 48..56: 64B-aligned zero-copy target
    pub authority_mask: u64,          // 56..64: kr_reduce_authority mask
    pub uma_util_milli: u16,          // 64..66: Veto GPU launch if >= 780
    pub epistemic_state: u16,         // 66..68: EpistemicState
    pub authority_state: u16,         // 68..70: AuthorityState
    pub invariant_state: u16,         // 70..72: InvariantState
    pub predicted_occupancy_permil: u16, // 72..74: 0..1000
    pub subgroup_tile_k: u16,         // 74..76: Must be < 449
    pub doorbell_fence_id: u32,       // 76..80: BAR0 completion fence
    pub reserved_tail: [u8; 48],      // 80..128: Pad to exact 128B
}

impl HybridRustXeLaunchDescriptor128B {
    /// CUDA-style occupancy & launch gate (`cudaOccupancyMaxActiveBlocks` + Krystal safety gate).
    pub fn validate_and_score_occupancy(&mut self) -> Result<u16, &'static str> {
        if self.vital_max_hp as usize != VITAL_MAX_HP {
            self.authority_state = AuthorityState::AuthVetoed as u16;
            return Err("ERR_XE_HP_VIOLATION_DELTA_S_ZERO");
        }
        if self.slm_shared_bytes > XE_SLM_BYTES_PER_SUBSLICE {
            self.authority_state = AuthorityState::AuthVetoed as u16;
            return Err("ERR_XE_SLM_EXCEEDS_64KB_SUBSLICE_LIMIT");
        }
        if self.subgroup_tile_k >= 449 {
            self.authority_state = AuthorityState::AuthVetoed as u16;
            return Err("ERR_XE_SUBGROUP_K_SPILL_LIMIT_449");
        }
        if self.uma_util_milli >= 780 {
            self.authority_state = AuthorityState::AuthVetoed as u16;
            return Err("ERR_XE_UMA_BUS_SATURATED_FALLBACK_CPU");
        }
        if (self.usm_input_offset & 63) != 0 || (self.usm_output_offset & 63) != 0 {
            self.authority_state = AuthorityState::AuthVetoed as u16;
            return Err("ERR_XE_USM_UNALIGNED_64B");
        }

        let max_threads_per_subslice = XE_EUS_PER_SUBSLICE * XE_HW_THREADS_PER_EU; // 112 HW threads
        let grf_factor = if self.grf_registers_per_thread <= XE_STANDARD_GRF_COUNT {
            1
        } else {
            2
        };
        let effective_hw_threads = max_threads_per_subslice / grf_factor;
        let wg_lanes = self.block_dim.volume() as u32;
        let simd = (self.simd_width as u32).max(8);
        let subgroups_per_wg = ((wg_lanes + simd - 1) / simd).max(1);

        let max_wg_by_threads = (effective_hw_threads / subgroups_per_wg).max(1);
        let max_wg_by_slm = if self.slm_shared_bytes == 0 {
            max_wg_by_threads
        } else {
            (XE_SLM_BYTES_PER_SUBSLICE / self.slm_shared_bytes).max(1)
        };
        let active_wg = if max_wg_by_threads < max_wg_by_slm {
            max_wg_by_threads
        } else {
            max_wg_by_slm
        };
        let active_threads = (active_wg * subgroups_per_wg).min(effective_hw_threads);
        let occ_permil = ((active_threads * 1000) / max_threads_per_subslice) as u16;

        self.predicted_occupancy_permil = occ_permil;
        self.epistemic_state = EpistemicState::EpiValidated as u16;
        self.authority_state = AuthorityState::AuthCommitApproved as u16;
        self.invariant_state = InvariantState::InvNominal6hp as u16;
        Ok(occ_permil)
    }
}
