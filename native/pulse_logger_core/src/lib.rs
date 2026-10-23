//! ==============================================================================
//! KRYSTAL-STACK: NATIVE PULSE SEQUENCE LOGGER (MODULE H-07 / E-07)
//! ==============================================================================
//! Hot-Path: Zero-allocation, zero-lock, no syscalls, no JSON formatting.
//! Invariant: VITAL_MAX_HP == 6
//! Hardware Target: Intel Tiger Lake AVX-512 / Iris Xe (80 EUs Gen12)
//! ==============================================================================

#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::sync::atomic::{AtomicU64, Ordering};

pub const VITAL_MAX_HP: u8 = 6;
pub const PULSE_BATCH_SIZE: usize = 16;
pub const PULSE_RING_DEFAULT_CAPACITY: usize = 65536;

/// Compact 8-byte pulse event frame with C ABI representation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PulseFrame {
    pub delta_ticks: u32,
    pub event_code: u8,
    pub invariant: u8,
    pub payload: u16,
}

// Compile-time static assertions ensuring exact 8-byte frame size and alignment
const _: () = assert!(core::mem::size_of::<PulseFrame>() == 8);
const _: () = assert!(core::mem::align_of::<PulseFrame>() == 8 || core::mem::align_of::<PulseFrame>() == 4 || core::mem::align_of::<PulseFrame>() == 2);

/// Fast inline read of CPU Time Stamp Counter (TSC).
#[inline(always)]
pub fn read_counter() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// SPSC Lock-free Ring Buffer layout with cache-line separation.
#[repr(C)]
pub struct PulseRing {
    // Producer cacheline (hot write path)
    pub write_index: AtomicU64,
    pub epoch_ticks: u64,
    pub capacity_mask: u32,
    pub batch_count: u32,
    pub batch_buffer: [PulseFrame; PULSE_BATCH_SIZE],

    // Consumer cacheline (cold read path)
    pub read_index: AtomicU64,

    // Frames storage pointer
    pub frames: *mut PulseFrame,
}

impl PulseRing {
    /// Initialize a pulse ring with a raw buffer and capacity (must be a power of two).
    pub fn init(&mut self, buffer: *mut PulseFrame, capacity: u32) -> Result<(), &'static str> {
        if buffer.is_null() || capacity == 0 || (capacity & (capacity - 1)) != 0 {
            return Err("Invalid buffer or capacity must be power of two");
        }

        self.write_index.store(0, Ordering::Relaxed);
        self.read_index.store(0, Ordering::Relaxed);
        self.epoch_ticks = read_counter();
        self.capacity_mask = capacity - 1;
        self.batch_count = 0;
        self.frames = buffer;

        Ok(())
    }

    /// Hot-Path H1: Ordinary aligned store with release publication.
    #[inline(always)]
    pub fn emit_h1_ordinary(&mut self, event_code: u8, payload: u16) {
        let ticks = read_counter();
        let delta = ticks.wrapping_sub(self.epoch_ticks) as u32;
        let w = self.write_index.load(Ordering::Relaxed);
        let idx = (w & (self.capacity_mask as u64)) as usize;

        let frame = PulseFrame {
            delta_ticks: delta,
            event_code,
            invariant: VITAL_MAX_HP,
            payload,
        };

        unsafe {
            core::ptr::write(self.frames.add(idx), frame);
        }

        self.write_index.store(w + 1, Ordering::Release);
    }

    /// Hot-Path H2: Non-temporal store (MOVNTI) bypassing cache.
    #[inline(always)]
    pub fn emit_h2_nontemporal(&mut self, event_code: u8, payload: u16) {
        let ticks = read_counter();
        let delta = ticks.wrapping_sub(self.epoch_ticks) as u32;
        let w = self.write_index.load(Ordering::Relaxed);
        let idx = (w & (self.capacity_mask as u64)) as usize;

        let frame = PulseFrame {
            delta_ticks: delta,
            event_code,
            invariant: VITAL_MAX_HP,
            payload,
        };

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let val = core::mem::transmute::<PulseFrame, i64>(frame);
            core::arch::x86_64::_mm_stream_si64(self.frames.add(idx) as *mut i64, val);
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            core::ptr::write(self.frames.add(idx), frame);
        }

        self.write_index.store(w + 1, Ordering::Release);
    }

    /// Hot-Path H3: 16-frame batch publication.
    #[inline(always)]
    pub fn emit_h3_batch16(&mut self, event_code: u8, payload: u16) {
        let ticks = read_counter();
        let delta = ticks.wrapping_sub(self.epoch_ticks) as u32;
        let cnt = self.batch_count as usize;

        self.batch_buffer[cnt] = PulseFrame {
            delta_ticks: delta,
            event_code,
            invariant: VITAL_MAX_HP,
            payload,
        };

        if cnt + 1 == PULSE_BATCH_SIZE {
            let w = self.write_index.load(Ordering::Relaxed);
            let mask = self.capacity_mask as u64;

            unsafe {
                for i in 0..PULSE_BATCH_SIZE {
                    let idx = ((w + i as u64) & mask) as usize;
                    core::ptr::write(self.frames.add(idx), self.batch_buffer[i]);
                }
            }

            self.write_index.store(w + PULSE_BATCH_SIZE as u64, Ordering::Release);
            self.batch_count = 0;
        } else {
            self.batch_count = (cnt + 1) as u32;
        }
    }

    /// Cold-Path Read: Read one frame from the ring.
    pub fn read_frame(&mut self) -> Option<PulseFrame> {
        let r = self.read_index.load(Ordering::Relaxed);
        let w = self.write_index.load(Ordering::Acquire);

        if r >= w {
            return None;
        }

        let idx = (r & (self.capacity_mask as u64)) as usize;
        let frame = unsafe { core::ptr::read(self.frames.add(idx)) };

        self.read_index.store(r + 1, Ordering::Release);
        Some(frame)
    }
}
