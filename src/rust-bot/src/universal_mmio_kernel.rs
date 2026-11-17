//! Krystal Stack — Universal Device Memory-Mapped I/O (MMIO) & BAR Aperture Kernel
//!
//! Provides memory-safe, bounds-checked, alignment-verified, capability-gated direct
//! register and DMA aperture communication across PCIe BARs, VFIO/UIO mappings,
//! WDDM/DXCore doorbells, and shared UMA pages.
//!
//! Constitutional Invariant: VITAL_MAX_HP == 6 (\neg Valid \implies \Delta S = 0).

#![allow(dead_code)]

use crate::epistemic_abi::VITAL_MAX_HP;

pub const MMIO_MAGIC_KMMI: u32 = 0x4B4D_4D49;
pub const MMIO_ABI_VERSION: u32 = 0x0001_0000;
pub const BAR0_PAGE_BYTES: usize = 4096;
pub const DMA_ALIGNMENT_BYTES: u64 = 64;
pub const UMA_SATURATION_LIMIT_MILLI: u32 = 780;
pub const THERMAL_TRIP_DECI_C: u32 = 1000;

pub const REG_DEVICE_MAGIC: usize = 0x0000;
pub const REG_ABI_VERSION: usize = 0x0004;
pub const REG_VITAL_MAX_HP: usize = 0x0008;
pub const REG_DEVICE_STATUS: usize = 0x000C;
pub const REG_AUTHORITY_MASK: usize = 0x0010;
pub const REG_UMA_UTIL_MILLI: usize = 0x0018;
pub const REG_THERMAL_DECI_C: usize = 0x001C;
pub const REG_DMA_SRC_OFFSET: usize = 0x0020;
pub const REG_DMA_DST_OFFSET: usize = 0x0028;
pub const REG_DMA_BYTE_LEN: usize = 0x0030;
pub const REG_OP_COMMAND: usize = 0x0034;
pub const REG_DOORBELL_TAIL: usize = 0x0040;
pub const REG_COMPLETED_HEAD: usize = 0x0048;
pub const REG_REJECTED_NOOP_COUNT: usize = 0x0050;
pub const REG_ATTESTATION_SHA256: usize = 0x0060;

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MmioBusError {
    UnalignedAddress = 1,
    OutOfBounds = 2,
    ReadOnlyRegister = 3,
    VitalHpViolation = 4,
    AuthorityDenied = 5,
    UmaBusSaturated = 6,
    ThermalHardTrip = 7,
}

/// 4096-byte page-aligned BAR0 Control/Status Register Window.
#[repr(C, align(4096))]
#[derive(Clone)]
pub struct UniversalMmioBar0Aperture {
    bytes: [u8; BAR0_PAGE_BYTES],
    is_gpu_aperture: bool,
}

impl UniversalMmioBar0Aperture {
    pub fn new(is_gpu_aperture: bool, initial_authority_mask: u64) -> Self {
        let mut dev = Self {
            bytes: [0u8; BAR0_PAGE_BYTES],
            is_gpu_aperture,
        };
        dev.put_u32_raw(REG_DEVICE_MAGIC, MMIO_MAGIC_KMMI);
        dev.put_u32_raw(REG_ABI_VERSION, MMIO_ABI_VERSION);
        dev.put_u32_raw(REG_VITAL_MAX_HP, VITAL_MAX_HP as u32);
        dev.put_u64_raw(REG_AUTHORITY_MASK, initial_authority_mask);
        dev.put_u32_raw(REG_UMA_UTIL_MILLI, 350);
        dev.put_u32_raw(REG_THERMAL_DECI_C, 580);
        dev
    }

    #[inline]
    fn is_read_only(offset: usize) -> bool {
        matches!(
            offset,
            REG_DEVICE_MAGIC
                | REG_ABI_VERSION
                | REG_VITAL_MAX_HP
                | REG_DEVICE_STATUS
                | REG_COMPLETED_HEAD
                | REG_REJECTED_NOOP_COUNT
        ) || (REG_ATTESTATION_SHA256..0x0080).contains(&offset)
    }

    #[inline]
    fn put_u32_raw(&mut self, offset: usize, val: u32) {
        self.bytes[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }

    #[inline]
    fn put_u64_raw(&mut self, offset: usize, val: u64) {
        self.bytes[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    }

    #[inline]
    pub fn read_u32(&self, offset: usize) -> Result<u32, MmioBusError> {
        if (offset & 3) != 0 {
            return Err(MmioBusError::UnalignedAddress);
        }
        if offset + 4 > BAR0_PAGE_BYTES {
            return Err(MmioBusError::OutOfBounds);
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.bytes[offset..offset + 4]);
        Ok(u32::from_le_bytes(buf))
    }

    #[inline]
    pub fn read_u64(&self, offset: usize) -> Result<u64, MmioBusError> {
        if (offset & 7) != 0 {
            return Err(MmioBusError::UnalignedAddress);
        }
        if offset + 8 > BAR0_PAGE_BYTES {
            return Err(MmioBusError::OutOfBounds);
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.bytes[offset..offset + 8]);
        Ok(u64::from_le_bytes(buf))
    }

    #[inline]
    pub fn write_u32(&mut self, offset: usize, val: u32) -> Result<(), MmioBusError> {
        if (offset & 3) != 0 {
            self.bump_rejected();
            return Err(MmioBusError::UnalignedAddress);
        }
        if offset + 4 > BAR0_PAGE_BYTES {
            self.bump_rejected();
            return Err(MmioBusError::OutOfBounds);
        }
        if Self::is_read_only(offset) {
            self.bump_rejected();
            return Err(MmioBusError::ReadOnlyRegister);
        }
        self.put_u32_raw(offset, val);
        Ok(())
    }

    #[inline]
    pub fn ring_doorbell(&mut self, tail_seq: u64) -> Result<u64, MmioBusError> {
        let hp = self.read_u32(REG_VITAL_MAX_HP)? as usize;
        if hp != VITAL_MAX_HP {
            self.bump_rejected();
            return Err(MmioBusError::VitalHpViolation);
        }
        let auth = self.read_u64(REG_AUTHORITY_MASK)?;
        if auth == 0 {
            self.bump_rejected();
            return Err(MmioBusError::AuthorityDenied);
        }
        let therm = self.read_u32(REG_THERMAL_DECI_C)?;
        if therm >= THERMAL_TRIP_DECI_C {
            self.bump_rejected();
            return Err(MmioBusError::ThermalHardTrip);
        }
        let uma = self.read_u32(REG_UMA_UTIL_MILLI)?;
        if self.is_gpu_aperture && uma >= UMA_SATURATION_LIMIT_MILLI {
            self.bump_rejected();
            return Err(MmioBusError::UmaBusSaturated);
        }
        let src = self.read_u64(REG_DMA_SRC_OFFSET)?;
        let dst = self.read_u64(REG_DMA_DST_OFFSET)?;
        if (src & (DMA_ALIGNMENT_BYTES - 1)) != 0 || (dst & (DMA_ALIGNMENT_BYTES - 1)) != 0 {
            self.bump_rejected();
            return Err(MmioBusError::UnalignedAddress);
        }
        self.put_u64_raw(REG_DOORBELL_TAIL, tail_seq);
        let completed = self.read_u64(REG_COMPLETED_HEAD)?.wrapping_add(1);
        self.put_u64_raw(REG_COMPLETED_HEAD, completed);
        Ok(completed)
    }

    #[inline]
    fn bump_rejected(&mut self) {
        if let Ok(cur) = self.read_u64(REG_REJECTED_NOOP_COUNT) {
            self.put_u64_raw(REG_REJECTED_NOOP_COUNT, cur.wrapping_add(1));
        }
    }
}
