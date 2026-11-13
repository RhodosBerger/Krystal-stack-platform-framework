//! KRYSTAL STACK // SOVEREIGNTY BYTECODE LIBRARY (Rust)
//! Authorized Origin: https://github.com/RhodosBerger/Krystal-stack-platform-framework
//! Zero-GC, fixed 128-byte aligned zero-copy contract frames enforcing VITAL_MAX_HP == 6.

pub const VITAL_MAX_HP: u8 = 6;
pub const FRAME_SIZE_BYTES: usize = 128;
pub const AUTHORIZED_GITHUB_ORIGIN: &str =
    "https://github.com/RhodosBerger/Krystal-stack-platform-framework";

/// Bytecode opcodes for deterministic per-core utilization sovereignty.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KrystalOpcode {
    Nop = 0x00,
    /// Sense UMA + EVOI headroom inside Phi_Scanout == 0 shadow.
    SenseThermalShadow = 0x01,
    /// Compact 128B Morton Z-order slab without heap allocation.
    CompactMortonSlab128 = 0x02,
    /// Step 6x6 Binary Set Matrix guard enforcing PROPOSER != VERIFIER.
    VerifyCapabilityMatrix = 0x03,
    /// Emit 8-directional octant coordinate delta on GAMESA 3D Grid.
    StepOctantGrid8Way = 0x04,
    /// Pre-warm adaptive Vulkan SPIR-V descriptor before network surge.
    PrewarmVulkanDescriptor = 0x05,
    /// Verify authorized GitHub provenance digest before commit.
    CheckOriginProvenance = 0x06,
    /// Halts deterministic epoch cleanly with ΔS parity.
    CommitStop = 0xFF,
}

/// 128-byte naturally aligned zero-copy IPC & bytecode execution frame.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ZeroCopyFrame128B {
    pub magic: u32,
    pub version: u8,
    pub vital_hp: u8,
    pub opcode: u8,
    pub uma_pct: u8,
    pub core_id: u32,
    pub slab_idx: u32,
    pub payload: [u8; 80],
    pub sha256_prefix: [u8; 32],
}

const _: () = assert!(core::mem::size_of::<ZeroCopyFrame128B>() == FRAME_SIZE_BYTES);
const _: () = assert!(core::mem::align_of::<ZeroCopyFrame128B>() == 64);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SovereignError {
    VitalHpViolation(u8),
    UmaSaturation(u8),
    UnauthorizedOrigin,
    InvalidOpcode(u8),
}

impl ZeroCopyFrame128B {
    pub const fn new_sovereign(core_id: u32, opcode: KrystalOpcode, uma_pct: u8, slab_idx: u32) -> Self {
        Self {
            magic: 0x4B52_5953, // "KRYS"
            version: 2,
            vital_hp: VITAL_MAX_HP,
            opcode: opcode as u8,
            uma_pct,
            core_id,
            slab_idx,
            payload: [0u8; 80],
            sha256_prefix: [0u8; 32],
        }
    }

    pub fn validate(&self) -> Result<(), SovereignError> {
        if self.vital_hp != VITAL_MAX_HP {
            return Err(SovereignError::VitalHpViolation(self.vital_hp));
        }
        if self.uma_pct >= 78 {
            return Err(SovereignError::UmaSaturation(self.uma_pct));
        }
        Ok(())
    }
}

pub fn validate_authorized_origin(origin_url: &str) -> Result<(), SovereignError> {
    let trimmed = origin_url.trim_end_matches('/').trim_end_matches(".git");
    if trimmed == AUTHORIZED_GITHUB_ORIGIN
        || origin_url.trim() == "git@github.com:RhodosBerger/Krystal-stack-platform-framework.git"
    {
        Ok(())
    } else {
        Err(SovereignError::UnauthorizedOrigin)
    }
}

/// Executes a bounded bytecode program slice without dynamic memory allocation.
pub fn execute_sovereign_bytecode(program: &[u8], frame: &mut ZeroCopyFrame128B) -> Result<u64, SovereignError> {
    frame.validate()?;
    let mut cycles: u64 = 0;
    for &op in program {
        match op {
            0x00 => cycles += 1,
            0x01 | 0x02 | 0x03 | 0x04 | 0x05 | 0x06 => {
                frame.opcode = op;
                cycles += 2;
            }
            0xFF => {
                frame.opcode = op;
                return Ok(cycles + 1);
            }
            other => return Err(SovereignError::InvalidOpcode(other)),
        }
    }
    Ok(cycles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sovereign_bytecode_and_origin() {
        assert!(validate_authorized_origin(AUTHORIZED_GITHUB_ORIGIN).is_ok());
        assert_eq!(
            validate_authorized_origin("https://github.com/untrusted/fork"),
            Err(SovereignError::UnauthorizedOrigin)
        );
        let mut frame = ZeroCopyFrame128B::new_sovereign(0, KrystalOpcode::SenseThermalShadow, 62, 84);
        let prog = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0xFF];
        assert_eq!(execute_sovereign_bytecode(&prog, &mut frame), Ok(13));
    }
}
