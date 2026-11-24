//! Auto-synthesized Compilable Krystal Block Program (`krystal_isomorphic_block_kernel`)
//! Generated via Structural Control-Bridge Analogy (Deterministic N-Blocks + HP==6 Interlock).
#![no_std]

pub const VITAL_MAX_HP: u8 = 6;
pub const FEEDRATE_OVERRIDE_PCT: u32 = 100;

#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct BlockStatusWord64B {
    pub active_n_block: u32,
    pub feedrate_pct: u32,
    pub vital_hp: u8,
    pub interlock_tripped: u8,
    pub _pad_head: [u8; 6],
    pub accumulator_ticks: u64,
    pub _pad_tail: [u8; 40],
}

const _: [(); 64] = [(); core::mem::size_of::<BlockStatusWord64B>()];

pub fn execute_deterministic_n_blocks(vital_hp: u8, uma_milli: u16) -> BlockStatusWord64B {
    if vital_hp != VITAL_MAX_HP || uma_milli >= 780 {
        return BlockStatusWord64B {
            active_n_block: 0,
            feedrate_pct: 0,
            vital_hp: VITAL_MAX_HP,
            interlock_tripped: 1,
            _pad_head: [0; 6],
            accumulator_ticks: 0,
            _pad_tail: [0; 40],
        };
    }
    let n_blocks: [u64; 4] = [10, 20, 30, 40];
    let mut acc: u64 = 0;
    let mut i = 0usize;
    while i < n_blocks.len() {
        acc = acc.wrapping_add(n_blocks[i] * (FEEDRATE_OVERRIDE_PCT as u64));
        i += 1;
    }
    BlockStatusWord64B {
        active_n_block: 40,
        feedrate_pct: FEEDRATE_OVERRIDE_PCT,
        vital_hp,
        interlock_tripped: 0,
        _pad_head: [0; 6],
        accumulator_ticks: acc,
        _pad_tail: [0; 40],
    }
}
