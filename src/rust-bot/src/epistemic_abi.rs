// ============================================================================
// KRYSTAL-STACK: CANONICAL RUST EPISTEMIC ABI
// Generated automatically from krystal_kernel.verification.semantic_protocol
// Constitutional Invariant: VITAL_MAX_HP == 6
// ============================================================================

#![allow(dead_code)]

pub const VITAL_MAX_HP: usize = 6;

pub const FAMILY_MASK: u16 = 0xF000;
pub const SUBKIND_MASK: u16 = 0x0F00;
pub const VALUE_MASK: u16 = 0x00FF;

#[inline]
pub const fn get_family(status: u16) -> u8 {
    ((status & FAMILY_MASK) >> 12) as u8
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EpistemicState {
    EpiUnknown                 = 0x1001,
    EpiObserved                = 0x1002,
    EpiDerived                 = 0x1003,
    EpiPredicted               = 0x1004,
    EpiCalibrated              = 0x1005,
    EpiValidated               = 0x1006,
    EpiFalsified               = 0x1007,
    EpiContradictory           = 0x1008,
    EpiQuarantined             = 0x1009,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TemporalState {
    TmpFresh                   = 0x2001,
    TmpLeased                  = 0x2002,
    TmpStale                   = 0x2003,
    TmpExpired                 = 0x2004,
    TmpMonotonicLatch          = 0x2005,
    TmpBistableHysteretic      = 0x2006,
    TmpShadowHypothetical      = 0x2007,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AuthorityState {
    AuthCommitApproved         = 0x3001,
    AuthDeferred               = 0x3002,
    AuthQuarantined            = 0x3003,
    AuthVetoed                 = 0x3004,
    AuthShadowOnly             = 0x3005,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SafetyState {
    SafePass                   = 0x4101,
    SafeBreach                 = 0x4102,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InvariantState {
    InvDepletedZero            = 0x4000,
    InvCriticalLock            = 0x4001,
    InvDegradedWarn            = 0x4003,
    InvNominal6hp              = 0x4006,
    InvBreachTripped           = 0x400E,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ActuationState {
    ActIdle                    = 0x5001,
    ActDispatched              = 0x5002,
    ActExecuted                = 0x5003,
    ActReverted                = 0x5004,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TransactionState {
    TxnIdle                    = 0x6001,
    TxnPrepared                = 0x6002,
    TxnVerified                = 0x6003,
    TxnCommitted               = 0x6004,
    TxnAborted                 = 0x6005,
    TxnCompensating            = 0x6006,
    TxnRolledBack              = 0x6007,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ObservationState {
    ObsRawSample               = 0x7001,
    ObsCalibratedSample        = 0x7002,
    ObsDriftDetected           = 0x7003,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivedState {
    DerComputedDelta           = 0x8001,
    DerPressureScore           = 0x8002,
    DerThermalHeadroom         = 0x8003,
    DerInterferenceIndex       = 0x8004,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PredictionState {
    PredPointEstimate          = 0x9001,
    PredConfidenceBound        = 0x9002,
    PredBreachWarning          = 0x9003,
    PredCounterfactualDelta    = 0x9004,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryState {
    MemArenaAlloc              = 0xA001,
    MemZeroCopyView            = 0xA002,
    MemQuarantined             = 0xA003,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum HardwareState {
    HwNominal                  = 0xB001,
    HwThrottledThermal         = 0xB002,
    HwRaplClamped              = 0xB003,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParadoxState {
    PdxDormant                 = 0xC001,
    PdxResonating              = 0xC002,
    PdxQuarantined             = 0xC003,
    PdxAutoHealed              = 0xC004,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MerkleState {
    MrkRootValid               = 0xD001,
    MrkLeafStale               = 0xD002,
    MrkHashMismatch            = 0xD003,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PolicyState {
    PolActive                  = 0xE001,
    PolCalibrating             = 0xE002,
    PolSuperseded              = 0xE003,
    PolRejected                = 0xE004,
    PolExperimental            = 0xE005,
}
