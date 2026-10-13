// GENERATED from native/spec/binary_layouts.yaml - DO NOT EDIT
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KrResourceEvidenceEnvelope {
    pub magic: [u8; 4],
    pub abi_version: u16,
    pub flags: u16,
    pub state_version: u64,
    pub observed_ns: u64,
    pub lease_ttl_ns: u64,
    pub value: f64,
    pub uncertainty: f64,
    pub domain_id: u32,
    pub epistemic_kind: u8,
    pub temporal_mode: u8,
    pub ownership: u8,
    pub reserved0: u8,
    pub crc32: u32,
    pub reserved1: u32,
}

const _: () = assert!(core::mem::size_of::<KrResourceEvidenceEnvelope>() == 64);
const _: () = assert!(core::mem::align_of::<KrResourceEvidenceEnvelope>() == 64);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, magic) == 0);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, abi_version) == 4);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, flags) == 6);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, state_version) == 8);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, observed_ns) == 16);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, lease_ttl_ns) == 24);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, value) == 32);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, uncertainty) == 40);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, domain_id) == 48);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, epistemic_kind) == 52);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, temporal_mode) == 53);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, ownership) == 54);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, reserved0) == 55);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, crc32) == 56);
const _: () = assert!(core::mem::offset_of!(KrResourceEvidenceEnvelope, reserved1) == 60);
