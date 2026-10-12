//! Cinematic Bridge - Native MotionScan KBSB v2 Codec, Deception Detector, Mortar Trajectory & Whisper Acoustic Engine
//!
//! Provides zero-copy 64-byte, 32-byte, and 16-byte binary packet encoding/decoding,
//! compile-time precomputed IEEE 802.3 CRC32 lookup tables, L.A. Noire deception classification,
//! acoustic formant whisper prediction, and high-performance ballistic trajectory calculations.

use serde::{Deserialize, Serialize};

pub const KMCP_MAGIC: [u8; 4] = *b"KMCP";
pub const KMCP_FRAME_SIZE: usize = 64;

pub const KMBP_MAGIC: [u8; 4] = *b"KMBP";
pub const KMBP_FRAME_SIZE: usize = 32;

pub const KDVP_MAGIC: [u8; 4] = *b"KDVP";
pub const KDVP_FRAME_SIZE: usize = 16;

pub const VITAL_MAX_HP: u8 = 6;

/// Precomputed compile-time IEEE 802.3 CRC32 lookup table for 8x throughput
pub const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// High-performance table-driven IEEE 802.3 CRC32 calculation
#[inline(always)]
pub fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        let idx = ((crc ^ (byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    !crc
}

#[derive(Debug, thiserror::Error)]
pub enum CinematicBridgeError {
    #[error("Invalid frame size: expected {expected}, got {got}")]
    InvalidFrameSize { expected: usize, got: usize },
    #[error("Invalid magic header: expected {expected:?}, got {got:?}")]
    InvalidMagic { expected: [u8; 4], got: [u8; 4] },
    #[error("Vital Max HP invariant breach: {0} > 6")]
    InvariantBreach(u8),
    #[error("CRC32 checksum mismatch: expected {expected:#x}, got {computed:#x}")]
    CrcMismatch { expected: u32, computed: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VisemeId {
    Rest = 0,
    Aa = 1,
    Ee = 2,
    Oo = 3,
    U = 4,
    Mbp = 5,
    Fv = 6,
    Th = 7,
    Ss = 8,
}

impl VisemeId {
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => VisemeId::Aa,
            2 => VisemeId::Ee,
            3 => VisemeId::Oo,
            4 => VisemeId::U,
            5 => VisemeId::Mbp,
            6 => VisemeId::Fv,
            7 => VisemeId::Th,
            8 => VisemeId::Ss,
            _ => VisemeId::Rest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EmotionId {
    TruthSteady = 0,
    DoubtEvasive = 1,
    LieDeceptive = 2,
    ShockImpact = 3,
}

impl EmotionId {
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => EmotionId::DoubtEvasive,
            2 => EmotionId::LieDeceptive,
            3 => EmotionId::ShockImpact,
            _ => EmotionId::TruthSteady,
        }
    }
}

/// 64-Byte Deterministic Motion Capture Packet (KMCP)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KBSBMotionScanPacket {
    pub timestamp_ms: u32,
    pub viseme: VisemeId,
    pub emotion: EmotionId,
    pub lod_level: u8,
    pub vital_hp: u8,
    pub jaw_open: f32,
    pub mouth_wide: f32,
    pub mouth_pucker: f32,
    pub lip_close: f32,
    pub lip_sneer: f32,
    pub brow_raise: f32,
    pub brow_furrow: f32,
    pub eye_blink: f32,
    pub eye_dart_x: f32,
    pub camera_pos: [f32; 3],
}

impl KBSBMotionScanPacket {
    pub fn compute_crc32(data: &[u8]) -> u32 {
        compute_crc32(data)
    }

    /// Serializes struct into a 64-byte binary buffer
    pub fn encode(&self) -> Result<[u8; KMCP_FRAME_SIZE], CinematicBridgeError> {
        if self.vital_hp > VITAL_MAX_HP {
            return Err(CinematicBridgeError::InvariantBreach(self.vital_hp));
        }

        let mut buf = [0u8; KMCP_FRAME_SIZE];
        buf[0..4].copy_from_slice(&KMCP_MAGIC);
        buf[4..8].copy_from_slice(&self.timestamp_ms.to_be_bytes());
        buf[8] = self.viseme as u8;
        buf[9] = self.emotion as u8;
        buf[10] = self.lod_level;
        buf[11] = self.vital_hp;

        // 8 blendshapes
        buf[12..16].copy_from_slice(&self.jaw_open.to_be_bytes());
        buf[16..20].copy_from_slice(&self.mouth_wide.to_be_bytes());
        buf[20..24].copy_from_slice(&self.mouth_pucker.to_be_bytes());
        buf[24..28].copy_from_slice(&self.lip_close.to_be_bytes());
        buf[28..32].copy_from_slice(&self.lip_sneer.to_be_bytes());
        buf[32..36].copy_from_slice(&self.brow_raise.to_be_bytes());
        buf[36..40].copy_from_slice(&self.brow_furrow.to_be_bytes());
        buf[40..44].copy_from_slice(&self.eye_blink.to_be_bytes());

        // Eye dart X
        buf[44..48].copy_from_slice(&self.eye_dart_x.to_be_bytes());

        // Camera position (X, Y, Z)
        buf[48..52].copy_from_slice(&self.camera_pos[0].to_be_bytes());
        buf[52..56].copy_from_slice(&self.camera_pos[1].to_be_bytes());
        buf[56..60].copy_from_slice(&self.camera_pos[2].to_be_bytes());

        // CRC32 over the first 60 bytes using fast table
        let crc = compute_crc32(&buf[0..60]);
        buf[60..64].copy_from_slice(&crc.to_be_bytes());

        Ok(buf)
    }

    /// Parses and verifies a 64-byte binary buffer
    pub fn decode(raw: &[u8]) -> Result<Self, CinematicBridgeError> {
        if raw.len() != KMCP_FRAME_SIZE {
            return Err(CinematicBridgeError::InvalidFrameSize {
                expected: KMCP_FRAME_SIZE,
                got: raw.len(),
            });
        }
        if raw[0..4] != KMCP_MAGIC {
            let mut got = [0u8; 4];
            got.copy_from_slice(&raw[0..4]);
            return Err(CinematicBridgeError::InvalidMagic {
                expected: KMCP_MAGIC,
                got,
            });
        }

        let expected_crc = u32::from_be_bytes(raw[60..64].try_into().unwrap());
        let computed_crc = compute_crc32(&raw[0..60]);
        if expected_crc != computed_crc {
            return Err(CinematicBridgeError::CrcMismatch {
                expected: expected_crc,
                computed: computed_crc,
            });
        }

        let vital_hp = raw[11];
        if vital_hp > VITAL_MAX_HP {
            return Err(CinematicBridgeError::InvariantBreach(vital_hp));
        }

        let timestamp_ms = u32::from_be_bytes(raw[4..8].try_into().unwrap());
        let viseme = VisemeId::from_u8(raw[8]);
        let emotion = EmotionId::from_u8(raw[9]);
        let lod_level = raw[10];

        let jaw_open = f32::from_be_bytes(raw[12..16].try_into().unwrap());
        let mouth_wide = f32::from_be_bytes(raw[16..20].try_into().unwrap());
        let mouth_pucker = f32::from_be_bytes(raw[20..24].try_into().unwrap());
        let lip_close = f32::from_be_bytes(raw[24..28].try_into().unwrap());
        let lip_sneer = f32::from_be_bytes(raw[28..32].try_into().unwrap());
        let brow_raise = f32::from_be_bytes(raw[32..36].try_into().unwrap());
        let brow_furrow = f32::from_be_bytes(raw[36..40].try_into().unwrap());
        let eye_blink = f32::from_be_bytes(raw[40..44].try_into().unwrap());
        let eye_dart_x = f32::from_be_bytes(raw[44..48].try_into().unwrap());

        let camera_pos = [
            f32::from_be_bytes(raw[48..52].try_into().unwrap()),
            f32::from_be_bytes(raw[52..56].try_into().unwrap()),
            f32::from_be_bytes(raw[56..60].try_into().unwrap()),
        ];

        Ok(Self {
            timestamp_ms,
            viseme,
            emotion,
            lod_level,
            vital_hp,
            jaw_open,
            mouth_wide,
            mouth_pucker,
            lip_close,
            lip_sneer,
            brow_raise,
            brow_furrow,
            eye_blink,
            eye_dart_x,
            camera_pos,
        })
    }
}

/// 32-Byte Deterministic Mortar Ballistics Packet (KMBP)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KBSBMortarBallisticPacket {
    pub timestamp_ms: u32,
    pub progress_s: f32, // [0.0, 1.0]
    pub vital_hp: u8,
    pub camera_mode: u8, // 0=Follow, 1=Plunge, 2=BulletTime
    pub shell_pos: [f32; 3],
    pub speed: f32,
    pub shake_intensity: f32,
}

impl KBSBMortarBallisticPacket {
    pub fn encode(&self) -> Result<[u8; KMBP_FRAME_SIZE], CinematicBridgeError> {
        if self.vital_hp > VITAL_MAX_HP {
            return Err(CinematicBridgeError::InvariantBreach(self.vital_hp));
        }

        let mut buf = [0u8; KMBP_FRAME_SIZE];
        buf[0..4].copy_from_slice(&KMBP_MAGIC);
        buf[4..8].copy_from_slice(&self.timestamp_ms.to_be_bytes());

        let q16_progress = (self.progress_s.clamp(0.0, 1.0) * 65535.0).round() as u16;
        buf[8..10].copy_from_slice(&q16_progress.to_be_bytes());
        buf[10] = self.vital_hp;
        buf[11] = self.camera_mode;

        buf[12..16].copy_from_slice(&self.shell_pos[0].to_be_bytes());
        buf[16..20].copy_from_slice(&self.shell_pos[1].to_be_bytes());
        buf[20..24].copy_from_slice(&self.shell_pos[2].to_be_bytes());

        let speed_u16 = (self.speed.max(0.0) * 100.0).min(65535.0).round() as u16;
        buf[24..26].copy_from_slice(&speed_u16.to_be_bytes());

        let shake_u16 = (self.shake_intensity.clamp(0.0, 1.0) * 65535.0).round() as u16;
        buf[26..28].copy_from_slice(&shake_u16.to_be_bytes());

        let crc = compute_crc32(&buf[0..28]);
        buf[28..32].copy_from_slice(&crc.to_be_bytes());

        Ok(buf)
    }

    pub fn decode(raw: &[u8]) -> Result<Self, CinematicBridgeError> {
        if raw.len() != KMBP_FRAME_SIZE {
            return Err(CinematicBridgeError::InvalidFrameSize {
                expected: KMBP_FRAME_SIZE,
                got: raw.len(),
            });
        }
        if raw[0..4] != KMBP_MAGIC {
            let mut got = [0u8; 4];
            got.copy_from_slice(&raw[0..4]);
            return Err(CinematicBridgeError::InvalidMagic {
                expected: KMBP_MAGIC,
                got,
            });
        }

        let expected_crc = u32::from_be_bytes(raw[28..32].try_into().unwrap());
        let computed_crc = compute_crc32(&raw[0..28]);
        if expected_crc != computed_crc {
            return Err(CinematicBridgeError::CrcMismatch {
                expected: expected_crc,
                computed: computed_crc,
            });
        }

        let vital_hp = raw[10];
        if vital_hp > VITAL_MAX_HP {
            return Err(CinematicBridgeError::InvariantBreach(vital_hp));
        }

        let timestamp_ms = u32::from_be_bytes(raw[4..8].try_into().unwrap());
        let q16_progress = u16::from_be_bytes(raw[8..10].try_into().unwrap());
        let progress_s = (q16_progress as f32) / 65535.0;
        let camera_mode = raw[11];

        let shell_pos = [
            f32::from_be_bytes(raw[12..16].try_into().unwrap()),
            f32::from_be_bytes(raw[16..20].try_into().unwrap()),
            f32::from_be_bytes(raw[20..24].try_into().unwrap()),
        ];

        let speed_u16 = u16::from_be_bytes(raw[24..26].try_into().unwrap());
        let speed = (speed_u16 as f32) / 100.0;

        let shake_u16 = u16::from_be_bytes(raw[26..28].try_into().unwrap());
        let shake_intensity = (shake_u16 as f32) / 65535.0;

        Ok(Self {
            timestamp_ms,
            progress_s,
            vital_hp,
            camera_mode,
            shell_pos,
            speed,
            shake_intensity,
        })
    }
}

/// 16-Byte Deterministic Deception Verdict Packet (KDVP)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KBSBDeceptionVerdictPacket {
    pub timestamp_ms: u32,
    pub verdict: InterrogationVerdict,
    pub vital_hp: u8,
    pub deception_score: f32, // [0.0, 1.0]
}

impl KBSBDeceptionVerdictPacket {
    pub fn encode(&self) -> Result<[u8; KDVP_FRAME_SIZE], CinematicBridgeError> {
        if self.vital_hp > VITAL_MAX_HP {
            return Err(CinematicBridgeError::InvariantBreach(self.vital_hp));
        }

        let mut buf = [0u8; KDVP_FRAME_SIZE];
        buf[0..4].copy_from_slice(&KDVP_MAGIC);
        buf[4..8].copy_from_slice(&self.timestamp_ms.to_be_bytes());

        let v_u8 = match self.verdict {
            InterrogationVerdict::Truth => 0u8,
            InterrogationVerdict::Doubt => 1u8,
            InterrogationVerdict::Lie => 2u8,
        };
        buf[8] = v_u8;
        buf[9] = self.vital_hp;

        let score_q16 = (self.deception_score.clamp(0.0, 1.0) * 65535.0).round() as u16;
        buf[10..12].copy_from_slice(&score_q16.to_be_bytes());

        let crc = compute_crc32(&buf[0..12]);
        buf[12..16].copy_from_slice(&crc.to_be_bytes());

        Ok(buf)
    }

    pub fn decode(raw: &[u8]) -> Result<Self, CinematicBridgeError> {
        if raw.len() != KDVP_FRAME_SIZE {
            return Err(CinematicBridgeError::InvalidFrameSize {
                expected: KDVP_FRAME_SIZE,
                got: raw.len(),
            });
        }
        if raw[0..4] != KDVP_MAGIC {
            let mut got = [0u8; 4];
            got.copy_from_slice(&raw[0..4]);
            return Err(CinematicBridgeError::InvalidMagic {
                expected: KDVP_MAGIC,
                got,
            });
        }

        let expected_crc = u32::from_be_bytes(raw[12..16].try_into().unwrap());
        let computed_crc = compute_crc32(&raw[0..12]);
        if expected_crc != computed_crc {
            return Err(CinematicBridgeError::CrcMismatch {
                expected: expected_crc,
                computed: computed_crc,
            });
        }

        let vital_hp = raw[9];
        if vital_hp > VITAL_MAX_HP {
            return Err(CinematicBridgeError::InvariantBreach(vital_hp));
        }

        let timestamp_ms = u32::from_be_bytes(raw[4..8].try_into().unwrap());
        let verdict = match raw[8] {
            1 => InterrogationVerdict::Doubt,
            2 => InterrogationVerdict::Lie,
            _ => InterrogationVerdict::Truth,
        };

        let score_q16 = u16::from_be_bytes(raw[10..12].try_into().unwrap());
        let deception_score = (score_q16 as f32) / 65535.0;

        Ok(Self {
            timestamp_ms,
            verdict,
            vital_hp,
            deception_score,
        })
    }
}

/// L.A. Noire Interrogation Deception Detector
pub struct DeceptionDetector;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterrogationVerdict {
    Truth,
    Doubt,
    Lie,
}

impl DeceptionDetector {
    /// Computes deception confidence and returns an interrogation verdict
    pub fn evaluate(packet: &KBSBMotionScanPacket) -> (f32, InterrogationVerdict) {
        let eye_component = packet.eye_dart_x.abs().min(1.0);
        let sneer_component = packet.lip_sneer.min(1.0);
        let furrow_component = packet.brow_furrow.min(1.0);

        let deception_score = (eye_component * 0.40) + (sneer_component * 0.40) + (furrow_component * 0.20);

        let verdict = if deception_score > 0.55 || packet.emotion == EmotionId::LieDeceptive {
            InterrogationVerdict::Lie
        } else if deception_score > 0.30 || packet.emotion == EmotionId::DoubtEvasive {
            InterrogationVerdict::Doubt
        } else {
            InterrogationVerdict::Truth
        };

        (deception_score, verdict)
    }
}

/// Ballistic Mortar Trajectory Predictor in native Rust
pub struct MortarTrajectoryPredictor {
    pub launch: [f32; 3],
    pub target: [f32; 3],
    pub apex_height: f32,
}

impl MortarTrajectoryPredictor {
    pub fn new(launch: [f32; 3], target: [f32; 3], apex_height: f32) -> Self {
        Self { launch, target, apex_height }
    }

    /// Evaluates projectile position at normalized progress s in [0.0, 1.0]
    #[inline(always)]
    pub fn sample_position(&self, s: f32) -> [f32; 3] {
        let s = s.clamp(0.0, 1.0);
        let parabola_y = 4.0 * self.apex_height * s * (1.0 - s);
        [
            self.launch[0] + (self.target[0] - self.launch[0]) * s,
            self.launch[1] + (self.target[1] - self.launch[1]) * s + parabola_y,
            self.launch[2] + (self.target[2] - self.launch[2]) * s,
        ]
    }

    /// Evaluates tangent velocity vector dP/ds
    #[inline(always)]
    pub fn sample_velocity(&self, s: f32) -> [f32; 3] {
        let s = s.clamp(0.0, 1.0);
        [
            self.target[0] - self.launch[0],
            (self.target[1] - self.launch[1]) + 4.0 * self.apex_height * (1.0 - 2.0 * s),
            self.target[2] - self.launch[2],
        ]
    }
}

/// Acoustic Spectral Formant Bands for Whisperer Pipeline
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcousticSpectralBands {
    pub f1_low: f32,      // 200 - 1000 Hz: vowel jaw openness (AA, OO)
    pub f2_mid: f32,      // 800 - 2500 Hz: tongue spread (EE)
    pub f3_high: f32,     // 2000 - 4000 Hz: acoustic voice tension
    pub sibilance: f32,   // > 4500 Hz: dental/alveolar fricatives (SS, TH, FV)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticPrediction {
    pub predicted_viseme: VisemeId,
    pub voice_stress_index: f32,
    pub cross_modal_divergence: f32,
    pub anomaly_flagged: bool,
}

/// Whisperer Acoustic Formant Predictor Pipeline in native Rust
pub struct AcousticWhisperPredictor;

impl AcousticWhisperPredictor {
    /// Predicts VisemeId and voice stress index from 4-band acoustic spectral energy
    pub fn predict_viseme(bands: &AcousticSpectralBands) -> (VisemeId, f32) {
        let total_energy = bands.f1_low + bands.f2_mid + bands.f3_high + bands.sibilance;
        if total_energy < 0.05 {
            return (VisemeId::Rest, 0.0);
        }

        let voice_stress = (bands.f3_high * 0.45 + bands.sibilance * 0.55).min(1.0);

        let viseme = if bands.sibilance > 0.55 {
            VisemeId::Ss
        } else if bands.f2_mid > 0.65 {
            VisemeId::Ee
        } else if bands.f1_low > 0.60 {
            VisemeId::Aa
        } else if bands.f3_high > 0.50 {
            VisemeId::Oo
        } else if bands.sibilance > 0.30 {
            VisemeId::Fv
        } else {
            VisemeId::U
        };

        (viseme, voice_stress)
    }

    /// Evaluates cross-modal divergence between facial micro-expression score and acoustic stress
    pub fn evaluate_cross_modal_divergence(
        bands: &AcousticSpectralBands,
        facial_deception_score: f32,
    ) -> AcousticPrediction {
        let (predicted_viseme, voice_stress_index) = Self::predict_viseme(bands);
        let cross_modal_divergence = (facial_deception_score - voice_stress_index).abs();
        let anomaly_flagged = cross_modal_divergence > 0.35;

        AcousticPrediction {
            predicted_viseme,
            voice_stress_index,
            cross_modal_divergence,
            anomaly_flagged,
        }
    }
}

/// Multi-Packet Batch Streaming Buffer
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KBSBMotionScanPacketBatch {
    pub packets: Vec<KBSBMotionScanPacket>,
}

impl KBSBMotionScanPacketBatch {
    pub fn new() -> Self {
        Self { packets: Vec::new() }
    }

    pub fn push(&mut self, pkt: KBSBMotionScanPacket) {
        self.packets.push(pkt);
    }

    pub fn encode_all(&self) -> Result<Vec<u8>, CinematicBridgeError> {
        let mut bytes = Vec::with_capacity(self.packets.len() * KMCP_FRAME_SIZE);
        for pkt in &self.packets {
            bytes.extend_from_slice(&pkt.encode()?);
        }
        Ok(bytes)
    }

    pub fn decode_stream(raw: &[u8]) -> Result<Self, CinematicBridgeError> {
        if raw.len() % KMCP_FRAME_SIZE != 0 {
            return Err(CinematicBridgeError::InvalidFrameSize {
                expected: KMCP_FRAME_SIZE,
                got: raw.len(),
            });
        }
        let count = raw.len() / KMCP_FRAME_SIZE;
        let mut packets = Vec::with_capacity(count);
        for i in 0..count {
            let slice = &raw[i * KMCP_FRAME_SIZE..(i + 1) * KMCP_FRAME_SIZE];
            packets.push(KBSBMotionScanPacket::decode(slice)?);
        }
        Ok(Self { packets })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmcp_encode_decode_roundtrip() {
        let packet = KBSBMotionScanPacket {
            timestamp_ms: 1250,
            viseme: VisemeId::Aa,
            emotion: EmotionId::LieDeceptive,
            lod_level: 0,
            vital_hp: 6,
            jaw_open: 0.85,
            mouth_wide: 0.35,
            mouth_pucker: 0.0,
            lip_close: 0.0,
            lip_sneer: 0.70,
            brow_raise: 0.2,
            brow_furrow: 0.5,
            eye_blink: 0.0,
            eye_dart_x: -0.75,
            camera_pos: [-10.0, 5.0, -12.0],
        };

        let encoded = packet.encode().expect("Encode should succeed");
        assert_eq!(encoded.len(), 64);
        assert_eq!(&encoded[0..4], b"KMCP");

        let decoded = KBSBMotionScanPacket::decode(&encoded).expect("Decode should succeed");
        assert_eq!(decoded.timestamp_ms, 1250);
        assert_eq!(decoded.viseme, VisemeId::Aa);
        assert_eq!(decoded.emotion, EmotionId::LieDeceptive);
        assert_eq!(decoded.vital_hp, 6);
        assert!((decoded.eye_dart_x - (-0.75)).abs() < 1e-4);
    }

    #[test]
    fn test_mortar_packet_32b_roundtrip() {
        let pkt = KBSBMortarBallisticPacket {
            timestamp_ms: 3200,
            progress_s: 0.5,
            vital_hp: 6,
            camera_mode: 1, // Plunge
            shell_pos: [-12.0, 18.5, -14.0],
            speed: 45.2,
            shake_intensity: 0.35,
        };

        let enc = pkt.encode().expect("KMBP encode ok");
        assert_eq!(enc.len(), 32);
        assert_eq!(&enc[0..4], b"KMBP");

        let dec = KBSBMortarBallisticPacket::decode(&enc).expect("KMBP decode ok");
        assert_eq!(dec.timestamp_ms, 3200);
        assert_eq!(dec.vital_hp, 6);
        assert_eq!(dec.camera_mode, 1);
        assert!((dec.progress_s - 0.5).abs() < 1e-3);
        assert!((dec.speed - 45.2).abs() < 0.1);
        assert!((dec.shake_intensity - 0.35).abs() < 1e-3);
    }

    #[test]
    fn test_verdict_packet_16b_roundtrip() {
        let pkt = KBSBDeceptionVerdictPacket {
            timestamp_ms: 500,
            verdict: InterrogationVerdict::Lie,
            vital_hp: 6,
            deception_score: 0.88,
        };

        let enc = pkt.encode().expect("KDVP encode ok");
        assert_eq!(enc.len(), 16);
        assert_eq!(&enc[0..4], b"KDVP");

        let dec = KBSBDeceptionVerdictPacket::decode(&enc).expect("KDVP decode ok");
        assert_eq!(dec.timestamp_ms, 500);
        assert_eq!(dec.verdict, InterrogationVerdict::Lie);
        assert_eq!(dec.vital_hp, 6);
        assert!((dec.deception_score - 0.88).abs() < 1e-3);
    }

    #[test]
    fn test_invariant_hp_enforcement_across_all_packets() {
        let inv_64 = KBSBMotionScanPacket {
            timestamp_ms: 0,
            viseme: VisemeId::Rest,
            emotion: EmotionId::TruthSteady,
            lod_level: 0,
            vital_hp: 7, // Breach
            jaw_open: 0.0,
            mouth_wide: 0.0,
            mouth_pucker: 0.0,
            lip_close: 1.0,
            lip_sneer: 0.0,
            brow_raise: 0.0,
            brow_furrow: 0.0,
            eye_blink: 0.0,
            eye_dart_x: 0.0,
            camera_pos: [0.0, 0.0, 0.0],
        };
        assert!(inv_64.encode().is_err());

        let inv_32 = KBSBMortarBallisticPacket {
            timestamp_ms: 0,
            progress_s: 0.1,
            vital_hp: 9, // Breach
            camera_mode: 0,
            shell_pos: [0.0, 0.0, 0.0],
            speed: 0.0,
            shake_intensity: 0.0,
        };
        assert!(inv_32.encode().is_err());

        let inv_16 = KBSBDeceptionVerdictPacket {
            timestamp_ms: 0,
            verdict: InterrogationVerdict::Truth,
            vital_hp: 12, // Breach
            deception_score: 0.0,
        };
        assert!(inv_16.encode().is_err());
    }

    #[test]
    fn test_acoustic_whisper_predictor_and_divergence() {
        // High sibilance -> SS
        let bands_s = AcousticSpectralBands {
            f1_low: 0.1,
            f2_mid: 0.2,
            f3_high: 0.3,
            sibilance: 0.85,
        };
        let (v_s, stress) = AcousticWhisperPredictor::predict_viseme(&bands_s);
        assert_eq!(v_s, VisemeId::Ss);
        assert!(stress > 0.5);

        // Acoustic calm (low stress) vs facial deception high (0.85) -> Anomaly flagged
        let calm_bands = AcousticSpectralBands {
            f1_low: 0.65, // Open vowel AA
            f2_mid: 0.30,
            f3_high: 0.10,
            sibilance: 0.05,
        };
        let pred = AcousticWhisperPredictor::evaluate_cross_modal_divergence(&calm_bands, 0.85);
        assert_eq!(pred.predicted_viseme, VisemeId::Aa);
        assert!(pred.anomaly_flagged);
        assert!(pred.cross_modal_divergence > 0.35);
    }

    #[test]
    fn test_batch_packet_stream() {
        let mut batch = KBSBMotionScanPacketBatch::new();
        for i in 0..5 {
            batch.push(KBSBMotionScanPacket {
                timestamp_ms: (i * 33) as u32,
                viseme: VisemeId::Ee,
                emotion: EmotionId::TruthSteady,
                lod_level: 0,
                vital_hp: 6,
                jaw_open: 0.3,
                mouth_wide: 0.7,
                mouth_pucker: 0.0,
                lip_close: 0.0,
                lip_sneer: 0.0,
                brow_raise: 0.1,
                brow_furrow: 0.1,
                eye_blink: 0.0,
                eye_dart_x: 0.0,
                camera_pos: [0.0, 1.0, 2.0],
            });
        }

        let stream_bytes = batch.encode_all().expect("Batch encode ok");
        assert_eq!(stream_bytes.len(), 5 * 64);

        let decoded_batch = KBSBMotionScanPacketBatch::decode_stream(&stream_bytes).expect("Batch decode ok");
        assert_eq!(decoded_batch.packets.len(), 5);
        assert_eq!(decoded_batch.packets[4].timestamp_ms, 132);
    }
}
