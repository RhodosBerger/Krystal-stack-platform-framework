//! Cinematic Bridge - Native MotionScan KBSB v2 Codec, Deception Detector, and Mortar Trajectory Engine
//!
//! Provides zero-copy 64-byte binary packet decoding, CRC32 verification,
//! L.A. Noire deception classification, and ballistic trajectory prediction.

use serde::{Deserialize, Serialize};

pub const KMCP_MAGIC: [u8; 4] = *b"KMCP";
pub const KMCP_FRAME_SIZE: usize = 64;
pub const VITAL_MAX_HP: u8 = 6;

#[derive(Debug, thiserror::Error)]
pub enum CinematicBridgeError {
    #[error("Invalid frame size: expected 64, got {0}")]
    InvalidFrameSize(usize),
    #[error("Invalid magic header: expected 'KMCP'")]
    InvalidMagic,
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
    /// Standard IEEE 802.3 CRC32 implementation without external dependencies
    pub fn compute_crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFFFFFFu32;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                let mask = (crc & 1) != 0;
                crc >>= 1;
                if mask {
                    crc ^= 0xEDB88320;
                }
            }
        }
        !crc
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

        // CRC32 over the first 60 bytes
        let crc = Self::compute_crc32(&buf[0..60]);
        buf[60..64].copy_from_slice(&crc.to_be_bytes());

        Ok(buf)
    }

    /// Parses and verifies a 64-byte binary buffer
    pub fn decode(raw: &[u8]) -> Result<Self, CinematicBridgeError> {
        if raw.len() != KMCP_FRAME_SIZE {
            return Err(CinematicBridgeError::InvalidFrameSize(raw.len()));
        }
        if raw[0..4] != KMCP_MAGIC {
            return Err(CinematicBridgeError::InvalidMagic);
        }

        let expected_crc = u32::from_be_bytes(raw[60..64].try_into().unwrap());
        let computed_crc = Self::compute_crc32(&raw[0..60]);
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
        // Weighted composite score: eye avoidance (0.4) + asymmetrical sneer (0.4) + furrow (0.2)
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
    pub fn sample_position(&self, s: f32) -> [f32; 3] {
        let s = s.clamp(0.0, 1.0);
        let parabola_y = 4.0 * self.apex_height * s * (1.0 - s);
        [
            self.launch[0] + (self.target[0] - self.launch[0]) * s,
            self.launch[1] + (self.target[1] - self.launch[1]) * s + parabola_y,
            self.launch[2] + (self.target[2] - self.launch[2]) * s,
        ]
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
    fn test_invariant_hp_enforcement() {
        let invalid_packet = KBSBMotionScanPacket {
            timestamp_ms: 500,
            viseme: VisemeId::Rest,
            emotion: EmotionId::TruthSteady,
            lod_level: 0,
            vital_hp: 7, // BREACH
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

        assert!(invalid_packet.encode().is_err());
    }

    #[test]
    fn test_deception_detector_classification() {
        let truthful_packet = KBSBMotionScanPacket {
            timestamp_ms: 100,
            viseme: VisemeId::Rest,
            emotion: EmotionId::TruthSteady,
            lod_level: 0,
            vital_hp: 6,
            jaw_open: 0.0,
            mouth_wide: 0.0,
            mouth_pucker: 0.0,
            lip_close: 1.0,
            lip_sneer: 0.0,
            brow_raise: 0.1,
            brow_furrow: 0.1,
            eye_blink: 0.0,
            eye_dart_x: 0.05,
            camera_pos: [0.0, 0.0, 0.0],
        };
        let (_, verdict_truth) = DeceptionDetector::evaluate(&truthful_packet);
        assert_eq!(verdict_truth, InterrogationVerdict::Truth);

        let lying_packet = KBSBMotionScanPacket {
            timestamp_ms: 200,
            viseme: VisemeId::Mbp,
            emotion: EmotionId::LieDeceptive,
            lod_level: 0,
            vital_hp: 6,
            jaw_open: 0.0,
            mouth_wide: 0.0,
            mouth_pucker: 0.0,
            lip_close: 1.0,
            lip_sneer: 0.85,
            brow_raise: 0.0,
            brow_furrow: 0.6,
            eye_blink: 0.0,
            eye_dart_x: -0.80, // Averting eyes
            camera_pos: [0.0, 0.0, 0.0],
        };
        let (score_lie, verdict_lie) = DeceptionDetector::evaluate(&lying_packet);
        assert!(score_lie > 0.55);
        assert_eq!(verdict_lie, InterrogationVerdict::Lie);
    }

    #[test]
    fn test_mortar_trajectory_prediction() {
        let pred = MortarTrajectoryPredictor::new([-10.0, 1.0, -10.0], [0.0, 0.0, 0.0], 15.0);
        let apex = pred.sample_position(0.5);
        assert!((apex[1] - 15.5).abs() < 1e-3);
    }
}
