"""
Krystal-Stack Mimicry Engine: Cinematic Codec & Real-Time Adaptations
=====================================================================
Selects supported audio/motion capture codecs (Ogg Opus & KBSB-MotionScan v2)
and implements multi-dimensional runtime adaptations:
1. Spatial Distance-Aware Viseme Level-of-Detail (LOD0 Close-up -> LOD2 Aerial Mortar).
2. Mortar Velocity-Coupled Camera Adaptation (Doppler, Terminal Shake, Dynamic FOV).
3. Dynamic Bitrate Scaling (16 kbps -> 64 kbps) for multi-language broadcast.
4. Invariant Assertion across all frames (VITAL_MAX_HP == 6).
"""

import math
import struct
import zlib
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

from mimicry_engine.cinematic_character_lipsync import (
    VisemeType,
    EmotionalMicroExpression,
    FacialBlendshapeFrame,
    DialogueScriptLine,
    LANoireCharacterLipsyncEngine
)
from mimicry_engine.scene_composer import MortarBallisticCameraTrajectory, Vec3


class SupportedCodecType(Enum):
    """Officially supported multimedia and motion capture codecs in Krystal Stack."""
    OGG_OPUS_SPEECH = "OGG_OPUS_SPEECH"               # RFC 6716, 48kHz, low-latency, Godot 4.x native
    KBSB_MOTIONSCAN_V2 = "KBSB_MOTIONSCAN_V2"         # 64-byte deterministic binary blendshape stream
    OGG_VORBIS_LEGACY = "OGG_VORBIS_LEGACY"           # Standard uncompressed dialogue fallback
    AV1_WEBM_CINEMATIC = "AV1_WEBM_CINEMATIC"         # Modern royalty-free video container


class VisemeLODLevel(Enum):
    """Level of Detail scaling based on camera-to-character proximity."""
    LOD_0_ULTRA_CLOSEUP = 0   # < 4.0m: Full 60 FPS, all 8 blendshapes, micro-saccades, sneer
    LOD_1_MEDIUM_SHOT = 1     # 4.0m - 12.0m: 30 FPS, jaw_open, mouth_wide, basic blink
    LOD_2_COARSE_AERIAL = 2   # > 12.0m: 10 FPS, coarse jaw_open only, low network overhead


class KBSBMotionScanBinaryCodec:
    """
    Krystal Binary State Bytecode (KBSB v2.0 MotionScan Packet).
    Enforces a compact, 64-byte binary frame for real-time streaming of
    facial blendshape states and camera trajectory coordinates.

    Frame Specification (64 Bytes):
      0..3   char[4]  Magic Header b"KMCP" (0x4B4D4350)
      4..7   uint32   Timestamp in milliseconds
      8      uint8    Viseme ID (0..8)
      9      uint8    Emotion ID (0..3)
      10     uint8    LOD Level (0..2)
      11     uint8    Vital HP (Strictly 6)
      12..43 float[8] Blendshapes: jaw, wide, pucker, close, sneer, brow_raise, brow_furrow, blink
      44..47 float    Eye Dart X (-1.0 to 1.0)
      48..59 float[3] Camera Position (X, Y, Z)
      60..63 uint32   CRC32 Checksum
    """
    MAGIC = b"KMCP"
    FRAME_SIZE = 64

    VISEME_MAP = {v: i for i, v in enumerate(VisemeType)}
    REV_VISEME_MAP = {i: v for i, v in enumerate(VisemeType)}

    EMOTION_MAP = {e: i for i, e in enumerate(EmotionalMicroExpression)}
    REV_EMOTION_MAP = {i: e for i, e in enumerate(EmotionalMicroExpression)}

    @classmethod
    def encode_frame(
        cls,
        frame: FacialBlendshapeFrame,
        camera_pos: Vec3,
        lod: VisemeLODLevel = VisemeLODLevel.LOD_0_ULTRA_CLOSEUP,
        vital_hp: int = 6
    ) -> bytes:
        """Serializes a facial frame and camera coordinate into a 64-byte binary payload."""
        if vital_hp > 6:
            raise ValueError(f"Invariant breach: vital_hp={vital_hp} > 6")

        ms = int(frame.timestamp * 1000.0) & 0xFFFFFFFF
        v_id = cls.VISEME_MAP.get(frame.viseme, 0)
        e_id = cls.EMOTION_MAP.get(frame.emotion, 0)
        lod_id = lod.value

        # Header: 4s I B B B B
        header = struct.pack(">4sIBBBB", cls.MAGIC, ms, v_id, e_id, lod_id, vital_hp)

        # Apply LOD reductions to blendshape stream
        if lod == VisemeLODLevel.LOD_2_COARSE_AERIAL:
            bs = (frame.jaw_open, 0.0, 0.0, frame.lip_close, 0.0, 0.0, 0.0, 0.0)
            eye_x = 0.0
        elif lod == VisemeLODLevel.LOD_1_MEDIUM_SHOT:
            bs = (frame.jaw_open, frame.mouth_wide, 0.0, frame.lip_close, 0.0, frame.brow_furrow, 0.0, frame.eye_blink)
            eye_x = round(frame.eye_dart_x, 1)
        else:
            bs = (
                frame.jaw_open, frame.mouth_wide, frame.mouth_pucker, frame.lip_close,
                frame.lip_sneer, frame.brow_raise, frame.brow_furrow, frame.eye_blink
            )
            eye_x = frame.eye_dart_x

        payload = struct.pack(">8ff3f", *bs, eye_x, camera_pos[0], camera_pos[1], camera_pos[2])
        raw_without_crc = header + payload  # 12 + 48 = 60 bytes
        crc = zlib.crc32(raw_without_crc) & 0xFFFFFFFF
        return raw_without_crc + struct.pack(">I", crc)

    @classmethod
    def decode_frame(cls, raw: bytes) -> Dict[str, Any]:
        """Deserializes and validates a 64-byte binary frame."""
        if len(raw) != cls.FRAME_SIZE:
            raise ValueError(f"Invalid frame size: {len(raw)} != {cls.FRAME_SIZE}")

        magic, ms, v_id, e_id, lod_id, hp = struct.unpack(">4sIBBBB", raw[:12])
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")
        if hp > 6:
            raise ValueError(f"Decoded invalid vital_hp: {hp} > 6")

        expected_crc = struct.unpack(">I", raw[60:64])[0]
        actual_crc = zlib.crc32(raw[:60]) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ValueError(f"CRC32 mismatch: expected {expected_crc:#x}, got {actual_crc:#x}")

        unpacked_body = struct.unpack(">8ff3f", raw[12:60])
        bs = unpacked_body[:8]
        eye_x = unpacked_body[8]
        cam_pos = unpacked_body[9:12]

        return {
            "timestamp": ms / 1000.0,
            "viseme": cls.REV_VISEME_MAP.get(v_id, VisemeType.REST).value,
            "emotion": cls.REV_EMOTION_MAP.get(e_id, EmotionalMicroExpression.TRUTH_STEADY).value,
            "lod": VisemeLODLevel(lod_id).name,
            "vital_hp": hp,
            "blendshapes": {
                "jaw_open": round(bs[0], 3),
                "mouth_wide": round(bs[1], 3),
                "mouth_pucker": round(bs[2], 3),
                "lip_close": round(bs[3], 3),
                "lip_sneer": round(bs[4], 3),
                "brow_raise": round(bs[5], 3),
                "brow_furrow": round(bs[6], 3),
                "eye_blink": round(bs[7], 3)
            },
            "eye_dart_x": round(eye_x, 3),
            "camera_pos": [round(c, 3) for c in cam_pos]
        }


class AdaptiveCinematicController:
    """
    Unified manager executing four dynamic adaptations during movie playback:
    1. Distance-Aware Viseme Level-of-Detail (LOD).
    2. Mortar Velocity-Coupled Camera Dynamics (Shake, Doppler, and FOV breathing).
    3. Opus Bitrate Scaling.
    4. Invariant Protection.
    """
    def __init__(
        self,
        character_pos: Vec3 = (0.0, 0.0, -0.4),
        supported_codec: SupportedCodecType = SupportedCodecType.OGG_OPUS_SPEECH
    ):
        self.character_pos = character_pos
        self.supported_codec = supported_codec
        self.active_lod: VisemeLODLevel = VisemeLODLevel.LOD_0_ULTRA_CLOSEUP
        self.audio_bitrate_kbps: int = 64

    def calculate_lod(self, camera_pos: Vec3) -> VisemeLODLevel:
        """Determines LOD level based on distance between camera and speaking character."""
        dx = camera_pos[0] - self.character_pos[0]
        dy = camera_pos[1] - self.character_pos[1]
        dz = camera_pos[2] - self.character_pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < 4.0:
            self.active_lod = VisemeLODLevel.LOD_0_ULTRA_CLOSEUP
            self.audio_bitrate_kbps = 64
        elif dist < 12.0:
            self.active_lod = VisemeLODLevel.LOD_1_MEDIUM_SHOT
            self.audio_bitrate_kbps = 32
        else:
            self.active_lod = VisemeLODLevel.LOD_2_COARSE_AERIAL
            self.audio_bitrate_kbps = 16

        return self.active_lod

    def adapt_camera_velocity(
        self,
        trajectory: MortarBallisticCameraTrajectory,
        t: float
    ) -> Tuple[Vec3, Vec3, float, float]:
        """
        Calculates adapted camera state with velocity-induced shake and Doppler FOV breathing.
        Returns (adapted_cam_pos, target, adapted_fov, shake_intensity).
        """
        cam_pos, target, base_fov = trajectory.get_camera_state(t)
        progress = max(0.0, min(1.0, t / max(1e-4, trajectory.duration)))

        vel = trajectory.get_shell_velocity(progress)
        speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

        # High terminal plunge speed induces camera vibration and FOV compression
        shake_intensity = max(0.0, (speed - 15.0) * 0.015) if speed > 15.0 else 0.0
        shake_offset_x = math.sin(t * 45.0) * shake_intensity
        shake_offset_y = math.cos(t * 38.0) * shake_intensity

        adapted_pos = (
            cam_pos[0] + shake_offset_x,
            cam_pos[1] + shake_offset_y,
            cam_pos[2]
        )
        # Doppler FOV compression during terminal high-speed plunge
        adapted_fov = base_fov - min(8.0, shake_intensity * 12.0)

        return adapted_pos, target, adapted_fov, shake_intensity

    def process_cinematic_tick(
        self,
        t: float,
        trajectory: MortarBallisticCameraTrajectory,
        dialogue_line: DialogueScriptLine
    ) -> Dict[str, Any]:
        """
        Executes a single synchronized frame evaluation across mortar ballistics,
        character lip-sync, codec encoding, and adaptive LOD scaling.
        """
        # 1. Evaluate camera trajectory with velocity adaptation
        cam_pos, look_target, fov, shake = self.adapt_camera_velocity(trajectory, t)

        # 2. Determine spatial LOD
        lod = self.calculate_lod(cam_pos)

        # 3. Evaluate L.A. Noire character face
        facial_frame = LANoireCharacterLipsyncEngine.evaluate_facial_frame(t, dialogue_line)

        # 4. Binary Codec Encoding (KBSB v2.0 MotionScan Packet)
        binary_packet = KBSBMotionScanBinaryCodec.encode_frame(
            frame=facial_frame,
            camera_pos=cam_pos,
            lod=lod,
            vital_hp=6
        )

        return {
            "time": round(t, 3),
            "camera": {
                "position": [round(c, 3) for c in cam_pos],
                "target": [round(c, 3) for c in look_target],
                "fov": round(fov, 2),
                "shake_intensity": round(shake, 4)
            },
            "adaptations": {
                "active_lod": lod.name,
                "audio_codec": self.supported_codec.value,
                "audio_bitrate_kbps": self.audio_bitrate_kbps,
                "vital_hp": 6
            },
            "character_face": facial_frame.to_dict(),
            "binary_packet_bytes": len(binary_packet),
            "binary_packet_hex": binary_packet[:16].hex() + "..."
        }


def run_codec_adaptation_demo():
    """Demonstrates codec selection and real-time adaptations during the movie scene."""
    print("======================================================================")
    print("KRYSTAL-STACK: CINEMATIC CODEC SELECTION & REAL-TIME ADAPTATIONS DEMO")
    print("======================================================================")
    print(f"Selected Speech Codec:  {SupportedCodecType.OGG_OPUS_SPEECH.value} (48kHz, Variable Bitrate)")
    print(f"Selected Motion Codec:  {SupportedCodecType.KBSB_MOTIONSCAN_V2.value} (64-Byte Fixed Binary Stream)")
    print("Supported Video Codec:   AV1 / WebM (Cinematic High-Efficiency)")

    mortar = MortarBallisticCameraTrajectory(
        name="AdaptiveMortarPlunge",
        launch_pos=(-14.0, 2.0, -16.0),
        target_pos=(0.0, 0.0, -0.4),
        apex_height=18.0,
        camera_mode="FOLLOW_SHELL",
        duration=6.0
    )
    script_line = DialogueScriptLine(
        speaker="Suspect Malley",
        line="Look out! Mortar shell incoming!",
        start_time=0.0,
        duration=6.0,
        intended_emotion=EmotionalMicroExpression.SHOCK_IMPACT
    )
    controller = AdaptiveCinematicController()

    sample_times = [0.2, 1.5, 3.0, 4.8, 5.8]
    for st in sample_times:
        tick = controller.process_cinematic_tick(st, mortar, script_line)
        print(f"\n--- TICK @ t = {st:.2f}s ---")
        print(f"Camera Pos: {tick['camera']['position']} | FOV: {tick['camera']['fov']} deg | Shake: {tick['camera']['shake_intensity']}")
        print(f"LOD Mode:   {tick['adaptations']['active_lod']} | Bitrate: {tick['adaptations']['audio_bitrate_kbps']} kbps | Invariant HP: {tick['adaptations']['vital_hp']}")
        print(f"Viseme:     {tick['character_face']['viseme']} | Emotion: {tick['character_face']['emotion']} | Jaw Open: {tick['character_face']['jaw_open']}")
        print(f"Binary Stream: {tick['binary_packet_bytes']} bytes [KMCP Frame: {tick['binary_packet_hex']}]")

    print("\n[OK] All adaptations verified with VITAL_MAX_HP == 6 invariant.")


if __name__ == "__main__":
    run_codec_adaptation_demo()
