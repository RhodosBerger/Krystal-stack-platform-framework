"""
Unit Test Suite for Cinematic Codec Selection & Real-Time Adaptations
=====================================================================
Verifies:
1. Supported codecs: Ogg Opus, KBSB-MotionScan v2 (64 bytes), AV1/WebM.
2. KBSBMotionScanBinaryCodec: serialization, deserialization, CRC32, vital_hp invariant.
3. AdaptiveCinematicController: distance-aware LOD scaling, bitrate adaptation,
   velocity-induced camera shake, Doppler FOV adaptation.
"""

import unittest
import struct

from mimicry_engine.cinematic_codec_adaptation import (
    SupportedCodecType,
    VisemeLODLevel,
    KBSBMotionScanBinaryCodec,
    AdaptiveCinematicController
)
from mimicry_engine.cinematic_character_lipsync import (
    VisemeType,
    EmotionalMicroExpression,
    FacialBlendshapeFrame,
    DialogueScriptLine
)
from mimicry_engine.scene_composer import MortarBallisticCameraTrajectory

class TestCinematicCodecAdaptation(unittest.TestCase):

    def test_01_binary_codec_roundtrip(self):
        frame = FacialBlendshapeFrame(
            timestamp=2.45,
            viseme=VisemeType.AA,
            emotion=EmotionalMicroExpression.LIE_DECEPTIVE,
            jaw_open=0.85,
            mouth_wide=0.4,
            lip_sneer=0.75,
            brow_furrow=0.5,
            eye_dart_x=-0.65,
            eye_blink=0.0
        )
        cam_pos = (-5.2, 12.0, -8.4)
        encoded = KBSBMotionScanBinaryCodec.encode_frame(
            frame=frame,
            camera_pos=cam_pos,
            lod=VisemeLODLevel.LOD_0_ULTRA_CLOSEUP,
            vital_hp=6
        )
        self.assertEqual(len(encoded), 64)
        self.assertTrue(encoded.startswith(b"KMCP"))

        # Decode and verify fields
        decoded = KBSBMotionScanBinaryCodec.decode_frame(encoded)
        self.assertAlmostEqual(decoded["timestamp"], 2.45, places=2)
        self.assertEqual(decoded["viseme"], "AA")
        self.assertEqual(decoded["emotion"], "LIE_DECEPTIVE")
        self.assertEqual(decoded["lod"], "LOD_0_ULTRA_CLOSEUP")
        self.assertEqual(decoded["vital_hp"], 6)
        self.assertAlmostEqual(decoded["blendshapes"]["jaw_open"], 0.85, places=2)
        self.assertAlmostEqual(decoded["blendshapes"]["lip_sneer"], 0.75, places=2)
        self.assertAlmostEqual(decoded["eye_dart_x"], -0.65, places=2)
        self.assertEqual(decoded["camera_pos"], [-5.2, 12.0, -8.4])

    def test_02_binary_codec_crc_and_invariant_validation(self):
        frame = FacialBlendshapeFrame(viseme=VisemeType.REST)
        # Test invariant protection (vital_hp > 6 must fail)
        with self.assertRaises(ValueError):
            KBSBMotionScanBinaryCodec.encode_frame(frame, (0, 0, 0), vital_hp=7)

        # Corrupt CRC check
        valid_raw = KBSBMotionScanBinaryCodec.encode_frame(frame, (0, 0, 0), vital_hp=6)
        corrupted = bytearray(valid_raw)
        corrupted[62] ^= 0xFF  # Flip CRC bit
        with self.assertRaises(ValueError):
            KBSBMotionScanBinaryCodec.decode_frame(bytes(corrupted))

    def test_03_distance_aware_lod_adaptation(self):
        controller = AdaptiveCinematicController(character_pos=(0.0, 0.0, 0.0))

        # Close-up (< 4m)
        lod_close = controller.calculate_lod((1.0, 1.0, 1.0))
        self.assertEqual(lod_close, VisemeLODLevel.LOD_0_ULTRA_CLOSEUP)
        self.assertEqual(controller.audio_bitrate_kbps, 64)

        # Medium shot (4m - 12m)
        lod_med = controller.calculate_lod((5.0, 2.0, 4.0))
        self.assertEqual(lod_med, VisemeLODLevel.LOD_1_MEDIUM_SHOT)
        self.assertEqual(controller.audio_bitrate_kbps, 32)

        # Aerial distant view (>= 12m)
        lod_far = controller.calculate_lod((10.0, 15.0, 10.0))
        self.assertEqual(lod_far, VisemeLODLevel.LOD_2_COARSE_AERIAL)
        self.assertEqual(controller.audio_bitrate_kbps, 16)

    def test_04_camera_velocity_and_shake_adaptation(self):
        controller = AdaptiveCinematicController()
        mortar = MortarBallisticCameraTrajectory(
            launch_pos=(-14.0, 2.0, -16.0),
            target_pos=(0.0, 0.0, 0.0),
            apex_height=18.0,
            duration=6.0
        )
        # Low speed near apex (t=3.0) vs high speed pre-impact (t=5.8)
        _, _, fov_apex, shake_apex = controller.adapt_camera_velocity(mortar, 3.0)
        _, _, fov_impact, shake_impact = controller.adapt_camera_velocity(mortar, 5.8)

        self.assertGreater(shake_impact, shake_apex)

    def test_05_cinematic_tick_pipeline(self):
        controller = AdaptiveCinematicController()
        mortar = MortarBallisticCameraTrajectory(duration=6.0)
        line = DialogueScriptLine(
            speaker="Detective",
            line="Where were you on the night of the mortar strike?",
            start_time=0.0,
            duration=6.0,
            intended_emotion=EmotionalMicroExpression.TRUTH_STEADY
        )
        tick = controller.process_cinematic_tick(2.5, mortar, line)
        self.assertIn("camera", tick)
        self.assertIn("adaptations", tick)
        self.assertIn("character_face", tick)
        self.assertEqual(tick["adaptations"]["vital_hp"], 6)
        self.assertEqual(tick["binary_packet_bytes"], 64)

if __name__ == "__main__":
    unittest.main()
