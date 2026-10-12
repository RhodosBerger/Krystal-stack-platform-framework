#!/usr/bin/env python3
"""
Integration Test Suite: Advanced Cinematic, Mortar Ballistics, L.A. Noire Lip-Sync & Rust Engine
================================================================================================
Validates end-to-end integration across:
1. Native KBSBMotionScan 64-Byte binary packet compatibility (cross-language zero-copy).
2. Live HTTP Endpoints in KrystalWebHub (ephemeral port):
   - GET  /api/cinematic/status
   - GET  /api/cinematic/scenes/interrogation
   - GET  /api/cinematic/trajectory/mortar
   - POST /api/cinematic/lipsync/evaluate
   - POST /api/cinematic/codec/packet (encode & decode roundtrip)
   - POST /api/cinematic/interrogation/action (Epistemic Authority integration)
3. VITAL_MAX_HP == 6 invariant enforcement across all routes and packet structures.
"""

import os
import sys
import json
import base64
import struct
import urllib.request
import urllib.error
import threading
import unittest
from http.server import HTTPServer

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_web_hub.server import KrystalHubHandler, state
from mimicry_engine.scene_composer import (
    MortarBallisticCameraTrajectory,
    build_noir_mortar_interrogation_scene,
    UrbanSpatialCompositionRules
)
from mimicry_engine.cinematic_character_lipsync import (
    LANoireCharacterLipsyncEngine,
    VisemeType,
    EmotionalMicroExpression,
    FacialBlendshapeFrame
)
from mimicry_engine.cinematic_codec_adaptation import (
    SupportedCodecType,
    VisemeLODLevel,
    KBSBMotionScanBinaryCodec,
    AdaptiveCinematicController
)


class TestAdvancedCinematicAndRustIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start ephemeral HTTP server on 127.0.0.1:0 to avoid port conflicts
        cls.server = HTTPServer(("127.0.0.1", 0), KrystalHubHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _http_get(self, path: str):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8")
            return resp.status, json.loads(data)

    def _http_post(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read().decode("utf-8")
                return resp.status, json.loads(data)
        except urllib.error.HTTPError as e:
            err_data = e.read().decode("utf-8")
            try:
                parsed = json.loads(err_data)
            except Exception:
                parsed = {"error": err_data}
            return e.code, parsed

    def test_01_binary_packet_zero_copy_layout(self):
        """Validates exact 64-byte binary alignment matching Rust KBSBMotionScanPacket."""
        frame = FacialBlendshapeFrame(
            timestamp=1.25,
            viseme=VisemeType.AA,
            emotion=EmotionalMicroExpression.LIE_DECEPTIVE,
            jaw_open=0.85,
            mouth_wide=0.35,
            mouth_pucker=0.0,
            lip_close=0.0,
            lip_sneer=0.70,
            brow_raise=0.20,
            brow_furrow=0.50,
            eye_blink=0.0,
            eye_dart_x=-0.75
        )
        cam_pos = (-10.0, 5.0, -12.0)
        encoded = KBSBMotionScanBinaryCodec.encode_frame(
            frame=frame,
            camera_pos=cam_pos,
            lod=VisemeLODLevel.LOD_0_ULTRA_CLOSEUP,
            vital_hp=6
        )

        self.assertEqual(len(encoded), 64)
        self.assertEqual(encoded[0:4], b"KMCP")

        # Unpack header
        magic, ts_ms, viseme_b, emotion_b, lod_b, vital_hp = struct.unpack(">4sIBBBB", encoded[0:12])
        self.assertEqual(magic, b"KMCP")
        self.assertEqual(ts_ms, 1250)
        self.assertEqual(viseme_b, 1)  # AA
        self.assertEqual(emotion_b, 2) # LIE_DECEPTIVE
        self.assertEqual(lod_b, 0)     # LOD_0_ULTRA_CLOSEUP
        self.assertEqual(vital_hp, 6)

        # Unpack blendshapes
        blendshapes = struct.unpack(">8f", encoded[12:44])
        self.assertAlmostEqual(blendshapes[0], 0.85, places=3)  # jaw_open
        self.assertAlmostEqual(blendshapes[4], 0.70, places=3)  # lip_sneer
        self.assertAlmostEqual(blendshapes[6], 0.50, places=3)  # brow_furrow

        # Unpack eye dart
        eye_dart = struct.unpack(">f", encoded[44:48])[0]
        self.assertAlmostEqual(eye_dart, -0.75, places=3)

        # Unpack camera position
        cx, cy, cz = struct.unpack(">3f", encoded[48:60])
        self.assertAlmostEqual(cx, -10.0, places=2)
        self.assertAlmostEqual(cy, 5.0, places=2)
        self.assertAlmostEqual(cz, -12.0, places=2)

    def test_02_server_cinematic_status(self):
        """Tests GET /api/cinematic/status."""
        status_code, body = self._http_get("/api/cinematic/status")
        self.assertEqual(status_code, 200)
        self.assertEqual(body["status"], "OK")
        self.assertTrue(body["available"])
        self.assertEqual(body["vital_max_hp"], 6)
        self.assertIn("OGG_OPUS_SPEECH", body["codecs"])
        self.assertIn("KBSB_MOTIONSCAN_V2", body["codecs"])
        self.assertIn("AA", body["visemes"])
        self.assertIn("LIE_DECEPTIVE", body["emotions"])

    def test_03_server_interrogation_scene(self):
        """Tests GET /api/cinematic/scenes/interrogation."""
        status_code, body = self._http_get("/api/cinematic/scenes/interrogation")
        self.assertEqual(status_code, 200)
        self.assertEqual(body["status"], "OK")
        scene = body["scene"]
        self.assertEqual(scene["scene_id"], "SCENE_NOIR_MORTAR_INTERROGATION")
        self.assertIn("trajectory", scene)
        self.assertEqual(scene["trajectory"]["type"], "MortarBallisticCameraTrajectory")

        # Urban spatial rules validation
        val = body["validation"]
        self.assertTrue(val["passed"])
        self.assertGreaterEqual(val["compliance_score"], 0.80)

    def test_04_server_mortar_trajectory(self):
        """Tests GET /api/cinematic/trajectory/mortar with custom ballistics."""
        status_code, body = self._http_get(
            "/api/cinematic/trajectory/mortar?launch_x=-15&launch_y=2&launch_z=-20&apex=25&mode=CINEMATIC_PLUNGE&steps=20"
        )
        self.assertEqual(status_code, 200)
        self.assertEqual(body["status"], "OK")
        self.assertEqual(body["mode"], "CINEMATIC_PLUNGE")
        self.assertEqual(body["apex_height"], 25.0)
        self.assertEqual(body["step_count"], 21)

        samples = body["samples"]
        # Apex should occur near progress ~0.5
        mid = samples[10]
        self.assertAlmostEqual(mid["progress"], 0.5, places=2)
        self.assertGreater(mid["shell_pos"][1], 20.0)

        # Impact should have higher shake
        first_step = samples[0]
        last_step = samples[-1]
        self.assertGreater(last_step["shake"], first_step["shake"])

    def test_05_server_lipsync_evaluate(self):
        """Tests POST /api/cinematic/lipsync/evaluate."""
        payload = {
            "text": "I didn't steal the bearer bonds, detective! I was at the speakeasy!",
            "emotion": "LIE_DECEPTIVE",
            "vital_hp": 6,
            "distance_m": 2.5,
            "fps": 30.0
        }
        status_code, body = self._http_post("/api/cinematic/lipsync/evaluate", payload)
        self.assertEqual(status_code, 200)
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["deception_verdict"], "LIE")
        self.assertGreater(body["deception_score"], 0.50)
        self.assertEqual(body["lod"], "LOD_0_ULTRA_CLOSEUP")
        self.assertIn("ascii_portrait", body)
        self.assertIn("godot_tracks", body)
        self.assertIn("jaw_open", body["godot_tracks"])

    def test_06_server_lipsync_invariant_breach_veto(self):
        """Tests POST /api/cinematic/lipsync/evaluate with vital_hp > 6 returning VETO."""
        payload = {
            "text": "Attempting unauthorized hp boost!",
            "emotion": "TRUTH_STEADY",
            "vital_hp": 8
        }
        status_code, body = self._http_post("/api/cinematic/lipsync/evaluate", payload)
        self.assertEqual(status_code, 400)
        self.assertEqual(body["status"], "VETO")
        self.assertEqual(body["vital_hp"], 8)

    def test_07_server_codec_packet_roundtrip(self):
        """Tests POST /api/cinematic/codec/packet encode & decode."""
        encode_payload = {
            "action": "encode",
            "timestamp_ms": 2500,
            "viseme": "OO",
            "emotion": "DOUBT_EVASIVE",
            "lod_level": 0,
            "vital_hp": 6,
            "jaw_open": 0.45,
            "mouth_wide": 0.20,
            "mouth_pucker": 0.85,
            "lip_close": 0.10,
            "lip_sneer": 0.15,
            "brow_furrow": 0.65,
            "eye_dart_x": 0.40,
            "camera_pos": [0.0, 1.2, 3.5]
        }
        status_code, enc_body = self._http_post("/api/cinematic/codec/packet", encode_payload)
        self.assertEqual(status_code, 200)
        self.assertEqual(enc_body["status"], "SUCCESS")
        self.assertEqual(enc_body["packet_size"], 64)
        hex_data = enc_body["hex_data"]

        # Decode via server
        decode_payload = {
            "action": "decode",
            "hex_data": hex_data
        }
        status_code, dec_body = self._http_post("/api/cinematic/codec/packet", decode_payload)
        self.assertEqual(status_code, 200)
        self.assertEqual(dec_body["status"], "SUCCESS")
        pkt = dec_body["packet"]
        self.assertEqual(pkt["viseme"], "OO")
        self.assertEqual(pkt["emotion"], "DOUBT_EVASIVE")
        self.assertEqual(pkt["vital_hp"], 6)
        self.assertAlmostEqual(pkt["blendshapes"]["mouth_pucker"], 0.85, places=2)
        self.assertAlmostEqual(pkt["blendshapes"]["jaw_open"], 0.45, places=2)

    def test_08_server_interrogation_epistemic_authority(self):
        """Tests POST /api/cinematic/interrogation/action with Epistemic Authority states."""
        # 1. Valid evidence -> COMMIT_APPROVED
        p1 = {
            "action_id": "CROSS_EXAMINE_01",
            "suspect_id": "vance",
            "evidence_refs": ["mortar_casing_brass", "manifest_lot_44"],
            "vital_hp": 6
        }
        code1, body1 = self._http_post("/api/cinematic/interrogation/action", p1)
        self.assertEqual(code1, 200)
        self.assertEqual(body1["authority_verdict"], "COMMIT_APPROVED")
        self.assertEqual(body1["lifecycle_action"], "WOULD_COMMIT")

        # 2. Contradictory evidence -> QUARANTINE_CONTRADICTORY_EVIDENCE
        p2 = {
            "action_id": "CROSS_EXAMINE_02",
            "suspect_id": "vance",
            "evidence_refs": ["alibi_statement", "contradictory_fingerprint"],
            "vital_hp": 6
        }
        code2, body2 = self._http_post("/api/cinematic/interrogation/action", p2)
        self.assertEqual(code2, 200)
        self.assertEqual(body2["authority_verdict"], "QUARANTINE_CONTRADICTORY_EVIDENCE")
        self.assertEqual(body2["lifecycle_action"], "QUARANTINE_AND_RECALIBRATE")

        # 3. Insufficient evidence -> DEFER_INSUFFICIENT_EVIDENCE
        p3 = {
            "action_id": "CROSS_EXAMINE_03",
            "suspect_id": "vance",
            "evidence_refs": [],
            "vital_hp": 6
        }
        code3, body3 = self._http_post("/api/cinematic/interrogation/action", p3)
        self.assertEqual(code3, 200)
        self.assertEqual(body3["authority_verdict"], "DEFER_INSUFFICIENT_EVIDENCE")
        self.assertEqual(body3["lifecycle_action"], "TRIGGER_ACTIVE_MICRO_PROBE")

        # 4. Invariant breach -> VETO_SAFETY_BREACH
        p4 = {
            "action_id": "CROSS_EXAMINE_04",
            "suspect_id": "vance",
            "evidence_refs": ["weapon"],
            "vital_hp": 7
        }
        code4, body4 = self._http_post("/api/cinematic/interrogation/action", p4)
        self.assertEqual(code4, 400)
        self.assertEqual(body4["verdict"], "VETO_SAFETY_BREACH")
        self.assertEqual(body4["lifecycle_action"], "REJECT_NOOP")


if __name__ == "__main__":
    unittest.main()
