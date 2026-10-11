#!/usr/bin/env python3
"""
Unit tests for Native Multi-Threaded Procedural Pipeline, Godot 4.x Vulkan Bridge,
Janet Synchronizer DSL, and Zero-Copy Binary Buffer Protocols.
"""

import os
import sys
import time
import math
import struct
import socket
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_janet.janet_bridge import JanetValidator
from krystal_kernel.native_procedural_renderer import (
    NativeProceduralRenderer,
    GodotProceduralIPCServer,
    INSTANCE_STRIDE_BYTES,
    INSTANCE_FLOAT_COUNT,
    STRUCT_FORMAT_INSTANCE,
    GLOBAL_NATIVE_PROCEDURAL_RENDERER
)


class TestNativeProceduralPipeline(unittest.TestCase):

    def test_janet_procedural_synchronizer_syntax_and_invariants(self):
        janet_file = os.path.join(WORKSPACE_ROOT, "krystal_janet", "procedural_synchronizer.janet")
        self.assertTrue(os.path.exists(janet_file), "procedural_synchronizer.janet must exist")

        val = JanetValidator.validate_file(janet_file)
        self.assertTrue(val["valid"], f"Janet syntax error: {val.get('error')}")
        self.assertIn("VITAL-MAX-HP", val["definitions"])
        self.assertIn("HARDWARE-AFFINITY-RULES", val["definitions"])
        self.assertIn("PROCEDURAL-MATH-GENES", val["definitions"])
        self.assertIn("GODOT-VULKAN-CONTRACT", val["definitions"])
        self.assertEqual(val["bracket_counts"]["("], val["bracket_counts"][")"])

    def test_native_procedural_renderer_batch_generation(self):
        renderer = NativeProceduralRenderer(max_workers=4)
        target_count = 8192
        raw_bytes, stats = renderer.generate_procedural_scene_binary(target_count=target_count)

        expected_bytes = target_count * INSTANCE_STRIDE_BYTES
        self.assertEqual(len(raw_bytes), expected_bytes)
        self.assertEqual(stats.total_instances, target_count)
        self.assertEqual(stats.draw_calls_guaranteed, 1)
        self.assertTrue(stats.l2_cache_resident)
        self.assertGreater(stats.instances_per_second, 50000.0)
        self.assertEqual(stats.vital_max_hp, VITAL_MAX_HP)

    def test_unpacked_binary_instance_buffer_integrity(self):
        renderer = NativeProceduralRenderer(max_workers=2)
        raw_bytes, stats = renderer.generate_procedural_scene_binary(target_count=64)

        # Inspect first instance
        unpacked = struct.unpack_from(STRUCT_FORMAT_INSTANCE, raw_bytes, 0)
        self.assertEqual(len(unpacked), INSTANCE_FLOAT_COUNT)

        # 4x4 matrix row 3 (affine 0, 0, 0, 1)
        m30, m31, m32, m33 = unpacked[12:16]
        self.assertAlmostEqual(m30, 0.0)
        self.assertAlmostEqual(m31, 0.0)
        self.assertAlmostEqual(m32, 0.0)
        self.assertAlmostEqual(m33, 1.0)

        # Color RGBA (unpacked[16:20])
        r, g, b, a = unpacked[16:20]
        self.assertTrue(0.0 <= r <= 1.0)
        self.assertTrue(0.0 <= g <= 1.0)
        self.assertTrue(0.0 <= b <= 1.0)
        self.assertEqual(a, 1.0)

        # Custom Data (unpacked[20:24])
        entropy, phase_id, time_param, flags = unpacked[20:24]
        self.assertTrue(0.0 <= entropy <= 1.0)
        self.assertIn(phase_id, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(flags, 1.0)

    def test_godot_procedural_tcp_ipc_communication(self):
        # Spin up test IPC server on dedicated port
        test_port = 8097
        server = GodotProceduralIPCServer(host="127.0.0.1", port=test_port)
        server.start()
        time.sleep(0.1)

        try:
            # Simulate Godot NativeProceduralVulkanBridge client connecting via TCP
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", test_port))

            # Send request packet: 8 bytes [target_instances=1024, seed=777]
            req = struct.pack("<2I", 1024, 777)
            client.sendall(req)

            # Receive 16-byte header: [magic, count, total_bytes, vital_max_hp]
            hdr_bytes = client.recv(16)
            self.assertEqual(len(hdr_bytes), 16)
            magic, count, total_bytes, vital_hp = struct.unpack("<4I", hdr_bytes)

            self.assertEqual(magic, 0x4B525953)  # "KRYS"
            self.assertEqual(count, 1024)
            self.assertEqual(total_bytes, 1024 * INSTANCE_STRIDE_BYTES)
            self.assertEqual(vital_hp, VITAL_MAX_HP)

            # Receive payload
            received_bytes = 0
            while received_bytes < total_bytes:
                chunk = client.recv(min(8192, total_bytes - received_bytes))
                if not chunk:
                    break
                received_bytes += len(chunk)

            self.assertEqual(received_bytes, total_bytes)
            client.close()
        finally:
            server.stop()

    def test_server_native_procedural_endpoints(self):
        import json
        from io import BytesIO
        from krystal_web_hub.server import KrystalHubHandler

        class DummyServer:
            pass

        class MockHandler(KrystalHubHandler):
            def __init__(self, path: str, method: str = "GET", body: dict = None):
                self.path = path
                self.command = method
                self.requestline = f"{method} {path} HTTP/1.1"
                self.wfile = BytesIO()
                self.server = DummyServer()
                self.client_address = ("127.0.0.1", 8080)
                self.close_connection = True
                self._code = 200

                if body is not None:
                    body_bytes = json.dumps(body).encode("utf-8")
                    self.rfile = BytesIO(body_bytes)
                    self.headers = {"Content-Length": str(len(body_bytes)), "Content-Type": "application/json"}
                else:
                    self.rfile = BytesIO()
                    self.headers = {}

            def send_response(self, code, message=None):
                self._code = code

            def send_header(self, keyword, value):
                pass

            def end_headers(self):
                pass

        # 1. GET /api/native_procedural/status
        h_get = MockHandler("/api/native_procedural/status", "GET")
        h_get.do_GET()
        res_get = json.loads(h_get.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_get["status"], "OK")
        self.assertEqual(res_get["vital_max_hp"], 6)
        self.assertEqual(res_get["renderer"], "NativeProceduralRenderer")

        # 2. POST /api/native_procedural/generate
        h_post = MockHandler("/api/native_procedural/generate", "POST", {
            "target_count": 2048,
            "center_x": 0.0,
            "center_z": 0.0,
            "time_sec": 1.5
        })
        h_post.do_POST()
        res_post = json.loads(h_post.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_post["status"], "OK")
        self.assertEqual(res_post["vital_max_hp"], 6)
        self.assertEqual(res_post["batch_stats"]["total_instances"], 2048)
        self.assertEqual(res_post["batch_stats"]["draw_calls_guaranteed"], 1)


if __name__ == "__main__":
    unittest.main()
