#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR AUTONOMOUS COMMAND LOOP ACTORS & SOCKETS
==============================================================================
File: tests/test_automated_command_actors.py
Description:
    Validates:
      1. Constitutional Invariant: VITAL_MAX_HP == 6 across all actors and packets.
      2. BulkParameterSocketPacket binary serialization, deserialization, and CRC32 integrity.
      3. BulkParameterSocketServer loopback socket lifecycle and packet ACK protocol.
      4. HardwareRegulationLoopActor bulk parameter ingestion via dedicated socket.
      5. NpuBatchTuningLoopActor OpenVINO tensor stream and bias configuration via socket.
      6. EpistemicGovernorLoopActor invariant audit and veto enforcement.
      7. AutonomousActorSystemCoordinator multi-actor supervisor and 100+ parameter throughput.
==============================================================================
"""

import time
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    ActuationState,
    AuthorityState,
    to_wire_token
)
from krystal_kernel.automated_command_actors import (
    SocketCommandOpcode,
    BulkParameterSocketPacket,
    BulkParameterSocketServer,
    send_bulk_parameters_via_client,
    ActorLoopState,
    BaseAutonomousCommandLoopActor,
    HardwareRegulationLoopActor,
    NpuBatchTuningLoopActor,
    EpistemicGovernorLoopActor,
    AutonomousActorSystemCoordinator,
    GLOBAL_ACTOR_SYSTEM_COORDINATOR
)


class TestAutomatedCommandActorsAndSockets(unittest.TestCase):
    """Test suite for Autonomous Command Loop Actors and Bulk Parameter Sockets."""

    def test_01_constitutional_invariant_6hp(self):
        """Invariant: VITAL_MAX_HP must strictly equal 6 in all actor state containers."""
        self.assertEqual(VITAL_MAX_HP, 6)

        coordinator = AutonomousActorSystemCoordinator()
        telemetry = coordinator.get_full_telemetry()
        self.assertEqual(telemetry["vital_max_hp"], 6)

        actor = HardwareRegulationLoopActor()
        self.assertEqual(actor.vital_max_hp, 6)

    def test_02_packet_serialization_and_crc32(self):
        """Packets must serialize to binary with 24-byte header and valid CRC32."""
        payload = {"param_a": 12.5, "param_b": "OPTIMAL", "matrix": [1, 2, 3]}
        pkt = BulkParameterSocketPacket(
            opcode=SocketCommandOpcode.OP_SET_BULK_PARAMETERS,
            transaction_id=123456789,
            param_count=len(payload),
            payload=payload
        )
        data = pkt.serialize()
        self.assertGreater(len(data), 28)

        # Deserialize valid packet
        deserialized = BulkParameterSocketPacket.deserialize(data)
        self.assertIsNotNone(deserialized)
        self.assertEqual(deserialized.opcode, SocketCommandOpcode.OP_SET_BULK_PARAMETERS)
        self.assertEqual(deserialized.transaction_id, 123456789)
        self.assertEqual(deserialized.param_count, len(payload))
        self.assertEqual(deserialized.payload["param_a"], 12.5)

        # Corrupt data must fail deserialization (CRC check)
        corrupted = bytearray(data)
        corrupted[30] ^= 0xFF
        self.assertIsNone(BulkParameterSocketPacket.deserialize(bytes(corrupted)))

    def test_03_hardware_actor_bulk_parameter_socket(self):
        """HardwareRegulationLoopActor must ingest bulk hardware parameters over socket."""
        actor = HardwareRegulationLoopActor()
        actor.start()
        try:
            port = actor.socket_server.bound_port
            self.assertGreater(port, 0)
            self.assertEqual(actor.state, ActorLoopState.ACTOR_RUNNING)

            params = {
                "pl1_watts": 30.0,
                "cpu_affinity_mask": 0x000F,
                "process_priority": "HIGH_PRIORITY_CLASS",
                "npu_streams": 4,
                "custom_voltage_offset_mv": -25.0
            }

            res = send_bulk_parameters_via_client(
                host="127.0.0.1",
                port=port,
                parameters=params
            )
            self.assertEqual(res["status"], "OK")
            self.assertEqual(res["applied_count"], len(params))
            self.assertEqual(res["vital_max_hp"], 6)

            # Check that actor updated internal values
            self.assertEqual(actor.active_pl1, 30.0)
            self.assertEqual(actor.active_affinity, 0x000F)
            self.assertEqual(actor.socket_server.packets_received, 1)
        finally:
            actor.stop()
            self.assertEqual(actor.state, ActorLoopState.ACTOR_TERMINATED)

    def test_04_npu_batch_tuning_actor_socket(self):
        """NpuBatchTuningLoopActor must ingest tensor model configurations over socket."""
        actor = NpuBatchTuningLoopActor()
        actor.start()
        try:
            port = actor.socket_server.bound_port
            params = {
                "stream_count": 4,
                "dp4a_vector_mode": True,
                "tensor_biases": {f"layer_{i}": 0.05 * i for i in range(10)}
            }
            res = send_bulk_parameters_via_client(
                host="127.0.0.1",
                port=port,
                parameters=params
            )
            self.assertEqual(res["status"], "OK")
            self.assertEqual(actor.stream_count, 4)
            self.assertTrue(actor.dp4a_vector_mode)
            self.assertEqual(len(actor.active_tensor_biases), 10)
        finally:
            actor.stop()

    def test_05_epistemic_governor_invariant_veto(self):
        """EpistemicGovernorLoopActor must approve valid batches and VETO invalid HP."""
        actor = EpistemicGovernorLoopActor()
        actor.start()
        try:
            port = actor.socket_server.bound_port

            # 1. Valid parameter batch with HP == 6
            valid_batch = {"thermal_headroom_target": 25.0, "vital_max_hp": 6}
            res_valid = send_bulk_parameters_via_client("127.0.0.1", port, valid_batch)
            self.assertEqual(res_valid["status"], "OK")
            self.assertTrue(res_valid["result"]["authorized"])
            self.assertEqual(res_valid["result"]["authority_verdict"], to_wire_token(AuthorityState.AUTH_COMMIT_APPROVED))

            # 2. Invalid batch attempting to violate constitutional invariant (HP == 7)
            breach_batch = {"thermal_headroom_target": 25.0, "vital_max_hp": 7}
            res_breach = send_bulk_parameters_via_client("127.0.0.1", port, breach_batch)
            self.assertEqual(res_breach["status"], "OK")
            self.assertFalse(res_breach["result"]["authorized"])
            self.assertEqual(res_breach["result"]["authority_verdict"], to_wire_token(AuthorityState.AUTH_VETOED))
            self.assertEqual(actor.governance_vetoes, 1)
        finally:
            actor.stop()

    def test_06_coordinator_bulk_streaming_throughput(self):
        """Coordinator must manage all actors and stream 100+ parameters in microsecond times."""
        coordinator = AutonomousActorSystemCoordinator()
        coordinator.start_all_actors()
        try:
            telemetry = coordinator.get_full_telemetry()
            self.assertEqual(telemetry["total_actors"], 3)
            self.assertTrue(telemetry["actors_active"])

            # Generate 150 bulk parameters
            bulk_params = {f"tuning_param_{i}": i * 0.1 for i in range(150)}
            bulk_params["pl1_watts"] = 28.0

            t0 = time.perf_counter_ns()
            res = coordinator.dispatch_bulk_parameters("ACTOR_HW_REGULATOR", bulk_params)
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0

            self.assertEqual(res["status"], "OK")
            self.assertEqual(res["applied_count"], 151)
            # Socket latency should be under 50 milliseconds on loopback
            self.assertLess(elapsed_us, 50_000.0)
        finally:
            coordinator.stop_all_actors()


if __name__ == "__main__":
    unittest.main()
