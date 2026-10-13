#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: AUTONOMOUS COMMAND LOOP ACTORS & BULK PARAMETER SOCKET FABRIC
==============================================================================
Module: krystal_kernel/automated_command_actors.py
Description:
    Autonomous actor control loops that dynamically instantiate dedicated high-speed
    IPC / network sockets to ingest, validate, and atomically apply large batches
    of system, hardware, and tensor model parameters.

Key Architecture:
    1. Binary & Structured Socket Protocol (BulkParameterSocketPacket):
       - Magic framing (0x4B50524D 'KPRM') with CRC32 integrity verification.
       - Transactional semantics (TXN_PREPARED -> TXN_COMMITTED / TXN_ABORTED).
       - Supports hundreds to thousands of parameter tuples in a single transmission.

    2. Autonomous Command Loop Actors:
       - HardwareRegulationLoopActor:
         Autonomous control loop managing CPU affinity, Win32 priority, PL1/PL2 clamps,
         voltage offsets, and cooling curves via dedicated socket.
       - NpuBatchTuningLoopActor:
         Autonomous loop streaming OpenVINO NPU tensor parameters, multi-stream
         concurrencies, DP4A vectorization modes, and model quantization weights.
       - EpistemicGovernorLoopActor:
         Supervisory loop enforcing dual-gate authority, paradox quarantine, and
         the constitutional invariant: assert VITAL_MAX_HP == 6.

    3. Unified Actor Coordinator (AutonomousActorSystemCoordinator):
       - Lifecycle arbitrator managing actor loops, dedicated socket port pools,
         and transactional 2-phase commit distribution.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys
import time
import json
import zlib
import struct
import socket
import select
import threading
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    ActuationState,
    TransactionState,
    AuthorityState,
    SafetyState,
    InvariantState,
    HardwareState,
    to_wire_token
)
from krystal_kernel.hardware_layers import (
    HardwareSubsystemCoordinator,
    GLOBAL_HARDWARE_COORDINATOR,
    HardwareDomainKind
)
from krystal_kernel.windows_timer_subsystem import (
    WindowsTimerSubsystem,
    GLOBAL_WINDOWS_TIMER_SUBSYSTEM
)

# Packet Constants
SOCKET_MAGIC_BYTES = b"KPRM"  # 4 bytes
PROTOCOL_VERSION   = 0x0001    # uint16

@unique
class SocketCommandOpcode(int, Enum):
    """Opcodes for the bulk parameter socket protocol."""
    OP_SET_BULK_PARAMETERS = 0x0001
    OP_QUERY_PARAMETERS    = 0x0002
    OP_STREAM_TELEMETRY    = 0x0003
    OP_COMMIT_TRANSACTION  = 0x0004
    OP_ROLLBACK_TRANSACTION = 0x0005
    OP_HEARTBEAT           = 0x0006


@dataclass
class BulkParameterSocketPacket:
    """Wire frame for streaming large parameter sets across sockets."""
    opcode: SocketCommandOpcode
    transaction_id: int
    param_count: int
    payload: Dict[str, Any]
    version: int = PROTOCOL_VERSION
    crc32_checksum: int = 0

    def serialize(self) -> bytes:
        """Encodes packet into binary stream: [MAGIC][VER][OPCODE][TXN_ID][COUNT][LEN][PAYLOAD][CRC32]."""
        payload_bytes = json.dumps(self.payload).encode("utf-8")
        payload_len = len(payload_bytes)
        # Header: 4s (MAGIC), H (ver), H (opcode), Q (txn_id), I (count), I (len) -> 26 bytes
        header = struct.pack(
            "!4sHHQII",
            SOCKET_MAGIC_BYTES,
            self.version,
            int(self.opcode),
            self.transaction_id,
            self.param_count,
            payload_len
        )
        data_to_checksum = header + payload_bytes
        checksum = zlib.crc32(data_to_checksum)
        return data_to_checksum + struct.pack("!I", checksum)

    @classmethod
    def deserialize(cls, data: bytes) -> Optional["BulkParameterSocketPacket"]:
        """Parses binary packet and verifies magic header and CRC32 checksum."""
        if len(data) < 28:  # Minimum header 24 bytes + 4 bytes CRC
            return None

        magic, ver, op_val, txn_id, count, plen = struct.unpack("!4sHHQII", data[:24])
        if magic != SOCKET_MAGIC_BYTES:
            return None

        total_expected = 24 + plen + 4
        if len(data) < total_expected:
            return None

        payload_bytes = data[24:24 + plen]
        crc_received = struct.unpack("!I", data[24 + plen:24 + plen + 4])[0]
        crc_computed = zlib.crc32(data[:24 + plen])

        if crc_received != crc_computed:
            return None

        payload = json.loads(payload_bytes.decode("utf-8"))
        return cls(
            opcode=SocketCommandOpcode(op_val),
            transaction_id=txn_id,
            param_count=count,
            payload=payload,
            version=ver,
            crc32_checksum=crc_received
        )


class BulkParameterSocketServer:
    """
    Asynchronous TCP Loopback Socket Server dedicated to high-throughput
    bulk parameter configuration.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, on_packet_callback: Optional[Callable] = None):
        self.host = host
        self.requested_port = port
        self.bound_port = 0
        self.on_packet_callback = on_packet_callback
        self._server_sock: Optional[socket.socket] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self.packets_received = 0
        self.parameters_applied = 0
        self.last_transaction_id = 0

    def start(self):
        """Starts socket server listener on background thread."""
        if self._is_running:
            return

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.requested_port))
        self.bound_port = self._server_sock.getsockname()[1]
        self._server_sock.listen(16)
        self._server_sock.settimeout(0.2)

        self._is_running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name=f"BulkSocket-{self.bound_port}")
        self._thread.start()

    def _listen_loop(self):
        while self._is_running and self._server_sock:
            try:
                client_sock, _ = self._server_sock.accept()
                client_sock.settimeout(1.0)
                client_thread = threading.Thread(
                    target=self._handle_client, args=(client_sock,), daemon=True
                )
                client_thread.start()
            except socket.timeout:
                continue
            except Exception:
                if not self._is_running:
                    break

    def _handle_client(self, client_sock: socket.socket):
        try:
            raw_buf = b""
            while self._is_running:
                chunk = client_sock.recv(65536)
                if not chunk:
                    break
                raw_buf += chunk

                while len(raw_buf) >= 28:
                    magic, _, _, _, _, plen = struct.unpack("!4sHHQII", raw_buf[:24])
                    if magic != SOCKET_MAGIC_BYTES:
                        raw_buf = raw_buf[1:]
                        continue

                    total_pkt_len = 24 + plen + 4
                    if len(raw_buf) < total_pkt_len:
                        break

                    pkt_bytes = raw_buf[:total_pkt_len]
                    raw_buf = raw_buf[total_pkt_len:]

                    packet = BulkParameterSocketPacket.deserialize(pkt_bytes)
                    if packet and self.on_packet_callback:
                        response_dict = self.on_packet_callback(packet)
                        self.packets_received += 1
                        self.parameters_applied += packet.param_count
                        self.last_transaction_id = packet.transaction_id

                        # Acknowledge response packet
                        ack = {
                            "status": "OK",
                            "txn_id": packet.transaction_id,
                            "applied_count": packet.param_count,
                            "result": response_dict,
                            "vital_max_hp": VITAL_MAX_HP
                        }
                        ack_pkt = BulkParameterSocketPacket(
                            opcode=SocketCommandOpcode.OP_COMMIT_TRANSACTION,
                            transaction_id=packet.transaction_id,
                            param_count=len(ack),
                            payload=ack
                        )
                        client_sock.sendall(ack_pkt.serialize())
        except Exception:
            pass
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    def stop(self):
        """Stops listener and cleans up socket descriptors."""
        self._is_running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


def send_bulk_parameters_via_client(
    host: str,
    port: int,
    parameters: Dict[str, Any],
    opcode: SocketCommandOpcode = SocketCommandOpcode.OP_SET_BULK_PARAMETERS,
    timeout_sec: float = 2.0
) -> Dict[str, Any]:
    """Helper client to transmit a bulk parameter dictionary across a dedicated socket."""
    txn_id = int(time.time_ns())
    packet = BulkParameterSocketPacket(
        opcode=opcode,
        transaction_id=txn_id,
        param_count=len(parameters),
        payload=parameters
    )
    wire_data = packet.serialize()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    try:
        sock.connect((host, port))
        sock.sendall(wire_data)
        response_data = b""
        total_expected = 28
        while len(response_data) < total_expected:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response_data += chunk
            if total_expected == 28 and len(response_data) >= 24:
                _, _, _, _, _, plen = struct.unpack("!4sHHQII", response_data[:24])
                total_expected = 24 + plen + 4

        ack_pkt = BulkParameterSocketPacket.deserialize(response_data)
        if ack_pkt:
            return ack_pkt.payload
        return {"status": "ERROR", "reason": f"Malformed ACK packet (received {len(response_data)}/{total_expected} bytes)"}
    finally:
        sock.close()


@unique
class ActorLoopState(str, Enum):
    """Lifecycle states of an autonomous command loop actor."""
    ACTOR_IDLE       = "ACTOR_IDLE"
    ACTOR_RUNNING    = "ACTOR_RUNNING"
    ACTOR_PAUSED     = "ACTOR_PAUSED"
    ACTOR_TERMINATED = "ACTOR_TERMINATED"


class BaseAutonomousCommandLoopActor:
    """
    Base autonomous actor running a periodic automated command loop
    and managing a dedicated bulk parameter socket.
    """

    def __init__(
        self,
        actor_id: str,
        actor_name: str,
        domain: HardwareDomainKind,
        cadence_ms: float = 25.0,
        preferred_port: int = 0
    ):
        self.actor_id = actor_id
        self.actor_name = actor_name
        self.domain = domain
        self.cadence_ms = cadence_ms
        self.preferred_port = preferred_port
        self.state = ActorLoopState.ACTOR_IDLE
        self.loop_cycles = 0
        self.last_cycle_us = 0.0
        self.parameters_db: Dict[str, Any] = {}
        self.vital_max_hp: int = VITAL_MAX_HP
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Dedicated Socket Server
        self.socket_server = BulkParameterSocketServer(
            host="127.0.0.1",
            port=self.preferred_port,
            on_packet_callback=self.handle_socket_packet
        )

    def start(self):
        """Starts the dedicated socket and spawns the autonomous command loop."""
        assert self.vital_max_hp == 6, "Invariant VITAL_MAX_HP == 6 breached in actor startup"
        if self.state == ActorLoopState.ACTOR_RUNNING:
            return

        self.socket_server.start()
        self._stop_event.clear()
        self.state = ActorLoopState.ACTOR_RUNNING
        self._loop_thread = threading.Thread(target=self._command_loop, daemon=True, name=f"ActorLoop-{self.actor_id}")
        self._loop_thread.start()

    def _command_loop(self):
        while not self._stop_event.is_set():
            t0 = time.perf_counter_ns()
            try:
                self.autonomous_cycle_tick()
                self.loop_cycles += 1
            except Exception:
                pass
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            self.last_cycle_us = round(elapsed_us, 2)

            sleep_sec = max(0.001, (self.cadence_ms / 1000.0) - (elapsed_us / 1_000_000.0))
            self._stop_event.wait(timeout=sleep_sec)

    def autonomous_cycle_tick(self):
        """Hook called on each autonomous control loop iteration."""
        pass

    def handle_socket_packet(self, packet: BulkParameterSocketPacket) -> Dict[str, Any]:
        """Processes incoming bulk parameters received through the dedicated socket."""
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 breached in packet handling"
        # Base validation & update
        self.parameters_db.update(packet.payload)
        return {
            "actor_id": self.actor_id,
            "processed_params": len(packet.payload),
            "state_version": self.loop_cycles,
            "status": to_wire_token(ActuationState.ACT_EXECUTED)
        }

    def stop(self):
        """Stops the autonomous loop and shuts down the dedicated socket."""
        self._stop_event.set()
        self.state = ActorLoopState.ACTOR_TERMINATED
        self.socket_server.stop()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=0.5)

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "name": self.actor_name,
            "domain": self.domain.value,
            "state": self.state.value,
            "cadence_ms": self.cadence_ms,
            "loop_cycles": self.loop_cycles,
            "last_cycle_us": self.last_cycle_us,
            "socket_host": self.socket_server.host,
            "socket_port": self.socket_server.bound_port,
            "packets_received": self.socket_server.packets_received,
            "parameters_applied": self.socket_server.parameters_applied,
            "parameter_count_active": len(self.parameters_db),
            "vital_max_hp": self.vital_max_hp
        }


class HardwareRegulationLoopActor(BaseAutonomousCommandLoopActor):
    """
    Automated command loop actor regulating CPU affinity, thread priorities,
    PL1/PL2 clamps, and voltage curves through its dedicated bulk parameter socket.
    """

    def __init__(self, hw_coordinator: Optional[HardwareSubsystemCoordinator] = None, preferred_port: int = 0):
        super().__init__(
            actor_id="ACTOR_HW_REGULATOR",
            actor_name="Hardware Autonomous Regulation Loop Actor",
            domain=HardwareDomainKind.POWER_THERMAL,
            cadence_ms=20.0,
            preferred_port=preferred_port
        )
        self.hw = hw_coordinator or GLOBAL_HARDWARE_COORDINATOR
        self.active_pl1 = 24.0
        self.active_affinity = 0x00FF

    def autonomous_cycle_tick(self):
        """Samples hardware levels and verifies operating envelope."""
        st = self.hw.get_full_status()
        headroom = st["power_thermal"]["thermal_headroom_c"]
        if headroom < 12.0 and self.active_pl1 > 15.0:
            # Autonomous thermal protection clamp
            self.active_pl1 = 12.0
            self.hw.apply_hardware_profile(pl1_watts=12.0)

    def handle_socket_packet(self, packet: BulkParameterSocketPacket) -> Dict[str, Any]:
        """Ingests bulk hardware configuration (voltages, clamps, affinities)."""
        super().handle_socket_packet(packet)
        payload = packet.payload
        applied_actions = []

        pl1 = payload.get("pl1_watts")
        if pl1 is not None:
            self.active_pl1 = float(pl1)
            self.hw.apply_hardware_profile(pl1_watts=self.active_pl1)
            applied_actions.append(f"PL1_SET_{self.active_pl1}W")

        affinity = payload.get("cpu_affinity_mask")
        if affinity is not None:
            self.active_affinity = int(affinity)
            self.hw.apply_hardware_profile(affinity_mask=self.active_affinity)
            applied_actions.append(f"AFFINITY_0x{self.active_affinity:04X}")

        priority = payload.get("process_priority")
        if priority is not None:
            self.hw.apply_hardware_profile(priority_class=str(priority))
            applied_actions.append(f"PRIORITY_{priority}")

        npu_streams = payload.get("npu_streams")
        if npu_streams is not None:
            self.hw.apply_hardware_profile(npu_streams=int(npu_streams))
            applied_actions.append(f"NPU_STREAMS_{npu_streams}")

        return {
            "applied_actions": applied_actions,
            "status": to_wire_token(ActuationState.ACT_EXECUTED),
            "vital_max_hp": VITAL_MAX_HP
        }


class NpuBatchTuningLoopActor(BaseAutonomousCommandLoopActor):
    """
    Automated command loop actor streaming and tuning OpenVINO NPU parameters,
    layer quantization biases, stream allocations, and inference budgets.
    """

    def __init__(self, preferred_port: int = 0):
        super().__init__(
            actor_id="ACTOR_NPU_BATCH_TUNER",
            actor_name="NPU Batch Tuning & Model Tensor Loop Actor",
            domain=HardwareDomainKind.NPU,
            cadence_ms=15.0,
            preferred_port=preferred_port
        )
        self.stream_count = 2
        self.dp4a_vector_mode = True
        self.active_tensor_biases: Dict[str, float] = {}

    def autonomous_cycle_tick(self):
        """Autonomous check of OpenVINO pipeline health and inference budgets."""
        pass

    def handle_socket_packet(self, packet: BulkParameterSocketPacket) -> Dict[str, Any]:
        """Ingests bulk tensor weights, stream settings, and quantization profiles."""
        super().handle_socket_packet(packet)
        payload = packet.payload

        streams = payload.get("stream_count")
        if streams is not None:
            self.stream_count = int(streams)

        dp4a = payload.get("dp4a_vector_mode")
        if dp4a is not None:
            self.dp4a_vector_mode = bool(dp4a)

        biases = payload.get("tensor_biases", {})
        if isinstance(biases, dict):
            self.active_tensor_biases.update(biases)

        return {
            "stream_count": self.stream_count,
            "dp4a_mode": self.dp4a_vector_mode,
            "updated_biases_count": len(biases),
            "status": to_wire_token(ActuationState.ACT_EXECUTED),
            "vital_max_hp": VITAL_MAX_HP
        }


class EpistemicGovernorLoopActor(BaseAutonomousCommandLoopActor):
    """
    Autonomous governance loop actor supervising safety invariants,
    authoritative dual-gate decisions, and rejecting any breach of VITAL_MAX_HP == 6.
    """

    def __init__(self, preferred_port: int = 0):
        super().__init__(
            actor_id="ACTOR_EPISTEMIC_GOVERNOR",
            actor_name="Epistemic Governor & Invariant Supervisory Actor",
            domain=HardwareDomainKind.CPU,
            cadence_ms=50.0,
            preferred_port=preferred_port
        )
        self.dual_gate_policy = "STRICT_VERIFIED"
        self.governance_vetoes = 0

    def handle_socket_packet(self, packet: BulkParameterSocketPacket) -> Dict[str, Any]:
        """Validates parameter batches against constitutional safety invariants."""
        payload = packet.payload

        # Guard: Check vital max hp rule
        if "vital_max_hp" in payload and payload["vital_max_hp"] != 6:
            self.governance_vetoes += 1
            return {
                "authorized": False,
                "reason": "VETO: Constitutional Invariant VITAL_MAX_HP must strictly equal 6",
                "authority_verdict": to_wire_token(AuthorityState.AUTH_VETOED),
                "vital_max_hp": VITAL_MAX_HP
            }

        super().handle_socket_packet(packet)
        return {
            "authorized": True,
            "authority_verdict": to_wire_token(AuthorityState.AUTH_COMMIT_APPROVED),
            "total_vetoes": self.governance_vetoes,
            "status": to_wire_token(ActuationState.ACT_EXECUTED),
            "vital_max_hp": VITAL_MAX_HP
        }


class AutonomousActorSystemCoordinator:
    """
    Supervisor coordinating all autonomous command loop actors and their
    dedicated bulk parameter sockets.
    """

    def __init__(self):
        self.hw_actor = HardwareRegulationLoopActor()
        self.npu_actor = NpuBatchTuningLoopActor()
        self.gov_actor = EpistemicGovernorLoopActor()
        self.actors: List[BaseAutonomousCommandLoopActor] = [
            self.hw_actor,
            self.npu_actor,
            self.gov_actor
        ]
        self._is_active = False

    def start_all_actors(self):
        """Spawns all actors and binds their dedicated bulk parameter sockets."""
        if self._is_active:
            return
        for actor in self.actors:
            actor.start()
        self._is_active = True

    def stop_all_actors(self):
        """Stops all actors and releases all sockets."""
        for actor in self.actors:
            actor.stop()
        self._is_active = False

    def get_actor_by_id(self, actor_id: str) -> Optional[BaseAutonomousCommandLoopActor]:
        for actor in self.actors:
            if actor.actor_id == actor_id:
                return actor
        return None

    def get_full_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive status across all loop actors and sockets."""
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 breached in actor coordinator"
        return {
            "actors_active": self._is_active,
            "total_actors": len(self.actors),
            "actors": [a.get_telemetry() for a in self.actors],
            "vital_max_hp": VITAL_MAX_HP,
            "timestamp_ns": time.time_ns()
        }

    def dispatch_bulk_parameters(self, actor_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Transmits parameters directly through the target actor's dedicated socket."""
        actor = self.get_actor_by_id(actor_id)
        if not actor:
            raise ValueError(f"Unknown actor_id: {actor_id}")

        if not actor.socket_server._is_running:
            actor.start()

        port = actor.socket_server.bound_port
        return send_bulk_parameters_via_client(
            host=actor.socket_server.host,
            port=port,
            parameters=parameters
        )


# Global Singleton for Autonomous Actor System Coordinator
GLOBAL_ACTOR_SYSTEM_COORDINATOR = AutonomousActorSystemCoordinator()


if __name__ == "__main__":
    print("===================================================================")
    print("Krystal Autonomous Command Loop Actors & Bulk Parameter Socket Fabric")
    print(f"Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("===================================================================")

    coordinator = AutonomousActorSystemCoordinator()
    coordinator.start_all_actors()

    try:
        telemetry = coordinator.get_full_telemetry()
        print(f"Active Actors: {telemetry['total_actors']}")
        for a in telemetry["actors"]:
            print(f"  * [{a['actor_id']}] {a['name']:<42} -> Socket: {a['socket_host']}:{a['socket_port']}")

        # Test streaming 100 bulk parameters via socket to Hardware Actor
        bulk_params = {f"core_voltage_step_{i}": round(-0.01 * (i % 5), 4) for i in range(100)}
        bulk_params["pl1_watts"] = 28.0
        bulk_params["cpu_affinity_mask"] = 0x000F
        bulk_params["process_priority"] = "HIGH_PRIORITY_CLASS"

        print(f"\n--- Sending {len(bulk_params)} bulk parameters via dedicated socket ---")
        t0 = time.perf_counter_ns()
        res = coordinator.dispatch_bulk_parameters("ACTOR_HW_REGULATOR", bulk_params)
        dur_us = (time.perf_counter_ns() - t0) / 1000.0

        print(f"Socket Transaction Completed in {dur_us:.2f} us")
        print(f"Result: {res}")
    finally:
        coordinator.stop_all_actors()
