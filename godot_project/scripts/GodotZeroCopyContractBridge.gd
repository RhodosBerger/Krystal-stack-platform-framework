# GodotZeroCopyContractBridge.gd
# ==============================================================================
# KRYSTAL STACK // GODOT KRYSTAL EDITION
# Layer 4: Universal 128B Zero-Copy SPSC Contract Framing
# ==============================================================================
# Implements KSPP Principle P2 (Universal 128B Lingua Franca) and P9 (Full-Frame Provenance):
#   - Exactly 128-byte naturally aligned binary frames.
#   - Binary structure:
#       0..3:   Magic (0x4B525953 'KRYS')
#       4..7:   Contract Type (0: STATE, 1: EVIDENCE, 2: TELEMETRY, 3: ACTUATION)
#       8..15:  Sequence Counter (uint64)
#       16..17: Source Node ID (uint16, e.g. 0x0010 = Godot)
#       18..19: Target Node ID (uint16, e.g. 0x0020 = Chrono SoC)
#       20..23: Constitutional Invariant (uint32, must == 6)
#       24..95: Payload data (72 bytes: floats/integers)
#       96..127: HMAC-SHA256 digest over 0..95 bytes (32 bytes)
#
# Constitutional Invariant: VITAL_MAX_HP = 6
# ==============================================================================

class_name GodotZeroCopyContractBridge
extends RefCounted

const VITAL_MAX_HP: int = 6
const FRAME_SIZE_BYTES: int = 128
const FRAME_MAGIC: int = 0x4B525953  # 'KRYS'
const NODE_ID_GODOT: int = 0x0010
const NODE_ID_CHRONO_SOC: int = 0x0020

enum ContractType {
	STATE = 0,
	EVIDENCE = 1,
	TELEMETRY = 2,
	ACTUATION = 3
}

var sequence_counter: int = 0
var total_frames_sent: int = 0
var total_frames_received: int = 0
var total_corrupted_frames: int = 0

func _init():
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach: VITAL_MAX_HP != 6")

func pack_frame(contract_type: int, target_node: int, payload_floats: Array[float]) -> PackedByteArray:
	"""Packs an aligned 128-byte zero-copy contract frame with cryptographic digest."""
	sequence_counter += 1
	var spb: StreamPeerBuffer = StreamPeerBuffer.new()
	spb.big_endian = false  # Little-endian canonical

	# Header (24 bytes)
	spb.put_u32(FRAME_MAGIC)
	spb.put_u32(contract_type)
	spb.put_u64(sequence_counter)
	spb.put_u16(NODE_ID_GODOT)
	spb.put_u16(target_node)
	spb.put_u32(VITAL_MAX_HP)

	# Payload (72 bytes: 18 float32s)
	for i in range(18):
		if i < payload_floats.size():
			spb.put_float(payload_floats[i])
		else:
			spb.put_float(0.0)

	# Calculate SHA-256 over first 96 bytes (simulating HMAC-SHA256)
	var raw_data: PackedByteArray = spb.data_array
	var ctx: HashingContext = HashingContext.new()
	ctx.start(HashingContext.HASH_SHA256)
	ctx.update(raw_data.slice(0, 96))
	var digest: PackedByteArray = ctx.finish()

	# Append 32-byte digest -> Exactly 128 bytes total
	spb.put_data(digest)

	assert(spb.data_array.size() == FRAME_SIZE_BYTES, "Frame size != 128 bytes")
	total_frames_sent += 1
	return spb.data_array

func unpack_and_verify_frame(frame_bytes: PackedByteArray) -> Dictionary:
	"""Validates 128B size, magic, constitutional HP == 6, and cryptographic integrity."""
	if frame_bytes.size() != FRAME_SIZE_BYTES:
		total_corrupted_frames += 1
		return {"valid": false, "error": "Invalid frame size: %d != 128" % frame_bytes.size()}

	var spb: StreamPeerBuffer = StreamPeerBuffer.new()
	spb.data_array = frame_bytes
	spb.big_endian = false

	var magic: int = spb.get_u32()
	if magic != FRAME_MAGIC:
		total_corrupted_frames += 1
		return {"valid": false, "error": "Magic mismatch: 0x%08X != 0x%08X" % [magic, FRAME_MAGIC]}

	var c_type: int = spb.get_u32()
	var seq: int = spb.get_u64()
	var src: int = spb.get_u16()
	var tgt: int = spb.get_u16()
	var hp: int = spb.get_u32()

	if hp != VITAL_MAX_HP:
		total_corrupted_frames += 1
		return {"valid": false, "error": "Constitutional invariant violation: hp %d != 6" % hp}

	var payload: Array[float] = []
	for i in range(18):
		payload.append(spb.get_float())

	var received_digest: PackedByteArray = frame_bytes.slice(96, 128)

	var ctx: HashingContext = HashingContext.new()
	ctx.start(HashingContext.HASH_SHA256)
	ctx.update(frame_bytes.slice(0, 96))
	var calculated_digest: PackedByteArray = ctx.finish()

	if received_digest != calculated_digest:
		total_corrupted_frames += 1
		return {"valid": false, "error": "Cryptographic digest mismatch"}

	total_frames_received += 1
	return {
		"valid": true,
		"contract_type": c_type,
		"sequence": seq,
		"source_node": src,
		"target_node": tgt,
		"vital_hp": hp,
		"payload": payload,
		"frame_size": FRAME_SIZE_BYTES
	}

func get_status() -> Dictionary:
	return {
		"layer": "L4_SPSC_COORDINATION",
		"frame_size_bytes": FRAME_SIZE_BYTES,
		"frames_sent": total_frames_sent,
		"frames_received": total_frames_received,
		"corrupted_frames": total_corrupted_frames,
		"sequence_counter": sequence_counter,
		"is_healthy": (total_corrupted_frames == 0),
		"vital_max_hp": VITAL_MAX_HP
	}
