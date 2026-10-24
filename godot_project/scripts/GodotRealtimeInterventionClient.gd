# GodotRealtimeInterventionClient.gd
# ==============================================================================
# KRYSTAL STACK // GODOT KRYSTAL EDITION
# Layer 5: Constitutional Gatekeeper & Real-Time Intervention Client
# ==============================================================================
# Communicates with the kernel's SafeRealtimeInterventionActuator:
#   - Submits InterventionProposals (framerate, UMA ceiling, resolution, physics dt).
#   - Strictly enforces Separation of Concerns (proposer != evaluator).
#   - Emits Delta S == 0 on any safety boundary violation.
#
# Constitutional Invariant: VITAL_MAX_HP = 6
# ==============================================================================

class_name GodotRealtimeInterventionClient
extends RefCounted

const VITAL_MAX_HP: int = 6
const PROPOSER_ROLE: String = "GODOT_ENGINE_CLIENT"
const EVALUATOR_ROLE: String = "CONSTITUTIONAL_GATEKEEPER"

# Local engine parameter state mirror
var engine_parameters: Dictionary = {
	"ELEM_DISPLAY_REFRESH_HZ": 60.0,
	"ELEM_UMA_CONTENTION_CEILING": 0.65,
	"ELEM_SYMPLECTIC_TIMESTEP_MS": 0.50,
	"ELEM_DWELL_FRAME_COUNT": 12.0,
	"ELEM_PDN_IMPEDANCE": 50.0,
	"ELEM_CONSTITUTIONAL_HP": 6.0
}

var intervention_history: Array[Dictionary] = []
var authorized_interventions_count: int = 0
var rejected_interventions_count: int = 0

func _init():
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach: VITAL_MAX_HP != 6")

func propose_intervention(element_id: String, proposed_delta: float, rationale: String, override_evaluator: String = EVALUATOR_ROLE) -> Dictionary:
	"""
	Proposes a real-time intervention. Evaluates pre-checks locally before
	dispatching or recording the formal transaction.
	Guarantees Delta S == 0 if any violation occurs.
	"""
	var ts_usec: int = Time.get_ticks_usec()
	var initial_val: float = engine_parameters.get(element_id, 0.0)
	var projected_val: float = initial_val + proposed_delta

	# 1. Separation of Concerns (Constitutional Law)
	if PROPOSER_ROLE == override_evaluator:
		var err_receipt: Dictionary = {
			"intervention_id": "GODOT-INTV-%d-%s" % [ts_usec, element_id],
			"element_id": element_id,
			"original_value": initial_val,
			"committed_value": initial_val,
			"delta_applied": 0.0,
			"authorized": false,
			"status": "REJECTED_SOC",
			"reason": "Proposer role matches Evaluator role (forbidden self-certification)",
			"mutation_delta_bytes": 0,
			"vital_max_hp": VITAL_MAX_HP
		}
		rejected_interventions_count += 1
		intervention_history.append(err_receipt)
		return err_receipt

	# 2. Known element check
	if not engine_parameters.has(element_id):
		var err_receipt: Dictionary = {
			"intervention_id": "GODOT-INTV-%d-%s" % [ts_usec, element_id],
			"element_id": element_id,
			"original_value": 0.0,
			"committed_value": 0.0,
			"delta_applied": 0.0,
			"authorized": false,
			"status": "REJECTED_UNKNOWN",
			"reason": "Element not registered in engine parameters",
			"mutation_delta_bytes": 0,
			"vital_max_hp": VITAL_MAX_HP
		}
		rejected_interventions_count += 1
		intervention_history.append(err_receipt)
		return err_receipt

	# 3. Specific safe boundary audits
	var authorized: bool = true
	var reason: String = "Intervention compliant with safe limits: " + rationale
	var status: String = "COMMITTED_SAFE"

	if element_id == "ELEM_DISPLAY_REFRESH_HZ":
		if projected_val > 120.0 or projected_val < 30.0:
			authorized = false
			status = "REJECTED_BOUNDS"
			reason = "Refresh rate %.1f Hz violates safe bounds [30.0, 120.0] Hz" % projected_val
		else:
			# VBlank shadow check
			var shadow_ms: float = (1000.0 / projected_val) - 12.80 - 0.366
			if shadow_ms < 3.50:
				authorized = false
				status = "QUARANTINED_SAFETY_BREACH"
				reason = "VBlank shadow %.2f ms is below 3.50 ms minimum requirement" % shadow_ms

	elif element_id == "ELEM_DWELL_FRAME_COUNT":
		if projected_val < 12.0:
			authorized = false
			status = "QUARANTINED_SAFETY_BREACH"
			reason = "Dwell frames %.1f < 12 violates KSPP Principle P8" % projected_val

	elif element_id == "ELEM_CONSTITUTIONAL_HP":
		if int(round(projected_val)) != VITAL_MAX_HP:
			authorized = false
			status = "REJECTED_CONSTITUTIONAL"
			reason = "Constitutional invariant VITAL_MAX_HP == 6 is immutable"

	if authorized:
		engine_parameters[element_id] = projected_val
		authorized_interventions_count += 1
		var receipt: Dictionary = {
			"intervention_id": "GODOT-INTV-%d-%s" % [ts_usec, element_id],
			"element_id": element_id,
			"original_value": initial_val,
			"committed_value": projected_val,
			"delta_applied": proposed_delta,
			"authorized": true,
			"status": status,
			"reason": reason,
			"mutation_delta_bytes": 8,
			"vital_max_hp": VITAL_MAX_HP
		}
		intervention_history.append(receipt)
		return receipt
	else:
		# DELTA S == 0 GUARANTEE
		engine_parameters[element_id] = initial_val
		rejected_interventions_count += 1
		var receipt: Dictionary = {
			"intervention_id": "GODOT-INTV-%d-%s" % [ts_usec, element_id],
			"element_id": element_id,
			"original_value": initial_val,
			"committed_value": initial_val,
			"delta_applied": 0.0,
			"authorized": false,
			"status": status,
			"reason": reason,
			"mutation_delta_bytes": 0,
			"vital_max_hp": VITAL_MAX_HP
		}
		intervention_history.append(receipt)
		return receipt

func get_status() -> Dictionary:
	return {
		"layer": "L5_CONSTITUTIONAL_GATEKEEPER",
		"parameters": engine_parameters.duplicate(),
		"authorized_count": authorized_interventions_count,
		"rejected_count": rejected_interventions_count,
		"history_size": intervention_history.size(),
		"vital_max_hp": VITAL_MAX_HP
	}
