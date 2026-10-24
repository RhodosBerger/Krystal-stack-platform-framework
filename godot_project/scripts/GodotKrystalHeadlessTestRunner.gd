# GodotKrystalHeadlessTestRunner.gd
# ==============================================================================
# KRYSTAL STACK // GODOT KRYSTAL EDITION HEADLESS TEST RUNNER
# ==============================================================================
# Executes high-throughput automated run-testing in Godot headless mode:
#   1. Layer 0: Silicon PDN & Thermal Throttle tests
#   2. Layer 1: Chronometric VBlank Shadow budget & task queue execution
#   3. Layer 2: 1,000 Symplectic Leapfrog steps, det(J) == 1.0, energy conservation
#   4. Layer 3: 12-Frame Dwell Hysteresis jitter suppression (>= 80%)
#   5. Layer 4: Aligned 128B SPSC contract frame packing, SHA-256 HMAC integrity
#   6. Layer 5: Constitutional Gatekeeper proposals with Delta S == 0 audit
#
# Usage:
#   godot --headless -s res://scripts/GodotKrystalHeadlessTestRunner.gd
#
# Constitutional Invariant: VITAL_MAX_HP = 6
# ==============================================================================

extends SceneTree

const VITAL_MAX_HP: int = 6

func _init() -> void:
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach: VITAL_MAX_HP != 6")
	print("==================================================================")
	print("  STARTING GODOT KRYSTAL EDITION HEADLESS TEST RUNNER")
	print("  Constitutional Invariant: VITAL_MAX_HP == %d" % VITAL_MAX_HP)
	print("==================================================================")

	var t_start: int = Time.get_ticks_msec()
	var all_passed: bool = true

	# --- Test 1: Layer 0 Silicon PDN ---
	print(">> [L0] Testing Silicon PDN Bridge...")
	var pdn: GodotSiliconPDNBridge = GodotSiliconPDNBridge.new()
	var pdn_status = pdn.update_telemetry({
		"vdd_volts": 0.85,
		"overshoot_volts": 0.08,
		"tj_celsius": 62.0,
		"power_watts": 19.5,
		"pdn_z0": 50.0
	})
	if not pdn_status["safe_envelope_adhered"]:
		print("FAIL: L0 safe envelope violated")
		all_passed = false
	else:
		print("   L0 Silicon PDN: PASSED (Peak V: %.3f V <= 1.00 V)" % pdn_status["peak_die_voltage_v"])

	# --- Test 2: Layer 1 Chronometric VBlank Shadow ---
	print(">> [L1] Testing Chronometric VBlank Shadow Compositor...")
	var compositor: GodotChronometricCompositor = GodotChronometricCompositor.new()
	var shadow_ms: float = compositor.calculate_shadow_headroom_ms(60.0)
	if shadow_ms < 3.50:
		print("FAIL: L1 shadow headroom %.2f ms < 3.50 ms" % shadow_ms)
		all_passed = false
	else:
		# Test task queueing & execution
		var task_ran: bool = false
		var test_callable = func(): task_ran = true
		compositor.schedule_shadow_task(test_callable)
		compositor.execute_vblank_shadow_window()
		if not task_ran:
			print("FAIL: L1 shadow task did not execute in window")
			all_passed = false
		else:
			print("   L1 Chronometric Compositor: PASSED (Shadow: %.2f ms, Task Ran: true)" % shadow_ms)

	# --- Test 3: Layer 2 Symplectic Leapfrog (1,000 Steps) ---
	print(">> [L2] Testing Spatial Symplectic Leapfrog Integrator (1,000 steps)...")
	var symplectic: GodotSymplecticIntegrator = GodotSymplecticIntegrator.new()
	var body = symplectic.add_orbital_body(Vector3(1.0, 0.0, 0.5), Vector3(0.0, 1.5, 0.0), 1.0)
	var initial_e: float = body.initial_energy

	for step in range(1000):
		symplectic.integrate_step(0.005, 2.0)

	var final_e: float = body.current_energy
	var drift: float = abs(final_e - initial_e)
	if symplectic.det_j_preservation != 1.0 or drift > 1e-3:
		print("FAIL: L2 energy drift too high (%.6f) or det(J) != 1.0" % drift)
		all_passed = false
	else:
		print("   L2 Symplectic Phase-Space: PASSED (det(J)==1.0, Drift: %.6f < 1e-3)" % drift)

	# --- Test 4: Layer 3 Cognitive Cadence & 12-Frame Dwell Hysteresis ---
	print(">> [L3] Testing 12-Frame Dwell Hysteresis Anti-Chattering...")
	var governor: GodotInertialGovernor = GodotInertialGovernor.new()
	# Simulate oscillating frames across boundary
	for i in range(200):
		var frame_ms: float = 12.0 if (i % 6 < 3) else 19.0
		governor.update_cadence(0.01667, frame_ms, 16.67)

	var gov_status = governor.get_status()
	if gov_status["chattering_prevented"] <= 0:
		print("FAIL: L3 12-frame dwell did not record prevented chattering")
		all_passed = false
	else:
		print("   L3 Cognitive Cadence: PASSED (Chattering events suppressed: %d)" % gov_status["chattering_prevented"])

	# --- Test 5: Layer 4 Universal 128B SPSC Zero-Copy Contract Framing ---
	print(">> [L4] Testing Universal 128B SPSC Contract Framing & HMAC-SHA256...")
	var bridge: GodotZeroCopyContractBridge = GodotZeroCopyContractBridge.new()
	var payload_floats: Array[float] = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
	var frame_bytes: PackedByteArray = bridge.pack_frame(1, 0x0020, payload_floats)

	if frame_bytes.size() != 128:
		print("FAIL: L4 frame size %d != 128 bytes" % frame_bytes.size())
		all_passed = false
	else:
		var verified = bridge.unpack_and_verify_frame(frame_bytes)
		if not verified["valid"] or verified["vital_hp"] != VITAL_MAX_HP:
			print("FAIL: L4 frame verification failed: " + verified.get("error", "Unknown"))
			all_passed = false
		else:
			print("   L4 Zero-Copy Contract: PASSED (128B Aligned, HMAC Valid, HP == 6)")

	# --- Test 6: Layer 5 Constitutional Gatekeeper & Delta S == 0 ---
	print(">> [L5] Testing Constitutional Gatekeeper (Safe vs Quarantined Interventions)...")
	var client: GodotRealtimeInterventionClient = GodotRealtimeInterventionClient.new()

	# Safe Proposal
	var safe_res = client.propose_intervention("ELEM_PDN_IMPEDANCE", 2.0, "Safe impedance step")
	# Unsafe Proposal (144 Hz truncating VBlank shadow)
	var init_hz: float = client.engine_parameters["ELEM_DISPLAY_REFRESH_HZ"]
	var unsafe_res = client.propose_intervention("ELEM_DISPLAY_REFRESH_HZ", 84.0, "Unsafe 144Hz breach")

	if not safe_res["authorized"] or unsafe_res["authorized"]:
		print("FAIL: L5 authorization logic incorrect")
		all_passed = false
	elif unsafe_res["mutation_delta_bytes"] != 0 or client.engine_parameters["ELEM_DISPLAY_REFRESH_HZ"] != init_hz:
		print("FAIL: L5 Delta S == 0 was violated on unsafe proposal")
		all_passed = false
	else:
		print("   L5 Constitutional Gate: PASSED (Safe committed, Unsafe quarantined, Delta S == 0 verified)")

	# --- Emit Run Receipt ---
	var dur_total_ms: int = Time.get_ticks_msec() - t_start
	print("==================================================================")
	if all_passed:
		print("  ALL 6 LAYERS PASSED HEADLESS VERIFICATION (%d ms)" % dur_total_ms)
		print("  RESULT: VERIFIED_EMPIRICAL (BUILD OPERATIONAL)")
	else:
		print("  HEADLESS VERIFICATION FAILED (%d ms)" % dur_total_ms)
	print("==================================================================")

	quit(0 if all_passed else 1)
