package dsc

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestOhmLawPhysicsAndZeroDivisionGuard(t *testing.T) {
	state := NewEngineState()

	target := &Entity{
		ID:             "target_turret",
		Name:           "Citadel Defense Turret",
		Coord:          HexCoord{Q: 1, R: 0},
		Layer:          LayerCenter,
		HP:             6,
		BaseResistance: 2.0,
		Conduct: ConductFlags{
			IsActive:    true,
			IsLocked:    false,
			Confidence:  1.0,
			IsHazardous: false,
			VitalHP:     6,
			VitalMaxHP:  6,
		},
	}
	if err := state.RegisterEntity(target); err != nil {
		t.Fatalf("failed to register entity: %v", err)
	}

	// Case 1: Standard Ohm's Law (V = 6.0, R = 2.0 -> I = 3.0, damage = 3, HP = 6 - 3 = 3)
	instr := EngineInstruction{
		Type:           "APPLY_OHM_VOLTAGE",
		SourceID:       "attacker_spark",
		TargetID:       "target_turret",
		AppliedVoltage: 6.0,
		Confidence:     0.95,
		IsHazardous:    false,
	}

	_, res := ResolveEngineState(state, instr)
	if res.Status != "OK" {
		t.Fatalf("expected status OK, got %s", res.Status)
	}
	if res.CalculatedCurrent != 3.0 {
		t.Errorf("expected current 3.0, got %f", res.CalculatedCurrent)
	}
	if res.DamageDealt != 3 {
		t.Errorf("expected damage 3, got %d", res.DamageDealt)
	}
	if res.TargetNewHP != 3 {
		t.Errorf("expected new HP 3, got %d", res.TargetNewHP)
	}

	// Case 2: Zero-division guard (Base resistance = 0.0, effective R clamped to MinEffectiveResistance = 0.1)
	target.BaseResistance = 0.0
	target.HP = 6
	instrZero := EngineInstruction{
		Type:           "APPLY_OHM_VOLTAGE",
		SourceID:       "attacker_spark",
		TargetID:       "target_turret",
		AppliedVoltage: 0.5,
		Confidence:     0.90,
		IsHazardous:    false,
	}
	_, resZero := ResolveEngineState(state, instrZero)
	if resZero.EffectiveResistance < MinEffectiveResistance {
		t.Errorf("expected effective resistance >= %f, got %f", MinEffectiveResistance, resZero.EffectiveResistance)
	}
	expectedCurrent := 0.5 / 0.1 // 5.0
	if math.Abs(resZero.CalculatedCurrent-expectedCurrent) > 1e-4 {
		t.Errorf("expected current %f, got %f", expectedCurrent, resZero.CalculatedCurrent)
	}
}

func TestSaturationArithmeticAndMemoryIntegrity(t *testing.T) {
	state := NewEngineState()

	target := &Entity{
		ID:             "aegis_core",
		Coord:          HexCoord{Q: 0, R: 0},
		Layer:          LayerCenter,
		HP:             4,
		BaseResistance: 1.0,
		Conduct: ConductFlags{
			IsActive:    true,
			Confidence:  1.0,
			VitalHP:     4,
			VitalMaxHP:  VitalMaxHP,
		},
	}
	_ = state.RegisterEntity(target)

	// Massive over-kill voltage (V = 100.0, R = 1.0 -> damage = 100)
	// Saturation arithmetic must clamp HP strictly to 0 (no negative underflow)
	instrOverkill := EngineInstruction{
		Type:           "APPLY_OHM_VOLTAGE",
		SourceID:       "hyper_cannon",
		TargetID:       "aegis_core",
		AppliedVoltage: 100.0,
		Confidence:     1.0,
	}

	_, resOverkill := ResolveEngineState(state, instrOverkill)
	if resOverkill.TargetNewHP != 0 {
		t.Errorf("expected HP clamped to 0, got %d", resOverkill.TargetNewHP)
	}
	if target.Conduct.IsActive {
		t.Errorf("expected entity to be inactive when neutralized")
	}

	// Verify target was removed from CenterOccupancy
	if _, occupied := state.CenterOccupancy[target.Coord.Key()]; occupied {
		t.Errorf("expected neutralized entity to be vacated from CenterOccupancy")
	}
}

func TestLIFOInterruptStack(t *testing.T) {
	state := NewEngineState()

	target := &Entity{
		ID:             "shielded_standee",
		Coord:          HexCoord{Q: 2, R: 2},
		Layer:          LayerCenter,
		HP:             6,
		BaseResistance: 1.0,
		Conduct: ConductFlags{
			IsActive:    true,
			Confidence:  1.0,
			VitalHP:     6,
			VitalMaxHP:  VitalMaxHP,
		},
	}
	_ = state.RegisterEntity(target)

	// Register 2 reactions in Interrupt Stack:
	// First interrupt: Reactive armor (+2.0 resistance)
	state.InterruptStack = append(state.InterruptStack, InterruptRecord{
		ID:              "int_1",
		SourceEntityID:  "support_drone",
		ActionType:      "REACTIVE_ARMOR",
		ResistanceDelta: 2.0,
		VoltageDelta:    0.0,
		IssuedTick:      state.Tick,
		ValidTicks:      5,
	})

	// Second interrupt (LIFO top): Grounding field (+3.0 resistance, -1.0 voltage)
	state.InterruptStack = append(state.InterruptStack, InterruptRecord{
		ID:              "int_2",
		SourceEntityID:  "grounding_pylon",
		ActionType:      "GROUNDING",
		ResistanceDelta: 3.0,
		VoltageDelta:    -1.0,
		IssuedTick:      state.Tick,
		ValidTicks:      5,
	})

	// Now apply 13.0 Volts
	// Total R = Base(1.0) + Int1(2.0) + Int2(3.0) = 6.0
	// Total V = 13.0 - 1.0 = 12.0
	// Current I = 12.0 / 6.0 = 2.0 -> damage = 2 -> HP = 6 - 2 = 4
	instr := EngineInstruction{
		Type:           "APPLY_OHM_VOLTAGE",
		SourceID:       "arc_strike",
		TargetID:       "shielded_standee",
		AppliedVoltage: 13.0,
		Confidence:     0.95,
	}

	_, res := ResolveEngineState(state, instr)
	if res.InterruptsProcessed != 2 {
		t.Errorf("expected 2 interrupts processed, got %d", res.InterruptsProcessed)
	}
	if res.EffectiveResistance != 6.0 {
		t.Errorf("expected effective resistance 6.0, got %f", res.EffectiveResistance)
	}
	if res.CalculatedCurrent != 2.0 {
		t.Errorf("expected calculated current 2.0, got %f", res.CalculatedCurrent)
	}
	if res.TargetNewHP != 4 {
		t.Errorf("expected new HP 4, got %d", res.TargetNewHP)
	}
	if len(state.InterruptStack) != 0 {
		t.Errorf("expected interrupt stack to be cleared after resolution")
	}
}

func TestPauliExclusionGuard(t *testing.T) {
	state := NewEngineState()

	// Entity 1 on (0, 0) LayerCenter
	e1 := &Entity{
		ID:    "unit_alpha",
		Coord: HexCoord{Q: 0, R: 0},
		Layer: LayerCenter,
		HP:    6,
	}
	if err := state.RegisterEntity(e1); err != nil {
		t.Fatalf("unexpected error registering e1: %v", err)
	}

	// Entity 2 tries to be placed on same cell (0, 0) LayerCenter -> MUST FAIL
	e2 := &Entity{
		ID:    "unit_beta",
		Coord: HexCoord{Q: 0, R: 0},
		Layer: LayerCenter,
		HP:    6,
	}
	err := state.RegisterEntity(e2)
	if err == nil {
		t.Fatalf("expected Pauli Exclusion violation when placing e2 on (0,0), got nil")
	}

	// Entity 3 tries to MOVE onto (0, 0) -> MUST TRIGGER PAULI_VIOLATION
	e3 := &Entity{
		ID:    "unit_gamma",
		Coord: HexCoord{Q: 1, R: 0},
		Layer: LayerCenter,
		HP:    6,
		Conduct: ConductFlags{
			IsActive:   true,
			Confidence: 1.0,
			VitalHP:    6,
			VitalMaxHP: VitalMaxHP,
		},
	}
	_ = state.RegisterEntity(e3)

	moveInstr := EngineInstruction{
		Type:        "MOVE_ENTITY",
		SourceID:    "unit_gamma",
		TargetCoord: HexCoord{Q: 0, R: 0}, // Already occupied by unit_alpha
		Confidence:  1.0,
	}

	_, resMove := ResolveEngineState(state, moveInstr)
	if resMove.Status != "PAULI_VIOLATION" {
		t.Errorf("expected status PAULI_VIOLATION, got %s", resMove.Status)
	}
	// Verify unit_gamma stayed at (1, 0)
	if e3.Coord.Q != 1 || e3.Coord.R != 0 {
		t.Errorf("expected unit_gamma to remain at (1, 0), got (%d, %d)", e3.Coord.Q, e3.Coord.R)
	}
}

func TestConductCodeValidationAndDeterministicFallback(t *testing.T) {
	state := NewEngineState()

	// Case 1: Low confidence (< 0.85) triggers fallback
	instrLowConf := EngineInstruction{
		Type:        "APPLY_OHM_VOLTAGE",
		SourceID:    "untrusted_agent",
		TargetID:    "any_target",
		Confidence:  0.72, // Below 0.85
		IsHazardous: false,
	}
	_, resLowConf := ResolveEngineState(state, instrLowConf)
	if resLowConf.Status != "FALLBACK_TRIGGERED" {
		t.Errorf("expected FALLBACK_TRIGGERED for low confidence, got %s", resLowConf.Status)
	}

	// Case 2: Hazardous operation triggers fallback
	instrHazard := EngineInstruction{
		Type:        "APPLY_OHM_VOLTAGE",
		SourceID:    "thermal_runaway_agent",
		TargetID:    "any_target",
		Confidence:  0.99,
		IsHazardous: true, // Hazard detected
	}
	_, resHazard := ResolveEngineState(state, instrHazard)
	if resHazard.Status != "FALLBACK_TRIGGERED" {
		t.Errorf("expected FALLBACK_TRIGGERED for hazard, got %s", resHazard.Status)
	}
}

func TestGeneratePackedInstanceBuffer(t *testing.T) {
	state := NewEngineState()

	e := &Entity{
		ID:             "vulkan_standee",
		Coord:          HexCoord{Q: 2, R: 3},
		Layer:          LayerCenter,
		HP:             6,
		BaseResistance: 2.0,
		Conduct: ConductFlags{
			IsActive:    true,
			IsLocked:    true,
			Confidence:  0.95,
			IsHazardous: false,
			VitalHP:     6,
			VitalMaxHP:  6,
		},
	}
	_ = state.RegisterEntity(e)

	buf := GeneratePackedInstanceBuffer(state)
	if len(buf) != 96 {
		t.Fatalf("expected packed buffer of 96 bytes (24 float32s), got %d bytes", len(buf))
	}

	// Verify world position in matrix indices 12, 13, 14 (offsets 48, 52, 56)
	posX := math.Float32frombits(binary.LittleEndian.Uint32(buf[48:52]))
	posY := math.Float32frombits(binary.LittleEndian.Uint32(buf[52:56]))
	posZ := math.Float32frombits(binary.LittleEndian.Uint32(buf[56:60]))

	// worldX = 1.0 * sqrt(3) * (2 + 3/2) = 1.7320508 * 3.5 ~= 6.062177
	expectedX := float32(math.Sqrt(3.0) * 3.5)
	if math.Abs(float64(posX-expectedX)) > 1e-3 {
		t.Errorf("expected posX %f, got %f", expectedX, posX)
	}
	if posY != 1.0 { // LayerCenter = 2 * 0.5 = 1.0
		t.Errorf("expected posY 1.0, got %f", posY)
	}
	// worldZ = 1.0 * 1.5 * 3 = 4.5
	if posZ != 4.5 {
		t.Errorf("expected posZ 4.5, got %f", posZ)
	}

	// Verify Layer ID at offset 80 (float 20)
	layerID := math.Float32frombits(binary.LittleEndian.Uint32(buf[80:84]))
	if layerID != float32(LayerCenter) {
		t.Errorf("expected layerID %d, got %f", LayerCenter, layerID)
	}
}
