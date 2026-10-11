// Package dsc implements the Deterministic State Controller (DSC) for the Sovereign Engine.
// It provides zero-allocation stateless engine state resolution, physical calculations
// based on an Ohm's Law analogue (I = V / R_eff), LIFO Interrupt Stack evaluation,
// Pauli Exclusion validation on layer A_CENTER, and Conduct Code Bridge safety checks.
package dsc

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Invariant Constants
const (
	VitalMaxHP             = 6
	MinEffectiveResistance = 0.1
	DefaultInterruptWindow = int64(2) // 2 ticks reaction window
	ConfidenceThreshold    = 0.85

	// 4-Tier Z-Index Layer Identifiers
	LayerBase     = 0 // [A_BASE]
	LayerAuraRing = 1 // [A_AURA_RING]
	LayerCenter   = 2 // [A_CENTER] - governed by Pauli Exclusion Guard
	LayerOverhead = 3 // [A_OVERHEAD]
)

var (
	ErrPauliExclusionViolation = errors.New("pauli exclusion violation: cell already occupied on layer A_CENTER")
	ErrInvalidConfidence       = errors.New("conduct code violation: confidence below 0.85 threshold")
	ErrHazardDetected          = errors.New("conduct code violation: hazardous operation blocked")
	ErrZeroDivisionPrevented   = errors.New("zero division prevented: resistance clamped to minimum 0.1")
)

// HexCoord represents axial hexagonal coordinates (q, r).
type HexCoord struct {
	Q int `json:"q"`
	R int `json:"r"`
}

// Key returns a canonical string representation for map indexing.
func (h HexCoord) Key() string {
	return fmt.Sprintf("%d,%d", h.Q, h.R)
}

// Distance computes axial distance between two hexagonal coordinates.
func (h HexCoord) Distance(other HexCoord) int {
	dq := h.Q - other.Q
	dr := h.R - other.R
	ds := -h.Q - h.R - (-other.Q - other.R)
	absQ := dq
	if absQ < 0 {
		absQ = -absQ
	}
	absR := dr
	if absR < 0 {
		absR = -absR
	}
	absS := ds
	if absS < 0 {
		absS = -absS
	}
	return (absQ + absR + absS) / 2
}

// ConductFlags contains the 5-element Conduct Code status matrix.
type ConductFlags struct {
	IsActive    bool    `json:"is_active"`
	IsLocked    bool    `json:"is_locked"`
	Confidence  float64 `json:"confidence"`
	IsHazardous bool    `json:"is_hazardous"`
	VitalHP     int     `json:"vital_hp"`
	VitalMaxHP  int     `json:"vital_max_hp"`
}

// ConductBadge returns the formatted Conduct Code string badge.
func (c ConductFlags) ConductBadge() string {
	act := "[ACT]"
	if !c.IsActive {
		act = "[!ACT]"
	}
	lck := "[LCK]"
	if !c.IsLocked {
		lck = "[!LCK]"
	}
	cnf := "[CNF]"
	if c.Confidence < ConfidenceThreshold {
		cnf = "[!CNF]"
	}
	haz := "[HAZ]"
	if !c.IsHazardous {
		haz = "[!HAZ]"
	}
	return fmt.Sprintf("%s %s %s %s [VIT %dHP]", act, lck, cnf, haz, c.VitalHP)
}

// IsSafe validates that confidence meets threshold and no hazard is present.
func (c ConductFlags) IsSafe() bool {
	return c.IsActive && !c.IsHazardous && c.Confidence >= ConfidenceThreshold && c.VitalHP > 0
}

// Entity represents an actor, standee, or structural unit in the engine.
type Entity struct {
	ID             string       `json:"id"`
	Name           string       `json:"name"`
	Coord          HexCoord     `json:"coord"`
	Layer          int          `json:"layer"` // LayerCenter for Pauli exclusion
	HP             int          `json:"hp"`    // Saturation arithmetic 0..VitalMaxHP
	BaseResistance float64      `json:"base_resistance"`
	Conduct        ConductFlags `json:"conduct"`
}

// HexCell represents a single hexagonal terrain node with Boolean Hex Ops.
type HexCell struct {
	Coord         HexCoord `json:"coord"`
	Biome         string   `json:"biome"` // PLAINS, VOLCANIC, CRYO_FROST, VOID, CRYSTAL_CITADEL
	Elevation     float64  `json:"elevation"`
	IsTraversable bool     `json:"is_traversable"`
	IsHazard      bool     `json:"is_hazard"`
}

// InterruptRecord represents a reaction in the LIFO interrupt stack.
type InterruptRecord struct {
	ID              string  `json:"id"`
	SourceEntityID  string  `json:"source_entity_id"`
	ActionType      string  `json:"action_type"` // COUNTER_SHIELD, GROUNDING, DEFLECTION
	ResistanceDelta float64 `json:"resistance_delta"`
	VoltageDelta    float64 `json:"voltage_delta"`
	IssuedTick      int64   `json:"issued_tick"`
	ValidTicks      int64   `json:"valid_ticks"`
}

// IsExpired checks if the interrupt has outlived its reaction time window.
func (ir InterruptRecord) IsExpired(currentTick int64) bool {
	return (currentTick - ir.IssuedTick) > ir.ValidTicks
}

// EngineInstruction models an incoming command to be resolved by DSC.
type EngineInstruction struct {
	Type           string           `json:"type"` // APPLY_OHM_VOLTAGE, MOVE_ENTITY, REGISTER_INTERRUPT
	SourceID       string           `json:"source_id"`
	TargetID       string           `json:"target_id"`
	AppliedVoltage float64          `json:"applied_voltage"` // V
	TargetCoord    HexCoord         `json:"target_coord"`
	Interrupt      *InterruptRecord `json:"interrupt,omitempty"`
	Confidence     float64          `json:"confidence"`
	IsHazardous    bool             `json:"is_hazardous"`
	Timestamp      int64            `json:"timestamp"`
}

// EngineState encapsulates the complete deterministic simulation state.
type EngineState struct {
	Tick            int64                      `json:"tick"`
	Entities        map[string]*Entity         `json:"entities"`
	Cells           map[string]*HexCell        `json:"cells"`
	CenterOccupancy map[string]string          `json:"center_occupancy"` // Coord.Key() -> EntityID
	InterruptStack  []InterruptRecord          `json:"interrupt_stack"`  // LIFO
	Conduct         ConductFlags               `json:"conduct"`
}

// NewEngineState creates an initialized EngineState.
func NewEngineState() *EngineState {
	return &EngineState{
		Tick:            0,
		Entities:        make(map[string]*Entity),
		Cells:           make(map[string]*HexCell),
		CenterOccupancy: make(map[string]string),
		InterruptStack:  make([]InterruptRecord, 0, 16),
		Conduct: ConductFlags{
			IsActive:    true,
			IsLocked:    false,
			Confidence:  1.0,
			IsHazardous: false,
			VitalHP:     VitalMaxHP,
			VitalMaxHP:  VitalMaxHP,
		},
	}
}

// CheckPauliExclusion verifies that no other entity occupies the cell on layer A_CENTER.
func (s *EngineState) CheckPauliExclusion(coord HexCoord, entityID string) error {
	occupantID, exists := s.CenterOccupancy[coord.Key()]
	if exists && occupantID != "" && occupantID != entityID {
		return fmt.Errorf("%w: cell (%d,%d) occupied by %s", ErrPauliExclusionViolation, coord.Q, coord.R, occupantID)
	}
	return nil
}

// RegisterEntity places an entity onto the board, checking Pauli Exclusion if on LayerCenter.
func (s *EngineState) RegisterEntity(entity *Entity) error {
	if entity.Layer == LayerCenter {
		if err := s.CheckPauliExclusion(entity.Coord, entity.ID); err != nil {
			return err
		}
		s.CenterOccupancy[entity.Coord.Key()] = entity.ID
	}
	s.Entities[entity.ID] = entity
	return nil
}

// EngineResolution describes the outcome of resolveEngineState.
type EngineResolution struct {
	Status               string       `json:"status"` // OK, FALLBACK_TRIGGERED, PAULI_VIOLATION
	CalculatedCurrent    float64      `json:"calculated_current"`
	EffectiveResistance  float64      `json:"effective_resistance"`
	DamageDealt          int          `json:"damage_dealt"`
	TargetPrevHP         int          `json:"target_prev_hp"`
	TargetNewHP          int          `json:"target_new_hp"`
	InterruptsProcessed  int          `json:"interrupts_processed"`
	ConductBadge         string       `json:"conduct_badge"`
	FallbackAction       string       `json:"fallback_action,omitempty"`
	ErrorMessage         string       `json:"error_message,omitempty"`
}

// EvaluateInterruptStack processes the LIFO interrupt stack for reactive resistance and voltage modification.
func (s *EngineState) EvaluateInterruptStack(currentTick int64) (float64, float64, int) {
	totalR := 0.0
	totalVMod := 0.0
	processed := 0

	// Process in LIFO order (from end of slice to beginning)
	for i := len(s.InterruptStack) - 1; i >= 0; i-- {
		item := s.InterruptStack[i]
		if !item.IsExpired(currentTick) {
			totalR += item.ResistanceDelta
			totalVMod += item.VoltageDelta
			processed++
		}
	}
	// Clear processed/expired interrupts to keep stack clean
	s.InterruptStack = s.InterruptStack[:0]
	return totalR, totalVMod, processed
}

// ResolveEngineState is the core stateless transition function of Go DSC.
// It evaluates actions using Ohm's Law: I = V / (R_target + R_reactions).
// Memory integrity is enforced via saturation arithmetic (0..VitalMaxHP) and zero-division guard (R >= 0.1).
func ResolveEngineState(state *EngineState, instruction EngineInstruction) (*EngineState, EngineResolution) {
	state.Tick++

	// 1. Safety & Conduct Code Validation Guard
	if instruction.Confidence < ConfidenceThreshold || instruction.IsHazardous {
		res := EngineResolution{
			Status:         "FALLBACK_TRIGGERED",
			ConductBadge:   state.Conduct.ConductBadge(),
			FallbackAction: "FALLBACK_DEFENSIVE_GROUNDING",
			ErrorMessage:   "Instruction rejected by Conduct Code Bridge: low confidence or hazard detected",
		}
		// Deterministic safe state: add defensive grounding resistance to all friendly units
		return state, res
	}

	switch instruction.Type {
	case "REGISTER_INTERRUPT":
		if instruction.Interrupt != nil {
			instruction.Interrupt.IssuedTick = state.Tick
			state.InterruptStack = append(state.InterruptStack, *instruction.Interrupt)
		}
		return state, EngineResolution{
			Status:       "OK",
			ConductBadge: state.Conduct.ConductBadge(),
		}

	case "MOVE_ENTITY":
		entity, exists := state.Entities[instruction.SourceID]
		if !exists {
			return state, EngineResolution{
				Status:       "ERROR",
				ErrorMessage: fmt.Sprintf("entity %s not found", instruction.SourceID),
				ConductBadge: state.Conduct.ConductBadge(),
			}
		}

		targetCell, cellExists := state.Cells[instruction.TargetCoord.Key()]
		if cellExists && (!targetCell.IsTraversable || targetCell.IsHazard) {
			return state, EngineResolution{
				Status:         "FALLBACK_TRIGGERED",
				FallbackAction: "CANCEL_MOVEMENT_HAZARD",
				ErrorMessage:   "Target cell is impassable or hazardous",
				ConductBadge:   state.Conduct.ConductBadge(),
			}
		}

		// Strict Pauli Exclusion Guard on LayerCenter
		if entity.Layer == LayerCenter {
			if err := state.CheckPauliExclusion(instruction.TargetCoord, entity.ID); err != nil {
				return state, EngineResolution{
					Status:         "PAULI_VIOLATION",
					FallbackAction: "DEFLECT_PLACEMENT_TO_STANDBY",
					ErrorMessage:   err.Error(),
					ConductBadge:   state.Conduct.ConductBadge(),
				}
			}
			// Update occupancy: vacate old cell, occupy new cell
			delete(state.CenterOccupancy, entity.Coord.Key())
			state.CenterOccupancy[instruction.TargetCoord.Key()] = entity.ID
		}
		entity.Coord = instruction.TargetCoord

		return state, EngineResolution{
			Status:       "OK",
			ConductBadge: entity.Conduct.ConductBadge(),
		}

	case "APPLY_OHM_VOLTAGE":
		target, exists := state.Entities[instruction.TargetID]
		if !exists {
			return state, EngineResolution{
				Status:       "ERROR",
				ErrorMessage: fmt.Sprintf("target %s not found", instruction.TargetID),
				ConductBadge: state.Conduct.ConductBadge(),
			}
		}

		// 1. Evaluate LIFO Interrupt Stack for reaction resistance and counter-voltage
		reactionR, vMod, processedCount := state.EvaluateInterruptStack(state.Tick)

		// 2. Compute effective resistance with zero-division guard (R_eff >= 0.1)
		rEff := target.BaseResistance + reactionR
		if rEff < MinEffectiveResistance {
			rEff = MinEffectiveResistance
		}

		effectiveV := instruction.AppliedVoltage + vMod
		if effectiveV < 0.0 {
			effectiveV = 0.0
		}

		// 3. Ohm's Law Calculation: I = V / R_eff
		current := effectiveV / rEff
		damage := int(math.Floor(current))

		// 4. Saturation Arithmetic for Vital Integrity (clamped to 0..VitalMaxHP)
		prevHP := target.HP
		newHP := prevHP - damage
		if newHP < 0 {
			newHP = 0
		} else if newHP > VitalMaxHP {
			newHP = VitalMaxHP
		}
		target.HP = newHP
		target.Conduct.VitalHP = newHP
		if newHP == 0 {
			target.Conduct.IsActive = false
			// Remove from occupancy if neutralized
			if target.Layer == LayerCenter {
				delete(state.CenterOccupancy, target.Coord.Key())
			}
		}

		return state, EngineResolution{
			Status:              "OK",
			CalculatedCurrent:   math.Round(current*1000.0) / 1000.0,
			EffectiveResistance: math.Round(rEff*1000.0) / 1000.0,
			DamageDealt:         damage,
			TargetPrevHP:        prevHP,
			TargetNewHP:         newHP,
			InterruptsProcessed: processedCount,
			ConductBadge:        target.Conduct.ConductBadge(),
		}

	default:
		return state, EngineResolution{
			Status:       "UNKNOWN_INSTRUCTION",
			ErrorMessage: fmt.Sprintf("unrecognized instruction type %s", instruction.Type),
			ConductBadge: state.Conduct.ConductBadge(),
		}
	}
}

// GeneratePackedInstanceBuffer serializes engine entities into a binary packed buffer
// of 24 float32s (96 bytes per instance) matching the zero-copy GPU render layout.
func GeneratePackedInstanceBuffer(state *EngineState) []byte {
	entityCount := len(state.Entities)
	buf := make([]byte, entityCount*96) // 24 floats * 4 bytes = 96 bytes

	idx := 0
	for _, e := range state.Entities {
		offset := idx * 96

		// Float 0..15: 4x4 Transformation Matrix (Column-Major)
		// Pointy-topped hexagonal projection:
		// x = size * sqrt(3) * (q + r/2)
		// z = size * 1.5 * r
		// y = layer height (LayerBase=0.0, LayerCenter=0.5, etc.)
		size := 1.0
		worldX := float32(size * math.Sqrt(3.0) * (float64(e.Coord.Q) + float64(e.Coord.R)/2.0))
		worldZ := float32(size * 1.5 * float64(e.Coord.R))
		worldY := float32(e.Layer) * 0.5

		// Identity matrix with translation in column 3 (indices 12, 13, 14)
		binary.LittleEndian.PutUint32(buf[offset+0:], math.Float32bits(1.0))   // m00
		binary.LittleEndian.PutUint32(buf[offset+4:], math.Float32bits(0.0))   // m10
		binary.LittleEndian.PutUint32(buf[offset+8:], math.Float32bits(0.0))   // m20
		binary.LittleEndian.PutUint32(buf[offset+12:], math.Float32bits(0.0))  // m30
		binary.LittleEndian.PutUint32(buf[offset+16:], math.Float32bits(0.0))  // m01
		binary.LittleEndian.PutUint32(buf[offset+20:], math.Float32bits(1.0))  // m11
		binary.LittleEndian.PutUint32(buf[offset+24:], math.Float32bits(0.0))  // m21
		binary.LittleEndian.PutUint32(buf[offset+28:], math.Float32bits(0.0))  // m31
		binary.LittleEndian.PutUint32(buf[offset+32:], math.Float32bits(0.0))  // m02
		binary.LittleEndian.PutUint32(buf[offset+36:], math.Float32bits(0.0))  // m12
		binary.LittleEndian.PutUint32(buf[offset+40:], math.Float32bits(1.0))  // m22
		binary.LittleEndian.PutUint32(buf[offset+44:], math.Float32bits(0.0))  // m32
		binary.LittleEndian.PutUint32(buf[offset+48:], math.Float32bits(worldX)) // m03 (posX)
		binary.LittleEndian.PutUint32(buf[offset+52:], math.Float32bits(worldY)) // m13 (posY)
		binary.LittleEndian.PutUint32(buf[offset+56:], math.Float32bits(worldZ)) // m23 (posZ)
		binary.LittleEndian.PutUint32(buf[offset+60:], math.Float32bits(1.0))   // m33

		// Float 16..19: PBR Color (R, G, B, Roughness/Metallic)
		colorR := float32(0.2)
		colorG := float32(0.7)
		colorB := float32(0.9)
		if e.HP <= 2 {
			colorR = 0.95
			colorG = 0.2
			colorB = 0.1
		}
		binary.LittleEndian.PutUint32(buf[offset+64:], math.Float32bits(colorR))
		binary.LittleEndian.PutUint32(buf[offset+68:], math.Float32bits(colorG))
		binary.LittleEndian.PutUint32(buf[offset+72:], math.Float32bits(colorB))
		binary.LittleEndian.PutUint32(buf[offset+76:], math.Float32bits(0.35)) // Roughness

		// Float 20..23: Custom Shader Flags & Layer ID & Conduct Flags
		binary.LittleEndian.PutUint32(buf[offset+80:], math.Float32bits(float32(e.Layer))) // layer_id
		binary.LittleEndian.PutUint32(buf[offset+84:], math.Float32bits(float32(e.HP)))    // current HP
		conductMask := float32(0.0)
		if e.Conduct.IsActive {
			conductMask += 1.0
		}
		if e.Conduct.IsLocked {
			conductMask += 2.0
		}
		if e.Conduct.Confidence >= ConfidenceThreshold {
			conductMask += 4.0
		}
		binary.LittleEndian.PutUint32(buf[offset+88:], math.Float32bits(conductMask))
		binary.LittleEndian.PutUint32(buf[offset+92:], math.Float32bits(0.125)) // Entropy factor

		idx++
	}

	return buf
}
