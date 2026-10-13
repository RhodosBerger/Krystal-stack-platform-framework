#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR CANONICAL SEMANTIC STATUS PROTOCOL v1.0
==============================================================================
File: tests/test_semantic_status_protocol.py
Description:
    Validates:
      1. 16-bit status code allocation matrix & nibble isolation.
      2. Orthogonal category separation (prevention of category errors like EPI_VETOED).
      3. Strict disambiguation of QUARANTINED across Epistemic, Authority, Paradox, Memory.
      4. Dual-gate authority decomposition f(SafetyGate, EvidenceGate) -> AuthorityState.
      5. Bidirectional wire token serialization (epi:validated, auth:vetoed, der:thermal_headroom).
      6. Generated C ABI header and Rust module integrity.
      7. Constitutional Invariant: VITAL_MAX_HP == 6.
==============================================================================
"""

import unittest

from krystal_kernel.verification.semantic_protocol import (
    VITAL_MAX_HP,
    FAMILY_MASK,
    FAMILY_SHIFT,
    SUBKIND_MASK,
    VALUE_MASK,
    StatusCodeFamily,
    EpistemicState,
    TemporalState,
    AuthorityState,
    SafetyState,
    InvariantState,
    ActuationState,
    TransactionState,
    ObservationState,
    DerivedState,
    PredictionState,
    MemoryState,
    HardwareState,
    ParadoxState,
    MerkleState,
    PolicyState,
    ALL_ENUM_CLASSES,
    CODE_TO_ENUM_MAP,
    NAME_TO_CODE_MAP,
    CODE_TO_WIRE_MAP,
    WIRE_TO_CODE_MAP,
    get_status_family,
    to_wire_token,
    from_wire_token,
    assert_orthogonal_category,
    evaluate_dual_gate,
    generate_c_header,
    generate_rust_module,
    generate_css_badge_classes
)


class TestSemanticStatusProtocol(unittest.TestCase):
    """Test suite for the Canonical Semantic Status Protocol v1.0."""

    def test_constitutional_invariant(self):
        """Invariant: VITAL_MAX_HP == 6."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(InvariantState.INV_NOMINAL_6HP, 0x4006)

    def test_unique_status_codes(self):
        """Every status code across all 15 enum classes must be unique."""
        all_codes = []
        for enum_cls in ALL_ENUM_CLASSES:
            for member in enum_cls:
                all_codes.append(int(member.value))

        self.assertEqual(len(all_codes), len(set(all_codes)), "Status codes must not collide!")
        self.assertGreaterEqual(len(all_codes), 60)

    def test_nibble_family_isolation(self):
        """Each enum class must strictly reside within its assigned top nibble."""
        expected_allocations = {
            EpistemicState: StatusCodeFamily.EPI,
            TemporalState: StatusCodeFamily.TMP,
            AuthorityState: StatusCodeFamily.AUTH,
            SafetyState: StatusCodeFamily.INV,  # Safety filter is sub-kind under Invariants
            InvariantState: StatusCodeFamily.INV,
            ActuationState: StatusCodeFamily.ACT,
            TransactionState: StatusCodeFamily.TXN,
            ObservationState: StatusCodeFamily.OBS,
            DerivedState: StatusCodeFamily.DER,
            PredictionState: StatusCodeFamily.PRED,
            MemoryState: StatusCodeFamily.MEM,
            HardwareState: StatusCodeFamily.HW,
            ParadoxState: StatusCodeFamily.PDX,
            MerkleState: StatusCodeFamily.MRK,
            PolicyState: StatusCodeFamily.POL,
        }

        for enum_cls, expected_family in expected_allocations.items():
            for member in enum_cls:
                code = int(member.value)
                actual_family = get_status_family(code)
                self.assertEqual(
                    actual_family,
                    expected_family,
                    f"Member {member.name} (0x{code:04X}) should have top nibble 0x{expected_family.value:X}, "
                    f"got 0x{actual_family.value:X}"
                )

    def test_category_separation_enforcement(self):
        """Passing an Authority state into an Epistemic category must raise TypeError."""
        # Valid category check
        assert_orthogonal_category(EpistemicState.EPI_VALIDATED, StatusCodeFamily.EPI)
        assert_orthogonal_category(AuthorityState.AUTH_VETOED, StatusCodeFamily.AUTH)

        # Cross-category violation: Attempting to treat AUTH_VETOED as an Epistemic status
        with self.assertRaises(TypeError):
            assert_orthogonal_category(AuthorityState.AUTH_VETOED, StatusCodeFamily.EPI)

        with self.assertRaises(TypeError):
            assert_orthogonal_category(EpistemicState.EPI_FALSIFIED, StatusCodeFamily.AUTH)

    def test_quarantined_disambiguation(self):
        """QUARANTINED must exist as 4 distinct namespaced codes with distinct meanings."""
        epi_q = EpistemicState.EPI_QUARANTINED
        auth_q = AuthorityState.AUTH_QUARANTINED
        pdx_q = ParadoxState.PDX_QUARANTINED
        mem_q = MemoryState.MEM_QUARANTINED

        # Must have different integer values
        codes = {epi_q, auth_q, pdx_q, mem_q}
        self.assertEqual(len(codes), 4, "All 4 QUARANTINED variants must have unique status codes!")

        # Must have different top nibble families
        self.assertEqual(get_status_family(epi_q), StatusCodeFamily.EPI)
        self.assertEqual(get_status_family(auth_q), StatusCodeFamily.AUTH)
        self.assertEqual(get_status_family(pdx_q), StatusCodeFamily.PDX)
        self.assertEqual(get_status_family(mem_q), StatusCodeFamily.MEM)

        # Wire tokens must be distinct
        self.assertEqual(to_wire_token(epi_q), "epi:quarantined")
        self.assertEqual(to_wire_token(auth_q), "auth:quarantined")
        self.assertEqual(to_wire_token(pdx_q), "pdx:quarantined")
        self.assertEqual(to_wire_token(mem_q), "mem:quarantined")

    def test_dual_gate_authority_decomposition(self):
        """Authority verdict is mathematically decomposed from SafetyGate and EvidenceGate."""
        # 1. Safety Breach always vetoes regardless of evidence
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_BREACH, EpistemicState.EPI_VALIDATED),
            AuthorityState.AUTH_VETOED
        )
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_BREACH, EpistemicState.EPI_OBSERVED),
            AuthorityState.AUTH_VETOED
        )

        # 2. Safety Pass + Validated evidence approves commit
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_PASS, EpistemicState.EPI_VALIDATED),
            AuthorityState.AUTH_COMMIT_APPROVED
        )

        # 3. Safety Pass + Contradictory evidence triggers quarantine
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_PASS, EpistemicState.EPI_CONTRADICTORY),
            AuthorityState.AUTH_QUARANTINED
        )
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_PASS, EpistemicState.EPI_QUARANTINED),
            AuthorityState.AUTH_QUARANTINED
        )

        # 4. Safety Pass + Incomplete or predicted evidence defers for more sampling
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_PASS, EpistemicState.EPI_OBSERVED),
            AuthorityState.AUTH_DEFERRED
        )
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_PASS, EpistemicState.EPI_DERIVED),
            AuthorityState.AUTH_DEFERRED
        )
        self.assertEqual(
            evaluate_dual_gate(SafetyState.SAFE_PASS, EpistemicState.EPI_PREDICTED),
            AuthorityState.AUTH_DEFERRED
        )

    def test_wire_serialization_roundtrip(self):
        """Every status code converts to a canonical wire token and back."""
        for code, wire in CODE_TO_WIRE_MAP.items():
            parsed_code = from_wire_token(wire)
            self.assertEqual(parsed_code, code)
            self.assertEqual(to_wire_token(code), wire)

        # Specific wire format samples
        self.assertEqual(to_wire_token(EpistemicState.EPI_VALIDATED), "epi:validated")
        self.assertEqual(to_wire_token(AuthorityState.AUTH_VETOED), "auth:vetoed")
        self.assertEqual(to_wire_token(DerivedState.DER_THERMAL_HEADROOM), "der:thermal_headroom")
        self.assertEqual(to_wire_token(PolicyState.POL_ACTIVE), "pol:active")
        self.assertEqual(to_wire_token(InvariantState.INV_NOMINAL_6HP), "inv:nominal_6hp")

    def test_code_generators(self):
        """C ABI header, Rust module, and CSS generators must produce valid text."""
        c_header = generate_c_header()
        self.assertIn("#define KRYSTAL_EPI_VALIDATED                0x1006", c_header)
        self.assertIn("#define KRYSTAL_AUTH_VETOED                  0x3004", c_header)
        self.assertIn("#define KRYSTAL_VITAL_MAX_HP 6", c_header)

        rust_mod = generate_rust_module()
        self.assertIn("pub enum EpistemicState {", rust_mod)
        self.assertIn("EpiValidated               = 0x1006,", rust_mod)
        self.assertIn("AuthVetoed                 = 0x3004,", rust_mod)
        self.assertIn("pub const VITAL_MAX_HP: usize = 6;", rust_mod)

        css_classes = generate_css_badge_classes()
        self.assertIn(".badge-epi-validated", css_classes)
        self.assertIn(".badge-auth-vetoed", css_classes)


if __name__ == "__main__":
    unittest.main()
