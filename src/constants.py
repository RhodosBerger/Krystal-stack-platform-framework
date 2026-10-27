#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CONSTANTS & CONSTITUTIONAL INVARIANTS (M1)
==============================================================================
Module: src/constants.py
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from typing import Final

VITAL_MAX_HP: Final[int] = 6


class ContractViolation(Exception):
    """Raised when a constitutional invariant or contract is breached."""
    pass
