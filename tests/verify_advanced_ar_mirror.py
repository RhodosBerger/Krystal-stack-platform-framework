"""
Krystal-Stack Automated Verification: Advanced AR Mirror & Antigravity Engine
=============================================================================
Legacy entrypoint maintained for backward compatibility.
The authoritative discoverable unittest suite resides in:
  tests/test_advanced_ar_mirror.py (TestAdvancedARMirror)
"""

import sys
import os
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tests.test_advanced_ar_mirror import TestAdvancedARMirror

def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAdvancedARMirror)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)

if __name__ == "__main__":
    run_tests()
