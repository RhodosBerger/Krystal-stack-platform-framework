#!/usr/bin/env python3
"""
SPEC REPOSITORY (Database Interface)
Stores and Validates User-Defined Protocols.
"""

import json
import os
from typing import Dict, List, Optional
from operational_standards import CLASS_C_STANDARD, MachineNorms

DB_FILE = "cms/user_specs_db.json"

class SpecRepository:
    def __init__(self):
        self.specs = self._load_db()

    def _load_db(self) -> Dict:
        if not os.path.exists(DB_FILE):
            return {}
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_db(self):
        with open(DB_FILE, 'w') as f:
            json.dump(self.specs, f, indent=4)

    def validate_and_save(self, name: str, criteria: Dict) -> Dict:
        """
        Validates criteria against Operational Standards.
        Returns: { "success": bool, "corrected_criteria": Dict, "message": str }
        """
        corrected = criteria.copy()
        messages = []
        valid = True

        # Validation Logic (Simple Rules Engine)
        # 1. Check Load Limit
        if corrected.get("max_load", 0) > CLASS_C_STANDARD.LOAD_CRITICAL:
            corrected["max_load"] = CLASS_C_STANDARD.LOAD_CONTINUOUS
            messages.append(f"Load clamped to {CLASS_C_STANDARD.LOAD_CONTINUOUS}% (Safety Standard).")
            # We don't fail, we "Correct" (LLM style behavior)

        # 2. Check Vibration
        if corrected.get("vib_limit", 0) > CLASS_C_STANDARD.VIB_CRITICAL:
            corrected["vib_limit"] = CLASS_C_STANDARD.VIB_WARNING
            messages.append("Vibration target reduced to ensure surface quality.")

        # Save
        self.specs[name] = corrected
        self._save_db()

        return {
            "success": True,
            "corrected_criteria": corrected,
            "message": " ".join(messages) or "Specs Accepted as Valid."
        }
    
    def get_spec(self, name: str) -> Optional[Dict]:
        return self.specs.get(name)

# Usage
if __name__ == "__main__":
    repo = SpecRepository()
    res = repo.validate_and_save("HyperAggressive", {"max_load": 200, "vib_limit": 1.5})
    print(res)
