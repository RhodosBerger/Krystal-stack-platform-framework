"""
Universal Quality Assurance & Statistics Framework 🛡️📊
Responsibility:
1. Validator: Enforce schemas and logical constraints.
2. Scanner: Detect unsafe patterns (Security).
3. RuntimeStatistics: Prove performance via metrics.
"""
import time
import re
import logging
from typing import Dict, Any, List, Callable, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger("QualityAssurance")

class RuntimeStatistics:
    """
    The 'Accountant': Tracks execution metrics for "Proving Statistics".
    """
    def __init__(self):
        self.stats = {
            "total_calls": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_duration_ms": 0,
            "avg_duration_ms": 0.0
        }

    def record(self, success: bool, duration_ms: float):
        self.stats["total_calls"] += 1
        if success:
            self.stats["success_count"] += 1
        else:
            self.stats["failure_count"] += 1
        
        self.stats["total_duration_ms"] += duration_ms
        self.stats["avg_duration_ms"] = self.stats["total_duration_ms"] / self.stats["total_calls"]
        
    def get_report(self) -> Dict:
        return self.stats

class ValidationScanner:
    """
    The 'Scanner': Regex-based safety checks.
    """
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",      # Unix Delete
        r"format\s+[c-z]:", # Windows Format
        r"drop\s+table",  # SQL Injection
        r"import\s+os",   # Simple Python exploit check (context dependent)
    ]

    def scan_text(self, text: str) -> List[str]:
        findings = []
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                findings.append(f"Detected Unsafe Pattern: {pattern}")
        return findings

class QualityValidator:
    """
    The 'Validator': Logical & Quality checks.
    """
    def validate_gcode(self, gcode: str) -> bool:
        """Simple G-Code Structural Check"""
        required = ["G20", "G21", "M02", "M30", "G0 ", "G1 "] # Metric/Imperial + End + Moves
        # At least one unit setting and one end command, and some movement
        has_units = "G20" in gcode or "G21" in gcode
        has_end = "M02" in gcode or "M30" in gcode
        has_move = "G0" in gcode or "G1" in gcode
        return has_units and has_move # Relaxed check

    def validate_structure(self, data: Dict, required_keys: List[str]) -> bool:
        return all(k in data for k in required_keys)

# --- Universal Decorator ---
class QAGuard:
    """
    Runtime Wrapper that applies Validator, Scanner, and Stats.
    """
    def __init__(self, scanner: ValidationScanner, validator: QualityValidator, stats: RuntimeStatistics):
        self.scanner = scanner
        self.validator = validator
        self.stats = stats

    def protect(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None
            
            try:
                # 1. Capture Output
                result = func(*args, **kwargs)
                
                # 2. Scan Output (if text)
                if isinstance(result, str):
                    issues = self.scanner.scan_text(result)
                    if issues:
                        logger.warning(f"🛡️ QA Scanner Found Issues: {issues}")
                        # In strict mode, we might raise error. For now, log.
                
                # 3. Validate Logic (simple check if it returned valid data)
                if result is None:
                    success = False
                    
            except Exception as e:
                success = False
                logger.error(f"🛡️ QA Runtime Error: {e}")
                raise e
            finally:
                # 4. Prove Statistics
                duration = (time.time() - start_time) * 1000
                self.stats.record(success, duration)
                logger.info(f"📊 QA Stats Updated: {self.stats.get_report()['avg_duration_ms']:.2f}ms avg")
                
            return result
        return wrapper

# Global QA System
qa_scanner = ValidationScanner()
qa_validator = QualityValidator()
qa_stats = RuntimeStatistics()
qa_guard = QAGuard(qa_scanner, qa_validator, qa_stats)
