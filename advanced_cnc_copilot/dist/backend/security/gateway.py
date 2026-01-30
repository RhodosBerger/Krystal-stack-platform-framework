"""
Security & Isolation Layer 🔒
Responsibility:
1. Control and limit external service connections.
2. Provide network isolation modes.
3. Audit external access attempts.
"""
import os
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from enum import Enum

class IsolationMode(Enum):
    FULL_NETWORK = "FULL_NETWORK"       # All external connections allowed
    LAN_ONLY = "LAN_ONLY"               # Only local network connections
    ISOLATED = "ISOLATED"               # No external connections at all
    ALLOWLIST = "ALLOWLIST"             # Only allowlisted domains

class SecurityGateway:
    """Central control point for all external communications."""
    
    def __init__(self):
        self.mode = IsolationMode.ALLOWLIST
        self.allowlist: Set[str] = {
            "localhost",
            "127.0.0.1",
            "host.docker.internal",
            # Add trusted domains here
        }
        self.blocklist: Set[str] = set()
        self.access_log: List[Dict[str, Any]] = []
        self.max_log_size = 500
        self._blocked_count = 0
        self._allowed_count = 0

    def set_mode(self, mode: IsolationMode):
        """Sets the isolation mode."""
        self.mode = mode
        self._log_event("MODE_CHANGE", f"Security mode changed to {mode.value}")

    def add_to_allowlist(self, domain: str) -> bool:
        """Adds a domain to the allowlist."""
        self.allowlist.add(domain.lower())
        self._log_event("ALLOWLIST_ADD", f"Added {domain} to allowlist")
        return True

    def remove_from_allowlist(self, domain: str) -> bool:
        """Removes a domain from the allowlist."""
        self.allowlist.discard(domain.lower())
        self._log_event("ALLOWLIST_REMOVE", f"Removed {domain} from allowlist")
        return True

    def add_to_blocklist(self, domain: str) -> bool:
        """Adds a domain to the blocklist (overrides allowlist)."""
        self.blocklist.add(domain.lower())
        self._log_event("BLOCKLIST_ADD", f"Added {domain} to blocklist")
        return True

    def is_allowed(self, target: str) -> Dict[str, Any]:
        """
        Checks if a connection to the target is allowed.
        Returns: { allowed: bool, reason: str }
        """
        target_lower = target.lower()
        
        # Always block if in blocklist
        if target_lower in self.blocklist:
            self._blocked_count += 1
            self._log_event("BLOCKED", f"Blocklisted: {target}")
            return {"allowed": False, "reason": "Domain is blocklisted"}

        # Check by mode
        if self.mode == IsolationMode.ISOLATED:
            self._blocked_count += 1
            self._log_event("BLOCKED", f"Isolated mode: {target}")
            return {"allowed": False, "reason": "System in ISOLATED mode - no external connections"}

        if self.mode == IsolationMode.LAN_ONLY:
            if self._is_local_address(target_lower):
                self._allowed_count += 1
                return {"allowed": True, "reason": "Local address allowed"}
            self._blocked_count += 1
            self._log_event("BLOCKED", f"LAN-only mode: {target}")
            return {"allowed": False, "reason": "LAN_ONLY mode - only local connections allowed"}

        if self.mode == IsolationMode.ALLOWLIST:
            for allowed in self.allowlist:
                if allowed in target_lower:
                    self._allowed_count += 1
                    return {"allowed": True, "reason": f"Matches allowlist: {allowed}"}
            self._blocked_count += 1
            self._log_event("BLOCKED", f"Not in allowlist: {target}")
            return {"allowed": False, "reason": "Not in allowlist"}

        # FULL_NETWORK mode - allow all
        self._allowed_count += 1
        return {"allowed": True, "reason": "FULL_NETWORK mode"}

    def _is_local_address(self, target: str) -> bool:
        """Checks if target is a local/LAN address."""
        local_patterns = [
            "localhost", "127.0.0.1", "0.0.0.0",
            "192.168.", "10.", "172.16.", "172.17.", "172.18.",
            "host.docker.internal"
        ]
        return any(p in target for p in local_patterns)

    def _log_event(self, event_type: str, message: str):
        """Logs a security event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "message": message
        }
        self.access_log.insert(0, entry)
        if len(self.access_log) > self.max_log_size:
            self.access_log = self.access_log[:self.max_log_size]

    def get_stats(self) -> Dict[str, Any]:
        """Returns security statistics."""
        return {
            "mode": self.mode.value,
            "allowlist_count": len(self.allowlist),
            "blocklist_count": len(self.blocklist),
            "blocked_attempts": self._blocked_count,
            "allowed_attempts": self._allowed_count,
            "recent_events": self.access_log[:20]
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns current security status."""
        return {
            "mode": self.mode.value,
            "allowlist": list(self.allowlist),
            "blocklist": list(self.blocklist),
            "is_isolated": self.mode == IsolationMode.ISOLATED
        }


# Global Instance
security_gateway = SecurityGateway()

# Default to ALLOWLIST mode with essential services
security_gateway.add_to_allowlist("localhost")
security_gateway.add_to_allowlist("127.0.0.1")
security_gateway.add_to_allowlist("host.docker.internal")
