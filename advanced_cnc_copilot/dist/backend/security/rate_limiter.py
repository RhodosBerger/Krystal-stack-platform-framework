"""
Rate Limiter & Resource Control 🚦
Responsibility:
1. Limit API request rates per user/endpoint.
2. Control resource consumption.
3. Prevent abuse and DoS attacks.
"""
import time
from typing import Dict, Any, Optional
from collections import defaultdict

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.rpm = requests_per_minute
        self.burst = burst_size
        self.tokens: Dict[str, float] = defaultdict(lambda: burst_size)
        self.last_update: Dict[str, float] = defaultdict(time.time)
        self.blocked_until: Dict[str, float] = {}

    def check(self, key: str) -> Dict[str, Any]:
        """
        Checks if request is allowed for the given key.
        Returns: { allowed: bool, remaining: int, reset_in: float }
        """
        now = time.time()
        
        # Check if currently blocked
        if key in self.blocked_until:
            if now < self.blocked_until[key]:
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_in": self.blocked_until[key] - now,
                    "reason": "Temporarily blocked due to rate limit violation"
                }
            else:
                del self.blocked_until[key]

        # Refill tokens based on time elapsed
        elapsed = now - self.last_update[key]
        refill = elapsed * (self.rpm / 60.0)
        self.tokens[key] = min(self.burst, self.tokens[key] + refill)
        self.last_update[key] = now

        # Check token availability
        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return {
                "allowed": True,
                "remaining": int(self.tokens[key]),
                "reset_in": 0
            }
        else:
            # Block for 10 seconds on violation
            self.blocked_until[key] = now + 10
            return {
                "allowed": False,
                "remaining": 0,
                "reset_in": 10,
                "reason": "Rate limit exceeded"
            }

    def get_limit_info(self, key: str) -> Dict[str, Any]:
        """Returns current limit info for a key."""
        return {
            "key": key,
            "remaining_tokens": int(self.tokens.get(key, self.burst)),
            "max_burst": self.burst,
            "requests_per_minute": self.rpm
        }


class ResourceController:
    """Controls and limits resource usage."""
    
    def __init__(self):
        self.limits = {
            "max_upload_size_mb": 50,
            "max_payload_size_mb": 100,
            "max_concurrent_jobs": 5,
            "max_llm_calls_per_hour": 100,
            "max_exports_per_hour": 20
        }
        self.usage: Dict[str, int] = defaultdict(int)
        self.usage_reset: Dict[str, float] = {}

    def check_limit(self, resource: str, amount: int = 1) -> Dict[str, Any]:
        """Checks if resource usage is within limits."""
        now = time.time()
        limit_key = f"max_{resource}"
        
        if limit_key not in self.limits:
            return {"allowed": True, "reason": "No limit defined"}

        # Reset hourly counters
        if resource in self.usage_reset:
            if now - self.usage_reset[resource] > 3600:
                self.usage[resource] = 0
                self.usage_reset[resource] = now
        else:
            self.usage_reset[resource] = now

        max_limit = self.limits[limit_key]
        current = self.usage[resource]
        
        if current + amount <= max_limit:
            self.usage[resource] += amount
            return {
                "allowed": True,
                "current": self.usage[resource],
                "limit": max_limit
            }
        else:
            return {
                "allowed": False,
                "current": current,
                "limit": max_limit,
                "reason": f"Resource limit exceeded for {resource}"
            }

    def set_limit(self, resource: str, limit: int):
        """Sets a resource limit."""
        self.limits[f"max_{resource}"] = limit

    def get_usage(self) -> Dict[str, Any]:
        """Returns current resource usage."""
        return {
            "limits": self.limits,
            "current_usage": dict(self.usage)
        }


# Global Instances
rate_limiter = RateLimiter(requests_per_minute=120, burst_size=20)
resource_controller = ResourceController()
