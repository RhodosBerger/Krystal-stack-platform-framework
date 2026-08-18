"""
GAMESA Profile Manager - Configuration Save/Load System

Provides:
- Named configuration profiles
- Profile inheritance and composition
- Auto-save on threshold triggers
- Profile comparison and diff
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
import json
import time
import copy


class ProfileType(Enum):
    """Profile categories."""
    GAMING = auto()
    PRODUCTIVITY = auto()
    CREATIVE = auto()
    POWER_SAVE = auto()
    BALANCED = auto()
    CUSTOM = auto()


@dataclass
class ProfileSettings:
    """Core profile settings."""
    # Learning parameters
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    discount_factor: float = 0.95

    # Thermal limits
    cpu_temp_target: float = 70.0
    gpu_temp_target: float = 75.0
    power_limit_w: float = 200.0

    # Performance targets
    min_fps: float = 30.0
    target_fps: float = 60.0
    latency_target_ms: float = 16.0

    # Resource allocation
    compute_priority: float = 0.5
    memory_priority: float = 0.3
    bandwidth_priority: float = 0.2

    # Thread boost
    use_p_cores: bool = True
    use_e_cores: bool = True
    boost_duration_ms: int = 5000

    # TPU settings
    prefer_gpu_inference: bool = True
    batch_size: int = 1


@dataclass
class Profile:
    """Complete profile with metadata."""
    name: str
    profile_type: ProfileType
    settings: ProfileSettings
    description: str = ""
    parent: Optional[str] = None
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)
    version: int = 1
    tags: List[str] = field(default_factory=list)


class ProfileManager:
    """
    Manages GAMESA configuration profiles.

    Features:
    - Create, load, save profiles
    - Profile inheritance
    - Auto-switching based on conditions
    - History tracking
    """

    def __init__(self):
        self._profiles: Dict[str, Profile] = {}
        self._active: Optional[str] = None
        self._history: List[tuple] = []  # (timestamp, profile_name, reason)
        self._auto_rules: List[tuple] = []  # (condition_fn, profile_name)
        self._init_default_profiles()

    def _init_default_profiles(self):
        """Initialize built-in profiles."""
        defaults = [
            Profile(
                name="gaming_performance",
                profile_type=ProfileType.GAMING,
                description="Maximum performance for gaming",
                settings=ProfileSettings(
                    learning_rate=0.15,
                    exploration_rate=0.1,
                    cpu_temp_target=80.0,
                    gpu_temp_target=85.0,
                    power_limit_w=250.0,
                    target_fps=144.0,
                    boost_duration_ms=10000
                ),
                tags=["gaming", "performance", "high-power"]
            ),
            Profile(
                name="power_saver",
                profile_type=ProfileType.POWER_SAVE,
                description="Minimize power consumption",
                settings=ProfileSettings(
                    learning_rate=0.05,
                    exploration_rate=0.05,
                    cpu_temp_target=60.0,
                    gpu_temp_target=65.0,
                    power_limit_w=65.0,
                    target_fps=30.0,
                    use_p_cores=False,
                    prefer_gpu_inference=False,
                    boost_duration_ms=1000
                ),
                tags=["battery", "quiet", "efficiency"]
            ),
            Profile(
                name="balanced",
                profile_type=ProfileType.BALANCED,
                description="Balance between performance and efficiency",
                settings=ProfileSettings(),
                tags=["default", "balanced"]
            ),
            Profile(
                name="creative_workload",
                profile_type=ProfileType.CREATIVE,
                description="Optimized for creative applications",
                settings=ProfileSettings(
                    learning_rate=0.08,
                    cpu_temp_target=75.0,
                    gpu_temp_target=80.0,
                    power_limit_w=180.0,
                    latency_target_ms=32.0,
                    memory_priority=0.5,
                    batch_size=4
                ),
                tags=["creative", "rendering", "production"]
            )
        ]

        for profile in defaults:
            self._profiles[profile.name] = profile

        self._active = "balanced"

    def create(self, name: str, profile_type: ProfileType,
               settings: Optional[ProfileSettings] = None,
               parent: Optional[str] = None,
               description: str = "") -> Profile:
        """
        Create new profile.

        Args:
            name: Unique profile name
            profile_type: Profile category
            settings: Custom settings (or inherit from parent)
            parent: Parent profile to inherit from
            description: Profile description

        Returns:
            Created profile
        """
        if parent and parent in self._profiles:
            base_settings = copy.deepcopy(self._profiles[parent].settings)
            if settings:
                # Overlay custom settings
                for key, val in asdict(settings).items():
                    if val != getattr(ProfileSettings(), key):
                        setattr(base_settings, key, val)
            settings = base_settings
        elif settings is None:
            settings = ProfileSettings()

        profile = Profile(
            name=name,
            profile_type=profile_type,
            settings=settings,
            description=description,
            parent=parent
        )
        self._profiles[name] = profile
        return profile

    def get(self, name: str) -> Optional[Profile]:
        """Get profile by name."""
        return self._profiles.get(name)

    def get_active(self) -> Optional[Profile]:
        """Get currently active profile."""
        return self._profiles.get(self._active)

    def activate(self, name: str, reason: str = "manual") -> bool:
        """
        Activate a profile.

        Args:
            name: Profile name to activate
            reason: Reason for activation

        Returns:
            True if activated successfully
        """
        if name not in self._profiles:
            return False

        self._history.append((time.time(), name, reason))
        self._active = name
        self._profiles[name].modified = time.time()
        return True

    def update(self, name: str, **kwargs) -> bool:
        """Update profile settings."""
        profile = self._profiles.get(name)
        if not profile:
            return False

        for key, val in kwargs.items():
            if hasattr(profile.settings, key):
                setattr(profile.settings, key, val)

        profile.modified = time.time()
        profile.version += 1
        return True

    def delete(self, name: str) -> bool:
        """Delete profile (cannot delete active or built-in)."""
        if name == self._active:
            return False
        if name in ["gaming_performance", "power_saver", "balanced", "creative_workload"]:
            return False

        if name in self._profiles:
            del self._profiles[name]
            return True
        return False

    def list_profiles(self, profile_type: Optional[ProfileType] = None) -> List[str]:
        """List profile names, optionally filtered by type."""
        if profile_type:
            return [p.name for p in self._profiles.values() if p.profile_type == profile_type]
        return list(self._profiles.keys())

    def add_auto_rule(self, condition: Callable[[Dict], bool], profile_name: str):
        """
        Add auto-switching rule.

        Args:
            condition: Function that takes telemetry dict and returns True to activate
            profile_name: Profile to activate when condition is True
        """
        if profile_name in self._profiles:
            self._auto_rules.append((condition, profile_name))

    def check_auto_rules(self, telemetry: Dict) -> Optional[str]:
        """
        Check auto rules and switch if needed.

        Returns:
            Name of activated profile, or None
        """
        for condition, profile_name in self._auto_rules:
            try:
                if condition(telemetry):
                    if profile_name != self._active:
                        self.activate(profile_name, "auto_rule")
                        return profile_name
            except Exception:
                pass
        return None

    def diff(self, name1: str, name2: str) -> Dict[str, tuple]:
        """
        Compare two profiles.

        Returns:
            Dict of setting_name -> (value1, value2) for differences
        """
        p1 = self._profiles.get(name1)
        p2 = self._profiles.get(name2)

        if not p1 or not p2:
            return {}

        diffs = {}
        s1 = asdict(p1.settings)
        s2 = asdict(p2.settings)

        for key in s1:
            if s1[key] != s2[key]:
                diffs[key] = (s1[key], s2[key])

        return diffs

    def export_json(self, name: str) -> str:
        """Export profile to JSON."""
        profile = self._profiles.get(name)
        if not profile:
            return "{}"

        data = {
            "name": profile.name,
            "type": profile.profile_type.name,
            "description": profile.description,
            "parent": profile.parent,
            "version": profile.version,
            "tags": profile.tags,
            "settings": asdict(profile.settings)
        }
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> Optional[Profile]:
        """Import profile from JSON."""
        try:
            data = json.loads(json_str)
            settings = ProfileSettings(**data.get("settings", {}))
            profile = Profile(
                name=data["name"],
                profile_type=ProfileType[data.get("type", "CUSTOM")],
                settings=settings,
                description=data.get("description", ""),
                parent=data.get("parent"),
                tags=data.get("tags", [])
            )
            self._profiles[profile.name] = profile
            return profile
        except Exception:
            return None

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get profile switch history."""
        return [
            {"timestamp": ts, "profile": name, "reason": reason}
            for ts, name, reason in self._history[-limit:]
        ]

    def get_stats(self) -> Dict:
        """Get profile manager statistics."""
        return {
            "total_profiles": len(self._profiles),
            "active_profile": self._active,
            "switches": len(self._history),
            "auto_rules": len(self._auto_rules),
            "profile_types": {pt.name: sum(1 for p in self._profiles.values() if p.profile_type == pt)
                             for pt in ProfileType}
        }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate profile manager."""
    print("=== GAMESA Profile Manager Demo ===\n")

    pm = ProfileManager()

    print(f"Default profiles: {pm.list_profiles()}")
    print(f"Active: {pm.get_active().name}\n")

    # Create custom profile
    pm.create(
        "my_gaming",
        ProfileType.GAMING,
        parent="gaming_performance",
        description="My custom gaming profile"
    )
    pm.update("my_gaming", target_fps=120.0, gpu_temp_target=82.0)

    print(f"Created 'my_gaming' inheriting from 'gaming_performance'")
    print(f"Diff from parent: {pm.diff('gaming_performance', 'my_gaming')}\n")

    # Add auto rule
    pm.add_auto_rule(
        lambda t: t.get("gpu_temp", 0) > 85,
        "power_saver"
    )

    # Simulate telemetry
    print("Checking auto rules:")
    result = pm.check_auto_rules({"gpu_temp": 70})
    print(f"  gpu_temp=70: {result}")
    result = pm.check_auto_rules({"gpu_temp": 90})
    print(f"  gpu_temp=90: {result}")

    print(f"\nSwitch history: {pm.get_history()}")
    print(f"\nStats: {pm.get_stats()}")


if __name__ == "__main__":
    demo()
