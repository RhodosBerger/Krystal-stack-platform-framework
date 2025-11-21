"""
GAMESA Plugin System - Extensible Architecture

Provides:
- Plugin discovery and loading
- Lifecycle management
- Hook system for extensions
- Plugin dependencies and conflicts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, Type
from enum import Enum, auto
from abc import ABC, abstractmethod
import time


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = auto()
    LOADED = auto()
    INITIALIZED = auto()
    ACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()


class HookPoint(Enum):
    """Available hook points in the system."""
    PRE_TICK = auto()
    POST_TICK = auto()
    PRE_DECISION = auto()
    POST_DECISION = auto()
    PRE_REWARD = auto()
    POST_REWARD = auto()
    TELEMETRY_RECEIVED = auto()
    STATE_CHANGED = auto()
    THERMAL_EVENT = auto()
    ERROR_OCCURRED = auto()


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    hooks: List[HookPoint] = field(default_factory=list)
    config_schema: Dict = field(default_factory=dict)


class Plugin(ABC):
    """
    Base class for GAMESA plugins.

    Subclass this to create custom plugins.
    """

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.state = PluginState.UNLOADED
        self._context: Optional[Any] = None

    def set_context(self, context: Any):
        """Set framework context."""
        self._context = context

    def initialize(self) -> bool:
        """
        Initialize plugin. Called once after loading.

        Returns:
            True if initialization successful
        """
        self.state = PluginState.INITIALIZED
        return True

    def activate(self) -> bool:
        """
        Activate plugin. Called when plugin should start working.

        Returns:
            True if activation successful
        """
        self.state = PluginState.ACTIVE
        return True

    def deactivate(self) -> bool:
        """
        Deactivate plugin. Called when plugin should stop working.

        Returns:
            True if deactivation successful
        """
        self.state = PluginState.PAUSED
        return True

    def shutdown(self):
        """Shutdown plugin. Called before unloading."""
        self.state = PluginState.UNLOADED

    # Hook methods - override as needed
    def on_pre_tick(self, cycle: int, telemetry: Dict) -> Dict:
        """Called before each tick. Can modify telemetry."""
        return telemetry

    def on_post_tick(self, cycle: int, results: Dict) -> Dict:
        """Called after each tick. Can modify results."""
        return results

    def on_pre_decision(self, state: List[float]) -> List[float]:
        """Called before decision. Can modify state."""
        return state

    def on_post_decision(self, action: List[float]) -> List[float]:
        """Called after decision. Can modify action."""
        return action

    def on_reward(self, reward: float) -> float:
        """Called on reward. Can modify reward."""
        return reward

    def on_state_change(self, old_state: str, new_state: str):
        """Called on framework state change."""
        pass

    def on_thermal_event(self, event: Dict):
        """Called on thermal events."""
        pass

    def on_error(self, error: Dict):
        """Called on errors."""
        pass


class PluginManager:
    """
    Manages plugin lifecycle and hook execution.
    """

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._hooks: Dict[HookPoint, List[str]] = {hp: [] for hp in HookPoint}
        self._load_order: List[str] = []
        self._context: Optional[Any] = None

    def set_context(self, context: Any):
        """Set framework context for plugins."""
        self._context = context
        for plugin in self._plugins.values():
            plugin.set_context(context)

    def register(self, plugin_class: Type[Plugin]):
        """Register a plugin class for later instantiation."""
        meta = plugin_class.get_metadata()
        self._plugin_classes[meta.name] = plugin_class

    def load(self, name: str, config: Dict = None) -> bool:
        """
        Load and initialize a plugin.

        Args:
            name: Plugin name
            config: Plugin configuration

        Returns:
            True if loaded successfully
        """
        if name in self._plugins:
            return False

        plugin_class = self._plugin_classes.get(name)
        if not plugin_class:
            return False

        meta = plugin_class.get_metadata()

        # Check dependencies
        for dep in meta.dependencies:
            if dep not in self._plugins:
                return False

        # Check conflicts
        for conflict in meta.conflicts:
            if conflict in self._plugins:
                return False

        # Instantiate and initialize
        try:
            plugin = plugin_class(config)
            plugin.state = PluginState.LOADED

            if self._context:
                plugin.set_context(self._context)

            if not plugin.initialize():
                return False

            self._plugins[name] = plugin
            self._load_order.append(name)

            # Register hooks
            for hook in meta.hooks:
                self._hooks[hook].append(name)

            return True
        except Exception:
            return False

    def unload(self, name: str) -> bool:
        """Unload a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        # Check if other plugins depend on this
        for other_name, other_plugin in self._plugins.items():
            if other_name == name:
                continue
            meta = other_plugin.get_metadata()
            if name in meta.dependencies:
                return False

        # Shutdown
        plugin.shutdown()

        # Remove from hooks
        meta = plugin.get_metadata()
        for hook in meta.hooks:
            if name in self._hooks[hook]:
                self._hooks[hook].remove(name)

        del self._plugins[name]
        self._load_order.remove(name)
        return True

    def activate(self, name: str) -> bool:
        """Activate a plugin."""
        plugin = self._plugins.get(name)
        if plugin and plugin.state == PluginState.INITIALIZED:
            return plugin.activate()
        return False

    def deactivate(self, name: str) -> bool:
        """Deactivate a plugin."""
        plugin = self._plugins.get(name)
        if plugin and plugin.state == PluginState.ACTIVE:
            return plugin.deactivate()
        return False

    def activate_all(self):
        """Activate all loaded plugins."""
        for name in self._load_order:
            self.activate(name)

    def deactivate_all(self):
        """Deactivate all active plugins."""
        for name in reversed(self._load_order):
            self.deactivate(name)

    def execute_hook(self, hook: HookPoint, *args, **kwargs) -> Any:
        """
        Execute all plugins registered for a hook.

        Args are passed through each plugin in order,
        allowing plugins to transform data.
        """
        result = args[0] if args else None

        for name in self._hooks[hook]:
            plugin = self._plugins.get(name)
            if not plugin or plugin.state != PluginState.ACTIVE:
                continue

            try:
                handler = {
                    HookPoint.PRE_TICK: plugin.on_pre_tick,
                    HookPoint.POST_TICK: plugin.on_post_tick,
                    HookPoint.PRE_DECISION: plugin.on_pre_decision,
                    HookPoint.POST_DECISION: plugin.on_post_decision,
                    HookPoint.PRE_REWARD: plugin.on_reward,
                    HookPoint.POST_REWARD: plugin.on_reward,
                    HookPoint.STATE_CHANGED: plugin.on_state_change,
                    HookPoint.THERMAL_EVENT: plugin.on_thermal_event,
                    HookPoint.ERROR_OCCURRED: plugin.on_error,
                }.get(hook)

                if handler:
                    if len(args) > 1:
                        result = handler(*args, **kwargs)
                    elif args:
                        result = handler(result, **kwargs)
                    else:
                        handler(**kwargs)
            except Exception as e:
                plugin.state = PluginState.ERROR

        return result

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict]:
        """List all loaded plugins."""
        result = []
        for name in self._load_order:
            plugin = self._plugins[name]
            meta = plugin.get_metadata()
            result.append({
                "name": name,
                "version": meta.version,
                "state": plugin.state.name,
                "hooks": [h.name for h in meta.hooks]
            })
        return result

    def list_available(self) -> List[str]:
        """List registered but not loaded plugins."""
        return [n for n in self._plugin_classes if n not in self._plugins]

    def get_stats(self) -> Dict:
        """Get plugin manager statistics."""
        states = {}
        for plugin in self._plugins.values():
            state_name = plugin.state.name
            states[state_name] = states.get(state_name, 0) + 1

        return {
            "registered": len(self._plugin_classes),
            "loaded": len(self._plugins),
            "states": states,
            "hooks_registered": sum(len(h) for h in self._hooks.values())
        }


# ============================================================
# EXAMPLE PLUGINS
# ============================================================

class TelemetryLoggerPlugin(Plugin):
    """Example plugin that logs telemetry."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="telemetry_logger",
            version="1.0.0",
            author="GAMESA",
            description="Logs telemetry data",
            hooks=[HookPoint.PRE_TICK, HookPoint.POST_TICK]
        )

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.log: List[Dict] = []

    def on_pre_tick(self, cycle: int, telemetry: Dict) -> Dict:
        self.log.append({"cycle": cycle, "telemetry": telemetry.copy()})
        return telemetry

    def on_post_tick(self, cycle: int, results: Dict) -> Dict:
        if self.log:
            self.log[-1]["results"] = results.copy()
        return results


class RewardShapingPlugin(Plugin):
    """Example plugin that modifies rewards."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="reward_shaping",
            version="1.0.0",
            description="Shapes rewards based on custom rules",
            hooks=[HookPoint.PRE_REWARD]
        )

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.bonus = config.get("bonus", 0.1) if config else 0.1

    def on_reward(self, reward: float) -> float:
        # Add bonus for high rewards
        if reward > 0.8:
            return min(1.0, reward + self.bonus)
        return reward


class ThermalGuardPlugin(Plugin):
    """Example plugin for thermal monitoring."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="thermal_guard",
            version="1.0.0",
            description="Monitors thermal events",
            hooks=[HookPoint.THERMAL_EVENT, HookPoint.PRE_TICK]
        )

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.threshold = config.get("threshold", 80) if config else 80
        self.alerts: List[Dict] = []

    def on_pre_tick(self, cycle: int, telemetry: Dict) -> Dict:
        temp = telemetry.get("gpu_temp", 0)
        if temp > self.threshold:
            self.alerts.append({"cycle": cycle, "temp": temp})
        return telemetry

    def on_thermal_event(self, event: Dict):
        self.alerts.append(event)


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate plugin system."""
    print("=== GAMESA Plugin System Demo ===\n")

    manager = PluginManager()

    # Register plugins
    manager.register(TelemetryLoggerPlugin)
    manager.register(RewardShapingPlugin)
    manager.register(ThermalGuardPlugin)

    print(f"Available plugins: {manager.list_available()}")

    # Load plugins
    manager.load("telemetry_logger")
    manager.load("reward_shaping", {"bonus": 0.15})
    manager.load("thermal_guard", {"threshold": 75})

    manager.activate_all()

    print(f"\nLoaded plugins: {manager.list_plugins()}")

    # Execute hooks
    print("\n--- Hook Execution ---")
    telemetry = {"cpu_util": 0.7, "gpu_temp": 78}

    result = manager.execute_hook(HookPoint.PRE_TICK, 1, telemetry)
    print(f"Pre-tick result: {result}")

    reward = manager.execute_hook(HookPoint.PRE_REWARD, 0.85)
    print(f"Shaped reward: 0.85 -> {reward}")

    # Check plugin state
    logger = manager.get_plugin("telemetry_logger")
    print(f"\nLogger captured {len(logger.log)} entries")

    guard = manager.get_plugin("thermal_guard")
    print(f"Thermal alerts: {guard.alerts}")

    print(f"\nStats: {manager.get_stats()}")


if __name__ == "__main__":
    demo()
