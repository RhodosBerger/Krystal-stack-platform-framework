"""
GAMESA Driver Pack - Deployment Orchestration Layer

Orchestrates Rust, Python, and C++ agents for unified deployment:
- Validates required toolchains
- Manages runtime dependencies
- Launches Cross-Forex bus components
- Provides deployment health checks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
import subprocess
import shutil
import os
import time
import threading


class ComponentType(Enum):
    """Component types in the runtime."""
    RUST = auto()
    PYTHON = auto()
    CPP = auto()
    BINARY = auto()


class ComponentStatus(Enum):
    """Component deployment status."""
    NOT_CHECKED = auto()
    AVAILABLE = auto()
    MISSING = auto()
    BUILDING = auto()
    READY = auto()
    RUNNING = auto()
    FAILED = auto()


@dataclass
class ToolchainRequirement:
    """A required toolchain."""
    name: str
    command: str
    min_version: Optional[str] = None
    install_hint: str = ""


@dataclass
class Component:
    """A deployable component."""
    name: str
    component_type: ComponentType
    path: str
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    toolchain: Optional[str] = None
    status: ComponentStatus = ComponentStatus.NOT_CHECKED
    process: Optional[subprocess.Popen] = None


class ToolchainValidator:
    """
    Validates required toolchains are available.
    """

    TOOLCHAINS = {
        "cargo": ToolchainRequirement(
            "cargo",
            "cargo --version",
            "1.70",
            "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        ),
        "cmake": ToolchainRequirement(
            "cmake",
            "cmake --version",
            "3.20",
            "Install cmake: apt install cmake / brew install cmake"
        ),
        "python": ToolchainRequirement(
            "python",
            "python3 --version",
            "3.9",
            "Install Python 3.9+"
        ),
        "gcc": ToolchainRequirement(
            "gcc",
            "gcc --version",
            None,
            "Install GCC: apt install build-essential"
        ),
        "clang": ToolchainRequirement(
            "clang",
            "clang --version",
            None,
            "Install Clang: apt install clang"
        )
    }

    def __init__(self):
        self._cache: Dict[str, Tuple[bool, str]] = {}

    def check(self, toolchain: str) -> Tuple[bool, str]:
        """Check if toolchain is available."""
        if toolchain in self._cache:
            return self._cache[toolchain]

        req = self.TOOLCHAINS.get(toolchain)
        if not req:
            return False, f"Unknown toolchain: {toolchain}"

        try:
            result = subprocess.run(
                req.command.split(),
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                self._cache[toolchain] = (True, version)
                return True, version
            else:
                self._cache[toolchain] = (False, req.install_hint)
                return False, req.install_hint

        except FileNotFoundError:
            self._cache[toolchain] = (False, req.install_hint)
            return False, req.install_hint
        except Exception as e:
            self._cache[toolchain] = (False, str(e))
            return False, str(e)

    def check_all(self, toolchains: List[str]) -> Dict[str, Tuple[bool, str]]:
        """Check multiple toolchains."""
        return {t: self.check(t) for t in toolchains}

    def validate_required(self, toolchains: List[str]) -> Tuple[bool, List[str]]:
        """Validate all required toolchains, return missing ones."""
        missing = []
        for t in toolchains:
            available, msg = self.check(t)
            if not available:
                missing.append(f"{t}: {msg}")
        return len(missing) == 0, missing


class DriverPack:
    """
    GAMESA Driver Pack - Deployment Orchestrator.

    Manages the full runtime stack:
    - Rust components (libgamesa_hex)
    - Python agents
    - C++ Vulkan layer
    - Cross-Forex bus coordination
    """

    def __init__(self, base_path: str = "."):
        self.base_path = os.path.abspath(base_path)
        self.validator = ToolchainValidator()

        # Components registry
        self._components: Dict[str, Component] = {}
        self._init_default_components()

        # Runtime state
        self._running = False
        self._processes: Dict[str, subprocess.Popen] = {}

        # Callbacks
        self._on_status_change: List[Callable[[str, ComponentStatus], None]] = []

    def _init_default_components(self):
        """Initialize default component definitions."""
        components = [
            Component(
                name="libgamesa_hex",
                component_type=ComponentType.RUST,
                path="src/rust/gamesa_hex",
                entry_point="cargo build --release",
                toolchain="cargo"
            ),
            Component(
                name="crystal_socketd",
                component_type=ComponentType.PYTHON,
                path="src/python",
                entry_point="crystal_socketd.py",
                dependencies=["crystal_protocol"]
            ),
            Component(
                name="guardian_hex",
                component_type=ComponentType.PYTHON,
                path="src/python",
                entry_point="guardian_hex.py",
                dependencies=["crystal_protocol"]
            ),
            Component(
                name="gamesad",
                component_type=ComponentType.PYTHON,
                path="src/python",
                entry_point="gamesad.py",
                dependencies=["crystal_socketd", "guardian_hex", "crystal_agents"]
            ),
            Component(
                name="vulkan_layer",
                component_type=ComponentType.CPP,
                path="src/cpp/vulkan_layer",
                entry_point="cmake --build build",
                toolchain="cmake"
            ),
            Component(
                name="cpu_agent_tool",
                component_type=ComponentType.PYTHON,
                path="tools",
                entry_point="cpu_agent_tool.py"
            )
        ]

        for comp in components:
            self._components[comp.name] = comp

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate_environment(self) -> Dict:
        """Validate deployment environment."""
        results = {
            "toolchains": {},
            "components": {},
            "ready": True,
            "missing": []
        }

        # Check toolchains
        required_toolchains = set()
        for comp in self._components.values():
            if comp.toolchain:
                required_toolchains.add(comp.toolchain)

        # Always check Python
        required_toolchains.add("python")

        for tc in required_toolchains:
            available, msg = self.validator.check(tc)
            results["toolchains"][tc] = {"available": available, "info": msg}
            if not available:
                results["ready"] = False
                results["missing"].append(f"Toolchain: {tc}")

        # Check component paths
        for name, comp in self._components.items():
            path = os.path.join(self.base_path, comp.path)
            exists = os.path.exists(path)
            results["components"][name] = {
                "type": comp.component_type.name,
                "path_exists": exists,
                "status": comp.status.name
            }

        return results

    def check_component(self, name: str) -> ComponentStatus:
        """Check status of a specific component."""
        comp = self._components.get(name)
        if not comp:
            return ComponentStatus.MISSING

        path = os.path.join(self.base_path, comp.path)

        if not os.path.exists(path):
            comp.status = ComponentStatus.MISSING
        elif comp.toolchain:
            available, _ = self.validator.check(comp.toolchain)
            comp.status = ComponentStatus.AVAILABLE if available else ComponentStatus.MISSING
        else:
            comp.status = ComponentStatus.AVAILABLE

        return comp.status

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    def build_component(self, name: str) -> Tuple[bool, str]:
        """Build a component."""
        comp = self._components.get(name)
        if not comp:
            return False, f"Unknown component: {name}"

        if comp.component_type == ComponentType.PYTHON:
            # Python doesn't need building
            comp.status = ComponentStatus.READY
            return True, "Python component ready"

        # Check toolchain
        if comp.toolchain:
            available, msg = self.validator.check(comp.toolchain)
            if not available:
                return False, f"Missing toolchain: {msg}"

        comp.status = ComponentStatus.BUILDING
        self._notify_status(name, ComponentStatus.BUILDING)

        try:
            work_dir = os.path.join(self.base_path, comp.path)

            if comp.component_type == ComponentType.RUST:
                result = subprocess.run(
                    ["cargo", "build", "--release"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            elif comp.component_type == ComponentType.CPP:
                # Create build directory
                build_dir = os.path.join(work_dir, "build")
                os.makedirs(build_dir, exist_ok=True)

                # CMake configure
                subprocess.run(
                    ["cmake", ".."],
                    cwd=build_dir,
                    capture_output=True,
                    timeout=60
                )

                # CMake build
                result = subprocess.run(
                    ["cmake", "--build", "."],
                    cwd=build_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            else:
                return False, f"Unsupported component type: {comp.component_type}"

            if result.returncode == 0:
                comp.status = ComponentStatus.READY
                self._notify_status(name, ComponentStatus.READY)
                return True, "Build successful"
            else:
                comp.status = ComponentStatus.FAILED
                self._notify_status(name, ComponentStatus.FAILED)
                return False, result.stderr

        except subprocess.TimeoutExpired:
            comp.status = ComponentStatus.FAILED
            return False, "Build timeout"
        except Exception as e:
            comp.status = ComponentStatus.FAILED
            return False, str(e)

    def build_all(self) -> Dict[str, Tuple[bool, str]]:
        """Build all components."""
        results = {}
        for name in self._components:
            results[name] = self.build_component(name)
        return results

    # --------------------------------------------------------
    # LAUNCH
    # --------------------------------------------------------

    def launch_component(self, name: str, **kwargs) -> Tuple[bool, str]:
        """Launch a component."""
        comp = self._components.get(name)
        if not comp:
            return False, f"Unknown component: {name}"

        if comp.status not in [ComponentStatus.READY, ComponentStatus.AVAILABLE]:
            # Try to prepare it
            self.check_component(name)
            if comp.component_type != ComponentType.PYTHON:
                success, msg = self.build_component(name)
                if not success:
                    return False, msg

        try:
            work_dir = os.path.join(self.base_path, comp.path)

            if comp.component_type == ComponentType.PYTHON:
                entry = os.path.join(work_dir, comp.entry_point)
                process = subprocess.Popen(
                    ["python3", entry],
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **kwargs
                )
            elif comp.component_type == ComponentType.RUST:
                # Run built binary
                binary = os.path.join(work_dir, "target/release", name)
                process = subprocess.Popen(
                    [binary],
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **kwargs
                )
            elif comp.component_type == ComponentType.CPP:
                binary = os.path.join(work_dir, "build", name)
                process = subprocess.Popen(
                    [binary],
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **kwargs
                )
            else:
                return False, "Unsupported component type"

            comp.process = process
            self._processes[name] = process
            comp.status = ComponentStatus.RUNNING
            self._notify_status(name, ComponentStatus.RUNNING)

            return True, f"Launched with PID {process.pid}"

        except Exception as e:
            comp.status = ComponentStatus.FAILED
            return False, str(e)

    def stop_component(self, name: str) -> bool:
        """Stop a running component."""
        process = self._processes.get(name)
        if not process:
            return False

        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        comp = self._components.get(name)
        if comp:
            comp.status = ComponentStatus.READY
            comp.process = None

        del self._processes[name]
        return True

    def stop_all(self):
        """Stop all running components."""
        for name in list(self._processes.keys()):
            self.stop_component(name)

    # --------------------------------------------------------
    # FULL DEPLOYMENT
    # --------------------------------------------------------

    def deploy(self) -> Tuple[bool, Dict]:
        """
        Full deployment of Cross-Forex runtime.

        1. Validate environment
        2. Build required components
        3. Launch runtime stack
        """
        results = {
            "validation": None,
            "build": {},
            "launch": {},
            "success": False
        }

        # 1. Validate
        validation = self.validate_environment()
        results["validation"] = validation

        if not validation["ready"]:
            return False, results

        # 2. Build non-Python components
        for name, comp in self._components.items():
            if comp.component_type != ComponentType.PYTHON:
                success, msg = self.build_component(name)
                results["build"][name] = {"success": success, "message": msg}

        # 3. Mark Python components ready
        for name, comp in self._components.items():
            if comp.component_type == ComponentType.PYTHON:
                comp.status = ComponentStatus.READY
                results["build"][name] = {"success": True, "message": "Python ready"}

        results["success"] = True
        return True, results

    # --------------------------------------------------------
    # STATUS & MONITORING
    # --------------------------------------------------------

    def _notify_status(self, name: str, status: ComponentStatus):
        """Notify status change callbacks."""
        for callback in self._on_status_change:
            try:
                callback(name, status)
            except Exception:
                pass

    def on_status_change(self, callback: Callable[[str, ComponentStatus], None]):
        """Register status change callback."""
        self._on_status_change.append(callback)

    def get_status(self) -> Dict:
        """Get deployment status."""
        return {
            "components": {
                name: {
                    "type": comp.component_type.name,
                    "status": comp.status.name,
                    "running": comp.process is not None and comp.process.poll() is None
                }
                for name, comp in self._components.items()
            },
            "running_processes": len([p for p in self._processes.values() if p.poll() is None]),
            "total_components": len(self._components)
        }

    def health_check(self) -> Dict:
        """Perform health check on running components."""
        results = {}

        for name, process in self._processes.items():
            poll = process.poll()
            if poll is None:
                results[name] = {"healthy": True, "status": "running"}
            else:
                results[name] = {"healthy": False, "status": "exited", "code": poll}

        return results


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate driver pack."""
    print("=== GAMESA Driver Pack Demo ===\n")

    pack = DriverPack()

    # Validate environment
    print("--- Environment Validation ---")
    validation = pack.validate_environment()

    print("Toolchains:")
    for tc, info in validation["toolchains"].items():
        status = "OK" if info["available"] else "MISSING"
        print(f"  {tc}: {status}")

    print("\nComponents:")
    for name, info in validation["components"].items():
        print(f"  {name}: {info['type']} - {info['status']}")

    if validation["ready"]:
        print("\nEnvironment ready for deployment!")
    else:
        print(f"\nMissing requirements: {validation['missing']}")

    # Show deployment status
    print("\n--- Deployment Status ---")
    status = pack.get_status()
    print(f"Total components: {status['total_components']}")
    print(f"Running: {status['running_processes']}")


if __name__ == "__main__":
    demo()
