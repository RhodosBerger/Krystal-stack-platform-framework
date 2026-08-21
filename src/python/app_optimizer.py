"""
GAMESA Real Application Optimizer - Universal Process Optimization

Optimizations for all real applications and programs:
1. ProcessProfiler - Profile any running process
2. ApplicationClassifier - Classify app type (game, browser, IDE, etc.)
3. DynamicOptimizer - Real-time per-process optimization
4. ResourceGovernor - Fair resource distribution across apps
5. LatencyOptimizer - Reduce input/output latency
6. MemoryOptimizer - Smart memory management per app
"""

import os
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path


# =============================================================================
# APPLICATION TYPES
# =============================================================================

class AppCategory(Enum):
    """Application categories."""
    GAME = "game"
    BROWSER = "browser"
    IDE = "ide"
    MEDIA_PLAYER = "media_player"
    CREATIVE = "creative"       # Photoshop, Blender, etc.
    OFFICE = "office"
    TERMINAL = "terminal"
    COMMUNICATION = "communication"  # Discord, Slack, etc.
    SYSTEM = "system"
    BACKGROUND = "background"
    UNKNOWN = "unknown"


class OptimizationProfile(Enum):
    """Optimization profiles."""
    LATENCY = "latency"         # Minimize latency (games, realtime)
    THROUGHPUT = "throughput"   # Maximize throughput (rendering, compile)
    BALANCED = "balanced"       # Balance all metrics
    EFFICIENCY = "efficiency"   # Minimize power (background)
    INTERACTIVE = "interactive" # Fast response (browsers, IDE)


# =============================================================================
# 1. PROCESS PROFILER
# =============================================================================

@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    name: str
    cmdline: str = ""
    exe_path: str = ""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    threads: int = 1
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    nice: int = 0
    state: str = "running"
    start_time: float = 0.0
    category: AppCategory = AppCategory.UNKNOWN


class ProcessProfiler:
    """
    Profile any running process on the system.
    """

    def __init__(self):
        self.process_cache: Dict[int, ProcessInfo] = {}
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self._page_size = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096

    def get_process(self, pid: int) -> Optional[ProcessInfo]:
        """Get process information."""
        try:
            proc_path = Path(f"/proc/{pid}")
            if not proc_path.exists():
                return None

            info = ProcessInfo(pid=pid, name="")

            # Name and cmdline
            try:
                info.name = (proc_path / "comm").read_text().strip()
            except (IOError, PermissionError):
                info.name = f"pid_{pid}"

            try:
                info.cmdline = (proc_path / "cmdline").read_text().replace('\x00', ' ').strip()
            except (IOError, PermissionError):
                pass

            try:
                info.exe_path = os.readlink(proc_path / "exe")
            except (IOError, PermissionError, OSError):
                pass

            # CPU and memory from stat
            try:
                stat = (proc_path / "stat").read_text().split()
                info.threads = int(stat[19])
                info.nice = int(stat[18])
                info.state = stat[2]
            except (IOError, PermissionError, IndexError):
                pass

            # Memory from statm
            try:
                statm = (proc_path / "statm").read_text().split()
                resident_pages = int(statm[1])
                info.memory_mb = (resident_pages * self._page_size) / (1024 * 1024)
            except (IOError, PermissionError, IndexError):
                pass

            # Calculate memory percent
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            total_kb = int(line.split()[1])
                            info.memory_percent = (info.memory_mb * 1024) / total_kb * 100
                            break
            except (IOError, PermissionError):
                pass

            # IO stats
            try:
                io_content = (proc_path / "io").read_text()
                for line in io_content.split('\n'):
                    if line.startswith("read_bytes:"):
                        info.io_read_mb = int(line.split()[1]) / (1024 * 1024)
                    elif line.startswith("write_bytes:"):
                        info.io_write_mb = int(line.split()[1]) / (1024 * 1024)
            except (IOError, PermissionError):
                pass

            # Cache and record history
            self.process_cache[pid] = info
            self.history[pid].append({
                "timestamp": time.time(),
                "cpu": info.cpu_percent,
                "memory": info.memory_mb,
            })

            return info

        except Exception:
            return None

    def get_all_processes(self) -> List[ProcessInfo]:
        """Get all running processes."""
        processes = []
        try:
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    pid = int(entry.name)
                    info = self.get_process(pid)
                    if info:
                        processes.append(info)
        except (IOError, PermissionError):
            pass
        return processes

    def get_top_processes(self, n: int = 10, sort_by: str = "memory") -> List[ProcessInfo]:
        """Get top N processes by resource usage."""
        processes = self.get_all_processes()

        if sort_by == "memory":
            processes.sort(key=lambda p: p.memory_mb, reverse=True)
        elif sort_by == "threads":
            processes.sort(key=lambda p: p.threads, reverse=True)

        return processes[:n]


# =============================================================================
# 2. APPLICATION CLASSIFIER
# =============================================================================

class ApplicationClassifier:
    """
    Classify applications by type for optimal tuning.
    """

    def __init__(self):
        # Known application patterns
        self.patterns = {
            AppCategory.GAME: [
                r"steam", r"wine", r"proton", r"lutris",
                r"minecraft", r"csgo", r"dota", r"valorant",
                r"game", r"unity", r"unreal", r"godot",
            ],
            AppCategory.BROWSER: [
                r"firefox", r"chrome", r"chromium", r"brave",
                r"opera", r"vivaldi", r"edge", r"safari",
                r"webkit", r"gecko",
            ],
            AppCategory.IDE: [
                r"code", r"vscode", r"idea", r"pycharm",
                r"eclipse", r"netbeans", r"atom", r"sublime",
                r"vim", r"nvim", r"emacs", r"gedit",
                r"android-studio", r"xcode",
            ],
            AppCategory.MEDIA_PLAYER: [
                r"vlc", r"mpv", r"mplayer", r"totem",
                r"spotify", r"rhythmbox", r"audacious",
                r"kodi", r"plex",
            ],
            AppCategory.CREATIVE: [
                r"gimp", r"krita", r"inkscape", r"blender",
                r"kdenlive", r"davinci", r"premiere", r"photoshop",
                r"lightroom", r"audacity", r"ardour",
            ],
            AppCategory.OFFICE: [
                r"libreoffice", r"openoffice", r"word", r"excel",
                r"calc", r"writer", r"impress", r"powerpoint",
                r"thunderbird", r"evolution", r"outlook",
            ],
            AppCategory.TERMINAL: [
                r"bash", r"zsh", r"fish", r"sh$",
                r"gnome-terminal", r"konsole", r"alacritty",
                r"kitty", r"terminator", r"xterm",
            ],
            AppCategory.COMMUNICATION: [
                r"discord", r"slack", r"teams", r"zoom",
                r"skype", r"telegram", r"signal", r"element",
            ],
            AppCategory.SYSTEM: [
                r"systemd", r"dbus", r"pulseaudio", r"pipewire",
                r"xorg", r"wayland", r"gnome-shell", r"plasmashell",
                r"kwin", r"mutter",
            ],
        }

        # Compile patterns
        self.compiled = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.patterns.items()
        }

    def classify(self, process: ProcessInfo) -> AppCategory:
        """Classify a process."""
        # Check name, cmdline, and exe path
        search_strings = [
            process.name,
            process.cmdline,
            process.exe_path,
        ]

        for category, patterns in self.compiled.items():
            for pattern in patterns:
                for search_str in search_strings:
                    if search_str and pattern.search(search_str):
                        return category

        # Heuristics for unknown
        if process.threads > 20:
            return AppCategory.CREATIVE  # Likely heavy app
        elif process.memory_mb > 500:
            return AppCategory.BROWSER  # Likely browser or similar

        return AppCategory.UNKNOWN

    def get_optimization_profile(self, category: AppCategory) -> OptimizationProfile:
        """Get optimization profile for category."""
        mapping = {
            AppCategory.GAME: OptimizationProfile.LATENCY,
            AppCategory.BROWSER: OptimizationProfile.INTERACTIVE,
            AppCategory.IDE: OptimizationProfile.INTERACTIVE,
            AppCategory.MEDIA_PLAYER: OptimizationProfile.THROUGHPUT,
            AppCategory.CREATIVE: OptimizationProfile.THROUGHPUT,
            AppCategory.OFFICE: OptimizationProfile.BALANCED,
            AppCategory.TERMINAL: OptimizationProfile.LATENCY,
            AppCategory.COMMUNICATION: OptimizationProfile.INTERACTIVE,
            AppCategory.SYSTEM: OptimizationProfile.EFFICIENCY,
            AppCategory.BACKGROUND: OptimizationProfile.EFFICIENCY,
            AppCategory.UNKNOWN: OptimizationProfile.BALANCED,
        }
        return mapping.get(category, OptimizationProfile.BALANCED)


# =============================================================================
# 3. DYNAMIC OPTIMIZER
# =============================================================================

@dataclass
class OptimizationAction:
    """Action to optimize a process."""
    pid: int
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class DynamicOptimizer:
    """
    Real-time per-process optimization.
    """

    def __init__(self):
        self.profiler = ProcessProfiler()
        self.classifier = ApplicationClassifier()
        self.active_optimizations: Dict[int, List[str]] = {}

        # Profile settings
        self.profile_settings = {
            OptimizationProfile.LATENCY: {
                "nice": -5,
                "ionice_class": 1,  # realtime
                "cpu_affinity": "performance",
                "scheduler": "SCHED_RR",
            },
            OptimizationProfile.THROUGHPUT: {
                "nice": -2,
                "ionice_class": 2,  # best-effort
                "cpu_affinity": "all",
                "scheduler": "SCHED_BATCH",
            },
            OptimizationProfile.INTERACTIVE: {
                "nice": -3,
                "ionice_class": 2,
                "cpu_affinity": "balanced",
                "scheduler": "SCHED_OTHER",
            },
            OptimizationProfile.BALANCED: {
                "nice": 0,
                "ionice_class": 2,
                "cpu_affinity": "balanced",
                "scheduler": "SCHED_OTHER",
            },
            OptimizationProfile.EFFICIENCY: {
                "nice": 10,
                "ionice_class": 3,  # idle
                "cpu_affinity": "efficiency",
                "scheduler": "SCHED_IDLE",
            },
        }

    def analyze(self, pid: int) -> List[OptimizationAction]:
        """Analyze process and suggest optimizations."""
        actions = []

        info = self.profiler.get_process(pid)
        if not info:
            return actions

        category = self.classifier.classify(info)
        profile = self.classifier.get_optimization_profile(category)
        settings = self.profile_settings[profile]

        # Nice adjustment
        if info.nice != settings["nice"]:
            actions.append(OptimizationAction(
                pid=pid,
                action="renice",
                params={"nice": settings["nice"]},
                reason=f"Adjust priority for {category.value} app",
            ))

        # CPU affinity
        if settings["cpu_affinity"] != "all":
            actions.append(OptimizationAction(
                pid=pid,
                action="set_affinity",
                params={"mode": settings["cpu_affinity"]},
                reason=f"Optimize CPU affinity for {profile.value}",
            ))

        # Memory optimization for heavy apps
        if info.memory_mb > 1000:
            actions.append(OptimizationAction(
                pid=pid,
                action="memory_advice",
                params={"advice": "MADV_HUGEPAGE"},
                reason="Enable huge pages for large memory app",
            ))

        return actions

    def optimize_all(self) -> Dict[str, List[OptimizationAction]]:
        """Analyze and optimize all processes."""
        results = {}

        top_processes = self.profiler.get_top_processes(20)

        for proc in top_processes:
            actions = self.analyze(proc.pid)
            if actions:
                results[proc.name] = actions

        return results

    def generate_optimization_script(self, actions: List[OptimizationAction]) -> str:
        """Generate shell script for optimizations."""
        lines = ["#!/bin/bash", "# GAMESA Auto-generated optimization script", ""]

        for action in actions:
            if action.action == "renice":
                nice = action.params.get("nice", 0)
                lines.append(f"renice {nice} -p {action.pid}  # {action.reason}")

            elif action.action == "set_affinity":
                mode = action.params.get("mode", "all")
                if mode == "performance":
                    lines.append(f"taskset -cp 0-3 {action.pid}  # {action.reason}")
                elif mode == "efficiency":
                    lines.append(f"taskset -cp 4-7 {action.pid}  # {action.reason}")

            elif action.action == "ionice":
                cls = action.params.get("class", 2)
                lines.append(f"ionice -c {cls} -p {action.pid}  # {action.reason}")

        return "\n".join(lines)


# =============================================================================
# 4. RESOURCE GOVERNOR
# =============================================================================

@dataclass
class ResourceQuota:
    """Resource quota for an application."""
    cpu_percent: float = 100.0
    memory_mb: float = 0.0  # 0 = unlimited
    io_weight: int = 100    # 1-1000
    network_mbps: float = 0.0  # 0 = unlimited


class ResourceGovernor:
    """
    Fair resource distribution across applications.
    """

    def __init__(self):
        self.quotas: Dict[str, ResourceQuota] = {}
        self.classifier = ApplicationClassifier()

        # Default quotas by category
        self.default_quotas = {
            AppCategory.GAME: ResourceQuota(cpu_percent=80, io_weight=500),
            AppCategory.BROWSER: ResourceQuota(cpu_percent=50, memory_mb=4096, io_weight=300),
            AppCategory.IDE: ResourceQuota(cpu_percent=60, io_weight=400),
            AppCategory.CREATIVE: ResourceQuota(cpu_percent=90, io_weight=600),
            AppCategory.MEDIA_PLAYER: ResourceQuota(cpu_percent=30, io_weight=200),
            AppCategory.BACKGROUND: ResourceQuota(cpu_percent=10, io_weight=50),
            AppCategory.SYSTEM: ResourceQuota(cpu_percent=20, io_weight=100),
        }

    def set_quota(self, app_name: str, quota: ResourceQuota):
        """Set resource quota for application."""
        self.quotas[app_name] = quota

    def get_quota(self, process: ProcessInfo) -> ResourceQuota:
        """Get resource quota for process."""
        # Check explicit quota
        if process.name in self.quotas:
            return self.quotas[process.name]

        # Use category default
        category = self.classifier.classify(process)
        return self.default_quotas.get(category, ResourceQuota())

    def distribute_resources(self, processes: List[ProcessInfo]) -> Dict[int, ResourceQuota]:
        """Distribute resources fairly across processes."""
        distribution = {}
        total_weight = 0

        # Calculate weights
        weights = {}
        for proc in processes:
            category = self.classifier.classify(proc)
            profile = self.classifier.get_optimization_profile(category)

            weight = {
                OptimizationProfile.LATENCY: 10,
                OptimizationProfile.THROUGHPUT: 8,
                OptimizationProfile.INTERACTIVE: 6,
                OptimizationProfile.BALANCED: 4,
                OptimizationProfile.EFFICIENCY: 2,
            }.get(profile, 4)

            weights[proc.pid] = weight
            total_weight += weight

        # Distribute proportionally
        for proc in processes:
            weight = weights[proc.pid]
            ratio = weight / total_weight if total_weight > 0 else 1.0 / len(processes)

            base_quota = self.get_quota(proc)
            distribution[proc.pid] = ResourceQuota(
                cpu_percent=min(100, base_quota.cpu_percent * ratio * 2),
                memory_mb=base_quota.memory_mb,
                io_weight=int(base_quota.io_weight * ratio * 2),
            )

        return distribution

    def generate_cgroup_config(self, pid: int, quota: ResourceQuota) -> Dict[str, str]:
        """Generate cgroup configuration."""
        config = {}

        if quota.cpu_percent < 100:
            # CPU quota in microseconds per period
            period = 100000
            quota_us = int(period * quota.cpu_percent / 100)
            config["cpu.max"] = f"{quota_us} {period}"

        if quota.memory_mb > 0:
            config["memory.max"] = f"{int(quota.memory_mb * 1024 * 1024)}"

        if quota.io_weight != 100:
            config["io.weight"] = f"default {quota.io_weight}"

        return config


# =============================================================================
# 5. LATENCY OPTIMIZER
# =============================================================================

class LatencyOptimizer:
    """
    Minimize input/output latency for responsive applications.
    """

    def __init__(self):
        self.latency_targets = {
            AppCategory.GAME: 1.0,          # 1ms target
            AppCategory.TERMINAL: 5.0,       # 5ms
            AppCategory.IDE: 10.0,           # 10ms
            AppCategory.BROWSER: 16.0,       # 16ms (60fps)
            AppCategory.COMMUNICATION: 20.0, # 20ms
        }

    def get_optimizations(self, category: AppCategory) -> Dict[str, Any]:
        """Get latency optimizations for category."""
        target = self.latency_targets.get(category, 50.0)

        optimizations = {
            "target_latency_ms": target,
            "recommendations": [],
        }

        if target <= 5:
            optimizations["recommendations"].extend([
                "Use SCHED_FIFO or SCHED_RR scheduler",
                "Pin to isolated CPU cores",
                "Disable CPU frequency scaling",
                "Use polling mode for input devices",
                "Enable threaded IRQs",
            ])
            optimizations["kernel_params"] = {
                "isolcpus": "2,3",
                "nohz_full": "2,3",
                "rcu_nocbs": "2,3",
            }

        elif target <= 16:
            optimizations["recommendations"].extend([
                "Use performance CPU governor",
                "Enable compositor bypass",
                "Reduce input polling interval",
            ])

        else:
            optimizations["recommendations"].extend([
                "Use schedutil governor",
                "Standard timer resolution",
            ])

        return optimizations

    def measure_input_latency(self) -> Dict[str, float]:
        """Measure current input latency (estimated)."""
        latencies = {}

        # Check timer resolution
        try:
            with open("/proc/timer_list") as f:
                content = f.read()
                if "hrtimer" in content:
                    latencies["timer_resolution_ns"] = 1000  # 1us with hrtimer
                else:
                    latencies["timer_resolution_ns"] = 1000000  # 1ms default
        except (IOError, PermissionError):
            latencies["timer_resolution_ns"] = 1000000

        # Check scheduler latency
        try:
            sched_latency = Path("/proc/sys/kernel/sched_latency_ns")
            if sched_latency.exists():
                latencies["sched_latency_ns"] = int(sched_latency.read_text().strip())
        except (IOError, PermissionError, ValueError):
            pass

        return latencies


# =============================================================================
# 6. MEMORY OPTIMIZER
# =============================================================================

class MemoryOptimizer:
    """
    Smart memory management per application.
    """

    def __init__(self):
        self.profiler = ProcessProfiler()
        self.memory_policies = {
            AppCategory.GAME: "aggressive_cache",
            AppCategory.BROWSER: "limit_and_swap",
            AppCategory.CREATIVE: "huge_pages",
            AppCategory.BACKGROUND: "compress",
        }

    def analyze_memory(self, pid: int) -> Dict[str, Any]:
        """Analyze memory usage of process."""
        analysis = {
            "pid": pid,
            "recommendations": [],
        }

        try:
            # Read smaps for detailed memory info
            smaps_path = Path(f"/proc/{pid}/smaps_rollup")
            if smaps_path.exists():
                content = smaps_path.read_text()

                for line in content.split('\n'):
                    if line.startswith("Rss:"):
                        analysis["rss_kb"] = int(line.split()[1])
                    elif line.startswith("Shared"):
                        analysis["shared_kb"] = int(line.split()[1])
                    elif line.startswith("Private"):
                        analysis["private_kb"] = int(line.split()[1])
                    elif line.startswith("Swap:"):
                        analysis["swap_kb"] = int(line.split()[1])

            # Recommendations based on analysis
            if analysis.get("swap_kb", 0) > 100000:  # >100MB in swap
                analysis["recommendations"].append("Process is swapping heavily - increase RAM or reduce memory pressure")

            if analysis.get("rss_kb", 0) > 2000000:  # >2GB
                analysis["recommendations"].append("Consider enabling huge pages for better TLB efficiency")

        except (IOError, PermissionError):
            pass

        return analysis

    def get_memory_policy(self, category: AppCategory) -> Dict[str, Any]:
        """Get memory policy for category."""
        policy_name = self.memory_policies.get(category, "default")

        policies = {
            "aggressive_cache": {
                "vm.swappiness": 10,
                "vm.vfs_cache_pressure": 50,
                "transparent_hugepages": "always",
                "oom_score_adj": -500,
            },
            "limit_and_swap": {
                "vm.swappiness": 60,
                "memory_limit_mb": 4096,
                "oom_score_adj": 0,
            },
            "huge_pages": {
                "transparent_hugepages": "always",
                "vm.nr_hugepages": 512,
                "oom_score_adj": -200,
            },
            "compress": {
                "vm.swappiness": 100,
                "zswap_enabled": True,
                "oom_score_adj": 500,
            },
            "default": {
                "vm.swappiness": 60,
                "oom_score_adj": 0,
            },
        }

        return policies.get(policy_name, policies["default"])


# =============================================================================
# UNIFIED APPLICATION OPTIMIZER
# =============================================================================

class ApplicationOptimizer:
    """
    Complete application optimization system.
    """

    def __init__(self):
        self.profiler = ProcessProfiler()
        self.classifier = ApplicationClassifier()
        self.dynamic = DynamicOptimizer()
        self.governor = ResourceGovernor()
        self.latency = LatencyOptimizer()
        self.memory = MemoryOptimizer()

    def optimize_system(self) -> Dict[str, Any]:
        """Optimize entire system."""
        results = {
            "timestamp": time.time(),
            "processes_analyzed": 0,
            "optimizations": [],
        }

        # Get all processes
        processes = self.profiler.get_all_processes()
        results["processes_analyzed"] = len(processes)

        # Classify and analyze each
        for proc in processes[:50]:  # Limit to top 50
            category = self.classifier.classify(proc)
            proc.category = category

            if category == AppCategory.UNKNOWN:
                continue

            # Get optimizations
            profile = self.classifier.get_optimization_profile(category)
            actions = self.dynamic.analyze(proc.pid)

            if actions:
                results["optimizations"].append({
                    "process": proc.name,
                    "pid": proc.pid,
                    "category": category.value,
                    "profile": profile.value,
                    "actions": [a.action for a in actions],
                })

        # Resource distribution
        results["resource_distribution"] = self.governor.distribute_resources(processes[:20])

        return results

    def optimize_process(self, pid: int) -> Dict[str, Any]:
        """Optimize specific process."""
        info = self.profiler.get_process(pid)
        if not info:
            return {"error": f"Process {pid} not found"}

        category = self.classifier.classify(info)
        profile = self.classifier.get_optimization_profile(category)

        return {
            "process": info.name,
            "pid": pid,
            "category": category.value,
            "profile": profile.value,
            "actions": self.dynamic.analyze(pid),
            "latency": self.latency.get_optimizations(category),
            "memory": self.memory.analyze_memory(pid),
            "quota": self.governor.get_quota(info),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system optimization status."""
        processes = self.profiler.get_top_processes(10)

        by_category = defaultdict(list)
        for proc in processes:
            category = self.classifier.classify(proc)
            by_category[category.value].append({
                "name": proc.name,
                "pid": proc.pid,
                "memory_mb": proc.memory_mb,
                "threads": proc.threads,
            })

        return {
            "top_processes": len(processes),
            "by_category": dict(by_category),
            "latency_metrics": self.latency.measure_input_latency(),
        }


def create_app_optimizer() -> ApplicationOptimizer:
    """Factory function."""
    return ApplicationOptimizer()


if __name__ == "__main__":
    optimizer = ApplicationOptimizer()

    print("=== GAMESA Application Optimizer ===\n")

    status = optimizer.get_system_status()

    print("Top Processes by Category:")
    for category, procs in status["by_category"].items():
        print(f"\n  {category}:")
        for p in procs[:3]:
            print(f"    - {p['name']} (PID {p['pid']}): {p['memory_mb']:.0f}MB, {p['threads']} threads")

    print(f"\nLatency Metrics: {status['latency_metrics']}")

    print("\n\nSystem Optimization Summary:")
    results = optimizer.optimize_system()
    print(f"  Processes analyzed: {results['processes_analyzed']}")
    print(f"  Optimizations suggested: {len(results['optimizations'])}")

    for opt in results["optimizations"][:5]:
        print(f"\n  {opt['process']} ({opt['category']}):")
        print(f"    Profile: {opt['profile']}")
        print(f"    Actions: {opt['actions']}")
