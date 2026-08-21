"""
System Introspection & Debug Console 🔬
Responsibility:
1. Deep system inspection and diagnostics.
2. Real-time memory and performance metrics.
3. Code introspection and module listing.
"""
import sys
import os
import gc
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class SystemIntrospector:
    """Deep system introspection for power users."""
    
    def __init__(self):
        self._start_time = time.time()
        self._request_count = 0

    def get_system_info(self) -> Dict[str, Any]:
        """Returns comprehensive system information."""
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "node": platform.node(),
            "uptime_seconds": time.time() - self._start_time,
            "uptime_human": self._format_uptime(time.time() - self._start_time),
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "thread_count": threading.active_count()
        }

    def get_memory_info(self) -> Dict[str, Any]:
        """Returns memory usage statistics."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            return {
                "rss_mb": round(mem.rss / 1024 / 1024, 2),
                "vms_mb": round(mem.vms / 1024 / 1024, 2),
                "percent": process.memory_percent(),
                "gc_counts": gc.get_count(),
                "gc_threshold": gc.get_threshold()
            }
        except ImportError:
            return {
                "gc_counts": gc.get_count(),
                "gc_threshold": gc.get_threshold(),
                "note": "Install psutil for detailed memory info"
            }

    def get_loaded_modules(self, filter_prefix: str = "backend") -> List[Dict[str, Any]]:
        """Lists loaded Python modules."""
        modules = []
        for name, module in sorted(sys.modules.items()):
            if filter_prefix and not name.startswith(filter_prefix):
                continue
            modules.append({
                "name": name,
                "file": getattr(module, "__file__", None),
                "package": getattr(module, "__package__", None)
            })
        return modules

    def get_environment_variables(self, safe: bool = True) -> Dict[str, str]:
        """Returns environment variables (optionally hiding sensitive ones)."""
        sensitive_keys = ["KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL"]
        env = {}
        for key, value in os.environ.items():
            if safe and any(s in key.upper() for s in sensitive_keys):
                env[key] = "***HIDDEN***"
            else:
                env[key] = value
        return env

    def get_api_endpoints(self) -> List[Dict[str, Any]]:
        """Lists all registered API endpoints."""
        try:
            from backend.main import app
            endpoints = []
            for route in app.routes:
                if hasattr(route, 'path'):
                    endpoints.append({
                        "path": route.path,
                        "name": getattr(route, 'name', None),
                        "methods": list(getattr(route, 'methods', [])) if hasattr(route, 'methods') else []
                    })
            return sorted(endpoints, key=lambda x: x['path'])
        except Exception as e:
            return [{"error": str(e)}]

    def run_garbage_collection(self) -> Dict[str, Any]:
        """Forces garbage collection and returns stats."""
        before = gc.get_count()
        collected = gc.collect()
        after = gc.get_count()
        return {
            "collected": collected,
            "before": before,
            "after": after
        }

    def get_thread_info(self) -> List[Dict[str, Any]]:
        """Lists all active threads."""
        threads = []
        for thread in threading.enumerate():
            threads.append({
                "name": thread.name,
                "ident": thread.ident,
                "daemon": thread.daemon,
                "alive": thread.is_alive()
            })
        return threads

    def _format_uptime(self, seconds: float) -> str:
        """Formats uptime as human-readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}h {minutes}m {secs}s"

    def increment_request_count(self):
        self._request_count += 1

    def get_request_count(self) -> int:
        return self._request_count


class DebugConsole:
    """Interactive debug console for advanced diagnostics."""
    
    def __init__(self):
        self.command_history: List[Dict[str, Any]] = []
        self.max_history = 100
        self.introspector = SystemIntrospector()

    def execute(self, command: str) -> Dict[str, Any]:
        """Executes a debug command."""
        self._log_command(command)
        
        cmd_parts = command.strip().lower().split()
        if not cmd_parts:
            return {"error": "No command provided"}
        
        cmd = cmd_parts[0]
        args = cmd_parts[1:] if len(cmd_parts) > 1 else []

        handlers = {
            "help": self._cmd_help,
            "sysinfo": self._cmd_sysinfo,
            "memory": self._cmd_memory,
            "modules": self._cmd_modules,
            "env": self._cmd_env,
            "endpoints": self._cmd_endpoints,
            "gc": self._cmd_gc,
            "threads": self._cmd_threads,
            "deps": self._cmd_deps,
            "history": self._cmd_history,
            "ping": self._cmd_ping,
            "echo": lambda a: {"output": " ".join(a) if a else "(empty)"}
        }

        if cmd in handlers:
            return handlers[cmd](args)
        else:
            return {"error": f"Unknown command: {cmd}", "available": list(handlers.keys())}

    def _cmd_help(self, args) -> Dict[str, Any]:
        return {
            "commands": {
                "help": "Show this help message",
                "sysinfo": "Display system information",
                "memory": "Show memory usage",
                "modules [prefix]": "List loaded Python modules",
                "env": "Show environment variables",
                "endpoints": "List API endpoints",
                "gc": "Run garbage collection",
                "threads": "List active threads",
                "deps": "List project dependencies (Python & Node)",
                "history": "Show command history",
                "ping": "Check if console is responsive",
                "echo [text]": "Echo back text"
            }
        }

    def _cmd_sysinfo(self, args) -> Dict[str, Any]:
        return {"system": self.introspector.get_system_info()}

    def _cmd_memory(self, args) -> Dict[str, Any]:
        return {"memory": self.introspector.get_memory_info()}

    def _cmd_modules(self, args) -> Dict[str, Any]:
        prefix = args[0] if args else "backend"
        return {"modules": self.introspector.get_loaded_modules(prefix)}

    def _cmd_env(self, args) -> Dict[str, Any]:
        return {"environment": self.introspector.get_environment_variables()}

    def _cmd_endpoints(self, args) -> Dict[str, Any]:
        return {"endpoints": self.introspector.get_api_endpoints()}

    def _cmd_gc(self, args) -> Dict[str, Any]:
        return {"garbage_collection": self.introspector.run_garbage_collection()}

    def _cmd_threads(self, args) -> Dict[str, Any]:
        return {"threads": self.introspector.get_thread_info()}

    def _cmd_deps(self, args) -> Dict[str, Any]:
        """Lists Python and Node.js dependencies from config files."""
        deps = {"python": [], "frontend_react": {}, "frontend_vue": {}}
        
        # 1. Python (requirements.txt)
        req_path = os.path.join(os.getcwd(), "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, "r") as f:
                deps["python"] = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        # 2. Node.js (package.json)
        import json
        for folder in ["frontend-react", "frontend-vue"]:
            pkg_path = os.path.join(os.getcwd(), folder, "package.json")
            if os.path.exists(pkg_path):
                with open(pkg_path, "r") as f:
                    try:
                        pkg = json.load(f)
                        deps[folder.replace("-", "_")] = {
                            "dependencies": pkg.get("dependencies", {}),
                            "devDependencies": pkg.get("devDependencies", {})
                        }
                    except:
                        deps[folder.replace("-", "_")] = "Error parsing package.json"
                        
        return {"dependencies": deps}

    def _cmd_history(self, args) -> Dict[str, Any]:
        return {"history": self.command_history[-20:]}

    def _cmd_ping(self, args) -> Dict[str, Any]:
        return {"pong": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    def _log_command(self, command: str):
        self.command_history.append({
            "command": command,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]


# Global Instances
system_introspector = SystemIntrospector()
debug_console = DebugConsole()
