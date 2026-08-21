#!/usr/bin/env python3
"""
Windows System Utility Framework - Cross-Forex Resource Market for Windows

Leverages Windows-specific system access to create an intelligent resource trading system.
Combines economic metaphors with Windows internals (Registry, memory, processes, timers).

Features:
- Resource trading market for CPU, memory, disk, network
- Windows Registry access and optimization
- Process monitoring and management
- System timers and scheduling
- UUID-based resource tracking
"""

import os
import sys
import time
import threading
import uuid
import json
import winreg
import psutil
import ctypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS AND DATA CLASSES
# ============================================================

class ResourceType(Enum):
    """Types of system resources that can be traded."""
    CPU_CORE = "cpu_core"
    CPU_TIME = "cpu_time"  # microseconds
    MEMORY = "memory"      # MB
    DISK_IO = "disk_io"    # IOPS
    NETWORK = "network"    # bandwidth MB/s
    GPU_COMPUTE = "gpu_compute"  # percentage
    GPU_MEMORY = "gpu_memory"    # MB
    HANDLE_COUNT = "handle_count"  # number of handles
    REGISTRY_QUOTA = "registry_quota"  # registry entries


class Priority(Enum):
    """Priority levels for resource requests."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AllocationRequest:
    """Request for system resources."""
    request_id: str
    agent_id: str
    resource_type: ResourceType
    amount: float
    priority: Priority = Priority.NORMAL
    duration_ms: int = 1000  # How long to hold the resource
    bid_credits: float = 1.0  # Economic value for resource
    timestamp: float = field(default_factory=time.time)


@dataclass
class Allocation:
    """Granted system resource allocation."""
    allocation_id: str
    request_id: str
    agent_id: str
    resource_type: ResourceType
    amount: float
    granted_at: float
    expires_at: float
    status: str = "active"  # active, expired, revoked


@dataclass
class SystemTelemetry:
    """System state telemetry for decision making."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_bytes_per_sec: float = 0.0
    process_count: int = 0
    handle_count: int = 0
    uptime_seconds: float = 0.0
    thermal_headroom: float = 20.0  # degrees C
    power_headroom: float = 50.0    # watts


# ============================================================
# RESOURCE POOLS
# ============================================================

class ResourcePool:
    """Base class for system resource pools."""
    
    def __init__(self, resource_type: ResourceType, total_capacity: float):
        self.resource_type = resource_type
        self.total_capacity = total_capacity
        self.allocated = 0.0
        self.allocations: Dict[str, Allocation] = {}
        self.lock = threading.RLock()
    
    def available(self) -> float:
        """Return available resource amount."""
        return self.total_capacity - self.allocated
    
    def allocate(self, request: AllocationRequest) -> Optional[Allocation]:
        """Allocate resources if available."""
        with self.lock:
            if request.amount <= self.available():
                self.allocated += request.amount
                
                allocation = Allocation(
                    allocation_id=str(uuid.uuid4()),
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    resource_type=request.resource_type,
                    amount=request.amount,
                    granted_at=time.time(),
                    expires_at=time.time() + (request.duration_ms / 1000.0)
                )
                
                self.allocations[allocation.allocation_id] = allocation
                return allocation
            return None
    
    def release(self, allocation_id: str):
        """Release allocated resources."""
        with self.lock:
            if allocation_id in self.allocations:
                allocation = self.allocations[allocation_id]
                self.allocated -= allocation.amount
                del self.allocations[allocation_id]


class CPUCorePool(ResourcePool):
    """Pool of CPU cores."""
    
    def __init__(self):
        # Use number of logical processors
        cpu_count = psutil.cpu_count(logical=True)
        super().__init__(ResourceType.CPU_CORE, float(cpu_count))
        
        # Track core affinity for processes
        self.core_affinity: Dict[str, List[int]] = {}  # agent_id -> [core_ids]


class MemoryPool(ResourcePool):
    """Pool of system memory."""
    
    def __init__(self):
        # Use available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        super().__init__(ResourceType.MEMORY, memory_gb * 1000)  # In MB


class DiskIOPool(ResourcePool):
    """Pool of disk I/O operations."""
    
    def __init__(self):
        # Estimate based on system capabilities
        super().__init__(ResourceType.DISK_IO, 100000.0)  # IOPS


class NetworkPool(ResourcePool):
    """Pool of network bandwidth."""
    
    def __init__(self):
        # Estimate based on common speeds (100 Mbps = ~12MB/s)
        super().__init__(ResourceType.NETWORK, 100.0)  # MB/s


# ============================================================
# WINDOWS-SPECIFIC INTEGRATION
# ============================================================

class WindowsRegistryManager:
    """Interface to Windows Registry with safety features."""
    
    def __init__(self):
        self.backup_history = []
    
    def read_value(self, key_path: str, value_name: str) -> Any:
        """Read a registry value."""
        try:
            hkey_map = {
                "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
                "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
                "HKEY_CLASSES_ROOT": winreg.HKEY_CLASSES_ROOT,
                "HKEY_USERS": winreg.HKEY_USERS,
            }

            # Parse key path
            parts = key_path.split("\\", 1)
            hkey_str = parts[0]
            subkey = parts[1] if len(parts) > 1 else ""

            hkey = hkey_map.get(hkey_str, winreg.HKEY_CURRENT_USER)  # Changed from HKEY_CURRENT_MACHINE to winreg.HKEY_CURRENT_MACHINE

            with winreg.OpenKey(hkey, subkey) as key:
                value, _ = winreg.QueryValueEx(key, value_name)
                return value
        except Exception as e:
            logger.error(f"Error reading registry {key_path}\\{value_name}: {e}")
            return None
    
    def write_value(self, key_path: str, value_name: str, value: Any, value_type=None) -> bool:
        """Write a registry value with automatic backup."""
        try:
            # Determine value type if not specified
            if value_type is None:
                if isinstance(value, str):
                    value_type = winreg.REG_SZ
                elif isinstance(value, int):
                    value_type = winreg.REG_DWORD
                elif isinstance(value, bytes):
                    value_type = winreg.REG_BINARY
                else:
                    value_type = winreg.REG_SZ  # Default
            
            hkey_map = {
                "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
                "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
            }
            
            parts = key_path.split("\\", 1)
            hkey_str = parts[0]
            subkey = parts[1] if len(parts) > 1 else ""
            
            hkey = hkey_map.get(hkey_str, winreg.HKEY_CURRENT_USER)
            
            # Backup before writing
            old_value = self.read_value(key_path, value_name)
            backup_entry = {
                "key_path": key_path,
                "value_name": value_name,
                "old_value": old_value,
                "timestamp": time.time()
            }
            self.backup_history.append(backup_entry)
            
            with winreg.OpenKey(hkey, subkey, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, value_name, 0, value_type, value)
                return True
        except Exception as e:
            logger.error(f"Error writing registry {key_path}\\{value_name}: {e}")
            return False
    
    def get_backup_count(self) -> int:
        """Get number of backup entries."""
        return len(self.backup_history)


class WindowsProcessManager:
    """Interface to Windows processes with UUID tracking."""
    
    def __init__(self):
        self.process_uuids: Dict[int, str] = {}  # pid -> uuid
        self.uuid_process: Dict[str, int] = {}  # uuid -> pid
        self.process_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def track_process(self, pid: int) -> str:
        """Assign a UUID to a process for tracking."""
        if pid not in self.process_uuids:
            process_uuid = str(uuid.uuid4())
            self.process_uuids[pid] = process_uuid
            self.uuid_process[process_uuid] = pid
        return self.process_uuids[pid]
    
    def get_process_uuid(self, pid: int) -> Optional[str]:
        """Get UUID for a process."""
        return self.process_uuids.get(pid)
    
    def get_process_by_uuid(self, process_uuid: str) -> Optional[int]:
        """Get process ID by UUID."""
        return self.uuid_process.get(process_uuid)
    
    def get_all_processes(self) -> List[Dict[str, Any]]:
        """Get all running processes with tracking info."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                pinfo = proc.info
                uuid = self.track_process(pinfo['pid'])

                # Get additional info
                p = psutil.Process(pinfo['pid'])
                pinfo['uuid'] = uuid
                pinfo['num_threads'] = p.num_threads()
                pinfo['handles'] = len(p.open_files()) + len(p.net_connections())

                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return processes
    
    def terminate_process_by_uuid(self, process_uuid: str) -> bool:
        """Terminate process by UUID."""
        pid = self.get_process_by_uuid(process_uuid)
        if pid:
            try:
                p = psutil.Process(pid)
                p.terminate()
                return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
        return False


class WindowsTimerManager:
    """Interface to Windows timers and scheduling."""
    
    def __init__(self):
        self.timers: Dict[str, threading.Timer] = {}
        self.scheduled_tasks: List[Dict] = []
        self.lock = threading.Lock()
    
    def schedule_task(self, 
                     task_id: str, 
                     delay_seconds: float, 
                     callback: Callable,
                     *args,
                     **kwargs) -> bool:
        """Schedule a task to run after delay."""
        with self.lock:
            if task_id in self.timers:
                # Cancel existing timer
                self.timers[task_id].cancel()
            
            timer = threading.Timer(delay_seconds, callback, args=args, kwargs=kwargs)
            timer.daemon = True
            timer.start()
            
            self.timers[task_id] = timer
            
            # Record in schedule
            self.scheduled_tasks.append({
                "task_id": task_id,
                "scheduled_at": time.time(),
                "run_at": time.time() + delay_seconds,
                "callback": callback.__name__ if hasattr(callback, '__name__') else str(callback)
            })
            
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        with self.lock:
            if task_id in self.timers:
                self.timers[task_id].cancel()
                del self.timers[task_id]
                return True
            return False
    
    def get_scheduled_count(self) -> int:
        """Get number of scheduled tasks."""
        return len(self.timers)


class WindowsResourceManager:
    """Main Windows system resource manager."""
    
    def __init__(self):
        # System managers
        self.registry = WindowsRegistryManager()
        self.processes = WindowsProcessManager()
        self.timers = WindowsTimerManager()
        
        # Resource pools
        self.pools: Dict[ResourceType, ResourcePool] = {
            ResourceType.CPU_CORE: CPUCorePool(),
            ResourceType.MEMORY: MemoryPool(),
            ResourceType.DISK_IO: DiskIOPool(),
            ResourceType.NETWORK: NetworkPool(),
        }
        
        # Economic parameters
        self.credits: Dict[str, float] = defaultdict(lambda: 100.0)  # agent_id -> credits
        self.trade_history = deque(maxlen=1000)
        
        # Telemetry
        self.telemetry_history = deque(maxlen=1000)
        
        # Safety limits
        self.safety_limits = {
            "max_cpu": 95.0,     # Don't allow system to exceed 95% CPU
            "max_memory": 90.0,  # Don't allow system to exceed 90% memory
            "min_thermal": 5.0,  # Keep at least 5C thermal headroom
        }
    
    def collect_telemetry(self) -> SystemTelemetry:
        """Collect system telemetry."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        # Disk
        disk_percent = psutil.disk_usage('/').percent  # Root drive usage

        # Network (simple bytes since last call)
        net_io = psutil.net_io_counters()

        # Process stats
        process_count = len(psutil.pids())

        # Handle count (with error handling for permissions)
        handle_count = 0
        for proc in psutil.process_iter():
            try:
                if proc.is_running():
                    # Use net_connections() instead of connections() as per deprecation warning
                    handle_count += len(proc.open_files()) + len(proc.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                # Skip processes we don't have access to
                continue

        # Uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time

        # Thermal (simplified - real system would use more accurate methods)
        thermal_headroom = max(0, 20.0 - (cpu_percent * 0.2))  # Estimate

        # Power (simplified)
        power_headroom = max(0, 100.0 - cpu_percent)

        telemetry = SystemTelemetry(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_bytes_per_sec=net_io.bytes_sent + net_io.bytes_recv,
            process_count=process_count,
            handle_count=handle_count,
            uptime_seconds=uptime,
            thermal_headroom=thermal_headroom,
            power_headroom=power_headroom
        )

        self.telemetry_history.append(telemetry)
        return telemetry
    
    def allocate_resource(self, request: AllocationRequest) -> Optional[Allocation]:
        """Allocate resource with safety checks."""
        # Check safety limits
        if self._would_violate_safety(request):
            logger.warning(f"Allocation request {request.request_id} would violate safety limits")
            return None
        
        # Check agent credits
        if self.credits[request.agent_id] < request.bid_credits:
            logger.warning(f"Agent {request.agent_id} insufficient credits for allocation")
            return None
        
        # Allocate from appropriate pool
        pool = self.pools.get(request.resource_type)
        if pool:
            allocation = pool.allocate(request)
            if allocation:
                # Deduct credits
                self.credits[request.agent_id] -= request.bid_credits
                
                # Record trade
                self.trade_history.append({
                    "timestamp": time.time(),
                    "request_id": request.request_id,
                    "allocation_id": allocation.allocation_id,
                    "agent": request.agent_id,
                    "resource": request.resource_type.value,
                    "amount": request.amount,
                    "cost": request.bid_credits
                })
                
                return allocation
        
        return None
    
    def _would_violate_safety(self, request: AllocationRequest) -> bool:
        """Check if allocation would violate safety limits."""
        telemetry = self.collect_telemetry()
        
        if request.resource_type == ResourceType.CPU_CORE:
            # Check if this allocation would exceed CPU limits
            pool = self.pools[ResourceType.CPU_CORE]
            new_utilization = (pool.allocated + request.amount) / pool.total_capacity * 100
            if new_utilization > self.safety_limits["max_cpu"]:
                return True
        
        elif request.resource_type == ResourceType.MEMORY:
            # Check if this allocation would exceed memory limits
            pool = self.pools[ResourceType.MEMORY]
            new_utilization = (pool.allocated + request.amount) / pool.total_capacity * 100
            if new_utilization > self.safety_limits["max_memory"]:
                return True
        
        return False
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status information for an agent."""
        return {
            "agent_id": agent_id,
            "credits": self.credits[agent_id],
            "allocated_resources": {
                rt.name: pool.allocated 
                for rt, pool in self.pools.items()
                if pool.allocated > 0
            },
            "total_resources_used": sum(pool.allocated for pool in self.pools.values()),
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        telemetry = self.collect_telemetry()
        
        return {
            "telemetry": telemetry,
            "resource_pools": {
                rt.name: {
                    "total": pool.total_capacity,
                    "allocated": pool.allocated,
                    "available": pool.available(),
                    "allocation_rate": (pool.allocated / pool.total_capacity) if pool.total_capacity > 0 else 0
                }
                for rt, pool in self.pools.items()
            },
            "registry_backups": self.registry.get_backup_count(),
            "active_processes": len(self.processes.get_all_processes()),
            "scheduled_tasks": self.timers.get_scheduled_count(),
            "trade_count": len(self.trade_history),
        }


# ============================================================
# DEMONSTRATION
# ============================================================

def demo():
    """Demonstrate the Windows System Utility Framework."""
    print("=" * 80)
    print("WINDOWS SYSTEM UTILITY FRAMEWORK - CROSS-FOREX RESOURCE MARKET")
    print("=" * 80)

    # Create manager
    manager = WindowsResourceManager()
    print("[OK] Windows Resource Manager initialized")

    # Show initial status
    status = manager.get_system_status()
    print("[OK] System status collected:")
    print(f"  - CPU cores: {status['resource_pools']['CPU_CORE']['total']}")
    print(f"  - Memory: {status['resource_pools']['MEMORY']['total']:.1f} MB")
    print(f"  - Active processes: {status['active_processes']}")
    print(f"  - Scheduled tasks: {status['scheduled_tasks']}")

    # Create agents
    agents = ["GameOptimizer", "BackupAgent", "AIProcessor", "ThermalGuard"]
    for agent in agents:
        manager.credits[agent] = 1000.0  # Give each agent 1000 credits
    print(f"[OK] Created {len(agents)} agents with 1000 credits each")

    # Demonstrate resource allocation
    print("\n--- Resource Allocation Demo ---")

    # Agent 1: Game Optimizer requests CPU cores
    cpu_request = AllocationRequest(
        request_id=str(uuid.uuid4()),
        agent_id="GameOptimizer",
        resource_type=ResourceType.CPU_CORE,
        amount=2.0,
        priority=Priority.HIGH,
        bid_credits=50.0
    )

    allocation = manager.allocate_resource(cpu_request)
    if allocation:
        print(f"[OK] GameOptimizer allocated {allocation.amount} CPU cores (ID: {allocation.allocation_id[:8]})")
    else:
        print("[ERROR] GameOptimizer CPU allocation failed")

    # Agent 2: Backup Agent requests disk I/O
    disk_request = AllocationRequest(
        request_id=str(uuid.uuid4()),
        agent_id="BackupAgent",
        resource_type=ResourceType.DISK_IO,
        amount=10000.0,
        priority=Priority.NORMAL,
        bid_credits=25.0
    )

    allocation = manager.allocate_resource(disk_request)
    if allocation:
        print(f"[OK] BackupAgent allocated {allocation.amount} IOPS (ID: {allocation.allocation_id[:8]})")
    else:
        print("[ERROR] BackupAgent disk allocation failed")

    # Agent 3: AI Processor requests memory
    memory_request = AllocationRequest(
        request_id=str(uuid.uuid4()),
        agent_id="AIProcessor",
        resource_type=ResourceType.MEMORY,
        amount=1024.0,  # 1GB
        priority=Priority.CRITICAL,
        bid_credits=200.0
    )

    allocation = manager.allocate_resource(memory_request)
    if allocation:
        print(f"[OK] AIProcessor allocated {allocation.amount} MB memory (ID: {allocation.allocation_id[:8]})")
    else:
        print("[ERROR] AIProcessor memory allocation failed")

    # Show agent status
    print("\n--- Agent Status ---")
    for agent in agents:
        status = manager.get_agent_status(agent)
        print(f"{agent}: {status['credits']:.1f} credits, "
              f"resources used: {status['total_resources_used']:.1f}")

    # Show system status
    print("\n--- Current System Status ---")
    sys_status = manager.get_system_status()
    for resource, stats in sys_status["resource_pools"].items():
        if stats["allocated"] > 0:
            print(f"{resource}: {stats['allocated']:.1f}/{stats['total']:.1f} "
                  f"({stats['allocation_rate']:.1%} allocated)")

    # Demonstrate process management
    print("\n--- Process Management Demo ---")
    processes = manager.processes.get_all_processes()[:5]  # Show first 5 processes
    for proc in processes:
        print(f"  PID {proc['pid']}: {proc['name']} (UUID: {proc['uuid'][:8]}...)")

    # Demonstrate registry access
    print("\n--- Registry Access Demo ---")
    # Read a safe registry value
    value = manager.registry.read_value("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "ProductName")
    if value:
        print(f"  Windows Product Name: {value}")
    else:
        print("  Could not read registry (may need admin privileges)")

    # Demonstrate timer scheduling
    print("\n--- Timer Scheduling Demo ---")
    def sample_task(task_name: str):
        print(f"  [TIMER] Executing scheduled task: {task_name}")

    manager.timers.schedule_task("demo_task", 2.0, sample_task, "Hello from Timer")
    print("  Scheduled task 'demo_task' to run in 2 seconds")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("This framework provides economic resource trading for Windows systems")
    print("with deep system access, UUID tracking, and intelligent scheduling")
    print("=" * 80)


if __name__ == "__main__":
    demo()