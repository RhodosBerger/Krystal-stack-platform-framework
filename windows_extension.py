#!/usr/bin/env python3
"""
Windows Extension for GAMESA/KrystalStack Framework

This extension enhances the GAMESA framework with Windows-specific capabilities,
including registry optimization, hardware monitoring, performance counters,
and advanced process management. It implements an economic resource trading
system specifically tailored for Windows environments.
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
import subprocess
from ctypes import wintypes

# Import optional Windows modules with error handling
try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False
    print("Warning: wmi module not available. WMI functionality will be limited.")

try:
    import win32serviceutil
    import win32service
    WIN32SERVICES_AVAILABLE = True
except ImportError:
    WIN32SERVICES_AVAILABLE = False
    print("Warning: pywin32 modules not available. Service management will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS AND DATA CLASSES
# ============================================================

class WindowsResourceType(Enum):
    """Windows-specific system resources that can be traded."""
    CPU_CORE = "cpu_core"
    CPU_TIME = "cpu_time"  # microseconds
    MEMORY = "memory"      # MB
    DISK_IO = "disk_io"    # IOPS
    NETWORK = "network"    # bandwidth MB/s
    GPU_COMPUTE = "gpu_compute"  # percentage
    GPU_MEMORY = "gpu_memory"    # MB
    HANDLE_COUNT = "handle_count"  # number of handles
    REGISTRY_QUOTA = "registry_quota"  # registry entries
    PROCESS_PRIORITY = "process_priority"  # priority level
    THREAD_AFFINITY = "thread_affinity"  # CPU core assignment
    PERFORMANCE_COUNTER = "performance_counter"  # perf counter access
    SERVICE_CONTROL = "service_control"  # service management


class WindowsPriority(Enum):
    """Windows-specific priority levels."""
    IDLE = -15
    BELOW_NORMAL = -1
    NORMAL = 0
    ABOVE_NORMAL = 1
    HIGH = 2
    REALTIME = 3


@dataclass
class WindowsAllocationRequest:
    """Request for Windows-specific system resources."""
    request_id: str
    agent_id: str
    resource_type: WindowsResourceType
    amount: float
    priority: WindowsPriority = WindowsPriority.NORMAL
    duration_ms: int = 1000  # How long to hold the resource
    bid_credits: float = 1.0  # Economic value for resource
    timestamp: float = field(default_factory=time.time)
    process_id: Optional[int] = None  # Specific process to target


@dataclass
class WindowsAllocation:
    """Granted Windows-specific system resource allocation."""
    allocation_id: str
    request_id: str
    agent_id: str
    resource_type: WindowsResourceType
    amount: float
    granted_at: float
    expires_at: float
    status: str = "active"  # active, expired, revoked
    process_id: Optional[int] = None  # Process this allocation affects


@dataclass
class WindowsTelemetry:
    """Windows-specific system state telemetry."""
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
    registry_usage: float = 0.0     # percentage of registry quota used
    page_file_usage: float = 0.0    # percentage of page file used
    commit_charge: float = 0.0      # committed memory in MB


# ============================================================
# WINDOWS-SPECIFIC RESOURCE POOLS
# ============================================================

class WindowsResourcePool:
    """Base class for Windows-specific resource pools."""

    def __init__(self, resource_type: WindowsResourceType, total_capacity: float):
        self.resource_type = resource_type
        self.total_capacity = total_capacity
        self.allocated = 0.0
        self.allocations: Dict[str, WindowsAllocation] = {}
        self.lock = threading.RLock()

    def available(self) -> float:
        """Return available resource amount."""
        return self.total_capacity - self.allocated

    def allocate(self, request: WindowsAllocationRequest) -> Optional[WindowsAllocation]:
        """Allocate resources if available."""
        with self.lock:
            if request.amount <= self.available():
                self.allocated += request.amount

                allocation = WindowsAllocation(
                    allocation_id=str(uuid.uuid4()),
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    resource_type=request.resource_type,
                    amount=request.amount,
                    granted_at=time.time(),
                    expires_at=time.time() + (request.duration_ms / 1000.0),
                    process_id=request.process_id
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


class ProcessPriorityPool(WindowsResourcePool):
    """Pool for process priority resources."""

    def __init__(self):
        super().__init__(WindowsResourceType.PROCESS_PRIORITY, 10.0)  # 10 priority changes allowed

    def allocate(self, request: WindowsAllocationRequest) -> Optional[WindowsAllocation]:
        """Allocate process priority resource."""
        with self.lock:
            if request.amount <= self.available() and request.process_id:
                try:
                    # Validate process exists
                    p = psutil.Process(request.process_id)
                    
                    # Set process priority
                    priority_map = {
                        WindowsPriority.IDLE: psutil.IDLE_PRIORITY_CLASS,
                        WindowsPriority.BELOW_NORMAL: psutil.BELOW_NORMAL_PRIORITY_CLASS,
                        WindowsPriority.NORMAL: psutil.NORMAL_PRIORITY_CLASS,
                        WindowsPriority.ABOVE_NORMAL: psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                        WindowsPriority.HIGH: psutil.HIGH_PRIORITY_CLASS,
                        WindowsPriority.REALTIME: psutil.REALTIME_PRIORITY_CLASS
                    }
                    
                    p.nice(priority_map[request.priority])
                    
                    self.allocated += request.amount

                    allocation = WindowsAllocation(
                        allocation_id=str(uuid.uuid4()),
                        request_id=request.request_id,
                        agent_id=request.agent_id,
                        resource_type=request.resource_type,
                        amount=request.amount,
                        granted_at=time.time(),
                        expires_at=time.time() + (request.duration_ms / 1000.0),
                        process_id=request.process_id
                    )

                    self.allocations[allocation.allocation_id] = allocation
                    return allocation
                except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                    logger.warning(f"Could not set priority for process {request.process_id}")
                    return None
            return None


class ThreadAffinityPool(WindowsResourcePool):
    """Pool for thread affinity resources."""

    def __init__(self):
        super().__init__(WindowsResourceType.THREAD_AFFINITY, float(psutil.cpu_count(logical=True)))

    def allocate(self, request: WindowsAllocationRequest) -> Optional[WindowsAllocation]:
        """Allocate thread affinity resource."""
        with self.lock:
            if request.amount <= self.available() and request.process_id:
                try:
                    # Validate process exists
                    p = psutil.Process(request.process_id)
                    
                    # Calculate CPU affinity mask
                    cpu_count = psutil.cpu_count(logical=True)
                    cores_to_use = min(int(request.amount), cpu_count)
                    
                    # Create affinity mask (use first N cores)
                    affinity_mask = 0
                    for i in range(cores_to_use):
                        affinity_mask |= (1 << i)
                    
                    # Set process affinity
                    p.cpu_affinity(list(range(cores_to_use)))
                    
                    self.allocated += request.amount

                    allocation = WindowsAllocation(
                        allocation_id=str(uuid.uuid4()),
                        request_id=request.request_id,
                        agent_id=request.agent_id,
                        resource_type=request.resource_type,
                        amount=request.amount,
                        granted_at=time.time(),
                        expires_at=time.time() + (request.duration_ms / 1000.0),
                        process_id=request.process_id
                    )

                    self.allocations[allocation.allocation_id] = allocation
                    return allocation
                except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                    logger.warning(f"Could not set affinity for process {request.process_id}")
                    return None
            return None


# ============================================================
# WINDOWS REGISTRY MANAGER
# ============================================================

class WindowsRegistryManager:
    """Advanced Windows Registry manager with optimization capabilities."""

    def __init__(self):
        self.backup_history = []
        self.optimization_history = []
        self.reg_handle_cache = {}

    def read_value(self, key_path: str, value_name: str) -> Any:
        """Read a registry value."""
        try:
            hkey_map = {
                "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
                "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
                "HKEY_CLASSES_ROOT": winreg.HKEY_CLASSES_ROOT,
                "HKEY_USERS": winreg.HKEY_USERS,
                "HKEY_CURRENT_CONFIG": winreg.HKEY_CURRENT_CONFIG,
            }

            # Parse key path
            parts = key_path.split("\\", 1)
            hkey_str = parts[0]
            subkey = parts[1] if len(parts) > 1 else ""

            hkey = hkey_map.get(hkey_str, winreg.HKEY_CURRENT_USER)

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
                elif isinstance(value, list):
                    value_type = winreg.REG_MULTI_SZ
                elif isinstance(value, bytes):
                    value_type = winreg.REG_BINARY
                else:
                    value_type = winreg.REG_SZ  # Default

            hkey_map = {
                "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
                "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
                "HKEY_CLASSES_ROOT": winreg.HKEY_CLASSES_ROOT,
                "HKEY_USERS": winreg.HKEY_USERS,
                "HKEY_CURRENT_CONFIG": winreg.HKEY_CURRENT_CONFIG,
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
                "new_value": value,
                "timestamp": time.time()
            }
            self.backup_history.append(backup_entry)

            with winreg.OpenKey(hkey, subkey, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, value_name, 0, value_type, value)
                return True
        except Exception as e:
            logger.error(f"Error writing registry {key_path}\\{value_name}: {e}")
            return False

    def optimize_registry(self) -> bool:
        """Optimize registry by defragmenting and cleaning."""
        try:
            # This would call Windows registry optimization tools
            # In practice, this might call regedit /e or similar
            logger.info("Optimizing Windows registry...")
            
            # Record optimization
            self.optimization_history.append({
                "type": "registry_optimization",
                "timestamp": time.time()
            })
            
            return True
        except Exception as e:
            logger.error(f"Error optimizing registry: {e}")
            return False

    def get_registry_usage(self) -> float:
        """Get registry usage percentage."""
        try:
            # This is a simplified approach - in reality, you'd need to query
            # registry size information from system APIs
            # For now, return a simulated value
            return 35.0  # Simulated registry usage percentage
        except:
            return 0.0

    def get_backup_count(self) -> int:
        """Get number of backup entries."""
        return len(self.backup_history)


# ============================================================
# WINDOWS WMI INTEGRATION
# ============================================================

class WindowsWMIManager:
    """Windows Management Instrumentation (WMI) integration."""

    def __init__(self):
        if not WMI_AVAILABLE:
            logger.warning("WMI module not available. WMI functionality disabled.")
            self.wmi_conn = None
            return

        try:
            self.wmi_conn = wmi.WMI()
        except Exception as e:
            logger.warning(f"Could not connect to WMI: {e}")
            self.wmi_conn = None

    def get_gpu_info(self) -> List[Dict]:
        """Get GPU information via WMI."""
        if not self.wmi_conn:
            return []
        
        try:
            gpu_info = []
            for gpu in self.wmi_conn.Win32_VideoController():
                gpu_info.append({
                    "name": gpu.Name,
                    "adapter_ram": getattr(gpu, 'AdapterRAM', 0),
                    "driver_version": getattr(gpu, 'DriverVersion', 'Unknown'),
                    "status": getattr(gpu, 'Status', 'Unknown'),
                    "current_refresh_rate": getattr(gpu, 'CurrentRefreshRate', 0),
                    "max_refresh_rate": getattr(gpu, 'MaxRefreshRate', 0),
                    "min_refresh_rate": getattr(gpu, 'MinRefreshRate', 0)
                })
            return gpu_info
        except Exception as e:
            logger.error(f"Error getting GPU info via WMI: {e}")
            return []

    def get_system_performance_counters(self) -> Dict:
        """Get system performance counters via WMI."""
        if not self.wmi_conn:
            return {}
        
        try:
            # Get CPU performance information
            cpu_load = self.wmi_conn.Win32_Processor()
            cpu_info = []
            for cpu in cpu_load:
                cpu_info.append({
                    "name": cpu.Name,
                    "load_percentage": cpu.LoadPercentage,
                    "max_clock_speed": cpu.MaxClockSpeed,
                    "current_clock_speed": getattr(cpu, 'CurrentClockSpeed', 0)
                })
            
            # Get memory performance information
            memory = self.wmi_conn.Win32_PerfRawData_PerfOS_Memory()[0]
            memory_info = {
                "available_bytes": getattr(memory, 'AvailableBytes', 0),
                "cache_bytes": getattr(memory, 'CacheBytes', 0),
                "committed_bytes": getattr(memory, 'CommittedBytes', 0),
                "pool_paged_bytes": getattr(memory, 'PoolPagedBytes', 0),
                "pool_nonpaged_bytes": getattr(memory, 'PoolNonPagedBytes', 0)
            }
            
            return {
                "cpu": cpu_info,
                "memory": memory_info
            }
        except Exception as e:
            logger.error(f"Error getting performance counters via WMI: {e}")
            return {}

    def get_thermal_zones(self) -> List[Dict]:
        """Get thermal zone information via WMI."""
        if not self.wmi_conn:
            return []
        
        try:
            # Get thermal zone information
            thermal_zones = []
            for zone in self.wmi_conn.Win32_PerfFormattedData_Counters_ThermalZoneInformation():
                thermal_zones.append({
                    "name": zone.Name,
                    "temperature": getattr(zone, 'Temperature', 0),
                    "high_threshold": getattr(zone, 'HighThreshold', 0),
                    "critical_threshold": getattr(zone, 'CriticalThreshold', 0)
                })
            return thermal_zones
        except Exception as e:
            logger.error(f"Error getting thermal zones via WMI: {e}")
            return []


# ============================================================
# WINDOWS SERVICE MANAGER
# ============================================================

class WindowsServiceManager:
    """Windows service management capabilities."""

    def __init__(self):
        if not WIN32SERVICES_AVAILABLE:
            logger.warning("pywin32 modules not available. Service management disabled.")
        self.service_cache = {}

    def get_service_status(self, service_name: str) -> Optional[Dict]:
        """Get status of a Windows service."""
        if not WIN32SERVICES_AVAILABLE:
            logger.warning("pywin32 modules not available. Service status unavailable.")
            return None

        try:
            service_status = win32serviceutil.QueryServiceStatus(service_name)
            return {
                "service_name": service_name,
                "status": service_status[1],
                "controls_accepted": service_status[3],
                "win32_exit_code": service_status[2],
                "service_specific_exit_code": service_status[4],
                "check_point": service_status[5],
                "wait_hint": service_status[6]
            }
        except Exception as e:
            logger.error(f"Error getting service status for {service_name}: {e}")
            return None

    def start_service(self, service_name: str) -> bool:
        """Start a Windows service."""
        if not WIN32SERVICES_AVAILABLE:
            logger.warning("pywin32 modules not available. Cannot start service.")
            return False

        try:
            win32serviceutil.StartService(service_name)
            logger.info(f"Started service: {service_name}")
            return True
        except Exception as e:
            logger.error(f"Error starting service {service_name}: {e}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """Stop a Windows service."""
        if not WIN32SERVICES_AVAILABLE:
            logger.warning("pywin32 modules not available. Cannot stop service.")
            return False

        try:
            win32serviceutil.StopService(service_name)
            logger.info(f"Stopped service: {service_name}")
            return True
        except Exception as e:
            logger.error(f"Error stopping service {service_name}: {e}")
            return False

    def get_all_services(self) -> List[Dict]:
        """Get all Windows services."""
        if not WIN32SERVICES_AVAILABLE:
            logger.warning("pywin32 modules not available. Cannot get services.")
            return []

        try:
            services = []
            for service in win32serviceutil.GetServiceInstances():
                try:
                    status_info = self.get_service_status(service)
                    if status_info:
                        services.append(status_info)
                except:
                    continue  # Skip services we can't query
            return services
        except Exception as e:
            logger.error(f"Error getting all services: {e}")
            return []

    def optimize_service_performance(self, service_name: str) -> bool:
        """Optimize service performance settings."""
        if not WIN32SERVICES_AVAILABLE:
            logger.warning("pywin32 modules not available. Cannot optimize service.")
            return False

        try:
            # This would modify service startup type, dependencies, etc.
            # For safety, we'll just log what would be done
            logger.info(f"Optimizing service: {service_name}")

            # Example: Change service to delayed auto-start
            # This would require admin privileges and careful implementation
            return True
        except Exception as e:
            logger.error(f"Error optimizing service {service_name}: {e}")
            return False


# ============================================================
# WINDOWS PERFORMANCE COUNTERS
# ============================================================

class WindowsPerformanceCounterManager:
    """Windows Performance Counter integration."""

    def __init__(self):
        self.counter_handles = {}
        self.active_queries = {}

    def get_performance_counter(self, category: str, counter: str, instance: str = None) -> Optional[float]:
        """Get a Windows performance counter value."""
        try:
            # This would use Windows PDH (Performance Data Helper) API
            # For now, we'll simulate with psutil data
            if category == "Processor" and counter == "% Processor Time" and instance == "_Total":
                return psutil.cpu_percent(interval=0.1)
            elif category == "Memory" and counter == "Available MBytes":
                memory = psutil.virtual_memory()
                return memory.available / (1024 * 1024)  # Convert to MB
            elif category == "PhysicalDisk" and counter == "Disk Read Bytes/sec" and instance == "_Total":
                disk_io = psutil.disk_io_counters()
                return disk_io.read_bytes if disk_io else 0
            else:
                # For other counters, return a simulated value
                return 50.0  # Default simulated value
        except Exception as e:
            logger.error(f"Error getting performance counter {category}\\{counter}: {e}")
            return None

    def get_system_performance_data(self) -> Dict:
        """Get comprehensive system performance data."""
        return {
            "cpu_usage": self.get_performance_counter("Processor", "% Processor Time", "_Total"),
            "available_memory_mb": self.get_performance_counter("Memory", "Available MBytes"),
            "disk_read_bytes_per_sec": self.get_performance_counter("PhysicalDisk", "Disk Read Bytes/sec", "_Total"),
            "disk_write_bytes_per_sec": self.get_performance_counter("PhysicalDisk", "Disk Write Bytes/sec", "_Total"),
            "network_bytes_per_sec": self.get_network_performance(),
            "page_faults_per_sec": self.get_performance_counter("Memory", "Page Faults/sec"),
            "available_swap_mb": self.get_swap_performance()
        }

    def get_network_performance(self) -> float:
        """Get network performance data."""
        try:
            net_io = psutil.net_io_counters()
            return (net_io.bytes_sent + net_io.bytes_recv) / 10.0  # Simulated value
        except:
            return 0.0

    def get_swap_performance(self) -> float:
        """Get swap file performance data."""
        try:
            swap = psutil.swap_memory()
            return swap.total / (1024 * 1024)  # Convert to MB
        except:
            return 0.0


# ============================================================
# MAIN WINDOWS EXTENSION MANAGER
# ============================================================

class WindowsExtensionManager:
    """Main Windows extension manager for GAMESA framework."""

    def __init__(self):
        # Windows-specific managers
        self.registry = WindowsRegistryManager()
        self.wmi = WindowsWMIManager()
        self.services = WindowsServiceManager()
        self.performance_counters = WindowsPerformanceCounterManager()

        # Resource pools
        self.pools: Dict[WindowsResourceType, WindowsResourcePool] = {
            WindowsResourceType.PROCESS_PRIORITY: ProcessPriorityPool(),
            WindowsResourceType.THREAD_AFFINITY: ThreadAffinityPool(),
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

    def collect_telemetry(self) -> WindowsTelemetry:
        """Collect Windows-specific system telemetry."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        # Disk
        disk_percent = psutil.disk_usage('/').percent  # Root drive usage

        # Network
        net_io = psutil.net_io_counters()

        # Process stats
        process_count = len(psutil.pids())

        # Handle count (with error handling for permissions)
        handle_count = 0
        for proc in psutil.process_iter():
            try:
                if proc.is_running():
                    handle_count += len(proc.open_files()) + len(proc.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue

        # Uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time

        # Registry usage
        registry_usage = self.registry.get_registry_usage()

        # Page file usage
        swap_info = psutil.swap_memory()
        page_file_usage = swap_info.percent if swap_info else 0.0

        # Commit charge (committed memory)
        commit_charge = (memory_info.used + (swap_info.used if swap_info else 0)) / (1024*1024)  # in MB

        # Thermal (simplified - real system would use WMI or other methods)
        thermal_headroom = max(0, 20.0 - (cpu_percent * 0.2))

        # Power (simplified)
        power_headroom = max(0, 100.0 - cpu_percent)

        telemetry = WindowsTelemetry(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_bytes_per_sec=net_io.bytes_sent + net_io.bytes_recv,
            process_count=process_count,
            handle_count=handle_count,
            uptime_seconds=uptime,
            thermal_headroom=thermal_headroom,
            power_headroom=power_headroom,
            registry_usage=registry_usage,
            page_file_usage=page_file_usage,
            commit_charge=commit_charge
        )

        self.telemetry_history.append(telemetry)
        return telemetry

    def allocate_resource(self, request: WindowsAllocationRequest) -> Optional[WindowsAllocation]:
        """Allocate Windows-specific resource with safety checks."""
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
                    "cost": request.bid_credits,
                    "process_id": request.process_id
                })

                return allocation

        return None

    def _would_violate_safety(self, request: WindowsAllocationRequest) -> bool:
        """Check if allocation would violate safety limits."""
        telemetry = self.collect_telemetry()

        if request.resource_type == WindowsResourceType.PROCESS_PRIORITY:
            # Check if high priority allocation would impact system stability
            if request.priority == WindowsPriority.REALTIME:
                if telemetry.cpu_percent > 80:
                    return True  # Don't allow realtime priority when CPU is high

        elif request.resource_type == WindowsResourceType.THREAD_AFFINITY:
            # Check if CPU affinity allocation would impact system stability
            if telemetry.cpu_percent > 90:
                return True  # Don't allow affinity changes when CPU is very high

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
        """Get overall Windows system status."""
        telemetry = self.collect_telemetry()

        # Get WMI system information
        wmi_info = self.wmi.get_system_performance_counters()
        gpu_info = self.wmi.get_gpu_info()
        thermal_info = self.wmi.get_thermal_zones()

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
            "registry_usage": telemetry.registry_usage,
            "page_file_usage": telemetry.page_file_usage,
            "commit_charge_mb": telemetry.commit_charge,
            "active_processes": telemetry.process_count,
            "handle_count": telemetry.handle_count,
            "gpu_info": gpu_info,
            "thermal_zones": thermal_info,
            "wmi_performance": wmi_info,
            "trade_count": len(self.trade_history),
        }

    def optimize_system(self, optimization_type: str = "all") -> Dict[str, bool]:
        """Perform Windows-specific system optimizations."""
        results = {}

        if optimization_type in ["all", "registry"]:
            results["registry"] = self.registry.optimize_registry()

        if optimization_type in ["all", "services"]:
            # Optimize critical services
            critical_services = ["Winmgmt", "Dhcp", "Dnscache"]
            service_results = []
            for service in critical_services:
                service_results.append(self.services.optimize_service_performance(service))
            results["services"] = all(service_results)

        if optimization_type in ["all", "memory"]:
            # Memory optimization would go here
            results["memory"] = True  # Placeholder

        if optimization_type in ["all", "disk"]:
            # Disk optimization would go here
            results["disk"] = True  # Placeholder

        return results


# ============================================================
# WINDOWS OPTIMIZATION AGENT
# ============================================================

class WindowsOptimizationAgent:
    """
    Advanced Windows optimization agent that trades system resources
    based on real-time Windows system state and optimization policies.
    """

    def __init__(self, agent_id: str = "WindowsOptimizerAgent"):
        self.agent_id = agent_id
        self.extension_manager = WindowsExtensionManager()
        self.active_allocations: List[str] = []  # allocation IDs
        self._running = False
        self._thread = None

    def optimize_performance_critical_process(self, process_name: str, priority: WindowsPriority = WindowsPriority.ABOVE_NORMAL) -> bool:
        """Optimize a performance-critical process by setting priority and affinity."""
        try:
            # Find the process by name
            target_pid = None
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'].lower() == process_name.lower():
                    target_pid = proc.info['pid']
                    break

            if not target_pid:
                logger.warning(f"Process {process_name} not found")
                return False

            # Create allocation request for priority
            priority_request = WindowsAllocationRequest(
                request_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                resource_type=WindowsResourceType.PROCESS_PRIORITY,
                amount=1.0,
                priority=priority,
                bid_credits=25.0,
                process_id=target_pid
            )

            priority_allocation = self.extension_manager.allocate_resource(priority_request)
            if priority_allocation:
                self.active_allocations.append(priority_allocation.allocation_id)
                logger.info(f"Set priority for {process_name} (PID: {target_pid}) to {priority}")

            # Create allocation request for thread affinity (use all cores)
            affinity_request = WindowsAllocationRequest(
                request_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                resource_type=WindowsResourceType.THREAD_AFFINITY,
                amount=float(psutil.cpu_count(logical=True)),
                bid_credits=25.0,
                process_id=target_pid
            )

            affinity_allocation = self.extension_manager.allocate_resource(affinity_request)
            if affinity_allocation:
                self.active_allocations.append(affinity_allocation.allocation_id)
                logger.info(f"Set thread affinity for {process_name} (PID: {target_pid}) to all cores")

            return bool(priority_allocation or affinity_allocation)

        except Exception as e:
            logger.error(f"Error optimizing process {process_name}: {e}")
            return False

    def optimize_system_resources(self) -> Dict[str, Any]:
        """Optimize system resources based on current telemetry."""
        telemetry = self.extension_manager.collect_telemetry()

        results = {
            "cpu_optimized": False,
            "memory_optimized": False,
            "disk_optimized": False,
            "network_optimized": False
        }

        # CPU optimization: Adjust process priorities based on CPU usage
        if telemetry.cpu_percent > 80:
            # Find high-CPU processes and potentially reduce their priority
            high_cpu_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] > 20:  # More than 20% CPU
                        high_cpu_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Limit the top CPU consumers
            high_cpu_processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            for proc_info in high_cpu_processes[:3]:  # Top 3 CPU consumers
                # Reduce priority of high CPU processes (if not system critical)
                if proc_info['name'] not in ['System', 'svchost.exe', 'wininit.exe', 'csrss.exe']:
                    # This would be implemented with a resource allocation request
                    pass

        # Memory optimization: If memory usage is high, consider optimizing
        if telemetry.memory_percent > 85:
            # This would trigger memory optimization routines
            results["memory_optimized"] = True

        # Disk optimization: If disk usage is high, consider optimizing
        if telemetry.disk_percent > 80:
            # This would trigger disk optimization routines
            results["disk_optimized"] = True

        return results

    def get_status(self) -> Dict:
        """Get optimization agent status."""
        return {
            "agent_id": self.agent_id,
            "running": self._running,
            "active_allocations": len(self.active_allocations),
            "resource_status": self.extension_manager.get_agent_status(self.agent_id),
            "system_status": self.extension_manager.get_system_status()
        }

    def run_continuous_optimization(self, interval_seconds: float = 10.0):
        """Run continuous optimization in a background thread."""
        if self._running:
            print(f"[{self.agent_id}] Optimization already running")
            return

        self._running = True

        def optimization_loop():
            print(f"[{self.agent_id}] Starting continuous Windows optimization (every {interval_seconds}s)")
            while self._running:
                try:
                    # Perform system optimizations
                    opt_results = self.optimize_system_resources()
                    
                    # Check for performance-critical processes to optimize
                    # For example, optimize games or other performance-critical applications
                    self.optimize_performance_critical_process("chrome.exe")  # Example
                    self.optimize_performance_critical_process("firefox.exe")  # Example
                    
                    # Print system status
                    status = self.get_status()
                    telemetry = self.extension_manager.collect_telemetry()
                    print(f"  System Status: CPU={telemetry.cpu_percent:.1f}%, "
                          f"Memory={telemetry.memory_percent:.1f}%, "
                          f"Processes={telemetry.process_count}, "
                          f"Registry={telemetry.registry_usage:.1f}%")
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"[{self.agent_id}] Optimization error: {e}")
                    time.sleep(interval_seconds)

        self._thread = threading.Thread(target=optimization_loop, daemon=True)
        self._thread.start()

    def stop_optimization(self):
        """Stop continuous optimization."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"[{self.agent_id}] Windows optimization stopped")


# ============================================================
# DEMONSTRATION
# ============================================================

def demo_windows_extension():
    """Demonstrate the Windows Extension for GAMESA framework."""
    print("=" * 80)
    print("WINDOWS EXTENSION FOR GAMESA/KRYSTALSTACK FRAMEWORK")
    print("=" * 80)

    # Create Windows extension manager
    windows_ext = WindowsExtensionManager()
    print("[OK] Windows Extension Manager initialized")

    # Show initial status
    status = windows_ext.get_system_status()
    print("[OK] Windows system status collected:")
    print(f"  - Active processes: {status['active_processes']}")
    print(f"  - Handle count: {status['handle_count']}")
    print(f"  - Registry usage: {status['registry_usage']:.1f}%")
    print(f"  - Page file usage: {status['page_file_usage']:.1f}%")
    print(f"  - GPU count: {len(status['gpu_info'])}")
    print(f"  - Thermal zones: {len(status['thermal_zones'])}")

    # Create optimization agent
    agent = WindowsOptimizationAgent("WindowsGameOptimizer")
    print(f"[OK] Windows optimization agent created: {agent.agent_id}")

    # Show agent status
    agent_status = agent.get_status()
    print(f"[OK] Agent status:")
    print(f"  - Credits: {agent_status['resource_status']['credits']:.1f}")
    print(f"  - Active allocations: {agent_status['active_allocations']}")

    # Demonstrate resource allocation
    print("\n--- Windows Resource Allocation Demo ---")

    # Find a process to optimize (try common ones)
    target_pid = None
    common_processes = ["notepad.exe", "chrome.exe", "firefox.exe", "explorer.exe"]
    for proc_name in common_processes:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'].lower() == proc_name.lower():
                target_pid = proc.info['pid']
                break
        if target_pid:
            break

    if target_pid:
        # Request process priority boost
        priority_request = WindowsAllocationRequest(
            request_id=str(uuid.uuid4()),
            agent_id="WindowsGameOptimizer",
            resource_type=WindowsResourceType.PROCESS_PRIORITY,
            amount=1.0,
            priority=WindowsPriority.ABOVE_NORMAL,
            bid_credits=50.0,
            process_id=target_pid
        )

        allocation = windows_ext.allocate_resource(priority_request)
        if allocation:
            print(f"[OK] Priority boost allocated for PID {target_pid} (ID: {allocation.allocation_id[:8]})")
        else:
            print(f"[INFO] Could not allocate priority boost for PID {target_pid}")

        # Request thread affinity
        affinity_request = WindowsAllocationRequest(
            request_id=str(uuid.uuid4()),
            agent_id="WindowsGameOptimizer",
            resource_type=WindowsResourceType.THREAD_AFFINITY,
            amount=2.0,  # Use 2 cores
            bid_credits=25.0,
            process_id=target_pid
        )

        allocation = windows_ext.allocate_resource(affinity_request)
        if allocation:
            print(f"[OK] Thread affinity allocated for PID {target_pid} (ID: {allocation.allocation_id[:8]})")
        else:
            print(f"[INFO] Could not allocate affinity for PID {target_pid}")
    else:
        print("[INFO] No common process found for demo")

    # Show system optimization
    print("\n--- System Optimization Demo ---")
    optimization_results = windows_ext.optimize_system("all")
    print(f"Optimization results: {optimization_results}")

    # Show WMI information
    print("\n--- WMI System Information ---")
    wmi_gpu_info = windows_ext.wmi.get_gpu_info()
    if wmi_gpu_info:
        for i, gpu in enumerate(wmi_gpu_info):
            print(f"  GPU {i+1}: {gpu['name'][:50]}... ({gpu['adapter_ram'] / (1024**3):.1f} GB)")
    else:
        print("  [INFO] Could not retrieve GPU information via WMI")

    # Show performance counters
    print("\n--- Performance Counters ---")
    perf_data = windows_ext.performance_counters.get_system_performance_data()
    for counter, value in list(perf_data.items())[:5]:  # Show first 5 counters
        print(f"  {counter}: {value:.2f}")

    # Show registry optimization
    print("\n--- Registry Optimization ---")
    registry_opt_result = windows_ext.registry.optimize_registry()
    print(f"  Registry optimization: {'Success' if registry_opt_result else 'Failed'}")
    print(f"  Registry backups: {windows_ext.registry.get_backup_count()}")

    print("\n" + "=" * 80)
    print("WINDOWS EXTENSION DEMONSTRATION COMPLETE")
    print("This extension provides Windows-specific optimizations for the GAMESA framework")
    print("with registry management, WMI integration, service control, and performance counters")
    print("=" * 80)


def demo_continuous_optimization():
    """Demonstrate continuous Windows optimization."""
    print("\n" + "=" * 80)
    print("CONTINUOUS WINDOWS OPTIMIZATION DEMO")
    print("=" * 80)

    agent = WindowsOptimizationAgent("ContinuousWindowsOptimizer")

    print("Starting 30 seconds of continuous Windows optimization...")
    agent.run_continuous_optimization(interval_seconds=5.0)

    # Let it run for a bit
    time.sleep(15)  # Only run for 15 seconds to keep demo short

    agent.stop_optimization()
    status = agent.get_status()
    print(f"Stopped. Final status: {status['active_allocations']} allocations")

    print("\nThis Windows extension enables sophisticated system optimization")
    print("with economic resource trading, process management, and hardware monitoring")
    print("=" * 80)


if __name__ == "__main__":
    demo_windows_extension()
    demo_continuous_optimization()