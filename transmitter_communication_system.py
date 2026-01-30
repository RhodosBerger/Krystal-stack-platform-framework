#!/usr/bin/env python3
"""
Transmitter Communication System with Logging and System Integration

This module implements a transmitter communication system with features for logging
from various channels, planning computing power changes, and system integration
with Guardian Framework components.
"""

import asyncio
import threading
import time
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import uuid
from collections import defaultdict, deque
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import platform
import subprocess
from pathlib import Path
import sys
import os
import copy
from functools import partial
import signal
import socket
import struct


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommunicationChannel(Enum):
    """Types of communication channels."""
    TCP_SOCKET = "tcp_socket"
    UDP_SOCKET = "udp_socket"
    LOCAL_QUEUE = "local_queue"
    SHARED_MEMORY = "shared_memory"
    MESSAGE_BROKER = "message_broker"
    PIPELINE_STREAM = "pipeline_stream"
    SYSTEM_BUS = "system_bus"


class LogSeverity(Enum):
    """Severity levels for logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComputingPowerLevel(Enum):
    """Levels of computing power."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"
    ADAPTIVE = "adaptive"


@dataclass
class LogEntry:
    """Entry in the communication log."""
    id: str
    timestamp: float
    channel: CommunicationChannel
    severity: LogSeverity
    message: str
    source: str
    metadata: Dict[str, Any]
    processed: bool = False
    processing_time: Optional[float] = None


@dataclass
class SystemDependency:
    """System dependency information."""
    name: str
    version: str
    status: str
    required: bool
    component: str
    path: Optional[str] = None


@dataclass
class ComputingPlan:
    """Plan for computing power allocation."""
    id: str
    name: str
    power_level: ComputingPowerLevel
    cpu_cores: int
    memory_mb: int
    gpu_enabled: bool
    priority: int
    creation_time: float
    dependencies: List[str]
    execution_time: Optional[float] = None


class TransmitterChannel:
    """Base class for communication channels."""
    
    def __init__(self, channel_type: CommunicationChannel, name: str):
        self.channel_type = channel_type
        self.name = name
        self.channel_id = f"CHAN_{name.upper()}_{uuid.uuid4().hex[:8].upper()}"
        self.is_active = False
        self.message_queue = queue.Queue()
        self.log_buffer = deque(maxlen=1000)
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'last_activity': time.time()
        }
        self.lock = threading.RLock()
    
    def send_message(self, message: Any, destination: str = None) -> bool:
        """Send a message through this channel."""
        try:
            with self.lock:
                # Process message based on channel type
                processed_message = self._process_message(message)
                
                # Add to log
                log_entry = LogEntry(
                    id=f"MSG_{uuid.uuid4().hex[:8].upper()}",
                    timestamp=time.time(),
                    channel=self.channel_type,
                    severity=LogSeverity.INFO,
                    message=f"Message sent: {str(processed_message)[:50]}...",
                    source=self.channel_id,
                    metadata={'destination': destination, 'size': len(str(processed_message))}
                )
                self.log_buffer.append(log_entry)
                
                # Update stats
                self.stats['messages_sent'] += 1
                self.stats['last_activity'] = time.time()
                
                return True
        except Exception as e:
            logger.error(f"Error sending message through {self.name}: {e}")
            self.stats['errors'] += 1
            return False
    
    def receive_message(self) -> Optional[Any]:
        """Receive a message from this channel."""
        try:
            with self.lock:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    
                    # Add to log
                    log_entry = LogEntry(
                        id=f"MSG_{uuid.uuid4().hex[:8].upper()}",
                        timestamp=time.time(),
                        channel=self.channel_type,
                        severity=LogSeverity.INFO,
                        message=f"Message received: {str(message)[:50]}...",
                        source=self.channel_id,
                        metadata={'size': len(str(message))}
                    )
                    self.log_buffer.append(log_entry)
                    
                    # Update stats
                    self.stats['messages_received'] += 1
                    self.stats['last_activity'] = time.time()
                    
                    return message
        except Exception as e:
            logger.error(f"Error receiving message from {self.name}: {e}")
            self.stats['errors'] += 1
        return None
    
    def _process_message(self, message: Any) -> Any:
        """Process message before sending."""
        # Default processing - in real implementation, this would depend on channel type
        return message


class TCPChannel(TransmitterChannel):
    """TCP socket communication channel."""
    
    def __init__(self, name: str, host: str = 'localhost', port: int = 8080):
        super().__init__(CommunicationChannel.TCP_SOCKET, name)
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
    
    def connect(self):
        """Connect the TCP socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.is_active = True
            logger.info(f"TCP channel {self.name} connected to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect TCP channel {self.name}: {e}")
            self.is_active = False
    
    def send_message(self, message: Any, destination: str = None) -> bool:
        """Send message via TCP."""
        if not self.is_active:
            self.connect()
        
        try:
            message_str = json.dumps(message) if not isinstance(message, str) else message
            message_bytes = message_str.encode('utf-8')
            
            # Send length first, then message
            length = struct.pack('!I', len(message_bytes))
            self.socket.sendall(length)
            self.socket.sendall(message_bytes)
            
            return super().send_message(message, destination)
        except Exception as e:
            logger.error(f"TCP send error: {e}")
            self.is_active = False
            return False


class LocalQueueChannel(TransmitterChannel):
    """Local queue communication channel."""
    
    def __init__(self, name: str):
        super().__init__(CommunicationChannel.LOCAL_QUEUE, name)
        self.queue = queue.Queue()
    
    def send_message(self, message: Any, destination: str = None) -> bool:
        """Send message via local queue."""
        try:
            self.queue.put(message)
            return super().send_message(message, destination)
        except Exception as e:
            logger.error(f"Local queue send error: {e}")
            return False
    
    def receive_message(self) -> Optional[Any]:
        """Receive message from local queue."""
        try:
            if not self.queue.empty():
                message = self.queue.get_nowait()
                # Log the received message
                log_entry = LogEntry(
                    id=f"MSG_{uuid.uuid4().hex[:8].upper()}",
                    timestamp=time.time(),
                    channel=self.channel_type,
                    severity=LogSeverity.INFO,
                    message=f"Message received: {str(message)[:50]}...",
                    source=self.channel_id,
                    metadata={'size': len(str(message))}
                )
                self.log_buffer.append(log_entry)
                
                self.stats['messages_received'] += 1
                self.stats['last_activity'] = time.time()
                
                return message
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Local queue receive error: {e}")
            self.stats['errors'] += 1
        return None


class TransmitterLogManager:
    """Manager for logging from various channels."""
    
    def __init__(self):
        self.channels: Dict[str, TransmitterChannel] = {}
        self.global_log_buffer = deque(maxlen=10000)
        self.severity_filters = set(LogSeverity)
        self.channel_filters = set(CommunicationChannel)
        self.log_stats = {
            'total_entries': 0,
            'by_severity': defaultdict(int),
            'by_channel': defaultdict(int),
            'processing_time_avg': 0.0
        }
        self.lock = threading.RLock()
    
    def add_channel(self, channel: TransmitterChannel):
        """Add a communication channel for logging."""
        self.channels[channel.channel_id] = channel
        logger.info(f"Added channel {channel.name} to log manager")
    
    def log_message(self, message: str, severity: LogSeverity, source: str, 
                   channel: CommunicationChannel, metadata: Dict[str, Any] = None) -> LogEntry:
        """Log a message from any channel."""
        if metadata is None:
            metadata = {}
        
        log_entry = LogEntry(
            id=f"LOG_{uuid.uuid4().hex[:8].upper()}",
            timestamp=time.time(),
            channel=channel,
            severity=severity,
            message=message,
            source=source,
            metadata=metadata
        )
        
        with self.lock:
            self.global_log_buffer.append(log_entry)
            self.log_stats['total_entries'] += 1
            self.log_stats['by_severity'][severity.value] += 1
            self.log_stats['by_channel'][channel.value] += 1
        
        return log_entry
    
    def get_logs_by_severity(self, severity: LogSeverity) -> List[LogEntry]:
        """Get logs filtered by severity."""
        with self.lock:
            return [entry for entry in self.global_log_buffer if entry.severity == severity]
    
    def get_logs_by_channel(self, channel: CommunicationChannel) -> List[LogEntry]:
        """Get logs filtered by channel."""
        with self.lock:
            return [entry for entry in self.global_log_buffer if entry.channel == channel]
    
    def get_recent_logs(self, count: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        with self.lock:
            return list(self.global_log_buffer)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get log statistics."""
        with self.lock:
            return copy.deepcopy(self.log_stats)


class SystemDependencyManager:
    """Manager for system dependencies and checks."""
    
    def __init__(self):
        self.dependencies: Dict[str, SystemDependency] = {}
        self.component_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.dependency_graph = defaultdict(list)
        self.check_results = {}
        self.lock = threading.RLock()
    
    def register_dependency(self, name: str, version: str, status: str, 
                          required: bool, component: str, path: str = None):
        """Register a system dependency."""
        with self.lock:
            dependency = SystemDependency(
                name=name,
                version=version,
                status=status,
                required=required,
                component=component,
                path=path
            )
            self.dependencies[name] = dependency
            self.component_dependencies[component].append(name)
            
            logger.info(f"Registered dependency: {name} for component {component}")
    
    def check_dependency(self, name: str) -> bool:
        """Check if a dependency is available and working."""
        with self.lock:
            if name not in self.dependencies:
                return False
            
            dep = self.dependencies[name]
            try:
                # Perform actual check based on dependency type
                if name.startswith('python-'):
                    # Check Python module
                    module_name = name.replace('python-', '')
                    __import__(module_name)
                elif name == 'openvino':
                    # Check OpenVINO installation
                    import openvino.runtime as ov
                elif name == 'numpy':
                    import numpy as np
                elif name == 'psutil':
                    import psutil
                elif name == 'threading':
                    import threading
                
                dep.status = 'available'
                self.check_results[name] = True
                return True
            except ImportError:
                dep.status = 'missing'
                self.check_results[name] = False
                return False
            except Exception as e:
                dep.status = f'error: {str(e)}'
                self.check_results[name] = False
                return False
    
    def check_all_dependencies(self) -> Dict[str, bool]:
        """Check all registered dependencies."""
        with self.lock:
            results = {}
            for name in self.dependencies:
                results[name] = self.check_dependency(name)
            return results
    
    def get_component_dependencies(self, component: str) -> List[SystemDependency]:
        """Get dependencies for a specific component."""
        with self.lock:
            dep_names = self.component_dependencies.get(component, [])
            return [self.dependencies[name] for name in dep_names if name in self.dependencies]
    
    def get_dependency_status(self) -> Dict[str, str]:
        """Get status of all dependencies."""
        with self.lock:
            return {name: dep.status for name, dep in self.dependencies.items()}


class ComputingPowerManager:
    """Manager for computing power allocation and planning."""
    
    def __init__(self):
        self.current_plan: Optional[ComputingPlan] = None
        self.planned_power: Dict[str, ComputingPlan] = {}
        self.power_history = deque(maxlen=100)
        self.system_monitoring = True
        self.monitoring_thread = None
        self.resource_usage = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_percent': 0.0,
            'disk_io': 0.0,
            'network_io': 0.0
        }
        self.lock = threading.RLock()
    
    def create_computing_plan(self, name: str, power_level: ComputingPowerLevel,
                            cpu_cores: int = None, memory_mb: int = None,
                            gpu_enabled: bool = False, priority: int = 1,
                            dependencies: List[str] = None) -> ComputingPlan:
        """Create a computing power allocation plan."""
        if dependencies is None:
            dependencies = []
        
        if cpu_cores is None:
            cpu_cores = mp.cpu_count()
        
        if memory_mb is None:
            memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))  # Total in MB
        
        plan = ComputingPlan(
            id=f"PLAN_{name.upper()}_{uuid.uuid4().hex[:8].upper()}",
            name=name,
            power_level=power_level,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu_enabled=gpu_enabled,
            priority=priority,
            creation_time=time.time(),
            dependencies=dependencies
        )
        
        with self.lock:
            self.planned_power[plan.id] = plan
            self.power_history.append(plan)
        
        logger.info(f"Created computing plan: {name} with power level {power_level.value}")
        return plan
    
    def activate_plan(self, plan_id: str) -> bool:
        """Activate a computing power plan."""
        with self.lock:
            if plan_id not in self.planned_power:
                logger.error(f"Plan {plan_id} not found")
                return False
            
            plan = self.planned_power[plan_id]
            self.current_plan = plan
            plan.execution_time = time.time()
            
            # Apply resource allocation based on plan
            self._apply_resource_allocation(plan)
            
            logger.info(f"Activated computing plan: {plan.name}")
            return True
    
    def _apply_resource_allocation(self, plan: ComputingPlan):
        """Apply resource allocation based on the plan."""
        # This would actually set system resource limits in a real implementation
        logger.info(f"Applying resource allocation for plan {plan.name}:")
        logger.info(f"  CPU Cores: {plan.cpu_cores}")
        logger.info(f"  Memory: {plan.memory_mb} MB")
        logger.info(f"  GPU: {'Enabled' if plan.gpu_enabled else 'Disabled'}")
        
        # In a real system, this would set actual resource limits
        # For example, using process priorities, CPU affinity, memory limits, etc.
    
    def get_optimal_plan(self, workload_requirements: Dict[str, Any]) -> Optional[ComputingPlan]:
        """Get the optimal computing plan for given workload requirements."""
        with self.lock:
            best_plan = None
            best_score = -1
            
            for plan in self.planned_power.values():
                score = self._calculate_plan_score(plan, workload_requirements)
                if score > best_score:
                    best_score = score
                    best_plan = plan
            
            return best_plan
    
    def _calculate_plan_score(self, plan: ComputingPlan, requirements: Dict[str, Any]) -> float:
        """Calculate a score for how well a plan fits the requirements."""
        score = 0.0
        
        # CPU requirement match
        if 'cpu_cores_required' in requirements:
            if plan.cpu_cores >= requirements['cpu_cores_required']:
                score += 2.0
            else:
                score += max(0, plan.cpu_cores / requirements['cpu_cores_required'])
        
        # Memory requirement match
        if 'memory_required_mb' in requirements:
            if plan.memory_mb >= requirements['memory_required_mb']:
                score += 2.0
            else:
                score += max(0, plan.memory_mb / requirements['memory_required_mb'])
        
        # GPU requirement
        if requirements.get('gpu_required', False) and plan.gpu_enabled:
            score += 3.0
        elif not requirements.get('gpu_required', False):
            score += 1.0  # Bonus for not requiring GPU if not needed
        
        # Priority bonus
        score += plan.priority * 0.1
        
        return score
    
    def start_monitoring(self):
        """Start system resource monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.system_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started computing power monitoring")
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        self.system_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Stopped computing power monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.system_monitoring:
            try:
                # Monitor system resources
                self.resource_usage['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                self.resource_usage['memory_percent'] = psutil.virtual_memory().percent
                
                # Simulate GPU monitoring (in real system, would use nvidia-ml-py or similar)
                self.resource_usage['gpu_percent'] = min(100, self.resource_usage['cpu_percent'] * 0.8)
                
                # Monitor I/O
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                
                time.sleep(1.0)  # Monitor every second
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)


class TransmitterCommunicationSystem:
    """Main transmitter communication system."""
    
    def __init__(self):
        self.channels: Dict[str, TransmitterChannel] = {}
        self.log_manager = TransmitterLogManager()
        self.dependency_manager = SystemDependencyManager()
        self.power_manager = ComputingPowerManager()
        self.system_id = f"TX_SYS_{uuid.uuid4().hex[:8].upper()}"
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.message_handlers = {}
        self.lock = threading.RLock()
    
    def add_channel(self, channel: TransmitterChannel):
        """Add a communication channel to the system."""
        self.channels[channel.channel_id] = channel
        self.log_manager.add_channel(channel)
        logger.info(f"Added channel {channel.name} to communication system")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def send_message(self, message: Any, channel_name: str, destination: str = None) -> bool:
        """Send a message through the specified channel."""
        with self.lock:
            for channel_id, channel in self.channels.items():
                if channel.name == channel_name or channel.channel_id == channel_name:
                    success = channel.send_message(message, destination)
                    if success:
                        self.log_manager.log_message(
                            f"Message sent through {channel.name}",
                            LogSeverity.INFO,
                            self.system_id,
                            channel.channel_type,
                            {'message_type': type(message).__name__, 'destination': destination}
                        )
                    return success
        
        logger.error(f"Channel {channel_name} not found")
        return False
    
    def process_incoming_messages(self):
        """Process incoming messages from all channels."""
        for channel_id, channel in self.channels.items():
            message = channel.receive_message()
            if message is not None:
                # Log the received message
                self.log_manager.log_message(
                    f"Message received from {channel.name}",
                    LogSeverity.INFO,
                    channel_id,
                    channel.channel_type,
                    {'message_type': type(message).__name__}
                )
                
                # Process with registered handlers
                message_type = type(message).__name__
                if message_type in self.message_handlers:
                    try:
                        self.executor.submit(self.message_handlers[message_type], message)
                    except Exception as e:
                        logger.error(f"Error processing message with handler: {e}")
    
    def register_system_dependencies(self):
        """Register system dependencies for all components."""
        # Guardian Framework dependencies
        self.dependency_manager.register_dependency(
            'python-numpy', '1.21.0', 'unknown', True, 'guardian_framework'
        )
        self.dependency_manager.register_dependency(
            'python-psutil', '5.8.0', 'unknown', True, 'guardian_framework'
        )
        self.dependency_manager.register_dependency(
            'python-threading', '1.0', 'unknown', True, 'guardian_framework'
        )
        
        # Grid Memory Controller dependencies
        self.dependency_manager.register_dependency(
            'python-numpy', '1.21.0', 'unknown', True, 'grid_memory_controller'
        )
        
        # Essential Encoder dependencies
        self.dependency_manager.register_dependency(
            'python-json', '1.0', 'unknown', True, 'essential_encoder'
        )
        self.dependency_manager.register_dependency(
            'python-struct', '1.0', 'unknown', True, 'essential_encoder'
        )
        
        # OpenVINO dependencies (if available)
        try:
            import openvino.runtime
            self.dependency_manager.register_dependency(
                'openvino', '2024.0', 'available', False, 'openvino_integration'
            )
        except ImportError:
            self.dependency_manager.register_dependency(
                'openvino', '2024.0', 'missing', False, 'openvino_integration'
            )
        
        logger.info("Registered system dependencies for all components")
    
    def initialize_system(self):
        """Initialize the complete communication system."""
        logger.info(f"Initializing transmitter communication system: {self.system_id}")
        
        # Create default channels
        tcp_channel = TCPChannel("tcp_default", port=8080)
        local_queue_channel = LocalQueueChannel("local_queue_default")
        
        self.add_channel(tcp_channel)
        self.add_channel(local_queue_channel)
        
        # Register system dependencies
        self.register_system_dependencies()
        
        # Create default computing plans
        self.power_manager.create_computing_plan(
            "low_power", ComputingPowerLevel.LOW,
            cpu_cores=max(1, mp.cpu_count() // 4),
            memory_mb=512,
            priority=1
        )
        
        self.power_manager.create_computing_plan(
            "medium_power", ComputingPowerLevel.MEDIUM,
            cpu_cores=max(2, mp.cpu_count() // 2),
            memory_mb=2048,
            priority=2
        )
        
        self.power_manager.create_computing_plan(
            "high_power", ComputingPowerLevel.HIGH,
            cpu_cores=mp.cpu_count(),
            memory_mb=4096,
            gpu_enabled=True,
            priority=3
        )
        
        # Start monitoring
        self.power_manager.start_monitoring()
        
        logger.info("Transmitter communication system initialized successfully")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self.lock:
            status = {
                'system_id': self.system_id,
                'is_running': self.is_running,
                'channel_count': len(self.channels),
                'registered_dependencies': len(self.dependency_manager.dependencies),
                'active_plan': self.power_manager.current_plan.name if self.power_manager.current_plan else None,
                'resource_usage': self.power_manager.resource_usage.copy(),
                'log_statistics': self.log_manager.get_statistics(),
                'timestamp': time.time()
            }
            return status
    
    def plan_computing_power_change(self, requirements: Dict[str, Any]) -> Optional[str]:
        """Plan a computing power change based on requirements."""
        optimal_plan = self.power_manager.get_optimal_plan(requirements)
        if optimal_plan:
            logger.info(f"Planned computing power change to: {optimal_plan.name}")
            return optimal_plan.id
        return None
    
    def execute_computing_power_change(self, plan_id: str) -> bool:
        """Execute a planned computing power change."""
        return self.power_manager.activate_plan(plan_id)
    
    def start_system(self):
        """Start the communication system."""
        self.is_running = True
        logger.info(f"Started transmitter communication system: {self.system_id}")
    
    def stop_system(self):
        """Stop the communication system."""
        self.is_running = False
        self.power_manager.stop_monitoring()
        self.executor.shutdown(wait=True)
        logger.info(f"Stopped transmitter communication system: {self.system_id}")


def demo_transmitter_communication_system():
    """Demonstrate the transmitter communication system."""
    print("=" * 80)
    print("TRANSMITTER COMMUNICATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create the transmitter communication system
    tx_system = TransmitterCommunicationSystem()
    tx_system.initialize_system()
    print(f"[OK] Created transmitter communication system: {tx_system.system_id}")
    
    # Show system status
    status = tx_system.get_system_status()
    print(f"\nSystem Status:")
    print(f"  System ID: {status['system_id']}")
    print(f"  Channels: {status['channel_count']}")
    print(f"  Dependencies: {status['registered_dependencies']}")
    print(f"  Active Plan: {status['active_plan']}")
    print(f"  CPU Usage: {status['resource_usage']['cpu_percent']:.1f}%")
    print(f"  Memory Usage: {status['resource_usage']['memory_percent']:.1f}%")
    
    # Check dependencies
    print(f"\n--- Dependency Check Demo ---")
    dependency_results = tx_system.dependency_manager.check_all_dependencies()
    available_deps = sum(1 for result in dependency_results.values() if result)
    total_deps = len(dependency_results)
    print(f"  Dependencies available: {available_deps}/{total_deps}")
    
    for dep_name, is_available in list(dependency_results.items())[:5]:  # Show first 5
        status = "[OK]" if is_available else "[FAIL]"
        print(f"    {status} {dep_name}: {'Available' if is_available else 'Missing'}")
    if len(dependency_results) > 5:
        print(f"    ... and {len(dependency_results) - 5} more")
    
    # Show available computing plans
    print(f"\n--- Computing Plans Demo ---")
    print(f"  Available plans:")
    for plan_id, plan in tx_system.power_manager.planned_power.items():
        print(f"    - {plan.name}: {plan.power_level.value} (CPU: {plan.cpu_cores}, Mem: {plan.memory_mb}MB)")
    
    # Plan a computing power change
    print(f"\n--- Computing Power Planning Demo ---")
    workload_reqs = {
        'cpu_cores_required': 4,
        'memory_required_mb': 2048,
        'gpu_required': True
    }
    planned_id = tx_system.plan_computing_power_change(workload_reqs)
    if planned_id:
        print(f"  Planned power change: {planned_id}")
        
        # Execute the planned change
        success = tx_system.execute_computing_power_change(planned_id)
        print(f"  Power change executed: {'SUCCESS' if success else 'FAILED'}")
    
    # Send some test messages
    print(f"\n--- Communication Demo ---")
    test_messages = [
        {"type": "system_status", "data": {"cpu": 50, "memory": 60}},
        {"type": "performance_alert", "data": {"metric": "latency", "value": 45}},
        {"type": "resource_request", "data": {"resource": "gpu", "priority": "high"}}
    ]
    
    for i, msg in enumerate(test_messages):
        success = tx_system.send_message(msg, "local_queue_default", f"destination_{i}")
        print(f"  Message {i+1} sent: {'SUCCESS' if success else 'FAILED'}")
    
    # Process incoming messages
    tx_system.process_incoming_messages()
    
    # Show recent logs
    print(f"\n--- Recent Logs Demo ---")
    recent_logs = tx_system.log_manager.get_recent_logs(3)
    for log in recent_logs:
        print(f"  [{log.severity.value}] {log.message[:50]}...")
    
    # Show log statistics
    log_stats = tx_system.log_manager.get_statistics()
    print(f"\nLog Statistics:")
    print(f"  Total entries: {log_stats['total_entries']}")
    print(f"  By severity: {dict(log_stats['by_severity'])}")
    
    # Show final status
    final_status = tx_system.get_system_status()
    print(f"\nFinal System Status:")
    print(f"  Active Plan: {final_status['active_plan']}")
    print(f"  Resource Usage: CPU={final_status['resource_usage']['cpu_percent']:.1f}%, Mem={final_status['resource_usage']['memory_percent']:.1f}%")
    
    print(f"\n" + "=" * 80)
    print("TRANSMITTER COMMUNICATION SYSTEM DEMONSTRATION COMPLETE")
    print("The system demonstrates:")
    print("- Multi-channel communication with logging")
    print("- System dependency management and checking")
    print("- Computing power planning and allocation")
    print("- Real-time resource monitoring")
    print("- Message handling and processing")
    print("- Integration with existing system components")
    print("=" * 80)


if __name__ == "__main__":
    demo_transmitter_communication_system()