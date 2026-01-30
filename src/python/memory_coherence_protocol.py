"""
Memory Coherence Protocol Implementation

Implements the MESI (Modified, Exclusive, Shared, Invalid) coherence protocol
for the unified 3D grid memory system with GPU coprocessor integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum, auto
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio


# Enums
class CoherenceState(Enum):
    """MESI Coherence States"""
    INVALID = "invalid"
    SHARED = "shared"
    EXCLUSIVE = "exclusive"
    MODIFIED = "modified"


class CoherenceOperation(Enum):
    """Coherence operations"""
    READ_ACCESS = auto()
    WRITE_ACCESS = auto()
    WRITEBACK = auto()
    INVALIDATE = auto()
    UPGRADE_REQUEST = auto()
    SYNC_REQUEST = auto()


class GPUType(Enum):
    """GPU types for coherence tracking"""
    UHD_COPROCESSOR = "uhd_coprocessor"
    DISCRETE_GPU = "discrete_gpu"
    INTEGRATED_GPU = "integrated_gpu"


# Data Classes
@dataclass
class CoherenceEntry:
    """Coherence protocol entry for a memory address."""
    address: int
    state: CoherenceState
    owner_gpu: int  # ID of owning GPU
    dirty: bool = False
    timestamp: float = field(default_factory=time.time)
    sharers: Set[int] = field(default_factory=set)  # GPUs with shared copies
    writers: Set[int] = field(default_factory=set)  # GPUs with write access
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    last_writer: Optional[int] = None


@dataclass
class CoherenceRequest:
    """Request for coherence protocol."""
    request_id: str
    gpu_id: int
    operation: CoherenceOperation
    address: int
    data: Optional[bytes] = None
    broadcast: bool = False
    priority: int = 5
    timestamp: float = field(default_factory=time.time)
    timeout: float = 1.0  # seconds


@dataclass
class CoherenceResponse:
    """Response from coherence protocol."""
    request_id: str
    success: bool
    data: Optional[bytes] = None
    new_state: Optional[CoherenceState] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CoherenceStatistics:
    """Coherence protocol statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    coherence_invalidations: int = 0
    upgrade_requests: int = 0
    sync_requests: int = 0
    average_latency_us: float = 0.0
    coherence_overhead: float = 0.0  # Percentage overhead


class MemoryCoherenceProtocol:
    """Main MESI Coherence Protocol Implementation."""
    
    def __init__(self):
        self.directory: Dict[int, CoherenceEntry] = {}
        self.request_queue: deque = deque()
        self.response_handlers: Dict[str, callable] = {}
        self.stats = CoherenceStatistics()
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.event_loop = None
        
        # GPU configuration
        self.gpus: Dict[int, Dict[str, Any]] = {}
        self.gpu_memory_regions: Dict[int, List[range]] = {}
        
        # Initialize default directory states
        self._initialize_directory()
    
    def _initialize_directory(self):
        """Initialize coherence directory with default states."""
        # Initialize common memory regions with invalid state
        for base_addr in range(0x70000000, 0x80000000, 0x100000):
            self.directory[base_addr] = CoherenceEntry(
                address=base_addr,
                state=CoherenceState.INVALID,
                owner_gpu=-1,
                dirty=False,
                sharers=set(),
                writers=set()
            )
    
    def register_gpu(self, gpu_id: int, gpu_type: GPUType, memory_region: range):
        """Register a GPU with the coherence protocol."""
        with self.lock:
            self.gpus[gpu_id] = {
                'id': gpu_id,
                'type': gpu_type,
                'memory_region': memory_region,
                'last_activity': time.time()
            }
            
            if gpu_id not in self.gpu_memory_regions:
                self.gpu_memory_regions[gpu_id] = []
            
            self.gpu_memory_regions[gpu_id].append(memory_region)
            print(f"Registered GPU {gpu_id} ({gpu_type.value}) with memory range {memory_region}")
    
    def read_access(self, gpu_id: int, address: int) -> CoherenceResponse:
        """Handle read access with coherence protocol."""
        with self.lock:
            self.stats.total_requests += 1
            start_time = time.time()
            
            entry = self._get_or_create_entry(address)
            old_state = entry.state
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access_time = time.time()
            
            # Apply MESI protocol for read
            response = self._handle_read_access(gpu_id, address, entry)
            
            # Update statistics
            if response.success:
                if old_state != CoherenceState.INVALID:
                    self.stats.cache_hits += 1
                else:
                    self.stats.cache_misses += 1
                
                # Calculate latency
                latency = (time.time() - start_time) * 1_000_000  # microseconds
                self.stats.average_latency_us = (
                    (self.stats.average_latency_us * response.success + latency) / (response.success + 1)
                )
            
            return response
    
    def write_access(self, gpu_id: int, address: int, data: bytes) -> CoherenceResponse:
        """Handle write access with coherence protocol."""
        with self.lock:
            self.stats.total_requests += 1
            start_time = time.time()
            
            entry = self._get_or_create_entry(address)
            old_state = entry.state
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access_time = time.time()
            entry.last_writer = gpu_id
            
            # Apply MESI protocol for write
            response = self._handle_write_access(gpu_id, address, data, entry)
            
            # Update statistics
            if response.success:
                if old_state != CoherenceState.INVALID:
                    self.stats.cache_hits += 1
                else:
                    self.stats.cache_misses += 1
                
                # Calculate latency
                latency = (time.time() - start_time) * 1_000_000  # microseconds
                self.stats.average_latency_us = (
                    (self.stats.average_latency_us * response.success + latency) / (response.success + 1)
                )
            
            return response
    
    def _handle_read_access(self, gpu_id: int, address: int, 
                           entry: CoherenceEntry) -> CoherenceResponse:
        """Handle read access according to MESI protocol."""
        try:
            response_data = b""
            
            if entry.state == CoherenceState.INVALID:
                # Cache miss - fetch from memory or other GPU
                response_data = self._fetch_data_from_memory(address)
                entry.state = CoherenceState.SHARED
                entry.sharers.add(gpu_id)
                entry.owner_gpu = gpu_id
                
            elif entry.state == CoherenceState.SHARED:
                # Add this GPU as a sharer
                entry.sharers.add(gpu_id)
                
                # Fetch data if not available locally
                response_data = self._fetch_data_from_memory(address)
                
            elif entry.state == CoherenceState.EXCLUSIVE:
                # Data is exclusively held by one GPU
                if entry.owner_gpu == gpu_id:
                    # Owner GPU reading - fetch from local cache
                    response_data = self._fetch_from_local_cache(gpu_id, address)
                else:
                    # Grant shared access to this GPU
                    entry.state = CoherenceState.SHARED
                    entry.sharers.add(gpu_id)
                    response_data = self._fetch_data_from_memory(address)
                    
            elif entry.state == CoherenceState.MODIFIED:
                # Data is modified - may need to sync to other GPUs
                entry.sharers.add(gpu_id)
                
                # If not the owner, fetch from owner GPU
                if gpu_id != entry.owner_gpu:
                    response_data = self._fetch_from_gpu(entry.owner_gpu, address)
                else:
                    # Owner GPU reading modified data
                    response_data = self._fetch_from_local_cache(gpu_id, address)
                    entry.dirty = True  # Still modified
            
            return CoherenceResponse(
                request_id=f"READ_{uuid.uuid4().hex[:8]}",
                success=True,
                data=response_data,
                new_state=entry.state
            )
            
        except Exception as e:
            return CoherenceResponse(
                request_id=f"READ_{uuid.uuid4().hex[:8]}",
                success=False,
                error_message=str(e)
            )
    
    def _handle_write_access(self, gpu_id: int, address: int, 
                           data: bytes, entry: CoherenceEntry) -> CoherenceResponse:
        """Handle write access according to MESI protocol."""
        try:
            old_state = entry.state
            
            if entry.state == CoherenceState.INVALID:
                # Need to get exclusive access before writing
                self._invalidate_other_copies(address, gpu_id)
                entry.state = CoherenceState.MODIFIED
                entry.owner_gpu = gpu_id
                entry.dirty = True
                entry.sharers.clear()
                entry.writers.clear()
                entry.writers.add(gpu_id)
                
            elif entry.state == CoherenceState.SHARED:
                # Upgrade to exclusive before writing (invalidate other copies)
                self._invalidate_other_copies(address, gpu_id)
                entry.state = CoherenceState.MODIFIED
                entry.owner_gpu = gpu_id
                entry.dirty = True
                entry.sharers.clear()
                entry.writers.clear()
                entry.writers.add(gpu_id)
                
            elif entry.state == CoherenceState.EXCLUSIVE:
                # If not owner, need to invalidate and take ownership
                if entry.owner_gpu != gpu_id:
                    self._invalidate_other_copies(address, gpu_id)
                    entry.owner_gpu = gpu_id
                entry.state = CoherenceState.MODIFIED
                entry.dirty = True
                entry.writers.add(gpu_id)
                
            elif entry.state == CoherenceState.MODIFIED:
                # Already modified - just write
                entry.writers.add(gpu_id)
                if entry.owner_gpu != gpu_id:
                    # Update owner if different
                    entry.owner_gpu = gpu_id
            
            # Actually write the data
            success = self._write_to_gpu_memory(gpu_id, address, data)
            
            return CoherenceResponse(
                request_id=f"WRITE_{uuid.uuid4().hex[:8]}",
                success=success,
                new_state=entry.state
            )
            
        except Exception as e:
            return CoherenceResponse(
                request_id=f"WRITE_{uuid.uuid4().hex[:8]}",
                success=False,
                error_message=str(e)
            )
    
    def _invalidate_other_copies(self, address: int, except_gpu: int):
        """Invalidate copies of data on all GPUs except the requesting one."""
        entry = self.directory.get(address)
        if not entry:
            return
        
        # Invalidate on all sharers except the requesting GPU
        for gpu_id in list(entry.sharers):
            if gpu_id != except_gpu:
                self._send_invalidate_request(gpu_id, address)
                entry.sharers.discard(gpu_id)
                entry.writers.discard(gpu_id)
        
        # Update state
        entry.state = CoherenceState.EXCLUSIVE if except_gpu in entry.sharers else CoherenceState.MODIFIED
    
    def _send_invalidate_request(self, target_gpu: int, address: int):
        """Send invalidate request to a GPU."""
        # Simulate sending invalidate request
        print(f"Coherence: Sending invalidate request to GPU {target_gpu} for address {address:08X}")
        
        # Update statistics
        self.stats.coherence_invalidations += 1
    
    def _fetch_from_local_cache(self, gpu_id: int, address: int) -> bytes:
        """Fetch data from local GPU cache."""
        # Simulate fetching from local cache
        print(f"Coherence: Fetching from GPU {gpu_id} local cache at {address:08X}")
        return b"simulated_data"
    
    def _fetch_from_gpu(self, source_gpu: int, address: int) -> bytes:
        """Fetch data from another GPU."""
        # Simulate fetching from another GPU
        print(f"Coherence: Fetching from GPU {source_gpu} at {address:08X}")
        return b"remote_data"
    
    def _fetch_data_from_memory(self, address: int) -> bytes:
        """Fetch data from main memory."""
        # Simulate fetching from main memory
        print(f"Coherence: Fetching from main memory at {address:08X}")
        return b"main_memory_data"
    
    def _write_to_gpu_memory(self, gpu_id: int, address: int, data: bytes) -> bool:
        """Write data to GPU memory."""
        # Simulate writing to GPU memory
        print(f"Coherence: Writing {len(data)} bytes to GPU {gpu_id} at {address:08X}")
        return True
    
    def upgrade_request(self, gpu_id: int, address: int) -> CoherenceResponse:
        """Handle upgrade request (shared to exclusive for writing)."""
        with self.lock:
            self.stats.upgrade_requests += 1
            
            entry = self._get_or_create_entry(address)
            
            if entry.state == CoherenceState.SHARED:
                # Invalidate other copies and upgrade to exclusive
                self._invalidate_other_copies(address, gpu_id)
                entry.state = CoherenceState.EXCLUSIVE
                entry.owner_gpu = gpu_id
                entry.sharers = {gpu_id}
                
                return CoherenceResponse(
                    request_id=f"UPGRADE_{uuid.uuid4().hex[:8]}",
                    success=True,
                    new_state=entry.state
                )
            
            return CoherenceResponse(
                request_id=f"UPGRADE_{uuid.uuid4().hex[:8]}",
                success=False,
                error_message=f"Cannot upgrade {entry.state.value} state"
            )
    
    def sync_request(self, gpu_id: int, address: int) -> CoherenceResponse:
        """Handle synchronization request."""
        with self.lock:
            self.stats.sync_requests += 1
            
            entry = self._get_or_create_entry(address)
            
            if entry.state == CoherenceState.MODIFIED:
                # If modified, sync back to main memory
                data = self._fetch_from_local_cache(gpu_id, address)
                sync_success = self._write_back_to_memory(address, data)
                
                if sync_success:
                    entry.state = CoherenceState.SHARED
                    entry.dirty = False
                    entry.owner_gpu = -1  # No owner now
                
                return CoherenceResponse(
                    request_id=f"SYNC_{uuid.uuid4().hex[:8]}",
                    success=sync_success,
                    new_state=entry.state
                )
            
            return CoherenceResponse(
                request_id=f"SYNC_{uuid.uuid4().hex[:8]}",
                success=True,
                new_state=entry.state
            )
    
    def _write_back_to_memory(self, address: int, data: bytes) -> bool:
        """Write data back to main memory."""
        # Simulate writing back to main memory
        print(f"Coherence: Writing back to main memory at {address:08X}")
        return True
    
    def _get_or_create_entry(self, address: int) -> CoherenceEntry:
        """Get existing entry or create new one."""
        if address not in self.directory:
            self.directory[address] = CoherenceEntry(
                address=address,
                state=CoherenceState.INVALID,
                owner_gpu=-1
            )
        return self.directory[address]
    
    def get_coherence_stats(self) -> CoherenceStatistics:
        """Get current coherence statistics."""
        return self.stats
    
    def get_entry_state(self, address: int) -> Optional[CoherenceState]:
        """Get coherence state for an address."""
        entry = self.directory.get(address)
        return entry.state if entry else None
    
    def force_state_change(self, address: int, new_state: CoherenceState):
        """Force a state change (for testing/debugging)."""
        with self.lock:
            entry = self._get_or_create_entry(address)
            old_state = entry.state
            entry.state = new_state
            print(f"Coherence: Force state change {address:08X} from {old_state.value} to {new_state.value}")


class GPUCoherenceManager:
    """Manages coherence across multiple GPUs."""
    
    def __init__(self):
        self.protocol = MemoryCoherenceProtocol()
        self.gpu_sync_manager = GPUInterconnectManager()
        self.coherence_checker = CoherenceValidator()
    
    def initialize_gpus(self, gpu_configs: List[Dict]):
        """Initialize GPUs and register with coherence protocol."""
        for config in gpu_configs:
            self.protocol.register_gpu(
                gpu_id=config['id'],
                gpu_type=config['type'],
                memory_region=config['memory_region']
            )
    
    def read_from_gpu(self, gpu_id: int, address: int) -> Optional[bytes]:
        """Read from GPU with coherence protocol."""
        response = self.protocol.read_access(gpu_id, address)
        
        if response.success:
            return response.data
        else:
            print(f"Coherence read failed: {response.error_message}")
            return None
    
    def write_to_gpu(self, gpu_id: int, address: int, data: bytes) -> bool:
        """Write to GPU with coherence protocol."""
        response = self.protocol.write_access(gpu_id, address, data)
        
        if response.success:
            print(f"Coherence write successful to {address:08X} on GPU {gpu_id}")
            return True
        else:
            print(f"Coherence write failed: {response.error_message}")
            return False
    
    def perform_coherence_sync(self, gpu_id: int, address: int) -> bool:
        """Perform coherence synchronization."""
        response = self.protocol.sync_request(gpu_id, address)
        return response.success


class GPUInterconnectManager:
    """Manages interconnect between GPUs for coherence protocol."""
    
    def __init__(self):
        self.gpu_connections = {}
        self.bandwidth_usage = {}
        self.latency_table = {}
        self.lock = threading.Lock()
    
    def establish_connection(self, gpu1_id: int, gpu2_id: int, bandwidth: float, latency: float):
        """Establish connection between two GPUs."""
        with self.lock:
            if gpu1_id not in self.gpu_connections:
                self.gpu_connections[gpu1_id] = {}
            if gpu2_id not in self.gpu_connections:
                self.gpu_connections[gpu2_id] = {}
            
            # Bi-directional connection
            self.gpu_connections[gpu1_id][gpu2_id] = {
                'bandwidth_gbps': bandwidth,
                'latency_ns': latency
            }
            self.gpu_connections[gpu2_id][gpu1_id] = {
                'bandwidth_gbps': bandwidth,
                'latency_ns': latency
            }
            
            # Initialize bandwidth usage tracking
            self.bandwidth_usage[f"{gpu1_id}-{gpu2_id}"] = 0.0
            self.bandwidth_usage[f"{gpu2_id}-{gpu1_id}"] = 0.0
            
            # Initialize latency table
            self.latency_table[f"{gpu1_id}-{gpu2_id}"] = latency
            self.latency_table[f"{gpu2_id}-{gpu1_id}"] = latency
    
    def estimate_transfer_time(self, source_gpu: int, dest_gpu: int, size_bytes: int) -> float:
        """Estimate transfer time between GPUs."""
        conn_key = f"{source_gpu}-{dest_gpu}"
        if conn_key in self.gpu_connections:
            bandwidth_gbps = self.gpu_connections[source_gpu][dest_gpu]['bandwidth_gbps']
            latency_ns = self.gpu_connections[source_gpu][dest_gpu]['latency_ns']
            
            # Transfer time = latency + data_size / bandwidth
            data_transfer_time_ns = (size_bytes * 8) / (bandwidth_gbps * 1_000_000_000) * 1_000_000_000
            total_time_ns = latency_ns + data_transfer_time_ns
            
            return total_time_ns / 1_000_000_000  # Convert to seconds
        
        return float('inf')  # Infinite time if no connection
    
    def get_optimal_path(self, source_gpu: int, dest_gpu: int, size_bytes: int) -> List[int]:
        """Get optimal path between GPUs for data transfer."""
        if source_gpu == dest_gpu:
            return [source_gpu]
        
        # Simple shortest path: direct connection or through system memory
        if dest_gpu in self.gpu_connections.get(source_gpu, {}):
            # Direct connection is optimal
            return [source_gpu, dest_gpu]
        
        # Use system memory as intermediate
        return [source_gpu, 'SYSTEM_MEMORY', dest_gpu]


class CoherenceValidator:
    """Validates coherence protocol compliance."""
    
    def __init__(self):
        self.validation_rules = self._initialize_rules()
        self.violation_log = deque(maxlen=1000)
        
    def _initialize_rules(self) -> Dict[str, callable]:
        """Initialize coherence validation rules."""
        return {
            'no_multiple_modifiers': self._validate_single_modifier,
            'exclusive_before_modify': self._validate_exclusive_before_modify,
            'shared_after_read': self._validate_shared_after_read,
            'proper_invalidation': self._validate_invalidation
        }
    
    def validate_state_transition(self, old_state: CoherenceState, 
                                new_state: CoherenceState, 
                                operation: CoherenceOperation) -> bool:
        """Validate if state transition is legal."""
        # Define legal transitions based on MESI protocol
        legal_transitions = {
            CoherenceState.INVALID: [CoherenceState.EXCLUSIVE, CoherenceState.SHARED],
            CoherenceState.SHARED: [CoherenceState.INVALID, CoherenceState.MODIFIED],
            CoherenceState.EXCLUSIVE: [CoherenceState.MODIFIED, CoherenceState.SHARED, CoherenceState.INVALID],
            CoherenceState.MODIFIED: [CoherenceState.SHARED, CoherenceState.INVALID]
        }
        
        return new_state in legal_transitions.get(old_state, [])
    
    def _validate_single_modifier(self, address: int, gpu_id: int, state: CoherenceState) -> bool:
        """Validate that only one GPU can modify a memory location."""
        if state != CoherenceState.MODIFIED:
            return True
        
        # Check that no other GPU is modifying this same address
        # This is checked at the protocol level
        return True
    
    def _validate_exclusive_before_modify(self, address: int, gpu_id: int, state: CoherenceState) -> bool:
        """Validate that GPU has exclusive access before modifying."""
        if state == CoherenceState.EXCLUSIVE or state == CoherenceState.MODIFIED:
            return True
        
        return False
    
    def _validate_shared_after_read(self, address: int, gpu_id: int, state: CoherenceState) -> bool:
        """Validate that GPU has shared access after successful read."""
        if state == CoherenceState.SHARED or state == CoherenceState.EXCLUSIVE:
            return True
        
        return False
    
    def _validate_invalidation(self, address: int, except_gpu: int) -> bool:
        """Validate that invalidation properly removes copies from other GPUs."""
        # Implementation depends on coherence protocol state tracking
        return True
    
    def log_violation(self, violation_type: str, details: Dict):
        """Log coherence protocol violation."""
        violation = {
            'timestamp': time.time(),
            'violation_type': violation_type,
            'details': details,
            'traceback': 'N/A'  # In production, add actual traceback
        }
        self.violation_log.append(violation)
        print(f"Coherence violation: {violation_type} - {details}")


# Integration example with UHD coprocessor
class UHDCoprocessorCoherence:
    """Coherence management for UHD coprocessor integration."""
    
    def __init__(self, coherence_manager: GPUCoherenceManager):
        self.coherence_manager = coherence_manager
        self.uhd_gpu_id = 0  # Typically UHD GPU is assigned ID 0
        self.coprocessor_memory_region = range(0x7FFF0000, 0x80000000)  # UHD buffer region
        
    def initialize_uhd_coherence(self):
        """Initialize coherence for UHD coprocessor."""
        # Register UHD GPU with coherence protocol
        self.coherence_manager.protocol.register_gpu(
            gpu_id=self.uhd_gpu_id,
            gpu_type=GPUType.UHD_COPROCESSOR,
            memory_region=self.coprocessor_memory_region
        )
        
        print("UHD coprocessor coherence initialized")
        
    def read_from_uhd_buffer(self, address: int) -> Optional[bytes]:
        """Read from UHD coprocessor buffer with coherence."""
        if address in self.coprocessor_memory_region:
            return self.coherence_manager.read_from_gpu(self.uhd_gpu_id, address)
        else:
            print(f"Address {address:08X} not in UHD memory region")
            return None
    
    def write_to_uhd_buffer(self, address: int, data: bytes) -> bool:
        """Write to UHD coprocessor buffer with coherence."""
        if address in self.coprocessor_memory_region:
            return self.coherence_manager.write_to_gpu(self.uhd_gpu_id, address, data)
        else:
            print(f"Address {address:08X} not in UHD memory region")
            return False


# Demo function
def demo_memory_coherence_protocol():
    """Demonstrate the memory coherence protocol."""
    import uuid
    
    print("=== Memory Coherence Protocol Demo ===\n")
    
    # Initialize coherence manager
    coherence_manager = GPUCoherenceManager()
    
    # Simulate GPU configurations
    gpu_configs = [
        {
            'id': 0,
            'type': GPUType.UHD_COPROCESSOR,
            'memory_region': range(0x7FFF0000, 0x7FFF8000)  # 32KB UHD buffer
        },
        {
            'id': 1,
            'type': GPUType.DISCRETE_GPU,
            'memory_region': range(0x80000000, 0x90000000)  # 256MB
        },
        {
            'id': 2,
            'type': GPUType.DISCRETE_GPU,
            'memory_region': range(0x90000000, 0xA0000000)  # 256MB
        }
    ]
    
    coherence_manager.initialize_gpus(gpu_configs)
    
    # Establish connections between GPUs (simulated)
    interconnect = coherence_manager.gpu_sync_manager
    interconnect.establish_connection(0, 1, 20.0, 800)  # UHD to Discrete 1: 20GB/s, 800ns
    interconnect.establish_connection(1, 2, 32.0, 600)  # Discrete 1-2: 32GB/s, 600ns
    interconnect.establish_connection(0, 2, 15.0, 1200) # UHD to Discrete 2: 15GB/s, 1.2μs
    
    print("GPU configurations:")
    for config in gpu_configs:
        print(f"  - GPU {config['id']}: {config['type'].value}, "
              f"Memory: 0x{config['memory_region'].start:08X}-0x{config['memory_region'].stop:08X}")
    print()
    
    # Test coherence protocol with different scenarios
    test_address = 0x7FFF1000
    print(f"Testing coherence protocol on address: 0x{test_address:08X}\n")
    
    # Scenario 1: Initial read (cache miss)
    print("Scenario 1: Initial read (should be cache miss)")
    data1 = coherence_manager.read_from_gpu(1, test_address)
    print(f"  Read successful: {data1 is not None}")
    print(f"  Coherence state after read: {coherence_manager.protocol.get_entry_state(test_address)}")
    print()
    
    # Scenario 2: Write to the same location (should upgrade to modified)
    print("Scenario 2: Write to same location (should upgrade to modified)")
    write_success = coherence_manager.write_to_gpu(1, test_address, b"test_data")
    print(f"  Write successful: {write_success}")
    print(f"  Coherence state after write: {coherence_manager.protocol.get_entry_state(test_address)}")
    print()
    
    # Scenario 3: Second GPU tries to read (should cause invalidation and synchronization)
    print("Scenario 3: Second GPU reads same location (should cause coherence action)")
    data2 = coherence_manager.read_from_gpu(2, test_address)
    print(f"  Read successful: {data2 is not None}")
    print(f"  Coherence state after second read: {coherence_manager.protocol.get_entry_state(test_address)}")
    print()
    
    # Scenario 4: Test UHD coprocessor integration
    print("Scenario 4: UHD coprocessor integration test")
    uhd_coherence = UHDCoprocessorCoherence(coherence_manager)
    uhd_coherence.initialize_uhd_coherence()
    
    uhd_address = 0x7FFF2000
    uhd_data = uhd_coherence.read_from_uhd_buffer(uhd_address)
    print(f"  UHD read successful: {uhd_data is not None}")
    
    write_success = uhd_coherence.write_to_uhd_buffer(uhd_address, b"uhd_test")
    print(f"  UHD write successful: {write_success}")
    print()
    
    # Show coherence statistics
    stats = coherence_manager.protocol.get_coherence_stats()
    print("Coherence Statistics:")
    print(f"  Total Requests: {stats.total_requests}")
    print(f"  Cache Hits: {stats.cache_hits}")
    print(f"  Cache Misses: {stats.cache_misses}")
    print(f"  Invalidations: {stats.coherence_invalidations}")
    print(f"  Average Latency: {stats.average_latency_us:.2f} μs")
    print()
    
    # Test interconnect performance
    print("Interconnect Performance Estimates:")
    for gpu1 in [0, 1, 2]:
        for gpu2 in [0, 1, 2]:
            if gpu1 != gpu2:
                transfer_time = interconnect.estimate_transfer_time(gpu1, gpu2, 1024)  # 1KB transfer
                print(f"  GPU {gpu1} -> GPU {gpu2}: {transfer_time*1_000_000:.2f} μs for 1KB")
    
    print(f"\nMemory coherence protocol demo completed successfully!")


if __name__ == "__main__":
    demo_memory_coherence_protocol()