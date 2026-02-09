"""
Holocube Storage Bridge - Memory Management Layer for FANUC RISE v2.1
Implements scalable eviction policy managing data lifecycle between RAM (Hex Grid), 
Kafka streaming engine, and MongoDB archival storage based on Hexadecimal Topology theory.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class HexCellStatus(Enum):
    HOT = "hot"  # Frequently accessed data
    WARM = "warm"  # Moderately accessed data
    COLD = "cold"  # Rarely accessed data, eligible for archival
    EVIDENCE = "evidence"  # Immutable forensic data


@dataclass
class HexCell:
    """Represents a single hexagonal cell in the 3D grid memory topology"""
    sector_id: str
    cell_id: str
    data: Any
    access_frequency: int = 0
    last_access: datetime = field(default_factory=datetime.utcnow)
    creation_time: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    status: HexCellStatus = HexCellStatus.HOT
    priority_score: float = 1.0  # Calculated based on access patterns and importance


@dataclass
class MemorySector:
    """Represents a sector in the 3D grid memory topology"""
    sector_id: str
    cells: Dict[str, HexCell] = field(default_factory=dict)
    density: float = 0.0  # Current memory density (0.0 to 1.0)
    max_capacity: int = 1000  # Max number of cells in sector
    current_size: int = 0  # Current size in bytes


class MockKafkaProducer:
    """Mock Kafka producer for when real Kafka is not available"""
    def __init__(self, **kwargs):
        pass
    
    def send(self, topic, value):
        logger.debug(f"Kafka mock send to {topic}: {value}")
        return None
    
    def flush(self):
        pass


class MockMongoClient:
    """Mock MongoDB client for when real MongoDB is not available"""
    def __init__(self, *args, **kwargs):
        self.collections = {}
    
    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection()
        return self.collections[name]
    
    def close(self):
        """Close method for MongoDB client compatibility"""
        pass  # Nothing to close in mock implementation


class MockCollection:
    """Mock MongoDB collection for when real MongoDB is not available"""
    def __init__(self):
        self.documents = []
    
    def insert_one(self, doc):
        # Create a mock inserted_id
        doc['_id'] = str(uuid.uuid4())
        self.documents.append(doc)
        return MockInsertResult(doc['_id'])
    
    def find_one(self, query):
        for doc in self.documents:
            # Simple mock matching - just check if keys match
            if all(k in doc and doc[k] == v for k, v in query.items()):
                return doc
        return None


class MockInsertResult:
    """Mock insert result for when real MongoDB is not available"""
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class StorageBridge:
    """
    Holocube Storage Bridge - Manages data lifecycle between RAM (Hex Grid), 
    Kafka streaming engine, and MongoDB archival storage
    """
    
    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092", 
                 mongodb_connection_string: str = "mongodb://localhost:27017/",
                 mongodb_database: str = "fanuc_rise_storage",
                 evidence_directory: str = "root/evidence_locker",
                 vault_directory: str = "root/live_ops"):
        """
        Initialize the Holocube Storage Bridge with Kafka and MongoDB connectors
        
        Args:
            kafka_bootstrap_servers: Kafka cluster connection string
            mongodb_connection_string: MongoDB connection string
            mongodb_database: Name of the database for storage
            evidence_directory: Path for immutable forensic logs
            vault_directory: Path for active operational data
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.mongodb_connection_string = mongodb_connection_string
        self.mongodb_database = mongodb_database
        self.evidence_directory = evidence_directory
        self.vault_directory = vault_directory
        
        # Initialize Kafka producer for memory events
        try:
            from kafka import KafkaProducer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks=1
            )
        except ImportError:
            logger.warning("Kafka library not available, using mock implementation")
            self.kafka_producer = MockKafkaProducer()
        
        # Initialize MongoDB client for scalable swap
        try:
            from pymongo import MongoClient
            self.mongo_client = MongoClient(mongodb_connection_string)
            self.storage_db = self.mongo_client[mongodb_database]
            self.swap_collection = self.storage_db['hex_grid_swap']
            self.evidence_collection = self.storage_db['evidence_logs']
        except ImportError:
            logger.warning("PyMongo library not available, using mock implementation")
            self.mongo_client = MockMongoClient()
            self.storage_db = self.mongo_client[mongodb_database]
            self.swap_collection = self.storage_db['hex_grid_swap']
            self.evidence_collection = self.storage_db['evidence_logs']
        
        # Initialize memory sectors (RAM representation)
        self.memory_sectors: Dict[str, MemorySector] = {}
        self.hex_topology_density_threshold = 0.80  # 80% threshold for offloading
        
        logger.info("Holocube Storage Bridge initialized with Kafka and MongoDB integration")
    
    def register_memory_sector(self, sector_id: str, max_capacity: int = 1000) -> MemorySector:
        """Register a new memory sector in the 3D grid topology"""
        sector = MemorySector(
            sector_id=sector_id,
            max_capacity=max_capacity
        )
        self.memory_sectors[sector_id] = sector
        logger.info(f"Registered memory sector: {sector_id} with capacity: {max_capacity}")
        return sector
    
    def allocate_hex_cell(self, sector_id: str, data: Any) -> str:
        """
        Allocate a new hex cell in the specified sector (simulates malloc)
        
        Args:
            sector_id: ID of the memory sector
            data: Data to store in the hex cell
            
        Returns:
            Cell ID of the allocated hex cell
        """
        if sector_id not in self.memory_sectors:
            raise ValueError(f"Sector {sector_id} not registered")
        
        sector = self.memory_sectors[sector_id]
        
        # Create new hex cell
        cell_id = f"CELL_{uuid.uuid4().hex[:8].upper()}"
        size_bytes = len(json.dumps(data).encode('utf-8')) if isinstance(data, (dict, list)) else len(str(data).encode('utf-8'))
        
        hex_cell = HexCell(
            sector_id=sector_id,
            cell_id=cell_id,
            data=data,
            size_bytes=size_bytes,
            status=HexCellStatus.HOT
        )
        
        # Add to sector
        sector.cells[cell_id] = hex_cell
        sector.current_size += size_bytes
        sector.density = sector.current_size / (sector.max_capacity * 1000)  # Assuming avg 1KB per cell
        
        # Publish memory allocation event to Kafka
        event_data = {
            'event_type': 'malloc',
            'cell_id': cell_id,
            'sector_id': sector_id,
            'size_bytes': size_bytes,
            'timestamp': datetime.utcnow().isoformat(),
            'density_after_allocation': sector.density
        }
        self.stream_event_to_kafka(event_data)
        
        # Check if sector density exceeds threshold and needs offloading
        if sector.density > self.hex_topology_density_threshold:
            logger.warning(f"Sector {sector_id} density ({sector.density:.2%}) exceeds threshold, initiating offload")
            asyncio.create_task(self._evict_cold_cells(sector_id))
        
        logger.debug(f"Allocated hex cell {cell_id} in sector {sector_id}")
        return cell_id
    
    def deallocate_hex_cell(self, sector_id: str, cell_id: str) -> bool:
        """
        Deallocate a hex cell in the specified sector (simulates free)
        
        Args:
            sector_id: ID of the memory sector
            cell_id: ID of the hex cell to deallocate
            
        Returns:
            True if successfully deallocated, False otherwise
        """
        if sector_id not in self.memory_sectors:
            logger.error(f"Sector {sector_id} not found")
            return False
        
        sector = self.memory_sectors[sector_id]
        if cell_id not in sector.cells:
            logger.error(f"Cell {cell_id} not found in sector {sector_id}")
            return False
        
        hex_cell = sector.cells[cell_id]
        sector.current_size -= hex_cell.size_bytes
        sector.density = sector.current_size / (sector.max_capacity * 1000)
        
        # Publish memory deallocation event to Kafka
        event_data = {
            'event_type': 'free',
            'cell_id': cell_id,
            'sector_id': sector_id,
            'size_bytes': hex_cell.size_bytes,
            'timestamp': datetime.utcnow().isoformat(),
            'density_after_deallocation': sector.density
        }
        self.stream_event_to_kafka(event_data)
        
        # Remove from memory
        del sector.cells[cell_id]
        
        logger.debug(f"Deallocated hex cell {cell_id} from sector {sector_id}")
        return True
    
    def access_hex_cell(self, sector_id: str, cell_id: str) -> Optional[Any]:
        """
        Access a hex cell and update its access statistics
        
        Args:
            sector_id: ID of the memory sector
            cell_id: ID of the hex cell to access
            
        Returns:
            Data stored in the hex cell, or None if not found
        """
        if sector_id not in self.memory_sectors:
            logger.error(f"Sector {sector_id} not found")
            return None
        
        sector = self.memory_sectors[sector_id]
        if cell_id not in sector.cells:
            logger.error(f"Cell {cell_id} not found in sector {sector_id}")
            return None
        
        hex_cell = sector.cells[cell_id]
        hex_cell.access_frequency += 1
        hex_cell.last_access = datetime.utcnow()
        
        # Update priority score based on access frequency and recency
        time_decay = max(0.1, 1.0 - ((datetime.utcnow() - hex_cell.creation_time).total_seconds() / 86400 / 30))  # Decay over 30 days
        hex_cell.priority_score = (hex_cell.access_frequency * 0.7) + (time_decay * 0.3)
        
        # Update status based on access patterns
        if hex_cell.access_frequency > 100:
            hex_cell.status = HexCellStatus.HOT
        elif hex_cell.access_frequency > 10:
            hex_cell.status = HexCellStatus.WARM
        else:
            hex_cell.status = HexCellStatus.COLD
        
        logger.debug(f"Accessed hex cell {cell_id} in sector {sector_id}")
        return hex_cell.data
    
    def stream_event_to_kafka(self, event_data: Dict[str, Any]):
        """
        Publish a memory allocation/deallocation event to Kafka for real-time analysis
        
        Args:
            event_data: Dictionary containing the event information
        """
        try:
            if hasattr(self.kafka_producer, 'send'):
                self.kafka_producer.send('memory.events', value=event_data)
                self.kafka_producer.flush()
            logger.debug(f"Streamed event to Kafka: {event_data['event_type']} in sector {event_data['sector_id']}")
        except Exception as e:
            logger.error(f"Failed to stream event to Kafka: {e}")
    
    def offload_to_mongo(self, sector_id: str, cell_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Offload cold hex cells to MongoDB for scalable archival storage
        
        Args:
            sector_id: ID of the memory sector to offload from
            cell_ids: Specific cell IDs to offload (if None, offloads all cold cells)
            
        Returns:
            Dictionary with offload statistics
        """
        if sector_id not in self.memory_sectors:
            logger.error(f"Sector {sector_id} not found")
            return {'success': False, 'error': f'Sector {sector_id} not found'}
        
        sector = self.memory_sectors[sector_id]
        
        # If no specific cells provided, offload all cold cells
        if cell_ids is None:
            cell_ids = [cid for cid, cell in sector.cells.items() 
                       if cell.status == HexCellStatus.COLD]
        
        offloaded_count = 0
        offloaded_size = 0
        offloaded_cells = []
        
        for cell_id in cell_ids:
            if cell_id in sector.cells:
                hex_cell = sector.cells[cell_id]
                
                # Prepare data for MongoDB storage
                mongo_doc = {
                    'cell_id': hex_cell.cell_id,
                    'sector_id': hex_cell.sector_id,
                    'data': hex_cell.data,
                    'access_frequency': hex_cell.access_frequency,
                    'last_access': hex_cell.last_access,
                    'creation_time': hex_cell.creation_time,
                    'size_bytes': hex_cell.size_bytes,
                    'status': hex_cell.status.value,
                    'priority_score': hex_cell.priority_score,
                    'offload_timestamp': datetime.utcnow()
                }
                
                try:
                    # Store in MongoDB swap collection
                    result = self.swap_collection.insert_one(mongo_doc)
                    logger.debug(f"Offloaded hex cell {cell_id} to MongoDB with ID: {result.inserted_id}")
                    
                    # Remove from memory sector
                    sector.current_size -= hex_cell.size_bytes
                    del sector.cells[cell_id]
                    
                    offloaded_count += 1
                    offloaded_size += hex_cell.size_bytes
                    offloaded_cells.append(cell_id)
                    
                except Exception as e:
                    logger.error(f"Failed to offload cell {cell_id} to MongoDB: {e}")
        
        # Update sector density after offloading
        sector.density = sector.current_size / (sector.max_capacity * 1000)
        
        # Publish offload event to Kafka
        event_data = {
            'event_type': 'offload_to_mongo',
            'sector_id': sector_id,
            'offloaded_count': offloaded_count,
            'offloaded_size': offloaded_size,
            'remaining_density': sector.density,
            'timestamp': datetime.utcnow().isoformat(),
            'cells_offloaded': offloaded_cells
        }
        self.stream_event_to_kafka(event_data)
        
        logger.info(f"Offloaded {offloaded_count} cold cells from sector {sector_id} to MongoDB")
        
        return {
            'success': True,
            'sector_id': sector_id,
            'offloaded_count': offloaded_count,
            'offloaded_size': offloaded_size,
            'remaining_density': sector.density,
            'cells_offloaded': offloaded_cells
        }
    
    def retrieve_from_mongo(self, cell_id: str) -> Optional[HexCell]:
        """
        Retrieve a hex cell from MongoDB archival storage back to active memory
        
        Args:
            cell_id: ID of the hex cell to retrieve
            
        Returns:
            Retrieved HexCell object, or None if not found
        """
        try:
            # Find in MongoDB swap collection
            mongo_doc = self.swap_collection.find_one({'cell_id': cell_id})
            if mongo_doc:
                # Create hex cell from retrieved data
                hex_cell = HexCell(
                    sector_id=mongo_doc['sector_id'],
                    cell_id=mongo_doc['cell_id'],
                    data=mongo_doc['data'],
                    access_frequency=mongo_doc['access_frequency'],
                    last_access=mongo_doc['last_access'],
                    creation_time=mongo_doc['creation_time'],
                    size_bytes=mongo_doc['size_bytes'],
                    status=HexCellStatus.WARM,  # Retrieved cells start as warm
                    priority_score=mongo_doc['priority_score']
                )
                
                # Publish retrieval event to Kafka
                event_data = {
                    'event_type': 'retrieve_from_mongo',
                    'cell_id': cell_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.stream_event_to_kafka(event_data)
                
                logger.debug(f"Retrieved hex cell {cell_id} from MongoDB")
                return hex_cell
            else:
                logger.warning(f"Hex cell {cell_id} not found in MongoDB")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve cell {cell_id} from MongoDB: {e}")
            return None
    
    async def _evict_cold_cells(self, sector_id: str):
        """
        Asynchronously evict cold cells from memory sector when density threshold is exceeded
        """
        if sector_id not in self.memory_sectors:
            return
        
        sector = self.memory_sectors[sector_id]
        
        # Identify cold cells for eviction (lowest priority first)
        cold_cells = [(cell_id, cell.priority_score) 
                     for cell_id, cell in sector.cells.items() 
                     if cell.status == HexCellStatus.COLD]
        
        # Sort by priority (ascending - lowest priority first)
        cold_cells.sort(key=lambda x: x[1])
        
        # Offload cells until density is below threshold
        while sector.density > self.hex_topology_density_threshold and cold_cells:
            cell_id, _ = cold_cells.pop(0)  # Get lowest priority cold cell
            self.offload_to_mongo(sector_id, [cell_id])
            
            # Refresh list of cold cells after each offload
            cold_cells = [(cell_id, cell.priority_score) 
                         for cell_id, cell in sector.cells.items() 
                         if cell.status == HexCellStatus.COLD]
            cold_cells.sort(key=lambda x: x[1])
    
    def log_evidence(self, evidence_data: Dict[str, Any], evidence_type: str = "forensic") -> str:
        """
        Log immutable evidence to the evidence locker with strict separation
        
        Args:
            evidence_data: Data to log as evidence
            evidence_type: Type of evidence being logged
            
        Returns:
            Evidence record ID
        """
        evidence_record = {
            'evidence_id': f"EVIDENCE_{uuid.uuid4().hex[:8].upper()}",
            'evidence_type': evidence_type,
            'data': evidence_data,
            'timestamp': datetime.utcnow().isoformat(),
            'immutable': True,  # Mark as immutable
            'integrity_hash': self._calculate_integrity_hash(evidence_data)
        }
        
        try:
            # Store in evidence collection (strictly immutable)
            result = self.evidence_collection.insert_one(evidence_record)
            logger.info(f"Logged immutable evidence: {evidence_record['evidence_id']}")
            
            # Publish evidence logging event to Kafka
            event_data = {
                'event_type': 'evidence_logged',
                'evidence_id': evidence_record['evidence_id'],
                'evidence_type': evidence_type,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.stream_event_to_kafka(event_data)
            
            return evidence_record['evidence_id']
        except Exception as e:
            logger.error(f"Failed to log evidence to MongoDB: {e}")
            return f"EVIDENCE_ERROR_{str(e)}"
    
    def _calculate_integrity_hash(self, data: Any) -> str:
        """
        Calculate integrity hash for evidence data to ensure immutability
        """
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(data_str).hexdigest()
    
    def get_sector_statistics(self, sector_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a memory sector
        """
        if sector_id not in self.memory_sectors:
            return {'error': f'Sector {sector_id} not found'}
        
        sector = self.memory_sectors[sector_id]
        
        # Count cells by status
        status_counts = {status.value: 0 for status in HexCellStatus}
        total_priority = 0
        total_size = 0
        
        for cell in sector.cells.values():
            status_counts[cell.status.value] += 1
            total_priority += cell.priority_score
            total_size += cell.size_bytes
        
        avg_priority = total_priority / len(sector.cells) if sector.cells else 0
        
        return {
            'sector_id': sector_id,
            'density': sector.density,
            'cell_count': len(sector.cells),
            'current_size_bytes': total_size,
            'max_capacity': sector.max_capacity,
            'status_distribution': status_counts,
            'average_priority': avg_priority,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def cleanup(self):
        """
        Clean up connections and resources
        """
        try:
            if hasattr(self, 'kafka_producer') and hasattr(self.kafka_producer, 'flush'):
                self.kafka_producer.flush()
        except Exception as e:
            logger.error(f"Error flushing Kafka producer: {e}")
        
        try:
            # Use the correct attribute name (mongo_client vs mongo_client)
            if hasattr(self, 'mongo_client') and hasattr(self.mongo_client, 'close'):
                self.mongo_client.close()
        except Exception as e:
            logger.error(f"Error closing MongoDB client: {e}")
        
        logger.info("Holocube Storage Bridge connections closed")


# Example usage
if __name__ == "__main__":
    # Initialize the storage bridge
    storage_bridge = StorageBridge(
        kafka_bootstrap_servers="localhost:9092",
        mongodb_connection_string="mongodb://localhost:27017/",
        mongodb_database="fanuc_rise_storage"
    )
    
    # Register a memory sector
    sector = storage_bridge.register_memory_sector("SECTOR_001", max_capacity=500)
    
    print("Holocube Storage Bridge initialized successfully")
    print("Ready to manage data lifecycle between RAM, Kafka, and MongoDB")
    print("Features:")
    print("- Hexadecimal Topology memory management")
    print("- Scalable eviction policy for cold cells")
    print("- Kafka streaming for real-time telemetry")
    print("- MongoDB archival storage for horizontal scalability")
    print("- Strict Evidence Separation for forensic logs")