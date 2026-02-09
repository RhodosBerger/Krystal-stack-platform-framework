"""
IoT Sensor Integration Framework
Real-time manufacturing data collection

ARCHITECTURE:
Sensors → MQTT Broker → Kafka → Database (Time-series)
                           ↓
                       Analytics
                       Dashboard

FEATURES:
- MQTT client for sensor data ingestion
- Support for multiple sensor types
- Buffering and batch insertion
- Real-time streaming to WebSocket clients
- Integration with telemetry table
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import time
from collections import deque
import threading


# =============================================================================
# SENSOR DATA MODELS
# =============================================================================

class SensorType(Enum):
    """Types of sensors in manufacturing"""
    SPINDLE_LOAD = "spindle_load"
    VIBRATION_X = "vibration_x"
    VIBRATION_Y = "vibration_y"
    VIBRATION_Z = "vibration_z"
    TEMPERATURE = "temperature"
    POWER_CONSUMPTION = "power_consumption"
    TOOL_WEAR = "tool_wear"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    POSITION_X = "position_x"
    POSITION_Y = "position_y"
    POSITION_Z = "position_z"


@dataclass
class SensorReading:
    """Single sensor reading"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'unit': self.unit,
            'metadata': self.metadata
        }


@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_id: str
    sensor_type: SensorType
    mqtt_topic: str
    sampling_rate_hz: float
    unit: str
    calibration_offset: float = 0.0
    calibration_scale: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None


# =============================================================================
# MQTT CLIENT (Simulated for now, real implementation needs paho-mqtt)
# =============================================================================

class MQTTSensorClient:
    """
    MQTT client for sensor data ingestion
    
    In production, this would use paho-mqtt library.
    For now, simulated for testing without MQTT broker.
    """
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        """
        Initialize MQTT client
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.subscriptions: Dict[str, Callable] = {}
        self.connected = False
        
        # Try to import paho-mqtt
        try:
            import paho.mqtt.client as mqtt
            self.mqtt = mqtt
            self._mqtt_available = True
        except ImportError:
            print("⚠️ paho-mqtt not installed - running in simulation mode")
            print("   Install with: pip install paho-mqtt")
            self._mqtt_available = False
    
    def connect(self):
        """Connect to MQTT broker"""
        if not self._mqtt_available:
            print(f"📡 [SIMULATED] Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.connected = True
            return
        
        try:
            self.client = self.mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            self.connected = True
            print(f"✅ Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            print(f"❌ Failed to connect to MQTT broker: {e}")
            self.connected = False
    
    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribe to MQTT topic
        
        Args:
            topic: MQTT topic (e.g., "sensors/machine1/spindle_load")
            callback: Function to call when message received
        """
        self.subscriptions[topic] = callback
        
        if self._mqtt_available and self.connected:
            self.client.subscribe(topic)
            print(f"📬 Subscribed to topic: {topic}")
        else:
            print(f"📬 [SIMULATED] Subscribed to topic: {topic}")
    
    def publish(self, topic: str, payload: str):
        """Publish message to topic"""
        if self._mqtt_available and self.connected:
            self.client.publish(topic, payload)
        else:
            print(f"📤 [SIMULATED] Published to {topic}: {payload[:50]}...")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected"""
        print(f"✅ MQTT Connected with result code {rc}")
        # Resubscribe to topics
        for topic in self.subscriptions.keys():
            client.subscribe(topic)
    
    def _on_message(self, client, userdata, msg):
        """Callback when message received"""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        if topic in self.subscriptions:
            self.subscriptions[topic](topic, payload)
    
    def disconnect(self):
        """Disconnect from broker"""
        if self._mqtt_available and hasattr(self, 'client'):
            self.client.loop_stop()
            self.client.disconnect()
        print("👋 Disconnected from MQTT broker")


# =============================================================================
# SENSOR DATA BUFFER
# =============================================================================

class SensorDataBuffer:
    """
    Buffer for sensor readings before database insertion
    
    Implements:
    - Fixed-size circular buffer
    - Thread-safe operations
    - Batch flushing
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize buffer
        
        Args:
            max_size: Maximum readings to buffer
        """
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, reading: SensorReading):
        """Add reading to buffer"""
        with self.lock:
            self.buffer.append(reading)
    
    def flush(self, count: Optional[int] = None) -> List[SensorReading]:
        """
        Flush readings from buffer
        
        Args:
            count: Number of readings to flush (None = all)
        
        Returns:
            List of readings
        """
        with self.lock:
            if count is None:
                readings = list(self.buffer)
                self.buffer.clear()
            else:
                readings = [self.buffer.popleft() for _ in range(min(count, len(self.buffer)))]
            
            return readings
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)


# =============================================================================
# SENSOR MANAGER
# =============================================================================

class SensorManager:
    """
    Central manager for all sensors
    
    Responsibilities:
    - Configure sensors
    - Receive data from MQTT
    - Buffer readings
    - Batch insert to database
    - Stream to WebSocket clients
    """
    
    def __init__(self, operation_id: int):
        """
        Initialize sensor manager
        
        Args:
            operation_id: Operation ID for telemetry records
        """
        self.operation_id = operation_id
        self.sensors: Dict[str, SensorConfig] = {}
        self.mqtt_client = MQTTSensorClient()
        self.buffer = SensorDataBuffer(max_size=10000)
        
        # WebSocket clients (for real-time streaming)
        self.websocket_clients: List[Any] = []
        
        # Statistics
        self.total_readings = 0
        self.readings_per_second = 0
        self._last_flush_time = time.time()
    
    def register_sensor(self, config: SensorConfig):
        """
        Register a sensor
        
        Args:
            config: Sensor configuration
        """
        self.sensors[config.sensor_id] = config
        
        # Subscribe to MQTT topic
        self.mqtt_client.subscribe(
            config.mqtt_topic,
            lambda topic, payload: self._on_sensor_data(config.sensor_id, payload)
        )
        
        print(f"🔧 Registered sensor: {config.sensor_id} ({config.sensor_type.value})")
    
    def start(self):
        """Start sensor data collection"""
        self.mqtt_client.connect()
        
        # Start periodic flush thread
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()
        
        print("🚀 Sensor manager started")
    
    def stop(self):
        """Stop sensor data collection"""
        self.mqtt_client.disconnect()
        print("🛑 Sensor manager stopped")
    
    def _on_sensor_data(self, sensor_id: str, payload: str):
        """
        Callback when sensor data received
        
        Args:
            sensor_id: Sensor ID
            payload: JSON payload
        """
        try:
            # Parse payload
            data = json.loads(payload)
            
            # Get sensor config
            if sensor_id not in self.sensors:
                return
            
            config = self.sensors[sensor_id]
            
            # Create reading
            raw_value = float(data.get('value', 0))
            
            # Apply calibration
            calibrated_value = (raw_value + config.calibration_offset) * config.calibration_scale
            
            # Validate range
            if config.min_value is not None and calibrated_value < config.min_value:
                calibrated_value = config.min_value
            if config.max_value is not None and calibrated_value > config.max_value:
                calibrated_value = config.max_value
            
            reading = SensorReading(
                sensor_id=sensor_id,
                sensor_type=config.sensor_type,
                timestamp=datetime.utcnow(),
                value=calibrated_value,
                unit=config.unit,
                metadata=data.get('metadata', {})
            )
            
            # Add to buffer
            self.buffer.add(reading)
            self.total_readings += 1
            
            # Stream to WebSocket clients (real-time)
            self._stream_to_websockets(reading)
        
        except Exception as e:
            print(f"❌ Error processing sensor data: {e}")
    
    def _periodic_flush(self):
        """Periodically flush buffer to database"""
        while True:
            time.sleep(1)  # Flush every second
            
            readings = self.buffer.flush()
            
            if readings:
                self._insert_to_database(readings)
                
                # Update statistics
                current_time = time.time()
                elapsed = current_time - self._last_flush_time
                self.readings_per_second = len(readings) / elapsed if elapsed > 0 else 0
                self._last_flush_time = current_time
    
    def _insert_to_database(self, readings: List[SensorReading]):
        """
        Insert readings to database
        
        Args:
            readings: List of readings to insert
        """
        # In production, would use database.repository.TelemetryRepository
        # For now, just log
        print(f"💾 Inserting {len(readings)} readings to database")
        
        # Example of what this would look like:
        # from database.connection import db_manager
        # from database.repository import TelemetryRepository
        #
        # with db_manager.session_scope() as session:
        #     samples = [
        #         {
        #             'operation_id': self.operation_id,
        #             'timestamp': r.timestamp,
        #             'sensor_id': r.sensor_id,
        #             'value': r.value,
        #             'unit': r.unit,
        #             'metadata': r.metadata
        #         }
        #         for r in readings
        #     ]
        #     TelemetryRepository.add_batch(session, samples)
    
    def _stream_to_websockets(self, reading: SensorReading):
        """Stream reading to connected WebSocket clients"""
        # Would send via WebSocket in production
        # print(f"📡 Streaming: {reading.sensor_id} = {reading.value} {reading.unit}")
        pass
    
    def get_statistics(self) -> Dict:
        """Get manager statistics"""
        return {
            'total_readings': self.total_readings,
            'buffer_size': self.buffer.size(),
            'readings_per_second': self.readings_per_second,
            'registered_sensors': len(self.sensors)
        }


# =============================================================================
# SENSOR SIMULATOR (For Testing)
# =============================================================================

class SensorSimulator:
    """
    Simulates sensor data for testing
    
    Generates realistic sensor values with noise
    """
    
    def __init__(self, mqtt_client: MQTTSensorClient):
        """
        Initialize simulator
        
        Args:
            mqtt_client: MQTT client to publish to
        """
        self.mqtt_client = mqtt_client
        self.running = False
    
    def start(self, sensors: List[SensorConfig], duration_seconds: int = 60):
        """
        Start simulating sensor data
        
        Args:
            sensors: List of sensors to simulate
            duration_seconds: How long to simulate
        """
        self.running = True
        
        import random
        import math
        
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < duration_seconds:
            for sensor in sensors:
                # Generate realistic value based on sensor type
                if sensor.sensor_type == SensorType.SPINDLE_LOAD:
                    value = 50 + 20 * math.sin(time.time()) + random.gauss(0, 2)
                elif sensor.sensor_type == SensorType.VIBRATION_X:
                    value = 0.3 + 0.2 * math.sin(time.time() * 10) + random.gauss(0, 0.05)
                elif sensor.sensor_type == SensorType.TEMPERATURE:
                    value = 35 + 10 * math.sin(time.time() / 10) + random.gauss(0, 1)
                else:
                    value = random.uniform(0, 100)
                
                # Publish to MQTT
                payload = json.dumps({
                    'value': value,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                self.mqtt_client.publish(sensor.mqtt_topic, payload)
            
            # Simulate sampling rate
            time.sleep(1.0 / sensors[0].sampling_rate_hz if sensors else 0.1)
        
        print("🛑 Sensor simulation stopped")
    
    def stop(self):
        """Stop simulation"""
        self.running = False


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("IoT Sensor Integration Framework - Demo")
    print("=" * 70)
    
    # Create sensor manager
    manager = SensorManager(operation_id=1)
    
    # Register sensors
    sensors = [
        SensorConfig(
            sensor_id="SPINDLE_01",
            sensor_type=SensorType.SPINDLE_LOAD,
            mqtt_topic="sensors/machine1/spindle_load",
            sampling_rate_hz=10.0,
            unit="%",
            min_value=0.0,
            max_value=100.0
        ),
        SensorConfig(
            sensor_id="VIB_X_01",
            sensor_type=SensorType.VIBRATION_X,
            mqtt_topic="sensors/machine1/vibration_x",
            sampling_rate_hz=100.0,
            unit="mm"
        ),
        SensorConfig(
            sensor_id="TEMP_01",
            sensor_type=SensorType.TEMPERATURE,
            mqtt_topic="sensors/machine1/temperature",
            sampling_rate_hz=1.0,
            unit="°C",
            min_value=-50.0,
            max_value=150.0
        )
    ]
    
    for sensor in sensors:
        manager.register_sensor(sensor)
    
    # Start manager
    manager.start()
    
    # Start simulator
    simulator = SensorSimulator(manager.mqtt_client)
    
    print("\n🎬 Starting sensor simulation for 10 seconds...")
    simulator_thread = threading.Thread(
        target=simulator.start,
        args=(sensors, 10),
        daemon=True
    )
    simulator_thread.start()
    
    # Monitor statistics
    for i in range(12):
        time.sleep(1)
        stats = manager.get_statistics()
        print(f"\n📊 Statistics (t={i}s):")
        print(f"   Total readings: {stats['total_readings']}")
        print(f"   Buffer size: {stats['buffer_size']}")
        print(f"   Readings/sec: {stats['readings_per_second']:.1f}")
    
    # Stop
    simulator.stop()
    manager.stop()
    
    print("\n✅ Demo complete!")
