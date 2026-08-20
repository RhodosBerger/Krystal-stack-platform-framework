"""
Telemetry Data Loader 📥
Fetches sensor data from TimescaleDB/Postgres for the Feature Engineering Pipeline.
"""
import os
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("TelemetryDataLoader")

class TelemetryDataLoader:
    def __init__(self, connection_string: str = None):
        self.conn_str = connection_string or os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/cnc_db")
        
    def fetch_sensor_data(self, 
                         machine_id: str, 
                         start_time: datetime, 
                         end_time: datetime, 
                         sensor_types: List[str] = None) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Fetch time-series data for a machine within a time window.
        
        Returns:
            Dict mapping sensor_id -> list of (timestamp, value)
        """
        sensor_data = {}
        
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Construct query
                    query = """
                        SELECT timestamp, sensor_type, value 
                        FROM telemetry 
                        WHERE machine_id = %s 
                        AND timestamp BETWEEN %s AND %s
                        ORDER BY timestamp ASC
                    """
                    
                    cur.execute(query, (machine_id, start_time, end_time))
                    rows = cur.fetchall()
                    
                    for row in rows:
                        s_type = row['sensor_type']
                        if sensor_types and s_type not in sensor_types:
                            continue
                            
                        if s_type not in sensor_data:
                            sensor_data[s_type] = []
                            
                        sensor_data[s_type].append((row['timestamp'], float(row['value'])))
                        
            logger.info(f"Fetched {sum(len(v) for v in sensor_data.values())} data points for {machine_id}")
            
        except Exception as e:
            logger.error(f"Failed to fetch telemetry: {e}")
            # Return empty or raise based on policy. For now, empty for robustness.
            return {}
            
        return sensor_data

# Example Usage
if __name__ == "__main__":
    loader = TelemetryDataLoader()
    # Mock data return for standalone testing if DB fails
    print("TelemetryDataLoader Initialized.")
