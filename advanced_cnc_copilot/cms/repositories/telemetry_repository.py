from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from ..models import Telemetry, Machine
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TelemetryRepository(BaseRepository[Telemetry]):
    """
    Repository for handling Telemetry data operations
    Implements the abstract methods and adds specific telemetry functionality
    Optimized for 1kHz telemetry ingestion rates as required by 'Neuro-Safety' reflex
    """
    
    def __init__(self, session: Session):
        super().__init__(session)
    
    def get_by_id(self, id: int) -> Optional[Telemetry]:
        """Get telemetry record by ID"""
        return self.session.query(Telemetry).filter(Telemetry.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[Telemetry]:
        """Get all telemetry records with pagination"""
        return self.session.query(Telemetry).offset(skip).limit(limit).all()
    
    def create(self, entity: Telemetry) -> Telemetry:
        """Create a new telemetry record with optimized bulk insertion capability"""
        self.session.add(entity)
        self.session.flush()  # Flush to get the ID without committing
        return entity
    
    def update(self, id: int, entity: Telemetry) -> Optional[Telemetry]:
        """Update an existing telemetry record"""
        existing = self.get_by_id(id)
        if existing:
            for key, value in entity.__dict__.items():
                if hasattr(existing, key) and key != '_sa_instance_state':
                    setattr(existing, key, value)
            return existing
        return None
    
    def delete(self, id: int) -> bool:
        """Delete a telemetry record by ID"""
        telemetry = self.get_by_id(id)
        if telemetry:
            self.session.delete(telemetry)
            return True
        return False
    
    def create_bulk(self, telemetry_records: List[Telemetry]) -> List[Telemetry]:
        """Bulk insert multiple telemetry records for high-frequency ingestion"""
        self.session.bulk_save_objects(telemetry_records)
        self.session.flush()  # Flush to assign IDs without committing
        return telemetry_records
    
    def get_recent_by_machine(self, machine_id: int, minutes: int = 10) -> List[Telemetry]:
        """Get recent telemetry data for a specific machine"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return (
            self.session.query(Telemetry)
            .filter(
                Telemetry.machine_id == machine_id,
                Telemetry.timestamp >= cutoff_time
            )
            .order_by(desc(Telemetry.timestamp))
            .all()
        )
    
    def get_latest_by_machine(self, machine_id: int) -> Optional[Telemetry]:
        """Get the most recent telemetry data for a specific machine"""
        return (
            self.session.query(Telemetry)
            .filter(Telemetry.machine_id == machine_id)
            .order_by(desc(Telemetry.timestamp))
            .first()
        )
    
    def get_by_time_range(self, machine_id: int, start_time: datetime, end_time: datetime) -> List[Telemetry]:
        """Get telemetry data for a specific machine within a time range"""
        return (
            self.session.query(Telemetry)
            .filter(
                Telemetry.machine_id == machine_id,
                Telemetry.timestamp >= start_time,
                Telemetry.timestamp <= end_time
            )
            .order_by(Telemetry.timestamp)
            .all()
        )
    
    def get_average_metrics_by_time_window(self, machine_id: int, window_minutes: int = 5) -> dict:
        """Get average metrics for a specific machine over a time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        result = (
            self.session.query(
                func.avg(Telemetry.spindle_load).label('avg_spindle_load'),
                func.avg(Telemetry.vibration_x).label('avg_vibration_x'),
                func.avg(Telemetry.dopamine_score).label('avg_dopamine_score'),
                func.avg(Telemetry.cortisol_level).label('avg_cortisol_level'),
                func.count(Telemetry.id).label('record_count')
            )
            .filter(
                Telemetry.machine_id == machine_id,
                Telemetry.timestamp >= cutoff_time
            )
            .first()
        )
        
        if result:
            return {
                'avg_spindle_load': float(result.avg_spindle_load) if result.avg_spindle_load else 0.0,
                'avg_vibration_x': float(result.avg_vibration_x) if result.avg_vibration_x else 0.0,
                'avg_dopamine_score': float(result.avg_dopamine_score) if result.avg_dopamine_score else 0.0,
                'avg_cortisol_level': float(result.avg_cortisol_level) if result.avg_cortisol_level else 0.0,
                'record_count': result.record_count or 0
            }
        
        return {
            'avg_spindle_load': 0.0,
            'avg_vibration_x': 0.0,
            'avg_dopamine_score': 0.0,
            'avg_cortisol_level': 0.0,
            'record_count': 0
        }
    
    def get_high_stress_events(self, machine_id: int, cortisol_threshold: float = 0.7, minutes: int = 30) -> List[Telemetry]:
        """Get telemetry records with high stress (cortisol) levels"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return (
            self.session.query(Telemetry)
            .filter(
                Telemetry.machine_id == machine_id,
                Telemetry.cortisol_level >= cortisol_threshold,
                Telemetry.timestamp >= cutoff_time
            )
            .order_by(desc(Telemetry.cortisol_level))
            .all()
        )
    
    def get_performance_trends(self, machine_id: int, hours: int = 24) -> dict:
        """Get performance trends for a machine over a specified period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Query for hourly averages
        hourly_stats = (
            self.session.query(
                func.date_trunc('hour', Telemetry.timestamp).label('hour'),
                func.avg(Telemetry.spindle_load).label('avg_spindle_load'),
                func.avg(Telemetry.vibration_x).label('avg_vibration_x'),
                func.avg(Telemetry.dopamine_score).label('avg_dopamine_score'),
                func.avg(Telemetry.cortisol_level).label('avg_cortisol_level'),
                func.count(Telemetry.id).label('count')
            )
            .filter(
                Telemetry.machine_id == machine_id,
                Telemetry.timestamp >= cutoff_time
            )
            .group_by(func.date_trunc('hour', Telemetry.timestamp))
            .order_by('hour')
            .all()
        )
        
        trends = []
        for stat in hourly_stats:
            trends.append({
                'timestamp': stat.hour.isoformat(),
                'avg_spindle_load': float(stat.avg_spindle_load) if stat.avg_spindle_load else 0.0,
                'avg_vibration_x': float(stat.avg_vibration_x) if stat.avg_vibration_x else 0.0,
                'avg_dopamine_score': float(stat.avg_dopamine_score) if stat.avg_dopamine_score else 0.0,
                'avg_cortisol_level': float(stat.avg_cortisol_level) if stat.avg_cortisol_level else 0.0,
                'count': stat.count
            })
        
        return {'trends': trends}
    
    def get_phantom_trauma_indicators(self, machine_id: int, minutes: int = 60) -> List[Telemetry]:
        """
        Get indicators of 'Phantom Trauma' - when system shows high stress but no corresponding 
        physical indicators, suggesting overly sensitive responses to safe conditions
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        # Look for high cortisol but low actual stress indicators (vibration, temperature, load)
        return (
            self.session.query(Telemetry)
            .filter(
                Telemetry.machine_id == machine_id,
                Telemetry.timestamp >= cutoff_time,
                Telemetry.cortisol_level > 0.6,  # High stress response
                Telemetry.vibration_x < 0.5,    # But low actual vibration
                Telemetry.spindle_load < 60     # And low actual load
            )
            .order_by(desc(Telemetry.cortisol_level))
            .all()
        )
    
    def get_neuro_safety_state(self, machine_id: int, minutes: int = 5) -> dict:
        """
        Get the current 'Neuro-Safety' state with dopamine and cortisol levels
        """
        recent_data = self.get_recent_by_machine(machine_id, minutes)
        
        if not recent_data:
            return {
                'dopamine_level': 0.5,
                'cortisol_level': 0.1,
                'stress_trend': 'neutral',
                'reward_trend': 'neutral',
                'neuro_state': 'stable'
            }
        
        # Calculate averages
        avg_dopamine = sum(t.dopamine_score for t in recent_data if t.dopamine_score is not None) / len(recent_data)
        avg_cortisol = sum(t.cortisol_level for t in recent_data if t.cortisol_level is not None) / len(recent_data)
        
        # Calculate trends
        if len(recent_data) >= 2:
            recent_early = recent_data[:len(recent_data)//2]
            recent_later = recent_data[len(recent_data)//2:]
            
            early_dopamine = sum(t.dopamine_score for t in recent_early if t.dopamine_score is not None) / len(recent_early)
            late_dopamine = sum(t.dopamine_score for t in recent_later if t.dopamine_score is not None) / len(recent_later)
            
            early_cortisol = sum(t.cortisol_level for t in recent_early if t.cortisol_level is not None) / len(recent_early)
            late_cortisol = sum(t.cortisol_level for t in recent_later if t.cortisol_level is not None) / len(recent_later)
            
            dopamine_trend = 'increasing' if late_dopamine > early_dopamine else 'decreasing'
            cortisol_trend = 'increasing' if late_cortisol > early_cortisol else 'decreasing'
        else:
            dopamine_trend = 'neutral'
            cortisol_trend = 'neutral'
        
        # Determine neuro state
        if avg_cortisol > 0.7:
            neuro_state = 'high_stress'
        elif avg_cortisol > 0.5 and avg_dopamine < 0.3:
            neuro_state = 'stressed_low_reward'
        elif avg_dopamine > 0.8 and avg_cortisol < 0.3:
            neuro_state = 'optimal'
        elif abs(avg_dopamine - avg_cortisol) < 0.2:
            neuro_state = 'balanced'
        else:
            neuro_state = 'stable'
        
        return {
            'dopamine_level': avg_dopamine,
            'cortisol_level': avg_cortisol,
            'stress_trend': cortisol_trend,
            'reward_trend': dopamine_trend,
            'neuro_state': neuro_state
        }