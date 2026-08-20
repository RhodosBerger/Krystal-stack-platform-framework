from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from ..models import Machine, get_session_local, create_database_engine
from ..repositories.telemetry_repository import TelemetryRepository
from ..services.dopamine_engine import DopamineEngine
from ..services.economics_engine import EconomicsEngine
from ..services.shadow_council import ShadowCouncil, DecisionPolicy, CreatorAgent, AuditorAgent, AccountantAgent

router = APIRouter()


def get_db():
    """Dependency to get database session"""
    engine = create_database_engine()
    db = get_session_local(engine)()
    try:
        yield db
    finally:
        db.close()


@router.get("/")
async def get_machines(db: Session = Depends(get_db)):
    """Get list of all registered machines"""
    machines = db.query(Machine).all()
    
    return {
        "machines": [
            {
                "id": m.id,
                "name": m.name,
                "serial_number": m.serial_number,
                "model": m.model,
                "created_at": m.created_at.isoformat() if m.created_at is not None else None,
                "last_seen": m.last_seen.isoformat() if m.last_seen is not None else None
            } for m in machines
        ]
    }


@router.get("/{machine_id}")
async def get_machine(machine_id: int, db: Session = Depends(get_db)):
    """Get details for a specific machine"""
    machine = db.query(Machine).filter(Machine.id == machine_id).first()
    
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    return {
        "id": machine.id,
        "name": machine.name,
        "serial_number": machine.serial_number,
        "model": machine.model,
        "created_at": machine.created_at.isoformat() if machine.created_at is not None else None,
        "last_seen": machine.last_seen.isoformat() if machine.last_seen is not None else None
    }


@router.get("/{machine_id}/performance-summary")
async def get_machine_performance_summary(machine_id: int, db: Session = Depends(get_db)):
    """Get performance summary for a specific machine"""
    repo = TelemetryRepository(db)
    
    # Get machine info
    machine = db.query(Machine).filter(Machine.id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    # Get recent telemetry for performance analysis
    recent_data = repo.get_recent_by_machine(machine_id, minutes=60)  # Last hour
    
    if not recent_data:
        return {
            "machine_id": machine_id,
            "machine_name": machine.name,
            "summary": "No recent telemetry data available",
            "uptime_percentage": 0.0,
            "average_efficiency": 0.0,
            "average_stress": 0.0,
            "recommendations": ["Collect more data to generate performance insights"]
        }
    
    # Calculate performance metrics
    total_records = len(recent_data)
    active_records = sum(1 for r in recent_data if _extract_float_value(r, 'spindle_load', 0) > 5.0)  # Consider active if load > 5%
    
    uptime_percentage = (active_records / total_records * 100) if total_records > 0 else 0.0
    
    # Calculate average metrics
    total_load = sum(_extract_float_value(r, 'spindle_load', 0) for r in recent_data)
    total_vibration = sum(_extract_float_value(r, 'vibration_x', 0) for r in recent_data)
    total_cortisol = sum(_extract_float_value(r, 'cortisol_level', 0) for r in recent_data)
    
    avg_load = total_load / total_records if total_records > 0 else 0.0
    avg_vibration = total_vibration / total_records if total_records > 0 else 0.0
    avg_stress = total_cortisol / total_records if total_records > 0 else 0.0
    
    # Initialize dopamine engine for additional insights
    dopamine_engine = DopamineEngine(repo)
    
    # Get current metrics for recommendation
    latest = repo.get_latest_by_machine(machine_id)
    if latest:
        current_metrics = {
            'spindle_load': _extract_float_value(latest, 'spindle_load', 0.0),
            'vibration_x': _extract_float_value(latest, 'vibration_x', 0.0),
            'temperature': _extract_float_value(latest, 'temperature', 35.0),
        }
        
        recommendation = dopamine_engine.get_process_recommendation(machine_id, current_metrics)
        phantom_trauma, trauma_description = dopamine_engine.detect_phantom_trauma(machine_id, current_metrics)
    else:
        recommendation = {'suggested_action': 'insufficient_data', 'confidence': 0.0, 'reasoning': []}
        phantom_trauma = False
        trauma_description = "No data available"
    
    return {
        "machine_id": machine_id,
        "machine_name": machine.name,
        "period_hours": 1,
        "total_records": total_records,
        "uptime_percentage": round(uptime_percentage, 2),
        "average_spindle_load": round(avg_load, 2),
        "average_vibration": round(avg_vibration, 3),
        "average_stress_level": round(avg_stress, 3),
        "current_recommendation": recommendation['suggested_action'],
        "recommendation_confidence": recommendation['confidence'],
        "phantom_trauma_detected": phantom_trauma,
        "phantom_trauma_details": trauma_description,
        "recommendations": recommendation['reasoning']
    }


def _extract_float_value(obj, attr_name, default=0.0):
    """Helper function to safely extract float values from SQLAlchemy objects"""
    try:
        value = getattr(obj, attr_name, None)
        if value is None:
            return default
        if hasattr(value, '__float__'):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        else:
            return float(value) if value is not None else default
    except Exception:
        return default