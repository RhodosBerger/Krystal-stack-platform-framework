from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from ..models import get_session_local, create_database_engine
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


@router.get("/recent/{machine_id}")
async def get_recent_telemetry(machine_id: int, minutes: int = 10, db: Session = Depends(get_db)):
    """Get recent telemetry data for a specific machine"""
    repo = TelemetryRepository(db)
    telemetry_records = repo.get_recent_by_machine(machine_id, minutes)
    
    return {
        "machine_id": machine_id,
        "records": [
            {
                "id": t.id,
                "timestamp": t.timestamp.isoformat(),
                "spindle_load": _extract_float_value(t, 'spindle_load', 0.0),
                "vibration_x": _extract_float_value(t, 'vibration_x', 0.0),
                "dopamine_score": _extract_float_value(t, 'dopamine_score', 0.0),
                "cortisol_level": _extract_float_value(t, 'cortisol_level', 0.0),
                "spindle_rpm": _extract_float_value(t, 'spindle_rpm', 0.0),
                "feed_rate": _extract_float_value(t, 'feed_rate', 0.0)
            } for t in telemetry_records
        ],
        "count": len(telemetry_records)
    }


@router.get("/latest/{machine_id}")
async def get_latest_telemetry(machine_id: int, db: Session = Depends(get_db)):
    """Get the most recent telemetry data for a specific machine"""
    repo = TelemetryRepository(db)
    latest = repo.get_latest_by_machine(machine_id)
    
    if not latest:
        raise HTTPException(status_code=404, detail="No telemetry data found for this machine")
    
    return {
        "id": latest.id,
        "machine_id": latest.machine_id,
        "timestamp": latest.timestamp.isoformat(),
        "spindle_load": _extract_float_value(latest, 'spindle_load', 0.0),
        "vibration_x": _extract_float_value(latest, 'vibration_x', 0.0),
        "dopamine_score": _extract_float_value(latest, 'dopamine_score', 0.0),
        "cortisol_level": _extract_float_value(latest, 'cortisol_level', 0.0),
        "spindle_rpm": _extract_float_value(latest, 'spindle_rpm', 0.0),
        "feed_rate": _extract_float_value(latest, 'feed_rate', 0.0)
    }


@router.get("/{machine_id}/neuro-status")
async def get_neuro_status(machine_id: int, db: Session = Depends(get_db)):
    """Get neuro-status with dopamine and cortisol levels for a specific machine"""
    repo = TelemetryRepository(db)
    latest = repo.get_latest_by_machine(machine_id)
    
    if not latest:
        raise HTTPException(status_code=404, detail="No telemetry data found for this machine")
    
    # Initialize dopamine engine for analysis
    dopamine_engine = DopamineEngine(repo)
    
    current_metrics = {
        'spindle_load': _extract_float_value(latest, 'spindle_load', 0.0),
        'vibration_x': _extract_float_value(latest, 'vibration_x', 0.0),
        'temperature': _extract_float_value(latest, 'temperature', 35.0),
    }
    
    # Get dopamine and cortisol levels
    dopamine_level = dopamine_engine.calculate_dopamine_response(machine_id, current_metrics)
    cortisol_level = dopamine_engine.calculate_cortisol_response(machine_id, current_metrics)
    
    # Get process recommendation
    recommendation = dopamine_engine.get_process_recommendation(machine_id, current_metrics)
    
    return {
        "machine_id": machine_id,
        "timestamp": latest.timestamp.isoformat(),
        "dopamine_level": dopamine_level,
        "cortisol_level": cortisol_level,
        "neuro_state": recommendation['suggested_action'],
        "confidence": recommendation['confidence'],
        "reasoning": recommendation['reasoning']
    }


@router.get("/{machine_id}/phantom-trauma-check")
async def check_phantom_trauma(machine_id: int, db: Session = Depends(get_db)):
    """Check for 'Phantom Trauma' - when system is overly sensitive to safe conditions"""
    repo = TelemetryRepository(db)
    latest = repo.get_latest_by_machine(machine_id)
    
    if not latest:
        raise HTTPException(status_code=404, detail="No telemetry data found for this machine")
    
    # Initialize dopamine engine for analysis
    dopamine_engine = DopamineEngine(repo)
    
    current_metrics = {
        'spindle_load': _extract_float_value(latest, 'spindle_load', 0.0),
        'vibration_x': _extract_float_value(latest, 'vibration_x', 0.0),
        'temperature': _extract_float_value(latest, 'temperature', 35.0),
    }
    
    # Check for phantom trauma
    is_phantom, description = dopamine_engine.detect_phantom_trauma(machine_id, current_metrics)
    
    return {
        "machine_id": machine_id,
        "timestamp": latest.timestamp.isoformat(),
        "is_phantom_trauma": is_phantom,
        "description": description,
        "current_metrics": current_metrics
    }


@router.get("/{machine_id}/economic-analysis")
async def get_economic_analysis(machine_id: int, db: Session = Depends(get_db)):
    """Get economic analysis for a specific machine"""
    repo = TelemetryRepository(db)
    latest = repo.get_latest_by_machine(machine_id)
    
    if not latest:
        raise HTTPException(status_code=404, detail="No telemetry data found for this machine")
    
    # Initialize economics engine
    economics_engine = EconomicsEngine(repo)
    
    # Create job data for analysis (using latest telemetry values)
    job_data = {
        'machine_id': machine_id,
        'estimated_duration_hours': 1.0,
        'actual_duration_hours': 1.0,
        'sales_price': 1000.0,
        'material_cost': 200.0,
        'labor_hours': 1.0,
        'part_count': 1
    }
    
    # Perform economic analysis
    analysis = economics_engine.analyze_job_economics(job_data)
    
    return {
        "machine_id": machine_id,
        "timestamp": latest.timestamp.isoformat(),
        "economic_analysis": analysis
    }


@router.get("/{machine_id}/shadow-council-evaluation")
async def get_shadow_council_evaluation(machine_id: int, db: Session = Depends(get_db)):
    """Get Shadow Council evaluation for a specific machine"""
    repo = TelemetryRepository(db)
    latest = repo.get_latest_by_machine(machine_id)
    
    if not latest:
        raise HTTPException(status_code=404, detail="No telemetry data found for this machine")
    
    # Initialize Shadow Council components
    decision_policy = DecisionPolicy()
    creator_agent = CreatorAgent(repo)
    auditor_agent = AuditorAgent(decision_policy)
    economics_engine = EconomicsEngine(repo)
    accountant_agent = AccountantAgent(economics_engine)
    
    shadow_council = ShadowCouncil(
        creator=creator_agent,
        auditor=auditor_agent,
        decision_policy=decision_policy
    )
    shadow_council.set_accountant(accountant_agent)
    
    # Prepare current state for evaluation
    current_state = {
        'spindle_load': _extract_float_value(latest, 'spindle_load', 0.0),
        'vibration_x': _extract_float_value(latest, 'vibration_x', 0.0),
        'temperature': _extract_float_value(latest, 'temperature', 35.0),
        'spindle_rpm': _extract_float_value(latest, 'spindle_rpm', 0.0),
        'feed_rate': _extract_float_value(latest, 'feed_rate', 0.0),
        'efficiency_score': 0.5,  # Placeholder
        'stress_level': 0.3,      # Placeholder
    }
    
    # Evaluate strategy through Shadow Council
    evaluation = shadow_council.evaluate_strategy(current_state, machine_id)
    
    return {
        "machine_id": machine_id,
        "timestamp": latest.timestamp.isoformat(),
        "shadow_council_evaluation": evaluation
    }