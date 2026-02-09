"""
Simple FastAPI server for LLM-powered manufacturing insights
Provides REST API for natural language queries and cross-session analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from backend.cms.cross_session_intelligence import CrossSessionIntelligence, DataPoint
from backend.cms.llm_integration_examples import LLMManufacturingAssistant

# Initialize FastAPI
app = FastAPI(
    title="CNC Copilot LLM API",
    description="Natural language interface to manufacturing intelligence",
    version="1.0.0"
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize intelligence
intelligence = CrossSessionIntelligence()
assistant = LLMManufacturingAssistant()


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str
    timestamp: str


class DataPointRequest(BaseModel):
    session_id: str
    data_type: str
    data: Dict[str, Any]
    machine_id: Optional[str] = None
    part_id: Optional[str] = None


class PredictionRequest(BaseModel):
    event_type: str
    current_indicators: Dict[str, Any]


class PredictionResponse(BaseModel):
    event_type: str
    prediction_confidence: float
    estimated_time_minutes: Optional[float]
    historical_cases: int
    llm_analysis: str


# API Endpoints

@app.get("/")
async def root():
    """API root"""
    return {
        "name": "CNC Copilot LLM API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "natural_language": "/api/ask",
            "add_data": "/api/data/add",
            "predict": "/api/predict",
            "insights": "/api/insights",
            "similar_sessions": "/api/sessions/similar"
        }
    }


@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask natural language question about manufacturing
    
    Example:
    POST /api/ask
    {
        "question": "Why did quality drop yesterday?",
        "context": "Machine CNC_001"
    }
    """
    try:
        # Add context if provided
        full_question = request.question
        if request.context:
            full_question = f"{request.context}\n\n{request.question}"
        
        # Get answer from LLM
        answer = assistant.chat(full_question)
        
        return QuestionResponse(
            answer=answer,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/add")
async def add_data_point(request: DataPointRequest):
    """
    Add data point for cross-session analysis
    
    Example:
    POST /api/data/add
    {
        "session_id": "session_123",
        "data_type": "vibration",
        "data": {"x": 2.5, "y": 2.8},
        "machine_id": "CNC_001"
    }
    """
    try:
        data_point = DataPoint(
            session_id=request.session_id,
            timestamp=datetime.now(),
            data_type=request.data_type,
            data=request.data,
            machine_id=request.machine_id,
            part_id=request.part_id
        )
        
        intelligence.add_data_point(data_point)
        
        return {
            "status": "success",
            "message": "Data point added",
            "session_id": request.session_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_event(request: PredictionRequest):
    """
    Predict future event based on current indicators
    
    Example:
    POST /api/predict
    {
        "event_type": "tool_failure",
        "current_indicators": {
            "vibration": 2.6,
            "load": 85,
            "temperature": 65
        }
    }
    """
    try:
        prediction = intelligence.predict_future_event(
            event_type=request.event_type,
            current_indicators=request.current_indicators
        )
        
        return PredictionResponse(
            event_type=prediction['event_type'],
            prediction_confidence=prediction['prediction_confidence'],
            estimated_time_minutes=prediction.get('estimated_time_minutes'),
            historical_cases=prediction['historical_cases'],
            llm_analysis=prediction.get('llm_analysis', 'No analysis available')
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/insights")
async def get_insights(days: int = 7):
    """
    Get LLM-generated insights for time period
    
    Example:
    GET /api/insights?days=7
    """
    try:
        report = intelligence.generate_insights_report(days=days)
        return report
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/similar")
async def find_similar_sessions(current_data: Dict[str, Any], top_k: int = 5):
    """
    Find similar historical sessions
    
    Example:
    POST /api/sessions/similar
    {
        "vibration": {"x": 2.5, "y": 2.8},
        "temperature": 65
    }
    """
    try:
        similar = intelligence.llm.find_similar_sessions(current_data, top_k=top_k)
        return {
            "similar_sessions": similar,
            "count": len(similar)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get data for specific session"""
    try:
        session_data = [
            {
                'timestamp': dp.timestamp.isoformat(),
                'data_type': dp.data_type,
                'data': dp.data
            }
            for dp in intelligence.data_repository
            if dp.session_id == session_id
        ]
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "data_points": session_data,
            "count": len(session_data)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explain/alert")
async def explain_alert(alert_data: Dict[str, Any]):
    """
    Get LLM explanation for an alert
    
    Example:
    POST /api/explain/alert
    {
        "alert_type": "high_vibration",
        "severity": "high",
        "data": {"vibration": 2.5}
    }
    """
    try:
        explanation = assistant.explain_alert(alert_data)
        
        return {
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend")
async def get_recommendation(situation: str):
    """
    Get LLM recommendation for a situation
    
    Example:
    POST /api/recommend
    {
        "situation": "Vibration increasing over last hour"
    }
    """
    try:
        recommendation = assistant.recommend_action(situation)
        
        return recommendation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_data_points": len(intelligence.data_repository),
        "unique_sessions": len(set(dp.session_id for dp in intelligence.data_repository)),
        "data_types": list(set(dp.data_type for dp in intelligence.data_repository)),
        "machines": list(set(dp.machine_id for dp in intelligence.data_repository if dp.machine_id)),
        "timestamp": datetime.now().isoformat()
    }


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "llm_provider": intelligence.llm.config.provider,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting CNC Copilot LLM API Server...")
    print("📡 API will be available at: http://localhost:8001")
    print("📚 Docs at: http://localhost:8001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
