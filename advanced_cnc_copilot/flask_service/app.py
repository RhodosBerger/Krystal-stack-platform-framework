#!/usr/bin/env python3
"""
FANUC RISE - FLASK MICROSERVICE
Real-time telemetry & WebSocket server
"""

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import json
from datetime import datetime
import threading
import time

# Import our core modules
import sys
sys.path.append('..')
from cms.sensory_cortex import SensoryCortex
from cms.impact_cortex import ImpactCortex
from cms.dopamine_engine import DopamineEngine
from cms.signaling_system import SignalingSystem
from cms.demo_data_generator import DemoDataGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fanuc-rise-secret-key-change-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
demo_gen = DemoDataGenerator()
cortex = ImpactCortex()
dopamine = DopamineEngine()
semaphore = SignalingSystem()

# Store active connections
active_connections = {}

# ======================
# REST API ENDPOINTS
# ======================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "flask": "running",
            "websocket": "running",
            "demo_generator": "active"
        }
    })

@app.route('/api/telemetry/current', methods=['GET'])
def get_current_telemetry():
    """Get current telemetry snapshot"""
    data = demo_gen.generate_telemetry()
    return jsonify(data)

@app.route('/api/telemetry/history', methods=['GET'])
def get_telemetry_history():
    """Get historical telemetry"""
    minutes = request.args.get('minutes', 60, type=int)
    data = demo_gen.generate_historical_batch(minutes)
    return jsonify(data)

@app.route('/api/dopamine/evaluate', methods=['POST'])
def evaluate_dopamine():
    """Evaluate dopamine based on current metrics"""
    metrics = request.json
    
    reward = dopamine.calculate_reward(
        load=metrics.get('load', 0),
        vibration=metrics.get('vibration', 0),
        temperature=metrics.get('temperature', 0)
    )
    
    return jsonify({
        "dopamine": reward,
        "cortisol": 100 - reward,
        "reasoning": dopamine.get_reasoning()
    })

@app.route('/api/signal/check', methods=['POST'])
def check_signal():
    """Check semaphore signal"""
    metrics = request.json
    
    signal = semaphore.evaluate(metrics)
    
    return jsonify({
        "signal": signal,
        "can_proceed": signal == "GREEN",
        "warning": semaphore.get_warning_message() if signal != "GREEN" else None
    })

@app.route('/api/impact/decide', methods=['POST'])
def make_decision():
    """Let impact cortex make decision"""
    context = request.json
    
    decision = cortex.make_decision(
        sensory_data=context.get('sensory'),
        dopamine_level=context.get('dopamine'),
        signal=context.get('signal')
    )
    
    return jsonify(decision)

# ======================
# WEBSOCKET EVENTS
# ======================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f'Client connected: {request.sid}')
    emit('connection_response', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f'Client disconnected: {request.sid}')
    if request.sid in active_connections:
        del active_connections[request.sid]

@socketio.on('join_machine')
def handle_join_machine(data):
    """Join machine room for real-time updates"""
    machine_id = data.get('machine_id', 'CNC_VMC_01')
    join_room(machine_id)
    active_connections[request.sid] = machine_id
    
    emit('joined_machine', {
        'machine_id': machine_id,
        'message': f'Subscribed to {machine_id} updates'
    })

@socketio.on('leave_machine')
def handle_leave_machine(data):
    """Leave machine room"""
    machine_id = data.get('machine_id')
    if machine_id:
        leave_room(machine_id)
    
    if request.sid in active_connections:
        del active_connections[request.sid]

@socketio.on('request_telemetry')
def handle_telemetry_request(data):
    """Manual telemetry request"""
    telemetry = demo_gen.generate_telemetry()
    emit('telemetry_update', telemetry)

# ======================
# BACKGROUND TELEMETRY STREAM
# ======================

def telemetry_broadcaster():
    """Background thread broadcasting telemetry every 1 second"""
    while True:
        time.sleep(1)
        
        # Generate telemetry
        telemetry = demo_gen.generate_telemetry()
        
        # Evaluate dopamine
        dopamine_score = dopamine.calculate_reward(
            load=telemetry['load'],
            vibration=telemetry['vibration']['z'],
            temperature=telemetry['spindle_temp']
        )
        telemetry['dopamine'] = dopamine_score
        telemetry['cortisol'] = 100 - dopamine_score
        
        # Check signal
        signal = semaphore.evaluate({
            'load': telemetry['load'],
            'vibration': telemetry['vibration']['z']
        })
        telemetry['signal'] = signal
        
        # Broadcast to all connected clients
        socketio.emit('telemetry_update', telemetry, room='CNC_VMC_01')

# Start background thread
thread = threading.Thread(target=telemetry_broadcaster, daemon=True)
thread.start()

# ======================
# MAIN ENTRY POINT
# ======================

if __name__ == '__main__':
    print("=" * 60)
    print("FANUC RISE - Flask Microservice")
    print("Real-time Telemetry & WebSocket Server")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:5000")
    print(f"WebSocket endpoint: ws://0.0.0.0:5000")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
