from flask import Blueprint, request, jsonify, Response
from backend.core.cortex_engine import CortexEngine
import threading

cortex_bp = Blueprint('cortex', __name__)

# Singleton Cortex Instance
cortex = CortexEngine()

@cortex_bp.route('/cortex/execute', methods=['POST'])
def execute_job():
    """
    Executes a manufacturing profile.
    JSON Body: { "profile": {...}, "optimize": true }
    """
    data = request.json
    profile = data.get("profile")
    optimize = data.get("optimize", True)

    def generate():
        for event in cortex.execute_job(profile, optimize):
            yield f"{event}\n"

    return Response(generate(), mimetype='text/plain')

@cortex_bp.route('/cortex/evolve', methods=['POST'])
def evolve_job():
    """
    Just runs the discovery engine and returns the optimized profile.
    """
    data = request.json
    profile = data.get("profile")
    best_mutant = cortex.optimizer.evolve_profile(profile)
    return jsonify(best_mutant)

@cortex_bp.route('/cortex/status', methods=['GET'])
def status():
    """
    Returns the current Covalent Spectrum state (Hex Trace).
    """
    # Create a dummy trace for current state
    spectrum = cortex.streamer.spectrum.get_state()
    trace = cortex.streamer.hex_logger.log_trace(spectrum, 999)
    return jsonify({
        "status": "ONLINE",
        "spectrum": str(spectrum),
        "hex_trace": trace,
        "mode": "CORTEX_ACTIVE"
    })
