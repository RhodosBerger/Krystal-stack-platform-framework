from flask import Blueprint, request, jsonify, Response, stream_with_context
import logging
from backend.cms.services.dopamine_engine import DopamineEngine
from backend.integrations.synapse.gcode_streamer import GCodeStreamer, MockRepo

synapse_bp = Blueprint('synapse', __name__)
logger = logging.getLogger(__name__)

# Initialize Engine (Mock Repo for now, or real one)
repo = MockRepo()
engine = DopamineEngine(repo)
streamer = GCodeStreamer(engine)

@synapse_bp.route('/stream', methods=['POST'])
def stream_gcode():
    """
    Endpoint to stream G-Code.
    Expects textual G-Code in body.
    Returns a streamed response of processed G-Code lines.
    """
    gcode_text = request.data.decode('utf-8')
    gcode_lines = gcode_text.splitlines()

    def generate():
        for line in streamer.stream(gcode_lines):
            yield f"{line}\n"

    return Response(stream_with_context(generate()), mimetype='text/plain')

@synapse_bp.route('/status', methods=['GET'])
def get_status():
    """
    Returns the current status of the Synapse.
    """
    return jsonify({
        "safety_interdictions": streamer.safety_interdictions,
        "is_paused": streamer.is_paused,
        "active_machine": streamer.machine_id
    })
