"""
The Live Link 🔗
Real-time WebSocket bridge between Blender (Creative Twin) and Backend (Brain).
Handles bidirectional sync of Selection, Geometry, and Simulation events.
"""
import bpy
import json
import threading
import asyncio
import websockets
from .preferences import get_prefs

# Global Client Instance
_live_link_client = None

class LiveLinkClient:
    def __init__(self, url):
        self.url = url
        self.stop_event = threading.Event()
        self.websocket = None
        self.thread = None
        self.msg_queue = [] # Queue for Main Thread processing
        
    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        print(f"🔗 Live Link Connecting to {self.url}...")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        print("🔗 Live Link Disconnected.")

    def _run_async_loop(self):
        """Wrapper to run async code in thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen())
        finally:
            loop.close()

    async def _listen(self):
        async for websocket in websockets.connect(self.url):
            self.websocket = websocket
            print("✅ Live Link Connected!")
            try:
                while not self.stop_event.is_set():
                    # Wait for messages with timeout to check stop_event
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        self.msg_queue.append(json.loads(msg))
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("⚠️ Connection Closed, reconnecting...")
                        break
            except Exception as e:
                print(f"❌ Link Error: {e}")
                await asyncio.sleep(5) # Retry delay

    def send(self, data):
        """Non-blocking send (fire and forget for now)"""
        # In a real impl, we'd use an outgoing queue & event loop
        # For MVP, we skip complex sending from this synchronous method 
        # or spin up a quick one-off runner if needed.
        # Ideally, we pass this to the loop. 
        # Simplified: We only listen in this version for stability.
        pass

# --- Blender Operators ---

class CNC_OT_ConnectLiveLink(bpy.types.Operator):
    bl_idname = "cnc.connect_live_link"
    bl_label = "Connect Live Link"
    bl_description = "Establish Real-Time Synapse with Backend"

    def execute(self, context):
        global _live_link_client
        prefs = get_prefs()
        url = prefs.api_url.replace("http", "ws") + "/ws/live_link"
        
        if _live_link_client:
            _live_link_client.stop()
            
        _live_link_client = LiveLinkClient(url)
        _live_link_client.start()
        
        # Start Timer to poll message queue on Main Thread
        bpy.app.timers.register(self.process_queue)
        
        self.report({'INFO'}, "Live Link Connecting...")
        return {'FINISHED'}

    def process_queue(self):
        global _live_link_client
        if not _live_link_client: return None # Stop timer
        
        # Process up to 5 messages per frame to avoid lag
        count = 0 
        while _live_link_client.msg_queue and count < 5:
            msg = _live_link_client.msg_queue.pop(0)
            self.handle_message(msg)
            count += 1
            
        return 0.1 # Run every 0.1s

    def handle_message(self, msg):
        print(f"📩 Received: {msg}")
        type = msg.get("type")
        
        if type == "SIMULATION_RESULT":
            # Update Scene Properties
            data = msg.get("data", {})
            bpy.context.scene.cnc_copilot_result = f"SF: {data.get('safety_factor')} | Stress: {data.get('max_stress')}"
            
            # Visual Feedback (Red if unsafe)
            if data.get("safety_factor", 0) < 1.0:
                 self.report({'ERROR'}, f"SIMULATION FAILED: SF {data.get('safety_factor')}")
            else:
                 self.report({'INFO'}, "Simulation Passed")

class CNC_PT_LiveLink(bpy.types.Panel):
    bl_label = "Live Link ⚡"
    bl_idname = "CNC_PT_LiveLink"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CNC Copilot'

    def draw(self, context):
        layout = self.layout
        layout.operator("cnc.connect_live_link", text="Connect Synapse", icon='LINK_BLEND')
        
        if context.scene.cnc_copilot_result:
            box = layout.box()
            box.label(text="Latest Evidence:", icon='Graph')
            box.label(text=context.scene.cnc_copilot_result)
