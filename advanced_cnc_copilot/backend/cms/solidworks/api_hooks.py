"""
SolidWorks API Hooks 🔗
Maps COM Events from SolidWorks to the Cortex "Simultaneous Analysis" stream.
Enables background listening for "ActiveDoc", "Selection", and "Save" events.
"""
import asyncio
import json
import time
from datetime import datetime

# Mocking connection to Backend for 'Simultaneous Analysis'
from backend.core.cortex_transmitter import cortex

class SolidWorksEventListener:
    def __init__(self, sw_app_object=None):
        self.sw_app = sw_app_object
        self.is_listening = False
        self._cache_last_doc = None

    def start_listening(self):
        """
        Subscribes to SW Application Events.
        In a real COM implementation, this uses pywin32 `KeepAlive`.
        """
        self.is_listening = True
        print("🔗 Hooked into SolidWorks 2026. Listening for geometry changes...")
        # Mock Loop for demo purposes
        # asyncio.create_task(self._mock_event_stream())

    def on_active_doc_change(self, doc_name):
        """
        Event: D0_Application_ActiveDocChangeNotify
        Triggered when user switches tabs.
        """
        cortex.transmit_intent(
            actor="SolidWorks_Hook",
            action="CONTEXT_SWITCH",
            reasoning=f"User focused on {doc_name}",
            context={"filename": doc_name}
        )

    def on_part_save(self, filename):
        """
        Event: D0_PartDoc_FileSaveNotify
        Triggered when user hits Ctrl+S.
        Action: Trigger Background Analysis.
        """
        print(f"💾 SW Saved: {filename} -> Triggering Simultaneous Analysis")
        
        cortex.transmit_intent(
            actor="SolidWorks_Hook",
            action="TRIGGER_ANALYSIS",
            reasoning="File Saved - Validating Integrity",
            context={"file": filename, "trigger": "SAVE_EVENT"}
        )
        
        # Dispatch to Analysis Agent (Mock)
        self._dispatch_analysis(filename)

    def on_selection_change(self, selection_count, object_type):
        """
        Event: D0_UserSelection_ChangeNotify
        Triggered when user clicks a face/edge.
        """
        if selection_count > 0:
            cortex.transmit_intent(
                actor="SolidWorks_Selection",
                action="INFER_INTENT",
                reasoning=f"User selected {object_type}",
                context={"count": selection_count, "type": object_type}
            )

    def _dispatch_analysis(self, filename):
        # Here we would call the /api/manufacturing/request endpoint
        timestamp = datetime.now().isoformat()
        print(f"🚀 [{timestamp}] Analysis Job Dispatched for {filename}")

# Manifest of Hooks for Documentation
HOOK_MAP = {
    "D0_Application_ActiveDocChangeNotify": "on_active_doc_change",
    "D0_PartDoc_FileSaveNotify": "on_part_save",
    "D0_UserSelection_ChangeNotify": "on_selection_change"
}
