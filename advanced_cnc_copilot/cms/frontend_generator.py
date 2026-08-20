"""
Frontend Generator & Layout Combinatorics Engine 🎨
Responsibility:
1. Parse existing HTML 'Concepts' (cms/dashboard/*.html).
2. Perform 'Combinatorics' to stitch them into specific 'Views'.
3. Serve as the Bridge between Python Backend and Frontend.
"""
import os
from typing import List, Dict

class FrontendGenerator:
    def __init__(self):
        self.dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard')
        self.components = self._scan_components()
        
    def _scan_components(self) -> Dict[str, str]:
        """Load all HTML fragments into memory"""
        comps = {}
        if not os.path.exists(self.dashboard_path):
            return comps
            
        for filename in os.listdir(self.dashboard_path):
            if filename.endswith(".html"):
                path = os.path.join(self.dashboard_path, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    comps[filename] = f.read()
        return comps

    def generate_layout(self, view_type: str, context: Dict = None) -> str:
        """
        Combinatorics Engine: Stitches components based on View Type.
        """
        base_css = """
        <style>
            :root { --primary: #00ff88; --bg: #0a0a0a; --panel: #1a1a1a; }
            body { background: var(--bg); color: #fff; font-family: 'Inter', sans-serif; margin: 0; display: grid; grid-template-rows: 60px 1fr; height: 100vh; }
            header { background: #000; border-bottom: 1px solid #333; display: flex; align-items: center; padding: 0 20px; justify-content: space-between; }
            .nav-btn { background: none; border: 1px solid #333; color: #888; padding: 8px 16px; margin-right: 10px; cursor: pointer; border-radius: 4px; }
            .nav-btn.active { border-color: var(--primary); color: var(--primary); }
            main { padding: 20px; display: grid; gap: 20px; overflow: auto; }
            .panel { background: var(--panel); border: 1px solid #333; border-radius: 8px; padding: 20px; }
            h2 { margin-top: 0; color: var(--primary); }
            iframe { width: 100%; height: 100%; border: none; }
        </style>
        """
        
        nav_bar = f"""
        <header>
            <div style="font-weight:bold; color:var(--primary);">FANUC RISE // {view_type.upper()}</div>
            <nav>
                <button class="nav-btn {'active' if view_type=='neuro' else ''}" onclick="window.location.href='/view/neuro'">Neuro</button>
                <button class="nav-btn {'active' if view_type=='operator' else ''}" onclick="window.location.href='/view/operator'">Operator</button>
                <button class="nav-btn {'active' if view_type=='knowledge' else ''}" onclick="window.location.href='/view/knowledge'">Knowledge</button>
                <button class="nav-btn {'active' if view_type=='cortex' else ''}" onclick="window.location.href='/view/cortex'">Cortex</button>
            </nav>
        </header>
        """
        
        content = ""
        
        if view_type == "neuro":
            # Direct Return of the main Neuro-Command Center
            return self.components.get("index.html", "<div>Neuro Dashboard Missing</div>")

        if view_type == "operator":
            # Combine XYZ Control + Adaptive G-Code Input
            gcode_editor = self.components.get("editor_gcode_adaptive.html", "<div>G-Code Editor Missing</div>")
            
            content = f"""
            <main style="grid-template-columns: 1fr 1fr;">
                <div class="panel">
                    <h2>🕹️ Machine Control</h2>
                    <p>Direct Connection to FANUC FOCAS (Simulated)</p>
                    <div id="xyz-container">Loading Control Panel...</div>
                </div>
                <div class="panel">
                    {gcode_editor}
                </div>
            </main>
            """
            
        elif view_type == "engineer":
            # New View: Parametric Design + Neural Chat
            parametric_editor = self.components.get("editor_parametric.html", "<div>Parametric Editor Missing</div>")
            neural_chat = self.components.get("comp_neural_chat.html", "<div>Chat Missing</div>")
            
            content = f"""
            <main style="grid-template-columns: 2fr 1fr;">
                 <div class="panel">
                    <h2>📐 Design Studio</h2>
                    {parametric_editor}
                 </div>
                 <div class="panel">
                    {neural_chat}
                 </div>
            </main>
            """
            
        elif view_type == "knowledge":
            # Combine 2FA Queue + Search + Presets
            # Load Search Component
            search_html = self.components.get("comp_ajax_search.html", "<div>Search Component Missing</div>")
            
            content = f"""
            <main style="grid-template-columns: 2fr 1fr;">
                <div class="panel">
                    <h2>🧠 Knowledge Base (Presets)</h2>
                    <!-- AJAX Search Injection -->
                    {search_html}
                    
                    <div id="preset-list" style="margin-top:20px;">
                        <h3>Recently Verified</h3>
                        <div style="color:#666;">Load full library to see all...</div>
                    </div>
                </div>
                <div class="panel">
                    <h2>🔐 2FA Verification Queue</h2>
                    <div id="pending-list">Checking Redis...</div>
                    <button onclick="fetch('/api/knowledge/pending').then(r=>r.json()).then(d=>document.getElementById('pending-list').innerText=JSON.stringify(d,null,2))" style="margin-top:10px;">Refresh Queue</button>
                </div>
            </main>
            """
            
        elif view_type == "cortex":
            # Combine Logs + Intent Map
            content = f"""
            <main style="grid-template-rows: 1fr 1fr;">
                <div class="panel">
                    <h2>👁️ Cortex Stream (Logs)</h2>
                    <pre id="log-stream" style="color:#0f0;">Connecting to Cortex Mirror...</pre>
                </div>
                <div class="panel">
                    <h2>🧠 Database of Intent</h2>
                    {self.components.get('comp_analytics_heatmap.html', '<div>Heatmap Loading Error</div>')}
                </div>

            </main>
            """
        
        elif view_type == "blueprint":
            # Phase 22: The Master Blueprint (Mission Control)
            blueprint_html = self.components.get("blueprint_master.html", "<div>Blueprint Missing</div>")
            
            # Fuse with Chat
            neural_chat = self.components.get("comp_neural_chat.html", "<div>Chat Missing</div>")
            
            content = f"""
            <main style="grid-template-columns: 3fr 1fr;">
                 <div style="height: 100%; overflow: hidden;">
                    {blueprint_html}
                 </div>
                 <div class="panel" style="display: flex; flex-direction: column;">
                    <h2>💬 Direct Uplink</h2>
                    <div style="flex-grow: 1; overflow: hidden;">
                        {neural_chat}
                    </div>
                 </div>
            </main>
            """
        
        elif view_type == "full_experience":
            # Phase 31: The Full Experience (Universal Blueprint)
            # This view consolidates Swarm, Spectrum, Chat, and Cortex into one master panel.
            universal_html = self.components.get("universal_blueprint.html", "<div>Universal Blueprint Missing</div>")
            
            # The Universal Blueprint already contains internal placeholders for component injection,
            # but we can do a second pass if needed, or simply return it as is.
            # We fulfill the placeholders using self.components
            content = universal_html.replace(
                "{self.components.get('comp_swarm_status.html', '<div>Swarm Hive Unavailable</div>')}",
                self.components.get('comp_swarm_status.html', '<div>Swarm Hive Unavailable</div>')
            ).replace(
                "{self.components.get('comp_spectrum_sim.html', '<div>Spectrum Simulator Error</div>')}",
                self.components.get('comp_spectrum_sim.html', '<div>Spectrum Simulator Error</div>')
            ).replace(
                "{self.components.get('comp_neural_chat.html', '<div>Uplink Offline</div>')}",
                self.components.get('comp_neural_chat.html', '<div>Uplink Offline</div>')
            )
            
            # Since universal_blueprint is a full HTML doc, we return it directly in generate_layout
            return f"<!DOCTYPE html><html>{content}</html>"

        elif view_type == "multiverse":
            # Phase 37: Multiverse Master (Standalone SaaS Explorer)
            multiverse_html = self.components.get("multiverse_master.html", "<div>Multiverse Master Missing</div>")
            return f"<!DOCTYPE html><html>{multiverse_html}</html>"

        elif view_type == "wizard":
            # Real-time Setup Wizard
            return self.components.get("wizard.html", "<div>Wizard Template Missing</div>")
            
        return f"<!DOCTYPE html><html><head><title>RISE | {view_type}</title>{base_css}</head><body>{nav_bar}{content}</body></html>"

frontend_generator = FrontendGenerator()
