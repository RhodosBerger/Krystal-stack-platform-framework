"""
Base 44 Preferences System ⚙️
Robust Configuration Module for CNC Copilot.
Handles:
- API Connection (Auth, URL)
- Dependency Management (Pip Paths)
- Simulation Quality Defaults
- UX Modes
"""
import bpy
import os

class CNC_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__ # Auto-detects 'blender_addon'

    # --- Section 1: Connection (The Brain) ---
    api_url: bpy.props.StringProperty(
        name="API URL",
        description="Address of the CNC Copilot Backend",
        default="http://localhost:8000"
    )
    
    api_key: bpy.props.StringProperty(
        name="API Key",
        description="Authentication Token (JWT)",
        subtype='PASSWORD',
        default=""
    )

    # --- Section 2: Ecosystem (The Factory) ---
    custom_pip_path: bpy.props.StringProperty(
        name="Custom Pip Path",
        description="Path to specific pip binary (optional)",
        subtype='FILE_PATH',
        default=""
    )
    
    auto_install_deps: bpy.props.BoolProperty(
        name="Auto-Install Missing Libs",
        description="Automatically fetch scipy/numpy on startup",
        default=True
    )

    # --- Section 3: Simulation (The Physicist) ---
    sim_quality: bpy.props.EnumProperty(
        name="Default Sim Quality",
        items=[
            ('LOW', "Draft (Fast)", "Low-poly proxy physics"),
            ('MEDIUM', "Standard", "Balanced accuracy"),
            ('HIGH', "Engineering (Slow)", "High-fidelity voxel backend"),
        ],
        default='MEDIUM'
    )

    # --- Section 4: UX (The Interface) ---
    ux_mode: bpy.props.EnumProperty(
        name="UX Experience",
        description="Choose your preferred workflow impression",
        items=[
            ('FAMILIAR', "Familiar (SolidWorks-like)", "Ribbon-style interface mimicking CAD tools"),
            ('NATIVE', "Native (Blender)", "Standard fast Blender panels"),
            ('HYBRID', "Neural Assistant (Base 44)", "Chat-driven and intent-based suggestions"),
        ],
        default='HYBRID'
    )

    def draw(self, context):
        layout = self.layout
        
        # Header
        box = layout.box()
        box.label(text="CNC Copilot | Base 44 Config", icon='PREFERENCES')
        
        # Connection
        row = layout.row()
        col = row.column()
        col.label(text="Connection", icon='URL')
        col.prop(self, "api_url")
        col.prop(self, "api_key")
        
        # Ecosystem
        row = layout.row()
        col = row.column()
        col.label(text="Ecosystem", icon='FILE_SCRIPT')
        col.prop(self, "custom_pip_path")
        col.prop(self, "auto_install_deps")
        
        # Simulation
        row = layout.row()
        col = row.column()
        col.label(text="Physics Defaults", icon='PHYSICS')
        col.prop(self, "sim_quality")
        
        # UX
        layout.separator()
        layout.label(text="User Experience", icon='VIEW3D')
        layout.prop(self, "ux_mode", expand=True)

# Helper to access prefs globally
def get_prefs():
    return bpy.context.preferences.addons[__package__].preferences
