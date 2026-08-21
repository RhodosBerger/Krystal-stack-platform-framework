bl_info = {
    "name": "Advanced CNC Copilot: Creative Twin",
    "author": "Fanuc RISE Team",
    "version": (1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > CNC Copilot",
    "description": "AI-Powered Manufacturing Twin with Multiple UX Modes (SW Impression)",
    "category": "3D View",
}

import bpy
from .ui_modes import CNC_PT_Familiar_Mode, CNC_PT_Native_Mode, CNC_PT_Hybrid_Mode
from .client import CNC_OT_CheckManufacturability
from .ui_dialogs import CNC_OT_AnalyzeStress, CNC_OT_AuditManufacturing, CNC_OT_EstimateCost
from .part_factory import CNC_OT_GeneratePart, CNC_OT_FunctionalityBuilder
from .render_studio import CNC_OT_SetupStudio, CNC_OT_AssignMaterial
from .interface_engine import CNC_OT_SmartTrigger, CNC_OT_ExplainUI, CNC_OT_NaturalCommand
from .dependency_manager import CNC_OT_InstallLib, CNC_OT_FetchAddon
from .preferences import CNC_Preferences, get_prefs
from .live_link import CNC_OT_ConnectLiveLink, CNC_PT_LiveLink

def register():
    bpy.utils.register_class(CNC_Preferences)
    bpy.utils.register_class(CNC_OT_CheckManufacturability)
    
    # Register Dialogs
    bpy.utils.register_class(CNC_OT_AnalyzeStress)
    bpy.utils.register_class(CNC_OT_AuditManufacturing)
    bpy.utils.register_class(CNC_OT_EstimateCost)
    
    # Register Generative & Render
    bpy.utils.register_class(CNC_OT_GeneratePart)
    bpy.utils.register_class(CNC_OT_FunctionalityBuilder)
    bpy.utils.register_class(CNC_OT_SetupStudio)
    bpy.utils.register_class(CNC_OT_AssignMaterial)

    # Register Interface Engine
    bpy.utils.register_class(CNC_OT_SmartTrigger)
    bpy.utils.register_class(CNC_OT_ExplainUI)
    bpy.utils.register_class(CNC_OT_NaturalCommand)

    # Register Ecosystem Manager
    bpy.utils.register_class(CNC_OT_InstallLib)
    bpy.utils.register_class(CNC_OT_FetchAddon)

    # Register Live Link
    bpy.utils.register_class(CNC_OT_ConnectLiveLink)
    bpy.utils.register_class(CNC_PT_LiveLink)

    # Register Panels
    bpy.utils.register_class(CNC_PT_Familiar_Mode)
    bpy.utils.register_class(CNC_PT_Native_Mode)
    bpy.utils.register_class(CNC_PT_Hybrid_Mode)
    
    # Init Properties
    bpy.types.Scene.cnc_copilot_result = bpy.props.StringProperty(name="Analysis Result", default="")

def unregister():
    bpy.utils.unregister_class(CNC_Preferences)
    bpy.utils.unregister_class(CNC_OT_CheckManufacturability)
    
    bpy.utils.unregister_class(CNC_OT_AnalyzeStress)
    bpy.utils.unregister_class(CNC_OT_AuditManufacturing)
    bpy.utils.unregister_class(CNC_OT_EstimateCost)
    
    bpy.utils.unregister_class(CNC_OT_GeneratePart)
    bpy.utils.unregister_class(CNC_OT_FunctionalityBuilder)
    bpy.utils.unregister_class(CNC_OT_SetupStudio)
    bpy.utils.unregister_class(CNC_OT_AssignMaterial)
    
    bpy.utils.unregister_class(CNC_OT_SmartTrigger)
    bpy.utils.unregister_class(CNC_OT_ExplainUI)
    bpy.utils.unregister_class(CNC_OT_NaturalCommand)

    bpy.utils.unregister_class(CNC_OT_InstallLib)
    bpy.utils.unregister_class(CNC_OT_FetchAddon)
    
    bpy.utils.unregister_class(CNC_OT_ConnectLiveLink)
    bpy.utils.unregister_class(CNC_PT_LiveLink)

    bpy.utils.unregister_class(CNC_PT_Familiar_Mode)
    bpy.utils.unregister_class(CNC_PT_Native_Mode)
    bpy.utils.unregister_class(CNC_PT_Hybrid_Mode)
    del bpy.types.Scene.cnc_copilot_result

if __name__ == "__main__":
    register()
