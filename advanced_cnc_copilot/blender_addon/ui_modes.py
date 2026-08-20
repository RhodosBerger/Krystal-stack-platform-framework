import bpy

class CNC_Panel_Base:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CNC Copilot"

    @classmethod
    def poll(cls, context):
        # Only show specific panels based on Preference selection
        prefs = context.preferences.addons[__package__].preferences
        return prefs.ux_mode == cls.ux_mode_key

# --- UX MODE 1: FAMILIAR (SolidWorks Impression) ---
class CNC_PT_Familiar_Mode(CNC_Panel_Base, bpy.types.Panel):
    bl_label = "Feature Manager (SolidWorks Mode)"
    bl_idname = "CNC_PT_Familiar"
    ux_mode_key = 'FAMILIAR'

    def draw(self, context):
        layout = self.layout
        
        # Ribbon-like Header
        row = layout.row(align=True)
        row.label(text="Features", icon='MODIFIER')
        row.label(text="Sketch", icon='EDITMODE_HLT')
        row.label(text="Evaluate", icon='MEASURE')
        
        layout.separator()
        
        # CommandManager-style buttons
        box = layout.box()
        box.label(text="Boss/Base")
        col = box.column(align=True)
        col.operator("mesh.primitive_cube_add", text="Extruded Boss/Base", icon='CUBE')
        col.operator("mesh.primitive_cylinder_add", text="Revolved Boss/Base", icon='MESH_CYLINDER')
        
        layout.separator()
        
        # "Evaluate" Section -> Neural Bridge
        box = layout.box()
        box.label(text="Design Study")
        row = box.row()
        row.operator("cnc.check_manufacturability", text="Interference Detection", icon='DRIVER')
        row.operator("cnc.check_manufacturability", text="Draft Analysis", icon='MOD_BEVEL')

# --- UX MODE 2: NATIVE (Blender Speed) ---
class CNC_PT_Native_Mode(CNC_Panel_Base, bpy.types.Panel):
    bl_label = "CNC Tools"
    bl_idname = "CNC_PT_Native"
    ux_mode_key = 'NATIVE'

    def draw(self, context):
        layout = self.layout
        layout.operator("cnc.check_manufacturability", text="Quick Check", icon='CHECKMARK')

# --- UX MODE 3: HYBRID (Neural Assistant) ---
class CNC_PT_Hybrid_Mode(CNC_Panel_Base, bpy.types.Panel):
    bl_label = "Neural Assistant"
    bl_idname = "CNC_PT_Hybrid"
    ux_mode_key = 'HYBRID'

    def draw(self, context):
        layout = self.layout
        
        # "Chat" Interface Impression
        box = layout.box()
        box.label(text="System: 'I see you are making a bracket.'", icon='INFO')
        box.label(text="Suggestion: Increase wall thickness by 2mm.", icon='LIGHT')
        
        row = box.row()
        row.button = True # Visual style
        row.label(text="Apply Fix?")
        row.operator("mesh.primitive_cube_add", text="Yes (Apply)", icon='CHECKBOX_HLT')
        
        layout.separator()
        layout.operator("cnc.check_manufacturability", text="Sync with Cortex", icon='SHADING_RENDERED')
