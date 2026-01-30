"""
Dynamic Interface Engine 🤖
Implements 'Smart Triggers' (AltGr), Context Awareness, and Natural Language Command Translation.
This is the bridge between the User's Intent and Blender's API.
"""
import bpy
import blf
import bgl
import threading
import requests
import json

# Global State for Overlay
overlay_active = False
overlay_text = "Neural Assistant Ready..."

def draw_callback_px(self, context):
    """Draws the Neural Assistant Overlay on screen"""
    global overlay_active, overlay_text
    if not overlay_active:
        return

    font_id = 0
    # simple HUD box
    bgl.glEnable(bgl.GL_BLEND)
    bgl.glColor4f(0.0, 0.0, 0.0, 0.7)
    
    # Bottom Left HUD
    x, y = 100, 100
    width, height = 600, 100
    
    # Modern Text
    blf.position(font_id, x + 20, y + 60, 0)
    blf.size(font_id, 24)
    blf.color(font_id, 0.2, 1.0, 0.6, 1.0) # Primary Green
    blf.draw(font_id, "NEURAL ASSISTANT (AltGr Active)")
    
    blf.position(font_id, x + 20, y + 20, 0)
    blf.size(font_id, 16)
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.draw(font_id, overlay_text)
    
    bgl.glDisable(bgl.GL_BLEND)

class CNC_OT_SmartTrigger(bpy.types.Operator):
    """
    The 'AltGr' Trigger.
    When held/tapped, it activates the Neural Listening Mode.
    """
    bl_idname = "cnc.smart_trigger"
    bl_label = "Smart Neural Trigger"
    bl_options = {'REGISTER', 'INTERNAL'}

    _handle = None

    def modal(self, context, event):
        global overlay_active, overlay_text
        
        # Tap AltGr (RIGHT_ALT) to toggle
        if event.type == 'RIGHT_ALT' and event.value == 'PRESS':
            overlay_active = not overlay_active
            if overlay_active:
                overlay_text = "Listening... (Type command or Hover UI)"
                self.report({'INFO'}, "Neural Assistant: ON")
            else:
                self.report({'INFO'}, "Neural Assistant: OFF")
            
            # Force redraw
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
            
        if overlay_active:
            # Context Aware Hover Logic
            if event.type == 'MOUSEMOVE':
                # In a real app, we'd raycast or check context.area.ui_type
                pass
            
            # Simple Quick Keys while active
            if event.type == 'G' and event.value == 'PRESS':
                overlay_text = "Generating Gear..."
                bpy.ops.cnc.generate_part(part_type='GEAR')
                context.area.tag_redraw()

            if event.type == 'B' and event.value == 'PRESS':
                overlay_text = "Generating Bracket..."
                bpy.ops.cnc.generate_part(part_type='BRACKET')
                context.area.tag_redraw()

            if event.type == 'ESC' and event.value == 'PRESS':
                overlay_active = False
                context.area.tag_redraw()
                
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            if self._handle is None:
                self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, (self, context), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run Neural Assistant")
            return {'CANCELLED'}

class CNC_OT_ExplainUI(bpy.types.Operator):
    bl_idname = "cnc.explain_ui"
    bl_label = "Explain This (SolidWorks)"
    
    def execute(self, context):
        # Mock Logic - In reality, would grab active button context
        self.report({'INFO'}, "Translating Blender Context to SolidWorks Terminology...")
        
        # Simulate Backend Call
        # resp = requests.post(API, json={"query": "modifier_boolean"})
        
        # Display feedback
        def show_popup(self, context):
            self.layout.label(text="Blender 'Boolean Modifier'")
            self.layout.label(text="Maps to: SolidWorks 'Combine/Subtract' Feature")
            self.layout.label(text="Usage: Non-destructive addition/subtraction.")

        context.window_manager.popup_menu(show_popup, title="Neural Explainer", icon='INFO')
        return {'FINISHED'}

class CNC_OT_NaturalCommand(bpy.types.Operator):
    bl_idname = "cnc.natural_command"
    bl_label = "Natural Language Command"
    
    prompt: bpy.props.StringProperty(name="Command", default="Make edges round")

    def execute(self, context):
        # Maps "Make edges round" -> Bevel
        if "round" in self.prompt.lower() or "fillet" in self.prompt.lower():
            bpy.ops.object.modifier_add(type='BEVEL')
            context.object.modifiers["Bevel"].width = 0.05
            self.report({'INFO'}, "Applied Bevel Modifier")
        
        elif "smooth" in self.prompt.lower():
             bpy.ops.object.shade_smooth()
             self.report({'INFO'}, "Applied Shade Smooth")
             
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
