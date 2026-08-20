"""
Simulation Panel & Physics Setup ⚛️
UI for the Simulation Agent ("The Physicist").
Allows setting Forces, Constraints, and Materials.
"""
import bpy
from .client import API_URL
import json
import requests
import threading

class CNC_OT_ApplyForce(bpy.types.Operator):
    bl_idname = "cnc.apply_force"
    bl_label = "Apply Force"
    bl_description = "Apply a Force Vector to the selected face"

    magnitude: bpy.props.FloatProperty(name="Magnitude (N)", default=1000.0)
    axis: bpy.props.EnumProperty(
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        name="Axis", default='Z'
    )

    def execute(self, context):
        obj = context.active_object
        # In a real app, we'd attach a custom property to the selected face index
        # For now, we attach to the Object as a "Load Definition"
        
        load_data = {
            "type": "FORCE",
            "magnitude": self.magnitude,
            "axis": self.axis,
            "target": obj.name
        }
        
        if "CNC_Loads" not in obj:
            obj["CNC_Loads"] = []
            
        # Append as JSON string implementation detail for simplicity
        loads = obj.get("CNC_Loads").to_list() if hasattr(obj["CNC_Loads"], "to_list") else []
        loads.append(load_data)
        obj["CNC_Loads"] = loads
        
        self.report({'INFO'}, f"Applied {self.magnitude}N Force on {self.axis}")
        return {'FINISHED'}

class CNC_OT_RunSimulation(bpy.types.Operator):
    bl_idname = "cnc.run_simulation"
    bl_label = "Run Simulation"
    bl_description = "Send Mesh & Setup to Backend Simulation Agent"

    sim_type: bpy.props.EnumProperty(
        items=[
            ('STATIC_STRESS', "Static Stress", "Von Mises Stress Analysis"),
            ('THERMAL', "Thermal", "Heat Distribution"),
            ('MODAL', "Modal Frequencies", "Vibration Modes"),
        ],
        name="Type", default='STATIC_STRESS'
    )

    def execute(self, context):
        obj = context.active_object
        if not obj:
             self.report({'WARNING'}, "No Object Selected")
             return {'CANCELLED'}

        self.report({'INFO'}, f"Sending {obj.name} to Physicist...")
        
        # Async Call
        threading.Thread(target=self.run_async, args=(context, obj)).start()
        return {'FINISHED'}

    def run_async(self, context, obj):
        payload = {
            "type": self.sim_type,
            "mesh_name": obj.name,
            "material": "STEEL_S355", # Fixed for MVP
            "loads": obj.get("CNC_Loads", []),
            "constraints": [] # TODO: Add constraints UI
        }
        
        try:
            # Mocking the endpoint for now as we haven't exposed SimAgent in main.py yet
            # In production: requests.post(f"{API_URL}/simulation/run", json=payload)
            import time
            time.sleep(1) # Simulate Network
            
            # Mock Result
            result = {
                "status": "COMPLETED",
                "max_stress_pa": 120e6,
                "safety_factor": 2.1
            }
            
            print(f"Simulation Result: {result}")
            # Thread-safe UI update would require a queue/timer
            
        except Exception as e:
            print(f"Sim Failed: {e}")

class CNC_PT_Simulation_Panel(bpy.types.Panel):
    bl_label = "Physics & Simulation"
    bl_idname = "CNC_PT_Simulation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CNC Copilot'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        layout.label(text="Setup", icon='Physics')
        row = layout.row()
        row.operator("cnc.assign_material", text="Material")
        
        layout.label(text="Loads & Constraints", icon='FORCE_FORCE')
        col = layout.column(align=True)
        col.operator("cnc.apply_force", text="Add Force (N)")
        # col.operator("cnc.add_constraint", text="Fix Geometry") # Future

        layout.separator()
        
        layout.label(text="Solver", icon='MOD_PHYSICS')
        layout.prop(context.scene, "cnc_sim_quality", text="Quality")
        
        box = layout.box()
        box.scale_y = 1.5
        box.operator("cnc.run_simulation", text="SOLVE PHYSICS", icon='PLAY')
        
        if obj and "CNC_Loads" in obj:
            layout.label(text="Active Loads:")
            for load in obj["CNC_Loads"]:
                layout.label(text=f"{load['type']} {load['magnitude']}N", icon='SMALL_TRI_RIGHT_VEC')
