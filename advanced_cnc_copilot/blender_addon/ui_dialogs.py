"""
Auxiliary Dialogs & Independent Prompt Triggers 💬
This module implements the "Persona" interfaces for independent evaluation.
"""
import bpy
import requests
import json
import threading

API_URL = "http://localhost:8000/api/manufacturing/request"

class CNC_OT_BaseDialog(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}
    
    prompt_persona: str = "Assistant"
    
    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def send_to_llm(self, prompt, payload):
        """Generic Async Sender"""
        full_payload = {
            "type": "CONSULTATION",
            "question": f"[{self.prompt_persona}] {prompt}",
            "context": payload
        }
        
        def _thread_target():
            try:
                # Mock Auth
                resp = requests.post(API_URL, json=full_payload)
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"[{self.prompt_persona}] Says: {data}")
                    # In real app: Update a specific UI StringProperty
            except Exception as e:
                print(f"Error: {e}")
                
        t = threading.Thread(target=_thread_target)
        t.start()

# --- Dialog 1: Stress Test (The Physicist) ---
class CNC_OT_AnalyzeStress(CNC_OT_BaseDialog):
    bl_idname = "cnc.analyze_stress"
    bl_label = "Stress Test Analysis"
    prompt_persona = "Physics Engine"

    load_case: bpy.props.EnumProperty(
        items=[
            ('GRAVITY', "Gravity Only", "Self-weight analysis"),
            ('FORCE_1KN', "Force 1kN", "Apply 1000N vertical load"),
            ('THERMAL', "Thermal Shock", "Heat from 20C to 200C"),
        ],
        name="Load Case"
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Configure Physics Simulation:", icon='PHYSICS')
        layout.prop(self, "load_case")
        layout.label(text="Model: " + context.active_object.name)
        
    def execute(self, context):
        msg = f"Running {self.load_case} on {context.active_object.name}..."
        print(msg)
        self.report({'INFO'}, msg)
        
        # Trigger LLM Evaluation
        payload = {"object_name": context.active_object.name, "load": self.load_case}
        self.send_to_llm(f"Evaluate structural integrity under {self.load_case}", payload)
        
        return {'FINISHED'}

# --- Dialog 2: Manufacturing Auditor (The Machinist) ---
class CNC_OT_AuditManufacturing(CNC_OT_BaseDialog):
    bl_idname = "cnc.audit_manufacturing"
    bl_label = "Manufacturing Audit"
    prompt_persona = "Master Machinist"

    machine_type: bpy.props.EnumProperty(
        items=[
            ('MILL_3AXIS', "3-Axis Mill", "Standard Vertical Mill"),
            ('MILL_5AXIS', "5-Axis Mill", "Complex Geometry"),
            ('LATHE', "Lathe", "Cylindrical Parts"),
        ],
        name="Target Machine"
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Audit for Machinability:", icon='MODIFIER')
        layout.prop(self, "machine_type")
        
    def execute(self, context):
        self.report({'INFO'}, f"Auditing for {self.machine_type}...")
        payload = {"object_name": context.active_object.name, "machine": self.machine_type}
        self.send_to_llm(f"Check for undercuts and tool access for {self.machine_type}", payload)
        return {'FINISHED'}

# --- Dialog 3: Cost Estimator (The Accountant) ---
class CNC_OT_EstimateCost(CNC_OT_BaseDialog):
    bl_idname = "cnc.estimate_cost"
    bl_label = "Cost Estimation"
    prompt_persona = "Cost Estimator"

    material: bpy.props.EnumProperty(
        items=[
            ('AL6061', "Aluminum 6061", "$5/kg"),
            ('SS304', "Stainless Steel 304", "$15/kg"),
            ('TI6AL4V', "Titanium Gr5", "$80/kg"),
        ],
        name="Material"
    )
    
    quantity: bpy.props.IntProperty(name="Batch Size", default=1, min=1)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Generate Quote:", icon='FUND')
        layout.prop(self, "material")
        layout.prop(self, "quantity")
        
    def execute(self, context):
        self.report({'INFO'}, "Calculating Estimate...")
        payload = {"material": self.material, "batch_size": self.quantity}
        self.send_to_llm(f"Calculate cost for {self.quantity} units in {self.material}", payload)
        return {'FINISHED'}
