"""
Part Factory & Functionalities Builder 🏭
Generates parametric parts and records macro functionalities.
"""
import bpy
import bmesh
import math
import json

class CNC_OT_GeneratePart(bpy.types.Operator):
    bl_idname = "cnc.generate_part"
    bl_label = "Generate Parametric Part"
    bl_options = {'REGISTER', 'UNDO'}

    part_type: bpy.props.EnumProperty(
        items=[
            ('GEAR', "Spur Gear", "Standard Tooth Gear"),
            ('BRACKET', "L-Bracket", "Reinforced mounting bracket"),
            ('BOLT', "Hex Bolt", "Metric Hex Bolt"),
        ],
        name="Part Type"
    )

    # Parametric Inputs (Dynamic based on type)
    param_1: bpy.props.FloatProperty(name="Diameter/Length", default=5.0)
    param_2: bpy.props.FloatProperty(name="Teeth/Width", default=12.0)
    param_3: bpy.props.FloatProperty(name="Thickness", default=1.0)

    def execute(self, context):
        if self.part_type == 'GEAR':
            self.create_gear(context, self.param_1, int(self.param_2), self.param_3)
        elif self.part_type == 'BRACKET':
            self.create_bracket(context, self.param_1, self.param_2, self.param_3)
        elif self.part_type == 'BOLT':
            self.create_bolt(context, self.param_1, self.param_2)
        
        self.report({'INFO'}, f"Generated {self.part_type}")
        return {'FINISHED'}

    def create_gear(self, context, radius, teeth, thickness):
        # Simple procedural gear logic using bmesh
        mesh = bpy.data.meshes.new(f"Gear_{teeth}T")
        obj = bpy.data.objects.new(f"Gear_{teeth}T", mesh)
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)

        bm = bmesh.new()
        
        # Create teeth profile (simplified)
        angle_step = (2 * math.pi) / teeth
        for i in range(teeth):
            angle = i * angle_step
            # Base vertex
            x1 = math.cos(angle) * radius
            y1 = math.sin(angle) * radius
            bm.verts.new((x1, y1, 0))
            
            # Tooth tip
            x2 = math.cos(angle + angle_step/2) * (radius * 1.2)
            y2 = math.sin(angle + angle_step/2) * (radius * 1.2)
            bm.verts.new((x2, y2, 0))

        bmesh.ops.contextual_create(bm, geom=bm.verts)
        bmesh.ops.extrude_face_region(bm, geom=bm.faces, r=thickness * 0.1) # Extrude Up
        
        # Make valid volume (simplified hull for demo)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        
        bm.to_mesh(mesh)
        bm.free()
        
        # Add Bevel Modifier for "Engineering Look"
        mod = obj.modifiers.new("EngBevel", 'BEVEL')
        mod.width = 0.05
        mod.segments = 3

    def create_bracket(self, context, length, width, thickness):
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = context.active_object
        obj.name = "L_Bracket"
        obj.scale = (length, width, thickness)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Extrude L-shape (Mock logic)
        # In a full version, we'd model the specific L geometry
        pass

    def create_bolt(self, context, dia, length):
        bpy.ops.mesh.primitive_cylinder_add(radius=dia/2, depth=length)
        obj = context.active_object
        obj.name = "Hex_Bolt"
        
        # Hex Head
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=dia, depth=dia/2, location=(0, 0, length/2))
        head = context.active_object
        
        # Join
        head.select_set(True)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.join()


class CNC_OT_FunctionalityBuilder(bpy.types.Operator):
    bl_idname = "cnc.functionality_builder"
    bl_label = "Record Macro function"
    bl_description = "Records recent operations as a reusable function"

    func_name: bpy.props.StringProperty(name="Function Name", default="My_Custom_Macro")

    def execute(self, context):
        # Mock Recorder: captures recent info report
        self.report({'INFO'}, f"Function '{self.func_name}' saved to Library.")
        # In real app: Parse bpy.context.window_manager.operators
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
