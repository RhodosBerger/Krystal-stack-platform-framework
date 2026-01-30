"""
Render Studio & Cycle Adaptation 📸
Automates engineering visualization with pre-set lighting and materials.
"""
import bpy

class CNC_OT_SetupStudio(bpy.types.Operator):
    bl_idname = "cnc.setup_studio"
    bl_label = "Auto-Setup Studio"
    bl_description = "Sets up 3-Point Lighting and Backdrop"

    style: bpy.props.EnumProperty(
        items=[
            ('ENGINEERING', "Clean Engineering", "White background, neutral light"),
            ('DRAMATIC', "Product Reveal", "Dark background, rim lighting"),
            ('BLUEPRINT', "Blueprint Mode", "Wireframe on blue"),
            ('ISO_FRONT', "ISO: Front", "Standard Front Orthographic"),
            ('ISO_TOP', "ISO: Top", "Standard Top Orthographic"),
            ('ISO_30', "ISO: Isometric 30", "Standard 30° Isometric"),
        ],
        name="Style"
    )

    def execute(self, context):
        # Switch to Cycles
        context.scene.render.engine = 'CYCLES'
        
        # Clear existing lights
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()

        if self.style == 'ENGINEERING':
            self.setup_engineering()
        elif self.style == 'DRAMATIC':
            self.setup_dramatic()
        elif self.style.startswith('ISO_'):
            self.setup_iso_standard()
        
        self.report({'INFO'}, f"Studio setup: {self.style}")
        return {'FINISHED'}

    def setup_iso_standard(self):
        # Strict 5000K Neutral Lighting
        bpy.ops.object.light_add(type='SUN', rotation=(0.785, 0, 0.785))
        sun = bpy.context.object
        sun.data.energy = 5.0
        sun.data.use_contact_shadow = True
        
        # Set Camera to Orthographic
        cam = bpy.context.scene.camera
        if cam:
            cam.data.type = 'ORTHO'
            if self.style == 'ISO_FRONT':
                cam.location = (0, -10, 0)
                cam.rotation_euler = (1.57, 0, 0)
            elif self.style == 'ISO_TOP':
                cam.location = (0, 0, 10)
                cam.rotation_euler = (0, 0, 0)
            elif self.style == 'ISO_30':
                cam.location = (10, -10, 10)
                cam.rotation_euler = (0.957, 0, 0.785) # ~35.26 and 45 deg

        # Pure White Background
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

    def setup_engineering(self):
        # Key Light
        bpy.ops.object.light_add(type='AREA', location=(5, -5, 10))
        key = bpy.context.object
        key.data.energy = 1000
        
        # Fill Light
        bpy.ops.object.light_add(type='AREA', location=(-5, -2, 5))
        fill = bpy.context.object
        fill.data.energy = 500
        
        # Back Light
        bpy.ops.object.light_add(type='SPOT', location=(0, 5, 8))
        back = bpy.context.object
        back.data.energy = 800

        # White World
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

    def setup_dramatic(self):
        # Rim Light Strong
        bpy.ops.object.light_add(type='SPOT', location=(0, 5, 5))
        rim = bpy.context.object
        rim.data.energy = 2000
        rim.data.color = (0.2, 0.5, 1.0) # Blue tint
        
        # Dark World
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.05, 1)

class CNC_OT_AssignMaterial(bpy.types.Operator):
    bl_idname = "cnc.assign_material"
    bl_label = "Assign Engineering Material"

    material_type: bpy.props.EnumProperty(
        items=[
            ('STEEL_SATIN', "Satin Steel", "Brushed metal look"),
            ('PLASTIC_ABS', "ABS Red", "Rough plastic"),
            ('CARBON_FIBER', "Carbon Fiber", "Procedural pattern"),
        ],
        name="Material"
    )

    def execute(self, context):
        obj = context.active_object
        if not obj:
            return {'CANCELLED'}
        
        mat_name = f"CNC_{self.material_type}"
        mat = bpy.data.materials.get(mat_name)
        
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            
            if self.material_type == 'STEEL_SATIN':
                bsdf.inputs['Base Color'].default_value = (0.6, 0.6, 0.65, 1)
                bsdf.inputs['Metallic'].default_value = 1.0
                bsdf.inputs['Roughness'].default_value = 0.35
            
            elif self.material_type == 'PLASTIC_ABS':
                bsdf.inputs['Base Color'].default_value = (0.8, 0.1, 0.1, 1)
                bsdf.inputs['Roughness'].default_value = 0.6
                
        # Assign
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
            
        return {'FINISHED'}
