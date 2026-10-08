# ====================================================================
# KRYSTAL-STACK: BLENDER GOOGLE MAPS 3D GEOMETRY IMPORT ADDON / SCRIPT
# ====================================================================
# Run in Blender: blender --background --python blender_import_google_maps.py
# Or load in Blender Scripting workspace and click 'Run Script'.
# ====================================================================

import bpy
import os

print("[Krystal Blender Bridge] Importing Google Maps Urban Geometry: Bratislava - Hradný Vrch & Podhradie...")

# 1. Clean existing default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 2. Import Wavefront .OBJ
obj_path = os.path.join(r"C:\Users\dusan\Documents\GitHub\Krystal-stack-platform-framework\krystal_web_hub\economic_engine\..\godot_assets\urban_cache", "google_maps_bratislava_castle_danube.obj")
if os.path.exists(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)
    city_obj = bpy.context.selected_objects[0]
    city_obj.name = "Bratislava - Hradný Vrch & Podhradie_Mesh"

    # 3. Apply Blender Modifier Stack for Real-World Tactility
    print("[Krystal Blender Bridge] Applying Blender Modifier Stack...")
    
    # Modifier A: Bevel (Softens sharp architectural corners)
    bev = city_obj.modifiers.new(name="Urban_Bevel", type='BEVEL')
    bev.width = 0.05
    bev.segments = 2
    bev.limit_method = 'ANGLE'
    bev.angle_limit = 0.523599 # 30 degrees

    # Modifier B: Solidify (Ensures water-tight walls)
    sol = city_obj.modifiers.new(name="Urban_Solidify", type='SOLIDIFY')
    sol.thickness = 0.08

    # Modifier C: Displace (Procedural stone/brick micro-texture)
    tex = bpy.data.textures.new("MasonryClouds", type='CLOUDS')
    tex.noise_scale = 0.4
    disp = city_obj.modifiers.new(name="Urban_Masonry_Displace", type='DISPLACE')
    disp.texture = tex
    disp.strength = 0.02

    # 4. Setup Sun and Sky Lighting
    light_data = bpy.data.lights.new(name="BohemianSun", type='SUN')
    light_data.energy = 3.5
    light_obj = bpy.data.objects.new(name="Sun_Light", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (15, 20, 25)

    print(f"[Krystal Blender Bridge] Successfully created city 'Bratislava - Hradný Vrch & Podhradie' with 4 buildings.")
else:
    print(f"[ERROR] OBJ file not found: {obj_path}")
