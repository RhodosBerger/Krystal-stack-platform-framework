# Spustenie: blender -b -P krystal_progressive_blender_mcp_job.py
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"C:\Users\dusan\Documents\GitHub\Krystal-stack-platform-framework\artifacts\cad_and_blender_production\krystal_projektant_mesh.obj")
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
for samples in [4, 16, 64, 256]:
    scene.cycles.samples = samples
    scene.render.filepath = rf"C:\Users\dusan\Documents\GitHub\Krystal-stack-platform-framework\artifacts\cad_and_blender_production\progressive_pass_{samples:03d}.png"
    bpy.ops.render.render(write_still=True)
