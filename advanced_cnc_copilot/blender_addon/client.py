import bpy
import requests
import json
import threading

# Configuration
API_URL = "http://localhost:8000/api/manufacturing/request"

class CNC_OT_CheckManufacturability(bpy.types.Operator):
    """Send Geometry to Neural CAD Bridge"""
    bl_idname = "cnc.check_manufacturability"
    bl_label = "Check Manufacturability"
    bl_description = "Analyze geometry for CNC feasibility using Neural Bridge"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Please select a Mesh object")
            return {'CANCELLED'}

        # 1. Extract Geometry (Vertices/Faces) - Simplified Voxelization
        mesh = obj.data
        verts = [v.co.__repr__() for v in mesh.vertices]
        faces = [f.vertices[:] for f in mesh.polygons]
        
        payload = {
            "type": "ANALYZE_GEOMETRY",
            "payload": {
                "vertex_count": len(verts),
                "face_count": len(faces),
                "dimensions": [d for d in obj.dimensions],
                "name": obj.name,
                # "geometry_blob": "..." # In real implementation, send GLB or Voxel Grid
            }
        }

        # 2. Async Call to Backend
        # Using a thread to avoid freezing Blender UI
        t = threading.Thread(target=self._send_request, args=(context, payload))
        t.start()

        self.report({'INFO'}, "Sent to Neural Bridge...")
        return {'FINISHED'}

    def _send_request(self, context, payload):
        try:
            # For now, mocking auth or assuming dev environment
            response = requests.post(API_URL, json=payload) #, headers={"Authorization": ...})
            
            if response.status_code == 200:
                data = response.json()
                # Schedule UI update in main thread
                # Note: In real Blender add-ons, use a queue or timer to update context safely
                print(f"Neural Bridge Response: {data}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Connection Failed: {e}")
