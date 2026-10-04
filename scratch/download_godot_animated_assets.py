import os
import urllib.request
import json

TARGET_DIR = r"c:\Users\dusan\Documents\GitHub\Krystal-stack-platform-framework\godot_project\assets\models"
os.makedirs(TARGET_DIR, exist_ok=True)

ASSETS = [
    {
        "id": "robot_expressive",
        "filename": "RobotExpressive.glb",
        "url": "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/RobotExpressive/RobotExpressive.glb",
        "name": "Krystal Cybernetic Robot Expressive",
        "category": "Character",
        "animations": ["Idle", "Walking", "Running", "Dance", "Death", "Sitting", "Standing", "Jump", "Yes", "No", "Wave", "Punch", "ThumbsUp"],
        "description": "Rigged humanoid robot with 13 rich skeletal animation clips, ideal for Godot 4.x 3D character controllers."
    },
    {
        "id": "cesium_man",
        "filename": "CesiumMan.glb",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/CesiumMan/glTF-Binary/CesiumMan.glb",
        "name": "Cesium Humanoid Explorer",
        "category": "Character",
        "animations": ["Anim_0_Walk"],
        "description": "Rigged humanoid walker with standard bone hierarchy and walking cycle animation."
    },
    {
        "id": "fox_quadruped",
        "filename": "Fox.glb",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Fox/glTF-Binary/Fox.glb",
        "name": "Sovereign Fox Quadruped",
        "category": "Creature",
        "animations": ["Survey", "Walk", "Run"],
        "description": "Rigged quadruped mammal with Survey (idle alert), Walk, and Run animation cycles."
    }
]

def download_and_verify():
    manifest = []
    print("=== Starting Godot 3D Animated Models Download ===")
    for asset in ASSETS:
        dest_path = os.path.join(TARGET_DIR, asset["filename"])
        print(f"Downloading {asset['name']} from {asset['url']}...")
        req = urllib.request.Request(asset["url"], headers={"User-Agent": "Krystal-Godot-PackageManager/1.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            print(f" -> Saved {asset['filename']} ({len(content):,} bytes) at {dest_path}")
            
        manifest.append({
            "id": asset["id"],
            "filename": asset["filename"],
            "local_path": f"res://assets/models/{asset['filename']}",
            "file_size_bytes": len(content),
            "name": asset["name"],
            "category": asset["category"],
            "animations": asset["animations"],
            "description": asset["description"]
        })
        
    manifest_path = os.path.join(TARGET_DIR, "models_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"vital_max_hp_rule": 6, "models": manifest}, f, indent=2)
    print(f"=== Successfully downloaded {len(manifest)} models and wrote manifest to {manifest_path} ===")

if __name__ == "__main__":
    download_and_verify()
