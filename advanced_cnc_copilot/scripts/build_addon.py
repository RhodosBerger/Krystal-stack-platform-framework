"""
Solution Packager (Base 44) 📦
Packages the 'cnc_copilot' Blender Add-on into a installable ZIP file.
Automatically excludes __pycache__, git, and test files.
"""
import os
import zipfile
import shutil
from datetime import datetime

# Configuration
ADDON_NAME = "cnc_copilot"
SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "advanced_cnc_copilot", "blender_addon")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "builds")
VERSION = "0.9.0_Base44"

def build_addon():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_filename = f"{ADDON_NAME}_v{VERSION}_{timestamp}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    
    print(f"📦 Packaging {ADDON_NAME} from {SOURCE_DIR}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk the directory
        for root, dirs, files in os.walk(SOURCE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Relative path for archive (must start with addon name folder)
                rel_path = os.path.relpath(file_path, os.path.dirname(SOURCE_DIR))
                
                # Filter exclusions
                if "__pycache__" in rel_path: continue
                if ".git" in rel_path: continue
                if ".DS_Store" in rel_path: continue
                
                print(f"  Adding: {rel_path}")
                zipf.write(file_path, rel_path)
                
    print(f"\n✅ Build Complete: {zip_path}")
    print(f"👉 Install this file in Blender -> Preferences -> Add-ons -> Install...")

if __name__ == "__main__":
    build_addon()
