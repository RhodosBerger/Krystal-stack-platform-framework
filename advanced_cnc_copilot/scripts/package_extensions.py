import shutil
import os
import datetime

def package_blender_addon():
    source_dir = "blender_addon"
    output_filename = f"fanuc_rise_blender_addon_{datetime.date.today().isoformat()}"
    output_dir = "extensions"
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"📦 Packaging Blender Addon from '{source_dir}'...")
    
    try:
        shutil.make_archive(output_path, 'zip', source_dir)
        print(f"✅ Addon packaged successfully: {output_path}.zip")
    except Exception as e:
        print(f"❌ Failed to package addon: {e}")

if __name__ == "__main__":
    package_blender_addon()
