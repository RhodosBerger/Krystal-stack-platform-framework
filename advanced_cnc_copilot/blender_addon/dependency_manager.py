"""
Ecosystem Manager & Dependency Fetcher 📦
Ensures the 'Open Source Puzzle' is complete by managing external dependencies.
Capabilities:
1. Pip Install (libs like scipy, shapely)
2. Add-on Fetcher (Git Clone/Zip Install)
3. Health Check
"""
import bpy
import sys
import subprocess
import os
import shutil
import urllib.request
import zipfile
import threading

class DependencyManager:
    @staticmethod
    def get_python_binary():
        """Returns the path to Blender's embedded Python binary."""
        return sys.executable

    @staticmethod
    def install_pip_package(package_name):
        python_exe = DependencyManager.get_python_binary()
        try:
            subprocess.check_call([python_exe, "-m", "pip", "install", package_name])
            return True, f"Installed {package_name}"
        except subprocess.CalledProcessError as e:
            return False, f"Failed to install {package_name}: {str(e)}"

    @staticmethod
    def fetch_addon_from_url(url, addon_name):
        """Downloads and installs a zipped addon."""
        try:
            # 1. Download
            temp_zip = os.path.join(bpy.app.tempdir, f"{addon_name}.zip")
            urllib.request.urlretrieve(url, temp_zip)
            
            # 2. Install (Blender API)
            bpy.ops.preferences.addon_install(filepath=temp_zip)
            
            # 3. Enable
            bpy.ops.preferences.addon_enable(module=addon_name)
            
            return True, f"Fetched & Enabled {addon_name}"
        except Exception as e:
            return False, f"Fetch Error: {str(e)}"

class CNC_OT_InstallLib(bpy.types.Operator):
    bl_idname = "cnc.install_lib"
    bl_label = "Install Python Library"
    bl_description = "Pip installs a library into Blender's Python"

    library_name: bpy.props.StringProperty(name="Library Name", default="scipy")

    def execute(self, context):
        self.report({'INFO'}, f"Installing {self.library_name}...")
        
        # Run in thread to not freeze UI
        threading.Thread(target=self.run_install, args=(self.library_name,)).start()
        return {'FINISHED'}

    def run_install(self, lib_name):
        success, msg = DependencyManager.install_pip_package(lib_name)
        print(f"[{'SUCCESS' if success else 'FAIL'}] {msg}")
        # Note: Reporting from thread to UI is tricky in Blender, printing to console for now

class CNC_OT_FetchAddon(bpy.types.Operator):
    bl_idname = "cnc.fetch_addon"
    bl_label = "Fetch Community Add-on"
    bl_description = "Downloads and installs known 'Puzzle Pieces'"

    addon_preset: bpy.props.EnumProperty(
        items=[
            ('SVERCHOK', "Sverchok (Parametric)", "https://github.com/nortikin/sverchok/archive/refs/heads/master.zip"),
            ('BLENDERCAM', "BlenderCAM (Manufacturing)", "https://github.com/BlenderCAM/BlenderCAM/archive/refs/heads/master.zip"),
            ('MEASUREIT', "MeasureIt_ARCH (Dimensions)", "https://github.com/Antonioya/MeasureIt_ARCH/archive/refs/heads/master.zip"),
        ],
        name="Add-on"
    )

    def execute(self, context):
        url = ""
        name = ""
        
        # Map Enum to URL (Simple Logic)
        if self.addon_preset == 'SVERCHOK':
            url = "https://github.com/nortikin/sverchok/archive/refs/heads/master.zip"
            name = "sverchok-master" 
        elif self.addon_preset == 'BLENDERCAM':
            url = "https://github.com/BlenderCAM/BlenderCAM/archive/refs/heads/master.zip"
            name = "BlenderCAM-master"
        elif self.addon_preset == 'MEASUREIT':
            url = "https://github.com/Antonioya/MeasureIt_ARCH/archive/refs/heads/master.zip"
            name = "MeasureIt_ARCH-master"

        self.report({'INFO'}, f"Fetching {name}...")
        threading.Thread(target=self.run_fetch, args=(url, name)).start()
        return {'FINISHED'}

    def run_fetch(self, url, name):
        success, msg = DependencyManager.fetch_addon_from_url(url, name)
        print(f"[{'SUCCESS' if success else 'FAIL'}] {msg}")

class CNC_PT_Ecosystem_Panel(bpy.types.Panel):
    bl_label = "Ecosystem Manager"
    bl_idname = "CNC_PT_Ecosystem"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CNC Copilot'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Python Libraries", icon='PYTHON')
        row = layout.row(align=True)
        row.operator("cnc.install_lib", text="Install scipy").library_name = "scipy"
        row.operator("cnc.install_lib", text="Install shapely").library_name = "shapely"
        
        layout.separator()
        
        layout.label(text="Community Puzzle", icon='COMMUNITY')
        layout.operator("cnc.fetch_addon", text="Get Sverchok").addon_preset = 'SVERCHOK'
        layout.operator("cnc.fetch_addon", text="Get BlenderCAM").addon_preset = 'BLENDERCAM'
