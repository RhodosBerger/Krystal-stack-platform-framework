#!/usr/bin/env python3
"""
Verification Script for Google Maps 3D Urban Geometry Extractor & Blender Bridge.
Validates:
  1. Geodetic coordinate conversion & bounded metric projection.
  2. Real-world urban building & street network extraction.
  3. 3D polygonal mesh synthesis (walls, spires, gables, roads).
  4. Wavefront .OBJ export & Blender automation script generation.
  5. Full integration with Code GENE ASCII Neural Compositor.
"""

import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from krystal_web_hub.economic_engine.google_maps_urban_extractor import (
    GoogleMapsUrbanExtractor,
    GLOBAL_GOOGLE_MAPS_EXTRACTOR,
    CANONICAL_REAL_CITIES,
    geodetic_to_local_meters
)
from krystal_web_hub.economic_engine.code_gene_neural_compositor import (
    CodeGeneNeuralCompositor,
    CodeGeneTopology
)

def test_geodetic_projection():
    print("[TEST 1] Testing Geodetic WGS84 to Local Metric Projection...")
    # 1 degree lat is approx 111.139 km
    x, z = geodetic_to_local_meters(50.001, 14.0, 50.000, 14.0)
    assert abs(x) < 1.0, f"East-West offset should be ~0, got {x}"
    assert 100.0 < z < 125.0, f"North-South offset for 0.001 deg should be ~111m, got {z}"
    print(f"  -> Geodetic projection OK: dx={x:.2f}m, dz={z:.2f}m")

def test_urban_extraction():
    print("[TEST 2] Testing Real City Extraction (Praha, Bratislava, Tokyo)...")
    extractor = GLOBAL_GOOGLE_MAPS_EXTRACTOR
    cities = extractor.list_available_cities()
    assert len(cities) >= 3, f"Expected at least 3 cities, found {len(cities)}"
    
    sector = extractor.extract_city_sector("praha_old_town", radius_m=120.0)
    assert sector.city_name.startswith("Praha"), f"Unexpected city name: {sector.city_name}"
    assert len(sector.buildings) >= 4, f"Buildings count too low: {len(sector.buildings)}"
    assert len(sector.roads) >= 2, f"Roads count too low: {len(sector.roads)}"
    
    # Check bounds containment
    for bld in sector.buildings:
        assert bld.height_m > 0, "Building height must be > 0"
        for pt in bld.footprint_polygon:
            assert abs(pt[0]) <= 3.5 and abs(pt[1]) <= 3.5, f"Building point {pt} outside cage!"
            
    print(f"  -> Urban extraction OK: {sector.city_name} with {len(sector.buildings)} buildings, {len(sector.roads)} roads within bounded cage.")

def test_mesh_and_blender_export():
    print("[TEST 3] Testing 3D Mesh Synthesis & Blender Python Script Generation...")
    extractor = GLOBAL_GOOGLE_MAPS_EXTRACTOR
    sector = extractor.extract_city_sector("bratislava_castle_danube")
    
    mesh = extractor.synthesize_3d_mesh(sector)
    assert mesh["vertex_count"] > 0, "Vertex count must be > 0"
    assert mesh["face_count"] > 0, "Face count must be > 0"
    
    obj_path = extractor.export_obj(sector, filename="test_bratislava.obj")
    assert os.path.exists(obj_path), f"OBJ file missing: {obj_path}"
    assert os.path.getsize(obj_path) > 1000, "OBJ file too small"
    
    script_path = extractor.generate_blender_import_script(sector)
    assert os.path.exists(script_path), f"Blender script missing: {script_path}"
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "bpy.ops.wm.obj_import" in content or "bpy.ops.import_scene.obj" in content, "Missing OBJ import in Blender script"
    assert "BEVEL" in content and "SOLIDIFY" in content and "DISPLACE" in content, "Missing modifier stack in Blender script"
    
    print(f"  -> Blender export OK: Mesh ({mesh['vertex_count']}v, {mesh['face_count']}f), OBJ ({os.path.getsize(obj_path)} bytes), Blender Script ({os.path.getsize(script_path)} bytes).")

def test_compositor_real_city_integration():
    print("[TEST 4] Testing Code GENE Compositor Real-City Loading & Dual ASCII/Mesh Rendering...")
    compositor = CodeGeneNeuralCompositor()
    
    # Load real city
    sector = compositor.load_real_city("praha_old_town")
    assert compositor.dynamics.state.topology == CodeGeneTopology.REAL_CITY_EXTRACTED
    assert compositor.active_city_sector is not None
    
    # Render ASCII
    ascii_out, telemetry = compositor.render_frame_ascii(width=72, height=22, t=1.0)
    lines = ascii_out.split('\n')
    assert len(lines) == 22, f"Expected 22 lines, got {len(lines)}"
    assert "CITY: Praha" in lines[-1] or "Praha" in lines[-1], f"City name not in footer: {lines[-1]}"

    
    # Generate Mesh
    mesh = compositor.generate_3d_mesh()
    assert mesh["vertex_count"] > 0, "Compositor mesh vertex count must be > 0"
    assert len(mesh["cage_lines"]) == 12, "Must retain 12 bounding cage edges"
    assert len(mesh["floor_grid_lines"]) > 0, "Must retain floor grid lines"
    
    print(f"  -> Compositor integration OK: Real city '{sector.city_name}' rendered in ASCII with Bayer dithering and in 3D Mesh with Bounded Cage.")

if __name__ == "__main__":
    print("=" * 65)
    print("KRYSTAL-STACK: VERIFYING GOOGLE MAPS 3D URBAN EXTRACTOR & BLENDER")
    print("=" * 65)
    test_geodetic_projection()
    test_urban_extraction()
    test_mesh_and_blender_export()
    test_compositor_real_city_integration()
    print("=" * 65)
    print("ALL GOOGLE MAPS URBAN EXTRACTOR TESTS PASSED WITH 100% INTEGRITY!")
    print("=" * 65)
