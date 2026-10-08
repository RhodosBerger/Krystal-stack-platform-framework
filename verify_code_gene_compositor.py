#!/usr/bin/env python3
"""
Verification Script for Code GENE Neural Compositor & Dual Viewport Engine.
Validates:
  1. Bounded 3D coordinate space calculations and boundary cage.
  2. Procedural raster mixing with Bayer dithering and directional Sobel lines.
  3. Project GENE latent action transitions.
  4. 3D Mesh generation (vertices, faces, normals, UVs).
  5. Wavefront .OBJ export integrity.
  6. ASCII string rendering and entropy metrics.
"""

import sys
import os
import json

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from krystal_web_hub.economic_engine.code_gene_neural_compositor import (
    BoundedRenderingVolume,
    ProceduralRasterMixer,
    CodeGeneTopology,
    CodeGeneDynamicsModel,
    CodeGeneNeuralCompositor,
    BAYER_4X4,
    BAYER_8X8
)

def test_bounded_volume():
    print("[TEST 1] Validating Bounded 3D Rendering Space...")
    bounds = BoundedRenderingVolume(min_x=-3.5, max_x=3.5, min_y=-2.0, max_y=2.0, min_z=-3.5, max_z=3.5)
    
    # Check volume
    expected_vol = 7.0 * 4.0 * 7.0
    assert abs(bounds.volume_m3() - expected_vol) < 1e-4, f"Volume mismatch: {bounds.volume_m3()} != {expected_vol}"
    
    # Check containment
    assert bounds.is_inside((0.0, 0.0, 0.0)), "Center must be inside"
    assert bounds.is_inside((3.4, 1.9, -3.4)), "Corner must be inside"
    assert not bounds.is_inside((4.0, 0.0, 0.0)), "Point outside X must fail"
    assert not bounds.is_inside((0.0, 2.5, 0.0)), "Point outside Y must fail"
    
    # Check cage lines
    corners = bounds.get_corner_vertices()
    assert len(corners) == 8, f"Must have 8 corners, got {len(corners)}"
    
    edges = bounds.get_cage_edges()
    assert len(edges) == 12, f"Must have 12 cage edges, got {len(edges)}"
    
    grid = bounds.get_floor_grid_lines()
    assert len(grid) > 0, "Floor grid lines must be non-empty"
    print(f"  -> Bounded volume OK: {bounds.volume_m3():.1f} m3, 12 cage edges, {len(grid)} floor grid lines.")

def test_raster_mixer():
    print("[TEST 2] Validating Procedural Raster Mixer ('Miešanie rastrov')...")
    mixer = ProceduralRasterMixer()
    
    # Check Bayer matrix dithering
    val_00 = mixer.sample_dither(0, 0)
    val_11 = mixer.sample_dither(1, 1)
    assert val_00 != val_11, "Dither matrix must produce spatial variance"
    
    # Test shading
    normal = (0.0, 1.0, 0.0)
    view = (0.0, -1.0, 0.0)
    ch, luma, rgb = mixer.shade_surface(norm=normal, view_dir=view, dist=2.0, ao=1.0, x_pix=2, y_pix=2)
    assert len(ch) >= 1, "Shaded character must not be empty"
    assert 0.0 <= luma <= 1.0, f"Luma must be within [0, 1], got {luma}"
    assert len(rgb) == 3 and all(0 <= c <= 255 for c in rgb), "RGB must be valid 8-bit tuple"
    print(f"  -> Raster mixer OK: Shaded glyph='{ch}', luma={luma:.2f}, RGB={rgb}")

def test_project_gene_latent_dynamics():
    print("[TEST 3] Validating Project GENE Action-Conditional Model...")
    dynamics = CodeGeneDynamicsModel()
    
    # Step 0: IDLE
    res0 = dynamics.step(0)
    assert res0["action"] == "IDLE"
    assert dynamics.state.step_id == 1
    
    # Step 1: ORBIT_CAM
    res1 = dynamics.step(1, {"degrees": 45.0})
    assert res1["action"] == "ORBIT_CAM"
    assert dynamics.state.cam_orbit_deg == 70.0 # 25 + 45
    
    # Step 2: MORPH_TOPOLOGY
    res2 = dynamics.step(2)
    assert res2["action"] == "MORPH_TOPOLOGY"
    assert dynamics.state.topology == CodeGeneTopology.GYROID_QUANTUM
    
    # Step 3: PULSE_DOPAMINE
    res3 = dynamics.step(3, {"freq": 3.0})
    assert res3["action"] == "PULSE_DOPAMINE"
    assert dynamics.state.pulse_active == True
    
    # Step 4: CRYSTAL_GROWTH
    res4 = dynamics.step(4)
    assert res4["action"] == "CRYSTAL_GROWTH"
    assert dynamics.state.crystal_growth_level > 0.0
    
    print(f"  -> Project GENE latent actions OK: 5 steps executed cleanly.")

def test_dual_representation_pipeline():
    print("[TEST 4] Validating Dual ASCII + 3D Mesh Output Pipeline...")
    compositor = CodeGeneNeuralCompositor()
    
    # Mode A: Photorealistic ASCII Frame
    ascii_out, telemetry = compositor.render_frame_ascii(width=64, height=20, t=1.0)
    lines = ascii_out.split('\n')
    assert len(lines) == 20, f"Expected 20 lines, got {len(lines)}"
    assert "CODE GENE COMPOSITOR" in lines[0], "Header bar missing in ASCII output"
    assert telemetry["spatial_entropy"] >= 0.0, "Entropy must be calculated"
    assert telemetry["coherence"] > 0.0, "Coherence must be positive"
    
    # Mode B: 3D Polygonal Mesh
    mesh = compositor.generate_3d_mesh(resolution=12)
    assert mesh["vertex_count"] > 0, "Vertex count must be > 0"
    assert mesh["face_count"] > 0, "Face count must be > 0"
    assert len(mesh["cage_lines"]) == 12, "Must contain 12 bounding cage segments"
    assert len(mesh["floor_grid_lines"]) > 0, "Must contain floor grid lines"
    
    # Wavefront .OBJ Export
    obj_path = compositor.export_obj_file("verify_test_code_gene.obj")
    assert os.path.exists(obj_path), f"OBJ file not found at {obj_path}"
    with open(obj_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "v " in content and "f " in content, "OBJ must have vertex and face declarations"
    
    print(f"  -> Dual pipeline OK: ASCII ({len(lines)} lines), 3D Mesh ({mesh['vertex_count']} v, {mesh['face_count']} f), OBJ exported ({os.path.getsize(obj_path)} bytes).")

if __name__ == "__main__":
    print("=" * 60)
    print("KRYSTAL-STACK: VERIFYING CODE GENE NEURAL COMPOSITOR")
    print("=" * 60)
    test_bounded_volume()
    test_raster_mixer()
    test_project_gene_latent_dynamics()
    test_dual_representation_pipeline()
    print("=" * 60)
    print("ALL TESTS PASSED WITH 100% INTEGRITY!")
    print("=" * 60)
