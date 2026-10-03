# ==============================================================================
# KRYSTAL-STACK: GEMINI 3D PROCEDURAL MODELER (.OBJ EXPORTER)
# ==============================================================================
# Mathematically models native 3D assets (.obj) for Godot and WebGL.
# Bypasses Blender by procedurally calculating vertices, faces, normals & UVs.
# Covers all 3 tribes of "Poslední Kmen":
#   1. Kryštálový Kmeň (Severní Štíty) - Hex Tiles, Shards, Crystal Shields
#   2. Jedovatý Kmeň (Pustina)         - Acid Slime Pools, Toxic Spore Towers
#   3. Druidi (Hlboký Les)             - Earth Roots, Ancient Runestones/Altars
# ==============================================================================

import os
import math

ASSET_DIR = os.path.join(os.path.dirname(__file__), 'godot_assets')
if not os.path.exists(ASSET_DIR):
    os.makedirs(ASSET_DIR)

def export_obj(filename, vertices, faces, normals=None):
    """Writes a standard Wavefront .obj file with optional normals."""
    filepath = os.path.join(ASSET_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Krystal-Stack Gemini Generated 3D Asset: {filename}\n")
        f.write("o ProceduralMesh\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        if normals:
            for vn in normals:
                f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
                
        for face in faces:
            # OBJ faces are 1-indexed
            if normals and len(normals) == len(vertices):
                face_str = " ".join([f"{idx + 1}//{idx + 1}" for idx in face])
            else:
                face_str = " ".join([str(idx + 1) for idx in face])
            f.write(f"f {face_str}\n")
            
    print(f"[Gemini 3D Modeler] Successfully exported: {filepath} ({len(vertices)} v, {len(faces)} f)")
    return filepath

# ------------------------------------------------------------------------------
# 1. HEX GAME BOARD TILE
# ------------------------------------------------------------------------------
def generate_hex_tile_obj():
    """Models a 3D beveled Hexagonal Tile for the Poslední Kmen tactical grid."""
    size = 1.0
    bevel_size = 0.90
    h_top = 0.45
    h_bevel = 0.50
    h_bot = 0.0
    vertices = []
    
    # 0..5: Outer Top Rim
    for i in range(6):
        a = math.radians(60 * i - 30)
        vertices.append((size * math.cos(a), h_top, size * math.sin(a)))
        
    # 6..11: Inner Top Bevel
    for i in range(6):
        a = math.radians(60 * i - 30)
        vertices.append((bevel_size * math.cos(a), h_bevel, bevel_size * math.sin(a)))
        
    # 12..17: Bottom Hexagon
    for i in range(6):
        a = math.radians(60 * i - 30)
        vertices.append((size * math.cos(a), h_bot, size * math.sin(a)))
        
    # 18: Center Top Socket
    vertices.append((0.0, h_bevel - 0.04, 0.0))
    # 19: Center Bottom
    vertices.append((0.0, h_bot, 0.0))

    faces = []
    # Inner top beveled surface
    for i in range(6):
        ni = (i + 1) % 6
        faces.append((6 + i, 6 + ni, 18))
        
    # Bevel chamfer ring
    for i in range(6):
        ni = (i + 1) % 6
        faces.append((i, 6 + i, 6 + ni))
        faces.append((i, 6 + ni, ni))
        
    # Side walls
    for i in range(6):
        ni = (i + 1) % 6
        faces.append((i, 12 + i, 12 + ni))
        faces.append((i, 12 + ni, ni))
        
    # Bottom cap
    for i in range(6):
        ni = (i + 1) % 6
        faces.append((12 + ni, 12 + i, 19))

    return export_obj("hex_tile.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 2. KRYŠTÁLOVÝ KMEŇ: CRYSTAL SHARD CLUSTER
# ------------------------------------------------------------------------------
def generate_crystal_shard_obj():
    """Models a 3D faceted Crystal Cluster (Main Spire + 3 orbiting shards)."""
    vertices = []
    faces = []
    
    def add_crystal_spire(center, h_top, h_bot, rad, rot_deg=0):
        cx, cy, cz = center
        base_idx = len(vertices)
        # Top apex
        vertices.append((cx, cy + h_top, cz))
        # Bottom apex
        vertices.append((cx, cy - h_bot, cz))
        # Middle 6-ring
        for i in range(6):
            a = math.radians(60 * i + rot_deg)
            vertices.append((cx + rad * math.cos(a), cy + (h_top - h_bot) * 0.15, cz + rad * math.sin(a)))
            
        top_idx = base_idx
        bot_idx = base_idx + 1
        ring_start = base_idx + 2
        
        for i in range(6):
            ni = (i + 1) % 6
            c_curr = ring_start + i
            c_next = ring_start + ni
            faces.append((top_idx, c_curr, c_next))
            faces.append((bot_idx, c_next, c_curr))
            
    # Main Central Crystal
    add_crystal_spire((0.0, 0.4, 0.0), h_top=1.8, h_bot=0.6, rad=0.45, rot_deg=15)
    # Secondary flanking shards
    add_crystal_spire((0.45, 0.1, 0.35), h_top=1.1, h_bot=0.3, rad=0.28, rot_deg=45)
    add_crystal_spire((-0.40, 0.05, 0.25), h_top=0.9, h_bot=0.25, rad=0.25, rot_deg=80)
    add_crystal_spire((0.05, 0.0, -0.45), h_top=1.3, h_bot=0.35, rad=0.32, rot_deg=110)

    return export_obj("crystal_shard.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 3. KRYŠTÁLOVÝ KMEŇ: CRYSTAL SHIELD BARRIER
# ------------------------------------------------------------------------------
def generate_crystal_shield_obj():
    """Models a hexagonal faceted energy barrier shield."""
    vertices = []
    faces = []
    r_outer = 1.2
    r_inner = 0.95
    t_thick = 0.12
    
    # Front hexagon (inner apex point + 6 ring)
    vertices.append((0.0, 0.0, t_thick)) # 0: Front center tip
    for i in range(6):
        a = math.radians(60 * i)
        vertices.append((r_outer * math.cos(a), r_outer * math.sin(a), 0.0))
    for i in range(6):
        a = math.radians(60 * i)
        vertices.append((r_inner * math.cos(a), r_inner * math.sin(a), t_thick * 0.6))
        
    vertices.append((0.0, 0.0, -t_thick)) # 13: Back center tip
    for i in range(6):
        a = math.radians(60 * i)
        vertices.append((r_outer * math.cos(a) * 0.9, r_outer * math.sin(a) * 0.9, -t_thick * 0.5))

    # Front facets
    for i in range(6):
        ni = (i + 1) % 6
        # Center to inner ring
        faces.append((0, 7 + i, 7 + ni))
        # Inner ring to outer rim
        faces.append((7 + i, 1 + i, 1 + ni))
        faces.append((7 + i, 1 + ni, 7 + ni))
        
    # Back facets
    for i in range(6):
        ni = (i + 1) % 6
        faces.append((13, 14 + ni, 14 + i))
        faces.append((1 + i, 14 + i, 14 + ni))
        faces.append((1 + i, 14 + ni, 1 + ni))

    return export_obj("crystal_shield.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 4. JEDOVATÝ KMEŇ: PROCEDURAL ACID SLIME POOL
# ------------------------------------------------------------------------------
def generate_acid_slime_obj():
    """Models a bubbling, viscous organic acid pool with mounds & ripples."""
    vertices = []
    faces = []
    n_rings = 4
    n_segments = 12
    
    vertices.append((0.0, 0.22, 0.0)) # Center dome
    
    for r in range(1, n_rings + 1):
        radius = r * 0.45
        for s in range(n_segments):
            angle = (2.0 * math.pi * s) / n_segments
            # Irregular organic wave perturbation
            perturb = 0.12 * math.sin(angle * 3.0) + 0.08 * math.cos(angle * 5.0)
            rad_perturbed = max(0.2, radius + perturb)
            height = 0.22 * math.exp(-((radius / 1.5) ** 2)) + 0.06 * math.sin(s * 1.5 + r)
            if r == n_rings:
                height = 0.02
            vertices.append((rad_perturbed * math.cos(angle), height, rad_perturbed * math.sin(angle)))
            
    # Connect rings
    for s in range(n_segments):
        ns = (s + 1) % n_segments
        faces.append((0, 1 + s, 1 + ns))
        
    for r in range(1, n_rings):
        r1_start = 1 + (r - 1) * n_segments
        r2_start = 1 + r * n_segments
        for s in range(n_segments):
            ns = (s + 1) % n_segments
            p1 = r1_start + s
            p2 = r1_start + ns
            p3 = r2_start + s
            p4 = r2_start + ns
            faces.append((p1, p3, p4))
            faces.append((p1, p4, p2))
            
    return export_obj("acid_slime.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 5. JEDOVATÝ KMEŇ: TOXIC SPORE TOWER / TOTEM
# ------------------------------------------------------------------------------
def generate_toxic_totem_obj():
    """Models a twisted tribal spore spire with venting orifices."""
    vertices = []
    faces = []
    n_slices = 8
    n_circle = 8
    
    for slice_i in range(n_slices):
        t = slice_i / (n_slices - 1)
        y = t * 2.2
        # Tapered and twisted waist
        twist = t * 1.8
        radius = (0.55 - 0.25 * math.sin(t * math.pi)) * (1.0 - t * 0.4)
        for c in range(n_circle):
            angle = (2.0 * math.pi * c) / n_circle + twist
            # Organic bulges
            bulge = 0.08 * math.sin(angle * 3.0 + t * 4.0)
            r_eff = radius + bulge
            vertices.append((r_eff * math.cos(angle), y, r_eff * math.sin(angle)))
            
    # Spore bulb apex
    apex_idx = len(vertices)
    vertices.append((0.0, 2.5, 0.0))
    # Base center
    base_idx = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    
    # Side quads
    for slice_i in range(n_slices - 1):
        s1 = slice_i * n_circle
        s2 = (slice_i + 1) * n_circle
        for c in range(n_circle):
            nc = (c + 1) % n_circle
            faces.append((s1 + c, s2 + c, s2 + nc))
            faces.append((s1 + c, s2 + nc, s1 + nc))
            
    # Apex cap
    top_start = (n_slices - 1) * n_circle
    for c in range(n_circle):
        nc = (c + 1) % n_circle
        faces.append((top_start + c, apex_idx, top_start + nc))
        
    # Base cap
    for c in range(n_circle):
        nc = (c + 1) % n_circle
        faces.append((c, nc, base_idx))

    return export_obj("toxic_totem.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 6. DRUIDI: EARTH ROOTS (ENTANGLING VINES)
# ------------------------------------------------------------------------------
def generate_earth_roots_obj():
    """Models gnarled, twisted tree roots bursting from the ground."""
    vertices = []
    faces = []
    
    def add_root_tendril(root_angle, reach_radius, height, max_steps=6):
        tendril_base = len(vertices)
        for step in range(max_steps):
            t = step / (max_steps - 1)
            y = (1.0 - (1.0 - t)**2) * height
            # Curving spiral trajectory
            r = reach_radius * (0.2 + 0.8 * (1.0 - t))
            ang = root_angle + t * 1.5
            cx = r * math.cos(ang)
            cz = r * math.sin(ang)
            
            thickness = 0.22 * (1.0 - t * 0.75)
            # 5-sided cylinder ring
            for k in range(5):
                ka = (2.0 * math.pi * k) / 5
                vx = cx + thickness * math.cos(ka)
                vy = y + thickness * 0.3 * math.sin(ka)
                vz = cz + thickness * math.sin(ka)
                vertices.append((vx, max(0.0, vy), vz))
                
        # Connect cylinder segments
        for step in range(max_steps - 1):
            r1 = tendril_base + step * 5
            r2 = tendril_base + (step + 1) * 5
            for k in range(5):
                nk = (k + 1) % 5
                faces.append((r1 + k, r2 + k, r2 + nk))
                faces.append((r1 + k, r2 + nk, r1 + nk))
                
        # Cap tip
        tip_start = tendril_base + (max_steps - 1) * 5
        tip_point = len(vertices)
        t_last = 1.0
        r_last = reach_radius * 0.2
        vertices.append((r_last * math.cos(root_angle + 1.5), height + 0.1, r_last * math.sin(root_angle + 1.5)))
        for k in range(5):
            nk = (k + 1) % 5
            faces.append((tip_start + k, tip_point, tip_start + nk))

    # 3 intertwined roots
    add_root_tendril(root_angle=0.0, reach_radius=0.9, height=1.4)
    add_root_tendril(root_angle=math.radians(120), reach_radius=1.1, height=1.8)
    add_root_tendril(root_angle=math.radians(240), reach_radius=0.8, height=1.2)

    return export_obj("earth_roots.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 7. DRUIDI: NATURE RUNIC MONOLITH / ALTAR
# ------------------------------------------------------------------------------
def generate_druid_monolith_obj():
    """Models an ancient druidic megalith stone pillar with engraved bevels."""
    vertices = []
    faces = []
    
    # 8-sided tapered pillar
    n_sides = 8
    h_top = 2.4
    h_bevel = 2.6
    
    for i in range(n_sides):
        a = (2.0 * math.pi * i) / n_sides
        # Base ring (wide)
        r_b = 0.60 + 0.08 * (i % 2)
        vertices.append((r_b * math.cos(a), 0.0, r_b * math.sin(a)))
        
    for i in range(n_sides):
        a = (2.0 * math.pi * i) / n_sides
        # Mid ring
        r_m = 0.45 + 0.05 * ((i + 1) % 2)
        vertices.append((r_m * math.cos(a), 1.2, r_m * math.sin(a)))
        
    for i in range(n_sides):
        a = (2.0 * math.pi * i) / n_sides
        # Top ring
        r_t = 0.35 + 0.04 * (i % 2)
        vertices.append((r_t * math.cos(a), h_top, r_t * math.sin(a)))
        
    # Apex tip
    apex = len(vertices)
    vertices.append((0.0, h_bevel, 0.0))
    # Base center
    base = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    
    # Wall quads
    for ring in range(2):
        r1 = ring * n_sides
        r2 = (ring + 1) * n_sides
        for i in range(n_sides):
            ni = (i + 1) % n_sides
            faces.append((r1 + i, r2 + i, r2 + ni))
            faces.append((r1 + i, r2 + ni, r1 + ni))
            
    # Apex cap
    t_start = 2 * n_sides
    for i in range(n_sides):
        ni = (i + 1) % n_sides
        faces.append((t_start + i, apex, t_start + ni))
        
    # Base cap
    for i in range(n_sides):
        ni = (i + 1) % n_sides
        faces.append((i, ni, base))

    return export_obj("druid_monolith.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 8. ECONOMIC BUILD: AETHER CONDUIT (Crystal Mine / Refinery)
# ------------------------------------------------------------------------------
def generate_aether_conduit_obj():
    """Models a crystal refinery mine with industrial base, conduits, and aether pylon."""
    vertices = []
    faces = []
    
    # Octagonal industrial base foundation
    n_seg = 8
    r_base = 1.1
    h_base = 0.4
    
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((r_base * math.cos(a), 0.0, r_base * math.sin(a)))
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((r_base * 0.9 * math.cos(a), h_base, r_base * 0.9 * math.sin(a)))
        
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        faces.append((i, n_seg + i, n_seg + ni))
        faces.append((i, n_seg + ni, ni))
        
    # Floating Aether Core Shard above base
    base_idx = len(vertices)
    vertices.append((0.0, 2.2, 0.0))  # Apex top
    vertices.append((0.0, 0.6, 0.0))  # Apex bottom
    for i in range(6):
        a = (2.0 * math.pi * i) / 6
        vertices.append((0.45 * math.cos(a), 1.4, 0.45 * math.sin(a)))
        
    top_p = base_idx
    bot_p = base_idx + 1
    ring_start = base_idx + 2
    for i in range(6):
        ni = (i + 1) % 6
        faces.append((top_p, ring_start + i, ring_start + ni))
        faces.append((bot_p, ring_start + ni, ring_start + i))
        
    return export_obj("aether_conduit.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 9. ECONOMIC BUILD: SLIME PIT (Bio-Refinery / Vat)
# ------------------------------------------------------------------------------
def generate_slime_pit_obj():
    """Models a bio-reactor vat pool with containment rim and sludge surface."""
    vertices = []
    faces = []
    n_seg = 12
    r_outer = 1.2
    r_inner = 0.95
    h_rim = 0.45
    h_sludge = 0.25
    
    # Outer bottom ring
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((r_outer * math.cos(a), 0.0, r_outer * math.sin(a)))
    # Outer top rim
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((r_outer * math.cos(a), h_rim, r_outer * math.sin(a)))
    # Inner top rim
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((r_inner * math.cos(a), h_rim, r_inner * math.sin(a)))
    # Sludge surface ring
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((r_inner * math.cos(a), h_sludge, r_inner * math.sin(a)))
    # Sludge center
    sludge_center = len(vertices)
    vertices.append((0.0, h_sludge + 0.08, 0.0))
    
    # Outer walls
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        faces.append((i, n_seg + i, n_seg + ni))
        faces.append((i, n_seg + ni, ni))
    # Rim top
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        faces.append((n_seg + i, 2 * n_seg + i, 2 * n_seg + ni))
        faces.append((n_seg + i, 2 * n_seg + ni, n_seg + ni))
    # Sludge pool
    s_start = 3 * n_seg
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        faces.append((sludge_center, s_start + i, s_start + ni))

    return export_obj("slime_pit.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 10. ECONOMIC BUILD: WORLD TREE (Living Grove / Sanctuary)
# ------------------------------------------------------------------------------
def generate_world_tree_obj():
    """Models an ancient World Tree canopy with twisting trunk and foliage crown."""
    vertices = []
    faces = []
    
    # Trunk cylinder (tapered & twisted)
    n_slices = 5
    n_circle = 6
    for sl in range(n_slices):
        t = sl / (n_slices - 1)
        y = t * 2.0
        rad = (0.55 - t * 0.25)
        twist = t * 0.8
        for c in range(n_circle):
            a = (2.0 * math.pi * c) / n_circle + twist
            vertices.append((rad * math.cos(a), y, rad * math.sin(a)))
            
    for sl in range(n_slices - 1):
        s1 = sl * n_circle
        s2 = (sl + 1) * n_circle
        for c in range(n_circle):
            nc = (c + 1) % n_circle
            faces.append((s1 + c, s2 + c, s2 + nc))
            faces.append((s1 + c, s2 + nc, s1 + nc))
            
    # Canopy Foliage (layered icosahedral dome)
    canopy_start = len(vertices)
    canopy_center = (0.0, 2.6, 0.0)
    c_rad = 1.3
    # 8-ring crown
    for i in range(8):
        a = (2.0 * math.pi * i) / 8
        vertices.append((c_rad * math.cos(a), 2.4, c_rad * math.sin(a)))
    for i in range(8):
        a = (2.0 * math.pi * i) / 8 + 0.3
        vertices.append((c_rad * 0.7 * math.cos(a), 3.2, c_rad * 0.7 * math.sin(a)))
    top_leaf = len(vertices)
    vertices.append((0.0, 3.6, 0.0))
    
    for i in range(8):
        ni = (i + 1) % 8
        faces.append((canopy_start + i, canopy_start + 8 + i, canopy_start + 8 + ni))
        faces.append((canopy_start + i, canopy_start + 8 + ni, canopy_start + ni))
        faces.append((top_leaf, canopy_start + 8 + ni, canopy_start + 8 + i))

    return export_obj("world_tree.obj", vertices, faces)

# ------------------------------------------------------------------------------
# 11. ECONOMIC BUILD: CAPACITOR TOWER (Mana Storage / Defense Pylon)
# ------------------------------------------------------------------------------
def generate_capacitor_tower_obj():
    """Models a high-voltage mana capacitor tower with rings and spire."""
    vertices = []
    faces = []
    n_seg = 6
    h_tower = 2.8
    r_tower = 0.4
    
    # Base
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((0.8 * math.cos(a), 0.0, 0.8 * math.sin(a)))
    for i in range(n_seg):
        a = (2.0 * math.pi * i) / n_seg
        vertices.append((r_tower * math.cos(a), h_tower, r_tower * math.sin(a)))
        
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        faces.append((i, n_seg + i, n_seg + ni))
        faces.append((i, n_seg + ni, ni))
        
    # Apex energy emitter
    apex = len(vertices)
    vertices.append((0.0, h_tower + 0.7, 0.0))
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        faces.append((apex, n_seg + i, n_seg + ni))

    return export_obj("capacitor_tower.obj", vertices, faces)

def bake_all_assets():
    print("=================================================================")
    print(" KRYSTAL-STACK: BAKING PROCEDURAL 3D ASSETS FOR GODOT ENGINE")
    print("=================================================================")
    generate_hex_tile_obj()
    generate_crystal_shard_obj()
    generate_crystal_shield_obj()
    generate_acid_slime_obj()
    generate_toxic_totem_obj()
    generate_earth_roots_obj()
    generate_druid_monolith_obj()
    # Economic Builds
    generate_aether_conduit_obj()
    generate_slime_pit_obj()
    generate_world_tree_obj()
    generate_capacitor_tower_obj()
    print("=================================================================")
    print(f" ALL ASSETS COMPILED INTO: {ASSET_DIR}")
    print("=================================================================")

if __name__ == "__main__":
    bake_all_assets()
