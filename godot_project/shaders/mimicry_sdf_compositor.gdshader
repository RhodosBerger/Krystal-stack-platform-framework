shader_type canvas_item;

// ==============================================================================
// KRYSTAL-STACK: MIMICRY COMPOSITOR & BLENDER MODIFIER SHADER
// Framework: Godot 4.x (Vulkan / Forward+ / Compatibility)
// Evaluates composite multi-part objects with smooth CSG blending (smin),
// procedural displacement, and array repetitions directly on the GPU.
// ==============================================================================

uniform sampler2D screen_texture : hint_screen_texture, filter_nearest;

// Uniforms driven by Mimicry Compositor Engine
uniform int u_recipe_mode : hint_range(0, 5) = 0; // 0=Turret, 1=Mech, 2=Obelisk, 3=Drone, 4=Spire, 5=Explorer
uniform float u_smooth_k : hint_range(0.01, 0.5) = 0.15;
uniform float u_displace_strength : hint_range(0.0, 0.2) = 0.05;
uniform int u_array_count : hint_range(1, 8) = 2;
uniform vec3 u_light_dir = vec3(0.577, 0.707, -0.408);
uniform vec4 u_primary_color : source_color = vec4(0.0, 0.941, 1.0, 1.0);  // Cyan
uniform vec4 u_accent_color : source_color = vec4(1.0, 0.0, 0.333, 1.0);   // Magenta

uniform vec2 cell_size = vec2(8.0, 16.0); // ASCII character cell dimensions
uniform float cam_dist : hint_range(2.0, 10.0) = 4.0;
uniform float rot_speed : hint_range(-2.0, 2.0) = 0.4;

// ── CSG Smooth Operators ─────────────────────────────────────────────────────

float smin(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

float smax(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return max(a, b) + h * h * k * 0.25;
}

float smooth_diff(float a, float b, float k) {
    return smax(a, -b, k);
}

// ── Primitives ───────────────────────────────────────────────────────────────

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

float sdf_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdf_cylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdf_torus(vec3 p, float r1, float r2) {
    vec2 q = vec2(length(p.xz) - r1, p.y);
    return length(q) - r2;
}

// ── Composite Scene Evaluation ───────────────────────────────────────────────

float evaluate_mimic_scene(vec3 p) {
    // Apply procedural displacement if enabled
    float disp = 0.0;
    if (u_displace_strength > 0.001) {
        disp = u_displace_strength * sin(p.x * 5.0) * cos(p.y * 5.0) * sin(p.z * 5.0);
    }

    float d = 1000.0;

    // 0: Cyber Turret MK-4
    if (u_recipe_mode == 0) {
        float d_base = sdf_box(p - vec3(0.0, -0.6, 0.0), vec3(0.8, 0.2, 0.8)) - 0.05;
        float d_core = sdf_sphere(p - vec3(0.0, -0.2, 0.0), 0.45);
        float d_mantlet = sdf_box(p - vec3(0.0, 0.15, 0.0), vec3(0.35, 0.25, 0.5));
        
        // Array barrels
        vec3 p_bar = p - vec3(0.0, 0.15, 0.6);
        p_bar.x = abs(p_bar.x) - 0.16;
        float d_barrel = sdf_cylinder(p_bar.xzy, 0.06, 0.6);

        d = smin(d_base, d_core, u_smooth_k);
        d = smin(d, d_mantlet, u_smooth_k);
        d = min(d, d_barrel);
    }
    // 1: Titan Mech Walker
    else if (u_recipe_mode == 1) {
        float d_cockpit = sdf_box(p - vec3(0.0, 0.4, 0.0), vec3(0.5, 0.4, 0.6)) - 0.06;
        // Legs with mirror X
        vec3 p_leg = p;
        p_leg.x = abs(p_leg.x) - 0.55;
        float d_thigh = sdf_box(p_leg - vec3(0.0, -0.3, 0.0), vec3(0.12, 0.45, 0.15));
        float d_foot = sdf_box(p_leg - vec3(0.0, -0.8, -0.05), vec3(0.22, 0.06, 0.35));
        d = smin(d_cockpit, d_thigh, u_smooth_k);
        d = min(d, d_foot);
    }
    // 2: Ancient Obelisk
    else if (u_recipe_mode == 2) {
        float d_plinth = sdf_box(p - vec3(0.0, -0.8, 0.0), vec3(0.8, 0.1, 0.8));
        float d_spire = sdf_cylinder(p - vec3(0.0, 0.2, 0.0), 0.35 - p.y * 0.1, 0.9);
        float d_cap = sdf_box(p - vec3(0.0, 1.4, 0.0), vec3(0.2, 0.2, 0.2));
        d = min(d_plinth, d_spire);
        d = min(d, d_cap);
    }
    // 3: Xenodrone
    else if (u_recipe_mode == 3) {
        float d_head = sdf_sphere(p - vec3(0.0, 0.3, 0.1), 0.45);
        vec3 p_rib = p - vec3(0.0, 0.0, -0.3);
        float d_spine = sdf_torus(p_rib, 0.3, 0.08);
        d = smin(d_head, d_spine, u_smooth_k);
    }
    // 4: Data Spire
    else if (u_recipe_mode == 4) {
        float d_core = sdf_box(p, vec3(0.25, 1.3, 0.25));
        vec3 p_deck = p;
        p_deck.y = mod(p_deck.y + 0.3, 0.4) - 0.2;
        float d_decks = sdf_box(p_deck, vec3(0.65, 0.04, 0.65));
        float d_ring = sdf_torus(p - vec3(0.0, 1.25, 0.0), 0.6, 0.05);
        d = min(d_core, d_decks);
        d = min(d, d_ring);
    }
    // 5: Solar Explorer
    else {
        float d_hab = sdf_sphere(p, 0.55);
        vec3 p_wing = p;
        p_wing.x = abs(p_wing.x) - 1.2;
        float d_wings = sdf_box(p_wing, vec3(0.65, 0.02, 0.35));
        d = min(d_hab, d_wings);
    }

    return d + disp;
}

vec3 calc_normal(vec3 p) {
    vec2 eps = vec2(0.005, 0.0);
    return normalize(vec3(
        evaluate_mimic_scene(p + eps.xyy) - evaluate_mimic_scene(p - eps.xyy),
        evaluate_mimic_scene(p + eps.yxy) - evaluate_mimic_scene(p - eps.yxy),
        evaluate_mimic_scene(p + eps.yyx) - evaluate_mimic_scene(p - eps.yyx)
    ));
}

void fragment() {
    vec2 screen_res = 1.0 / SCREEN_PIXEL_SIZE;
    
    // 1. Quantize screen into ASCII character blocks
    vec2 cell_coord = floor(FRAGCOORD.xy / cell_size);
    vec2 cell_center_uv = (cell_coord * cell_size + cell_size * 0.5) / screen_res;
    vec2 cell_local_uv = fract(FRAGCOORD.xy / cell_size);

    // 2. Camera Setup (Orbiting)
    float aspect = screen_res.x / screen_res.y;
    vec2 p_ndc = (cell_center_uv * 2.0 - 1.0) * vec2(aspect, -1.0);

    float angle = TIME * rot_speed;
    vec3 ro = vec3(cam_dist * sin(angle), 1.0, -cam_dist * cos(angle));
    vec3 ta = vec3(0.0, 0.0, 0.0);

    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(vec3(0.0, 1.0, 0.0), ww));
    vec3 vv = cross(ww, uu);
    vec3 rd = normalize(p_ndc.x * uu + p_ndc.y * vv + 1.8 * ww);

    // 3. Raymarch
    float dist = 0.0;
    bool hit = false;
    vec3 p = ro;

    for (int i = 0; i < 32; i++) {
        p = ro + rd * dist;
        float d = evaluate_mimic_scene(p);
        if (d < 0.005) {
            hit = true;
            break;
        }
        dist += d;
        if (dist > 12.0) break;
    }

    vec3 final_rgb = vec3(0.015, 0.02, 0.035); // Ambient background

    if (hit) {
        vec3 norm = calc_normal(p);
        vec3 l_dir = normalize(u_light_dir);
        float diff = max(0.0, dot(norm, l_dir));
        float rim = 1.0 - max(0.0, dot(norm, -rd));
        float luma = clamp(diff * 0.7 + pow(rim, 2.0) * 0.5, 0.0, 1.0);

        // Procedural ASCII glyph selection
        float glyph_alpha = 0.0;
        if (rim > 0.7) {
            // Directional edge strokes
            float deg = degrees(atan(norm.y, norm.x));
            if (deg < 0.0) deg += 180.0;
            if (deg > 67.5 && deg < 112.5) glyph_alpha = smoothstep(0.2, 0.0, abs(cell_local_uv.x - 0.5));
            else if (deg > 22.5 && deg <= 67.5) glyph_alpha = smoothstep(0.2, 0.0, abs(cell_local_uv.x + cell_local_uv.y - 1.0));
            else if (deg >= 112.5 && deg < 157.5) glyph_alpha = smoothstep(0.2, 0.0, abs(cell_local_uv.x - cell_local_uv.y));
            else glyph_alpha = smoothstep(0.2, 0.0, abs(cell_local_uv.y - 0.5));
        } else {
            // Density fills
            if (luma > 0.7) glyph_alpha = 1.0;
            else if (luma > 0.4) glyph_alpha = (abs(cell_local_uv.x - 0.5) < 0.35 && abs(cell_local_uv.y - 0.5) < 0.35) ? 0.8 : 0.0;
            else if (luma > 0.15) glyph_alpha = length(cell_local_uv - vec2(0.5)) < 0.22 ? 0.6 : 0.0;
        }

        vec3 color_tone = mix(u_primary_color.rgb, u_accent_color.rgb, pow(rim, 2.0));
        final_rgb = color_tone * luma * glyph_alpha;
        final_rgb += color_tone * 0.05; // Rim ambient
    }

    COLOR = vec4(final_rgb, 1.0);
}
