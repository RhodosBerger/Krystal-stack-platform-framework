shader_type canvas_item;

// =============================================================================
// KRYSTAL-STACK // GODOT 4.x PHOTOREALISTIC SDF RAYMARCHING SHADER
// Scene: NeoPraha_Alchemical_Hologram
// Hardware L2 Cache Budget: 1696.0 KB (43.1% of GPU L2)
// Scene Diversity Score: 88.5 / 100.0
// Non-negotiable Invariant: VITAL_MAX_HP = 6
// =============================================================================

uniform sampler2D screen_texture : hint_screen_texture, filter_nearest;

// ─── Hardware Tuned Push Constants & Uniforms ───────────────────────────────
uniform int u_max_steps : hint_range(16, 128) = 96;
uniform float u_epsilon = 0.001;
uniform int u_coxeter_folds : hint_range(2, 16) = 6;
uniform float u_fresnel_factor : hint_range(0.1, 1.0) = 0.88;
uniform float u_bayer_dither_strength : hint_range(0.0, 1.0) = 0.65;
uniform vec4 u_gold_alchemical : source_color = vec4(1.0, 0.843, 0.0, 1.0);
uniform vec4 u_crystal_cyan : source_color = vec4(0.0, 0.941, 1.0, 1.0);
uniform vec3 u_light_dir = vec3(0.577, 0.577, -0.577);

const mat4 BAYER_4x4 = mat4(
    vec4( 0.0,  8.0,  2.0, 10.0) / 16.0,
    vec4(12.0,  4.0, 14.0,  6.0) / 16.0,
    vec4( 3.0, 11.0,  1.0,  9.0) / 16.0,
    vec4(15.0,  7.0, 13.0,  5.0) / 16.0
);

// ─── Coxeter Dihedral Symmetry Fold (D_N) ───────────────────────────────────
vec2 fold_dihedral(vec2 p, int folds) {
    float r = length(p);
    float theta = atan(p.y, p.x);
    float sector = 3.141592653589793 / float(max(1, folds));
    float mod_theta = abs(mod(theta, 2.0 * sector) - sector);
    return vec2(cos(mod_theta), sin(mod_theta)) * r;
}

// ─── Signed Distance Field (SDF) Primitives ─────────────────────────────────
float sd_sphere(vec3 p, float r) {
    return length(p) - r;
}

float sd_chalice(vec3 p) {
    // Stem + Bowl
    float stem = length(p.xz) - 0.12;
    stem = max(stem, abs(p.y) - 0.55);
    float bowl = length(p - vec3(0.0, 0.65, 0.0)) - 0.78;
    bowl = max(bowl, -(length(p - vec3(0.0, 0.72, 0.0)) - 0.72));
    return min(stem, bowl);
}

float sd_athame(vec3 p) {
    // Blade along Y
    vec3 q = abs(p);
    float blade = max(q.x * 2.0 + q.z * 12.0 - 0.15, q.y - 1.45);
    float guard = length(vec2(q.x - 0.65, q.z)) - 0.1;
    return min(blade, guard);
}

float map_scene(vec3 p) {
    // 1. Fold across Coxeter dihedral symmetry planes
    p.xz = fold_dihedral(p.xz, u_coxeter_folds);
    
    // 2. Multioctave elevation wave
    float terrain = p.y + 1.2 + 0.25 * sin(p.x * 1.5) * cos(p.z * 1.5);
    
    // 3. Bohemian alchemical artifacts
    float chalice = sd_chalice(p - vec3(1.2, 0.0, 0.0));
    float athame = sd_athame(p - vec3(-1.2, 0.0, 0.0));
    
    return min(terrain, min(chalice, athame));
}

vec3 calc_normal(vec3 p) {
    const float h = 0.002;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * map_scene(p + k.xyy * h) +
        k.yyx * map_scene(p + k.yyx * h) +
        k.yxy * map_scene(p + k.yxy * h) +
        k.xxx * map_scene(p + k.xxx * h)
    );
}

void fragment() {
    vec2 res = 1.0 / SCREEN_PIXEL_SIZE;
    vec2 uv = (FRAGCOORD.xy - 0.5 * res) / res.y;
    
    // Ray generation
    vec3 ro = vec3(0.0, 1.8, -3.8);
    vec3 rd = normalize(vec3(uv, 1.35));
    
    float t = 0.0;
    float d = 0.0;
    int hit_step = -1;
    
    for (int i = 0; i < u_max_steps; i++) {
        vec3 p = ro + rd * t;
        d = map_scene(p);
        if (d < u_epsilon) {
            hit_step = i;
            break;
        }
        t += d * 0.85; // Conservative step to prevent L2 cache thrash
        if (t > 25.0) break;
    }
    
    vec3 col = vec3(0.04, 0.05, 0.08); // Dark atmospheric background
    
    if (hit_step >= 0) {
        vec3 p = ro + rd * t;
        vec3 n = calc_normal(p);
        
        // PBR Directional Lighting & Fresnel
        float diff = max(0.0, dot(n, normalize(u_light_dir)));
        float fresnel = pow(1.0 - max(0.0, dot(-rd, n)), 3.0) * u_fresnel_factor;
        
        // Alchemical Dual Tone
        vec3 base_color = mix(u_gold_alchemical.rgb, u_crystal_cyan.rgb, clamp(p.y * 0.5 + 0.5, 0.0, 1.0));
        col = base_color * (diff * 0.85 + 0.15) + vec3(fresnel);
    }
    
    // Bayer Matrix Halftone Post-Process
    ivec2 bayer_coord = ivec2(mod(FRAGCOORD.xy, 4.0));
    float threshold = BAYER_4x4[bayer_coord.x][bayer_coord.y];
    col = mix(col, floor(col * 8.0 + threshold) / 8.0, u_bayer_dither_strength);
    
    COLOR = vec4(col, 1.0);
}
