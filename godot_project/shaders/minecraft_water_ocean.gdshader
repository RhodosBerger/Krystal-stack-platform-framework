shader_type spatial;
render_mode blend_mix, depth_draw_always, cull_back, diffuse_burley, specular_schlick_ggx;

// ==============================================================================
// KRYSTAL-STACK: MINECRAFT-GRADE GLSL WATER & OCEAN SHADER
// Compatible with: Godot 4.2+ (Forward+ / Vulkan)
// Inspired by: BSL Shaders Water, Complementary Reimagined Emerald Oceans
// Features: Multi-Octave Gerstner Waves, Beer-Lambert Extinction, Underwater Caustics,
//           Screen-Space Refraction, Dynamic Edge Shore Foam.
// ==============================================================================

uniform sampler2D screen_texture : hint_screen_texture, filter_linear_mipmap;
uniform sampler2D depth_texture : hint_depth_texture, filter_linear;

// --- Water Optical Color Properties ---
uniform vec4 water_shallow_color : source_color = vec4(0.0, 0.72, 0.82, 0.65); // Azure Cyan
uniform vec4 water_deep_color : source_color = vec4(0.01, 0.12, 0.32, 0.95);    // Deep Oceanic Indigo
uniform vec3 absorption_coefficients = vec3(0.25, 0.08, 0.03); // Red absorbs fastest, Blue slowest

// --- Gerstner Wave Parameters ---
uniform int wave_octaves : hint_range(1, 4) = 3;
uniform float wave_amplitude : hint_range(0.0, 1.0) = 0.15;
uniform float wave_frequency : hint_range(0.1, 2.0) = 0.6;
uniform float wave_speed : hint_range(0.1, 4.0) = 1.2;
uniform float wave_steepness : hint_range(0.0, 1.0) = 0.45; // Trochoidal wave sharp crest factor

// --- Optical FX & Refraction ---
uniform float refraction_strength : hint_range(0.0, 0.05) = 0.018;
uniform float chromatic_dispersion : hint_range(0.0, 0.01) = 0.003;
uniform float fresnel_power : hint_range(1.0, 10.0) = 5.0;

// --- Caustics & Shoreline Foam ---
uniform float caustic_scale : hint_range(1.0, 20.0) = 6.0;
uniform float caustic_speed : hint_range(0.1, 3.0) = 0.8;
uniform float caustic_intensity : hint_range(0.0, 2.0) = 0.85;
uniform float foam_threshold : hint_range(0.05, 1.5) = 0.35;
uniform vec4 foam_color : source_color = vec4(0.92, 0.96, 1.0, 0.9);

varying vec3 v_world_pos;
varying vec3 v_normal;

// Gerstner Wave Displacement & Normal Accumulator
vec3 evaluate_gerstner_wave(vec2 dir, float steepness, float amplitude, float frequency, float speed, vec3 p, inout vec3 normal) {
    float k = 2.0 * 3.14159265 * frequency;
    float c = sqrt(9.8 / k);
    vec2 d = normalize(dir);
    float f = k * (dot(d, p.xz) - c * speed * TIME * 0.2);
    float a = steepness / k;
    
    // Wave derivatives for normal perturbation
    normal.x -= d.x * (a * k * cos(f));
    normal.y -= a * k * sin(f);
    normal.z -= d.y * (a * k * cos(f));
    
    return vec3(
        d.x * (a * cos(f)),
        amplitude * sin(f),
        d.y * (a * cos(f))
    );
}

// Procedural Caustic Generator (Voronoi wave interference)
float calculate_caustic_pattern(vec2 uv, float time) {
    vec2 p = uv * caustic_scale;
    vec2 p1 = p + vec2(sin(time * 0.7), cos(time * 0.6)) * 0.4;
    vec2 p2 = p * 1.3 - vec2(cos(time * 0.5), sin(time * 0.8)) * 0.35;
    
    float c1 = sin(p1.x * 2.5 + sin(p1.y * 3.0 + time)) * 0.5 + 0.5;
    float c2 = cos(p2.x * 3.1 + cos(p2.y * 2.8 + time * 1.2)) * 0.5 + 0.5;
    return pow(c1 * c2, 1.8) * 2.2;
}

void vertex() {
    v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
    vec3 wave_displacement = vec3(0.0);
    vec3 wave_normal = vec3(0.0, 1.0, 0.0);
    
    // Octave 1: Primary Swell
    wave_displacement += evaluate_gerstner_wave(vec2(1.0, 0.3), wave_steepness, wave_amplitude, wave_frequency, wave_speed, v_world_pos, wave_normal);
    
    // Octave 2: Cross Sea Wave
    if (wave_octaves >= 2) {
        wave_displacement += evaluate_gerstner_wave(vec2(0.4, 1.0), wave_steepness * 0.7, wave_amplitude * 0.5, wave_frequency * 1.8, wave_speed * 1.4, v_world_pos, wave_normal);
    }
    
    // Octave 3: Micro Ripple
    if (wave_octaves >= 3) {
        wave_displacement += evaluate_gerstner_wave(vec2(-0.7, 0.7), wave_steepness * 0.5, wave_amplitude * 0.25, wave_frequency * 3.2, wave_speed * 1.9, v_world_pos, wave_normal);
    }
    
    VERTEX += wave_displacement;
    v_normal = normalize(wave_normal);
}

void fragment() {
    // 1. Reconstruct Scene Depth behind water surface
    float raw_depth = texture(depth_texture, SCREEN_UV).r;
    vec3 ndc = vec3(SCREEN_UV * 2.0 - 1.0, raw_depth);
    vec4 view_pos = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
    view_pos /= view_pos.w;
    float scene_depth = -view_pos.z;
    
    // Surface depth
    float surface_depth = -VERTEX.z;
    float water_depth = max(scene_depth - surface_depth, 0.0);
    
    // 2. Normal Vector & Fresnel Specular Term
    vec3 N = normalize(v_normal);
    vec3 V = normalize(VIEW);
    float NdotV = max(dot(N, V), 0.0);
    float fresnel = pow(1.0 - NdotV, fresnel_power);
    fresnel = mix(0.02, 1.0, fresnel); // Schlick F0 for water is 0.02
    
    // 3. Screen-Space Refraction with Chromatic Dispersion
    vec2 distortion = N.xz * refraction_strength * clamp(water_depth * 0.5, 0.0, 1.0);
    vec2 uv_r = SCREEN_UV + distortion + vec2(chromatic_dispersion, 0.0);
    vec2 uv_g = SCREEN_UV + distortion;
    vec2 uv_b = SCREEN_UV + distortion - vec2(chromatic_dispersion, 0.0);
    
    float bg_r = texture(screen_texture, uv_r).r;
    float bg_g = texture(screen_texture, uv_g).g;
    float bg_b = texture(screen_texture, uv_b).b;
    vec3 refracted_background = vec3(bg_r, bg_g, bg_b);
    
    // 4. Beer-Lambert Light Extinction
    vec3 extinction = exp(-absorption_coefficients * water_depth * 2.5);
    vec3 watercolor = mix(water_deep_color.rgb, water_shallow_color.rgb, clamp(water_depth * 0.2, 0.0, 1.0));
    vec3 absorbed_underwater = refracted_background * extinction + watercolor * (1.0 - extinction);
    
    // 5. Dynamic Underwater Caustics
    float caustic = calculate_caustic_pattern(v_world_pos.xz + N.xz * 0.2, TIME * caustic_speed);
    float caustic_mask = clamp(exp(-water_depth * 0.8), 0.0, 1.0); // Caustics fade in very deep water
    absorbed_underwater += vec3(caustic * caustic_intensity * caustic_mask);
    
    // 6. Shoreline Contact Foam
    vec3 final_color = absorbed_underwater;
    if (water_depth < foam_threshold) {
        float foam_factor = 1.0 - (water_depth / foam_threshold);
        float foam_noise = sin(v_world_pos.x * 8.0 + TIME * 3.0) * cos(v_world_pos.z * 8.0 + TIME * 2.5);
        foam_factor = smoothstep(0.2, 0.8, foam_factor + foam_noise * 0.15);
        final_color = mix(final_color, foam_color.rgb, foam_factor * foam_color.a);
    }
    
    // Output Surface Parameters
    ALBEDO = final_color;
    NORMAL = N;
    ROUGHNESS = 0.02;
    METALLIC = 0.0;
    SPECULAR = fresnel;
    ALPHA = mix(water_shallow_color.a, 1.0, clamp(water_depth * 0.4 + fresnel * 0.6, 0.0, 1.0));
}
