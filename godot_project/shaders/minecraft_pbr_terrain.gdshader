shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

// ==============================================================================
// KRYSTAL-STACK: MINECRAFT-GRADE GLSL PBR TERRAIN SHADER (LabPBR 1.3)
// Compatible with: Godot 4.2+ (Forward+ / Vulkan / Mobile)
// Inspired by: BSL Shaders, Complementary Reimagined, SEUS PTGI
// ==============================================================================

// --- Textures (LabPBR 1.3 Standard) ---
uniform sampler2D texture_albedo : source_color, filter_linear_mipmap_anisotropic;
uniform sampler2D texture_normal : hint_normal, filter_linear_mipmap_anisotropic; // RG: Normal XY, B: AO, A: Height (POM)
uniform sampler2D texture_specular : hint_default_black, filter_linear_mipmap_anisotropic; // R: Smoothness, G: F0/Metal, B: Porosity, A: Emission

// --- Parallax Occlusion Mapping (POM) ---
uniform bool pom_enabled = true;
uniform float pom_depth_scale : hint_range(0.0, 0.1) = 0.04;
uniform int pom_min_layers : hint_range(4, 64) = 8;
uniform int pom_max_layers : hint_range(8, 128) = 32;
uniform bool pom_self_shadow = true;
uniform float pom_shadow_strength : hint_range(0.0, 1.0) = 0.7;

// --- Wind & Foliage Physics ---
uniform bool wind_waving = false;
uniform float wind_speed : hint_range(0.0, 5.0) = 1.2;
uniform float wind_amplitude : hint_range(0.0, 0.5) = 0.08;
uniform vec2 wind_direction = vec2(1.0, 0.5);

// --- Weather & Wetness Simulation ---
uniform float wetness_intensity : hint_range(0.0, 1.0) = 0.0;
uniform vec4 puddle_tint : source_color = vec4(0.05, 0.05, 0.05, 1.0);

// --- Constants & Invariants ---
const float GOLDEN_RATIO = 1.61803398875;
const int VITAL_MAX_HP = 6;

varying vec3 v_tangent_view_dir;
varying vec3 v_tangent_light_dir;
varying vec3 v_world_pos;

// Parallax Occlusion Mapping with Binary Search refinement
vec2 calculate_pom_uv(vec2 base_uv, vec3 view_dir_tangent, out float parallax_height) {
    if (!pom_enabled) {
        parallax_height = 0.0;
        return base_uv;
    }
    
    // Adjust sample count based on viewing angle
    float num_layers = mix(float(pom_max_layers), float(pom_min_layers), max(dot(vec3(0.0, 0.0, 1.0), view_dir_tangent), 0.0));
    float layer_depth = 1.0 / num_layers;
    float current_layer_depth = 0.0;
    
    // Shift vector in UV space
    vec2 p = view_dir_tangent.xy * pom_depth_scale / max(view_dir_tangent.z, 0.05);
    vec2 delta_uv = p / num_layers;
    
    vec2 current_uv = base_uv;
    float current_height_value = 1.0 - texture(texture_normal, current_uv).a;
    
    // Steep Parallax Mapping
    while (current_layer_depth < current_height_value) {
        current_uv -= delta_uv;
        current_height_value = 1.0 - texture(texture_normal, current_uv).a;
        current_layer_depth += layer_depth;
    }
    
    // Parallax Occlusion Interpolation (linear between previous and current step)
    vec2 prev_uv = current_uv + delta_uv;
    float next_diff = current_height_value - current_layer_depth;
    float prev_diff = (1.0 - texture(texture_normal, prev_uv).a) - (current_layer_depth - layer_depth);
    
    float weight = next_diff / (next_diff - prev_diff);
    vec2 final_uv = mix(current_uv, prev_uv, weight);
    parallax_height = current_layer_depth;
    return final_uv;
}

// Parallax Self-Shadowing calculation
float calculate_pom_shadow(vec2 uv, vec3 light_dir_tangent, float initial_height) {
    if (!pom_self_shadow || light_dir_tangent.z <= 0.0) {
        return 1.0;
    }
    
    int shadow_steps = 16;
    float step_size = 1.0 / float(shadow_steps);
    vec2 delta_uv = (light_dir_tangent.xy * pom_depth_scale / light_dir_tangent.z) * step_size;
    
    float current_shadow_depth = initial_height - step_size;
    vec2 current_uv = uv + delta_uv;
    float shadow = 1.0;
    
    for (int i = 0; i < shadow_steps; i++) {
        float sample_height = 1.0 - texture(texture_normal, current_uv).a;
        if (sample_height > current_shadow_depth) {
            shadow = 1.0 - pom_shadow_strength;
            break;
        }
        current_shadow_depth -= step_size;
        current_uv += delta_uv;
    }
    return shadow;
}

void vertex() {
    v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
    
    // Foliage Wind Displacement (Minecraft-style sinusoidal plant sway)
    if (wind_waving) {
        float wind_time = TIME * wind_speed;
        vec2 norm_wind = normalize(wind_direction);
        float sway = sin(wind_time * 2.0 + v_world_pos.x * 0.5 + v_world_pos.z * 0.5) *
                     cos(wind_time * 1.5 + v_world_pos.y * 0.8);
        
        // Sway increases towards the top of the block/geometry
        float height_factor = clamp(UV.y, 0.0, 1.0);
        VERTEX.xz += norm_wind * sway * wind_amplitude * height_factor;
    }
    
    // Construct Tangent-Bitangent-Normal (TBN) Matrix
    vec3 T = normalize(TANGENT);
    vec3 B = normalize(BINORMAL);
    vec3 N = normalize(NORMAL);
    mat3 tbn = mat3(T, B, N);
    
    // Transform View Vector to Tangent Space
    vec3 view_dir_world = normalize(CAMERA_POSITION_WORLD - v_world_pos);
    v_tangent_view_dir = tbn * view_dir_world;
    
    // Key Light Direction (default sun vector)
    vec3 light_dir_world = normalize(vec3(0.5, 0.8, 0.3));
    v_tangent_light_dir = tbn * light_dir_world;
}

void fragment() {
    float parallax_depth = 0.0;
    vec2 pom_uv = calculate_pom_uv(UV, normalize(v_tangent_view_dir), parallax_depth);
    
    // Sample Textures
    vec4 albedo_sample = texture(texture_albedo, pom_uv);
    vec4 normal_sample = texture(texture_normal, pom_uv);
    vec4 spec_sample = texture(texture_specular, pom_uv);
    
    // --- LabPBR 1.3 Decoding ---
    // 1. Perceptual Smoothness -> Linear Roughness: roughness = (1.0 - smoothness)^2
    float perceptual_smoothness = spec_sample.r;
    float linear_roughness = pow(1.0 - perceptual_smoothness, 2.0);
    
    // 2. F0 (Reflectance) & Metal Mask:
    // LabPBR: Green 0..228 = Dielectric F0, 229..255 = Hardcoded Metals
    float f0_raw = spec_sample.g;
    float metallic_val = 0.0;
    float specular_val = 0.5;
    
    if (f0_raw > (228.0 / 255.0)) {
        metallic_val = 1.0;
        specular_val = 1.0;
    } else {
        metallic_val = 0.0;
        specular_val = f0_raw * (255.0 / 228.0);
    }
    
    // 3. Porosity (Water absorption / Puddle physics)
    float porosity = spec_sample.b;
    
    // 4. Emission: Alpha 0..254 = glowing brightness, 255 = disabled
    vec3 emissive_color = vec3(0.0);
    if (spec_sample.a < (254.5 / 255.0)) {
        float emission_strength = spec_sample.a * 5.0;
        emissive_color = albedo_sample.rgb * emission_strength;
    }
    
    // 5. Normal Tangent Vector & Material AO (Normal B channel)
    vec2 normal_xy = normal_sample.rg * 2.0 - 1.0;
    float normal_z = sqrt(max(1.0 - dot(normal_xy, normal_xy), 0.0));
    vec3 reconstructed_normal = normalize(vec3(normal_xy, normal_z));
    float material_ao = normal_sample.b;
    
    // 6. Wetness & Rain Puddles Modulation
    if (wetness_intensity > 0.0) {
        // Porous materials darken and smooth out when wet
        float wet_factor = wetness_intensity * (1.0 - porosity * 0.5);
        albedo_sample.rgb = mix(albedo_sample.rgb, albedo_sample.rgb * 0.65, wet_factor);
        linear_roughness = mix(linear_roughness, 0.04, wet_factor);
        specular_val = mix(specular_val, 0.9, wet_factor);
    }
    
    // 7. Parallax Self-Shadowing
    float pom_shadow = calculate_pom_shadow(pom_uv, normalize(v_tangent_light_dir), parallax_depth);
    material_ao *= pom_shadow;
    
    // Output to Godot PBR Material Pipeline
    ALBEDO = albedo_sample.rgb;
    NORMAL = reconstructed_normal;
    ROUGHNESS = clamp(linear_roughness, 0.04, 1.0);
    METALLIC = metallic_val;
    SPECULAR = specular_val;
    AO = material_ao;
    EMISSION = emissive_color;
}
