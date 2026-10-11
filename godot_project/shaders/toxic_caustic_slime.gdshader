// =====================================================================
// KRYSTAL-STACK: TOXIC CAUSTIC SLIME SHADER
// Godot 4.2+ Forward+ / Vulkan
// Specialized for: Hnijící Slatiny (Toxic Tribe)
// Features: Bubbling vertex ripples, caustic bioluminescence, acid DoT glow
// =====================================================================
shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_lambert, specular_schlick_ggx;

uniform vec4 toxic_color : source_color = vec4(0.18, 0.95, 0.05, 0.95);
uniform vec4 mire_purple : source_color = vec4(0.55, 0.08, 0.85, 1.0);
uniform float bubble_speed : hint_range(0.2, 4.0) = 1.6;
uniform float wave_amplitude : hint_range(0.0, 0.25) = 0.06;
uniform float caustic_scale : hint_range(2.0, 30.0) = 14.0;
uniform float emission_energy : hint_range(0.5, 4.0) = 2.0;

void vertex() {
    // Dynamic bubbling and sludge undulating waves
    float w1 = sin(VERTEX.x * 2.8 + TIME * bubble_speed);
    float w2 = cos(VERTEX.z * 2.4 + TIME * bubble_speed * 0.85);
    float bubble = sin((VERTEX.x + VERTEX.z) * 4.0 + TIME * bubble_speed * 1.4) * 0.5 + 0.5;
    
    VERTEX.y += (w1 * w2 * wave_amplitude) + (bubble * wave_amplitude * 0.6);
    NORMAL = normalize(NORMAL + vec3(w1 * 0.1, 0.0, w2 * 0.1));
}

void fragment() {
    // Caustic multi-frequency noise
    vec2 c_uv = UV * caustic_scale;
    float c1 = sin(c_uv.x + sin(c_uv.y + TIME * bubble_speed * 0.7));
    float c2 = cos(c_uv.y + cos(c_uv.x + TIME * bubble_speed * 0.9));
    float caustic = clamp(c1 * c2 * 0.5 + 0.5, 0.0, 1.0);

    // Gradient blend between acid lime green and necrotic violet
    float blend_factor = sin(UV.x * 3.14 + TIME * 0.5) * 0.5 + 0.5;
    vec3 base_col = mix(toxic_color.rgb, mire_purple.rgb, blend_factor * 0.4);

    ALBEDO = base_col + vec3(caustic * 0.25);
    ROUGHNESS = 0.08; // Viscous liquid sheen
    METALLIC = 0.15;
    
    // Bioluminescent acid glow
    float pulse = sin(TIME * bubble_speed * 1.5) * 0.3 + 0.7;
    EMISSION = toxic_color.rgb * (caustic * emission_energy * pulse);
}
