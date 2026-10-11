// =====================================================================
// KRYSTAL-STACK: AETHERIC VORTEX SWIRL SHADER (STUDNA DUŠÍ)
// Godot 4.2+ Forward+ / Vulkan
// Specialized for: Studna Duší (Well of Souls - Central Island Vortex)
// Features: Polar logarithmic swirl, depth fade, radiant soul emission
// =====================================================================
shader_type spatial;
render_mode blend_add, depth_draw_always, cull_disabled, unshaded;

uniform vec4 core_color : source_color = vec4(0.0, 0.95, 1.0, 1.0);
uniform vec4 rim_color : source_color = vec4(0.4, 0.1, 0.9, 0.8);
uniform float swirl_speed : hint_range(0.2, 5.0) = 1.8;
uniform float spiral_arms : hint_range(1.0, 8.0) = 3.0;
uniform float vortex_depth : hint_range(0.5, 6.0) = 2.5;

void fragment() {
    // Convert UV to centered polar coordinates [-1, 1]
    vec2 centered_uv = UV * 2.0 - 1.0;
    float r = length(centered_uv);
    float angle = atan(centered_uv.y, centered_uv.x);

    // Discard outside circular well boundary
    if (r > 1.0) {
        discard;
    }

    // Logarithmic spiral distortion
    float spiral = angle * spiral_arms + log(max(0.01, r)) * vortex_depth - (TIME * swirl_speed);
    float arm_intensity = sin(spiral) * 0.5 + 0.5;

    // Radiant gradient from center to outer rim
    float center_glow = pow(1.0 - r, 2.2);
    vec3 mixed_color = mix(rim_color.rgb, core_color.rgb, center_glow + arm_intensity * 0.4);

    ALBEDO = mixed_color * (1.2 + center_glow * 2.5);
    ALPHA = clamp((center_glow * 1.5 + arm_intensity * 0.5) * (1.0 - pow(r, 4.0)), 0.0, 1.0);
}
