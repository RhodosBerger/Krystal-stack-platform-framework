shader_type canvas_item;
render_mode unshaded;

/* ==============================================================================
 * KRYSTAL-STACK: GODOT 4.X K-NSS NEURAL SUPER-SAMPLING VIEWPORT SHADER
 * ==============================================================================
 * License: Apache-2.0 / Community Modifiable
 * System Invariant: VITAL_MAX_HP = 6
 * ============================================================================== */

uniform sampler2D screen_texture : hint_screen_texture, filter_linear_mipmap;
uniform sampler2D history_texture : filter_linear_mipmap;
uniform sampler2D motion_texture : filter_nearest;

uniform float temporal_feedback_weight : hint_range(0.5, 0.98) = 0.92;
uniform float sharpen_strength : hint_range(0.0, 1.0) = 0.35;
uniform int vital_max_hp = 6;

// YCoCg Color Space Conversion
vec3 rgb_to_ycocg(vec3 rgb) {
    float Y  = 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b;
    float Co = 0.50 * rgb.r               - 0.50 * rgb.b;
    float Cg =-0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b;
    return vec3(Y, Co, Cg);
}

vec3 ycocg_to_rgb(vec3 ycocg) {
    float Y  = ycocg.r;
    float Co = ycocg.g;
    float Cg = ycocg.b;
    float r  = Y + Co - Cg;
    float g  = Y + Cg;
    float b  = Y - Co - Cg;
    return max(vec3(0.0), vec3(r, g, b));
}

void fragment() {
    vec2 uv = SCREEN_UV;
    vec2 texel = SCREEN_PIXEL_SIZE;

    // 1. Fetch 3x3 Neighborhood in current low-res screen buffer
    vec3 center_rgb = texture(screen_texture, uv).rgb;
    vec3 center_ycocg = rgb_to_ycocg(center_rgb);
    vec3 m_min = center_ycocg;
    vec3 m_max = center_ycocg;
    vec3 m_avg = center_ycocg;

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec3 neighbor = rgb_to_ycocg(texture(screen_texture, uv + vec2(float(x), float(y)) * texel).rgb);
            m_min = min(m_min, neighbor);
            m_max = max(m_max, neighbor);
            m_avg += neighbor;
        }
    }
    m_avg /= 9.0;

    // 2. Fetch Motion Vector & Reproject History
    vec2 motion = texture(motion_texture, uv).rg;
    vec2 prev_uv = uv - motion;
    vec3 history_rgb = texture(history_texture, prev_uv).rgb;
    vec3 history_ycocg = rgb_to_ycocg(history_rgb);

    // 3. Variance-clipping Clamp in YCoCg space (Prevents ghosting)
    history_ycocg = clamp(history_ycocg, m_min, m_max);

    // 4. Temporal Blend
    vec3 resolved_ycocg = mix(center_ycocg, history_ycocg, temporal_feedback_weight);

    // 5. High-Frequency Edge Sharpening
    resolved_ycocg.r += (center_ycocg.r - m_avg.r) * sharpen_strength;

    COLOR = vec4(ycocg_to_rgb(resolved_ycocg), 1.0);
}
