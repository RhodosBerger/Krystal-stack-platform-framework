shader_type canvas_item;

// Krystal-Stack: Holographic ASCII Screen-Space Shader
// Framework: Godot 4.x (Vulkan / Mobile Compatibility)
// Converts any 3D Viewport into a multi-chromatic ASCII hologram.

uniform sampler2D screen_texture : hint_screen_texture, filter_nearest;
uniform sampler2D depth_texture : hint_depth_texture, filter_nearest;

uniform vec2 cell_size = vec2(8.0, 16.0); // Font cell dimensions in pixels
uniform float edge_threshold : hint_range(0.05, 1.0) = 0.22;
uniform float holographic_glow : hint_range(0.5, 3.0) = 1.4;
uniform float scanline_frequency : hint_range(10.0, 300.0) = 120.0;
uniform float chromatic_disparity : hint_range(0.0, 0.02) = 0.005;

// Holographic color grading
uniform vec3 tint_near = vec3(0.0, 0.95, 1.0);  // Cyan
uniform vec3 tint_far = vec3(1.0, 0.05, 0.45);  // Magenta

void fragment() {
    vec2 screen_res = 1.0 / SCREEN_PIXEL_SIZE;
    
    // 1. Quantize screen into ASCII character blocks
    vec2 cell_coord = floor(FRAGCOORD.xy / cell_size);
    vec2 cell_center_uv = (cell_coord * cell_size + cell_size * 0.5) / screen_res;
    vec2 cell_local_uv = fract(FRAGCOORD.xy / cell_size); // 0.0 to 1.0 within cell

    // 2. Chromatic Aberration Sampling (Holographic 3D Anaglyph)
    float r = texture(screen_texture, cell_center_uv + vec2(chromatic_disparity, 0.0)).r;
    float g = texture(screen_texture, cell_center_uv).g;
    float b = texture(screen_texture, cell_center_uv - vec2(chromatic_disparity, 0.0)).b;
    vec3 base_color = vec3(r, g, b);

    // 3. Luminance & Edge Detection
    float luma = dot(base_color, vec3(0.299, 0.587, 0.114));
    
    // Sample 4-neighborhood to compute spatial gradient
    vec2 texel = cell_size / screen_res;
    float luma_up    = dot(texture(screen_texture, cell_center_uv + vec2(0.0, -texel.y)).rgb, vec3(0.333));
    float luma_down  = dot(texture(screen_texture, cell_center_uv + vec2(0.0,  texel.y)).rgb, vec3(0.333));
    float luma_left  = dot(texture(screen_texture, cell_center_uv + vec2(-texel.x, 0.0)).rgb, vec3(0.333));
    float luma_right = dot(texture(screen_texture, cell_center_uv + vec2( texel.x, 0.0)).rgb, vec3(0.333));
    
    float edge_x = luma_right - luma_left;
    float edge_y = luma_down - luma_up;
    float edge_mag = sqrt(edge_x * edge_x + edge_y * edge_y);

    // 4. Procedural Glyph Synthesis (Rasterizing ASCII glyphs inside each cell)
    float glyph_alpha = 0.0;
    
    if (edge_mag > edge_threshold) {
        // Directional stroke
        float angle = atan(edge_y, edge_x);
        float deg = degrees(angle);
        if (deg < 0.0) deg += 180.0;
        
        if (deg >= 67.5 && deg < 112.5) {
            // Vertical line '|'
            glyph_alpha = smoothstep(0.18, 0.0, abs(cell_local_uv.x - 0.5));
        } else if (deg >= 22.5 && deg < 67.5) {
            // Diagonal '/'
            glyph_alpha = smoothstep(0.18, 0.0, abs(cell_local_uv.x + cell_local_uv.y - 1.0));
        } else if (deg >= 112.5 && deg < 157.5) {
            // Diagonal '\'
            glyph_alpha = smoothstep(0.18, 0.0, abs(cell_local_uv.x - cell_local_uv.y));
        } else {
            // Horizontal '-'
            glyph_alpha = smoothstep(0.18, 0.0, abs(cell_local_uv.y - 0.5));
        }
    } else {
        // Density block dots/rings
        if (luma > 0.75) {
            // Dense block '█'
            glyph_alpha = 1.0;
        } else if (luma > 0.45) {
            // Cross '+' or center square
            glyph_alpha = (abs(cell_local_uv.x - 0.5) < 0.35 && abs(cell_local_uv.y - 0.5) < 0.35) ? 0.8 : 0.0;
        } else if (luma > 0.15) {
            // Center dot '.'
            glyph_alpha = length(cell_local_uv - vec2(0.5)) < 0.25 ? 0.6 : 0.0;
        } else {
            glyph_alpha = 0.0; // Space
        }
    }

    // 5. Holographic Scanlines & Interference
    float scanline = sin(FRAGCOORD.y * scanline_frequency / screen_res.y + TIME * 4.0) * 0.15 + 0.85;
    vec3 holo_tint = mix(tint_near, tint_far, sin(TIME * 0.5 + cell_center_uv.y * 3.0) * 0.5 + 0.5);

    vec3 final_rgb = base_color * holo_tint * glyph_alpha * scanline * holographic_glow;
    
    // Add ambient holographic baseline glow
    final_rgb += holo_tint * 0.03 * scanline;

    COLOR = vec4(final_rgb, 1.0);
}
