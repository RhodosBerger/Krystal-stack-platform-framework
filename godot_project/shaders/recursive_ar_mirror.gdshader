shader_type canvas_item;

// Krystal-Stack: Recursive AR Mirror & Dihedral Symmetry Shader
// Framework: Godot 4.x (Vulkan / Forward+ / Compatibility)
// Applies recursive kaleidoscopic IFS folding, Fresnel reflections,
// and chromatic aberration derived from Antigravity Prompt Engine.

uniform sampler2D screen_texture : hint_screen_texture, filter_nearest;

// Uniforms driven by Antigravity Templates & Vulkan Push Constants
uniform int u_mirror_folds : hint_range(2, 16) = 6;
uniform float u_fold_angle = 0.5235987; // pi / u_mirror_folds
uniform int u_recursion_limit : hint_range(1, 16) = 8;
uniform float u_fresnel_factor : hint_range(0.1, 1.0) = 0.88;
uniform float u_chromatic_aberration : hint_range(0.0, 0.05) = 0.008;
uniform float u_scanline_density : hint_range(10.0, 300.0) = 120.0;
uniform vec4 u_primary_color : source_color = vec4(0.0, 0.941, 1.0, 1.0);  // Cyan
uniform vec4 u_accent_color : source_color = vec4(1.0, 0.0, 0.333, 1.0);   // Neon Magenta

uniform vec2 cell_size = vec2(8.0, 16.0); // ASCII character cell dimensions
uniform float mirror_zoom : hint_range(0.5, 5.0) = 1.25;
uniform float rotation_speed : hint_range(-2.0, 2.0) = 0.35;

vec2 fold_dihedral(vec2 uv, int folds) {
    // Center around origin
    vec2 p = uv - vec2(0.5);
    
    // Convert to polar
    float r = length(p) * mirror_zoom;
    float theta = atan(p.y, p.x) + (TIME * rotation_speed);
    
    // Dihedral symmetry fold (D_N)
    float sector = 3.1415926535 / float(max(1, folds));
    float mod_theta = abs(mod(theta, 2.0 * sector) - sector);
    
    // Reconstruct folded Cartesian point
    vec2 folded = vec2(cos(mod_theta), sin(mod_theta)) * r;
    return folded + vec2(0.5);
}

void fragment() {
    vec2 screen_res = 1.0 / SCREEN_PIXEL_SIZE;
    
    // 1. Quantize coordinates into character grid cells
    vec2 cell_coord = floor(FRAGCOORD.xy / cell_size);
    vec2 cell_center_uv = (cell_coord * cell_size + cell_size * 0.5) / screen_res;
    vec2 cell_local_uv = fract(FRAGCOORD.xy / cell_size);

    // 2. Perform Recursive Mirror Reflections
    vec2 current_uv = cell_center_uv;
    vec3 accumulated_color = vec3(0.0);
    float reflection_weight = 1.0;
    float total_weight = 0.0;

    for (int i = 0; i < u_recursion_limit; i++) {
        // Fold coordinates across symmetry axes
        vec2 folded_uv = fold_dihedral(current_uv, u_mirror_folds);
        
        // Chromatic aberration pass
        float r_sample = texture(screen_texture, folded_uv + vec2(u_chromatic_aberration * float(i + 1), 0.0)).r;
        float g_sample = texture(screen_texture, folded_uv).g;
        float b_sample = texture(screen_texture, folded_uv - vec2(u_chromatic_aberration * float(i + 1), 0.0)).b;
        vec3 pass_sample = vec3(r_sample, g_sample, b_sample);

        // Approximate Fresnel reflection decay: R = R0 + (1 - R0) * (1 - cos(theta))^5
        float fresnel = u_fresnel_factor + (1.0 - u_fresnel_factor) * pow(clamp(length(folded_uv - vec2(0.5)) * 1.5, 0.0, 1.0), 5.0);
        
        // Accumulate weighted reflection pass
        accumulated_color += pass_sample * reflection_weight * fresnel;
        total_weight += reflection_weight;

        // Recursive contraction scale for next bounce
        current_uv = fract(folded_uv * 1.6180339 - vec2(0.309));
        reflection_weight *= 0.55;
    }

    if (total_weight > 0.0) {
        accumulated_color /= total_weight;
    }

    // 3. Luminance and Edge Quantization
    float luma = dot(accumulated_color, vec3(0.299, 0.587, 0.114));
    
    // Procedural ASCII Glyph Alpha within cell
    float glyph_alpha = 0.0;
    if (luma > 0.7) {
        glyph_alpha = 1.0; // Block
    } else if (luma > 0.45) {
        // Diamond / Cross '◇' / '+'
        float dist_manhattan = abs(cell_local_uv.x - 0.5) + abs(cell_local_uv.y - 0.5);
        glyph_alpha = smoothstep(0.45, 0.35, dist_manhattan);
    } else if (luma > 0.2) {
        // Subtle central rune dot '·'
        glyph_alpha = length(cell_local_uv - vec2(0.5)) < 0.2 ? 0.7 : 0.0;
    } else {
        glyph_alpha = 0.05; // Faint background field
    }

    // 4. Color Palette Synthesis
    vec3 graded_palette = mix(u_primary_color.rgb, u_accent_color.rgb, sin(TIME * 1.2 + luma * 3.1415) * 0.5 + 0.5);
    
    // Scanline & Laser Interference
    float scanline = sin(FRAGCOORD.y * u_scanline_density / screen_res.y + TIME * 3.0) * 0.12 + 0.88;

    vec3 final_rgb = accumulated_color * graded_palette * glyph_alpha * scanline;
    // Add specular rim glow from Fresnel reflections
    final_rgb += u_accent_color.rgb * pow(luma, 3.0) * 0.3;

    COLOR = vec4(final_rgb, 1.0);
}
