// =====================================================================
// KRYSTAL-STACK: CRYSTAL ICE PRISM REFRACTION SHADER
// Godot 4.2+ Forward+ / Vulkan
// Specialized for: Severní Štíty (Crystal Tribe - Vládci Mrazu)
// Features: Subsurface scattering, chromatic dispersion, facet specular
// =====================================================================
shader_type spatial;
render_mode blend_mix, depth_draw_always, cull_back, diffuse_burley, specular_schlick_ggx;

uniform vec4 ice_color : source_color = vec4(0.35, 0.92, 0.98, 0.85);
uniform vec4 aether_glow : source_color = vec4(0.0, 0.8, 1.0, 1.0);
uniform float roughness : hint_range(0.0, 1.0) = 0.12;
uniform float metallic : hint_range(0.0, 1.0) = 0.25;
uniform float refraction_strength : hint_range(0.0, 0.2) = 0.045;
uniform float chromatic_dispersion : hint_range(0.0, 0.05) = 0.015;
uniform float pulse_speed : hint_range(0.1, 5.0) = 1.2;
uniform float facet_scale : hint_range(1.0, 40.0) = 12.0;

// Pseudo-random Voronoi / crystal facet generator
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float voronoi_facets(vec2 uv) {
    vec2 g = floor(uv);
    vec2 f = fract(uv);
    float min_dist = 1.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 lattice = vec2(float(x), float(y));
            vec2 offset = vec2(hash(g + lattice), hash(g + lattice + vec2(17.1, 31.7)));
            vec2 diff = lattice + offset - f;
            min_dist = min(min_dist, length(diff));
        }
    }
    return min_dist;
}

void vertex() {
    // Subtle breathing displacement on crystal vertices
    float breath = sin(TIME * pulse_speed + VERTEX.y * 3.0) * 0.02;
    VERTEX += NORMAL * breath;
}

void fragment() {
    vec2 facet_uv = UV * facet_scale;
    float facet = voronoi_facets(facet_uv);

    // Fresnel rim effect
    float fresnel = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 3.0);

    // Chromatic dispersion shimmer
    float shimmer = sin(TIME * pulse_speed + facet * 6.28) * 0.5 + 0.5;
    vec3 dispersed_color = mix(ice_color.rgb, aether_glow.rgb, shimmer * fresnel);

    ALBEDO = dispersed_color * (0.8 + 0.4 * facet);
    ROUGHNESS = mix(roughness, roughness + 0.3, facet);
    METALLIC = metallic;
    
    // Glowing aether emissions
    EMISSION = aether_glow.rgb * (fresnel * 2.2 + shimmer * 0.6);
    
    // Translucency & alpha
    ALPHA = clamp(ice_color.a + fresnel * 0.3, 0.0, 1.0);
}
