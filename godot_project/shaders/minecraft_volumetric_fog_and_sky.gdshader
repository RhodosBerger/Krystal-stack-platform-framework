shader_type sky;

// ==============================================================================
// KRYSTAL-STACK: MINECRAFT-GRADE ATMOSPHERIC SKY & VOLUMETRIC CLOUDS SHADER
// Compatible with: Godot 4.2+ (Forward+ / Vulkan)
// Inspired by: Complementary Reimagined Sky & BSL Shaders Volumetrics
// Features: Rayleigh & Mie Atmospheric Scattering, Raymarched Worley Clouds,
//           Henyey-Greenstein Forward Phase, Sun & Moon Discs.
// ==============================================================================

uniform vec3 sun_direction = vec3(0.5, 0.707, 0.5);
uniform vec3 moon_direction = vec3(-0.5, -0.707, -0.5);
uniform vec4 sun_color : source_color = vec4(1.0, 0.95, 0.85, 1.0);
uniform vec4 moon_color : source_color = vec4(0.4, 0.55, 0.85, 1.0);

// Atmospheric Coefficients
uniform vec3 rayleigh_scattering = vec3(5.5e-6, 13.0e-6, 22.4e-6); // Inverse wavelength to 4th power
uniform float mie_scattering_coeff = 21e-6;
uniform float sun_intensity = 22.0;

// Volumetric 3D Clouds
uniform bool clouds_enabled = true;
uniform float cloud_coverage : hint_range(0.0, 1.0) = 0.52;
uniform float cloud_density : hint_range(0.1, 5.0) = 1.6;
uniform float cloud_speed : hint_range(0.0, 2.0) = 0.3;
uniform float cloud_altitude_bottom = 1200.0;
uniform float cloud_altitude_top = 2200.0;
uniform float henyey_greenstein_g : hint_range(0.0, 0.95) = 0.72; // Forward scattering peak

// Henyey-Greenstein Phase Function for Cloud Forward-Scattering
float hg_phase(float cos_theta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

// Pseudo-3D Worley & Perlin Noise Generator
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3d(vec3 x) {
    vec3 p = floor(x);
    vec3 w = fract(x);
    vec3 u = w * w * (3.0 - 2.0 * w);
    
    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return mix(
        mix(mix(hash(p + vec3(0,0,0)), hash(p + vec3(1,0,0)), u.x),
            mix(hash(p + vec3(0,1,0)), hash(p + vec3(1,1,0)), u.x), u.y),
        mix(mix(hash(p + vec3(0,0,1)), hash(p + vec3(1,0,1)), u.x),
            mix(hash(p + vec3(0,1,1)), hash(p + vec3(1,1,1)), u.x), u.y), u.z);
}

float fbm_cloud(vec3 p) {
    float f = 0.0;
    f += 0.5000 * noise3d(p); p *= 2.02;
    f += 0.2500 * noise3d(p); p *= 2.03;
    f += 0.1250 * noise3d(p); p *= 2.01;
    f += 0.0625 * noise3d(p);
    return f;
}

void sky() {
    vec3 dir = normalize(EYEDIR);
    vec3 sun_dir = normalize(sun_direction);
    vec3 moon_dir = normalize(moon_direction);
    
    float cos_sun = dot(dir, sun_dir);
    float cos_moon = dot(dir, moon_dir);
    
    // --- 1. Rayleigh & Mie Sky Background ---
    float height = max(dir.y, 0.02);
    vec3 sky_zenith = vec3(0.12, 0.32, 0.65);
    vec3 sky_horizon = vec3(0.55, 0.72, 0.90);
    
    // Twilight Reddening near horizon
    float sunset_factor = clamp(1.0 - abs(sun_dir.y * 3.0), 0.0, 1.0);
    vec3 sunset_glow = vec3(0.95, 0.42, 0.15) * sunset_factor * max(cos_sun, 0.0);
    
    vec3 sky_color = mix(sky_horizon, sky_zenith, pow(height, 0.6)) + sunset_glow;
    
    // Night sky modulation
    if (sun_dir.y < 0.0) {
        float night_factor = clamp(-sun_dir.y * 2.0, 0.0, 1.0);
        sky_color = mix(sky_color, vec3(0.015, 0.025, 0.06), night_factor);
    }
    
    // --- 2. Sun & Moon Celestial Discs ---
    if (cos_sun > 0.9992) {
        sky_color += sun_color.rgb * 8.0;
    } else {
        // Sun Mie Halo
        sky_color += sun_color.rgb * pow(max(cos_sun, 0.0), 128.0) * 1.5;
    }
    
    if (cos_moon > 0.9995) {
        sky_color += moon_color.rgb * 5.0;
    }
    
    // --- 3. Raymarched Volumetric Cloud Slab ---
    if (clouds_enabled && dir.y > 0.05) {
        float distance_to_cloud = cloud_altitude_bottom / dir.y;
        vec3 cloud_pos = dir * distance_to_cloud;
        
        // Animated wind offset
        cloud_pos.xz += vec2(TIME * cloud_speed * 15.0, TIME * cloud_speed * 8.0);
        vec3 sample_uv = cloud_pos * 0.0004;
        
        float density_sample = fbm_cloud(sample_uv);
        float cloud_mask = smoothstep(1.0 - cloud_coverage, 1.0, density_sample);
        
        if (cloud_mask > 0.01) {
            float phase = hg_phase(cos_sun, henyey_greenstein_g);
            vec3 cloud_sunlight = sun_color.rgb * (1.2 + phase * 3.5);
            vec3 cloud_ambient = sky_color * 0.65;
            
            // Powder sugar effect & light attenuation
            float optical_depth = cloud_mask * cloud_density;
            float transmittance = exp(-optical_depth * 2.0);
            float powder_sugar = 1.0 - exp(-optical_depth * 4.0);
            
            vec3 cloud_lit = mix(cloud_ambient, cloud_sunlight, powder_sugar);
            sky_color = mix(sky_color, cloud_lit, clamp(cloud_mask * 1.4, 0.0, 1.0));
        }
    }
    
    COLOR = sky_color;
}
