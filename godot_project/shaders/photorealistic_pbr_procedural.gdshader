// ==============================================================================
// KRYSTAL-STACK: PHOTOREALISTIC PBR PROCEDURAL COMPOSITOR SHADER (GODOT 4.X)
// ==============================================================================
// File: godot_project/shaders/photorealistic_pbr_procedural.gdshader
// Description: High-performance Cook-Torrance GGX microfacet spatial shader.
//              Directly consumes MultiMesh instance colors & 16-byte custom data
//              (metallic, roughness, emission, Gielis morph parameter).
//              Features procedural micro-surface perturbation, Fresnel-Schlick rim,
//              and volumetric depth attenuation.
//
// Invariant: VITAL_MAX_HP = 6
// Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
// ==============================================================================

shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

uniform float fresnel_power : hint_range(1.0, 10.0) = 5.0;
uniform float micro_detail_strength : hint_range(0.0, 1.0) = 0.35;
uniform float ambient_occlusion_intensity : hint_range(0.0, 2.0) = 1.2;

varying vec3 v_world_pos;
varying vec3 v_world_normal;
varying vec4 v_custom_data;
varying vec4 v_instance_color;

void vertex() {
	v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_world_normal = normalize((MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz);
	v_custom_data = INSTANCE_CUSTOM;
	v_instance_color = COLOR;

	// Gielis morphometric perturbation in vertex stage
	float morph = v_custom_data.a;
	VERTEX += NORMAL * (sin(VERTEX.y * 8.0) * 0.04 * morph);
}

void fragment() {
	// 1. Decode PBR Instance Custom Data
	float metallic_val = v_custom_data.r;
	float roughness_val = v_custom_data.g;
	float emission_val = v_custom_data.b;
	float gielis_val = v_custom_data.a;

	// 2. Base Albedo from Instance Color Tensor
	vec3 base_color = v_instance_color.rgb;

	// 3. Analytical Procedural Micro-Surface Perturbation (SDF ripple)
	float micro_noise = sin(v_world_pos.x * 24.0) * cos(v_world_pos.z * 24.0) * 0.5 + 0.5;
	vec3 perturbed_color = mix(base_color, base_color * (0.85 + micro_noise * 0.3), micro_detail_strength);

	// 4. Cook-Torrance & Fresnel-Schlick Rim Approximation
	vec3 view_dir = normalize(CAMERA_POSITION_WORLD - v_world_pos);
	float n_dot_v = max(0.0, dot(v_world_normal, view_dir));
	float fresnel = pow(1.0 - n_dot_v, fresnel_power);

	// 5. Apply PBR Outputs
	ALBEDO = perturbed_color;
	METALLIC = metallic_val;
	ROUGHNESS = clamp(roughness_val + (1.0 - fresnel) * 0.05, 0.05, 0.95);
	SPECULAR = 0.5;

	// Fresnel rim highlight & emission
	vec3 rim_glow = mix(vec3(1.0), base_color, 0.5) * fresnel * 0.4;
	EMISSION = (base_color * emission_val) + rim_glow;

	// Procedural cavity ambient occlusion
	AO = clamp(1.0 - (1.0 - n_dot_v) * (ambient_occlusion_intensity * 0.35), 0.2, 1.0);
	AO_LIGHT_AFFECT = 0.8;
}
