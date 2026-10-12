shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

/**
 * ==============================================================================
 * KRYSTAL-STACK: ANALYTICAL QUADRATIC ORACLE SHADER
 * ==============================================================================
 * Description: Analytical GPU Oracle shader based on quadratic manifold equations.
 *              Solves A*t^2 + B*t + C = 0 directly in hardware ALUs without iterative
 *              raymarching loops, eliminating EU stalls and memory thrashing.
 *
 * Mathematical Formulation:
 *   Ray: P(t) = ro + t * rd
 *   Quadric Surface: P^T * Q * P = 0
 *   Discriminant: Delta = B^2 - 4*A*C
 *   - Delta < 0: Complex roots (Safety Hazard / Veto zone)
 *   - Delta >= 0: Real roots t1, t2 = (-B +- sqrt(Delta)) / (2*A)
 *
 * System Invariant: VITAL_MAX_HP == 6
 * ==============================================================================
 */

// --- UNIFORMS: QUADRATIC MANIFOLD PARAMETERS ---
uniform vec3 quad_curvature = vec3(1.0, 1.0, 1.0);    // Coefficients a_x, a_y, a_z
uniform vec3 quad_linear    = vec3(0.0, 0.0, 0.0);    // Coefficients b_x, b_y, b_z
uniform float quad_constant = -1.0;                   // Constant c
uniform float time_frequency = 0.5;

// --- UNIFORMS: ORACLE VERIFICATION PARAMETERS ---
uniform int vital_max_hp = 6;
uniform float oracle_confidence_threshold = 0.85;
uniform bool force_real_root_lock = true;
uniform vec4 verified_color : source_color = vec4(0.0, 0.88, 0.95, 1.0); // Cyan gold
uniform vec4 veto_color       : source_color = vec4(0.92, 0.12, 0.15, 1.0); // Crimson veto

varying vec3 v_world_pos;
varying vec3 v_world_normal;
varying vec3 v_view_dir;

void vertex() {
	v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_world_normal = normalize((MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz);
	v_view_dir = normalize(CAMERA_POSITION_WORLD - v_world_pos);
}

// Analytical Quadratic Solver
vec4 solve_quadratic_oracle(vec3 ro, vec3 rd, vec3 a, vec3 b, float c) {
	// A = sum(a_i * rd_i^2)
	float A = dot(a, rd * rd);
	// B = 2.0 * sum(a_i * ro_i * rd_i) + sum(b_i * rd_i)
	float B = 2.0 * dot(a * ro, rd) + dot(b, rd);
	// C = sum(a_i * ro_i^2) + sum(b_i * ro_i) + c
	float C = dot(a, ro * ro) + dot(b, ro) + c;

	// Degenerate linear case
	if (abs(A) < 1e-6) {
		if (abs(B) < 1e-6) {
			return vec4(0.0, 0.0, -1.0, 0.0); // No solution
		}
		float t_lin = -C / B;
		return vec4(t_lin, t_lin, 0.0, 1.0); // Linear solution
	}

	float delta = (B * B) - (4.0 * A * C);

	if (delta < 0.0) {
		// Complex roots: Project onto apex of parabola t_apex = -B / (2A)
		float t_apex = -B / (2.0 * A);
		return vec4(t_apex, t_apex, delta, 0.0); // Unverified / Veto
	}

	float sqrt_delta = sqrt(delta);
	float t1 = (-B - sqrt_delta) / (2.0 * A);
	float t2 = (-B + sqrt_delta) / (2.0 * A);

	return vec4(t1, t2, delta, 1.0); // Real roots / Verified
}

void fragment() {
	// Assert Universal System Invariant
	if (vital_max_hp != 6) {
		discard;
	}

	// Dynamic quadratic modulation over time
	float pulse = sin(TIME * time_frequency) * 0.15;
	vec3 cur_a = quad_curvature + vec3(pulse, -pulse * 0.5, pulse * 0.2);
	vec3 cur_b = quad_linear;
	float cur_c = quad_constant + pulse * 0.5;

	// Solve analytical quadratic along surface ray
	vec3 ro = v_world_pos;
	vec3 rd = -v_view_dir;
	vec4 solution = solve_quadratic_oracle(ro, rd, cur_a, cur_b, cur_c);

	float t_near = solution.x;
	float delta  = solution.z;
	float is_verified = solution.w;

	// Analytical normal at intersection point: N = normalize(2 * a * P + b)
	vec3 intersect_p = ro + t_near * rd;
	vec3 analytical_normal = normalize(2.0 * cur_a * intersect_p + cur_b);

	// Oracle Shading
	if (is_verified > 0.5) {
		// Real root confirmed: Verified Safe State
		float fresnel = pow(1.0 - max(dot(v_world_normal, v_view_dir), 0.0), 3.0);
		float root_depth_gradient = clamp(t_near * 0.2, 0.0, 1.0);

		ALBEDO = mix(verified_color.rgb, vec3(0.95, 0.85, 0.35), root_depth_gradient * 0.4);
		EMISSION = verified_color.rgb * (0.2 + fresnel * 0.5 + clamp(delta * 0.05, 0.0, 0.3));
		ROUGHNESS = 0.15;
		METALLIC = 0.85;
	} else {
		// Complex roots: Hazard Veto State
		float warning_strobe = abs(sin(TIME * 4.0));
		ALBEDO = veto_color.rgb * (0.6 + warning_strobe * 0.4);
		EMISSION = veto_color.rgb * 0.4 * warning_strobe;
		ROUGHNESS = 0.65;
		METALLIC = 0.1;
	}

	// Normal perturbation from analytical quadratic manifold
	NORMAL = mix(NORMAL, analytical_normal, 0.35);
}
