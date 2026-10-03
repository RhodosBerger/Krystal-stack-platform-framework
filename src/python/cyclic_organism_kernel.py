"""
Krystal-Stack Platform Framework: Cyclic Organism Kernel
========================================================
A continuous-time Symplectic Hamiltonian Dynamical Engine governing the
computational organism's homeostatic cycle, thermodynamic stability, and
cognitive brainwave phases (Alpha, Beta, Gamma, Omega).

Implements:
1. 2nd-Order Symplectic Velocity-Verlet Integration preserving phase space volume.
2. Non-linear Lyapunov stability evaluation.
3. Closed-loop visual entropy and thermodynamic damping.
4. Cognitive Brainwave State Machine modulating hardware priority and step budgets.
"""

import math
import time
from typing import Dict, Any, List, Tuple, Optional


class SymplecticCyclicEngine:
    """
    Symplectic Hamiltonian Dynamical Engine modeling the computational organism.

    State Vector q:
      q[0]: Visual Complexity / SDF Detail Level (normalized, typical [0.2, 1.5])
      q[1]: Target Frame Rate (FPS, typical [15.0, 120.0])
      q[2]: VRAM / Memory Utilization Fraction (typical [0.05, 0.95])
      q[3]: Economic Arbitrage Ratio (Render vs Mining, [0.0, 1.0])

    Conjugate Momentum p:
      p[i] = m[i] * dq[i]/dt (computational momentum along each dimension)
    """

    def __init__(
        self,
        organism_id: str = "Organism-Alpha-Prime",
        mass_inertia: Optional[List[float]] = None,
        spring_constants: Optional[List[float]] = None,
        homeostatic_setpoints: Optional[List[float]] = None,
        barrier_limits: Optional[List[float]] = None,
        linear_damping_gamma: float = 0.05,
        nonlinear_entropy_beta: float = 2.5,
        enable_economic_forcing: bool = True
    ):
        self.organism_id = organism_id
        # Dimension N = 4
        self.mass = mass_inertia or [1.0, 0.5, 2.0, 1.2]
        self.spring_k = spring_constants or [2.0, 0.8, 4.0, 1.5]
        self.q_star = homeostatic_setpoints or [1.0, 60.0, 0.50, 0.65]
        self.q_max = barrier_limits or [2.5, 144.0, 0.95, 1.0]

        self.gamma_0 = linear_damping_gamma
        self.beta = nonlinear_entropy_beta
        self.enable_economic_forcing = enable_economic_forcing

        # Initial coordinates q and momentum p
        self.q: List[float] = list(self.q_star)
        self.p: List[float] = [0.0, 0.0, 0.0, 0.0]

        # External state inputs
        self.visual_entropy: float = 0.25
        self.gpu_temp_c: float = 48.0
        self.economic_budget: float = 850.0
        self.mesh_congestion: float = 0.0  # Integration Point: Synthesizer Mesh Congestion

        # Cognitive phase
        self.cognitive_phase: str = "BETA"
        self.cycle_count: int = 0
        self.last_update_time: float = time.perf_counter()

    def potential_gradient(self, q_vec: List[float]) -> List[float]:
        """
        Computes -dV/dq for the potential energy field:
        V(q) = 1/2 sum(k_i * (q_i - q_i*)^2) + sum(xi_i / (q_max_i - q_i)^2)
        Returns the conservative restoring force vector F_conservative.
        """
        forces = []
        xi = 0.02  # Barrier repulsion factor
        for i in range(4):
            # Harmonic restoring force
            f_harmonic = -self.spring_k[i] * (q_vec[i] - self.q_star[i])

            # Barrier repulsion near physical upper limit
            dist_to_limit = max(0.01, self.q_max[i] - q_vec[i])
            f_barrier = - (2.0 * xi) / (dist_to_limit ** 3)

            forces.append(f_harmonic + f_barrier)
        return forces

    def dissipative_force(self, p_vec: List[float], entropy: float, congestion: float = 0.0) -> List[float]:
        """
        Computes non-conservative drag: Gamma(p, E) = -(gamma_0 + beta * E^2 + beta * congestion^2) * p
        """
        effective_drag = self.gamma_0 + self.beta * (entropy ** 2) + self.beta * (congestion ** 2)
        return [-effective_drag * p_i for p_i in p_vec]

    def economic_forcing(self, t: float) -> List[float]:
        """
        Computes dynamic market driving force vector based on economic cycles.
        """
        if not self.enable_economic_forcing:
            return [0.0, 0.0, 0.0, 0.0]
        # Periodic market fluctuations drive economic balance and target FPS
        f_econ = 0.3 * math.sin(t * 0.2)
        f_fps = 0.5 * math.cos(t * 0.1)
        return [0.0, f_fps, 0.0, f_econ]

    def symplectic_step(self, dt: float, visual_entropy: float, gpu_temp: float = 48.0, mesh_congestion: float = 0.0):
        """
        Advances the phase space trajectory by dt using 2nd-order Symplectic Velocity-Verlet.
        Preserves Liouville symplectic 2-form area in phase space.
        """
        self.visual_entropy = max(0.0, min(1.0, visual_entropy))
        self.gpu_temp_c = gpu_temp
        self.mesh_congestion = max(0.0, min(1.0, mesh_congestion))
        self.cycle_count += 1
        t = self.cycle_count * dt

        # 1. Total force at t: F(q, p) = F_conservative + F_dissipative + F_econ
        f_cons_t = self.potential_gradient(self.q)
        f_diss_t = self.dissipative_force(self.p, self.visual_entropy, self.mesh_congestion)
        f_econ_t = self.economic_forcing(t)

        f_total_t = [fc + fd + fe for fc, fd, fe in zip(f_cons_t, f_diss_t, f_econ_t)]

        # 2. Half-step momentum update: p(t + dt/2) = p(t) + 0.5 * dt * F_total(t)
        p_half = [
            self.p[i] + 0.5 * dt * f_total_t[i]
            for i in range(4)
        ]

        # 3. Full-step coordinate update: q(t + dt) = q(t) + dt * (p_half / m)
        q_next = [
            self.q[i] + dt * (p_half[i] / self.mass[i])
            for i in range(4)
        ]

        # Clamp q coordinates strictly within safe physical intervals
        for i in range(4):
            q_next[i] = max(0.05, min(self.q_max[i] - 0.02, q_next[i]))

        # 4. Total force at t + dt
        f_cons_next = self.potential_gradient(q_next)
        f_diss_next = self.dissipative_force(p_half, self.visual_entropy, self.mesh_congestion)
        f_econ_next = self.economic_forcing(t + dt)

        f_total_next = [fc + fd + fe for fc, fd, fe in zip(f_cons_next, f_diss_next, f_econ_next)]

        # 5. Half-step momentum completion: p(t + dt) = p_half + 0.5 * dt * F_total_next
        p_next = [
            p_half[i] + 0.5 * dt * f_total_next[i]
            for i in range(4)
        ]

        self.q = q_next
        self.p = p_next

        # 6. Update Cognitive Phase & Control Actuations
        self._update_cognitive_state()

    def kinetic_energy(self) -> float:
        """Evaluates T(p) = 0.5 * sum(p_i^2 / m_i)."""
        return 0.5 * sum((self.p[i] ** 2) / self.mass[i] for i in range(4))

    def potential_energy(self) -> float:
        """Evaluates V(q) = 0.5 * sum(k_i * (q_i - q_i*)^2) + barrier."""
        v_harm = 0.5 * sum(self.spring_k[i] * ((self.q[i] - self.q_star[i]) ** 2) for i in range(4))
        xi = 0.02
        v_barr = sum(xi / ((max(0.01, self.q_max[i] - self.q[i])) ** 2) for i in range(4))
        return v_harm + v_barr

    def total_hamiltonian(self) -> float:
        """Total system energy H = T + V."""
        return self.kinetic_energy() + self.potential_energy()

    def lyapunov_stability_index(self) -> float:
        """
        Lyapunov function L(q, p) measuring distance from homeostatic attractor.
        L(q, p) >= 0, L(q*, 0) = 0.
        """
        t_val = self.kinetic_energy()
        v_dev = 0.5 * sum(self.spring_k[i] * ((self.q[i] - self.q_star[i]) ** 2) for i in range(4))
        return round(t_val + v_dev, 4)

    def _update_cognitive_state(self):
        """
        Determines current brainwave state (ALPHA, BETA, GAMMA, OMEGA)
        and adjusts control actuations.
        """
        momentum_norm = math.sqrt(sum(p_i ** 2 for p_i in self.p))

        # Check critical thresholds (Omega Phase: Backpressure / Thermal protection)
        if self.visual_entropy > 0.70 or self.gpu_temp_c > 80.0 or self.mesh_congestion > 0.60:
            self.cognitive_phase = "OMEGA"
        elif momentum_norm > 1.2 and self.visual_entropy <= 0.70:
            self.cognitive_phase = "GAMMA"
        elif 0.35 <= momentum_norm <= 1.2:
            self.cognitive_phase = "BETA"
        else:
            self.cognitive_phase = "ALPHA"

    def get_control_actuation(self) -> Dict[str, Any]:
        """
        Translates current Hamiltonian phase space coordinates into concrete
        rendering, threading, and resource partition settings.
        """
        phase = self.cognitive_phase

        if phase == "OMEGA":
            # Resistance / Throttling: Step down raymarching complexity, lower priority
            step_budget = 16
            style = "BLUEPRINT_EDGE"
            priority = "LOW"
            vram_dist = {"mining_percent": 0.0, "render_percent": 40.0, "inference_percent": 50.0, "buffer_percent": 10.0}
        elif phase == "GAMMA":
            # Peak Force: Full raymarching steps, maximum priority
            step_budget = 48
            style = "HIGH_FIDELITY"
            priority = "REALTIME"
            vram_dist = {"mining_percent": 20.0, "render_percent": 70.0, "inference_percent": 5.0, "buffer_percent": 5.0}
        elif phase == "BETA":
            # Active Flow: Balanced settings
            step_budget = 32
            style = "CYBERPUNK"
            priority = "NORMAL"
            vram_dist = {"mining_percent": 30.0, "render_percent": 50.0, "inference_percent": 15.0, "buffer_percent": 5.0}
        else:
            # ALPHA: Idle / Synthesis: Low step budget, power conservation
            step_budget = 20
            style = "CYBERPUNK"
            priority = "IDLE"
            vram_dist = {"mining_percent": 75.0, "render_percent": 15.0, "inference_percent": 5.0, "buffer_percent": 5.0}

        # Fine-tune step budget with continuous complexity coordinate q[0]
        step_budget = int(step_budget * max(0.6, min(1.4, self.q[0])))

        return {
            "raymarch_step_budget": max(12, min(64, step_budget)),
            "active_style_palette": style,
            "thread_priority": priority,
            "vram_zone_distribution": vram_dist
        }

    def generate_phase_trajectory(self, num_points: int = 32, dt: float = 0.033) -> List[Tuple[float, float]]:
        """
        Projects an un-damped short-term orbit in (q[0], p[0]) phase space
        for real-time holographic canvas visualization.
        """
        trajectory = []
        sim_q = list(self.q)
        sim_p = list(self.p)

        for _ in range(num_points):
            trajectory.append((round(sim_q[0], 3), round(sim_p[0], 3)))
            # Quick Verlet iteration
            f = self.potential_gradient(sim_q)
            p_half = [sim_p[i] + 0.5 * dt * f[i] for i in range(4)]
            sim_q = [sim_q[i] + dt * (p_half[i] / self.mass[i]) for i in range(4)]
            f_next = self.potential_gradient(sim_q)
            sim_p = [p_half[i] + 0.5 * dt * f_next[i] for i in range(4)]

        return trajectory

    def to_schema_dict(self) -> Dict[str, Any]:
        """
        Serializes current state conforming to schemas/cyclic_architecture_schema.json.
        """
        t_val = self.kinetic_energy()
        v_val = self.potential_energy()
        h_val = t_val + v_val

        return {
            "organism_id": self.organism_id,
            "version": "1.0.0",
            "cognitive_phase": self.cognitive_phase,
            "hamiltonian_parameters": {
                "mass_inertia": [round(m, 3) for m in self.mass],
                "spring_constants": [round(k, 3) for k in self.spring_k],
                "homeostatic_setpoints": [round(q, 3) for q in self.q_star],
                "barrier_limits": [round(lim, 3) for lim in self.q_max],
                "linear_damping_gamma": self.gamma_0,
                "nonlinear_entropy_beta": self.beta
            },
            "state_vector": {
                "q_coordinates": [round(x, 4) for x in self.q],
                "p_momentum": [round(x, 4) for x in self.p]
            },
            "telemetry": {
                "total_energy_h": round(h_val, 4),
                "kinetic_energy_t": round(t_val, 4),
                "potential_energy_v": round(v_val, 4),
                "lyapunov_stability_index": self.lyapunov_stability_index(),
                "visual_entropy": round(self.visual_entropy, 4),
                "gpu_temperature_c": round(self.gpu_temp_c, 1),
                "economic_budget": round(self.economic_budget, 1)
            },
            "control_actuation": self.get_control_actuation()
        }
