# Krystal Stack: Technical & Mathematical Architecture

> **Language:** English (EN)  
> **Level:** Systems Architecture & Microarchitectural Specification  
> **Invariant:** `assert VITAL_MAX_HP == 6`

---

## 1. The Heterogeneous Execution Triad

Krystal Stack strictly segregates local physical execution tiers from remote asynchronous service planes:

$$\mathcal{E}_{\text{local}} = \{T_0, T_1, T_2, T_3\} \quad \text{and} \quad \mathcal{S}_{\text{remote}} = \{R_1\}$$

1. **Tier 0 ($T_0$) – DirectSim Reference Oracle:**  
   Deterministic integer rules, formal rollback arbiter, bit-exact verification.
2. **Tier 1 ($T_1$) – CPU AVX-512 / VNNI Fast-Path:**  
   Branch-heavy tactical logic, A* pathfinding, spatial BVH collision trees, and invariant gates.
3. **Tier 2 ($T_2$) – Iris Xe Vulkan Compute & DP4A:**  
   SDF raymarching, 4096-particle simulations, 10-block procedural scenes with $\le 256\,\text{MB}$ UMA ceiling.
4. **Tier 3 ($T_3$) – Dedicated Local Hardware NPU:**  
   Declared in accordance with physical silicon reality (on Tiger Lake i5: `PHYSICALLY_UNAVAILABLE`).
5. **Remote Tier ($R_1$) – Remote Neural Service:**  
   Cloud advisory intelligence (Gemini / Vertex AI) that never blocks game loop execution.

---

## 2. Dihedral Symmetry Folds ($D_n$) & Lipschitz Isometry

Procedural crystalline raymarching folds sample points $\mathbf{x} \in \mathbb{R}^3$ across $n$ reflection planes:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - 2 \min\big(0, \mathbf{x}_k \cdot \mathbf{n}_k\big) \mathbf{n}_k, \quad \mathbf{n}_k = \left(\cos\frac{k\pi}{n}, \sin\frac{k\pi}{n}, 0\right)$$
Since reflection is an orthogonal isometry, the transformation satisfies:
$$\|\mathbf{x}_{k+1} - \mathbf{y}_{k+1}\| = \|\mathbf{x}_k - \mathbf{y}_k\| \implies L = 1.0$$
preserving metric distances and preventing distance field overstepping.

---

## 3. Harmonic Cosine Topology & Radius Derivative

Procedural boundaries are governed by orthogonal trigonometric decomposition:
$$r(\theta) = R_0 + \sum_{k=1}^K A_k \cos(k \omega \theta + \phi_k)$$
$$\frac{dr}{d\theta} = -\sum_{k=1}^K A_k (k \omega) \sin(k \omega \theta + \phi_k)$$

The steepness metric $\xi(\theta) = \frac{|\frac{dr}{d\theta}|}{r(\theta)}$ dynamically gates rendering strategy:
* $\xi(\theta) \le 1.0 \implies$ 2.5D heightfield mesh.
* $\xi(\theta) > 1.0 \implies$ 3D SDF volumetric overhangs and cavern spires.
