# Krystal Stack: Technische & Mathematische Architektur

> **Sprache:** Deutsch (DE)  
> **Ebene:** Systemarchitektur & Mikroarchitektur  
> **Invariante:** `assert VITAL_MAX_HP == 6`

---

## 1. Die Heterogene Ausführungstriade

Krystal Stack trennt strikt lokale physische Rechenebenen von externen asynchronen Cloud-Diensten:

$$\mathcal{E}_{\text{local}} = \{T_0, T_1, T_2, T_3\} \quad \text{und} \quad \mathcal{S}_{\text{remote}} = \{R_1\}$$

1. **Tier 0 ($T_0$) – DirectSim Referenz-Orakel:**  
   Deterministische Integer-Regeln, formale Rollback-Autorität, bit-exakte Verifikation.
2. **Tier 1 ($T_1$) – CPU AVX-512 / VNNI Fast-Path:**  
   Verzweigungsintensive KI-Logik, A*-Wegfindung, räumliche BVH-Bäume und Sicherheitsgatter.
3. **Tier 2 ($T_2$) – Iris Xe Vulkan Compute & DP4A:**  
   SDF-Raymarching, 4096-Partikelsimulation, 10-Block-Szenen mit UMA-Grenze $\le 256\,\text{MB}$.
4. **Tier 3 ($T_3$) – Dedizierte lokale Hardware-NPU:**  
   Mathematisch deklariert entsprechend der tatsächlichen Hardware (Tiger Lake i5: `PHYSICALLY_UNAVAILABLE`).
5. **Remote Tier ($R_1$) – Externer Neuronaler Dienst:**  
   Asynchrone Cloud-Modelle (Gemini / Vertex AI), die die Ausführung des lokalen Kerns niemals blockieren.

---

## 2. Dieder-Faltung ($D_n$) & Lipschitz-Isometrie

Das prozedurale Raymarching spiegelt Abtastpunkte $\mathbf{x} \in \mathbb{R}^3$ an $n$ Spiegelebenen:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - 2 \min\big(0, \mathbf{x}_k \cdot \mathbf{n}_k\big) \mathbf{n}_k, \quad \mathbf{n}_k = \left(\cos\frac{k\pi}{n}, \sin\frac{k\pi}{n}, 0\right)$$
Da jede Reflexion eine orthogonale Isometrie darstellt, gilt Lipschitz-Konstante $L = 1.0$, wodurch metrische Verzerrungen im Distanzfeld (SDF) verhindert werden.

---

## 3. Kosinus-Topologie & Radius-Ableitung

Geländeprofile basieren auf orthogonaler trigonometrischer Zerlegung:
$$r(\theta) = R_0 + \sum_{k=1}^K A_k \cos(k \omega \theta + \phi_k)$$
$$\frac{dr}{d\theta} = -\sum_{k=1}^K A_k (k \omega) \sin(k \omega \theta + \phi_k)$$

Das Steigungsmaß $\xi(\theta) = \frac{|\frac{dr}{d\theta}|}{r(\theta)}$ steuert das Renderverfahren:
* $\xi(\theta) \le 1.0 \implies$ 2.5D Höhenfeld (Heightmap).
* $\xi(\theta) > 1.0 \implies$ Echtes 3D-SDF-Raymarching für Überhänge und Höhlensysteme.
