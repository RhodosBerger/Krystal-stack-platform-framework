# Krystal Stack: Technická a Matematická Architektura

> **Jazyk:** Český jazyk (CS)  
> **Úroveň:** Systémová a mikroarchitektonická specifikace  
> **Invariant:** `assert VITAL_MAX_HP == 6`

---

## 1. Heterogenní Výpočetní Triáda

Krystal Stack striktně odděluje lokální fyzické exekuční vrstvy od vzdálených asynchronních cloudových služeb:

$$\mathcal{E}_{\text{local}} = \{T_0, T_1, T_2, T_3\} \quad \text{a} \quad \mathcal{S}_{\text{remote}} = \{R_1\}$$

1. **Tier 0 ($T_0$) – DirectSim Referenční Orákulum:**  
   Deterministická celočíselná herní pravidla, formální rollback autorita, bitově přesná verifikace.
2. **Tier 1 ($T_1$) – CPU AVX-512 / VNNI Fast-Path:**  
   Větvená taktická logika, A* pathfinding, prostorové stromy BVH a invariantní brány.
3. **Tier 2 ($T_2$) – Iris Xe Vulkan Compute & DP4A:**  
   SDF raymarching, simulace 4096 částic, 10-blokové procedurální scény s limitem UMA $\le 256\,\text{MB}$.
4. **Tier 3 ($T_3$) – Dedikovaný Lokální Hardware NPU:**  
   Deklarováno podle skutečného křemíku (na Tiger Lake i5: `PHYSICALLY_UNAVAILABLE`).
5. **Remote Tier ($R_1$) – Vzdálená Neuronová Služba:**  
   Asynchronní cloudové modely (Gemini / Vertex AI), které nikdy neblokují lokální běhovou smyčku.

---

## 2. Dihedrální Skládání ($D_n$) a Lipschitzova Izometrie

Raymarching procedurálních krystalů skládá vzorkovací bod $\mathbf{x} \in \mathbb{R}^3$ přes $n$ zrcadlových rovin:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - 2 \min\big(0, \mathbf{x}_k \cdot \mathbf{n}_k\big) \mathbf{n}_k, \quad \mathbf{n}_k = \left(\cos\frac{k\pi}{n}, \sin\frac{k\pi}{n}, 0\right)$$
Jelikož každé zrcadlení představuje ortogonální izometrii, Lipschitzova konstanta transformace je $L = 1.0$, což vylučuje zkreslení vzdálenostního pole (SDF).

---

## 3. Kosinová Topologie a Radius Derivace

Geometrický profil scény vychází z trigonometrického rozvoje:
$$r(\theta) = R_0 + \sum_{k=1}^K A_k \cos(k \omega \theta + \phi_k)$$
$$\frac{dr}{d\theta} = -\sum_{k=1}^K A_k (k \omega) \sin(k \omega \theta + \phi_k)$$

Poměr strmosti $\xi(\theta) = \frac{|\frac{dr}{d\theta}|}{r(\theta)}$ řídí volbu renderovací strategie:
* $\xi(\theta) \le 1.0 \implies$ 2.5D výšková mapa terénu.
* $\xi(\theta) > 1.0 \implies$ Plný 3D SDF raymarching pro jeskynní převisy.
