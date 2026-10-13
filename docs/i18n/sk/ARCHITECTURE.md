# Krystal Stack: Technická a Matematická Architektúra

> **Jazyk:** Slovenský jazyk (SK)  
> **Úroveň:** Systémová a mikroskopická špecifikácia  
> **Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6$

---

## 1. Heterogénna Triáda Výpočtových Úrovní

Krystal Stack striktne oddeľuje lokálne fyzické exekučné jednotky od vzdialených služieb:

$$\mathcal{E}_{\text{local}} = \{T_0, T_1, T_2, T_3\} \quad \text{a} \quad \mathcal{S}_{\text{remote}} = \{R_1\}$$

1. **Tier 0 ($T_0$) – DirectSim Reference Oracle:**  
   Deterministické celočíselné herné pravidlá, formálna verifikácia, rollback autorita.
2. **Tier 1 ($T_1$) – CPU AVX-512 / VNNI Fast-Path:**  
   Branch-heavy AI logika, A* pathfinding, priestorové BVH stromy a bezpečnostné brány.
3. **Tier 2 ($T_2$) – Iris Xe Vulkan Compute & DP4A:**  
   SDF raymarching, simulácia 4096 častíc, 10-blokové procedurálne scény s UMA stropom $\le 256\,\text{MB}$.
4. **Tier 3 ($T_3$) – Dedikovaný Lokálny NPU:**  
   Matematicky deklarovaný podľa hostiteľského čipu (na Tiger Lake i5: `PHYSICALLY_UNAVAILABLE`).
5. **Remote Tier ($R_1$) – Vzdialená Neurónová Služba:**  
   Asynchrónny poradný cloudový model (Gemini / Vertex AI), nikdy nesmie blokovať herné jadro.

---

## 2. Dihedrálne Skladanie ($D_n$) a Lipschitzova Izometria

Raymarching procedurálnych kryštálov a symetrických arén využíva skladanie bodu $\mathbf{x} \in \mathbb{R}^3$ cez reflexné roviny:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - 2 \min\big(0, \mathbf{x}_k \cdot \mathbf{n}_k\big) \mathbf{n}_k, \quad \mathbf{n}_k = \left(\cos\frac{k\pi}{n}, \sin\frac{k\pi}{n}, 0\right)$$
Pretože reflexia je izometrická transformácia, Lipschitzova konštanta je identicky rovná $L = 1.0$, čo zabraňuje skresleniu vzdialenostných polí (SDF).

---

## 3. Kosínusová Topológia a Radius Derivát

Profil terénu je generovaný ortogonálnym rozvojom:
$$r(\theta) = R_0 + \sum_{k=1}^K A_k \cos(k \omega \theta + \phi_k)$$
$$\frac{dr}{d\theta} = -\sum_{k=1}^K A_k (k \omega) \sin(k \omega \theta + \phi_k)$$

Pomer strmosti $\xi(\theta) = \frac{|\frac{dr}{d\theta}|}{r(\theta)}$ automaticky prepína medzi 2.5D výškovou mapou ($\xi \le 1.0$) a plným 3D SDF raymarchingom previsov a jaskýň ($\xi > 1.0$).
