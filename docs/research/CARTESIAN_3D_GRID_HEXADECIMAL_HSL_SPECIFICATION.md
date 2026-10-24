# 3D Cartesian Grid Coefficients on Hexadecimal HSL Spectrum

**Project:** Krystal-Stack Platform Framework  
**Document ID:** `KRYSTAL-RESEARCH-CARTESIAN-HEX-HSL-01`  
**Classification:** Computational Geometry, Color Science & Spatial Invariants  
**Author:** Dušan Kopecký & Krystal-Stack Architecture Council (2026)  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Epistemic Verdict:** `DOKAZANE` / `VERIFIED_WITH_SCOPE` (51/51 testov zelených)  

---

## 1. Executive Summary

Používateľská požiadavka definuje úlohu:
> *„Definovať koeficienty 3D gridu na hexadecimálnom HSL spektre karteziánskych bodov.“*

Tento dokument a sprievodný modul [**`krystal_kernel/cartesian_hex_hsl_grid.py`**](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cartesian_hex_hsl_grid.py) formalizujú transformáciu spojitého alebo diskrétneho 3D karteziánskeho priestoru $(x, y, z) \in \mathbb{R}^3$ do plne kvantizovaného **hexadecimálneho HSL spektra** (Hue, Saturation, Lightness), ktoré zachováva ústavný invariant mriežky `VITAL_MAX_HP == 6` rozdelením do **6 symetrických hexadecimálnych sektorov** po $60^\circ$.

---

## 2. Matematická formulácia priestorových koeficientov 3D gridu

Každý karteziánsky bod $\mathbf{P} = (x, y, z)$ sa najprv rozloží do sférických a polárnych veličín:
$$r_{xz} = \sqrt{x^2 + z^2}, \quad r_{xyz} = \sqrt{x^2 + y^2 + z^2}$$
$$\theta = \operatorname{atan2}(z, x) \in [-\pi, \pi], \quad \phi = \operatorname{asin}\left(\frac{y}{\max(10^{-6}, r_{xyz})}\right)$$

Transformácia je riadená parametrickou triedou [**`GridCoefficients3D`**](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cartesian_hex_hsl_grid.py#L65):

### 2.1 Priestorový odtieň (Hue $H \in [0.0, 1.0)$)
Odtieň spája lineárny karteziánsky gradient, azimutálny vír, radiálne harmonické vlnenie a fázu zlatého rezu $\Phi$:

$$H(x, y, z) = \left( \Phi_{\text{phase}} + \underbrace{\alpha_x x + \alpha_y y + \alpha_z z}_{\text{Lineárny karteziánsky sklon}} + \underbrace{\frac{\omega_{\text{rot}}}{2\pi} \theta}_{\text{Azimutálny vír}} + \underbrace{\kappa_{\text{harm}} \sin\left(\frac{2\pi r_{xz}}{\lambda_{\text{wave}}}\right)}_{\text{Radiálne harmonické vlnenie}} \right) \bmod 1.0$$

* $\alpha_x = 0.035\ \text{m}^{-1}$, $\alpha_y = 0.050\ \text{m}^{-1}$, $\alpha_z = 0.025\ \text{m}^{-1}$: smerové priestorové gradienty.
* $\omega_{\text{rot}} = 0.750$: váha uhlového víru okolo osi $Y$.
* $\lambda_{\text{wave}} = 12.0\text{ m}$: priestorová vlnová dĺžka harmonických vĺn.
* $\kappa_{\text{harm}} = 0.125$: hĺbka harmonickej modulácie.
* $\Phi_{\text{phase}} = \frac{1 + \sqrt{5}}{2} \approx 1.6180339887$: kotva zlatého rezu.

### 2.2 Priestorová sýtosť (Saturation $S \in [0.0, 1.0]$)
Sýtosť modeluje radiálny útlm od stredu gridu smerom k okrajom s bezpečnostnými prahmi:

$$S(x, y, z) = \operatorname{clamp}\Big( \beta_{\text{base}} - \beta_{\text{decay}} \cdot r_{xz}, \ \beta_{\text{min}}, \ \beta_{\text{max}} \Big)$$

* $\beta_{\text{base}} = 0.85$: maximálna centrálna sýtosť.
* $\beta_{\text{decay}} = 0.015\ \text{m}^{-1}$: radiálny koeficient desaturácie.
* $\beta_{\text{min}} = 0.20$: spodný prah sýtosti (zabraňuje úplnému zošednutiu).
* $\beta_{\text{max}} = 1.00$: horný saturačný strop.

### 2.3 Priestorová svetlosť a hĺbková plasticita (Lightness $L \in [0.0, 1.0]$)
Svetlosť zohľadňuje výškový gradient (vertikálne osvetlenie) a Fresnelovu hĺbkovú plasticitu:

$$L(x, y, z) = \operatorname{clamp}\Big( \gamma_{\text{base}} + \gamma_y \cdot y + \gamma_{\text{fresnel}} \cdot \left(1 - e^{-0.1 \cdot r_{xz}}\right), \ \gamma_{\text{min}}, \ \gamma_{\text{max}} \Big)$$

* $\gamma_{\text{base}} = 0.50$: neutrálny stred jasu.
* $\gamma_y = 0.035\ \text{m}^{-1}$: vertikálny výškový gradient.
* $\gamma_{\text{fresnel}} = 0.120$: hĺbkový plastický koeficient.
* $\gamma_{\text{min}} = 0.08$: ochrana pred čiernym orezaním (black crush).
* $\gamma_{\text{max}} = 0.92$: ochrana pred prepálením bielej (white clip).

---

## 3. Hexadecimálne HSL spektrum a 6 ústavných sektorov

Podľa ústavného pravidla $\text{VITAL\_MAX\_HP} \equiv 6$ je plné $360^\circ$ spektrum rozdelené do **6 rovnomerných sektorov po $60^\circ$**:

$$\text{SectorIndex} = \lfloor 6.0 \cdot H(x, y, z) \rfloor \bmod 6$$

| Sektor | Rozsah Hue | Hex Kód | Symbolický názov | Sémantický živel / Funkcia v Krystal Stacku |
| :---: | :---: | :---: | :--- | :--- |
| **0** | $0^\circ \le \theta < 60^\circ$ | `0x10` | `RED_EMBER` | Termálna záťaž, kinetická energia, ohňový element |
| **1** | $60^\circ \le \theta < 120^\circ$ | `0x20` | `AMBER_GOLD` | Ekonomická hodnota, zlatý rez, výnosový tok |
| **2** | $120^\circ \le \theta < 180^\circ$ | `0x30` | `EMERALD_BIO` | Bio-homeostáza, vegetácia, druidský bióm |
| **3** | $180^\circ \le \theta < 240^\circ$ | `0x40` | `CYAN_CRYSTAL` | Krystalická optika, HeNe laserová interferencia, vlnoplocha |
| **4** | $240^\circ \le \theta < 300^\circ$ | `0x50` | `SAPPHIRE_VOID` | Pamäťová zbernica UMA, Level-Zero USM, fázový tieň |
| **5** | $300^\circ \le \theta < 360^\circ$ | `0x60` | `AMETHYST_AETHER` | Epistemická reflexia, SLM kognitívna úvaha, psyché |

---

## 4. Dátové formáty a binárne kódovanie

Výsledok evaluácie každého karteziánskeho bodu [**`CartesianGridPoint`**](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cartesian_hex_hsl_grid.py#L105) poskytuje tri úrovne reprezentácie:

### 4.1 32-bitový hexadecimálny deskriptor HSLA (`0xHHSSLLAA`)
Kvantizácia každého kanála do jedného bajtu ($0x00$ – $0xFF$):
$$\text{hex\_hsla\_u32} = (\lfloor 255 \cdot H \rfloor \ll 24) \mid (\lfloor 255 \cdot S \rfloor \ll 16) \mid (\lfloor 255 \cdot L \rfloor \ll 8) \mid 0xFF$$

### 4.2 24-bitový reťazec sRGB (`#RRGGBB`)
Deterministická analytická konverzia $HSL \to RGB$:
$$C = (1 - |2L - 1|) \cdot S, \quad X = C \cdot (1 - |(6H \bmod 2) - 1|), \quad m = L - C/2$$
Prevod na hexadecimálny reťazec: `f"#{int(R*255):02X}{int(G*255):02X}{int(B*255):02X}"`.

### 4.3 96-bajtový MultiMesh GPU layout (Vulkan / Godot 4.x kompatibilita)
Každý bod možno serializovať priamo do pamäťového rámca pre hardvérové inštancovanie:
- **Offset 0..63 (16 floatov):** $4\times 4$ afinná transformačná matica (pozícia $(x, y, z)$ + uniformná škála).
- **Offset 64..79 (4 floaty):** Vektor farby $(R, G, B, A)$.
- **Offset 80..95 (4 floaty):** Zákaznícke shadovacie dáta: `[Hue, Saturation, Lightness, HexSectorCode]`.

### 4.4 128-bajtový Zero-Copy Contract Frame
Dávka karteziánskych bodov generuje [**`ZeroCopyContractFrame`**](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/zero_copy_contracts.py) s `contract_type = 0x04` (`COMPUTATIONAL_GEOMETRY`), SHA-256 prepojením a striktným zachovaním `VITAL_MAX_HP == 6`.

---

## 5. Podporované typy 3D gridov

Implementácia v triede [**`CartesianHexHslGrid`**](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cartesian_hex_hsl_grid.py#L135) podporuje tri kľúčové topológie generovania:
1. **`generate_cubic_lattice(dim_x, dim_y, dim_z, step)`:** Pravidelná 3D karteziánska kubická mriežka.
2. **`generate_fibonacci_phyllotaxis_lattice(point_count, radius_scale, height_span)`:** 3D valcová špirála na báze zlatého uhla $\theta_g = 2\pi (1 - 1/\Phi)$, garantujúca rovnomerný rozptyl bodov do všetkých 6 sektorov.
3. **`generate_hex_axial_lattice(radius_steps, hex_radius)`:** Axionálne hexagonálne pole $(q, r)$ projektované do 3D sínusového terénu.

---

## 6. Verifikácia a výsledky testov

Všetky invarianty a výpočtové vzorce boli overené v [**`tests/test_cartesian_hex_hsl_grid.py`**](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_cartesian_hex_hsl_grid.py):
- 8/8 testov špecifických pre 3D grid prebehlo za 0.002s.
- Celá regresná sada (51 testov v 8 moduloch) prebehla za 1.109s s výsledkom `OK`.
- Žiadne odchýlky, pretečenia ani zlyhania invariantu `VITAL_MAX_HP == 6`.
