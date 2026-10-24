# KRYSTAL UNIFIED SOVEREIGN RUNTIME & KOORDINAČNÉ INTERPOLÁCIE

**Špecifikácia a Architektonický Manifest:** `SOVEREIGN-UNIFIED-RUNTIME-V1`  
**Dizajnová Filozofia:** *"Podľa nášho gusta"* (Striktná realizácia KSPP v1 ústavných princípov P1–P10)  
**Ústavný Invariant:** `VITAL_MAX_HP == 6` ($\neg\text{Valid}(A) \implies \Delta S \equiv 0$)  
**Implementácia:** [`krystal_kernel/sovereign_runtime.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_runtime.py)  
**Verifikácia:** [`tests/test_sovereign_runtime.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_sovereign_runtime.py) (8/8 PASS) | Plná regresia: 94/94 PASS  
**Theory DAG:** `THEORY-SOVEREIGN-UNIFIED-RUNTIME` (`DOKAZANE`)

---

## 1. Architektonická Vízia: Runtime „Podľa Nášho Gusta“

Doterajšie runtime prostredia v priemysle trpia chronickými chorobami:
- **Nekontrolovaný memory bloat & garbage collector pauzy:** Jitter v rádoch desiatok milisekúnd.
- **Naivné lineárne interpolácie (Lerp):** Spôsobujú trhanie farieb, energetickú nestabilitu a numerické prepady terénu.
- **Desynchronizácia taktovania:** Asynchrónne buffery zapĺňajú pamäťovú zbernicu UMA práve vtedy, keď GPU skenuje obraz na displej.
- **Neexistencia ústavných invariantov:** Tiché chyby prepisujú globálny stav.

**Krystal Unified Sovereign Runtime** bol postavený na mieru našim overeným teóriám a princípom:

```mermaid
graph TD
    subgraph "Sovereign Coordination Channels (128B SPSC Rings)"
        WSL[WSL Bridge] <-->|128B Zero-Copy| SOC[Chrono-Symplectic SoC]
        GODOT[Godot HoloBridge] <-->|128B Zero-Copy| SOC
        TERRAIN[Terrain Manifold] <-->|128B Zero-Copy| GODOT
        SOC <-->|128B Zero-Copy| LLM[Slovak LLM Cascade]
        LLM <-->|128B Zero-Copy| VPP[VPP Grid Broker]
        VPP <-->|128B Zero-Copy| SOC
    end

    subgraph "Adapted 4-Way Interpolation Suite"
        I1[Spatial Hex-HSL Interpolator]
        I2[Cadence Harmonic Interpolator]
        I3[Symplectic Phase-Space Interpolator]
        I4[Voltage Impedance Interpolator]
    end

    subgraph "2-Phase Shadow Cycle Engine"
        C1[Phase 1: Shadow Dispatch (VBlank 3.5ms)] --> C2{Invariant Gate: VITAL_MAX_HP == 6}
        C2 -->|Valid| C3[Phase 2: Commit State & Sign HMAC]
        C2 -->|Breach / Over-budget| C4[Delta S == 0 Rollback]
    end

    SOC --> I1
    SOC --> I2
    SOC --> I3
    SOC --> I4
    I1 --> C1
    I2 --> C1
    I3 --> C1
    I4 --> C1
```

---

## 2. Protokoly Vzájomnej Koordinácie (128B SPSC Kanály)

Komunikácia medzi subsystémami neprebieha cez reťazce ani JSON, ale výhradne cez **Universal 128B Lingua Franca** ([`ZeroCopyContractFrame`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/zero_copy_contracts.py)):
1. **Zarovnanie:** Presne dve 64-bajtové cache linky procesora (`_Alignas(64)`).
2. **Kryptografická Integrita:** 96-bajtový prefix chránený HMAC-SHA256 podpisom v bajtoch 96..127.
3. **Kruhová Pamäť (SPSC Ring):** Bounded kapacita 64 rámcov na kanál. Najstaršie správy sú pri preplnení deterministicky uvoľňované.
4. **Kanálová Topológia:**
   - `WSL_TO_CHRONO_SOC`: Telemetria z Linux jadra a `/dev/dxg`.
   - `GODOT_TO_CHRONO_SOC`: Požiadavky na holoprojekciu a geometrické transformácie.
   - `TERRAIN_TO_GODOT`: Prúd symplektických eróznych dát a výškových polí.
   - `CHRONO_SOC_TO_CASCADE`: Plánovanie dávok inferencie v tieni VBlank.
   - `CASCADE_TO_VPP`: Telemetria spotreby energie na vygenerovaný token.
   - `VPP_TO_CHRONO_SOC`: Spätné riadenie napäťových limitov a RAPL rozpočtov.

---

## 3. Prispôsobené Matematické Interpolácie (Adapted Interpolators)

Namiesto jednorozmerného naivného lerpu systém implementuje 4 špecializované interpolátory prispôsobené fyzike konkrétnych riešení:

### 3.1. Priestorový Hex-HSL Interpolátor (`SpatialHexHslInterpolator`)
- **Úloha:** Prechod medzi spojitým 3D svetom a diskrétnou 6-sektorovou mriežkou farieb.
- **Polohová Interpolácia:** Kubický Hermiteov smoothstep:
  $$s(t) = 3t^2 - 2t^3 \quad \text{pre } t \in [0.0, 1.0]$$
  $$P(t) = P_1 + s(t) \cdot (P_2 - P_1)$$
- **Farebná Interpolácia na Kružnici:** Najkratšia uhlová vzdialenosť na Hue kružnici zabraňuje farebným anomáliám (prechod cez čiernu alebo opačné spektrum):
  $$\Delta H = (H_2 - H_1) \pmod{1.0}, \quad \text{ak } \Delta H > 0.5 \implies \Delta H \gets \Delta H - 1.0$$
  $$H(t) = (H_1 + s(t) \cdot \Delta H) \pmod{1.0}$$
- **Kompaktné 32-bitové Balenie:** Výsledok generuje priamo `0xHHSSLLAA` u32 a priraďuje sektor `0x10` (Red) až `0x60` (Cyan).

### 3.2. Harmonický Taktový Interpolátor (`CadenceHarmonicInterpolator`)
- **Úloha:** Prechod medzi 8 harmonickými rytmami ($\alpha=100\text{ ms}$, $\beta=50\text{ ms}$, $\gamma=25\text{ms}$, $\Omega=125\text{ms}$, VBlank $16.66\text{ ms}$, Leapfrog, Dopamine combat $\tau_0 = 850\text{ ms}$).
- **Princíp P8 (12-Frame Dwell Hysteresis):** Zmena taktu vyžaduje minimálne $N_{\text{dwell}} = 12$ po sebe idúcich požiadaviek pred povýšením, čím sa eliminuje zbernicový a kognitívny tras (thrashing).
- **Rekurentný Filter Merkle-Momentum:**
  $$M_k = 0.85 M_{k-1} + 0.15 T_{\text{target}}$$

### 3.3. Symplektický Fázový Interpolátor (`SymplecticPhaseSpaceInterpolator`)
- **Úloha:** Fyzika a sedimentárna erózia terénu.
- **2. Rád Störmer-Verlet Leapfrog:**
  $$p_{t+1/2} = p_t - \frac{\Delta t}{2} \nabla V(q_t)$$
  $$q_{t+1} = q_t + \Delta t \frac{p_{t+1/2}}{m}$$
  $$p_{t+1} = p_{t+1/2} - \frac{\Delta t}{2} \nabla V(q_{t+1})$$
- **Konzervatívny Invariant:**
  $$\det(\mathbf{J}) \equiv 1.0 \implies \text{Objem fázového priestoru sa nemení.}$$
  Hamiltonián $E = E_{\text{kin}} + E_{\text{pot}}$ zostáva striktne ohraničený bez numerického kolapsu.

### 3.4. Napäťový Impedančný Interpolátor (`VoltageImpedanceInterpolator`)
- **Úloha:** Zamedzenie napäťových rázov pri prechodoch zbernice UMA do burst režimu.
- **RLC Exponenciálne Tlmenie:**
  $$V(t) = V_{\text{target}} + (V_0 - V_{\text{target}}) \cdot e^{-\Delta t / \tau_{\text{RC}}}$$
  kde $\tau_{\text{RC}} = Z_0 \cdot C_{\text{bus}} \approx 5.0\text{ ns}$ pri impedancii $Z_0 = 50.0\ \Omega$.
- **Indukčná Spätná EMF:**
  $$V_{\text{ind}} = -L \frac{dI}{dt}$$
  Pri prudkom náraste prúdu sa napätie dynamicky zovrie (clamp) do bezpečného koridoru $\pm 150\text{ mV}$.

---

## 4. Dvojfázový Tieňový Exekučný Cyklus (Two-Phase Shadow Engine)

Cyklus beží synchronizovane so zobrazovacím panelom (60 FPS / $16.666\text{ ms}$):
- **Phase 1: Shadow Dispatch ($3.5\text{ ms}$ rozpočet):** Výpočty prebiehajú striktne počas vertikálneho zatemnenia (VBlank), kedy je UMA zbernica voľná ($\mu_{\text{UMA}} < 0.78$).
- **Phase 2: Reconcile & Commit:**
  - Ak výpočet stihol $3.5\text{ ms}$ a `vital_max_hp == 6`: Zmena stavu je autorizovaná a podpísaná HMAC-SHA256 podpisom (`state_committed = True`).
  - Ak došlo k prečerpaniu rozpočtu alebo prekročeniu UMA tlaku: Akcia je okamžite zamietnutá a stav ostáva nezmenený ($\Delta S \equiv 0$).

---

## 5. Výsledky Testovacej Validácie

- **Jednotkové Testy:** [`tests/test_sovereign_runtime.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_sovereign_runtime.py) (**8/8 testov úspešných** za 0.001 s).
- **Plná Regresná Sada Repozitára:** **94/94 testov úspešných** za 0.024 s bez jedinej chyby.
- **Teoretický Graf (Theory DAG):** 27 uzlov v stave `DOKAZANE`, 0 cyklov, 0 varovaní.
