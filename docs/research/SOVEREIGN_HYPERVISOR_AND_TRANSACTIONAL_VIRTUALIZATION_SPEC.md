# KRYSTAL STACK: SOVEREIGN HYPERVISOR & TRANSACTIONAL VIRTUALIZATION SPECIFICATION
## Paravirtualization Control Plane, Domain-Lock Arbitration, 2x2 Factorial Ablation & Capability Gates

**Dokument ID:** `KRYSTAL-RESEARCH-HYPERVISOR-SPEC-2026-V1`  
**Klasifikácia:** Paravirtualizačný riadiaci model, Transakčná virtualizácia, Doménová arbitráž  
**Autor:** Antigravity Cognitive Invariants Group & Krystal Architecture Council (2026)  
**Ústavný Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6 \land (\neg\text{Valid}(H) \implies \Delta S_{\text{authoritative}} \equiv 0)$  
**Záväzok grafu teórií:** Graf závislostí teórií zostáva **striktne zmrazený na 56 uzloch** (0 cyklov, žiadny uzol 57).  
**Dôkazový status:** `VERIFIED_WITH_SCOPE` (125/125 testov zelených, 0.091s) | Kompilátor: 6/6 fáz autorizovaných (1.55 ms)

---

## 1. Exekutívny súhrn & Architektonický model

Tento dokument formalizuje architektúru **Sovereign Hypervisor Toolsetu** ako transakčnej riadiacej roviny virtualizácie (Transactional Virtualization Control Plane) nad hosťovskými doménami:

$$\boxed{
\begin{array}{c}
\text{Guest Proposal} \longrightarrow \text{Authority Check} \longrightarrow \text{Domain Lock} \longrightarrow \text{Reversibility Gate} \\
\longrightarrow \text{Hypercall Capability} \longrightarrow \text{Execute} \longrightarrow \text{Validate} \longrightarrow \text{Commit / Compensate}
\end{array}
}$$

### 1.1 Zásadná epistemická hranica rozsahu (Epistemic Scope Boundary)
Súčasný modul [`sovereign_hypervisor_toolset.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_hypervisor_toolset.py) implementuje softvérový a paravirtualizačný riadiaci model nad procesmi Windows/Linux:

$$\boxed{\text{Klasifikácia: } [\text{PARAVIRTUALIZATION CONTROL MODEL}] \quad \lor \quad [\text{HYPERVISOR-READY}]}$$

**Striktné vymedzenie:** Tento modul **sám osebe nedokazuje hardvérovú izoláciu (Hardware Virtualization Isolation)** v zmysle procesorového režimu Ring -1 (Intel VT-x VMX root/non-root, EPT/NPT, IOMMU/VT-d interrupt remapping, VM exit traps). Preto je termín „hardware-isolated hypervisor“ vyhradený až pre budúcu fázu natívneho kremíka KSAC alebo nízkoúrovňového mikrokernela s IOMMU ovládačom.

---

## 2. Doménovo orientované zamykanie (Domain-Scoped Locking vs. Global Mutex)

Namiesto monolitického globálneho zámku hypervízora (`GlobalHypervisorMutex`), ktorý vedie k serializácii nezávislých úloh, hypervízor prenáša koncept `TurnToken(domain)` priamo na úroveň hypercallov:

$$\boxed{\text{Concurrent}(A, B) \iff \text{Locks}(A) \cap \text{Locks}(B) = \emptyset}$$

Dve hosťovské domény (napr. `DOMU_VULKAN_GRAPHICS` a `DOMU_LLM_COGNITION`) postupujú paralelne, pokiaľ ich požadované množiny prostriedkov nemajú prienik:
- Vulkan pristupuje k `GPU_QUEUE_0` a hexadecimálnym riadkom `04–07`.
- LLM pristupuje k `LLM_CACHE_0` a hexadecimálnym riadkom `08–11`.
- Prienik je prázdny $\implies$ súbežná exekúcia bez čakania.
- Ak nastane konflikt (napr. obaja žiadajú `SHARED_DMA_RING_BUFFER`), Turn Coordinator serializuje výhradne sporný zdroj bez zablokovania nesúvisiacich subsystémov (ako je CNC spindle riadenie v Dom0).

---

## 3. Trojstupňová reverzibilita a kompenzácia DMA: $\Delta S_{\text{authoritative}} = 0$ vs. $\Delta S_{\text{physical}} = 0$

Riadiaca rovina rozlišuje 3 triedy reverzibility operácií:

1. **`REVERSIBLE`:**
   Čisto softvérové alebo pamäťové mutácie v rámci hosťa. Pri zlyhaní sa aplikuje okamžitý lokálny rollback pamäte.
2. **`COMPENSATABLE`:**
   Operácie so zbernicovým side-effectom (DMA burst, I/O packet). Pri zlyhaní sa aktivuje registrovaná kompenzačná akcia (rewind DMA ring pointerov, invalidácia neúplných rámcov).
   
   $$\boxed{\text{COMPENSATABLE Fault} \implies \text{ConsistencyRestored}}$$
   
   *Vedecká presnosť:* Ak DMA radič fyzicky vykonal transakciu na zbernici, táto udalosť historicky nastala ($\Delta S_{\text{physical}} \neq 0$). Kompenzačná operácia obnovuje **logickú konzistenciu autoritatívneho stavu** ($\Delta S_{\text{authoritative}} = 0$), pričom fyzický svet nesie kompenzačnú stopu.
3. **`IRREVERSIBLE`:**
   Fyzické akcie (zapnutie CNC vretena, laserový rez, nevratné sieťové odoslanie).
   Token je odmietnutý ešte pred začatím exekúcie, ak operácia neobsahuje explicitnú pre-commit validáciu:
   $$\text{Irreversible}(A) \land \neg\text{PreCommitValidated}(A) \implies \text{RejectAndHalt} \quad (\Delta S \equiv 0)$$

---

## 4. Diagnostické metriky: VUTE a TUD

Tradičné vyťaženie časového slotu (Turn Efficiency - TE):
$$\text{TE} = \frac{\tau_{\text{exec}}}{Q_{\text{granted}}}$$
samotné nepostačuje, pretože operácia s $\text{TE} = 0.98$ mohla skončiť chybou, invalidáciou alebo porušením invariantu.

Hypervízor zavádza prísnejšie telemetrické metriky:

### 4.1 Verified Useful Time Efficiency (VUTE)
$$\boxed{\text{VUTE} = \frac{\tau_{\text{validated}}}{Q_{\text{granted}}}}$$
Pri rollbacku alebo zlyhaní validácie klesá $\text{VUTE} \equiv 0.0$. Vysoká hodnota VUTE zaručuje, že pridelený hardvérový čas vyprodukoval skutočnú, verifikovanú prácu.

### 4.2 Policy-Weighted Utility Density (TUD)
$$\boxed{\text{TUD} = \text{Priority} \times \text{VUTE}}$$
*Epistemické upresnenie:* TUD nie je fyzikálna energetická hustota, ale **administratívne/politicky vážená hustota úžitku (Policy-Weighted Utility Density)**, kde `Priority` predstavuje váhu priradenú riadiacou politikou domény.

---

## 5. 2x2 Faktoriálny ablačný experiment (Ablation Decomposition)

Ablačný nástroj [`HypervisorComponentAblationTool`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_hypervisor_toolset.py) testuje hypervízor v 4 konfiguráciách:

- **$H_0$:** Statické kvantum ($500\,\mu\text{s}$) + Monolitický hypervisor zámok (Serial Dom0/DomU) $\implies U_0 = 0.9936$.
- **$H_1$:** Adaptívne kvantum (EMA + UMA) + Monolitický hypervisor zámok (Serial) $\implies U_1 = 1.4941$.
- **$H_2$:** Statické kvantum ($500\,\mu\text{s}$) + Paravirtualizované doménové tokeny (Concurrent) $\implies U_2 = 1.6848$.
- **$H_3$:** Adaptívne kvantum (EMA + UMA) + Paravirtualizované doménové tokeny (Full Agile) $\implies U_3 = 2.5334$.

### 5.1 Výpočet prírastkov (Gains):
$$G_{\text{quantum}} = \frac{U_1}{U_0} - 1.0 = \frac{1.4941}{0.9936} - 1.0 \approx +50.37\%$$
$$G_{\text{parallelism}} = \frac{U_2}{U_0} - 1.0 = \frac{1.6848}{0.9936} - 1.0 \approx +69.57\%$$
$$G_{\text{total}} = \frac{U_3}{U_0} - 1.0 = \frac{2.5334}{0.9936} - 1.0 \approx +154.97\%$$

### 5.2 Analýza interakcie (Aditívna vs. Multiplikatívna):
1. **Aditívna škála (Classical 2x2 Factorial Interaction):**
   $$I_{\text{add}} = U_3 - U_2 - U_1 + U_0 = 2.5334 - 1.6848 - 1.4941 + 0.9936 = 0.3481$$
   $$\frac{I_{\text{add}}}{U_0} = \frac{0.3481}{0.9936} \approx +35.04\%$$
   Relatívne k $H_0$ baseline prináša aditívna interakcia $+35.04\%$, čo potvrdzuje, že $G_{\text{total}} = G_{\text{quantum}} + G_{\text{parallelism}} + I_{\text{rel}}$ ($50.37\% + 69.57\% + 35.04\% = 154.98\%$).
2. **Multiplikatívna škála:**
   $$U_{\text{expected}} = \frac{U_1 \cdot U_2}{U_0} = \frac{1.4941 \times 1.6848}{0.9936} \approx 2.5330 \approx U_3 = 2.5334$$
   $$\frac{U_3}{U_{\text{expected}}} - 1.0 = \frac{2.5334}{2.5330} - 1.0 \approx +0.016\% \approx 0.0\%$$
   
   $$\boxed{+35.0\%\text{ additive interaction relative to } H_0; \text{ multiplicative interaction approximately neutral.}}$$
   
   Tento výsledok má zásadnú metodickú hodnotu: **adaptívne kvantá a doménový paralelizmus sú multiplikatívne nezávislé faktory**. Ich spojenie netrpí vzájomnou deštrukciou ani saturačným stropom, ale správa sa takmer presne ako ideálny ortogonálny produkt ziskov.

---

## 6. Logické rozdelenie pamäte UMA (Logical UMA Resource Partitioning)

Mriežka 16x16 hexadecimálnych buniek ($32\,\text{KiB}$ L1D stopa, 256 buniek $\times$ 128B) je paravirtualizovane rozdelená medzi domény:
- Riadky `00–03` (64 buniek): `DOM0_CENTRAL_MACHINE`
- Riadky `04–07` (64 buniek): `DOMU_VULKAN_GRAPHICS`
- Riadky `08–11` (64 buniek): `DOMU_LLM_COGNITION`
- Riadky `12–13` (32 buniek): `DOMU_WSL_NETWORK`
- Riadok `14`     (16 buniek): `DOMU_TERRAIN_MESH`
- Riadok `15`     (16 buniek): `SHARED_DMA_RING_BUFFER`

$$\boxed{\text{Klasifikácia: } [\text{LOGICAL UMA RESOURCE PARTITION}]}$$

Toto rozdelenie slúži ako softvérová kontrolná rovina a auditné orakulum. Premenu na fyzickú izoláciu zabezpečí až prepojenie na IOMMU mapovanie stránok a hardvérové cache coloring/CAT registre.

---

## 7. HypercallTurnToken ako Proof-Carrying Capability

Token [`HypercallTurnToken`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_hypervisor_toolset.py) nie je obyčajný identifikátor, ale nositeľ formálneho kryptografického oprávnenia:

```text
HypercallTurnToken
─────────────────────────────────────────────
token_id:            HMAC-SHA256 digest
guest_domain:        HypervisorGuestDomain
resource_domains:    FrozenSet[ResourceDomain]
allocated_quantum:   float (us)
allocated_energy:    float (uJ)
reversibility:       OperationReversibilityClass
pre_commit_status:   bool
nonce:               uint64
expires_at:          float (timestamp)
signature:           HMAC(K_receipt, payload)
```

Základný bezpečnostný kontrakt:
$$\boxed{\text{Execute}(H) \implies \text{Verify}(\text{Token}_H)}$$
$$\neg\text{Verify}(\text{Token}_H) \implies \text{DenyHypercall} \land \Delta S_{\text{authoritative}} \equiv 0$$

Týmto sa prepája celý Krystal reťazec do jedného capability modelu:
$$\boxed{\text{SovereignPromptbook} \longrightarrow \text{KSL} \longrightarrow \text{AEP} \longrightarrow \text{Hypervisor} \longrightarrow \text{Hardware}}$$

---

## 8. Bezpečnostná zásada: IOMMU ako autorita nad fyzickým DMA

Ak hypervízor riadi skutočné hardvérové zariadenia, softvérový Python zámok nepostačuje na zastavenie kompromitovaného hosťa, ktorý má priamy prístup k DMA bus-masteringu.

V produkčnom KSAC/bare-metal stacku platí ústavné pravidlo:
$$\boxed{\text{DMARequest} \implies \text{IOMMU Permission Gate}}$$
$$\text{ResourceCapabilityToken} \longrightarrow \text{IOMMU Page Table Mapping} \longrightarrow \text{Physical DMA Engine}$$
a pre GPU akcelerátory:
$$\text{TurnToken} \longrightarrow \text{Queue / Engine Context Ownership} \longrightarrow \text{Hardware Command Stream Submission}$$

---

## 9. Zjednotená trojúrovňová sémantika exekúcie (Unified Tri-Level Execution Semantics)

Najdôležitejším architektonickým zistením výskumu je, že Krystal Stack nevytvára tri odlišné ad-hoc schedulery. Používa **jediný unifikovaný exekučný kontrakt** na troch rôznych úrovniach abstrakcie:

```
              ┌────────────────────────────────────────────────────────┐
              │  ÚROVEŇ 1: Application Operation (SovereignPromptbook) │
              └───────────────────────────┬────────────────────────────┘
                                          │
                                          ▼
              ┌────────────────────────────────────────────────────────┐
              │  ÚROVEŇ 2: Subsystem Turn (TurnCoordinator / Governor) │
              └───────────────────────────┬────────────────────────────┘
                                          │
                                          ▼
              ┌────────────────────────────────────────────────────────┐
              │  ÚROVEŇ 3: Hypervisor Hypercall (HypervisorToolset)    │
              └────────────────────────────────────────────────────────┘
```

Na všetkých troch úrovniach platia identické štyri zákony:
1. **$\text{Proposal} \neq \text{Authority}$:** Klient, subsystém aj hosť iba navrhujú akciu.
2. **$\text{Capability precedes execution}$:** Exekúcia vyžaduje platný kryptografický token.
3. **$\text{Irreversible} \implies \text{PreCommitValidation}$:** Nevratné zásahy vyžadujú dôkaz pred spustením.
4. **$\text{Commit} \implies \text{Receipt}$:** Každý stavový prechod emituje zreťazený nemenný receipt.

---

## 10. Návrh hardvérového validačného experimentu (Next CDT)

Pre transformáciu z `[PARAVIRTUALIZATION CONTROL MODEL]` do `[HARDWARE-ISOLATED]` navrhujeme nasledujúci najlacnejší rozlišujúci test (CDT):

### Experiment: Hardware-Gated Capability Enforcement (HGCE-01)
- **Cieľ:** Preukázať, že hosťovská doména `DOMU_WSL_NETWORK` nedokáže zapísať do pamäte `DOM0_CENTRAL_MACHINE` ani v prípade, že je softvérový Python arbiter bypassnutý.
- **Mechanizmus:** Prepojenie `HypercallTurnToken` s hardvérovou IOMMU maskou alebo FPGA BRAM mailbox registrom s bitovou šírkou zodpovedajúcou sektorom `SubsystemNodeId`.
- **Falzifikátor ($H_0$):** Ak zápis prejde bez hardvérového trap interruptu pri absencii tokenu, hypotéza o hardvérovej autorite je falzifikovaná.
- **Úspech ($H_1$):** Hardvérový bus fault nastane do $1$ hodinového cyklu bez poškodenia stavu Dom0 ($\Delta S \equiv 0$).
