# Theory of Phase-Locked Epistemic Action (PLEA)
## Meta-Theory Axiom Pack (MTAP v1) & The PLEA Engine

> **Ústavný invariant:** $\text{VITAL\_MAX\_HP} == 6$  
> **Jadrový postulát:** *Poznanie v autonómnom systéme nie je statický graf textových uzlov, ale stojatá vlna v zdieľanej pamäti viazaná na fázový tieň (VBlank) hardvéru, s merateľnou kryptografickou zotrvačnosťou a topologickým vetvením pri empirickom rozpore.*

---

## I. Balík piatich základných teórií (Axiomatické reaktanty)

### Teória $\mathcal{T}_1$: Termodynamika epistemickej práce (Landauerov-Krystalov princíp)
* **Axióma:** Každé zvýšenie epistemického stavu tvrdenia v mriežke poznania ($\tau_{\text{low}} \to \tau_{\text{high}}$, napr. z `HYPOTHESIS` na `MEASURED_WITH_SCOPE`) je fyzikálny proces znižovania informačnej entropie $\Delta H$, ktorý vyžaduje nenulovú spotrebu hardvérového rozpočtu (čas na zbernici $\Delta t_{\text{UMA}}$, elektrickú prácu $\Delta E_{\text{PL1}}$ a tepelnú disipáciu $\Delta Q$).
* **Formálny tvar:**
  $$\Delta \tau > 0 \implies \int_{t_0}^{t_1} P_{\text{bus}}(t)\,dt \ge E_{\text{min}}(\Delta H) \quad \land \quad \text{Neoverená špekulácia} \equiv \text{Tepelný šum}$$
* **Systémový dôsledok:** Halucinácia alebo nepodložená hypotéza nie je len logická chyba – je to parazitický odber prúdu a šírky pásma LPDDR5, ktorý ohrieva kremík bez vytvorenia verifikovanej informačnej hodnoty.

---

### Teória $\mathcal{T}_2$: Topologická separácia konfliktov (Homotópia rozporov)
* **Axióma:** Keď dva kauzálne overené prúdy telemetrie z rovnakého časového okna vykazujú ortogonálne výsledky ($E_A \bot E_B$), stavový priestor systému má v danom bode netriviálnu topologickú prekážku (dieru/singularitu). Akýkoľvek pokus o lineárnu interpoláciu alebo spriemerovanie týchto tvrdení vedie do neplatného (nefyzikálneho) stavu mimo varietu systému.
* **Formálny tvar:**
  $$E_A \bot E_B \implies \frac{E_A + E_B}{2} \notin \mathcal{M}_{\text{valid}} \quad \text{(Zákaz konvexného spriemerovania pravdy)}$$
* **Systémový dôsledok:** Konflikt (`CONFLICT_NODE`) sa nesmie vyhladiť; musí sa uložiť ako rozvetvená topologická štruktúra (bifurkácia), ktorá existuje dovtedy, kým nový diskriminačný experiment nezrúti jednu z vetiev.

---

### Teória $\mathcal{T}_3$: Chronometrická fázová dualita (Teória fázového tieňa)
* **Axióma:** Deterministické vnímanie/vykresľovanie (Scanout / Actuation) a asynchrónna reflexia (Epistemic Reasoning) sú kanonicky združené premenné v čase. Nemôžu okupovať rovnakú fázu nosnej vlny zbernice bez vzájomnej deštruktívnej interferencie (stutter / torn-frame).
* **Formálny tvar:**
  $$\Phi_{\text{Actuation}}(t) \cdot \Phi_{\text{Epistemic}}(t) = 0 \quad \forall t \quad \text{(Podmienka ortogonálneho fázového multiplexu)}$$
* **Systémový dôsledok:** Kognitívna vrstva smie fyzicky existovať na zbernici výlučne vo „fázovom tieni“ (VBlank okná, medzisnímkové bubliny $2.2\text{ ms}$) deterministického jadra.

---

### Teória $\mathcal{T}_4$: Kryptografická kauzálna zotrvačnosť (Merkle-Momentum)
* **Axióma:** Každý 128-bajtový rámec nesie cez pole `predecessor_hash[32]` (rozštiepené cez CL0 a CL1) a `frame_hmac[32]` kumulatívnu históriu všetkých predchádzajúcich stavov. Dlhšie neprerušené reťazce fyzicky overených rámcov (`PHYSICAL-MEASURED`) získavajú vyššiu „epistemickú hmotnosť“ (zotrvačnosť) než izolované alebo čerstvo vygenerované špekulácie.
* **Formálny tvar:**
  $$M_{\text{epistemic}}(\text{Chain}_k) = \sum_{i=1}^{k} \gamma^{k-i} \cdot \text{Weight}(\text{Evidence}_i), \quad \gamma \in (0, 1)$$
* **Systémový dôsledok:** Krátkodobý šum alebo osamelý návrh od `PROPOSER` nemôže prevážiť ani prepísať trajektóriu s vysokou kauzálnou zotrvačnosťou bez úmerne silného experimentálneho dôkazu.

---

### Teória $\mathcal{T}_5$: Invariantná symetria prežitia (Noetherovej-Krystalova veta)
* **Axióma:** Každá spojitá symetria a bezpečnosť riadiaceho systému je zviazaná so zákonom zachovania ústavného invariantu $\text{VITAL\_MAX\_HP} == 6$. Akákoľvek transformácia (zníženie rozlíšenia DRS, stlačenie KV-cache, zmena PL1 limitu, rollback experimentu) je prípustná vtedy a len vtedy, ak je izometriou voči tomuto invariantu.
* **Formálny tvar:**
  $$\forall \mathcal{F} \in \text{Actions}: \quad \mathcal{I}(\mathcal{F}(S_t)) \equiv \mathcal{I}(S_t) \equiv 6 \quad \text{(kde } \mathcal{I} \text{ je operátor integrity)}$$

---

## II. Syntéza: Teória Fázovo-Viazanej Epistemickej Akcie (PLEA)

Keď týchto päť teórií ($\mathcal{T}_1 \times \mathcal{T}_2 \times \mathcal{T}_3 \times \mathcal{T}_4 \times \mathcal{T}_5$) reaguje spoločne, systém podlieha **základnej pohybovej rovnici Krystal Stacku**:

V každom VBlank okne $\Delta t_{\text{blank}}$ systém vyberá takú trajektóriu poznania a akcie $\pi^*$, ktorá minimalizuje **Epistemický Akčný Integrál $\mathcal{S}$**:

$$\boxed{\mathcal{S}[\pi] = \int_{t_{\text{blank}}} \Big( \underbrace{\alpha \cdot \text{UMA\_Cost}(\pi)}_{\text{z }\mathcal{T}_1\text{ a }\mathcal{T}_3} + \underbrace{\beta \cdot \text{TopoConflict}(\pi)}_{\text{z }\mathcal{T}_2} - \underbrace{\gamma \cdot \text{CausalInertia}(\pi)}_{\text{z }\mathcal{T}_4} \Big)\,dt \quad \text{pri väzbe } \underbrace{\text{HP}(\pi) \equiv 6}_{\text{z }\mathcal{T}_5}}$$

### Systémová interpretácia rovnice v reálnom čase:
1. **Nízke zaťaženie zbernice (UMA Idle, $\mu < 0.50$):**  
   Termodynamická cena $\text{UMA\_Cost}$ je minimálna $\implies$ systém automaticky investuje ušetrené milisekundy a watty do rozuzlenia topologických konfliktov ($\text{TopoConflict}$) cez mikro-experimenty v tieni.
2. **Vysoké zaťaženie zbernice (UMA Saturation, $\mu \to 0.78$):**  
   Termodynamická cena $\text{UMA\_Cost}$ prudko stúpa $\implies$ systém okamžite zmrazí špekulatívne vetvy s nízkou kauzálnou zotrvačnosťou ($\text{CausalInertia}$) a spolieha sa výlučne na vopred overené inerciálne reťazce v Tier 0 / Tier 1.

---

## III. Architektúra komponentov PLEA subsystému

| Komponent (Modul) | Zdrojové teórie | Presná funkcia v Krystal Stacku | Napojenie na 128B `ZeroCopyContractFrame` |
| :--- | :--- | :--- | :--- |
| **`EpistemicCalorimeter`** | $\mathcal{T}_1 + \mathcal{T}_3$ | Meria Jouly (RAPL/MMIO) a prenesené MB na UMA zbernici pre každú hypotézu. Parazitické špekulácie usmrcuje. | `timestamp_ns` (offset 32), `flags` (offset 16). |
| **`BifurcatedConflictRing`** | $\mathcal{T}_2 + \mathcal{T}_4$ | Pri $E_A \bot E_B$ vytvorí paralelné 128B SPSC pruhy (Vetva $\alpha$ a Vetva $\beta$) so spoločným `predecessor_hash`. Zákaz spriemerovania. | `predecessor_hash[32]` (offset 48..79) ako fork digest, `generation_id` (offset 24) ako branch ID. |
| **`VBlankExperimentInjector`**| $\mathcal{T}_3 + \mathcal{T}_1$ | Vykonáva sub-milisekundové CDT experimenty výlučne v $2.2\text{ ms}$ VBlank okne. Pri prekročení zastaví pred scanoutom. | `metric1` / `metric2` (offset 88..95), `approved_authority_mask` (offset 84). |
| **`CausalInertiaDamper`** | $\mathcal{T}_4 + \mathcal{T}_5$ | Počíta kumulatívnu Merkle-hmotnosť. Návrhy s nízkou hmotnosťou nemôžu mutovať systémové profily ani invariant `VITAL_MAX_HP == 6`. | `sequence_id` (offset 20), `predecessor_hash_raw` (offset 48), `signature_or_digest_raw` (offset 96). |

---

## IV. Schéma riadiaceho toku PLEA Engine

```text
                     [ TELEMETRIA & HYPOTÉZY (PROPOSER) ]
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │       EpistemicCalorimeter        │ ◄── (T1: Meria J/MB na hypotézu;
                    │   (Energetický & UMA rozpočet)    │      zabíja parazitické úvahy)
                    └─────────────────┬─────────────────┘
                                      │
                   Ak nastane konflikt (Evidence_A ⊥ Evidence_B)
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │     BifurcatedConflictRing        │ ◄── (T2: Zákaz spriemerovania;
                    │  [Vetva Alpha]  ⊥  [Vetva Beta]   │      rozštiepi 128B reťazec)
                    └────────┬─────────────────┬────────┘
                             │                 │
                             ▼                 ▼
                    ┌───────────────────────────────────┐
                    │     VBlankExperimentInjector      │ ◄── (T3: Vykoná CDT test presne
                    │  (Beh v 2.2 ms tieni medzi FPS)   │      počas VSync blanking okna)
                    └─────────────────┬─────────────────┘
                                      │
                         Víťazná vetva + Fyzický Receipt
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │       CausalInertiaDamper         │ ◄── (T4 + T5: Povýši len ak
                    │ (Merkle-hmotnosť & HP == 6 Gate)  │      prekoná kauzálnu zotrvačnosť)
                    └───────────────────────────────────┘
```
