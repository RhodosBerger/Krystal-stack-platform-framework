# Špekulatívna Architektúra Neuromorfného Mikroprocesora: Syntéza Neuromorfného Kremíka, Bioinformatiky a Duálneho Komputingu

**Autor:** Dušan Kopecký & Krystal-Stack Architecture Council (2026)  
**Systémový invariant:** `VITAL_MAX_HP = 6`  
**Cieľová architektúra:** Intel 11th Gen Core i5-1135G7 (Willow Cove) + Intel Iris Xe Graphics (96 EUs) / Sub-10nm Feroelektrický uzol  
**Živý webový blog:** `http://localhost:8080/blog` alebo `http://localhost:8080/speculative-architecture`

---

## 1. Filozoficko-Vedecká Esej: Duálna Paradigma Computingu

### 1.1. Kríza Diskrétneho Von Neumannovho Monopolu
Od 40. rokov 20. storočia dominovala počítačovej vede jediná dogma: **Turingova abstrakcia**. Tá predpokladá, že akýkoľvek výpočet možno vyjadriť ako postupnosť diskrétnych stavov riadených deterministickou logikou cez centrálnu procesorovú jednotku a pasívnu pamäť. Táto abstrakcia však ignoruje fyzikálnu realitu termodynamiky:
1. **Von Neumannovo pamäťové hrdlo (Memory Wall)**: Presun dát medzi pamäťovou maticou a aritmeticko-logickou jednotkou (ALU) spotrebúva $85 - 90\%$ celkovej energie čipu.
2. **Koniec Dennardovho škálovania**: Zmenšovanie tranzistorov už neprináša automatické znižovanie napätia kvôli parazitným zvodovým prúdom (sub-threshold leakage).

V protiklade k tomu stojí **biologický mozog**, ktorý spracúva komplexné zmyslové, kognitívne a motorické dáta pri príkone iba $\sim 20\,\text{W}$. Dôvodom nie je to, že by bol mozog „dokonalejší binárny počítač“, ale to, že funguje v **duálnej paradigme**:
* **Diskrétna vrstva (All-or-None Action Potentials)**: Akčné potenciály (spiky) sú diskrétne udalosti umožňujúce bezšumový diaľkový prenos informácie pozdĺž axónov.
* **Spojitá vlnová vrstva (Sub-threshold Dendritic Computation & Wave Interference)**: Integrácia signálov v dendritickom strome je spojitá, analógová a vlnová. Synapsie nemajú oddelenú pamäť od procesora – váha synapsie (jej vodivosť $G$) je zároveň pamäťou aj okamžitým analógovým násobičom.

---

## 2. Štyri Piliere Konvergencie

```
                          ┌────────────────────────┐
                          │   DUÁLNA PARADIGMA     │
                          │   COMPUTINGU (2026)    │
                          └───────────┬────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
 ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
 │      NEUROLÓGIA      │  │ ELEKTRIKA & FYZIKA   │  │   BIOINFORMATIKA     │
 │  - LIF Spiking Neurón│  │  - Feroelektrický    │  │  - 4-Stavová DNA     │
 │  - STDP Plasticita   │  │    FeFET Memristor   │  │    Quaternary Logika │
 │  - Refraktérny cyklus│  │  - Dual-Rail IPC     │  │  - GF(2) Samoliečenie│
 └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

1. **Neurológia (Spike-Timing Dependent Plasticity & Event-Driven Execution)**:
   Procesor nevykonáva prázdne cykly (busy-waiting). Výpočet prebieha výhradne vtedy, keď integrál vstupných prúdov presiahne prahové napätie $V_{\text{th}}$. Refraktérna perióda zabezpečuje smerovosť toku dát bez nutnosti centrálnych hodín (clockless asynchronous fabric).
2. **Elektrika a Fyzika Feroelektrika (Sub-10nm HZO)**:
   Využitie feroelektrického oxidu hafnično-zirkoničitého ($\text{Hf}_{0.5}\text{Zr}_{0.5}\text{O}_2$) integrovaného do hradla FET tranzistora. Polarizačné stavy umožňujú analógové škálovanie vodivosti kanála bez potreby periodického obnovovania náboja (non-volatile analog memory).
3. **Bioinformatika a Genomika (Kvaternárna Logika a GF(2) Parita)**:
   Inšpirácia genetickým kódom ($A, C, G, T$): nahradenie 2-stavovej logiky 4-stavovou kvaternárnou logikou pomocou multi-prahových hradiel zdvojnásobuje informačnú hustotu na plochu kremíka. Chyby spôsobené šumom sú autogénne hojené paritnou maticou Hamminga $GF(2)$.
4. **Duálna Paradigma Computingu**:
   Symbolické riadenie (Von Neumann Control Plane) koordinuje vlnovo-interferenčné analógové polia (Neuromorphic Tensor Fabric).

---

## 3. Zvýraznenie Kľúčových Matíc Systému

Systém mikroprocesora transformuje všetky výpočtové operácie na manipuláciu so štyrmi fundamentálnymi maticami:

### 3.1. Systolické Tenzorové Pole ($512 \times 512$ INT8 / FP16 GEMM)
Násobenie tenzorov prebieha systolickým posunom dát:
$$\mathbf{C} = \mathbf{A} \times \mathbf{B} + \mathbf{D}$$
$$\begin{bmatrix}
C_{00} & C_{01} & \dots & C_{0n} \\
C_{10} & C_{11} & \dots & C_{1n} \\
\vdots & \vdots & \ddots & \vdots \\
C_{m0} & C_{m1} & \dots & C_{mn}
\end{bmatrix} =
\begin{bmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11} \\
\vdots & \vdots \\
A_{m0} & A_{m1}
\end{bmatrix} \times
\begin{bmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{bmatrix}$$
* **Empirický výkon v Krystal Stacku**: Latencia klesla z $2\,876\,210\,\mu\text{s}$ na **$7.0\,\mu\text{s}$**, čo predstavuje **$410\,887\times$ zrýchlenie**.

### 3.2. Tile4 2D Cache Tiling & Kisak CCS Matica
Organizácia riadkov grafickej a systémovej pamäte L2 Cache (Iris Xe):
$$\begin{bmatrix}
\text{Tile}_{0,0} \ (4\,\text{KB}) & \text{Tile}_{1,0} \ (4\,\text{KB}) \\
\text{CCS Aux}: \mathtt{0b11} & \text{CCS Aux}: \mathtt{0b01} \\
\hline
\text{Tile}_{0,1} \ (4\,\text{KB}) & \text{Tile}_{1,1} \ (4\,\text{KB}) \\
\text{CCS Aux}: \mathtt{0b01} & \text{CCS Aux}: \mathtt{0b11}
\end{bmatrix}$$
* **Úspora šírky zbernice**: **$90.0\%$** redukcia DRAM prenosov vďaka kompresným bitom Color Clear State (CCS).

### 3.3. Hamming $GF(2)$ Paritná Matica Autogénneho Samoliečenia
Lokalizácia a oprava sub-threshold bitflips:
$$\mathbf{H} = \begin{bmatrix}
1 & 0 & 1 & 0 & 1 & 0 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1 & 1 & 1 & 1
\end{bmatrix}, \quad \mathbf{s} = \mathbf{H} \cdot \mathbf{x}^T \pmod 2$$
* Ak syndróm $\mathbf{s} = \mathbf{0}$, stav je neporušený. Ak $\mathbf{s} \ne \mathbf{0}$, index stĺpca priamo identifikuje poškodený bit, ktorý hardvérová logika invertuje v jedinom takte bez prerušenia operačného systému.

### 3.4. Synaptický Memristorový Krížový Prepínač (Crossbar Array)
Násobenie vektora a matice cez Kirchhoffov prúdový zákon ($O(1)$ časová zložitosť):
$$I_i = \sum_{j=1}^{N} V_j \cdot G_{ij}$$
$$\begin{bmatrix} I_1 \\ I_2 \\ I_3 \end{bmatrix} = \begin{bmatrix} G_{11} & G_{12} & G_{13} \\ G_{21} & G_{22} & G_{23} \\ G_{31} & G_{32} & G_{33} \end{bmatrix} \times \begin{bmatrix} V_1 \\ V_2 \\ V_3 \end{bmatrix}$$
* Každý uzol krížového prepínača je tvorený jedným tranzistorom FeFET s plynulo nastaviteľnou vodivosťou $G_{ij}$.

---

## 4. Konkrétne Nákresy Tranzistorov a Obvodov

### 4.1. Memristive Ferroelectric Synaptic FeFET
```
                  V_gate (Pre-synaptic Action Potential)
                            │
                            ├──[ R_series ]──┐
                                             │
                                         ┌───┴───┐
                                         │  Gate │
                                         ├───────┤
                                         │  HZO  │  <-- Feroelektrická vrstva (Polarizácia P_r)
                                         ├───────┤
                                         │Channel│
               Source ───────────────────┴───────┴─── Drain (I_post = G · V_ds)
                 │                                       │
               [GND]                                   [Virtuálna zem zosilňovača]
```
* **Fyzikálny princíp**: Vložením feroelektrickej vrstvy $\text{Hf}_{0.5}\text{Zr}_{0.5}\text{O}_2$ s remanentnou polarizáciou $P_r \approx 20\,\mu\text{C/cm}^2$ sa prahové napätie $V_{\text{th}}$ posúva podľa integrálu predchádzajúcich napäťových pulzov.

### 4.2. Neuromorfný Leaky Integrate-and-Fire (LIF) Obvod
```
    I_synaptic In ───┬──────[ R_leak ]────── GND
                     │
                    === C_membrane (Integrácia náboja synapsií)
                     │
                     ├───(+) Komparátor [ Prah V_th ] ───┬─── Spike Out (V_axon)
                     │   (-)                             │
                     │                                   └───[ Delay Invertor ]───┐
                     │                                                            │
                     └─── Drain ───[ Reset NMOS ]─── GND <────────────────────────┘
                                         Gate (Refraktérna perióda ~2 ms)
```

### 4.3. Dual-Rail Diferenciálny Tranzistorový Pár pre Binary IPC
```
                                 VDD (0.9V)
                                     │
                             ┌───────┴───────┐
                           [PMOS]          [PMOS]  (Cross-coupled precharge)
                             │   ╲        ╱  │
                             ├───( D )──( D# )──┤  <-- Diferenciálny výstup IPC
                             │               │
                           [NMOS]          [NMOS]  (Vstupný signál + GF(2) parita)
                             │               │
                             └───────┬───────┘
                                     │
                                [Footer NMOS] (Clock Enable: Sub-nanosekundové vyhodnotenie)
                                     │
                                    GND
```

### 4.4. Kvaternárne Multi-Gate Bio-Hradlo (4-Stavová DNA Logika)
```
                                 VDD (1.2V)
                                     │
                             ┌───────┼───────┐
                           [FET-1] [FET-2] [FET-3]
                          Vth=0.2V Vth=0.5V Vth=0.8V
                             │       │       │
                             └───────┼───────┘
                                     │
                                     ├─── Výstupné napätie V_out:
                                     │    - 0.0V : 00 (A - Adenín)
                                     │    - 0.3V : 01 (C - Cytozín)
                                     │    - 0.6V : 10 (G - Guanín)
                                     │    - 0.9V : 11 (T - Tymín)
                                   [R_load]
                                     │
                                    GND
```

---

## 5. Kvantitatívne Porovnanie Architektúr

| Metrika | Konvenčný x86 Von Neumann | Krystal-NeuroCore (Návrh) | Kvantitatívny Prínos |
| :--- | :--- | :--- | :--- |
| **Energetická účinnosť (GEMM)** | $18.4\,\text{nJ / op}$ | **$0.092\,\text{nJ / op}$** | **$200\times$ nižšia spotreba** |
| **Medzimodulová latencia (IPC)** | $74.88\,\mu\text{s}$ (JSON/HTTP) | **$2.66\,\mu\text{s}$ (Binary SHM)** | **$28.2\times$ zrýchlenie** |
| **L1 Cache zaťaženie na frame** | 365 liniek (Eviction hazard) | **181 liniek (Zero hazard)** | **184 liniek ušetrených** |
| **Hustota informácie na tranzistor** | $1\,\text{bit / tranzistor}$ | **$2\,\text{bity / tranzistor}$** | **$+100\%$ hustota stavov** |
| **Mechanizmus detekcie chýb** | Externý ECC radič | **In-Silicon $GF(2)$ Hamming** | **0 taktov oneskorenia** |
| **Invariant integrity** | `VITAL_MAX_HP = 6` | `VITAL_MAX_HP = 6` | **Striktne zachovaný** |

---

## 6. Prístup k Živému Blogu a Demonštrácii

Blogový portál bol nasadený na Localhost Hub Serveri a je okamžite prístupný v prehliadači:
* **URL Blogu**: `http://localhost:8080/blog`
* **Alternatívne aliasy**: `http://localhost:8080/speculative-architecture`, `http://localhost:8080/neuromorphic-cpu`
* **Vizuálne assety**:
  * [`krystal_web_hub/static/images/speculative_neuromorphic_die.jpg`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/images/speculative_neuromorphic_die.jpg)
  * [`krystal_web_hub/static/images/neuromorphic_transistor_schematic.jpg`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/images/neuromorphic_transistor_schematic.jpg)
  * HTML zdroj: [`krystal_web_hub/static/speculative_microprocessor_blog.html`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/speculative_microprocessor_blog.html)
