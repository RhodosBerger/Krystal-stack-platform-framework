# Epistemický kompilátor: Keď jazyk dostane pamäťovú bariéru
## Filozofický a architektonický manifest Krystal Stacku (ENLG v2)

> **Ústavný invariant:** $\text{VITAL\_MAX\_HP} == 6$  
> **Jadrová téza:** *Stupne poznania nie sú prídavné mená, sú to dátové typy.*  
> **Tri-Plane Parity:** $\text{Zero-Copy Memory Model} \iff \text{Governed Execution Model} \iff \text{Epistemic Language Model}$

---

Po desaťročia sme v informatike budovali neprekonateľné hradby tam, kde išlo o bity, adresy a registre, no nechávali sme dokorán otvorené dvere tam, kde išlo o význam a pravdu.

Ak sa procesor na architektúre ARM64 alebo RISC-V pokúsi prečítať 128-bajtový rámec zo zdieľanej pamäte skôr, než producent dokončí zápis druhej cache linky a vykoná inštrukciu `STLR` (Store-Release), nazveme to *torn read* – roztrhnutým čítaním, kritickou chybou pamäťového modelu. Ak sa proces bez príslušného kryptografického tokenu pokúsi zmeniť afinitu jadier alebo prepísať MSR register spotreby energie, operačný systém ho okamžite zastaví pre porušenie autorizácie.

Ale čo sa stane, keď veľký jazykový model (LLM) vidí v telemetrii simulovaný odhad, vynechá šum zbernice a s dokonalou gramatickou istotou vyhlási: *„Systém je plne verifikovaný a stabilný“*? V tradičných AI architektúrach sa nestane vôbec nič. Prúd tokenov plynulo pokračuje ďalej. To, čo by sme na úrovni kremíka označili za fatálne porušenie koherencie, sme na úrovni myslenia a jazyka doteraz tolerovali ako neškodnú „halucináciu“ alebo štylistickú nedokonalosť.

Práve na tomto zlome vzniká myšlienka **epistemického kompilátora** (ENLG).

---

### 1. Ilúzia plynulosti a gravitačný ťah k sebapotvrdeniu

Prirodzený jazyk nebol evolučne navrhnutý na uchovávanie prísnej hranice medzi domnienkou a dôkazom. Bol navrhnutý na presviedčanie a rozprávanie príbehov. Veľké jazykové modely, trénované na miliardách ľudských textov, zdedili túto vlastnosť v jej najčistejšej podobe: trpia neustálym gravitačným ťahom k *predčasnej promócii* (premature promotion).

V latentnom priestore modelu ležia slová *„navrhujem“*, *„simulácia naznačuje“*, *„meranie potvrdilo“* a *„je dokázané“* nebezpečne blízko seba. Stačí mierny posun v kontextovom okne a pravdepodobnostný dekodér bez váhania prekročí priepasť, ktorá vo fyzickom svete oddeľuje zbožné prianie od empirického faktu:

* Z *„kód existuje“* (`[CODE-PRESENT]`) sa v priebehu jednej vety stáva *„mechanizmus funguje“* (`[PHYSICAL-MEASURED]`).
* Zo *„simulácia prebehla“* sa stáva *„výsledok je zmeraný“*.
* Z *„testy v CI sú zelené“* sa stáva *„systém je bezchybný“*.

Klasický kompilátor (ako `clang` alebo `rustc`) nám nikdy nedovolí priradiť neinicializovaný ukazovateľ do premennej, ktorá očakáva platnú referenciu. Vyhodí typovú chybu ešte predtým, než program vôbec vznikne. Prečo by sme teda mali dovoliť kognitívnemu jadru priradiť typ `VERIFIED` tvrdeniu, ktorého vstupom je len typ `SIMULATED`?

Epistemický kompilátor mení perspektívu: **stupne poznania nie sú prídavné mená, sú to dátové typy.** A prechod z nižšieho stupňa poznania na vyšší nie je otázkou rétoriky, ale typovej konverzie, ktorá vyžaduje priložený dôkaz (Receipt).

---

### 2. Trojitý izomorfizmus: Jediný zákon pre tri svety

Najhlbšia krása indícií, z ktorých vyrastá architektúra Krystal Stacku, spočíva v objave **Tri-Plane Parity** – trojitého izomorfizmu. Ukazuje sa, že spoľahlivosť v nízkoúrovňovom hardvéri, v riadení operačného systému a v umelej inteligencii nestojí na troch rôznych princípoch. Stojí na jednom jedinom univerzálnom zákone bariéry pred zverejnením:

1. **V rovine pamäte (Zero-Copy Memory Model):** Konzument nesmie vidieť rámec skôr, než sú všetky jeho bajty v oboch cache linkách dopísané a potvrdené pamäťovou bariérou (`Publish` $\implies$ `FullyInitialized`).
2. **V rovine konania (Governed Execution Model):** Akčný člen nesmie zmeniť fyzický stav systému skôr, než prebehne autorizácia a kauzálne meranie dopadu (`Execute` $\implies$ `Authorized`).
3. **V rovine poznania (Epistemic Language Model):** Jazykový model nesmie povýšiť hypotézu na poznanie skôr, než prejde bránou dôkazov (`AssertAsKnowledge` $\implies$ `EvidentiallyAuthorized`).

V tomto svetle je nepodložené tvrdenie AI presným ekvivalentom *torn readu* na zbernici. Je to čítanie neúplnej, rozštiepenej reality, v ktorej prvá polovica (myšlienka alebo kód) už v pamäti leží, ale druhá polovica (fyzické overenie na kremíku) ešte nikdy nedorazila. `ArtifactGate` v epistemickom kompilátore nerobí nič iné, než to, čo robí inštrukcia `LDAR` (Load-Acquire) na procesore: odmieta prepustiť informáciu ďalej, kým nie je kauzálna reťaz kompletne uzavretá.

---

### 3. Obrátenie šípky myslenia: Falsifier-First a krása nevedomosti

Ľudské aj strojové uvažovanie prirodzene postupuje od tvrdenia k hľadaniu argumentov, ktoré ho podporujú. Epistemický kompilátor túto šípku násilne láme a otáča.

Vzor **Falsifier-First** (najprv vyvracač) núti systém urobiť niečo hlboko neprirodzené pre štatistický prediktor ďalšieho slova: ešte predtým, než smie vysloviť argument v prospech svojej hypotézy, musí presne zadefinovať najmenšie možné pozorovanie, ktoré ju zničí. Spolu s vytvorením „kontrafaktuálnych dvojčiat“ ($H_0$ – žiadny efekt, $H_1$ – šum, $H_2$ – artefakt merania alebo tepelný throttling) tým kompilátor vytvára imunitný systém priamo vo vnútri aktivovaného kontextu. Model už nehľadá spôsob, ako obhájiť svoj záver; stáva sa sudcom v spore medzi vlastným návrhom a troma alternatívnymi vysvetleniami.

Rovnako radikálny je aj vzor **Residual-Preservation** a **Semantic NOOP**. Pre bežný jazykový model je odpoveď `UNKNOWN` (neviem / nezmerané) alebo `CONFLICT` (dôkazy si protirečia) zlyhaním, ktorému sa snaží vyhnúť zahladením rozporov do vágneho priemeru. Pre epistemický kompilátor je to presne naopak:

* Priznanie `UNKNOWN` je kryptograficky atestovaný stav poznania.
* Keď dva prúdy telemetrie stoja proti sebe ($Evidence_A \bot Evidence_B$), pravda neleží uprostred. Pravda leží v priznaní rozporu (`CONFLICT_NODE`) a vo vygenerovaní najlacnejšieho rozlišovacieho experimentu, ktorý spor rozsekne.
* Ak chýba dôkaz, jedinou korektnou operáciou je sémantické `NOOP` ($\Delta S = 0$). Mlčanie o tom, čo nebolo zmerané, nie je prázdnotou – je to zachovanie ústavného invariantu integrity.

---

### 4. Kompilácia slov na činy

Možno najsilnejším momentom celej koncepcie je zistenie, že v skutočne uzavretom systéme **výstupom premýšľania nemá byť odpoveď, ale experiment**.

Tradičný prompt končí bodkou za zhrnutím. Epistemický kompilátor končí sekciou `NEXT_EXPERIMENT`: presným predpisom merania, kritéria úspechu, kritéria zlyhania, deterministického rollbacku a podmienky zastavenia. Slová sa tak nestrácajú vo vzduchu; kompilujú sa priamo do stavového automatu (`PLANNED` $\to$ `BASELINE_MEASURED` $\to$ `ACTUATED` $\to$ `POST_MEASURED` $\to$ `AUDITED`), ktorý siahne na reálny hardvér, odmeria odozvu a vráti sa späť s novým, vyšším typom dôkazu.

Epistemický kompilátor teda nie je len sadou pravidiel pre krajšie formátovanie výstupov. Je to pokus o vnesenie fyzikálnej poctivosti do sveta symbolov. Učí nás, že medzi *„navrhnúť“* (`PROPOSER`) a *„potvrdiť“* (`VERIFIER`) musí stáť neprekročiteľná hranica právomocí – a že skutočná inteligencia sa nepozná podľa toho, ako presvedčivo dokáže odpovedať na všetko, ale podľa toho, ako nekompromisne dokáže strážiť hranicu medzi tým, čo si len myslí, a tým, čo naozaj vie.
