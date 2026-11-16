# Krystal Stack: Keď heterogénny kremík nerozhoduje naslepo, ale na základe dôkazu

Moderné výpočtové stanice a notebooky s procesormi Intel v sebe spájajú vektorové jadrá `AVX-512` (`oneDNN`), integrovanú grafiku `Intel Iris Xe` na zdieľanej pamäťovej zbernici (`UMA`) a lokálne jazykové či tenzorové modely. Väčšina dnešných frameworkov však medzi týmito jednotkami prepína buď natvrdo, alebo na základe neoverených heuristík — čo pri krátkych úlohách vedie k stovkám milisekúnd stratenej kompilácie a pri vyťaženej zbernici k trhaniu obrazu.

**Krystal Stack** prináša odlišnú inžiniersku disciplínu: **žiadny modul, jazykový model ani používateľské rozhranie si sám necertifikuje svoj výstup (`PROPOSER != VERIFIER`)**. Každé prepnutie zariadenia, každý maticový súčin aj každý súbojový ťah prechádza merateľnou bránou so zárukou, že neplatný zásah stav systému nikdy nepoškodí ($\Delta S = 0$).

---

## Čo Krystal Stack preukázateľne doručuje už dnes (`[NAMERANÉ A OVERENÉ NA HOSTITEĽOVI]`)

1. **Ergonómia akcelerátora pre Intel kremík (`IntelOneApiAccelerator`)**
   Vrstva postavená priamo nad OpenVINO a Intel oneAPI spája bezkópiové hostiteľské tenzory na CPU (`ov.Tensor(..., shared_memory=True)` s jadrom `brgconv_avx512_f32`), asynchrónne prúdy `IntelStream` na `Iris Xe` (`convolution_gpu_bfyx_os_iyx_osv16__f32`) a perzistentnú `CACHE_DIR`, ktorá zráža opakovanú kompiláciu iGPU zo stoviek milisekúnd na jednotky až nízke desiatky milisekúnd.
2. **Matematická kontrola maticových operácií a numerického posunu**
   Systém neskrýva rozdiely plávajúcej rádovej čiarky `IEEE-754 float32` pod koberec: presne identifikuje hranicu `K = 449`, kde `OpenBLAS` prechádza zo sekvenčného FMA na blokovú akumuláciu, stráži relatívnu odchýlku medzi CPU a GPU (`~2.5e-6 < 1e-3`) a overuje maticové súčiny v čase $O(n^2)$ pomocou **Freivaldsovho algoritmu**.
3. **Neuro-symbolický CLI Shell a citovaný jazykový most**
   Prirodzený jazyk slúži na pohodlné vyjadrenie zámeru, nie na nekontrolované spúšťanie príkazov. Každá veta z LLM musí niesť citáciu overeného faktu `[fact_id]` a každý navrhnutý príkaz v `KrystalOneApiHeterogeneousShell` prechádza prísnou bielou listinou `ShellCommandVerifier`. Ak lokálny model na porte `:1234` nebeží, systém nič nefabuluje — prepne na deterministické maticové smerovanie v OpenVINO a jasne to označí.
4. **4-panelová diagnostika pamäťových tokov v štýle `DxDiag` a extrakcia vzorov**
   Namiesto neprehľadných logov dostáva vývojár okamžitý prehľad o stave kremíka, štyroch pamäťových prúdoch, `128 B` rámcoch so `SHA-256` potvrdením a kandidátnych vzoroch (`PATTERN_CANDIDATE`), ktoré priamo vedú ďalšiu optimalizáciu.

---

## Štyri okamžité polia nasadenia

- **Deterministické hry a multi-žánrová synchronizácia:** adaptívny rollback horizont ($\text{EWMA} + 2\sigma^+$ s 12-krokovou hysteréziou) a aréna, v ktorej je každá karta priamo zviazaná so skutočným stavom hardvéru.
- **Lokálne spracovanie obrazu a prepojenie s Blenderom:** 5-stupňový most pre procedurálnu stavbu 3D scén a rozumné rozdelenie práce medzi EEVEE/Cycles a OpenVINO tenzory bez preťaženia zbernice `UMA`.
- **Bezpeční lokálni inžinierski kopiloti:** riadenie konfigurátora (`U01..U06`) cez OpenAPI 3.1 rozhranie bez rizika halucinovaných čísel alebo podvrhnutých shell príkazov.
- **Samo-auditujúce výpočtové a priemyselné pipeline:** konzistenčný bot priebežne overuje tvrdenia v kóde a ukladá výsledky do centralizovanej znalostnej bázy.

---

## Kam Krystal Stack smeruje ďalej (`[VÍZIA A BUDÚCA ROADMAPA]`)

Nasledujúce kroky sú explicitne vedené ako plánované rozšírenia, ktoré budú do jadra prijaté až po hardvérovom premeraní:
- **Priame hardvérové čítanie telemetrie zbernice a spotreby (RAPL / PMC / Level Zero Sysman):** nahradenie používateľských proxy modelov priamymi počítadlami priepustnosti IMC a joulov na každom kroku inferencie.
- **Plné bezkópiové zdieľanie povrchov medzi Vulkanom, Blenderom a OpenVINO GPU (`USM / DMA-BUF / Win32 Handle`):** odstránenie zostávajúcich medzikópií pri prechode renderovaných snímok do konvolučných filtrov.
- **Dedikované NPU vetvy pre generácie Meteor Lake / Lunar Lake / Panther Lake:** rozšírenie maticového smerovača promptov o štvrtú doménu pre trvalý nízkoenergetický beh na pozadí.
- **Automatická kalibrácia pri každej aktualizácii ovládača:** okamžité prepočítanie bodov zlomu $n^*$ a prahov redukcie $K$ pri štarte stanice, aby krivky očakávaného výkonu vždy zodpovedali skutočnému kremíku.
