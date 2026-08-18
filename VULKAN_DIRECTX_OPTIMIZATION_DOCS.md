# Vulkan & DirectX Optimization: GPU ako Pamäťové Médium
*Architektúra pre Compute Shaders a Dátovú Syntézu*

**Dátum:** 14. Január 2026
**Modul:** GPU Compute Bridge

---

## 1. Koncept: GPU ako "Super-RAM"
Bežný systém používa RAM. Keď dôjde, použije SSD (Swap), čo je pomalé.
Náš systém používa **VRAM (Video RAM)** ako medzistupeň.
*   **DirectX 12 / Vulkan** umožňujú pristupovať k VRAM ako k "Byte AddressBuffer".
*   **Rýchlosť:** VRAM (GDDR6) má priepustnosť 500GB/s+ (oproti DDR4 RAM s 50GB/s).

## 2. Compute Shaders a "Grid Computing"
Dáta neukladáme lineárne. Ukladáme ich do 3D Textúr (Grid).
*   Každý "Pixel" v textúre nie je farba, ale **Dátový Blok**.
*   **Compute Shader:** Malý program bežiaci na tisíckach jadier GPU naraz, ktorý tieto dáta triedi, komprimuje a analyzuje.

## 3. Syntéza pre OpenVINO
Aby OpenVINO (AI) rozumelo dátam z GPU, musíme ich "Syntetizovať".
1.  **Raw Data (GPU):** Milióny bodov v mriežke.
2.  **Reduction Shader:** GPU vypočíta priemery, odchýlky a "tepelné mapy" (Heatmaps).
3.  **Tensor (CPU):** Výsledný malý vektor (napr. 1KB dát) posielame do OpenVINO.

## 4. Generovanie Presetov
Na základe syntézy systém generuje **Runtime Presety**.
*   Ak GPU zistí, že dáta sú chaotické -> Vygeneruje preset `CHAOS_DAMPENING.json` (Vysoká kompresia).
*   Ak sú dáta lineárne -> Vygeneruje preset `STREAM_ACCELERATION.json` (Pre-fetching).

---

*Tento systém obchádza štandardný Windows Memory Manager a používa vlastný alokátor.*
