# Krystal Stack — Prémiový Dizajn Manuál (Estetika AMD Adrenalin × Funkčná Hĺbka Intel Arc & NVIDIA Panel)

> **Ústavný invariant:** `VITAL_MAX_HP == 6` (`\neg Valid \implies \Delta S = 0`) · **Zbernica UMA:** `< 0.78`

## 1. Filozofia vizuálnej adopcie
Rozhranie spája tri svety do jedného suverénneho štandardu:
1. **Estetická identita inšpirovaná AMD Software: Adrenalin Edition:**
   - Hlboké obsidiánovo-uhlíkové povrchy (`#0a0c11`, `#121620`, `#171c28`) s jemným kovovým prechodom,
   - Ostrá karmínovo-rubínová signálna linka na hornej hrane aktívnych kariet (`#ed1c24` → `#ff3355`) v kontraste s azúrovým telemetrickým lúčom (`#00e5ff`) pre zdravý stav zbernice UMA,
   - Skosené taktické rohy (`clip-path` bevel `8px`), kruhové oblúkové ukazovatele frekvencií a hustá typografia s pevnou šírkou číslic (`tabular-nums`).
2. **Funkčná hĺbka panelov NVIDIA Control Panel / NVIDIA App:**
   - Profily šité na mieru pre konkrétne herné enginy (`ENG_PRESET_01..08`), prepínač kvality **Krystal-DLSS (`DLAA / Quality / Balanced / Performance / UltraPerf`)**, nízkolatenčná synchronizácia **Explicit Sync Reflex** a predhriatie shader cache (`ANV GPL` / zero-heap `CSO`).
3. **Hardvérová telemetria na úrovni Intel Graphics Command Center / oneAPI:**
   - Živé sledovanie `96` EU (`768` FP32 stream procesorov), bezpečných frekvenčných obálok čipsetu Tiger Lake (`400 – 4200 MHz` CPU, `400 – 1300 MHz` Iris Xe), chladiacich kriviek Arrhenia (`THRM_01..05`), zdieľanej `14,0 GB` UMA VRAM a unifikovaného pamäťového swapu.

## 2. Kaskádové CSS tokeny (`krystal-adrenalin-cascade`)
```css
:root {
  --kr-bg-obsidian: #090b10;
  --kr-card-carbon: #121621;
  --kr-card-elevated: #181e2c;
  --kr-border-steel: rgba(255, 255, 255, 0.09);
  --kr-adrenalin-red: #ed1c24;
  --kr-adrenalin-glow: rgba(237, 28, 36, 0.28);
  --kr-xe-cyan: #00e5ff;
  --kr-ok-emerald: #10b981;
  --kr-warn-amber: #f59e0b;
  --kr-text-primary: #f1f5f9;
  --kr-text-muted: #94a3b8;
}
```

## 3. Pravidlá animácií GSAP (`Morphing` & `Type-Vykresľovanie`)
- **Prechod informačných celkov (`GSAP Flip` + SVG `attr:{d}` morphing):** Prednačítané kombinácie blokov (`AST Logika`, `DLSS & Vulkan`, `Čipset & Unifikovaný Swap`, `MCP & Gemma 4 E2B`) morfujú v priebehu `380 ms` (`power3.out`) bez trhania layoutu.
- **Kinetické vykresľovanie viet (`GSAP TextPlugin`):** Každá nová veta vyprodukovaná kompilátorom jedinečných vysvetlení a modelom `Gemma 4 E2B` sa vykresľuje plynulým prúdom znakov s okamžitým ukotvením citovaných faktov.
