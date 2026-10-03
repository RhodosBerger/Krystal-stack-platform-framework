# Krystal-Stack Research: 12 Pozemských Úkazov a Sektorov Čínskeho Zverokruhu

**Dátum:** 3. Október 2026  
**Autor:** Krystal-Stack Geomancy, Celestial Mechanics & Spatial Topology Group  
**Kľúčový Invariant:** `VITAL_MAX_HP = 6` (Pravidlo maximálnej celistvosti systému)  
**Zlatý Pomer:** $\phi = 1.61803398875$, $\phi^{-1} \approx 0.61803398875$

---

## 1. Úvod a Kozmologická Architektúra

Zatiaľ čo západný zverokruh (Baran až Ryby) riadi v Krystal-Stack nebeskú sféru, optiku zväčšenia a spektrálne imunity, **čínska astrológia a 12 pozemských vetiev (地支 Dìzhī / 生肖 Shēngxiào)** bola integrovaná ako riadiaci princíp pre **pozemské úkazy (terrestrial phenomena)** a **fyzické rozdelenie sveta do 12 samostatných sektorov**.

Každé zviera a pozemská vetva reprezentuje konkrétny geomorfologický, atmosférický a hydrologický jav, ktorý priamo ovplyvňuje:
- Alchýmiu a rýchlosť ťažby surovín v sektore.
- Taktovanie a chladenie hardvérových jadier (VRAM cache, L1 bandwidth).
- Šébrové a časticové efekty na plátne herného sveta.
- Odolnosť sektorových pevností (striktne viazaná na `VITAL_MAX_HP = 6`).

---

## 2. Prehľad 12 Sektorov a Pozemských Úkazov

| Sektor | Zverokruh | Vetva (Čas) | Element / Polarita | Pozemský Úkaz | Hlavný Vplyv na Systém |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sector_01_rat` | **Potkan (鼠)** | 1. Zǐ (23:00 - 01:00) | Voda / Yang | Nočná hydrologická spodná voda | Chladenie VRAM, filtrácia aéteru |
| `sector_02_ox` | **Byvol (牛)** | 2. Chǒu (01:00 - 03:00) | Zem / Yin | Kryo-tektonické vrstvenie žuly | Odolnosť pevností, stabilita zbernice |
| `sector_03_tiger` | **Tiger (虎)** | 3. Yín (03:00 - 05:00) | Drevo / Yang | Ranné blesky a lesná búrka | Bojový nárast rýchlosti, L1 cache burst |
| `sector_04_rabbit` | **Zajac (兔)** | 4. Mǎo (05:00 - 07:00) | Drevo / Yin | Ranná hmla a rast fyto-spór | Regenerácia zdravia, jantárová miazga |
| `sector_05_dragon` | **Drak (龙)** | 5. Chén (07:00 - 09:00) | Zem / Yang | Geomagnetická búrka & gejzír | Dračia mana, ionizovaný dym, požiar |
| `sector_06_snake` | **Had (蛇)** | 6. Sì (09:00 - 11:00) | Oheň / Yin | Zemný plyn a kyselinová štrbina | Toxická korózia, kaustické bahno |
| `sector_07_horse` | **Kôň (马)** | 7. Wǔ (11:00 - 13:00) | Oheň / Yang | Poludňajší solárny žiar | Maximálny takt procesora (Core Clock Boost) |
| `sector_08_goat` | **Koza (羊)** | 8. Wèi (13:00 - 15:00) | Zem / Yin | Sprašová usadenina & alúvium | Chemická rovnováha, stabilita alchýmie |
| `sector_09_monkey` | **Opica (猴)** | 9. Shēn (15:00 - 17:00) | Kov / Yang | Horský vír & ionosféra | Aerostatický sklz, rýchly swap pamäte |
| `sector_10_rooster` | **Kohút (鸡)** | 10. Yǒu (17:00 - 19:00) | Kov / Yin | Zrkadlenie žíl & magnetit | Brnenie, feromagnetický štít zbraní |
| `sector_11_dog` | **Pes (狗)** | 11. Xū (19:00 - 21:00) | Zem / Yang | Seizmická hliadka kôry | Detekcia podzemného pohybu, protiprieskum |
| `sector_12_pig` | **Prasa (猪)** | 12. Hài (21:00 - 23:00) | Voda / Yin | Aluviálna bažina & sediment | Dlhodobý archív pamäte, usadzovanie kalov |

---

## 3. Topológia Bagua a Priestorové Usporiadanie

12 sektorov je usporiadaných do kruhového kompasu **Bagua (八卦)** s 30-stupňovým krokom okolo centrálneho kryštálového jadra:
- **Sever (0°):** Potkan (Zǐ) – nočná spodná voda.
- **Východ (90°):** Zajac (Mǎo) – ranná hmla a fyto-spóry.
- **Juh (180°):** Kôň (Wǔ) – zenitový solárny žiar.
- **Západ (270°):** Kohút (Yǒu) – magnetitové zrkadlá rudných žíl.

Každý sektor disponuje vlastnými hexadecimálnymi súradnicami `hex_coordinates`, ktoré sa integrujú s existujúcim `SectorConquestEngine` a `ACTIVE_ECONOMIC_MATCH`.

---

## 4. Implementácia v Architektúre Krystal-Stack

1. **Python Core Engine:** [chinese_zodiac_terrestrial_sectors.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/chinese_zodiac_terrestrial_sectors.py)
   - Dataclassy `TerrestrialPhenomenon`, `ZodiacSector`, `ChineseZodiacSectorEngine`.
   - Modulácia intenzity javov `trigger_phenomenon` a výpočet vládnucej vetvy cyklu `evaluate_global_terrestrial_cycle`.
2. **Janet DSL Subproject:** [chinese_zodiac_terrestrial_sectors.janet](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/chinese_zodiac_terrestrial_sectors.janet)
   - 28. validovaný súbor v subsystéme Janet.
3. **REST API na porte 8089:** [krystal_engine_core.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/krystal_engine_core.py)
   - `GET /api/zodiac-sectors/all`
   - `GET /api/zodiac-sectors/cycle?turn=N`
   - `POST /api/zodiac-sectors/trigger-phenomenon`
   - `GET /chinese-zodiac-sectors` (webové štúdio)
4. **Interaktívne Webové Štúdio:** [chinese_zodiac_sectors_studio.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/chinese_zodiac_sectors_studio.html)
   - Rotujúce kruhové plátno Bagua s 12 sektormi a čínskymi glyfmi.
   - Časticové emisie pre každý element.
   - Inšpekčný panel s detailmi terénu, fenoménu a tlačidlom na zosilnenie úkazu.
5. **Krystal WebOS Desktop Integrácia:** [krystal_webos_desktop.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/krystal_webos_desktop.html)
   - Ikona `☯️ Čínsky Zverokruh`, položka v Štart menu a plávajúce okno.
6. **Platformový Invariant:**
   - Striktné dodržanie `VITAL_MAX_HP = 6`.
