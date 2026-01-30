# Virtuálna 3D Pamäť (Holocube): Analógia Galaxie
*Topologický prístup k správe dát v Gamesa/KrystalStack*

**Dátum:** 14. Január 2026
**Modul:** Holocube Grid

---

## 1. Definícia Priestoru (The Cube)
Opúšťame lineárne adresovanie (`0x0000` -> `0xFFFF`).
Zavádzame vektorové adresovanie: `V(x, y, z)`.

### Osi Súradníc:
*   **Os X (Kontext):** Typ dát. (0-10: Audio, 11-20: Textúra, 21-30: Logika).
*   **Os Y (Priorita):** "Teplota" dát. (0 = Cold Storage/SSD, 100 = L1 Cache/Hot).
*   **Os Z (Čas):** Verzia dát. (Z=0: Prítomnosť, Z=+1: Predikcia, Z=-1: História).

## 2. Mechanika: Pamäťová Gravitácia
V tomto systéme platí fyzikálny zákon: **"Dáta, ktoré sa používajú spolu, sa priťahujú."**

### Analógia
Ak procesor často žiada `Objekt_A` (Textúra) a `Objekt_B` (Shader) v tom istom cykle, systém zvýši ich vzájomnú "Gravitačnú Konštantu".
*   V ďalšom cykle "Defragmentácie" sa `Objekt_B` fyzicky presunie na súradnicu vedľa `Objekt_A`.
*   **Výsledok:** Načítanie bloku pamäte (Memory Page) natiahne oba objekty naraz. Zero Latency.

## 3. Raycast Reading (Lúčové Čítanie)
Namiesto čítania po bajtoch, CPU vystrelí "Lúč" cez Grid.
*   **Príkaz:** `GET_RAY(Start: [10,10,0], Direction: [0,0,1])`
*   **Význam:** "Daj mi vývoj tohto objektu v čase."
*   Systém vráti pole hodnôt z `Z=0` až `Z=5` (Prítomnosť + 5 krokov predikcie) v jednej transakcii.

---

## 4. Využitie pre Rendering (DirectX/Vulkan)
Pre 3D scény je to natívne prostredie.
*   Súradnice v pamäti môžu zodpovedať súradniciam vo svete hry.
*   Ak hráč stojí na pozícii `(100, 100)`, pamäťový systém automaticky načíta Grid Sector okolo týchto súradníc do L2 Cache.
