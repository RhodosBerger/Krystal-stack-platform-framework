# Tabuľkový generátor riadený prioritami bota a kalibrácia modelov (`CSV` + `JSONL`)

- **Ústavný invariant:** `VITAL_MAX_HP == 6`
- **Vygenerovaný minimálne postačujúci objem:** **`180` riadkov (`33.87 KB` CSV)** · potlačených duplicitných riadkov: `26`
- **Úspora objemu voči surovému logovaniu 5 000 tikov:** **`40.37×` menšia stopa** pri 100 % pokrytí všetkých 5 cieľových režimov (`CFG_01..CFG_05`).
- **Rozdelenie bez priesaku (chronologické):** `train = 108` · `val = 36` · `test = 36`

## Porovnanie modelov a kalibrácia na testovacej sade (`ml-best-practices` + 7-kroková kalibrácia)

| Model | Presnosť (`Accuracy`) | `Macro-F1` | `Brier Score` (nižšie = lepšie) | `ECE` (nižšie = lepšie) | Poznámka |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `M0_NAIVE_PRIOR` | `0.2222` | `0.0727` | `0.8002` | `0.0093` | Naivný väčšinový baseline |
| `M1_REGULARIZED_LINEAR` | `0.9167` | `0.916` | `0.1404` | `0.0715` | L2 multinomiálna logistická regresia |
| `M2_UNCALIBRATED_MLP` | `0.9167` | `0.916` | `0.1361` | `0.0557` | Dvojvrstvová sieť pred kalibráciou |
| **`M2_TEMP_CALIBRATED_MLP`** | **`0.9167`** | **`0.916`** | **`0.1359`** (`±0.0664`) | **`0.0517`** (`±0.0251`) | Teplotné škálovanie `T* = 1.05`, 95 % CI `[0.8056, 1.0]` |

## Záver analýzy dát
Stratifikovaný odber okolo kritických prahov (`UMA = 0.78`, `K = 449`, `Δsenzor = 1.0 m`, `payload = 16 KB`, `dávka = 64`) spojený s L2 filtrom novosti zachytáva plnú rozhodovaciu hranicu v ~24 KB stope a umožňuje spoľahlivo trénovať aj teplotne kalibrovať následné modely bez zbytočného zahltenia disku.
