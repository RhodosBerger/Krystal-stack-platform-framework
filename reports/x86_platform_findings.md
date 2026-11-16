# Výpočtová správa: Nové zistenia pre platformu x86-64 (Willow Cove / AVX-512)

- **Procesor**: `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz` (`4` fyzické / `8` logické jadrá, `little`-endian)
- **ISA príznaky (`WIN32_IsProcessorFeaturePresent`)**: `{"SSE2": true, "SSE3": true, "AVX": true, "AVX2": true, "AVX512F": true}`
- **Vzorky na meranie**: `n = 15` (`LOCAL_MEASURED`)

## Prehľad 5 empirických zistení

| ID | Skúmaný jav na x86-64 | Kľúčová nameraná hodnota | Dôsledok pre Krystal Stack |
|---|---|---|---|
| `X01` | Priepustnosť L1d -> L2 -> L3 -> UMA DRAM | `24.98` -> `55.1` -> `34.56` -> `17.81` GB/s (`1.4x` L1d/DRAM) | Držať mikro-tenzory `< 48 KB` v L1d na CPU namiesto prenosu cez UMA |
| `X02` | Škálovanie `1T -> 4T` (fyzické) vs. `8T` (SMT) pod `brgconv_avx512_f32` | `1T->4T`: `2.264x`, `4T->8T`: `1.129x` | Pri nízkolatenčnom AVX-512 behu viazať vlákna na fyzické jadrá, ak SMT neprináša zisk |
| `X03` | Subnormálne čísla (`~3.6e-42`) v NumPy vs. OpenVINO `oneDNN` | NumPy spomalenie `14.004x` (zachová subnormály=`True`); oneDNN spomalenie `37.367x` (FTZ/DAZ nulovanie=`False`) | Vysvetľuje odlišné správanie pod prahom `1.18e-38` a chráni oneDNN pred mikrokódovou pascou |
| `X04` | Krok po `64 B` cache linajkach (`stride 16`) a zarovnanie `ov.Tensor` | Spomalenie `stride 16 / stride 1`: `9.586x`; zero-copy `64B`=`True`, `+4B`=`True` | Susediace uloženie (`C-contiguous`) je kritické; rozptýlené indexy ničia využitie 64 B linajok |
| `X05` | Little-endian binárny `128 B` rámec (`2x 64 B`) vs. UTF-8 JSON | `5.038x` rýchlejšie (`637348.6` rámcov/s, `77.8 MB/s`) | Potvrdzuje zmysel 128 B zmluvných rámcov oproti textovému JSON v kritickej slučke |

## Neodmerané hranice (`UNKNOWN`)

- Priame hardvérové počítadlá RAPL (Joule) a IMC šírky pásma nie sú v používateľskom priestore Windows čítané (UNKNOWN).
- Frekvencia jednotlivých jadier pod AVX-512 downclockingom nie je odčítaná z MSR IA32_PERF_STATUS (UNKNOWN).
