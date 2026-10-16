# Non-Invasive OS Coordination & Reversible Policy Engine Specification
**Document ID:** `ARCH-POLICY-ENGINE-2026-V2`  
**Milestone:** `V-NON-INVASIVE-OS-02`  
**Classification:** OS Koordinačná vrstva, reverzibilná správa subsystémov & Krystal Policy Engine  
**Global Invariant:** `VITAL_MAX_HP == 6` (Udržiavaný v globálnom zmluvnom kontexte repozitára)  

---

## 1. Účel a základná architektonická doktrína

Základným inžinierskym postulátom tejto špecifikácie je:
> **„Jeden policy mozog, tenké capability-based adaptéry, hostiteľský OS ostáva autorita, všetko merateľné, reverzibilné a auditovateľné.“**

Koordinačná vrstva neprepisuje OS kernel a nevytvára proprietárne neprenosné bypassy. Systém optimalizuje koexistenciu subsystémov výhradne cez **stabilné, existujúce rozhrania hostiteľského OS** s garanciou reverzibility.

---

## 2. Štyri kľúčové architektonické piliere (Revízia V2)

### 2.1. Anti-Batch Invariant: „Jedna nevalidovaná transakcia naraz“
* Pôvodné striktné pravidlo „jedna zmena naraz“ bolo príliš reštriktívne pre koordináciu – znemožňovalo overiť cross-subsystémovú synergiu (napr. súčasné nastavenie CPU affinity na P-cores a zníženie I/O queue latency).
* **Nový invariant:** V stave `TESTING / UNVALIDATED` smie byť v celom systéme **výhradne jedna jediná transakcia**.
* Transakcia môže obsahovať viacero koordinovaných akcií naprieč subsystémami, no tvorí **jeden atomický celok** s jednotnou hypotézou, orakulom a spoločným rollback plánom.

### 2.2. Rollback triedy (Rollback Classes)
Deterministický rollback nefunguje na všetkých fyzikálnych subsystémoch identicky. Obnovenie registrov nespôsobí okamžité fyzikálne ochladenie čipu ani okamžité znovunaplnenie zahodenej L3 cache. Preto zavádzame 4 explicitné triedy rollbacku:

| Trieda rollbacku | Význam a mechanizmus | Príklady subsystémov |
| :--- | :--- | :--- |
| **`EXACT_STATE`** | Okamžité a plné obnovenie diskrétneho stavu. | CPU cpuset affinity, cgroups limit, premenná prostredia, process priority. |
| **`CONFIG_RESTORE`** | Obnovenie riadiaceho parametra/registra; fyzikálny stav (teplota, P-state) konverguje cez dwell time. | Thermal power limit (PL1/PL2), clock scaling governor, GPU frequency floor. |
| **`DRAIN_AND_CONVERGE`** | Asynchrónne vyprázdnenie frontov a vyrovnanie tokov pred návratom. | Page-cache dirty writeback flush, network socket buffer drain. |
| **`IRREVERSIBLE_SIDE_EFFECT`** | Zmena parametra je vratná, no zanecháva nevratný vedľajší efekt (napr. strata horúcej cache). | Invalidation of compiled JIT traces, dropped filesystem buffer cache. |

### 2.3. Oddelenie eBPF do Observability (Mimo Actuation hierarchie)
* eBPF v našej architektúre neslúži ako kontrolná plocha na priamy zápis/actuation.
* **eBPF patrí výhradne do vrstvy pasívnej telemetrie (Observability & Tracing)** spolu s HW performance counters, ETW a procfs pollingom.
* **Hierarchia kontrolných plôch pre Actuation** má 10 úrovní (od najmenej invazívnej po najviac):
  1. `App Config` (aplikačná konfigurácia, flags)
  2. `Env Vars` (premenné prostredia)
  3. `systemd drop-ins` (service slices)
  4. `cgroups v2` / `Windows Job Objects`
  5. `sysfs` & `procfs` (dynamické runtime zápisy)
  6. `sysctl` (kernel tunables)
  7. `IRQ affinity` (smerovanie prerušení)
  8. `Parametre modulov` (`/etc/modprobe.d/` runtime reload)
  9. `Boot parametre` (cmdline - len ako predposledná možnosť)
  10. `Kernel patch` (posledná núdzová možnosť, zakázaná pokiaľ existujú userspace rozhrania)

### 2.4. Oddelenie Linuxu a Windows cez Capability Providery
Aby sa nemiešali rozhrania rôznych platforiem, tenké adaptéry nevolajú operačný systém priamo, ale dopytujú sa cez **Capability Provider**:
* `LinuxCapabilityProvider`: poskytuje `cgroups_v2`, `sysfs`, `sysctl`, `systemd`, `sched_setaffinity`.
* `WindowsCapabilityProvider`: poskytuje `JobObjects`, `SetProcessInformation(ProcessPowerThrottling / EcoQoS)`, `RegistryDisplayParameters`, `DXGI`, `ETW`.
* Adaptér dynamicky zistí dostupné schopnosti hostiteľského OS a odmietne nekompatibilný zásah bez rizika behovej chyby.

---

### 2.5. Komponentové oddelenie: Proposer != Evaluator
* Oddelenie rolí nie je iba logická podmienka (`if id1 == id2`), ale **samostatná fyzická komponentová architektúra**:
  * **`PolicyProposer`**: generuje hypotézy a navrhuje transakcie. Nemá oprávnenie hodnotiť úspech ani commitovať transakciu.
  * **`ValidationEvaluator`**: nezávislá entita s vlastným orakulom a telemetrickým vstupom. Ako jediná má právo vydať verdict `APPROVE` alebo nariadiť `ROLLBACK`.

---

## 3. Architektonická schéma systému

```
  +-------------------------------------------------------------------------------+
  |                        OBSERVABILITY VRSTVA (Telemetria)                      |
  |  - eBPF Tracepointy    - ETW (Windows)    - HW Performance Counters (TSC/PMC)  |
  +---------------------------------------+---------------------------------------+
                                          | Pasívna telemetria (Baseline & Post)
                                          v
  +-------------------------------------------------------------------------------+
  |                        POLICY PROPOSER (Centrálny mozog)                      |
  |  - Analyzuje úzke miesta a generuje koordinačnú transakciu                    |
  |  - Priraďuje kontrolné plochy (úrovne 1-10) a Rollback triedy                 |
  +---------------------------------------+---------------------------------------+
                                          | Navrhnutá transakcia (1 aktívna naraz)
                                          v
  +-------------------------------------------------------------------------------+
  |                         TRANSACTIONAL CHANGE STACK                            |
  |  - Anti-Batch: Presne 1 nevalidovaná transakcia v stave TESTING               |
  |  - LIFO Rollback Stack s evidenciou RollbackClass                            |
  +---------------------------------------+---------------------------------------+
                                          | Koordinovaný zásah cez adaptéry
                                          v
  +-------------------------------------------------------------------------------+
  |                 CAPABILITY-BASED TENKÉ ADAPTÉRY (CPU, Mem, IO, Accel, Therm)  |
  |  - LinuxCapabilityProvider        vs        - WindowsCapabilityProvider       |
  |    (cgroups, sysfs, sysctl)                    (Job Objects, EcoQoS, Registry) |
  +---------------------------------------+---------------------------------------+
                                          | Zmeraný efekt po zmene
                                          v
  +-------------------------------------------------------------------------------+
  |                        VALIDATION EVALUATOR (Audit Gate)                      |
  |  - Komponentovo oddelený od Proposera                                         |
  |  - Porovnáva baseline vs post-change voči falzifikátoru                       |
  |  - Verdikt: COMMIT (Supported)  alebo  LIFO ROLLBACK                          |
  +-------------------------------------------------------------------------------+
```

---

## 4. Šablóna koordinačnej transakcie

```json
{
  "transaction_id": "tx_coordination_uma_01",
  "hypothesis": "Pin compute workers to P-cores and lower I/O queue latency during active scanout",
  "actions": [
    {
      "subsystem": "CPU",
      "control_surface": "cgroups_v2_or_job_object",
      "tier": 4,
      "rollback_class": "EXACT_STATE",
      "rollback_payload": { "cpuset": "0-7" }
    },
    {
      "subsystem": "IO",
      "control_surface": "sysfs_or_registry",
      "tier": 5,
      "rollback_class": "CONFIG_RESTORE",
      "rollback_payload": { "scheduler": "mq-deadline" }
    }
  ],
  "target_metric": "VSync Jitter and LLC Contention",
  "expected_gain": "-35% Jitter",
  "side_effects": "+1.5W CPU package power",
  "falsifier": "Jitter exceeds 2.5ms or render FPS drops below 118",
  "status": "testing"
}
```

---

## 5. Globálne invarianty

1. **`VITAL_MAX_HP == 6`:** Zachovaný v globálnych kontraktoch Krystal Stacku (zmluvy, tokeny, receipts).
2. **Autorita hostiteľa:** Hostiteľský OS ostáva nespochybniteľnou autoritou; zásahy sú len koordinované odporúčania cez existujúce rozhrania.
