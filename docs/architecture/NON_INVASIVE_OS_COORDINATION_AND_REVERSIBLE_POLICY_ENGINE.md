# Non-Invasive OS Coordination & Reversible Policy Engine Specification
**Document ID:** `ARCH-POLICY-ENGINE-2026-V1`  
**Milestone:** `V-NON-INVASIVE-OS-01`  
**Classification:** OS Koordinačná vrstva, reverzibilná správa subsystémov & Krystal Policy Engine  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  

---

## 1. Účel a základný princíp

Základným inžinierskym postulátom tejto špecifikácie je:
> **„Neprepisovať operačný systém, ale postaviť koordinačnú vrstvu, ktorá zasahuje do subsystémov výhradne cez stabilné, existujúce rozhrania a každú zmenu dokáže deterministicky vrátiť späť (Transactional Rollback).“**

Namiesto zásahov do zdrojového kódu kernelu, inštalácie proprietárnych nestabilných modulov alebo binárneho patchovania OS systém operuje ako inteligentný **Krystal Policy Engine**, ktorý:
1. Rešpektuje pôvodnú architektúru hostiteľského OS (Windows / Linux).
2. Preferuje rozhrania s najmenšou mierou invazivity (od aplikačnej konfigurácie po sysctl).
3. Vykonáva zmeny striktne po jednej (zákaz „desať zmien naraz“).
4. Každú zmenu formuluje ako falzifikovateľnú hypotézu a meria stav pred a po.
5. Pri akejkoľvek regresii alebo falzifikácii vykoná okamžitý LIFO rollback.

---

## 2. Fáza Discovery: Zmapovanie subsystémov OS

Pred akýmkoľvek zásahom koordinačná vrstva vykonáva pasívny discovery audit 10 subsystémov:

| Subsystém | Skúmané charakteristiky a parametre | Existujúce stabilné rozhrania |
| :--- | :--- | :--- |
| **1. CPU** | Topológia (P-core vs E-core), frekvenčné domény, scaling governory, EPP (Energy Performance Preference), scheduler time-slice, cpuset affinity | `/sys/devices/system/cpu/`, `PowerRegisterSuspendResumeNotification`, Win32 `SetThreadSelectedCpuSets` |
| **2. Pamäť** | RAM kapacita, LPDDR zbernica, dirty writeback thresholds, swapinness, zram kompresné algoritmy, HugePages, Transparent HugePages (THP) | `/proc/sys/vm/`, Windows VirtualAlloc telemetry, MEMORYSTATUSEX |
| **3. I/O Storage** | I/O schedulery (BFQ, Kyber, mq-deadline), queue depth, read-ahead buffer, write-cache flushes, blkio limits | `/sys/block/*/queue/`, DeviceIoControl (FSCTL), storage bus counters |
| **4. Filesystem** | Mount options (noatime, commit=30), journaling režimy, page-cache flush intervals | `/proc/mounts`, Win32 GetVolumeInformationByHandle |
| **5. Sieť** | TCP socket buffery (rmem, wmem), TCP congestion control (BBR, CUBIC), NIC ring buffery, GRO/GSO/TSO offloading | `sysctl net.ipv4.*`, ethtool, IP Helper API |
| **6. Akcelerátory** | GPU/NPU compute queues, UMA clock domains, power limits, DirectX/Vulkan context priority, Level Zero / OpenVINO scheduler | DXGI adapter telemetry, OpenVINO Core, Level Zero sysman |
| **7. Termály** | Teplotné zóny, junction temperatures, power limits (PL1, PL2, Tau), passive/active cooling policies | `/sys/class/thermal/`, Intel DPTF / Energy Driver, Windows Thermal Zones |
| **8. cgroups** | cgroups v2 resource hierarchy, cpu.weight, cpu.max, memory.high, memory.max, io.weight | `/sys/fs/cgroup/`, Windows Job Objects |
| **9. systemd / Init** | Service slices, unit drop-ins, CPUQuota, MemoryHigh, IOWeight | `systemctl set-property`, systemd drop-in files (`.d/*.conf`) |
| **10. Ovládače** | Driver runtime parameters, GPU scheduling flags (HAGS on/off), TDR timeouts | `/sys/module/*/parameters/`, Windows Registry Display Parameters |

---

## 3. Hierarchia kontrolných plôch (Control Surface Hierarchy)

Pri voľbe zásahu systém striktne postupuje zhora nadol. Nižšia vrstva sa smie použiť **výhradne vtedy**, ak vyššia vrstva neumožňuje dosiahnuť merateľný efekt:

```
[NAJMENEJ INVAZÍVNE / NAJVIAC PREFEROVANÉ]
  1. Aplikačná konfigurácia (App Config, flags, runtime JSON)
  2. Premenné prostredia (Environment Variables)
  3. Systemd Drop-ins (service slices, resource restrictions)
  4. cgroups v2 (CPU/Memory/IO resource limits, Job Objects)
  5. sysfs & procfs (runtime dynamically writable tunables)
  6. sysctl (kernel network/vm parameters)
  7. IRQ affinity (smerovanie sieťových a GPU prerušení na dedikované jadrá)
  8. eBPF pozorovanie (výhradne pasívna telemetria a tracepointy)
  9. Parametre modulov (/etc/modprobe.d/ runtime reloading)
 10. Boot parametre (GRUB / systemd-boot cmdline - len ako predposledná možnosť)
 11. Kernel patch (LEN A LEN VTEDY, KEĎ NIČ INÉ NEFUNGUJE)
[NAJVIAC INVAZÍVNE / ZÁKAZ BEZ EXPLICITNÉHO SCHVÁLENIA]
```

---

## 4. Architektúra Krystal Policy Engine

Systém rozdeľuje zodpovednosti do troch izolovaných funkčných entít s prísnym zákazom miešania rolí:

```
                 +---------------------------------------------+
                 |            TELEMETRIA (OBSERVER)            |
                 |   - Čisto pasívne čítanie telemetrie        |
                 |   - Meria baseline a post-change metriky    |
                 +----------------------+----------------------+
                                        | Raw Telemetry
                                        v
                 +---------------------------------------------+
                 |        KRYSTAL POLICY ENGINE (PROPOSER)     |
                 |   - Centrálny riadiaci mozog                |
                 |   - Generuje cross-subsystémové hypotézy    |
                 |   - Udržiava transakčný stack zmien         |
                 +----------------------+----------------------+
                                        | Navrhnutá zmena
                                        v
     +--------------------------------------------------------------------+
     |                         TENKÉ ADAPTÉRY                             |
     |  [CPUAdapter] [MemoryAdapter] [IOAdapter] [AccelAdapter] [Thermal] |
     |      (Zásahy iba cez existujúce rozhrania s rollbackom)            |
     +----------------------------------+---------------------------------+
                                        | Meranie po zmene
                                        v
                 +---------------------------------------------+
                 |          VALIDATION GATE (EVALUATOR)        |
                 |   - Nezávislý audit (Anti-Self-Certification)|
                 |   - Porovnáva post-change s falzifikátorom  |
                 |   - Verdikt: APPROVED / ROLLBACK            |
                 +---------------------------------------------+
```

1. **Telemetria len pozoruje (Telemetry / Observer):** Číta počítadlá, jitter, latenciu, LLC cache miss rate a teplotu. Nemá právo meniť žiadnu konfiguráciu.
2. **Politika navrhuje (Policy Engine / Proposer):** Analyzuje úzke miesta a generuje izolovanú hypotézu s očakávaným ziskom a presným rollback skriptom.
3. **Validácia rozhoduje (Validation Gate / Evaluator):** Nezávislý audit porovnáva nameranú zmenu voči falzifikátoru. Ak nastane regresia alebo rozpor, spúšťa okamžitý návrat stavu.

---

## 5. Pravidlo izolovaných transakčných zmien (Anti-Batch Invariant)

* **Zákaz hromadných zmien:** Je striktne zakázané aplikovať viac ako 1 zmenu konfigurácie súčasne („žiadne desať zmien naraz“).
* **Záväzný životný cyklus zmeny:**
  $$\text{Baseline Telemetria} \longrightarrow \text{Aplikácia zmeny } S_i \longrightarrow \text{Post-Change Telemetria} \longrightarrow \text{Audit} \longrightarrow (\text{Commit} \lor \text{Rollback})$$
* **Povinný Rollback Receipt:** Každá aplikovaná zmena ukladá reverzný payload do LIFO zásobníka `TransactionalChangeStack`. Akýkoľvek neočakávaný stav alebo chyba spôsobí okamžité obnovenie pôvodných hodnôt v opačnom poradí.

---

## 6. Šablóna hypotézy optimalizačného mechanizmu

Pre každý navrhnutý mechanizmus engine eviduje:

```json
{
  "subsystem": "CPU",
  "control_surface": "cgroups_v2",
  "mechanism": "Pinned LLM compute worker to physical P-cores, excluding HyperThreads and E-cores during display VBlank",
  "target_metric": "VSync Jitter and LLC Cache Miss Rate",
  "expected_gain": "-35% LLC contention during active scanout",
  "side_effects": "Slightly increased power draw on P-cores (+1.2W)",
  "reversibility": "Immediate runtime rollback via cpu.cpuset write",
  "falsifier": "VSync jitter exceeds 2.5ms or render FPS drops below 118 FPS",
  "rollback_payload": { "previous_cpuset": "0-7" },
  "status": "hypothesis"
}
```

---

## 7. Hardvérové a konštitučné invarianty

1. **`VITAL_MAX_HP == 6`:** Žiadna zmena konfigurácie nesmie porušiť konštitučnú integritu Krystal Stacku.
2. **Zero-Destruction Guarantee:** Akékoľvek prerušenie spojenia, crash procesu alebo výpadok telemetrie musí automaticky iniciovať reset subsystémov do bezpečného východiskového stavu hostiteľského OS.
