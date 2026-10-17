# Non-Invasive OS Coordination & Reversible Policy Engine Specification (V3)
**Document ID:** `ARCH-POLICY-ENGINE-2026-V3`  
**Milestone:** `V-NON-INVASIVE-OS-03`  
**Classification:** OS Koordinačná vrstva, odolnostné testovanie, domain locking & Krystal Policy Engine  
**Global Invariant:** `GLOBAL_VITAL_MAX_HP == 6`  

---

## 1. Upravená architektonická doktrína (Revízia V3)

> **„Jeden policy mozog, tenké capability-based adaptéry, hostiteľský OS ostáva autorita. Reverzibilná je konfigurácia (nie vždy fyzikálny stav systému), transakcie sú izolované na úrovni domén a kernel patch je za tvrdou bezpečnostnou bránou.“**

### Tri kľúčové spresnenia:
1. **Konfigurácia vs. Fyzikálny stav:**
   * Systém garantuje **deterministickú reverzibilitu konfigurácie, riadiacich registrov a systémových politík**.
   * Fyzikálny a dynamický stav systému (teplota čipu, obsadenie L3/L2 cache, naplnenie sieťových socketov) podlieha fyzikálnym zákonom, hysterézii a termálnej zotrvačnosti. Rollback triedy (`EXACT_STATE`, `CONFIG_RESTORE`, `DRAIN_AND_CONVERGE`, `IRREVERSIBLE_SIDE_EFFECT`) exaktne modelujú tento rozdiel.
2. **Domain-Scoped Transaction Locking:**
   * Namiesto jediného hrubého globálneho zámku zavádzame **doménovo viazané zámky** (`DomainLockManager`).
   * Zmeny v disjunktných doménach (napr. `NETWORK_DOMAIN` vs. `STORAGE_DOMAIN`) môžu bežať paralelne bez vzájomnej blokácie, zatiaľ čo transakcie v prekrývajúcich sa doménach (napr. CPU affinity a Thermal power limit v `COMPUTE_DOMAIN`) ostávajú striktne serializované („jedna nevalidovaná transakcia na doménu“).
3. **Kernel Patch za tvrdou bezpečnostnou bránou (`HardExclusionSafetyGate`):**
   * Kernel patch nie je bežná úroveň v hierarchii kontrolných plôch.
   * Bežná hierarchia Actuation má **9 štandardných userspace úrovní** (1..9).
   * Zásah do kernelu je vyčlenený do izolovanej zóny vyžadujúcej explicitný kryptografický/manuálny override token a je prísne zakázaný pre akékoľvek automatizované navrhovanie politík.

---

## 2. Architektonické domény subsystémov (Domain Partitioning)

Pre zabránenie úzkym hrdlám globálneho zámku sú subsystémy rozdelené do ortogonálnych domén:

```
  +-------------------------------------------------------------------------+
  |                           SUBSYSTEM DOMAINS                             |
  |                                                                         |
  |  [ COMPUTE & THERMAL DOMAIN ]       [ STORAGE & I/O DOMAIN ]            |
  |  - CPU (P/E cores, EPP, scheduler) - Storage I/O schedulers            |
  |  - Accelerator (GPU/NPU queues)     - Filesystem mount / write-cache    |
  |  - Thermal (PL1/PL2, cooling)                                           |
  |                                     [ NETWORK DOMAIN ]                  |
  |  [ PLATFORM SLICES DOMAIN ]         - TCP socket buffers & congestion   |
  |  - cgroups v2 / Windows Job Objects - NIC offloading & ring buffers     |
  |  - systemd / Windows Services                                           |
  +-------------------------------------------------------------------------+
```

* **Pravidlo paralelizmu:** Transakcia zasahujúca výhradne `STORAGE_DOMAIN` neblokuje transakciu v `NETWORK_DOMAIN`.
* **Pravidlo kolízie:** Ak transakcia $T_A$ a transakcia $T_B$ zdieľajú aspoň jeden subsystém alebo doménu (napr. $T_A$ mení CPU a $T_B$ mení Thermal), $T_B$ musí čakať na dokončenie auditu $T_A$.

---

## 3. Hierarchia kontrolných plôch a Hard Exclusion Gate

```
[ŠTANDARDNÉ KONTROLNÉ PLOCHY PRE ACTUATION]
  Tier 1: App Config (aplikačné JSON/flags)
  Tier 2: Env Vars (premenné prostredia)
  Tier 3: systemd drop-ins / Service overrides
  Tier 4: cgroups v2 / Windows Job Objects
  Tier 5: sysfs & procfs (dynamické runtime tunables)
  Tier 6: sysctl (kernel network/vm tunables)
  Tier 7: IRQ affinity (smerovanie prerušení)
  Tier 8: Parametre modulov (/etc/modprobe.d/ runtime reload)
  Tier 9: Boot parametre (cmdline - len ako predposledná možnosť)

-----------------------------------------------------------------------------
================== HARD EXCLUSION SAFETY BARRIER ============================
  [ HardExclusionSafetyGate: Kernel Patch ]
  - VYČLENENÝ MIMO BEŽNEJ HIERARCHIE.
  - Zakázaný pre akýkoľvek automatizovaný policy engine.
  - Vyžaduje explicitný kryptografický autorizačný token operátora.
=============================================================================
```

---

## 4. Odolnostný protokol (Resilience & Chaos Engineering)

Aby systém nezlyhal pri chybách prostredia alebo timeoutoch testovania, zavádza sa **Odolnostný protokol**:

```
        +-------------------------------------------------------------+
        |                 TRANSACTION LIFECYCLE MONITOR               |
        +------------------------------+------------------------------+
                                       |
        +------------------------------+------------------------------+
        | Timeout Exceeded (> 5s)      | Partial Failure during Apply |
        v                              v                              v
  [ AUTOMATIC ROLLBACK ]     [ ATOMIC REVERSE UNWIND ]      [ ADAPTER CRASH ]
  Okamžité zrušenie          Vrátenie úspešne aplikovaných  Fallback do
  a obnova konfigurácie      krokov v LIFO poradí           bezpečných hodnôt
```

1. **Transaction Timeout:** Ak transakcia v stave `TESTING` neposkytne audit do definovaného timeoutu (default 5.0s), je automaticky vyhlásená za zlyhanú a nasleduje okamžitý rollback.
2. **Partial Apply Failure:** Ak transakcia pozostáva z 3 akcií a druhá akcia zlyhá (napr. OS odmietne parameter), systém okamžite zroluje prvú akciu späť a transakciu odmietne bez zanechania polovičného stavu.
3. **Self-Healing Rollback:** Ak adaptér zlyhá počas samotného rollbacku, systém aplikuje záchrannú sadu bezpečných hodnôt (*Safe Baseline Defaults*) a zaloguje bezpečnostný incident.
