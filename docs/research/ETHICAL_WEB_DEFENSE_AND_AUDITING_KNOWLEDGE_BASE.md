# Krystal-Stack: White-Hat Web Application Defense & Ethical Auditing Knowledge Base
## Inšpirované metodikou "Hackni vlastný web" (Audituj a zabezpeč vlastný systém)

**Autor:** Krystal-Stack Security Architecture & Research Team  
**Dátum:** Október 2026  
**Status:** Kanonická bezpečnostná ontológia & Audítorská príručka (Defense-in-Depth)  

---

## 1. Filozofia & Etický Rámec: "Hackni vlastný web skôr, než to urobí útočník"

Základným princípom publikácie a metodiky *"Hackni vlastný web"* je proaktívny prístup k bezpečnosti: **vývojár alebo administrátor musí poznať útočné vektory a vzorce zraniteľností, aby dokázal navrhnúť nepriestrelnú obranu.**

V prostredí **Krystal-Stack Platform Framework** (ktorý spája Python, Janet Lisp, Java 21, WordPress Docker a heterogénne výpočtové API) aplikujeme viacvrstvovú architektúru **Defense-in-Depth**, kde zlyhanie jednej bezpečnostnej vrstvy nikdy neohrozí integritu celého systému.

```
+-----------------------------------------------------------------------------------------+
|                        DEFENSE-IN-DEPTH BEZPEČNOSTNÝ MODEL                              |
+-----------------------------------------------------------------------------------------+
|  VRSTVA 1: Obvodová ochrana (Perimeter & Edge)                                         |
|  - Nginx reverzný proxy s BBQ (Block Bad Queries) firewallom                           |
|  - Rate limiting (300 req/min), fail2ban pravidlá, blokovanie IP rozsahov              |
+-----------------------------------------------------------------------------------------+
|  VRSTVA 2: Aplikačný firewall (Layer 7 WAF)                                            |
|  - AdaptiveApplicationFirewall (DPI - Deep Packet Inspection)                          |
|  - Heuristická analýza SQLi, XSS, Path Traversal a Command Injection                   |
+-----------------------------------------------------------------------------------------+
|  VRSTVA 3: Autentifikácia & Riadenie identity (Identity & Zero Trust)                  |
|  - RFC 6238 TOTP 2FA (Microsoft Authenticator s 2-digit number matchingom)             |
|  - Kryptografické hashovanie hesiel: Argon2id (m=64MB, t=3, p=4) a PBKDF2-SHA512       |
|  - Wordfence-style adaptívne uzamykanie IP pri brute-force útokoch                     |
+-----------------------------------------------------------------------------------------+
|  VRSTVA 4: Biznis logika & Dátová integrita (Engine & Invariants)                      |
|  - Nemenný invariant 6 Max HP (zamedzenie pretečenia a logických glitchov v boji)      |
|  - Antispam Bee pasce (honeypot polia a časové prahy odoslania formulárov)             |
+-----------------------------------------------------------------------------------------+
|  VRSTVA 5: Izolácia kontajnerov & Úložiska (Storage & Isolation)                       |
|  - Docker compose izolované siete bez priameho prístupu do host OS                     |
|  - Zákaz vykonávania skriptov v upload adresároch (noexec / PHP script execution off)  |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Katalóg 10 Kľúčových Vzorcov Útokov & Obranných Mechanizmov

Vychádzajúc zo štandardov OWASP Top 10 a praktických príkladov testovania webu:

### 1. Injekčné útoky (SQL Injection - SQLi)
- **Vzorec útoku:** Útočník vkladá SQL operátory (`' OR '1'='1`, `UNION SELECT`, `benchmark()`, `sleep()`) do vstupných parametrov za účelom manipulácie s databázou.
- **Detekcia v Krystal-Stack:**
  - BBQ regex vo firewalle: `(union.*select|concat\(|into.*outfile|load_file\(|benchmark\(|sleep\()`.
- **Defenzívne riešenie:**
  - Výhradné použitie parametrizovaných dopytov (Prepared Statements) alebo objektových ORM rozhraní.
  - Zamedzenie dynamického skladania SQL reťazcov (`f"SELECT * FROM users WHERE name = '{user}'"` je striktne zakázané).

### 2. Zlomená autentifikácia & Brute-Force útoky
- **Vzorec útoku:** Slovníkové útoky a credential stuffing na prihlasovacie formuláre (`/wp-login.php`, `/api/auth/login`).
- **Defenzívne riešenie v Krystal-Stack:**
  - **Wordfence Brute Force Engine:** Sledovanie neúspešných pokusov podľa IP adresy; po 5 neúspešných pokusoch nasleduje 30-minútový lockout.
  - **Microsoft Authenticator 2FA:** Povinná dvojfaktorová autentifikácia cez RFC 6238 TOTP tokeny s 2-digit number matching výzvou na mobilnom zariadení.
  - **Heslá:** Ukladanie výhradne ako Argon2id solene hash reťazce.

### 3. Cross-Site Scripting (XSS - Reflected, Stored, DOM)
- **Vzorec útoku:** Vloženie škodlivého JavaScriptu (`<script>alert(1)</script>`, `onerror=`, `javascript:`) do vstupov, ktorý sa vykoná v prehliadači obete a odcudzí session cookies.
- **Defenzívne riešenie v Krystal-Stack:**
  - Kontextové escapovanie výstupov (`html.escape` v Pythone, bezpečné DOM uzly `element.textContent` namiesto `innerHTML`).
  - Striktné bezpečnostné hlavičky v Nginx:
    ```nginx
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;";
    add_header X-XSS-Protection "1; mode=block";
    add_header X-Content-Type-Options "nosniff";
    ```

### 4. Path Traversal & Local File Inclusion (LFI/RFI)
- **Vzorec útoku:** Manipulácia s cestami k súborom cez sekvencie `../`, `..\\`, `/etc/passwd`, `boot.ini` za účelom prečítania citlivých konfiguračných súborov.
- **Defenzívne riešenie v Krystal-Stack:**
  - Bezpečná normalizácia cesty:
    ```python
    target = os.path.normpath(os.path.join(STATIC_DIR, rel_path))
    if not target.startswith(os.path.normpath(STATIC_DIR)):
        return 403_FORBIDDEN
    ```
  - BBQ regex filter blokujúci akékoľvek `../` pred doručením aplikácii.

### 5. Cross-Site Request Forgery (CSRF)
- **Vzorec útoku:** Prinútenie prehliadača obete odoslať nechcenú akciu (napr. zmenu hesla) na zraniteľný web, kde je používateľ prihlásený.
- **Defenzívne riešenie v Krystal-Stack:**
  - Unikátne kryptografické Anti-CSRF tokeny pre každú session.
  - Nastavenie `SameSite=Strict` alebo `SameSite=Lax` a príznak `Secure; HttpOnly` pre všetky autentifikačné cookies.

### 6. Spam boty & Automatizované formuláre (Antispam Bee filozofia)
- **Vzorec útoku:** Automatizované skripty zaplavujúce registračné formuláre, komentáre a API fiktívnymi údajmi bez vykonania reálneho JavaScriptu.
- **Defenzívne riešenie v Krystal-Stack:**
  - **Skryté honeypot pole:** `krystal_trap_honey_bee` skryté pred reálnymi používateľmi cez CSS. Ak bot pole vyplní, požiadavka je okamžite ticho zahodená.
  - **Časová pasca (Timing trap):** Človek potrebuje na vyplnenie formulára aspoň 3 sekundy. Ak je formulár odoslaný rýchlejšie ako za 3.0 s, ide o bota.

### 7. Zraniteľnosti pri nahrávaní súborov (File Upload Vulnerabilities)
- **Vzorec útoku:** Nahratie PHP/Python/CGI skriptu maskovaného ako obrázok (`shell.php.jpg` alebo zneužitie dvojitej prípony) a jeho následné spustenie cez HTTP URL.
- **Defenzívne riešenie v Krystal-Stack:**
  - Striktný whitelist povolených prípon (`.jpg`, `.png`, `.webp`, `.obj`).
  - Overenie skutočného MIME typu cez hlavičkové magic bajty (napr. `\xff\xd8\xff` pre JPEG), nie len podľa názvu súboru.
  - Generovanie náhodných mien súborov bez zachovania pôvodného názvu od používateľa.
  - Konfigurácia webservera: zákaz spúšťania skriptov v adresári `/uploads/` (`location ~* ^/uploads/.*\.php$ { deny all; }`).

### 8. Nebezpečná deserializácia & Pretečenie pamäte (DDoS / Memory Exhaustion)
- **Vzorec útoku:** Odoslanie gigabajtového payloadu alebo zacyklených objektov, ktoré vyčerpajú pamäť RAM servera.
- **Defenzívne riešenie v Krystal-Stack:**
  - Striktné obmedzenie veľkosti tela HTTP požiadavky:
    ```python
    content_length = int(self.headers.get('Content-Length', 0))
    if content_length > 10 * 1024 * 1024: # max 10 MB
        return 413_PAYLOAD_TOO_LARGE
    ```
  - Zákaz používania nebezpečného `pickle.loads` na nespoľahlivé vstupy; použitie výhradne striktného `json.loads`.

### 9. Logické chyby v hre a biznis procesoch (Invariant Violations)
- **Vzorec útoku:** Manipulácia s parametrami požiadaviek (napr. odoslanie `hp = 999` alebo zápornej ceny tovaru v JSON tele).
- **Defenzívne riešenie v Krystal-Stack:**
  - **Nemenný invariant 6 Max HP:** Každá procedurálne generovaná alebo prijímaná entita je automaticky orezaná:
    $$\text{hp} = \min(\text{max}(0, \text{hp}), 6)$$
  - Zlyhanie invariantu vyvoláva okamžité odmietnutie transakcie v ekonomickej účtovnej knihe (`EconomicLedger`).

### 10. Bezpečnostná miskonfigurácia & Chýbajúce HTTP hlavičky
- **Vzorec útoku:** Získavanie informácií o verzii softvéru z hlavičiek `Server:` a `X-Powered-By:` za účelom vyhľadania známych CVE zraniteľností.
- **Defenzívne riešenie v Krystal-Stack:**
  - Skrytie identifikačných hlavičiek v Nginx (`server_tokens off;`).
  - Nasadenie balíka bezpečnostných hlavičiek:
    - `Strict-Transport-Security: max-age=31536000; includeSubDomains; preload`
    - `X-Frame-Options: SAMEORIGIN` (ochrana pred Clickjackingom)
    - `X-Content-Type-Options: nosniff`
    - `Referrer-Policy: strict-origin-when-cross-origin`

---

## 3. Prepojenie na Existujúcu Krystal-Stack Architektúru

Všetky vyššie uvedené vzorce sú priamo integrované v bežiacich moduloch repozitára:

| Modul v Krystal-Stack | Úloha v Bezpečnostnom Rámci |
| :--- | :--- |
| [monitoring_clusters_and_firewall.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/monitoring_clusters_and_firewall.py) | **AdaptiveApplicationFirewall**: Layer 7 inšpekcia paketov, adaptívny rate-limiting a automatický IP ban. |
| [wordpress_security_and_java_transpiler.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/wordpress_security_and_java_transpiler.py) | **WordPressSecurityEngine**: BBQ Firewall regex filtre, Antispam Bee pasce a Wordfence brute force lockouts. |
| [microsoft_authenticator_2fa.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/microsoft_authenticator_2fa.py) | **MicrosoftAuthenticator2FAEngine**: RFC 6238 TOTP, 2-digit number matching, ochrana pred replay útokmi a záložné SHA-256 kódy. |
| [nginx_bbq_security.conf](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/deploy/nginx_bbq_security.conf) | Nginx reverzný proxy s ochranou pred bad queries, obmedzením rýchlosti a zákazom spúšťania skriptov. |
| [docker-compose.wordpress.yml](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docker-compose.wordpress.yml) | Izolované Docker prostredie pre WordPress, MariaDB a Nginx s internou sieťou `krystal_secure_net`. |
| [.agents/rules/consistent-procedural-generation-patterns.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/.agents/rules/consistent-procedural-generation-patterns.md) | Vynútenie nemenného pravidla **6 Max HP** a deterministickej seed hygieny. |

---

## 4. 25-Bodový Kontrolný Zoznam pre Vlastný Audit Webových Aplikácií

Pred nasadením akejkoľvek aplikácie alebo API do produkcie vykonajte tento audit:

- [ ] **1. SQLi:** Sú všetky SQL dopyty parametrizované?
- [ ] **2. BBQ Firewall:** Sú aktívne regex vzory pre blokovanie `union select`, `concat(`, `benchmark(`?
- [ ] **3. XSS:** Používa sa `textContent` alebo automatické escapovanie namiesto `innerHTML`?
- [ ] **4. CSP:** Je nastavená prísna hlavička `Content-Security-Policy`?
- [ ] **5. Clickjacking:** Je prítomná hlavička `X-Frame-Options: SAMEORIGIN`?
- [ ] **6. MIME Sniffing:** Je prítomná hlavička `X-Content-Type-Options: nosniff`?
- [ ] **7. HSTS:** Je vynútené HTTPS cez `Strict-Transport-Security`?
- [ ] **8. Cookies:** Majú session cookies príznaky `Secure; HttpOnly; SameSite=Strict`?
- [ ] **9. Heslá:** Ukladajú sa heslá cez Argon2id alebo PBKDF2 s unikátnou soľou?
- [ ] **10. 2FA:** Je pre administrátorov aktívny TOTP 2FA s ochranou proti replay útokom?
- [ ] **11. Brute-Force:** Je nastavený lockout po maximálne 5 neúspešných pokusoch?
- [ ] **12. Path Traversal:** Blokuje aplikácia `../` a normalizuje cesty cez `os.path.normpath`?
- [ ] **13. File Uploads:** Je zakázané spúšťanie skriptov v adresári s nahratými súbormi?
- [ ] **14. File Types:** Overujú sa nahrávané súbory podľa magic bajtov, nie prípony?
- [ ] **15. Veľkosť tela:** Odmieta server požiadavky väčšie ako povolený limit (napr. 10 MB)?
- [ ] **16. Chybové hlášky:** Sú podrobné tracebacky vypnuté v produkčnom režime?
- [ ] **17. Directory Indexing:** Je vypnuté listovanie adresárov (`autoindex off;`)?
- [ ] **18. Default Credentials:** Boli zmenené všetky predvolené heslá (admin, db_password)?
- [ ] **19. Honeypot:** Obsahujú formuláre skryté pasce pre spambotov (`Antispam Bee`)?
- [ ] **20. Timing Trap:** Odmietajú sa formuláre odoslané za menej ako 3 sekundy?
- [ ] **21. Rate Limiting:** Je obmedzený počet požiadaviek z jednej IP (max 300 req/min)?
- [ ] **22. Secrets v Git:** Sú všetky API kľúče a heslá mimo gitu v `.env` súbore?
- [ ] **23. Docker izolácia:** Bežia kontajnery pod neprivilegovaným používateľom (`non-root`)?
- [ ] **24. Vital Invarianty:** Sú herné/ekonomické hodnoty orezané (napr. Max HP = 6)?
- [ ] **25. Logovanie:** Sú zaznamenávané bezpečnostné incidenty a pokusy o útok?

---

*Tento dokument slúži ako stály znalostný štandard pre vývoj a audit webových služieb na platforme Krystal-Stack.*
