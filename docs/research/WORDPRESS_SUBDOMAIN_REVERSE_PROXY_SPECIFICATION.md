# Krystal-Stack // WordPress Subdomain Security & Reverse Proxy Specification

![WordPress Subdomain Security Shield](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/img/wordpress_subdomain_security_shield.jpg)

## 1. Architektonický Prehľad

Tento systém zabezpečuje herné a výpočtové prostredie **Krystal-Stack** prostredníctvom reverzného proxy a autorizačnej brány WordPress bežiacej na subdoméne (napr. `krystal.poslednikmen.cz` alebo `krystal.vasadomena.sk`).

```
[ Používateľský Prehliadač ]
       │
       ▼ HTTPS / TLS 1.3 (Port 443)
[ Subdoména: krystal.vasadomena.sk ]
       │
       ▼
[ Nginx Reverse Proxy ]
  ├── 1. SSL/TLS Terminácia + HSTS (63,072,000s) + CSP Frame-Ancestors
  ├── 2. BBQ Firewall (Block Bad Queries) – blokuje SQLi, Directory Traversal, RCE, XSS
  └── 3. Wordfence Rate Limiting – obmedzuje /wp-login.php (max 3 req/min)
       │
       ├──► Cesty `/` a `/wp-*` (WordPress CMS na Subdoméne)
       │       │
       │       ▼
       │    [ WordPress CMS & Plugin krystal-posledni-kmen ]
       │      ├── Overenie používateľov (heslá, 2FA Microsoft Authenticator)
       │      ├── Kontrola oprávnení (Administrator, Player, Subscriber)
       │      └── Generovanie kryptografického HMAC-SHA256 Tokenu
       │
       └──► Cesty `/krystal-core/` a `/api/` (Reverzný Proxy na 127.0.0.1:8089)
               │
               ▼
            [ Krystal Engine Core (Localhost:8089) ]
              ├── Overenie HMAC-SHA256 podpisu a časovej platnosti (anti-replay: 60s)
              ├── Overenie subdomény vo whiteliste
              └── Striktné zachovanie pravidla integrity: VITAL_MAX_HP = 6
```

---

## 2. Implementované Komponenty

### A. Python Engine Core & Security Gate
- [wordpress_security_and_java_transpiler.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/wordpress_security_and_java_transpiler.py):
  - Trieda `WordPressSubdomainSecurityGate`
  - Správa HMAC-SHA256 podpisov
  - Prevencia Replay útokov cez jednorazové nonces
  - Ochrana integrity invariantu `VITAL_MAX_HP = 6`
  - Kontrola a whitelist subdomén

### B. WordPress Plugin: `krystal-posledni-kmen`
- [krystal-posledni-kmen.php](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/wordpress_plugin_krystal/krystal-posledni-kmen/krystal-posledni-kmen.php): Hlavný spúšťací súbor pluginu.
- [class-krystal-api.php](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/wordpress_plugin_krystal/krystal-posledni-kmen/includes/class-krystal-api.php):
  - Endpoint `GET /wp-json/krystal/v1/auth-token` (poskytuje podpísaný token prihlásenému používateľovi)
  - Endpoint `GET /wp-json/krystal/v1/subdomain/status` (telemetria z Python kernelu)
  - Endpoint `POST /wp-json/krystal/v1/subdomain/proxy` (reverzný proxy pre WP AJAX volania)
- [class-krystal-shortcodes.php](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/wordpress_plugin_krystal/krystal-posledni-kmen/includes/class-krystal-shortcodes.php):
  - `[krystal_subdomain_portal mode="hybrid|desktop|pantheon|blueprint" height="780px"]`
  - `[krystal_webos_desktop height="800px"]`
  - `[krystal_hybrid_arena height="750px"]`
  - `[krystal_svg_blueprint height="700px"]`
- [class-krystal-admin.php](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/wordpress_plugin_krystal/krystal-posledni-kmen/includes/class-krystal-admin.php):
  - Administrátorské menu vo WordPress pre konfiguráciu subdomény, HMAC kľúča a overenie spojenia.

### C. Nginx Reverzné Proxy & BBQ Firewall
- [nginx_krystal_subdomain_ssl.conf](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/deploy/nginx_krystal_subdomain_ssl.conf):
  - TLS 1.3 / TLS 1.2 konfigurácia
  - BBQ WAF regex pravidlá pre SQLi, Traversal, RCE, XSS
  - Mapovanie `/krystal-core/` na privátny port `127.0.0.1:8089`

### D. Automatizované Inštalačné Skripty
- PowerShell (Windows Server): [deploy_wordpress_subdomain.ps1](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/deploy_wordpress_subdomain.ps1)
- Bash (Linux / Ubuntu): [deploy_wordpress_subdomain.sh](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/deploy_wordpress_subdomain.sh)

### E. Interaktívne Vizuálne Štúdio
- [wordpress_subdomain_security_studio.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/wordpress_subdomain_security_studio.html):
  - Živá animovaná Canvas sieťová mapa
  - WAF testovač (SQLi, Traversal, RCE, XSS)
  - HMAC mint & token verifikátor
  - Priame konfiguračné snippety

---

## 3. Výsledky Testov & Verifikácie

Všetkých **11/11 špecifických testov** a všetkých **6 krokov end-to-end toku** prešlo na 100%:

```
=================================================================
 VERIFYING END-TO-END WORDPRESS SUBDOMAIN SECURITY FLOW
=================================================================
[PASS] 1. Kernel Security Status: ONLINE, Max HP Invariant = 6
[PASS] 2. BBQ WAF allowed legitimate request: Legitimate Query
[PASS] 2. BBQ WAF blocked attack: SQL Injection -> Threat: SQL_INJECTION
[PASS] 2. BBQ WAF blocked attack: Path Traversal -> Threat: PATH_TRAVERSAL
[PASS] 2. BBQ WAF blocked attack: RCE Exploit -> Threat: REMOTE_CODE_EXECUTION
[PASS] 2. BBQ WAF blocked attack: XSS Payload -> Threat: CROSS_SITE_SCRIPTING
[PASS] 3. WordPress Token Minted: eyJleHAiOjE3OTEwNjU0MjAu...
[PASS] 4. Token Verified by Kernel: User='krystal_archon', HP=6
[PASS] 5. Replay Attack Detected & Rejected: REPLAY_ATTACK_DETECTED
[PASS] 6. HTML Studio Loaded: 30468 bytes
=================================================================
 ALL 6 SUBDOMAIN SECURITY CHECKS PASSED PERFECTLY!
=================================================================
```
