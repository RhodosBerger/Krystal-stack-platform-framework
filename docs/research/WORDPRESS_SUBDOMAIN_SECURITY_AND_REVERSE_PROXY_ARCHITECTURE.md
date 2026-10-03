# Architektúra Zabezpečenia Krystal-Stack cez WordPress na Subdoméne

Tento dokument definuje kompletnú architektúru, konfigurácie a postup nasadenia pre zabezpečenie hernej a výpočtovej platformy **Krystal-Stack** (FastRingBuffer, 3D/SDF Raymarching, WebOS, Panteón, procedurálne ostrovy) prostredníctvom webu **WordPress bežiaceho na subdoméne** (napr. `krystal.poslednikmen.cz` alebo `krystal.vasadomena.sk`).

---

## 1. Architektonický Vzor: Subdomain Security Gateway & Reverse Proxy

V produkčnom prostredí nesmie byť vysokorýchlostný Python kernel (`krystal_engine_core`, port `8089`) nikdy priamo vystavený do verejného internetu. Namiesto toho je chránený tromi vrstvami:

```
[ Používateľský Prehliadač ]
       │
       ▼ HTTPS / TLS 1.3 (Port 443)
[ Subdoména: krystal.vasadomena.sk ]
       │
       ▼
[ Nginx Reverse Proxy ]
  ├── 1. SSL/TLS Terminácia + HSTS (63,072,000s) + CSP
  ├── 2. BBQ Firewall (Block Bad Queries) – blokuje SQLi, Directory Traversal, RCE, XSS
  └── 3. Wordfence Rate Limiting – obmedzuje /wp-login.php (3 pokusy/min)
       │
       ├──► Ak URL začína na `/wp-` alebo `/` (WordPress CMS)
       │       │
       │       ▼
       │    [ WordPress CMS na Subdoméne ]
       │      ├── Overenie používateľov (heslá, 2FA Microsoft Authenticator)
       │      ├── Plugin `krystal-posledni-kmen`
       │      └── Generovanie kryptografického HMAC-SHA256 Bearer Tokenu
       │
       └──► Ak URL začína na `/krystal-core/` alebo `/api/`
               │
               ▼
            [ Krystal Engine Core (127.0.0.1:8089) ]
              ├── Overenie HMAC podpisu a časovej platnosti tokenu (anti-replay: 60s)
              ├── Overenie subdomény vo whiteliste
              └── Striktné zachovanie pravidla integrity: VITAL_MAX_HP = 6
```

---

## 2. DNS Konfigurácia Subdomény

U vášho poskytovateľa domény pridajte nasledujúci DNS záznam:

| Typ záznamu | Názov (Host) | Hodnota (Cieľ) | TTL | Účel |
| :--- | :--- | :--- | :--- | :--- |
| **A** | `krystal` | `VEREJNA_IP_SERVERA` | 300 s | Smerovanie subdomény na Nginx server |
| **CNAME** (alt.) | `krystal` | `vasadomena.sk.` | 300 s | Alias na hlavnú doménu |

---

## 3. Získanie SSL/TLS Certifikátu (Let's Encrypt Certbot)

Pre subdoménu vygenerujte platný SSL certifikát:

```bash
# Pre samostatnú subdoménu:
sudo certbot certonly --nginx -d krystal.vasadomena.sk

# Alebo s wildcard certifikátom pre celú doménu:
sudo certbot certonly --manual --preferred-challenges dns -d "*.vasadomena.sk" -d "vasadomena.sk"
```

Certifikáty umiestnite alebo namapujte do `/etc/nginx/certs/`:
- Certifikát: `/etc/nginx/certs/subdomain_cert.pem`
- Privátny kľúč: `/etc/nginx/certs/subdomain_key.pem`

---

## 4. Konfigurácia Nginx (Reverzné Proxy & BBQ Firewall)

Konfiguračný súbor sa nachádza v repozitári: [`deploy/nginx_krystal_subdomain_ssl.conf`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/deploy/nginx_krystal_subdomain_ssl.conf).

### Kľúčové bloky v Nginx:
1. **BBQ Firewall Regex**:
   ```nginx
   if ($query_string ~* "([a-z0-9_.]*|%2e%2e|/)(boot\.ini|etc/passwd|winnt|windows/system32|eval\(|base64_decode\(|passthru\(|system\(|shell_exec\()") {
       return 403;
   }
   if ($query_string ~* "(union.*select|benchmark\(|sleep\(|into.*outfile|load_file\(|waitfor.*delay)") {
       return 403;
   }
   ```
2. **Reverzný Proxy do Krystal Engine Core**:
   ```nginx
   location /krystal-core/ {
       limit_req zone=krystal_subdomain_api burst=20 nodelay;
       rewrite ^/krystal-core/(.*)$ /$1 break;
       proxy_pass http://127.0.0.1:8089;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_set_header Host $host;
       proxy_set_header X-WP-Subdomain $host;
   }
   ```

---

## 5. Konfigurácia WordPress (`wp-config.php`)

Do súboru `wp-config.php` na vašej WordPress inštalácii doplňte nasledujúce direktívy:

```php
// Nastavenie subdomény
define( 'WP_HOME', 'https://krystal.vasadomena.sk' );
define( 'WP_SITEURL', 'https://krystal.vasadomena.sk' );

// Umožnenie zdieľania cookies medzi hlavnou doménou a subdoménou (ak žiaduce)
define( 'COOKIE_DOMAIN', '.vasadomena.sk' );

// Bezpečnostné sprísnenie
define( 'FORCE_SSL_ADMIN', true );
define( 'DISALLOW_FILE_EDIT', true );
define( 'WP_AUTO_UPDATE_CORE', 'minor' );

// Zdieľaný kryptografický kľúč pre Krystal-Stack Subdomain Security Gate
define( 'KRYSTAL_SUBDOMAIN_SECRET', 'krystal_wp_subdomain_secure_secret_2026' );
```

---

## 6. WordPress Plugin: `krystal-posledni-kmen`

Plugin sa nachádza v adresári [`wordpress_plugin_krystal/krystal-posledni-kmen/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/wordpress_plugin_krystal/krystal-posledni-kmen/).

### Dostupné Shortcody:

1. **Plný Subdomain Portál s automatickou ochranou**:
   ```
   [krystal_subdomain_portal mode="hybrid" height="780px" require_login="true"]
   ```
   *Pokiaľ používateľ nie je prihlásený, zobrazí sa štýlová vstupná brána (Security Gate) s tlačidlom pre prihlásenie. Po prihlásení plugin vygeneruje časovo obmedzený HMAC token a bezpečne vloží herné štúdio.*

2. **WebOS Desktop (Multitasking Window Manager)**:
   ```
   [krystal_webos_desktop height="800px"]
   ```

3. **Hybrid Portal Arena (3D & SDF Engine)**:
   ```
   [krystal_hybrid_arena height="750px"]
   ```

4. **Evolved SVG Master Blueprint**:
   ```
   [krystal_svg_blueprint height="700px"]
   ```

5. **Grécky Panteón & Bohémia Pamäťové Axiómy**:
   ```
   [krystal_subdomain_portal mode="pantheon" height="750px"]
   ```

---

## 7. Kryptografický Mechanizmus HMAC Bearer Tokenu

Pri generovaní tokenu WordPress vypočíta štruktúrovaný JSON payload:
- `uid`: ID používateľa vo WordPress
- `usr`: Používateľské meno
- `rol`: Rola (`administrator`, `player`, `subscriber`)
- `sub`: Subdoména (`krystal.vasadomena.sk`)
- `iat`: Čas vydania (UNIX timestamp)
- `exp`: Expirácia (`iat + 60s` – prevencia zneužitia)
- `nce`: 16-znakový kryptografický jednorazový nonce
- `vhp`: Striktné pravidlo **`VITAL_MAX_HP = 6`**

Token je zakódovaný do Base64Url a podpísaný pomocou `hash_hmac('sha256', payload, secret)`. Python `WordPressSubdomainSecurityGate` token overí, skontroluje či nonce nebol v danom okne už použitý (**ochrana proti Replay Attack**) a vráti reláciu.

---

## 8. Kontrola Zdravia & Interaktívne Štúdio

Pre správcov a vývojárov je k dispozícii interaktívne štúdio:
- URL: `https://krystal.vasadomena.sk/wordpress-subdomain-security`
- Ponúka:
  - Živú mapu sieťovej topológie (Canvas animácia dátových tokov)
  - Simulátor BBQ Firewallu a testovanie SQLi/RCE vzorov
  - Generovanie a overovanie HMAC tokenov v reálnom čase
  - Priebežný záznam auditu prístupov (Audit Stream)
