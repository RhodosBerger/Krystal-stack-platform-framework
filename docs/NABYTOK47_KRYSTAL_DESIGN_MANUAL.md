# KRYSTAL-STACK × NÁBYTOK 47 — DESIGN MANUAL, ACF GRID LOGIC & GODOT UI BRIDGE

## 1. Non-Indexed Subdomain & Subfolder Topology (Team.blue / WebSupport)
- **Primary Brand Site (Indexed)**: `https://nabytok47.sk`
- **Subfolder Sandbox (Strictly `noindex, nofollow`)**: `https://nabytok47.sk/krystal-hub/`
- **Subdomain Bridge Alias (Strictly `noindex, nofollow`)**: `https://krystal.nabytok47.sk` → maps via WebSupport DNS CNAME + OpenResty/Apache rewrite to `/krystal-hub/` while injecting `X-Robots-Tag: noindex, nofollow, noarchive, nosnippet` and SEOPress `seopress_titles_noindex = true`.

```apache
# WebSupport (.htaccess) Subdomain -> Subfolder + Non-Indexed Header
<IfModule mod_headers.c>
    SetEnvIf Request_URI "^/krystal-hub" KRYSTAL_NOINDEX=1
    SetEnvIf Host "^krystal\.nabytok47\.sk$" KRYSTAL_NOINDEX=1
    Header set X-Robots-Tag "noindex, nofollow, noarchive, nosnippet" env=KRYSTAL_NOINDEX
</IfModule>
```

---

## 2. Visual Identity & Stylist Token System (`nbsr-` Oxygen Prefix)
Unifies **Nábytok 47** artisanal bespoke joinery aesthetics with **Krystal-Stack V-Ray Optic Telemetry**:

| Token | CSS Variable | Value | Role Across Oxygen 6 & Godot 4.x UI |
| :--- | :--- | :--- | :--- |
| **Obsidian Workshop** | `--nbsr-krystal-bg` | `#0E1116` | Deep V-Ray raymarch viewport canvas |
| **Slate Grid Surface** | `--nbsr-krystal-surface` | `#171C24` | 8×8 KMB1 mechanical grid cells & ACF option cards |
| **Artisanal Warm Oak** | `--nbsr-krystal-oak` | `#C89B6E` | Nábytok 47 Egger/Kronospan woodgrain accent & Mariáš Trump trim |
| **Caustic Optic Cyan** | `--nbsr-krystal-cyan` | `#38BDF8` | Active V-Ray radiance & bitboard trajectory vector |
| **Heading Typography** | `--nbsr-font-display` | `'Montserrat', sans-serif` | Geometric precision headers & HUD telemetry |
| **Narrative Lore Serif** | `--nbsr-font-lore` | `'Lora', serif` | Fantasy LLM quest prose & Mariáš codex |
| **Artisan Signature** | `--nbsr-font-accent` | `'Caveat', cursive` | Handcrafted blueprint annotations |

---

## 3. ACF-Equivalent Grid Logic <-> Oxygen 6 <-> Godot 4.x Mapping

| ACF-Style Meta Key / Term | Type | Oxygen Visual Builder Binding | Godot 4.x (`KrystalHybridLoreAggregator.gd`) Consumer |
| :--- | :--- | :--- | :--- |
| `krystal_grid_bitboard_hex` | `uint64 hex` | Renders active cells on the 8×8 CSS Grid (`0x8100000800000080`) | Syncs `bitboard_state` occupancy bitmask |
| `krystal_kmb1_binary_hex` | `24B hex` | Displays CRC32-verified mechanical payload badge | Decodes/encodes `StreamPeerBuffer` `KMB1` packet |
| `krystal_vray_active_lens` | `enum term` | Switches stylist theme (`SOLAR_CLARITY`, `CAUSTIC_SUSPICION`, `UMBRAL_NOIR_MORTAR`) | Updates Forward+ viewport irradiance & shader pass |
| `krystal_marias_trump_suit` | `enum term` | Highlights 40-pt Hláška Marriage cards & pending click queue | Emits `marias_semantic_animation_triggered` Bezier arc |
| `krystal_nabytok_decor_sku` | `SKU string` | Links Egger/Kronospan physical catalog (`EGGER-H3170-OAK`) | Binds PBR roughness/albedo to 3D furniture/citadel assets |
| `krystal_vital_max_hp` | `const int` | System health invariant badge (`6`) | Enforces `VITAL_MAX_HP = 6` |

---

## 4. Custom Stylist Query Preset (Oxygen Loops & WP_Query)
```php
// Custom Oxygen / WP Stylist Query for Non-Indexed Hybrid Grid Nodes
$krystal_stylist_query = [
    'post_type'      => 'krystal_grid_node',
    'posts_per_page' => 64,
    'meta_key'       => 'krystal_vray_radiance',
    'orderby'        => 'meta_value_num',
    'order'          => 'DESC',
    'tax_query'      => [[
        'taxonomy' => 'krystal_acf_grid_term',
        'field'    => 'slug',
        'terms'    => ['active-bitboard-cell', 'marias-pending-action'],
    ]],
];
```
