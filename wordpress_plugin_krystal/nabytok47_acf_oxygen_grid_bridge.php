<?php
/**
 * Plugin Name: Krystal-Stack x Nábytok 47 — ACF-Equivalent Grid & Oxygen Bridge (NoIndex Subsite)
 * Description: Bridges Godot 4.x 8x8 KMB1 Mechanical Bitboard binaries, V-Ray Algebraic Signals,
 *              and Mariáš turn states into native WordPress/Oxygen 6 Dynamic Fields (ACF-style terms)
 *              for Team.blue / WebSupport hosting, while strictly blocking SEO indexing (noindex/nofollow)
 *              on `/krystal-hub/*` subfolder and `krystal.nabytok47.sk` subdomain routes.
 * Version: 1.0.0
 * Author: Dušan Kopecký & Krystal-Stack Architecture Council
 */

if (!defined('ABSPATH')) {
    exit;
}

define('KRYSTAL_VITAL_MAX_HP', 6);

/**
 * 1. STRICT NON-INDEXED GUARD (Subdomain `krystal.nabytok47.sk` & Subfolder `/krystal-hub/`)
 * Prevents Google / Bing / SEOPress sitemap & crawler indexing.
 */
add_action('template_redirect', function () {
    $host = isset($_SERVER['HTTP_HOST']) ? strtolower(sanitize_text_field($_SERVER['HTTP_HOST'])) : '';
    $uri  = isset($_SERVER['REQUEST_URI']) ? strtolower(sanitize_text_field($_SERVER['REQUEST_URI'])) : '';

    if (strpos($host, 'krystal.') === 0 || strpos($uri, '/krystal-hub') === 0) {
        header('X-Robots-Tag: noindex, nofollow, noarchive, nosnippet', true);
        add_filter('wp_robots', function (array $robots): array {
            $robots['noindex']   = true;
            $robots['nofollow']  = true;
            $robots['noarchive'] = true;
            return $robots;
        }, 999);
        // SEOPress compatibility hook on nabytok47.sk
        add_filter('seopress_titles_noindex', '__return_true', 999);
        add_filter('seopress_titles_nofollow', '__return_true', 999);
    }
});

/**
 * 2. ACF-EQUIVALENT CUSTOM POST TYPE, TAXONOMY TERMS & META SCHEMA FOR OXYGEN DYNAMIC DATA
 * Works zero-plugin-overhead alongside Oxygen 6 `bd-bind` / Dynamic Data without requiring ACF Pro.
 */
add_action('init', function () {
    register_post_type('krystal_grid_node', [
        'labels'              => ['name' => 'Krystal Grid Nodes', 'singular_name' => 'Krystal Grid Node'],
        'public'              => true,
        'publicly_queryable'  => true,
        'exclude_from_search' => true,
        'show_in_rest'        => true,
        'supports'            => ['title', 'editor', 'custom-fields', 'page-attributes'],
        'rewrite'             => ['slug' => 'krystal-hub/grid-node', 'with_front' => false],
    ]);

    register_taxonomy('krystal_acf_grid_term', ['krystal_grid_node', 'page'], [
        'labels'       => ['name' => 'ACF Grid Logic Terms', 'singular_name' => 'ACF Grid Term'],
        'public'       => false,
        'show_ui'      => true,
        'show_in_rest' => true,
        'hierarchical' => true,
    ]);

    // RH Isolation Slovakia — Custom Service & Thermal/Acoustic Material Taxonomies
    register_taxonomy('rh_isolation_taxonomy', ['krystal_grid_node', 'page'], [
        'labels'       => ['name' => 'RH Isolation Taxonomies', 'singular_name' => 'RH Isolation Taxonomy'],
        'public'       => false,
        'show_ui'      => true,
        'show_in_rest' => true,
        'hierarchical' => true,
    ]);

    $acf_equivalent_fields = [
        'krystal_grid_bitboard_hex' => 'string',  // e.g. 0x8100000800000080
        'krystal_kmb1_binary_hex'   => 'string',  // 24-byte CRC32-verified mechanical frame
        'krystal_vray_active_lens'  => 'string',  // SOLAR_CLARITY | CAUSTIC_SUSPICION | UMBRAL_NOIR_MORTAR
        'krystal_vray_radiance'     => 'number',  // Algebraic light-transport / thermal flux output
        'krystal_marias_trump_suit' => 'string',  // HEARTS_SRDCIA (40-pt marriage)
        'krystal_nabytok_decor_sku' => 'string',  // Bridges Egger/Kronospan material SKU <-> 3D Godot shader
        'rh_isolation_u_value_wm2k' => 'number',  // Thermal transmittance U-value (W/m²K) for RH Isolation Slovakia
        'rh_isolation_r_resistance' => 'number',  // Thermal resistance R (m²K/W)
        'rh_isolation_acoustic_db'  => 'integer', // Acoustic attenuation rating (dB)
        'krystal_vital_max_hp'      => 'integer', // Invariant = 6
    ];

    foreach ($acf_equivalent_fields as $meta_key => $type) {
        register_post_meta('', $meta_key, [
            'show_in_rest'      => true,
            'single'            => true,
            'type'              => $type,
            'sanitize_callback' => 'sanitize_text_field',
            'auth_callback'     => function () {
                return current_user_can('edit_posts');
            },
        ]);
    }
});

/**
 * 3. ACF-COMPATIBLE HELPER FUNCTIONS FOR OXYGEN PHP / DYNAMIC QUERIES
 */
function krystal_get_acf_grid_field(string $field_name, int $post_id = 0) {
    $pid = $post_id ?: get_the_ID();
    if (function_exists('get_field')) {
        $val = get_field($field_name, $pid);
        if ($val !== null && $val !== false && $val !== '') {
            return $val;
        }
    }
    return get_post_meta($pid, $field_name, true);
}

/**
 * 4. GODOT 4.X <-> NÁBYTOK 47 / OXYGEN REST BRIDGE (`/wp-json/krystal/v1/grid-sync`)
 */
add_action('rest_api_init', function () {
    register_rest_route('krystal/v1', '/grid-sync/(?P<id>\d+)', [
        [
            'methods'             => 'GET',
            'permission_callback' => '__return_true',
            'callback'            => function (WP_REST_Request $req) {
                $pid = (int) $req['id'];
                return rest_ensure_response([
                    'status'              => 'OK',
                    'post_id'             => $pid,
                    'vital_max_hp'        => KRYSTAL_VITAL_MAX_HP,
                    'seo_indexed'         => false,
                    'bitboard_uint64_hex' => get_post_meta($pid, 'krystal_grid_bitboard_hex', true) ?: '0x8100000000000081',
                    'kmb1_binary_hex'     => get_post_meta($pid, 'krystal_kmb1_binary_hex', true) ?: 'KMB1_READY',
                    'vray_active_lens'    => get_post_meta($pid, 'krystal_vray_active_lens', true) ?: 'SOLAR_CLARITY',
                    'marias_trump_suit'   => get_post_meta($pid, 'krystal_marias_trump_suit', true) ?: 'HEARTS_SRDCIA',
                    'nabytok_decor_sku'   => get_post_meta($pid, 'krystal_nabytok_decor_sku', true) ?: 'EGGER-H3170-OAK',
                ]);
            },
        ],
    ]);
});
