<?php
/**
 * Admin Settings Page for Krystal-Stack Subdomain Security & Engine Connectivity
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

class Krystal_Admin_Settings {

    public function init() {
        add_action( 'admin_menu', array( $this, 'register_admin_menu' ) );
        add_action( 'admin_init', array( $this, 'register_settings' ) );
    }

    public function register_admin_menu() {
        add_menu_page(
            'Krystal Subdoména & Bezpečnosť',
            'Krystal Engine',
            'manage_options',
            'krystal-subdomain-settings',
            array( $this, 'render_settings_page' ),
            'dashicons-shield',
            65
        );
    }

    public function register_settings() {
        register_setting( 'krystal_subdomain_group', 'krystal_subdomain_host' );
        register_setting( 'krystal_subdomain_group', 'krystal_engine_backend_url' );
        register_setting( 'krystal_subdomain_group', 'krystal_engine_subdomain_public_url' );
        register_setting( 'krystal_subdomain_group', 'krystal_subdomain_secret' );
        register_setting( 'krystal_subdomain_group', 'krystal_token_ttl_seconds' );
    }

    public function render_settings_page() {
        if ( ! current_user_can( 'manage_options' ) ) {
            return;
        }

        $host = get_option( 'krystal_subdomain_host', 'krystal.poslednikmen.cz' );
        $backend_url = get_option( 'krystal_engine_backend_url', 'http://127.0.0.1:8089' );
        $public_url = get_option( 'krystal_engine_subdomain_public_url', '/krystal-core' );
        $secret = get_option( 'krystal_subdomain_secret', 'krystal_wp_subdomain_secure_secret_2026' );
        $ttl = get_option( 'krystal_token_ttl_seconds', '60' );

        // Test connectivity to Python engine
        $status_response = wp_remote_get( rtrim( $backend_url, '/' ) . '/api/wordpress/subdomain/security-status', array( 'timeout' => 3 ) );
        $is_online = ! is_wp_error( $status_response ) && wp_remote_retrieve_response_code( $status_response ) === 200;
        ?>
        <div class="wrap" style="max-width: 900px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;">
            <h1 style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                <span style="color: #d8ae4b;">🏛️</span> Krystal-Stack: Zabezpečenie Subdomény a Reverzné Proxy
            </h1>

            <!-- Health Status Banner -->
            <div style="background: #0d131f; border-left: 4px solid <?php echo $is_online ? '#4ade80' : '#f87171'; ?>; padding: 16px 20px; border-radius: 4px; color: #fff; margin-bottom: 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #8e9cae;">Stav Spojenia s Kernelom (Port 8089)</div>
                        <div style="font-size: 18px; font-weight: bold; margin-top: 4px; color: <?php echo $is_online ? '#4ade80' : '#f87171'; ?>;">
                            <?php echo $is_online ? '● KRYSTAL ENGINE KERNEL: ONLINE' : '○ KRYSTAL ENGINE KERNEL: NEDOSTUPNÝ'; ?>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: rgba(216, 174, 75, 0.15); border: 1px solid #d8ae4b; color: #d8ae4b; font-family: monospace; font-size: 11px; padding: 4px 10px; border-radius: 2px;">
                            VITAL MAX HP: 6
                        </span>
                    </div>
                </div>
            </div>

            <!-- Architecture Flow Representation -->
            <div style="background: #06090e; border: 1px solid #1f2d3d; border-radius: 4px; padding: 20px; color: #8e9cae; font-family: monospace; font-size: 11px; line-height: 1.5; margin-bottom: 24px; overflow-x: auto;">
                <div style="color: #d8ae4b; font-weight: bold; margin-bottom: 8px;">// REVERZNÁ ARCHITEKTÚRA SUBDOMÉNY</div>
                <div>[ Prehliadač ] ──HTTPS:443──► [ Nginx Proxy (Subdoména) ]</div>
                <div style="margin-left: 175px;">├── BBQ Firewall (SQLi, Traversal, RCE, XSS Blocked)</div>
                <div style="margin-left: 175px;">├── Wordfence Rate Limiting (/wp-login.php)</div>
                <div style="margin-left: 175px;">├── / ──────► WordPress CMS na Subdoméne (HMAC Token Auth)</div>
                <div style="margin-left: 175px;">└── /krystal-core/ ─► Krystal Engine Core (Localhost:8089)</div>
            </div>

            <form method="post" action="options.php">
                <?php settings_fields( 'krystal_subdomain_group' ); ?>
                <?php do_settings_sections( 'krystal_subdomain_group' ); ?>

                <table class="form-table" role="presentation">
                    <tr>
                        <th scope="row"><label for="krystal_subdomain_host">Názov Subdomény</label></th>
                        <td>
                            <input name="krystal_subdomain_host" type="text" id="krystal_subdomain_host" value="<?php echo esc_attr( $host ); ?>" class="regular-text" />
                            <p class="description">Napr. <code>krystal.poslednikmen.cz</code> alebo <code>krystal.vasadomena.sk</code>.</p>
                        </td>
                    </tr>
                    <tr>
                        <th scope="row"><label for="krystal_engine_backend_url">Interná URL Python Engine</label></th>
                        <td>
                            <input name="krystal_engine_backend_url" type="text" id="krystal_engine_backend_url" value="<?php echo esc_attr( $backend_url ); ?>" class="regular-text" />
                            <p class="description">Privátna adresa kernelu (typicky <code>http://127.0.0.1:8089</code>).</p>
                        </td>
                    </tr>
                    <tr>
                        <th scope="row"><label for="krystal_engine_subdomain_public_url">Verejná Reverzná Cesta (Proxy Path)</label></th>
                        <td>
                            <input name="krystal_engine_subdomain_public_url" type="text" id="krystal_engine_subdomain_public_url" value="<?php echo esc_attr( $public_url ); ?>" class="regular-text" />
                            <p class="description">Verejný prefix smerovaný Nginxom (odporúčané <code>/krystal-core</code>).</p>
                        </td>
                    </tr>
                    <tr>
                        <th scope="row"><label for="krystal_subdomain_secret">HMAC Zdieľaný Kľúč (Secret)</label></th>
                        <td>
                            <input name="krystal_subdomain_secret" type="password" id="krystal_subdomain_secret" value="<?php echo esc_attr( $secret ); ?>" class="regular-text" />
                            <p class="description">Používaný pre podpisovanie jednorazových Bearer tokenov používateľov.</p>
                        </td>
                    </tr>
                    <tr>
                        <th scope="row"><label for="krystal_token_ttl_seconds">Platnosť Tokenu (TTL v sekundách)</label></th>
                        <td>
                            <input name="krystal_token_ttl_seconds" type="number" id="krystal_token_ttl_seconds" value="<?php echo esc_attr( $ttl ); ?>" class="small-text" min="10" max="300" />
                            <p class="description">Časové okno pre prevenciu Replay útokov (predvolené 60 sekúnd).</p>
                        </td>
                    </tr>
                </table>

                <?php submit_button( 'Uložiť Nastavenia Subdomény' ); ?>
            </form>

            <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;" />

            <h3>Rýchle Shortcody pre Stránky na Subdoméne:</h3>
            <p>Vložte tieto značky do vizuálneho editora (Oxygen Builder, Elementor, Gutenberg):</p>
            <ul style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 16px 24px; border-radius: 4px; font-family: monospace;">
                <li style="margin-bottom: 8px;"><code>[krystal_subdomain_portal mode="hybrid" height="780px"]</code> — Plný zabezpečený portál s login bránou</li>
                <li style="margin-bottom: 8px;"><code>[krystal_webos_desktop height="800px"]</code> — Multitaskingový WebOS Window Manager</li>
                <li style="margin-bottom: 8px;"><code>[krystal_subdomain_portal mode="pantheon" height="750px"]</code> — Grécky Panteón & Bohémia</li>
                <li><code>[krystal_svg_blueprint height="700px"]</code> — Vektorové SVG Blueprint štúdio</li>
            </ul>
        </div>
        <?php
    }
}
