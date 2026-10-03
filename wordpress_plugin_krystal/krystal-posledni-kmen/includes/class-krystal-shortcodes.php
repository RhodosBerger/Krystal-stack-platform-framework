<?php
/**
 * Registers WordPress Shortcodes to render the Krystal-Stack UI within Oxygen Builder.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

class Krystal_Shortcodes {

    public function init() {
        // [krystal_game_dashboard]
        add_shortcode( 'krystal_game_dashboard', array( $this, 'render_dashboard' ) );
        // [krystal_godot_arena]
        add_shortcode( 'krystal_godot_arena', array( $this, 'render_godot_arena' ) );
        // [krystal_gnome_duel_window]
        add_shortcode( 'krystal_gnome_duel_window', array( $this, 'render_gnome_duel_window' ) );
        // [krystal_subdomain_portal]
        add_shortcode( 'krystal_subdomain_portal', array( $this, 'render_subdomain_portal' ) );
        // [krystal_webos_desktop]
        add_shortcode( 'krystal_webos_desktop', array( $this, 'render_webos_desktop' ) );
        // [krystal_hybrid_arena]
        add_shortcode( 'krystal_hybrid_arena', array( $this, 'render_hybrid_arena' ) );
        // [krystal_svg_blueprint]
        add_shortcode( 'krystal_svg_blueprint', array( $this, 'render_svg_blueprint' ) );
    }

    public function render_dashboard( $atts ) {
        // Enqueue JS for the frontend logic
        wp_enqueue_script( 'krystal-game-js', false );
        
        $js = "
            document.addEventListener('DOMContentLoaded', function() {
                const playBtn = document.getElementById('krystal-play-btn');
                if(playBtn) {
                    playBtn.addEventListener('click', function() {
                        playBtn.innerText = 'Správcovský uzol overuje (MRP)...';
                        fetch('/wp-json/krystal/v1/play-card', { method: 'POST' })
                            .then(res => res.json())
                            .then(data => {
                                document.getElementById('krystal-terminal').innerText = '> ' + data.message;
                                playBtn.innerText = 'Útok: Krystalové Meteory (3 Mana)';
                            });
                    });
                }
            });
        ";
        wp_add_inline_script( 'krystal-game-js', $js );

        // Return the HTML that Oxygen Builder will wrap
        ob_start();
        ?>
        <div class="krystal-wrapper" style="background: var(--oxy-surface); border: 1px solid var(--oxy-accent); border-radius: 12px; padding: 30px; color: #fff; font-family: 'Inter', sans-serif;">
            <h2 style="color: var(--oxy-accent); margin-top: 0; font-family: 'Cinzel', serif;">POSLEDNÍ KMEN // KRYSTAL-STACK</h2>
            
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div style="flex: 1; padding: 15px; background: rgba(0,0,0,0.5); border-radius: 8px;">
                    <div style="font-size: 10px; color: #8b92a5;">HRÁČ 1 (KRYSTALOVÝ KMEN)</div>
                    <div style="font-size: 24px;">HP: ❤️❤️❤️❤️❤️❤️</div>
                    <div style="font-size: 18px; color: var(--oxy-accent);">MANA (LEDGER): 10</div>
                </div>
                <div style="flex: 1; padding: 15px; background: rgba(0,0,0,0.5); border-radius: 8px;">
                    <div style="font-size: 10px; color: #8b92a5;">AI PROTIVNÍK (JEDOVATÝ KMEN)</div>
                    <div style="font-size: 24px;">HP: ❤️❤️❤️❤️❤️❤️</div>
                    <div style="font-size: 18px; color: var(--tribe-toxic);">MANA (LEDGER): 10</div>
                </div>
            </div>

            <button id="krystal-play-btn" style="background: var(--oxy-accent); color: #000; border: none; padding: 12px 24px; border-radius: 4px; font-weight: bold; cursor: pointer; width: 100%; margin-bottom: 20px;">
                Zahrať kartu: Kryštálové Meteory (Cena: 3 Mana)
            </button>
            
            <div style="background: #000; color: #0f0; padding: 15px; border-radius: 4px; font-family: monospace; min-height: 80px;" id="krystal-terminal">
                > Systém pripravený. Čakám na ťah...
            </div>
        </div>
        <?php
        return ob_get_clean();
    }

    public function render_godot_arena( $atts ) {
        // Here we embed the actual Godot HTML5 / WebAssembly export. 
        // We assume the user has exported their Godot project to a folder like /wp-content/uploads/godot_arena/
        $atts = shortcode_atts( array(
            'url' => '/wp-content/uploads/godot_arena/index.html',
            'height' => '600px'
        ), $atts );

        ob_start();
        ?>
        <div class="krystal-godot-wrapper" style="width: 100%; border-radius: 12px; overflow: hidden; border: 2px solid var(--oxy-accent); box-shadow: 0 0 30px rgba(102, 252, 241, 0.2);">
            <!-- The header mirrors the HUD seen in the user's screenshots -->
            <div style="background: rgba(11, 12, 16, 0.95); padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="font-family: 'Cinzel', serif; color: var(--oxy-accent); font-weight: bold; font-size: 1.2rem;">KRYSTAL-STACK // ARENA (GODOT ENGINE)</div>
                <div style="font-family: monospace; font-size: 0.8rem; color: #8b92a5;">
                    <span style="color: #4CAF50;">● WebGL 2.0</span> | Render Target: HTML5 Canvas | FPS: 60
                </div>
            </div>
            
            <iframe 
                src="<?php echo esc_url( $atts['url'] ); ?>" 
                style="width: 100%; height: <?php echo esc_attr( $atts['height'] ); ?>; border: none; background: #000;"
                allow="autoplay; fullscreen; xr-spatial-tracking"
                title="Poslední Kmen Godot Arena">
            </iframe>
        </div>
        <?php
        return ob_get_clean();
    }

    public function render_gnome_duel_window( $atts ) {
        $atts = shortcode_atts( array(
            'duel_id' => 'room_alpha',
            'aspect_ratio' => '16:9',
            'player_name' => 'Krystal Duelist',
            'opponent_name' => 'Toxic Warmonger'
        ), $atts );

        ob_start();
        ?>
        <div class="krystal-gnome-window" data-aspect="<?php echo esc_attr( $atts['aspect_ratio'] ); ?>" style="max-width: 900px; margin: 0 auto; background: #1e1e24; border-radius: 10px; box-shadow: 0 12px 40px rgba(0,0,0,0.7); overflow: hidden; border: 1px solid rgba(255,255,255,0.1); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <!-- GNOME CSD Header Bar -->
            <div style="background: linear-gradient(180deg, #2e2e36 0%, #24242c 100%); padding: 8px 16px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.5);">
                <div style="display: flex; gap: 8px; align-items: center;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: #e05f56;"></div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: #f4be4f;"></div>
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: #57c344;"></div>
                    <span style="margin-left: 10px; font-size: 13px; font-weight: 600; color: #d0d0d8;">Adwaita Duel: <?php echo esc_html( $atts['player_name'] ); ?> vs <?php echo esc_html( $atts['opponent_name'] ); ?> [<?php echo esc_html( $atts['aspect_ratio'] ); ?>]</span>
                </div>
                <div style="font-size: 11px; color: #66fcf1; font-family: monospace;">MCP STREAM ACTIVE (BULLET-TIME READY)</div>
            </div>

            <!-- Duel Close-Up Arena Viewport -->
            <div style="position: relative; height: 420px; background: radial-gradient(circle at center, #151820 0%, #0b0c10 100%); display: flex; align-items: center; justify-content: space-between; padding: 0 40px; overflow: hidden;">
                <!-- Left Fighter: Cold Melee Weapon -->
                <div style="text-align: center; z-index: 2;">
                    <div style="font-size: 32px; filter: drop-shadow(0 0 10px #66fcf1);">⚔️</div>
                    <div style="color: #66fcf1; font-weight: bold; margin-top: 8px;"><?php echo esc_html( $atts['player_name'] ); ?></div>
                    <div style="color: #ff4757; font-size: 14px;">HP: ❤️❤️❤️❤️❤️❤️</div>
                    <div style="font-size: 11px; color: #a4b0be;">Kryštálová Čepeľ (Cold Weapon)</div>
                </div>

                <!-- Center: Bullet Time Collision Zone & Mortar Plunging Arc -->
                <div style="text-align: center; border: 1px dashed rgba(102, 252, 241, 0.3); border-radius: 50%; width: 180px; height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.3);">
                    <div style="font-size: 24px; animation: pulse 1.5s infinite;">💣</div>
                    <div style="font-size: 11px; color: #ff9f43; font-weight: bold; margin-top: 4px;">PLUNGING MORTAR</div>
                    <div style="font-size: 9px; color: #8395a7;">Cadence: 0.40s | CEP: 1.2m</div>
                    <div style="font-size: 9px; color: #57c344; margin-top: 4px;">UBISOFT BULLET TIME</div>
                </div>

                <!-- Right Fighter: Cold Melee Flail / Opponent Viewport -->
                <div style="text-align: center; z-index: 2;">
                    <div style="font-size: 32px; filter: drop-shadow(0 0 10px #8a2be2);">🛡️</div>
                    <div style="color: #c56cf0; font-weight: bold; margin-top: 8px;"><?php echo esc_html( $atts['opponent_name'] ); ?></div>
                    <div style="color: #ff4757; font-size: 14px;">HP: ❤️❤️❤️❤️❤️🖤</div>
                    <div style="font-size: 11px; color: #a4b0be;">Toxický Censer Flail</div>
                </div>
            </div>
        </div>
        <?php
        return ob_get_clean();
    }

    /**
     * Renders secure Krystal Subdomain Portal with HMAC session authentication
     */
    public function render_subdomain_portal( $atts ) {
        $atts = shortcode_atts( array(
            'mode' => 'hybrid', // hybrid | desktop | arena | blueprint | pantheon
            'height' => '780px',
            'require_login' => 'true'
        ), $atts );

        $is_logged_in = is_user_logged_in();
        $require_login = filter_var( $atts['require_login'], FILTER_VALIDATE_BOOLEAN );

        if ( $require_login && ! $is_logged_in ) {
            $login_url = wp_login_url( get_permalink() );
            ob_start();
            ?>
            <div class="krystal-subdomain-gate" style="background: #06090e; border: 1px solid #d8ae4b; border-radius: 2px; padding: 40px 24px; color: #fff; font-family: 'Cinzel', serif; text-align: center; max-width: 680px; margin: 30px auto; box-shadow: 0 4px 24px rgba(0,0,0,0.8);">
                <div style="font-size: 32px; margin-bottom: 12px;">🏛️</div>
                <div style="font-size: 11px; letter-spacing: 0.2em; color: #d8ae4b; text-transform: uppercase; margin-bottom: 8px;">KRYSTAL-STACK // SUBDOMAIN SECURITY GATE</div>
                <h3 style="color: #f7f9fa; margin: 0 0 16px 0; font-size: 20px;">PRÍSTUP VYŽADUJE OVERENÝ ÚČET</h3>
                <p style="font-family: 'Inter', sans-serif; font-size: 13px; color: #8e9cae; line-height: 1.6; margin-bottom: 24px;">
                    Tento procedurálny herný a výpočtový priestor je chránený kryptografickým HMAC tokenom. Pre prístup k simulácii a 3D aréne sa prosím prihláste do svojho účtu na subdoméne.
                </p>
                <div style="display: flex; justify-content: center; gap: 16px;">
                    <a href="<?php echo esc_url( $login_url ); ?>" style="background: #d8ae4b; color: #06090e; text-decoration: none; padding: 10px 24px; font-weight: bold; font-size: 13px; letter-spacing: 0.1em; border-radius: 2px; text-transform: uppercase;">
                        Prihlásiť sa cez WordPress
                    </a>
                </div>
            </div>
            <?php
            return ob_get_clean();
        }

        // User is authenticated: generate HMAC signed bearer token
        $user = wp_get_current_user();
        $roles = (array) $user->roles;
        $role = ! empty( $roles ) ? $roles[0] : 'subscriber';
        $token = Krystal_API_Connector::create_subdomain_token( $user->ID, $user->user_login, $role );

        // Determine target path based on mode
        $engine_base = get_option( 'krystal_engine_subdomain_public_url', '' );
        if ( empty( $engine_base ) ) {
            $engine_base = '/krystal-core'; // Proxied by Nginx
        }

        $mode_map = array(
            'hybrid' => '/hybrid-portal-arena',
            'desktop' => '/desktop',
            'arena' => '/game',
            'blueprint' => '/evolved-svg-studio',
            'pantheon' => '/static/greek_bohemia_pantheon_studio.html',
            'terrain' => '/terrain-synthesis'
        );

        $rel_path = isset( $mode_map[ $atts['mode'] ] ) ? $mode_map[ $atts['mode'] ] : '/hybrid-portal-arena';
        $target_url = rtrim( $engine_base, '/' ) . $rel_path . '?token=' . urlencode( $token );

        ob_start();
        ?>
        <div class="krystal-subdomain-container" style="width: 100%; border: 1px solid #1f2d3d; border-radius: 2px; overflow: hidden; background: #06090e; margin: 20px 0;">
            <!-- Secure HUD Bar -->
            <div style="background: #090e15; border-bottom: 1px solid #1f2d3d; padding: 8px 16px; display: flex; justify-content: space-between; align-items: center; font-family: monospace; font-size: 11px;">
                <div style="color: #d8ae4b; display: flex; align-items: center; gap: 8px;">
                    <span>🛡️ SUBDOMAIN SECURED</span>
                    <span style="color: #58d8ff;">[<?php echo esc_html( strtoupper( $atts['mode'] ) ); ?>]</span>
                </div>
                <div style="color: #8e9cae; display: flex; gap: 16px;">
                    <span>USER: <b style="color: #f7f9fa;"><?php echo esc_html( $user->user_login ); ?></b> (<?php echo esc_html( $role ); ?>)</span>
                    <span style="color: #4ade80;">● HMAC VALIDATED</span>
                    <span style="color: #f87171;">VITAL HP: 6</span>
                </div>
            </div>
            <!-- Embedded Viewport -->
            <iframe 
                src="<?php echo esc_url( $target_url ); ?>" 
                style="width: 100%; height: <?php echo esc_attr( $atts['height'] ); ?>; border: none; background: #06090e;"
                allow="autoplay; fullscreen; xr-spatial-tracking"
                title="Krystal-Stack Subdomain Portal">
            </iframe>
        </div>
        <?php
        return ob_get_clean();
    }

    public function render_webos_desktop( $atts ) {
        $atts = is_array( $atts ) ? $atts : array();
        $atts['mode'] = 'desktop';
        return $this->render_subdomain_portal( $atts );
    }

    public function render_hybrid_arena( $atts ) {
        $atts = is_array( $atts ) ? $atts : array();
        $atts['mode'] = 'hybrid';
        return $this->render_subdomain_portal( $atts );
    }

    public function render_svg_blueprint( $atts ) {
        $atts = is_array( $atts ) ? $atts : array();
        $atts['mode'] = 'blueprint';
        return $this->render_subdomain_portal( $atts );
    }
}
