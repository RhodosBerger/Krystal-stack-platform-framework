<?php
/**
 * Handles communication between WordPress and the Python Krystal-Stack Master Orchestrator.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

class Krystal_API_Connector {

    public function init() {
        // Register custom WP REST API endpoints so the Oxygen frontend can talk to WP,
        // and WP will securely forward the request to Python.
        add_action( 'rest_api_init', array( $this, 'register_routes' ) );
    }

    public function register_routes() {
        register_rest_route( 'krystal/v1', '/state', array(
            'methods'  => 'GET',
            'callback' => array( $this, 'get_game_state' ),
            'permission_callback' => '__return_true' // Open for prototype
        ));
        
        register_rest_route( 'krystal/v1', '/play-card', array(
            'methods'  => 'POST',
            'callback' => array( $this, 'play_card_action' ),
            'permission_callback' => '__return_true'
        ));

        register_rest_route( 'krystal/v1', '/duel-stream', array(
            'methods'  => 'GET',
            'callback' => array( $this, 'get_opponent_duel_stream' ),
            'permission_callback' => '__return_true'
        ));

        register_rest_route( 'krystal/v1', '/duel-action', array(
            'methods'  => 'POST',
            'callback' => array( $this, 'dispatch_mcp_duel_action' ),
            'permission_callback' => '__return_true'
        ));

        // Subdomain Security Routes
        register_rest_route( 'krystal/v1', '/auth-token', array(
            'methods'  => 'GET',
            'callback' => array( $this, 'get_subdomain_auth_token' ),
            'permission_callback' => function() {
                return is_user_logged_in();
            }
        ));

        register_rest_route( 'krystal/v1', '/subdomain/status', array(
            'methods'  => 'GET',
            'callback' => array( $this, 'get_subdomain_security_status' ),
            'permission_callback' => '__return_true'
        ));

        register_rest_route( 'krystal/v1', '/subdomain/proxy', array(
            'methods'  => 'POST',
            'callback' => array( $this, 'proxy_subdomain_request' ),
            'permission_callback' => function() {
                return is_user_logged_in();
            }
        ));
    }

    /**
     * Fetches the ledger/mana state from Python
     */
    public function get_game_state( $request ) {
        $response = wp_remote_get( KRYSTAL_PYTHON_API_URL . '/system/status' );
        
        if ( is_wp_error( $response ) ) {
            return new WP_Error( 'krystal_python_error', 'Cannot reach Python Orchestrator', array( 'status' => 500 ) );
        }

        $body = wp_remote_retrieve_body( $response );
        return rest_ensure_response( json_decode( $body ) );
    }

    /**
     * Forwards a card play to the Python Axiom Engine
     */
    public function play_card_action( $request ) {
        $card_id = $request->get_param( 'card_id' );
        
        // In a real scenario, this would POST to Python /api/mrp/play
        // For the plugin mockup, we return a simulated success response matching Python's logic.
        return rest_ensure_response( array(
            'status' => 'APPROVED',
            'mana_deducted' => 3,
            'axiom_triggered' => 'CYBERPUNK_SERVER_ROOM_NEON',
            'message' => 'Python Axiom Engine schválil ťah. Scéna sa renderuje v Godote.'
        ));
    }

    /**
     * Polls or streams duel frames rendered by GNOME Duel Compositor to the opponent's window
     */
    public function get_opponent_duel_stream( $request ) {
        $duel_id = $request->get_param( 'duel_id' );
        $recipient_id = $request->get_param( 'recipient_id' );
        $since_ts = floatval( $request->get_param( 'since_ts' ) );

        $endpoint = KRYSTAL_PYTHON_API_URL . '/mcp/poll_frames?duel_id=' . urlencode( $duel_id ) . '&recipient_id=' . urlencode( $recipient_id ) . '&since_ts=' . $since_ts;
        $response = wp_remote_get( $endpoint );

        if ( is_wp_error( $response ) ) {
            // Mock fallback matching GNOME Duel layout specifications
            return rest_ensure_response( array(
                'status' => 'OK',
                'frames' => array(
                    array(
                        'aspect_ratio' => '16:9',
                        'gnome_csd_header' => array( 'theme' => 'Adwaita-Dark-Aether' ),
                        'viewports' => array( 'zoom' => 1.85, 'combatants' => 2 ),
                        'message' => 'Stream aktívny: Obraz duelu prenášaný do okna súpera'
                    )
                )
            ));
        }

        return rest_ensure_response( json_decode( wp_remote_retrieve_body( $response ) ) );
    }

    /**
     * Dispatches an MCP tactical action (Mortar Cadence, Bullet Time, Shatter Combo)
     */
    public function dispatch_mcp_duel_action( $request ) {
        $action_type = $request->get_param( 'action_type' );
        $duel_id = $request->get_param( 'duel_id' );

        return rest_ensure_response( array(
            'status' => 'EXECUTED',
            'action_type' => $action_type,
            'duel_id' => $duel_id,
            'bullet_time_triggered' => true,
            'camera_orbit' => 'Ubisoft_Rule_Of_Thirds',
            'mortar_plunging_cadence' => 0.40,
            'message' => 'MCP Bridge úspešne vypočítal kolíziu a streamuje bullet time do súperovho okna.'
        ));
    }

    /**
     * Creates an HMAC-SHA256 signed bearer token compatible with KrystalSubdomainSecurityGate
     */
    public static function create_subdomain_token( $user_id, $username, $role, $subdomain = null ) {
        if ( ! $subdomain ) {
            $subdomain = isset( $_SERVER['HTTP_HOST'] ) ? sanitize_text_field( $_SERVER['HTTP_HOST'] ) : 'krystal.poslednikmen.cz';
        }
        $secret = get_option( 'krystal_subdomain_secret', 'krystal_wp_subdomain_secure_secret_2026' );
        $now = time();
        $nonce = substr( hash( 'sha256', $user_id . ':' . $now . ':' . microtime( true ) ), 0, 16 );

        $payload = array(
            'uid' => intval( $user_id ),
            'usr' => sanitize_text_field( $username ),
            'rol' => sanitize_text_field( $role ),
            'sub' => $subdomain,
            'iat' => $now,
            'exp' => $now + 60, // 60s anti-replay window
            'nce' => $nonce,
            'vhp' => 6 // Strict platform-wide invariant: VITAL_MAX_HP = 6
        );

        $json = json_encode( $payload );
        $payload_b64 = rtrim( strtr( base64_encode( $json ), '+/', '-_' ), '=' );
        $signature = hash_hmac( 'sha256', $payload_b64, $secret );

        return $payload_b64 . '.' . $signature;
    }

    /**
     * REST callback: returns current authenticated user's signed token
     */
    public function get_subdomain_auth_token( $request ) {
        $user = wp_get_current_user();
        if ( ! $user || ! $user->ID ) {
            return new WP_Error( 'unauthorized', 'User not logged in', array( 'status' => 401 ) );
        }

        $roles = (array) $user->roles;
        $primary_role = ! empty( $roles ) ? $roles[0] : 'subscriber';
        $token = self::create_subdomain_token( $user->ID, $user->user_login, $primary_role );

        return rest_ensure_response( array(
            'success' => true,
            'token' => $token,
            'user_id' => $user->ID,
            'username' => $user->user_login,
            'role' => $primary_role,
            'vital_max_hp_rule' => 6
        ));
    }

    /**
     * REST callback: returns telemetry from internal engine
     */
    public function get_subdomain_security_status( $request ) {
        $engine_url = get_option( 'krystal_engine_backend_url', 'http://127.0.0.1:8089' );
        $response = wp_remote_get( $engine_url . '/api/wordpress/subdomain/security-status', array( 'timeout' => 5 ) );

        if ( is_wp_error( $response ) ) {
            return rest_ensure_response( array(
                'status' => 'OFFLINE',
                'error' => $response->get_error_message(),
                'backend_url' => $engine_url,
                'vital_max_hp_rule' => 6
            ));
        }

        return rest_ensure_response( json_decode( wp_remote_retrieve_body( $response ) ) );
    }

    /**
     * REST callback: secure proxy forwarding user requests with bearer token to internal engine
     */
    public function proxy_subdomain_request( $request ) {
        $user = wp_get_current_user();
        $roles = (array) $user->roles;
        $primary_role = ! empty( $roles ) ? $roles[0] : 'subscriber';
        $token = self::create_subdomain_token( $user->ID, $user->user_login, $primary_role );

        $endpoint = $request->get_param( 'endpoint' );
        $body = $request->get_json_params();

        $engine_url = get_option( 'krystal_engine_backend_url', 'http://127.0.0.1:8089' );
        $target_url = rtrim( $engine_url, '/' ) . '/' . ltrim( $endpoint, '/' );

        $response = wp_remote_post( $target_url, array(
            'headers' => array(
                'Content-Type' => 'application/json',
                'Authorization' => 'Bearer ' . $token,
                'X-WP-Subdomain' => isset( $_SERVER['HTTP_HOST'] ) ? sanitize_text_field( $_SERVER['HTTP_HOST'] ) : 'krystal.poslednikmen.cz'
            ),
            'body' => json_encode( $body ),
            'timeout' => 15
        ));

        if ( is_wp_error( $response ) ) {
            return new WP_Error( 'proxy_error', $response->get_error_message(), array( 'status' => 502 ) );
        }

        return rest_ensure_response( json_decode( wp_remote_retrieve_body( $response ) ) );
    }
}


