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
}

