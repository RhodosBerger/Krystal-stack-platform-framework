<?php
/**
 * Plugin Name: Krystal-Stack: Poslední Kmen & ERP Engine
 * Plugin URI: https://krystal-stack.com
 * Description: Integrates the Krystal-Stack Axiomatic Engine (ERP, Accounting, AI Bot) with WordPress and Oxygen Builder to power the "Poslední Kmen" game state and SaaS dashboards.
 * Version: 1.0.0
 * Author: Krystal Architecture Team
 * Author URI: https://krystal-stack.com
 * Text Domain: krystal-posledni-kmen
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit; // Exit if accessed directly
}

define( 'KRYSTAL_PLUGIN_DIR', plugin_dir_path( __FILE__ ) );
define( 'KRYSTAL_PYTHON_API_URL', 'http://localhost:8085/api' ); // Connects to master_erp_orchestrator.py

// 1. Include Core Classes
require_once KRYSTAL_PLUGIN_DIR . 'includes/class-krystal-api.php';
require_once KRYSTAL_PLUGIN_DIR . 'includes/class-krystal-shortcodes.php';
require_once KRYSTAL_PLUGIN_DIR . 'includes/class-krystal-admin.php';

// 2. Initialize Plugin
function krystal_posledni_kmen_init() {
    $krystal_api = new Krystal_API_Connector();
    $krystal_api->init();

    $krystal_shortcodes = new Krystal_Shortcodes();
    $krystal_shortcodes->init();

    $krystal_admin = new Krystal_Admin_Settings();
    $krystal_admin->init();
}
add_action( 'plugins_loaded', 'krystal_posledni_kmen_init' );

// 3. Oxygen Builder Integration (Registering Custom Elements/CSS)
function krystal_oxygen_integration() {
    if ( class_exists( 'CT_Component' ) ) {
        // Enqueue our specific dark-theme CSS variables if Oxygen is active
        add_action( 'wp_enqueue_scripts', function() {
            wp_register_style( 'krystal-oxygen-theme', false );
            wp_enqueue_style( 'krystal-oxygen-theme' );
            $css_vars = "
                :root {
                    --oxy-bg-base: #0b0c10;
                    --oxy-surface: rgba(18, 22, 28, 0.7);
                    --oxy-accent: #66fcf1;
                    --tribe-crystal: #66fcf1;
                    --tribe-toxic: #8a2be2;
                    --tribe-druid: #b8860b;
                }
            ";
            wp_add_inline_style( 'krystal-oxygen-theme', $css_vars );
        });
    }
}
add_action( 'init', 'krystal_oxygen_integration' );
