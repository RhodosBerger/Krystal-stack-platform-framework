/**
 * KRYSTAL-STACK: ASSET & DEPENDENCY HYDRATION LOADER
 * ==============================================================================
 * Provides a resilient dual-layer dependency pipeline:
 * 1. Attempts to load local vendor bundles (/static/vendor/...) for 100% offline air-gapped dev.
 * 2. Gracefully falls back to high-speed CDNs (Cloudflare / jsDelivr) if local assets are missing.
 * 3. Verifies library availability (Three.js r128, OrbitControls) before bootstrapping 3D scenes.
 * 4. Dispatches 'krystal:dependencies-ready' window event.
 * ==============================================================================
 */

(function() {
    'use strict';

    const DEPENDENCY_MANIFEST = [
        {
            name: 'Three.js Core',
            globalCheck: () => window.THREE !== undefined,
            localUrl: '/static/vendor/three.min.js',
            cdnUrl: 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'
        },
        {
            name: 'Three.js OrbitControls',
            globalCheck: () => window.THREE && window.THREE.OrbitControls !== undefined,
            localUrl: '/static/vendor/OrbitControls.js',
            cdnUrl: 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'
        }
    ];

    function loadScript(url) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.async = false;
            script.onload = () => resolve(url);
            script.onerror = () => reject(new Error(`Failed to load script from ${url}`));
            document.head.appendChild(script);
        });
    }

    async function loadWithFallback(dep) {
        if (dep.globalCheck()) {
            console.log(`[AssetLoader] ${dep.name} is already available in window scope.`);
            return true;
        }

        // 1. Try local vendor bundle
        try {
            console.log(`[AssetLoader] Fetching ${dep.name} from local vendor bundle: ${dep.localUrl}...`);
            await loadScript(dep.localUrl);
            if (dep.globalCheck()) {
                console.log(`[AssetLoader] Successfully loaded ${dep.name} locally (Offline-Ready).`);
                return true;
            }
        } catch (localErr) {
            console.warn(`[AssetLoader] Local bundle unavailable for ${dep.name}. Falling back to CDN...`);
        }

        // 2. Fallback to CDN
        try {
            console.log(`[AssetLoader] Fetching ${dep.name} from CDN: ${dep.cdnUrl}...`);
            await loadScript(dep.cdnUrl);
            if (dep.globalCheck()) {
                console.log(`[AssetLoader] Successfully loaded ${dep.name} via CDN fallback.`);
                return true;
            }
        } catch (cdnErr) {
            console.error(`[AssetLoader] CRITICAL: Failed to load ${dep.name} from both local and CDN sources!`, cdnErr);
            return false;
        }

        return false;
    }

    async function hydrateAllDependencies() {
        console.log("[AssetLoader] Starting Krystal-Stack dependency hydration cycle...");
        let allSuccess = true;

        for (const dep of DEPENDENCY_MANIFEST) {
            const ok = await loadWithFallback(dep);
            if (!ok) allSuccess = false;
        }

        window.__KRYSTAL_DEPS_READY = allSuccess;
        const event = new CustomEvent('krystal:dependencies-ready', { detail: { success: allSuccess } });
        window.dispatchEvent(event);
        console.log(`[AssetLoader] Hydration complete. All dependencies active: ${allSuccess}`);
    }

    window.checkDependencyHealth = function() {
        return {
            three_loaded: window.THREE !== undefined,
            orbit_controls_loaded: window.THREE && window.THREE.OrbitControls !== undefined,
            all_ready: window.__KRYSTAL_DEPS_READY === true
        };
    };

    // Auto-execute on DOMContentLoaded or immediately if already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hydrateAllDependencies);
    } else {
        hydrateAllDependencies();
    }
})();
