# Progressive Web App (PWA) Configuration
# Phase 1: Foundation - Mobile-First Experience

"""
Paradigm: PWA as Chameleon

Traditional Web App = Picture in Frame:
- View only in browser
- No offline mode
- Reload every visit
- No notifications

PWA = Living Organism:
- Install on device (habitat)
- Work offline (hibernation)
- Cache data (fat storage)
- Push notifications (sense perception)
- Adapt to environment (responsiveness)

Chameleon Abilities:
- Color change = Responsive design
- Tail grip = Offline caching
- Eye movement = Service worker
- Camouflage = Native app feel
"""

# Web App Manifest (app identity card)
MANIFEST_CONFIG = {
    "name": "CNC Copilot Platform",
    "short_name": "CNC Copilot",
    "description": "AI-Powered Manufacturing Intelligence",
    
    # App identity (passport photo)
    "start_url": "/",
    "display": "standalone",  # Full-screen (no browser UI)
    "theme_color": "#667eea",  # Brand color
    "background_color": "#0f0c29",  # Splash screen
    
    # Icons (profile pictures at different sizes)
    "icons": [
        {
            "src": "/static/icons/icon-72x72.png",
            "sizes": "72x72",
            "type": "image/png",
            "purpose": "any maskable"
        },
        {
            "src": "/static/icons/icon-96x96.png",
            "sizes": "96x96",
            "type": "image/png"
        },
        {
            "src": "/static/icons/icon-128x128.png",
            "sizes": "128x128",
            "type": "image/png"
        },
        {
            "src": "/static/icons/icon-144x144.png",
            "sizes": "144x144",
            "type": "image/png"
        },
        {
            "src": "/static/icons/icon-152x152.png",
            "sizes": "152x152",
            "type": "image/png"
        },
        {
            "src": "/static/icons/icon-192x192.png",
            "sizes": "192x192",
            "type": "image/png"
        },
        {
            "src": "/static/icons/icon-384x384.png",
            "sizes": "384x384",
            "type": "image/png"
        },
        {
            "src": "/static/icons/icon-512x512.png",
            "sizes": "512x512",
            "type": "image/png"
        }
    ],
    
    # Orientation (landscape vs portrait)
    "orientation": "any",
    
    # Scope (boundaries of app)
    "scope": "/",
    
    # Categories (app store classification)
    "categories": ["business", "productivity", "utilities"],
    
    # Screenshots (app preview)
    "screenshots": [
        {
            "src": "/static/screenshots/dashboard.png",
            "sizes": "1280x720",
            "type": "image/png"
        }
    ],
    
    # Platform-specific
    "related_applications": [],
    "prefer_related_applications": False
}

"""
Service Worker Analogy: Personal Assistant

Service Worker = AI Assistant that:
- Works in background (even when app closed)
- Intercepts requests (gatekeeper)
- Caches resources (memory)
- Handles offline (works independently)
- Syncs data (organizes notes)

Life Cycle:
1. Install = Hiring assistant
2. Activate = Assistant starts working
3. Idle = Waiting for tasks
4. Fetch = Handling requests
5. Terminate = Takes break (saves energy)
"""

# Service Worker JavaScript
SERVICE_WORKER_JS = '''
/**
 * Service Worker - The Background Worker
 * Analogy: Invisible Butler
 */

const CACHE_NAME = 'cnc-copilot-v1';
const urlsToCache = [
    '/',
    '/static/css/main.css',
    '/static/js/app.js',
    '/static/icons/icon-192x192.png',
    '/dashboard/',
    '/offline.html', // Fallback page
];

/**
 * Installation Phase
 * Like: Butler learning house layout
 */
self.addEventListener('install', event => {
    console.log('[SW] Installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('[SW] Caching app shell');
                return cache.addAll(urlsToCache);
            })
            .then(() => self.skipWaiting()) // Take control immediately
    );
});

/**
 * Activation Phase
 * Like: Butler taking over duties
 */
self.addEventListener('activate', event => {
    console.log('[SW] Activating...');
    
    // Clean old caches (like throwing out expired food)
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => self.clients.claim()) // Control all tabs
    );
});

/**
 * Fetch Event - Network Request Interceptor
 * Like: Butler screening visitors
 * 
 * Strategies:
 * 1. Cache First (photos - rarely change)
 * 2. Network First (API data - always fresh)
 * 3. Cache Only (app shell - offline)
 * 4. Network Only (auth - never cache)
 * 5. Stale While Revalidate (best of both)
 */
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);
    
    // Strategy selection (butler's decision tree)
    if (url.pathname.startsWith('/api/')) {
        // API requests: Network first (fresh data priority)
        event.respondWith(networkFirst(event.request));
    } else if (url.pathname.match(/\.(png|jpg|css|js)$/)) {
        // Static assets: Cache first (speed priority)
        event.respondWith(cacheFirst(event.request));
    } else {
        // HTML pages: Stale while revalidate (balanced)
        event.respondWith(staleWhileRevalidate(event.request));
    }
});

/**
 * Cache First Strategy
 * Analogy: Check pantry before going to store
 */
async function cacheFirst(request) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    
    if (cached) {
        console.log('[SW] Serving from cache:', request.url);
        return cached;
    }
    
    console.log('[SW] Fetching from network:', request.url);
    const response = await fetch(request);
    
    // Cache for next time
    cache.put(request, response.clone());
    return response;
}

/**
 * Network First Strategy
 * Analogy: Call friend first, check notes if no answer
 */
async function networkFirst(request) {
    try {
        console.log('[SW] Fetching from network:', request.url);
        const response = await fetch(request);
        
        // Update cache
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, response.clone());
        
        return response;
    } catch (error) {
        console.log('[SW] Network failed, trying cache:', request.url);
        const cached = await caches.match(request);
        
        if (cached) {
            return cached;
        }
        
        // Last resort: offline page
        return caches.match('/offline.html');
    }
}

/**
 * Stale While Revalidate Strategy
 * Analogy: Use old map while getting new one
 */
async function staleWhileRevalidate(request) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    
    // Fetch fresh version in background
    const fetchPromise = fetch(request).then(response => {
        cache.put(request, response.clone());
        return response;
    });
    
    // Return cached if available, otherwise wait for network
    return cached || fetchPromise;
}

/**
 * Background Sync
 * Analogy: Todo list that works even when offline
 */
self.addEventListener('sync', event => {
    if (event.tag === 'sync-data') {
        event.waitUntil(syncData());
    }
});

async function syncData() {
    // Get pending data (like items in outbox)
    const db = await openDatabase();
    const pending = await db.getPendingData();
    
    // Try to send each item
    for (const item of pending) {
        try {
            await fetch('/api/sync/', {
                method: 'POST',
                body: JSON.stringify(item)
            });
            
            // Success: remove from queue
            await db.removePending(item.id);
        } catch (error) {
            console.log('[SW] Sync failed for item:', item.id);
            // Will retry on next sync
        }
    }
}

/**
 * Push Notifications
 * Analogy: Doorbell that works anyw here
 */
self.addEventListener('push', event => {
    const data = event.data.json();
    
    const options = {
        body: data.message,
        icon: '/static/icons/icon-192x192.png',
        badge: '/static/icons/badge-72x72.png',
        vibrate: [200, 100, 200], // Vibration pattern
        data: {
            url: data.url
        },
        actions: [
            {
                action: 'view',
                title: 'View'
            },
            {
                action: 'dismiss',
                title: 'Dismiss'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

/**
 * Notification Click Handler
 * Analogy: Clicking doorbell camera to see who it is
 */
self.addEventListener('notificationclick', event => {
    event.notification.close();
    
    if (event.action === 'view') {
        event.waitUntil(
            clients.openWindow(event.notification.data.url)
        );
    }
});
''';

"""
IndexedDB Analogy: Personal Filing Cabinet

Compared to LocalStorage (sticky notes):
- Much larger capacity (full filing cabinet vs notes)
- Structured data (organized folders vs scattered)
- Async access (don't block while searching)
- Transactions (atomic operations)

Use Cases:
- Offline data sync (backup copies)
- Large datasets (technical drawings)
- Queue management (pending tasks)
"""

# IndexedDB Schema
INDEXEDDB_SCHEMA = '''
const dbName = 'CncCopilotDB';
const dbVersion = 1;

// Open database (like unlocking filing cabinet)
const request = indexedDB.open(dbName, dbVersion);

request.onupgradeneeded = event => {
    const db = event.target.result;
    
    // Create object stores (filing cabinet drawers)
    
    // Machine data drawer
    if (!db.objectStoreNames.contains('machines')) {
        const machineStore = db.createObjectStore('machines', { keyPath: 'id' });
        machineStore.createIndex('status', 'status', { unique: false });
        machineStore.createIndex('organization', 'organization_id', { unique: false });
    }
    
    // Telemetry data drawer
    if (!db.objectStoreNames.contains('telemetry')) {
        const telemetryStore = db.createObjectStore('telemetry', { 
            keyPath: 'id', 
            autoIncrement: true 
        });
        telemetryStore.createIndex('machine_timestamp', ['machine_id', 'timestamp']);
    }
    
    // Pending sync drawer (outbox)
    if (!db.objectStoreNames.contains('pending_sync')) {
        db.createObjectStore('pending_sync', { 
            keyPath: 'id', 
            autoIncrement: true 
        });
    }
};

// Database operations (filing actions)
class Database {
    static async addMachine(machine) {
        // Like filing new document
        const db = await this.getDB();
        const tx = db.transaction('machines', 'readwrite');
        const store = tx.objectStore('machines');
        await store.put(machine);
        await tx.complete;
    }
    
    static async getMachine(id) {
        // Like retrieving specific file
        const db = await this.getDB();
        const tx = db.transaction('machines', 'readonly');
        const store = tx.objectStore('machines');
        return await store.get(id);
    }
    
    static async queryMachinesByStatus(status) {
        // Like searching files by category
        const db = await this.getDB();
        const tx = db.transaction('machines', 'readonly');
        const store = tx.objectStore('machines');
        const index = store.index('status');
        return await index.getAll(status);
    }
}
''';

"""
App Install Prompt Analogy: Salesperson Pitch

Before PWA:
- User must manually add to home screen
- Hidden in browser menu
- Low discoverability

With Prompt:
- Proactive suggestion (salesperson approach)
- timing matters (don't be pushy)
- Show value first (demonstrate product)
- Easy installation (one-click purchase)
"""

# Install Prompt JavaScript
INSTALL_PROMPT_JS = '''
/**
 * PWA Install Prompt
 * Analogy: Smart salesperson
 */

let deferredPrompt = null;

// Capture install event (salesperson arrives)
window.addEventListener('beforeinstallprompt', (e) => {
    console.log('[PWA] Install prompt available');
    
    // Prevent default (don't show automatically)
    e.preventDefault();
    
    // Store for later (wait for right moment)
    deferredPrompt = e;
    
    // Show custom install button
    showInstallPromotion();
});

function showInstallPromotion() {
    /**
     * Show install button at opportune moment
     * Like: approach customer after they've browsed
     */
    
    // Check if user has visited 3+ times (interested customer)
    const visitCount = parseInt(localStorage.getItem('visit_count') || '0');
    
    if (visitCount >= 3) {
        const installBtn = document.getElementById('install-btn');
        installBtn.style.display = 'block';
        
        // Handle click (customer says yes)
        installBtn.addEventListener('click', async () => {
            if (deferredPrompt) {
                // Show native prompt (complete purchase)
                deferredPrompt.prompt();
                
                // Wait for user choice
                const { outcome } = await deferredPrompt.userChoice;
                
                console.log('[PWA] Install outcome:', outcome);
                
                // Clean up
                deferredPrompt = null;
                installBtn.style.display = 'none';
            }
        });
    }
    
    // Increment visit count
    localStorage.setItem('visit_count', visitCount + 1);
}

// Track install success (sale completed)
window.addEventListener('appinstalled', () => {
    console.log('[PWA] App installed successfully');
    
    // Analytics tracking
    trackEvent('pwa_install', { method: 'banner' });
    
    // Hide install button
    const installBtn = document.getElementById('install-btn');
    if (installBtn) {
        installBtn.style.display = 'none';
    }
});
''';

"""
Offline Experience Analogy: Emergency Supplies

Like having:
- Canned food (cached HTML/CSS/JS)
- First aid kit (offline page)
- Backup generator (service worker)
- Water storage (IndexedDB)
- Radio (background sync)

Offline Strategies:
1. Full offline (airplane mode works)
2. Partial offline (core features only)
3. Graceful degradation (show what's available)
4. Optimistic UI (assume success, sync later)
"""

# Push Notification Setup (Django)
PUSH_NOTIFICATIONS_SETTINGS = {
    "FCM_API_KEY": "[Firebase Cloud Messaging Key]",
    "GCM_API_KEY": "[Google Cloud Messaging Key]",
    "APNS_CERTIFICATE": "/path/to/cert.pem",  # iOS
    "WP_PRIVATE_KEY": "[Windows Push Key]",
}

# Example: Send push notification
def send_push_notification(user, title, message, url):
    """
    Send notification to user's device
    Analogy: Sending telegram
    """
    from push_notifications.models import GCMDevice
    
    devices = GCMDevice.objects.filter(user=user)
    
    devices.send_message(
        title=title,
        message=message,
        extra={
            'url': url,
            'vibrate': [200, 100, 200],
        }
    )

"""
PWA Checklist (Launch Requirements)

Like Rocket Launch Checklist:
✅ HTTPS enabled (secure connection)
✅ Web app manifest (identity)
✅ Service worker registered (background worker)
✅ Icons (all sizes)
✅ Responsive design (works on all devices)
✅ Fast load time (<3 seconds)
✅ Works offline (cached shell)
✅ Install prompt (user engagement)
✅ Push notifications (re-engagement)
✅ App-like experience (no browser UI)
"""
