/**
 * Health Navigator - Service Worker
 * Progressive Web App features: offline support, caching, background sync
 */

const CACHE_NAME = 'health-navigator-v4';
const STATIC_CACHE = 'health-navigator-static-v4';
const DYNAMIC_CACHE = 'health-navigator-dynamic-v4';

// URLs to cache on install
const STATIC_ASSETS = [
  '/',
  '/login',
  '/register',
  '/chat',
  '/static/css/base.css',
  '/static/css/layout.css',
  '/static/css/components.css',
  '/static/css/animations.css',
  '/static/css/accessibility.css',
  '/static/css/style.css',
  '/static/js/main.js',
  '/static/js/animations.js',
  '/static/js/service-worker.js',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js',
  'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;500;600;700&display=swap'
];

// API endpoints that should use network-first strategy
const API_ENDPOINTS = [
  /\/api\/.*/,
  /\/health/
];

// ==============================
// Install Event
// ==============================
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');

  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log('[Service Worker] Caching static assets');
      return cache.addAll(STATIC_ASSETS.map(url => new Request(url, { cache: 'reload' })))
        .catch((error) => {
          console.error('[Service Worker] Failed to cache static assets:', error);
          // Continue even if some assets fail to cache
          return Promise.resolve();
        });
    })
  );

  // Force the waiting service worker to become active
  self.skipWaiting();
});

// ==============================
// Activate Event
// ==============================
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          // Delete old caches that don't match the current version
          if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
            console.log('[Service Worker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );

  // Take control of all pages immediately
  return self.clients.claim();
});

// ==============================
// Fetch Event - Network Strategy
// ==============================
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip chrome extensions and other protocols
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // API endpoints: Network First with Cache fallback
  if (API_ENDPOINTS.some(pattern => pattern.test(url.pathname))) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }

  // Static assets: Cache First
  if (url.pathname.startsWith('/static/') || url.pathname.endsWith('.css') || url.pathname.endsWith('.js')) {
    event.respondWith(cacheFirstStrategy(request));
    return;
  }

  // HTML pages: Network First with Cache fallback
  if (request.headers.get('accept').includes('text/html')) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }

  // Images: Cache First with stale-while-revalidate
  if (request.destination === 'image') {
    event.respondWith(staleWhileRevalidateStrategy(request));
    return;
  }

  // Default: Network First
  event.respondWith(networkFirstStrategy(request));
});

// ==============================
// Caching Strategies
// ==============================

// Network First: Try network, fallback to cache
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    // Cache successful responses
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('[Service Worker] Network failed, trying cache:', request.url);
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    // Return offline fallback for HTML requests
    if (request.headers.get('accept').includes('text/html')) {
      return caches.match('/offline') || new Response('Offline - No cached version available', {
        status: 503,
        statusText: 'Service Unavailable',
        headers: new Headers({ 'Content-Type': 'text/plain' })
      });
    }
    throw error;
  }
}

// Cache First: Try cache, fallback to network
async function cacheFirstStrategy(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.error('[Service Worker] Cache and network failed:', request.url);
    throw error;
  }
}

// Stale While Revalidate: Serve cache, update in background
async function staleWhileRevalidateStrategy(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cachedResponse = await cache.match(request);

  // Fetch in background to update cache
  const fetchPromise = fetch(request).then((networkResponse) => {
    if (networkResponse && networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  });

  // Return cached response immediately, or wait for network
  return cachedResponse || fetchPromise;
}

// ==============================
// Background Sync
// ==============================
self.addEventListener('sync', (event) => {
  console.log('[Service Worker] Background sync:', event.tag);

  if (event.tag === 'sync-messages') {
    event.waitUntil(syncMessages());
  }
});

async function syncMessages() {
  // Implement background sync for offline messages
  try {
    // Get all pending messages from IndexedDB
    // Send them to the server
    // Clear pending messages on success
    console.log('[Service Worker] Background sync completed');
  } catch (error) {
    console.error('[Service Worker] Background sync failed:', error);
  }
}

// ==============================
// Push Notifications (optional)
// ==============================
self.addEventListener('push', (event) => {
  if (!event.data) {
    return;
  }

  const data = event.data.json();
  const options = {
    body: data.body || 'New update from Health Navigator',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    image: data.image,
    data: {
      url: data.url || '/',
      conversationId: data.conversationId
    },
    actions: [
      {
        action: 'open',
        title: 'Open'
      },
      {
        action: 'dismiss',
        title: 'Dismiss'
      }
    ],
    requireInteraction: false,
    silent: false
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Health Navigator', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'dismiss') {
    return;
  }

  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((clientList) => {
      // Focus or open a window to the appropriate URL
      const url = event.notification.data.url || '/';

      for (const client of clientList) {
        if (client.url.includes(url) && 'focus' in client) {
          return client.focus();
        }
      }

      if (clients.openWindow) {
        return clients.openWindow(url);
      }
    })
  );
});

// ==============================
// Message Handling
// ==============================
self.addEventListener('message', (event) => {
  const { data, ports } = event;

  if (data && data.type === 'SKIP_WAITING') {
    self.skipWaiting();
    if (ports && ports[0]) {
      ports[0].postMessage({ type: 'SKIPPED_WAITING' });
    }
  }

  if (data && data.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(DYNAMIC_CACHE).then((cache) => {
        return cache.addAll(data.urls);
      })
    );
  }

  if (data && data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.delete(DYNAMIC_CACHE).then(() => {
        if (ports && ports[0]) {
          ports[0].postMessage({ type: 'CACHE_CLEARED' });
        }
      })
    );
  }
});

// ==============================
// Periodic Background Sync (for periodic updates)
// ==============================
self.addEventListener('periodicsync', (event) => {
  console.log('[Service Worker] Periodic sync:', event.tag);

  if (event.tag === 'sync-conversations') {
    event.waitUntil(syncConversations());
  }
});

async function syncConversations() {
  // Implement periodic sync for conversation updates
  try {
    const response = await fetch('/api/conversations', {
      headers: {
        'Cache-Control': 'no-cache'
      }
    });
    if (response.ok) {
      const conversations = await response.json();
      // Update IndexedDB with latest conversations
      console.log('[Service Worker] Synced conversations:', conversations);
    }
  } catch (error) {
    console.error('[Service Worker] Periodic sync failed:', error);
  }
}

// ==============================
// Cache cleanup on storage pressure
// ==============================
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      // Keep only the two most recent caches
      const cachePromises = cacheNames
        .sort()
        .slice(0, -2)
        .map((cacheName) => {
          console.log('[Service Worker] Deleting old cache on storage pressure:', cacheName);
          return caches.delete(cacheName);
        });
      return Promise.all(cachePromises);
    })
  );
});

console.log('[Service Worker] Service Worker file loaded');
