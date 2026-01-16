// Main JavaScript - Global functionality for Health Navigator

document.addEventListener('DOMContentLoaded', function () {
    // ==================
    // NAVBAR SCROLL EFFECT
    // ==================
    const navbar = document.querySelector('.glass-navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 50) {
            navbar?.classList.add('scrolled');
        } else {
            navbar?.classList.remove('scrolled');
        }

        lastScroll = currentScroll;
    });

    // ==================
    // ANIMATED BACKGROUND
    // ==================
    const bgAnimated = document.querySelector('.bg-animated');
    if (bgAnimated) {
        // Add mouse movement parallax effect
        document.addEventListener('mousemove', (e) => {
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;

            const before = bgAnimated.querySelector('::before');
            if (before) {
                bgAnimated.style.setProperty('--mouse-x', `${x * 100}%`);
                bgAnimated.style.setProperty('--mouse-y', `${y * 100}%`);
            }
        });
    }

    // ==================
    // LOADING ANIMATIONS
    // ==================
    // Add fade-in animation to elements
    const animateOnLoad = document.querySelectorAll('.animate-on-load');
    animateOnLoad.forEach((element, index) => {
        setTimeout(() => {
            element.classList.add('fade-in');
        }, index * 100);
    });

    // ==================
    // TOOLTIPS (Bootstrap)
    // ==================
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }

    // ==================
    // SMOOTH SCROLLING
    // ==================
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#' && href !== '') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // ==================
    // KEYBOARD SHORTCUTS
    // ==================
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus search/input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('.message-input') ||
                document.querySelector('.form-input');
            if (searchInput) {
                searchInput.focus();
            }
        }

        // Escape to close modals/alerts
        if (e.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal.show');
            openModals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });

            // Close sidebar on mobile
            const sidebar = document.querySelector('.chat-sidebar.show');
            if (sidebar) {
                sidebar.classList.remove('show');
            }
        }
    });

    // ==================
    // FORM ENHANCEMENTS
    // ==================
    // Auto-focus first input in forms
    const firstInput = document.querySelector('.auth-form input:not([type="hidden"])');
    if (firstInput && !document.querySelector('.alert-error')) {
        setTimeout(() => {
            firstInput.focus();
        }, 500);
    }

    // Add floating label effect
    const floatingInputs = document.querySelectorAll('.form-input');
    floatingInputs.forEach(input => {
        // Check if has value on load
        if (input.value) {
            input.classList.add('has-value');
        }

        input.addEventListener('input', function () {
            if (this.value) {
                this.classList.add('has-value');
            } else {
                this.classList.remove('has-value');
            }
        });

        input.addEventListener('focus', function () {
            this.classList.add('is-focused');
        });

        input.addEventListener('blur', function () {
            this.classList.remove('is-focused');
        });
    });

    // ==================
    // NOTIFICATION SYSTEM
    // ==================
    window.showNotification = function (message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="bi bi-${getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="notification-close">
                <i class="bi bi-x"></i>
            </button>
        `;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);

        // Close button
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            closeNotification(notification);
        });

        // Auto close
        if (duration > 0) {
            setTimeout(() => {
                closeNotification(notification);
            }, duration);
        }
    };

    function closeNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }

    function getNotificationIcon(type) {
        const icons = {
            'success': 'check-circle-fill',
            'error': 'exclamation-circle-fill',
            'warning': 'exclamation-triangle-fill',
            'info': 'info-circle-fill'
        };
        return icons[type] || icons.info;
    }

    // ==================
    // PERFORMANCE MONITORING
    // ==================
    // Log page load time
    window.addEventListener('load', () => {
        const loadTime = performance.now();
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    });

    // ==================
    // NETWORK STATUS
    // ==================
    window.addEventListener('online', () => {
        console.log('Connection restored');
        if (window.showNotification) {
            showNotification('Connection restored', 'success');
        }
    });

    window.addEventListener('offline', () => {
        console.log('Connection lost');
        if (window.showNotification) {
            showNotification('No internet connection', 'error', 0);
        }
    });

    // ==================
    // RESPONSIVE HELPERS
    // ==================
    function updateViewportHeight() {
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }

    updateViewportHeight();
    window.addEventListener('resize', updateViewportHeight);

    // ==================
    // ACCESSIBILITY ENHANCEMENTS
    // ==================
    // Focus visible for keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-nav');
        }
    });

    document.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-nav');
    });

    // ==================
    // ERROR HANDLING
    // ==================
    window.addEventListener('error', (e) => {
        console.error('Global error:', e.error);
        // Could send to error tracking service here
    });

    window.addEventListener('unhandledrejection', (e) => {
        console.error('Unhandled promise rejection:', e.reason);
        // Could send to error tracking service here
    });

    // ==================
    // COPY TO CLIPBOARD
    // ==================
    window.copyToClipboard = async function (text) {
        try {
            await navigator.clipboard.writeText(text);
            if (window.showNotification) {
                showNotification('Copied to clipboard', 'success');
            }
            return true;
        } catch (err) {
            console.error('Failed to copy:', err);
            if (window.showNotification) {
                showNotification('Failed to copy', 'error');
            }
            return false;
        }
    };

    // ==================
    // LOCAL STORAGE HELPERS
    // ==================
    window.storage = {
        set: (key, value) => {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('Storage error:', e);
                return false;
            }
        },
        get: (key, defaultValue = null) => {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('Storage error:', e);
                return defaultValue;
            }
        },
        remove: (key) => {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('Storage error:', e);
                return false;
            }
        },
        clear: () => {
            try {
                localStorage.clear();
                return true;
            } catch (e) {
                console.error('Storage error:', e);
                return false;
            }
        }
    };

    // ==================
    // THEME SYSTEM (Optional)
    // ==================
    const savedTheme = window.storage.get('theme', 'dark');
    document.documentElement.setAttribute('data-theme', savedTheme);

    window.setTheme = function (theme) {
        document.documentElement.setAttribute('data-theme', theme);
        window.storage.set('theme', theme);
    };

    // ==================
    // INITIALIZATION COMPLETE
    // ==================
    console.log('Health Navigator initialized');
    console.log('Version: 1.0.0');
    console.log('Environment: Production');
});

// ==================
// UTILITY FUNCTIONS (Global)
// ==================

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function (...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Format date
function formatDate(date) {
    const d = new Date(date);
    const now = new Date();
    const diff = now - d;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;

    return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: d.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Validate email
function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Generate unique ID
function generateUniqueId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

// Export utilities
window.utils = {
    debounce,
    throttle,
    formatDate,
    formatFileSize,
    isValidEmail,
    generateUniqueId
};