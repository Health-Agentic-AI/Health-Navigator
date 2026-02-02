/**
 * Health Navigator - Micro-interactions & Animations
 * Handles button ripples, scroll reveals, and other interactive effects
 */

(function() {
    'use strict';

    // ====================
    // RIPPLE EFFECT
    // ====================
    function createRipple(event, element) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;

        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;
        ripple.classList.add('ripple');

        // Remove existing ripples
        const existingRipple = element.querySelector('.ripple');
        if (existingRipple) {
            existingRipple.remove();
        }

        element.appendChild(ripple);

        ripple.addEventListener('animationend', () => {
            ripple.remove();
        });
    }

    // Initialize ripple effect on buttons
    function initRippleEffect() {
        const rippleButtons = document.querySelectorAll('.btn-primary, .btn-secondary, .btn-submit, .quick-action-btn');
        rippleButtons.forEach(button => {
            button.addEventListener('click', function(e) {
                if (!this.classList.contains('btn-loading')) {
                    createRipple(e, this);
                }
            });
        });
    }

    // ====================
    // SCROLL REVEAL
    // ====================
    function initScrollReveal() {
        const observerOptions = {
            root: null,
            rootMargin: '0px',
            threshold: 0.1
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                    // Optional: stop observing after reveal
                    // observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        const revealElements = document.querySelectorAll('.scroll-reveal');
        revealElements.forEach(el => observer.observe(el));
    }

    // ====================
    // STAGGERED ANIMATIONS
    // ====================
    function initStaggeredAnimations() {
        const staggerContainers = document.querySelectorAll('.stagger-children');
        staggerContainers.forEach(container => {
            const children = container.children;
            Array.from(children).forEach((child, index) => {
                child.style.animationDelay = `${index * 50}ms`;
            });
        });
    }

    // ====================
    // TYPING EFFECT
    // ====================
    function typeWriter(element, text, speed = 50) {
        let i = 0;
        element.textContent = '';

        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }

        type();
    }

    // Expose typing effect globally
    window.typeWriter = typeWriter;

    // ====================
    // FORM INTERACTIONS
    // ====================
    function initFormInteractions() {
        // Floating labels
        const floatInputs = document.querySelectorAll('.input-float .form-input');
        floatInputs.forEach(input => {
            input.addEventListener('focus', function() {
                this.parentElement.classList.add('focused');
            });
            input.addEventListener('blur', function() {
                if (!this.value) {
                    this.parentElement.classList.remove('focused');
                }
            });
            // Check initial state
            if (input.value) {
                input.parentElement.classList.add('focused');
            }
        });

        // Input validation visual feedback
        const requiredInputs = document.querySelectorAll('input[required], textarea[required]');
        requiredInputs.forEach(input => {
            input.addEventListener('blur', function() {
                if (this.value.trim()) {
                    this.classList.add('has-value');
                } else {
                    this.classList.remove('has-value');
                }
            });
        });
    }

    // ====================
    // BUTTON LOADING STATES
    // ====================
    function setButtonLoading(button, loading) {
        if (loading) {
            button.classList.add('btn-loading');
            button.disabled = true;
        } else {
            button.classList.remove('btn-loading');
            button.disabled = false;
        }
    }

    function setButtonSuccess(button, message = '') {
        button.classList.add('btn-success');
        button.classList.remove('btn-loading');
        button.disabled = false;

        if (message) {
            button.setAttribute('data-original-text', button.textContent);
            button.textContent = message;
        }

        setTimeout(() => {
            button.classList.remove('btn-success');
            if (message) {
                button.textContent = button.getAttribute('data-original-text') || '';
            }
        }, 2000);
    }

    // Expose button functions globally
    window.setButtonLoading = setButtonLoading;
    window.setButtonSuccess = setButtonSuccess;

    // ====================
    // TOAST NOTIFICATIONS
    // ====================
    function showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span>${message}</span>
            <button class="btn-close" onclick="this.parentElement.remove()"></button>
        `;

        // Add to container
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;
            document.body.appendChild(container);
        }

        container.appendChild(toast);

        // Auto remove
        setTimeout(() => {
            toast.classList.add('toast-hiding');
            toast.addEventListener('animationend', () => {
                toast.remove();
            });
        }, duration);
    }

    // Expose toast function globally
    window.showToast = showToast;

    // ====================
    // LAZY LOADING IMAGES
    // ====================
    function initLazyImages() {
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.classList.add('loaded');
                        imageObserver.unobserve(img);
                    }
                }
            });
        });

        const lazyImages = document.querySelectorAll('img[data-src]');
        lazyImages.forEach(img => imageObserver.observe(img));
    }

    // ====================
    // SMOOTH SCROLL
    // ====================
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    // ====================
    // KEYBOARD NAVIGATION
    // ====================
    function initKeyboardNav() {
        // Add keyboard navigation class when using tab
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-nav');
            }
        });

        // Remove when using mouse
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-nav');
        });
    }

    // ====================
    // REDUCED MOTION DETECTION
    // ====================
    function prefersReducedMotion() {
        return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    // Expose reduced motion check
    window.prefersReducedMotion = prefersReducedMotion;

    // ====================
    // DEBOUNCE FUNCTION
    // ====================
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

    // Expose debounce
    window.debounce = debounce;

    // ====================
    // THROTTLE FUNCTION
    // ====================
    function throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Expose throttle
    window.throttle = throttle;

    // ====================
    // ANIMATE ON SCROLL
    // ====================
    function initAnimateOnScroll() {
        const animatedElements = document.querySelectorAll('[data-animate]');

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !prefersReducedMotion()) {
                    const animation = entry.target.dataset.animate;
                    entry.target.classList.add(`animate-${animation}`);
                }
            });
        }, { threshold: 0.1 });

        animatedElements.forEach(el => observer.observe(el));
    }

    // ====================
    // COUNT UP ANIMATION
    // ====================
    function countUp(element, target, duration = 2000) {
        const start = 0;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = Math.floor(start + (target - start) * easeOut);

            element.textContent = current.toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = target.toLocaleString();
            }
        }

        requestAnimationFrame(update);
    }

    // Expose count up
    window.countUp = countUp;

    // ====================
    // PROGRESS BAR ANIMATION
    // ====================
    function animateProgress(element, target, duration = 1000) {
        const start = 0;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = start + (target - start) * easeOut;

            element.style.width = `${current}%`;
            element.setAttribute('aria-valuenow', Math.round(current));

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.style.width = `${target}%`;
                element.setAttribute('aria-valuenow', target);
            }
        }

        requestAnimationFrame(update);
    }

    // Expose progress animation
    window.animateProgress = animateProgress;

    // ====================
    // INITIALIZE ALL
    // ====================
    function init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeAll);
        } else {
            initializeAll();
        }
    }

    function initializeAll() {
        initRippleEffect();
        initScrollReveal();
        initStaggeredAnimations();
        initFormInteractions();
        initLazyImages();
        initSmoothScroll();
        initKeyboardNav();
        initAnimateOnScroll();

        // Emit ready event
        document.dispatchEvent(new CustomEvent('animations:ready'));
    }

    // Start
    init();

    // Re-init on dynamic content
    window.reinitAnimations = initializeAll;

})();
