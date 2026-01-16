// Register Page JavaScript
document.addEventListener('DOMContentLoaded', () => {
    const registerForm = document.getElementById('registerForm');
    const passwordInput = document.getElementById('passwordInput');
    const togglePassword = document.getElementById('togglePassword');
    const submitBtn = document.getElementById('submitBtn');
    const passwordStrength = document.getElementById('passwordStrength');

    // Toggle password visibility
    if (togglePassword && passwordInput) {
        togglePassword.addEventListener('click', () => {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);

            const icon = togglePassword.querySelector('i');
            if (type === 'password') {
                icon.classList.remove('bi-eye-slash');
                icon.classList.add('bi-eye');
            } else {
                icon.classList.remove('bi-eye');
                icon.classList.add('bi-eye-slash');
            }
        });
    }

    // Password strength indicator
    if (passwordInput && passwordStrength) {
        passwordInput.addEventListener('input', function () {
            const password = this.value;
            const strength = calculatePasswordStrength(password);
            updatePasswordStrength(strength);
        });
    }

    function calculatePasswordStrength(password) {
        if (!password) return 0;

        let strength = 0;

        // Length check
        if (password.length >= 8) strength += 1;
        if (password.length >= 12) strength += 1;

        // Character variety checks
        if (/[a-z]/.test(password)) strength += 1;
        if (/[A-Z]/.test(password)) strength += 1;
        if (/[0-9]/.test(password)) strength += 1;
        if (/[^a-zA-Z0-9]/.test(password)) strength += 1;

        // Normalize to 0-4 scale
        return Math.min(4, Math.floor(strength / 1.5));
    }

    function updatePasswordStrength(strength) {
        // Remove all strength classes
        passwordStrength.classList.remove('weak', 'medium', 'strong', 'very-strong');

        const strengthText = passwordStrength.querySelector('.strength-text');

        switch (strength) {
            case 0:
                strengthText.textContent = 'Too weak';
                break;
            case 1:
                passwordStrength.classList.add('weak');
                strengthText.textContent = 'Weak';
                break;
            case 2:
                passwordStrength.classList.add('medium');
                strengthText.textContent = 'Fair';
                break;
            case 3:
                passwordStrength.classList.add('strong');
                strengthText.textContent = 'Strong';
                break;
            case 4:
                passwordStrength.classList.add('very-strong');
                strengthText.textContent = 'Very strong';
                break;
        }
    }

    // Form validation and submission
    if (registerForm) {
        registerForm.addEventListener('submit', function (e) {
            const password = passwordInput.value;
            const strength = calculatePasswordStrength(password);

            // Require at least medium strength
            if (strength < 2) {
                e.preventDefault();
                alert('Please choose a stronger password (at least 8 characters with mixed case, numbers, and symbols)');
                return;
            }

            // Add loading state
            submitBtn.classList.add('loading');
            submitBtn.disabled = true;

            // Form will submit normally
            // If you want AJAX submission, handle it here
        });
    }

    // Input focus effects
    const inputs = document.querySelectorAll('.form-input');
    inputs.forEach(input => {
        input.addEventListener('focus', function () {
            this.parentElement.classList.add('focused');
        });

        input.addEventListener('blur', function () {
            this.parentElement.classList.remove('focused');
        });
    });

    // Username validation (no spaces, lowercase)
    const usernameInput = document.querySelector('input[name="username"]');
    if (usernameInput) {
        usernameInput.addEventListener('input', function () {
            this.value = this.value.toLowerCase().replace(/\s/g, '');
        });
    }

    // Smooth animations on load
    setTimeout(() => {
        document.querySelector('.auth-card')?.classList.add('fade-in');
    }, 100);
});