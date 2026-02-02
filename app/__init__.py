"""
Health Navigator - Flask Application Factory
Centralized application initialization with configuration management
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

# Initialize extensions (bind later in create_app)
db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.environ.get("REDIS_URL", "memory://")
)


def create_app(config=None):
    """
    Application factory pattern for creating Flask app.

    Args:
        config: Optional AppConfig instance. If None, loads from environment.

    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)

    # Load configuration
    from app.config import AppConfig, init_app as init_config
    if config is None:
        config = AppConfig.from_env()

    # Validate and apply configuration
    config.validate()
    init_config(app, config)

    # Store config in app for access
    app.config['app_config'] = config

    # Initialize CSRF protection
    csrf.init_app(app)

    # Security headers middleware
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses."""
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.gstatic.com https://stackpath.bootstrapcdn.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com https://use.fontawesome.com; "
            "font-src 'self' https://fonts.gstatic.com https://use.fontawesome.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://generativelanguage.googleapis.com; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers['Content-Security-Policy'] = csp
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'

        # HSTS only in production
        if config.env == 'production':
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        return response

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
