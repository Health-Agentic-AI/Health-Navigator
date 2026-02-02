from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv(r'C:\My Projects\Health-Navigator\credentials.env')

db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.environ.get("REDIS_URL", "memory://")
)

def create_app():
    app = Flask(__name__)

    # Database Configuration - PostgreSQL
    # Using psycopg2 as driver
    app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql+psycopg2://{os.environ.get("POSTGRES_USERNAME", "postgres")}:{os.environ.get("POSTGRES_PASSWORD", "password")}@{os.environ.get("POSTGRES_HOST", "localhost")}:{os.environ.get("POSTGRES_PORT", "5432")}/{os.environ.get("DATABASE_NAME", "medical_db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.environ.get("FLASK_SECRET_KEY", "dev-key-please-change")

    # Security Configuration - Session
    app.config.update(
        # Session cookie security
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "False") == "True",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
        SESSION_REFRESH_EACH_REQUEST=True,
    )

    # CSRF Configuration
    app.config['WTF_CSRF_TIME_LIMIT'] = None  # No timeout for CSRF tokens
    app.config['WTF_CSRF_SSL_STRICT'] = True  # Require HTTPS for CSRF in production

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    limiter.init_app(app)

    # Setup structured logging
    from app.logging_config import setup_logging, add_request_id_middleware
    setup_logging(app)
    add_request_id_middleware(app)

    # Initialize model registry
    from app.workflow.model_registry import init_default_models
    try:
        init_default_models()
        app.logger.info("Model registry initialized with default models")
    except Exception as e:
        app.logger.warning(f"Could not initialize model registry: {e}")

    # Security headers
    @app.after_request
    def add_security_headers(response):
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
        if os.environ.get("FLASK_ENV") != "development":
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        return response

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
