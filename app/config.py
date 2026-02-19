"""
Health Navigator - Configuration Management
Environment-based configuration classes with validation
"""

import os
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
from dotenv import load_dotenv
import logging
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# Load environment variables from .env file
# Try to load from project root
try:
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / '.env')
    load_dotenv(project_root / 'credentials.env')
except Exception:
    pass  # Fallback to environment variables


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def _get_bool(env_var: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.environ.get(env_var, str(default))
    return value.lower() in ('true', '1', 'yes', 'on')


def _get_int(env_var: str, default: int = 0, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    """Get integer value from environment variable with optional bounds checking."""
    value = os.environ.get(env_var, str(default))
    try:
        int_value = int(value)
        if min_value is not None and int_value < min_value:
            raise ConfigValidationError(f"{env_var} must be at least {min_value}")
        if max_value is not None and int_value > max_value:
            raise ConfigValidationError(f"{env_var} must be at most {max_value}")
        return int_value
    except ValueError:
        raise ConfigValidationError(f"{env_var} must be an integer")


def _get_list(env_var: str, default: Optional[str] = None) -> List[str]:
    """Get list value from comma-separated environment variable."""
    value = os.environ.get(env_var, default or '')
    return [item.strip() for item in value.split(',') if item.strip()]


@dataclass
class DatabaseConfig:
    """Database configuration with PostgreSQL-first, MySQL-fallback support."""
    host: str
    port: int
    username: str
    password: str
    database: str
    mysql_host: str
    mysql_port: int
    mysql_username: str
    mysql_password: str
    mysql_database: str
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    connect_timeout: int = 3
    _active_uri: Optional[str] = field(default=None, init=False, repr=False)
    _active_backend: Optional[str] = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create database config from environment variables."""
        default_database = os.environ.get('DATABASE_NAME', 'medical_db')
        return cls(
            host=os.environ.get('POSTGRES_HOST', 'localhost'),
            port=_get_int('POSTGRES_PORT', 5432, 1, 65535),
            username=os.environ.get('POSTGRES_USERNAME', 'postgres'),
            password=os.environ.get('POSTGRES_PASSWORD', 'password'),
            database=default_database,
            mysql_host=os.environ.get('MYSQL_HOST', 'localhost'),
            mysql_port=_get_int('MYSQL_PORT', 3306, 1, 65535),
            mysql_username=os.environ.get('MYSQL_USERNAME', 'root'),
            mysql_password=os.environ.get('MYSQL_PASSWORD', 'password'),
            mysql_database=os.environ.get('MYSQL_DATABASE', default_database),
            pool_size=_get_int('DB_POOL_SIZE', 20, 1, 100),
            max_overflow=_get_int('DB_MAX_OVERFLOW', 40, 0, 100),
            pool_timeout=_get_int('DB_POOL_TIMEOUT', 30, 1, 300),
            pool_recycle=_get_int('DB_POOL_RECYCLE', 3600, 60, 86400),
            echo=_get_bool('DB_ECHO', False),
            connect_timeout=_get_int('DB_CONNECT_TIMEOUT', 3, 1, 30)
        )

    def get_postgres_uri(self) -> str:
        """Assemble PostgreSQL connection URI from discrete connection fields."""
        return f'postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}'

    def get_mysql_uri(self) -> str:
        """Assemble MySQL connection URI from discrete connection fields."""
        return f'mysql+pymysql://{self.mysql_username}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}'

    def get_candidate_uris(self) -> List[Tuple[str, str]]:
        """Return connection candidates in priority order."""
        return [
            ('postgresql', self.get_postgres_uri()),
            ('mysql', self.get_mysql_uri())
        ]

    def _test_uri(self, uri: str) -> None:
        """Test DB connectivity using SQLAlchemy."""
        engine = create_engine(
            uri,
            pool_pre_ping=True,
            connect_args={'connect_timeout': self.connect_timeout}
        )
        try:
            with engine.connect() as connection:
                connection.execute(text('SELECT 1'))
        finally:
            engine.dispose()

    def resolve_uri(self, force_refresh: bool = False) -> Tuple[str, str]:
        """
        Resolve and cache the active DB URI, trying PostgreSQL first then MySQL.

        Returns:
            Tuple[str, str]: (resolved_uri, backend_type)
        """
        if not force_refresh and self._active_uri and self._active_backend:
            return self._active_uri, self._active_backend

        last_error = None
        for backend, uri in self.get_candidate_uris():
            try:
                self._test_uri(uri)
                self._active_uri = uri
                self._active_backend = backend
                logger.info(f"Database connection resolved using {backend}")
                return uri, backend
            except Exception as exc:
                last_error = exc
                logger.warning(f"Failed to connect using {backend}, trying fallback: {exc}")

        raise ConfigValidationError(
            "Unable to connect to PostgreSQL or MySQL. Check DB credentials and availability."
        ) from last_error

    def get_uri(self) -> str:
        """Get resolved database URI (PostgreSQL first, then MySQL fallback)."""
        uri, _ = self.resolve_uri()
        return uri

    def get_database_type(self) -> str:
        """Get resolved database backend type."""
        _, backend = self.resolve_uri()
        return backend


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str
    session_secure: bool = False
    session_httponly: bool = True
    session_samesite: str = 'Lax'
    session_lifetime_hours: int = 24
    csrf_enabled: bool = True
    csrf_time_limit: Optional[int] = None
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digit: bool = True
    password_require_special: bool = True

    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Create security config from environment variables."""
        secret_key = os.environ.get('FLASK_SECRET_KEY')
        if not secret_key or secret_key == 'dev-key-please-change':
            if os.environ.get('FLASK_ENV') == 'production':
                raise ConfigValidationError('FLASK_SECRET_KEY must be set in production')
            secret_key = 'dev-key-please-change-in-production'

        return cls(
            secret_key=secret_key,
            session_secure=_get_bool('SESSION_COOKIE_SECURE', False),
            session_httponly=_get_bool('SESSION_COOKIE_HTTPONLY', True),
            session_samesite=os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax'),
            session_lifetime_hours=_get_int('PERMANENT_SESSION_LIFETIME_HOURS', 24, 1, 168),
            csrf_enabled=_get_bool('CSRF_ENABLED', True),
            csrf_time_limit=None if _get_bool('WTF_CSRF_TIME_LIMIT_NONE', False) else None,
            password_min_length=_get_int('PASSWORD_MIN_LENGTH', 8, 4, 128),
            password_require_uppercase=_get_bool('PASSWORD_REQUIRE_UPPERCASE', True),
            password_require_lowercase=_get_bool('PASSWORD_REQUIRE_LOWERCASE', True),
            password_require_digit=_get_bool('PASSWORD_REQUIRE_DIGIT', True),
            password_require_special=_get_bool('PASSWORD_REQUIRE_SPECIAL', True)
        )


@dataclass
class RedisConfig:
    """Redis configuration for caching and rate limiting."""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    url: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create Redis config from environment variables."""
        url = os.environ.get('REDIS_URL')
        return cls(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=_get_int('REDIS_PORT', 6379, 1, 65535),
            db=_get_int('REDIS_DB', 0, 0, 15),
            password=os.environ.get('REDIS_PASSWORD'),
            url=url
        )

    def get_url(self) -> str:
        """Get the Redis connection URL."""
        if self.url:
            return self.url
        auth = f':{self.password}@' if self.password else ''
        return f'redis://{auth}{self.host}:{self.port}/{self.db}'


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    storage: str = 'memory'  # 'memory' or 'redis'
    default_limits: List[str] = field(default_factory=lambda: ['200 per day', '50 per hour'])
    auth_endpoints: str = '5 per minute'
    chat_endpoints: str = '20 per minute'
    upload_endpoints: str = '10 per minute'

    @classmethod
    def from_env(cls) -> 'RateLimitConfig':
        """Create rate limit config from environment variables."""
        return cls(
            enabled=_get_bool('RATE_LIMIT_ENABLED', True),
            storage=os.environ.get('RATE_LIMIT_STORAGE', 'redis' if os.environ.get('REDIS_URL') else 'memory'),
            default_limits=_get_list('RATE_LIMIT_DEFAULT', '200 per day,50 per hour'),
            auth_endpoints=os.environ.get('RATE_LIMIT_AUTH', '5 per minute'),
            chat_endpoints=os.environ.get('RATE_LIMIT_CHAT', '20 per minute'),
            upload_endpoints=os.environ.get('RATE_LIMIT_UPLOAD', '10 per minute')
        )


@dataclass
class APIConfig:
    """External API configuration."""
    llm_provider: str = 'google'  # google or z.ai
    google_api_key: Optional[str] = None
    google_model: str = 'gemini-2.0-flash-exp'
    vision_model: str = 'gemini-2.0-flash-exp'
    zai_api_key: Optional[str] = None
    zai_model: str = 'glm-4.7'
    zai_base_url: str = 'https://api.z.ai/api/paas/v4/'
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create API config from environment variables."""
        return cls(
            llm_provider=os.environ.get('LLM_PROVIDER', 'google'),
            google_api_key=os.environ.get('GOOGLE_API_KEY'),
            google_model=os.environ.get('GOOGLE_MODEL', 'gemini-2.0-flash-exp'),
            vision_model=os.environ.get('GOOGLE_VISION_MODEL', 'gemini-2.0-flash-exp'),
            zai_api_key=os.environ.get('ZAI_API_KEY'),
            zai_model=os.environ.get('ZAI_MODEL', 'glm-4.7'),
            zai_base_url=os.environ.get('ZAI_BASE_URL', 'https://api.z.ai/api/paas/v4/'),
            temperature=float(os.environ.get('API_TEMPERATURE', '0.7')),
            max_tokens=_get_int('API_MAX_TOKENS', 4096, 128, 32768),
            timeout=_get_int('API_TIMEOUT', 120, 10, 600)
        )


@dataclass
class ModelConfig:
    """ML model configuration."""
    models_dir: str = 'models'
    heart_disease_model: str = 'heart_disease_model.onnx'
    stroke_model: str = 'stroke_model.onnx'
    cancer_model: str = 'cancer_model.onnx'
    medical_image_model: str = 'medical_image_model.onnx'
    confidence_threshold: float = 0.6
    max_upload_size_mb: int = 50
    allowed_extensions: List[str] = field(default_factory=lambda: ['png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx'])

    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create model config from environment variables."""
        return cls(
            models_dir=os.environ.get('MODELS_DIR', 'models'),
            heart_disease_model=os.environ.get('HEART_DISEASE_MODEL', 'heart_disease_model.onnx'),
            stroke_model=os.environ.get('STROKE_MODEL', 'stroke_model.onnx'),
            cancer_model=os.environ.get('CANCER_MODEL', 'cancer_model.onnx'),
            medical_image_model=os.environ.get('MEDICAL_IMAGE_MODEL', 'medical_image_model.onnx'),
            confidence_threshold=float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', '0.6')),
            max_upload_size_mb=_get_int('MAX_UPLOAD_SIZE_MB', 50, 1, 500),
            allowed_extensions=_get_list('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif,pdf,txt,docx')
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = 'INFO'
    format: str = 'json'  # 'json' or 'text'
    log_file: Optional[str] = None
    log_dir: str = 'logs'
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create logging config from environment variables."""
        return cls(
            level=os.environ.get('LOG_LEVEL', 'INFO'),
            format=os.environ.get('LOG_FORMAT', 'json'),
            log_file=os.environ.get('LOG_FILE'),
            log_dir=os.environ.get('LOG_DIR', 'logs'),
            max_bytes=_get_int('LOG_MAX_BYTES', 10485760, 1048576, 104857600),
            backup_count=_get_int('LOG_BACKUP_COUNT', 5, 1, 50)
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    # Environment
    env: str = 'development'
    debug: bool = False
    testing: bool = False

    # Server
    host: str = '0.0.0.0'
    port: int = 5000

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig.from_env)
    security: SecurityConfig = field(default_factory=SecurityConfig.from_env)
    redis: RedisConfig = field(default_factory=RedisConfig.from_env)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig.from_env)
    api: APIConfig = field(default_factory=APIConfig.from_env)
    models: ModelConfig = field(default_factory=ModelConfig.from_env)
    logging: LoggingConfig = field(default_factory=LoggingConfig.from_env)

    # Paths
    base_dir: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    uploads_dir: str = 'uploads'

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create full app config from environment variables."""
        env = os.environ.get('FLASK_ENV', 'development')

        if env not in ['development', 'testing', 'production']:
            raise ConfigValidationError(f"FLASK_ENV must be one of: development, testing, production. Got: {env}")

        return cls(
            env=env,
            debug=env == 'development',
            testing=env == 'testing',
            host=os.environ.get('APP_HOST', '0.0.0.0'),
            port=_get_int('APP_PORT', 5000, 1024, 65535),
            uploads_dir=os.environ.get('UPLOADS_DIR', 'uploads')
        )

    def validate(self) -> None:
        """Validate configuration and raise errors if invalid."""
        # Validate secret key in production
        if self.env == 'production' and self.security.secret_key == 'dev-key-please-change-in-production':
            raise ConfigValidationError('FLASK_SECRET_KEY must be set in production')

        provider = self.api.llm_provider.strip().lower()
        if provider not in {'google', 'z.ai'}:
            raise ConfigValidationError("LLM_PROVIDER must be either 'google' or 'z.ai'")

        # Validate active LLM provider credentials
        if provider == 'google' and not self.api.google_api_key:
            logger.warning('GOOGLE_API_KEY not set while LLM_PROVIDER=google - AI features will be limited')
        if provider == 'z.ai' and not self.api.zai_api_key:
            logger.warning('ZAI_API_KEY not set while LLM_PROVIDER=z.ai - AI features will be limited')

        # Validate database connection
        postgres_default = (not self.database.password or self.database.password == 'password')
        mysql_default = (not self.database.mysql_password or self.database.mysql_password == 'password')
        if postgres_default and mysql_default:
            logger.warning('Using default PostgreSQL and MySQL passwords - not recommended for production')


@lru_cache
def get_config() -> AppConfig:
    """Get cached application configuration."""
    return AppConfig.from_env()


def init_app(app, config: Optional[AppConfig] = None) -> None:
    """
    Initialize Flask application with configuration.

    Args:
        app: Flask application instance
        config: Application configuration (uses env if not provided)
    """
    if config is None:
        config = get_config()

    # Flask basic config
    app.config['SECRET_KEY'] = config.security.secret_key
    resolved_uri, resolved_backend = config.database.resolve_uri()
    app.config['SQLALCHEMY_DATABASE_URI'] = resolved_uri
    app.config['DATABASE_TYPE'] = resolved_backend
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = config.database.echo
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': config.database.pool_size,
        'max_overflow': config.database.max_overflow,
        'pool_timeout': config.database.pool_timeout,
        'pool_recycle': config.database.pool_recycle,
    }

    # Session config
    app.config['SESSION_COOKIE_SECURE'] = config.security.session_secure
    app.config['SESSION_COOKIE_HTTPONLY'] = config.security.session_httponly
    app.config['SESSION_COOKIE_SAMESITE'] = config.security.session_samesite
    app.config['PERMANENT_SESSION_LIFETIME'] = config.security.session_lifetime_hours * 3600
    app.config['SESSION_REFRESH_EACH_REQUEST'] = True

    # CSRF config
    if config.security.csrf_enabled:
        app.config['WTF_CSRF_TIME_LIMIT'] = config.security.csrf_time_limit
        app.config['WTF_CSRF_SSL_STRICT'] = True

    # Upload config
    app.config['MAX_CONTENT_LENGTH'] = config.models.max_upload_size_mb * 1024 * 1024

    # Logging level
    app.logger.setLevel(getattr(logging, config.logging.level))

    # Validate configuration
    if not config.testing:
        config.validate()


__all__ = [
    'AppConfig',
    'DatabaseConfig',
    'SecurityConfig',
    'RedisConfig',
    'RateLimitConfig',
    'APIConfig',
    'ModelConfig',
    'LoggingConfig',
    'get_config',
    'init_app',
]
