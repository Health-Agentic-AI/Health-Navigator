"""
Structured Logging Configuration for Health Navigator
Provides JSON-formatted logging with correlation IDs for request tracing.
"""

import logging
import logging.handlers
import json
import os
import uuid
import time
from datetime import datetime
from typing import Any, Dict
from flask import request, g
from functools import wraps


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter that outputs logs as structured JSON.
    Includes timestamp, level, logger name, message, and extra context.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.
        """
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the log record
        # These are added via logger.info("msg", extra={"key": "value"})
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "exc_info", "exc_text", "stack_info",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class RequestIdFilter(logging.Filter):
    """
    Logging filter that adds a request_id to each log record.
    Uses Flask's g object to store the request ID for the duration of the request.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request_id to the log record if available in Flask's g object.
        """
        try:
            from flask import g
            if hasattr(g, "request_id"):
                record.request_id = g.request_id
            if hasattr(g, "user_id"):
                record.user_id = g.user_id
        except (ImportError, RuntimeError):
            # Flask context not available
            pass
        return True


def setup_logging(app: Any) -> None:
    """
    Configure structured logging for the Flask application.

    Args:
        app: Flask application instance
    """
    # Determine log level from environment
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatters
    json_formatter = JSONFormatter()

    # Console handler (stdout) - for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(json_formatter)
    console_handler.addFilter(RequestIdFilter())
    root_logger.addHandler(console_handler)

    # File handler - for production
    # Rotate logs daily, keep 30 days
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "health_navigator.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(json_formatter)
    file_handler.addFilter(RequestIdFilter())
    root_logger.addHandler(file_handler)

    # Error file handler - separate file for errors
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "health_navigator_errors.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    error_handler.addFilter(RequestIdFilter())
    root_logger.addHandler(error_handler)

    # Set specific logger levels
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Log startup message
    app.logger.info("Health Navigator logging initialized", extra={"log_level": log_level})


def add_request_id_middleware(app: Any) -> None:
    """
    Add middleware to generate and store a unique request ID for each request.
    Also tracks user_id if authenticated.

    Args:
        app: Flask application instance
    """

    @app.before_request
    def before_request():
        """Generate request ID before processing each request."""
        # Generate unique request ID
        g.request_id = str(uuid.uuid4())

        # Track user ID if authenticated
        if "session" in app.__dict__:
            # Check if user is authenticated
            try:
                user_id = session.get("user_id") if session else None
                if user_id:
                    g.user_id = user_id
            except:
                pass

        # Store request start time for duration tracking
        g.start_time = time.time()

    @app.after_request
    def after_request(response):
        """Add request ID header and log request completion."""
        # Add request ID to response headers for debugging
        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id

        # Log request completion
        if hasattr(g, "start_time") and hasattr(g, "request_id"):
            duration = time.time() - g.start_time
            app.logger.info(
                "Request completed",
                extra={
                    "request_id": g.request_id,
                    "method": request.method,
                    "path": request.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "ip_address": request.remote_addr,
                    "user_agent": request.user_agent.string[:200] if request.user_agent else None,
                }
            )

        return response


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for logging with request context
def log_with_context(func):
    """
    Decorator that ensures logging context is available.
    Useful for background tasks or non-request contexts.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate a request ID for this task
        request_id = str(uuid.uuid4())

        # Add to logging context
        extra = {"request_id": request_id}
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Starting task: {func.__name__}", extra=extra)

        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed task: {func.__name__}", extra=extra)
            return result
        except Exception as e:
            logger.error(f"Task failed: {func.__name__}", exc_info=True, extra=extra)
            raise

    return wrapper


# Export the main classes and functions
__all__ = [
    "JSONFormatter",
    "RequestIdFilter",
    "setup_logging",
    "add_request_id_middleware",
    "get_logger",
    "log_with_context",
]
