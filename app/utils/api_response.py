"""
API Response Wrapper Utility for Health Navigator
Provides consistent response format for all API endpoints.
"""

from typing import Any, Dict, Optional, Tuple
from flask import jsonify, Response
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class APIResponse:
    """
    Utility class for creating consistent API responses.
    All responses follow a standard format with status, data, and metadata.
    """

    @staticmethod
    def success(
        data: Any,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status_code: int = 200
    ) -> Tuple[Response, int]:
        """
        Create a successful API response.

        Args:
            data: The response data (can be dict, list, str, etc.)
            message: Optional success message
            metadata: Optional metadata dictionary (timing, pagination, etc.)
            status_code: HTTP status code (default: 200)

        Returns:
            Tuple of (flask Response, status_code)
        """
        response = {
            "status": "success",
            "data": data
        }

        if message:
            response["message"] = message

        if metadata:
            response["metadata"] = metadata

        logger.debug(f"API success response", extra={"status_code": status_code})
        return jsonify(response), status_code

    @staticmethod
    def error(
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400
    ) -> Tuple[Response, int]:
        """
        Create an error API response.

        Args:
            code: Error code (e.g., 'VALIDATION_ERROR', 'NOT_FOUND')
            message: Human-readable error message
            details: Optional additional error details
            status_code: HTTP status code (default: 400)

        Returns:
            Tuple of (flask Response, status_code)
        """
        error_response = {
            "status": "error",
            "error": {
                "code": code,
                "message": message
            }
        }

        if details:
            error_response["error"]["details"] = details

        logger.warning(f"API error response: {code}", extra={"status_code": status_code, "message": message})
        return jsonify(error_response), status_code

    @staticmethod
    def created(
        data: Any,
        message: str = "Resource created successfully",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Response, int]:
        """Convenience method for 201 Created responses."""
        return APIResponse.success(data, message, metadata, 201)

    @staticmethod
    def no_content(message: str = "Operation successful") -> Tuple[Response, int]:
        """Convenience method for 204 No Content responses."""
        return jsonify({"status": "success", "message": message}), 204

    @staticmethod
    def not_found(
        resource: str = "Resource",
        details: Optional[Dict[str, Any]] = None
    ) -> Tuple[Response, int]:
        """Convenience method for 404 Not Found responses."""
        return APIResponse.error(
            code="NOT_FOUND",
            message=f"{resource} not found",
            details=details,
            status_code=404
        )

    @staticmethod
    def unauthorized(
        message: str = "Authentication required",
        details: Optional[Dict[str, Any]] = None
    ) -> Tuple[Response, int]:
        """Convenience method for 401 Unauthorized responses."""
        return APIResponse.error(
            code="UNAUTHORIZED",
            message=message,
            details=details,
            status_code=401
        )

    @staticmethod
    def forbidden(
        message: str = "Access denied",
        details: Optional[Dict[str, Any]] = None
    ) -> Tuple[Response, int]:
        """Convenience method for 403 Forbidden responses."""
        return APIResponse.error(
            code="FORBIDDEN",
            message=message,
            details=details,
            status_code=403
        )

    @staticmethod
    def validation_error(
        message: str,
        errors: Optional[Dict[str, list]] = None
    ) -> Tuple[Response, int]:
        """
        Create a validation error response.

        Args:
            message: General validation error message
            errors: Dict of field-specific validation errors

        Returns:
            Tuple of (flask Response, 400)
        """
        details = None
        if errors:
            details = {"fields": errors}

        return APIResponse.error(
            code="VALIDATION_ERROR",
            message=message,
            details=details,
            status_code=400
        )

    @staticmethod
    def rate_limited(
        message: str = "Too many requests",
        retry_after: Optional[int] = None
    ) -> Tuple[Response, int]:
        """
        Create a rate limit exceeded response.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying

        Returns:
            Tuple of (flask Response, 429)
        """
        details = None
        if retry_after:
            details = {"retry_after": retry_after}

        return APIResponse.error(
            code="RATE_LIMIT_EXCEEDED",
            message=message,
            details=details,
            status_code=429
        )

    @staticmethod
    def server_error(
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None
    ) -> Tuple[Response, int]:
        """Convenience method for 500 Internal Server Error responses."""
        return APIResponse.error(
            code="INTERNAL_ERROR",
            message=message,
            details=details,
            status_code=500
        )


# Error codes constant class for reference
class ErrorCode:
    """Standard error codes used throughout the application."""

    # Authentication & Authorization
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    SESSION_EXPIRED = "SESSION_EXPIRED"

    # Validation
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"

    # Resources
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    CONFLICT = "CONFLICT"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Workflow & Processing
    WORKFLOW_ERROR = "WORKFLOW_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    INTERRUPT_ERROR = "INTERRUPT_ERROR"

    # File Handling
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_TYPE_NOT_ALLOWED = "FILE_TYPE_NOT_ALLOWED"
    FILE_VALIDATION_FAILED = "FILE_VALIDATION_FAILED"

    # Server
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


def with_error_handling(func):
    """
    Decorator for API routes that provides consistent error handling.
    Catches exceptions and returns appropriate error responses.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            return APIResponse.validation_error(str(e))
        except PermissionError as e:
            logger.warning(f"Permission error in {func.__name__}: {e}")
            return APIResponse.forbidden(str(e))
        except FileNotFoundError as e:
            logger.warning(f"Not found in {func.__name__}: {e}")
            return APIResponse.not_found(str(e))
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return APIResponse.server_error()

    # Preserve function name and docstring
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__

    return wrapper


__all__ = [
    "APIResponse",
    "ErrorCode",
    "with_error_handling",
]
