"""
Health Navigator - Audit & Compliance Models
Tracking user actions, data access, and regulatory compliance
"""

from datetime import datetime
from sqlalchemy.sql import expression
from app import db
import logging

logger = logging.getLogger(__name__)


class AuditLog(db.Model):
    """
    Audit log for tracking user actions and data access.
    Supports compliance with HIPAA, GDPR, and other regulatory requirements.
    """
    __tablename__ = 'audit_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    action = db.Column(db.String(100), nullable=False, index=True)
    resource_type = db.Column(db.String(50))  # 'conversation', 'profile', 'prediction', 'attachment'
    resource_id = db.Column(db.Integer)
    ip_address = db.Column(db.String(45))  # IPv6 can be up to 45 chars
    user_agent = db.Column(db.String(500))
    status = db.Column(db.String(20), default='success')  # success, failure, attempted
    details = db.Column(db.JSON)  # Additional context
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship
    user = db.relationship('User', backref=db.backref('audit_logs', lazy='dynamic'))

    def __repr__(self):
        return f'<AuditLog {self.id}: {self.action} by User {self.user_id} at {self.timestamp}>'

    @classmethod
    def log_action(
        cls,
        action: str,
        user_id: int = None,
        resource_type: str = None,
        resource_id: int = None,
        ip_address: str = None,
        user_agent: str = None,
        status: str = 'success',
        details: dict = None
    ) -> 'AuditLog':
        """
        Log an action to the audit log.

        Args:
            action: Action performed (e.g., 'login', 'view_conversation', 'run_prediction')
            user_id: User who performed the action
            resource_type: Type of resource affected
            resource_id: ID of the resource
            ip_address: IP address of the request
            user_agent: User agent string
            status: Status of the action
            details: Additional context as JSON

        Returns:
            Created AuditLog entry
        """
        try:
            log_entry = cls(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                status=status,
                details=details
            )
            db.session.add(log_entry)
            db.session.commit()
            logger.info(f"Audit log: {action} by user {user_id}")
            return log_entry
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            db.session.rollback()
            return None

    def to_dict(self) -> dict:
        """Convert audit log to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'ip_address': self.ip_address,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'status': self.status,
            'details': self.details
        }


class Consent(db.Model):
    """
    User consent tracking for GDPR/privacy compliance.
    Tracks user consent for data processing and communications.
    """
    __tablename__ = 'consents'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True, index=True)
    consent_type = db.Column(db.String(50), nullable=False, index=True)  # 'data_processing', 'marketing', 'analytics'
    granted = db.Column(db.Boolean, default=False, nullable=False)
    granted_at = db.Column(db.DateTime, nullable=True)
    revoked_at = db.Column(db.DateTime)
    version = db.Column(db.String(20), default='1.0')  # For tracking consent version updates
    ip_address = db.Column(db.String(45))
    consent_metadata = db.Column(db.JSON)  # Additional consent context

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = db.relationship('User', backref=db.backref('consents', lazy='dynamic'))

    def __repr__(self):
        return f'<Consent {self.consent_type} for user {self.user_id}: {self.granted}>'

    def to_dict(self) -> dict:
        """Convert consent to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'consent_type': self.consent_type,
            'granted': self.granted,
            'granted_at': self.granted_at.isoformat() if self.granted_at else None,
            'revoked_at': self.revoked_at.isoformat() if self.revoked_at else None,
            'version': self.version
        }


class DataRetentionPolicy(db.Model):
    """
    Data retention policies for compliance with data protection regulations.
    Defines how long different types of data should be retained.
    """
    __tablename__ = 'data_retention_policies'

    id = db.Column(db.Integer, primary_key=True)
    data_type = db.Column(db.String(50), unique=True, nullable=False)  # 'conversation', 'attachment', 'profile'
    retention_period_days = db.Column(db.Integer, nullable=False)
    deletion_action = db.Column(db.String(50), default='hard_delete')  # 'hard_delete', 'anonymize', 'archive'
    policy_description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<DataRetentionPolicy {self.data_type}: {self.retention_period_days} days>'


def log_user_action(action: str, **kwargs) -> AuditLog:
    """
    Convenience function to log user actions from anywhere in the app.
    Can be imported and used in routes, workflows, and other modules.

    Args:
        action: Action identifier
        **kwargs: Additional arguments passed to AuditLog.log_action

    Returns:
        Created AuditLog entry or None if logging failed
    """
    return AuditLog.log_action(action, **kwargs)


# Consent type constants
class ConsentType:
    DATA_PROCESSING = 'data_processing'      # Main consent for processing personal data
    MARKETING = 'marketing'                  # Consent for marketing communications
    ANALYTICS = 'analytics'                    # Consent for analytics tracking
    COOKIES = 'cookies'                      # Consent for non-essential cookies
    MEDICAL_ANALYSIS = 'medical_analysis'    # Specific consent for AI medical analysis


# Action type constants for audit logging
class ActionType:
    # Authentication
    LOGIN = 'login'
    LOGOUT = 'logout'
    REGISTER = 'register'
    PASSWORD_CHANGE = 'password_change'

    # Profile
    PROFILE_VIEW = 'profile_view'
    PROFILE_UPDATE = 'profile_update'
    PROFILE_DELETE = 'profile_delete'

    # Conversations
    CONVERSATION_CREATE = 'conversation_create'
    CONVERSATION_VIEW = 'conversation_view'
    CONVERSATION_EXPORT = 'conversation_export'
    CONVERSATION_DELETE = 'conversation_delete'

    # Messages
    MESSAGE_SEND = 'message_send'
    MESSAGE_RECEIVE = 'message_receive'

    # Attachments
    ATTACHMENT_UPLOAD = 'attachment_upload'
    ATTACHMENT_DOWNLOAD = 'attachment_download'
    ATTACHMENT_DELETE = 'attachment_delete'

    # Predictions
    PREDICTION_RUN = 'prediction_run'
    PREDICTION_VIEW = 'prediction_view'

    # Data Management
    DATA_EXPORT = 'data_export'
    DATA_DELETE_REQUEST = 'data_delete_request'

    # Admin
    USER_VIEW = 'user_view'
    USER_MODIFY = 'user_modify'
    USER_DELETE = 'user_delete'


__all__ = [
    'AuditLog',
    'Consent',
    'DataRetentionPolicy',
    'log_user_action',
    'ConsentType',
    'ActionType',
]
