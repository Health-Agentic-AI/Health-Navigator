from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, current_app, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from app import db, limiter
from app.models import User, Conversation, PatientProfile, Allergy, Medication, PastMedicalHistory, PastSurgery, FamilyHistory, Message, Attachment
from app.compliance import AuditLog, Consent, log_user_action, ConsentType, ActionType
from app.workflow.workflow import app as workflow_app
from app.utils.api_response import APIResponse, ErrorCode
from langgraph.types import Command
import uuid
import os
import json
import logging
import filetype
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from langgraph.errors import GraphInterrupt
from datetime import datetime, timedelta
from sqlalchemy import text
from io import BytesIO

# Configure logging
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

# Rate limiter instance for this blueprint
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.environ.get("REDIS_URL", "memory://")
)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx'}

# Allowed MIME types for additional security
ALLOWED_MIME_TYPES = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/gif': 'gif',
    'application/pdf': 'pdf',
    'text/plain': 'txt',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
}

# --- Helper Functions ---

def get_current_user():
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_type(file_stream, filename):
    """
    Validate file type using magic numbers (file content) in addition to extension.
    This prevents users from disguising malicious files with allowed extensions.

    Args:
        file_stream: File object to validate
        filename: Original filename

    Returns:
        tuple: (is_valid, error_message, detected_mime_type)
    """
    try:
        # Check extension first
        if not allowed_file(filename):
            return False, f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}", None

        # Get file info from magic number detection
        file_stream.seek(0)
        header = file_stream.read(261)  # Read first 261 bytes for magic number detection
        file_stream.seek(0)

        kind = filetype.guess(header)

        if kind is None:
            # Couldn't determine type, fall back to extension check only
            logger.warning(f"Could not detect file type for {filename}, allowing based on extension")
            return True, None, None

        detected_mime = kind.mime
        detected_extension = kind.extension

        # Check if detected MIME type is allowed
        if detected_mime not in ALLOWED_MIME_TYPES:
            return False, f"File content type ({detected_mime}) not allowed. File may be corrupted or renamed.", detected_mime

        # Check if detected extension matches the claimed extension
        claimed_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if detected_extension != claimed_extension:
            logger.warning(f"Extension mismatch for {filename}: claimed '{claimed_extension}', detected '{detected_extension}'")
            # Allow common variations (e.g., jpg/jpeg)
            if {detected_extension, claimed_extension} == {'jpg', 'jpeg'}:
                return True, None, detected_mime
            return False, f"File extension '{claimed_extension}' doesn't match actual file type '{detected_extension}'", detected_mime

        return True, None, detected_mime

    except Exception as e:
        logger.error(f"Error validating file type: {e}", exc_info=True)
        return False, "Error validating file type", None

# --- Routes ---

@main_bp.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('main.chat'))
    return redirect(url_for('main.login'))

@main_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter((User.username == username) | (User.email == email)).first():
            return render_template('register.html', error="Username or Email already exists")

        hashed_password = generate_password_hash(password)
        new_user = User(full_name=full_name, username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        # Create empty profile
        profile = PatientProfile(user_id=new_user.id)
        db.session.add(profile)
        db.session.commit()

        session['user_id'] = new_user.id
        return redirect(url_for('main.chat'))

    return render_template('register.html')

@main_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            return redirect(url_for('main.chat'))

        return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

@main_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('main.login'))

@main_bp.route('/chat')
def chat():
    user = get_current_user()
    if not user:
        return redirect(url_for('main.login'))

    # Get recent conversations
    conversations = Conversation.query.filter_by(user_id=user.id).order_by(Conversation.last_updated_at.desc()).all()
    return render_template('chat.html', user=user, conversations=conversations)

@main_bp.route('/api/conversations', methods=['POST'])
@limiter.limit("20 per minute")
def create_conversation():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    title = data.get('title', 'New Consultation')

    conversation = Conversation(
        user_id=user.id,
        title=title
        # messages will be initialized as empty via default or just not passed
    )
    db.session.add(conversation)
    db.session.commit()

    return jsonify({'id': conversation.id, 'title': conversation.title})

@main_bp.route('/api/conversations/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    conversation = Conversation.query.get_or_404(conversation_id)
    if conversation.user_id != user.id:
        return jsonify({'error': 'Forbidden'}), 403

    # Manually serialize messages
    messages_list = []
    for msg in conversation.messages:
        messages_list.append({
            'role': msg.sender_type,
            'content': msg.content,
            'timestamp': str(msg.created_at),
            'attachments': [a.original_name for a in msg.attachments]
        })

    return jsonify({
        'id': conversation.id,
        'title': conversation.title,
        'messages': messages_list
    })

@main_bp.route('/api/chat/message', methods=['POST'])
@limiter.limit("20 per minute")
def send_message():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    # Handle file uploads
    uploaded_files = {} # filename -> path

    # Get all files with the key 'files' (from frontend FormData)
    files = request.files.getlist('files')

    if files:
        if len(files) > 20:
             return jsonify({'error': 'Too many files. Maximum 20 allowed.'}), 400

        upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', str(user.id))
        os.makedirs(upload_folder, exist_ok=True)

        for file in files:
            # Skip empty file inputs
            if not file or not file.filename:
                continue

            if allowed_file(file.filename):
                # Validate file type using magic numbers
                is_valid, error_msg, detected_mime = validate_file_type(file.stream, file.filename)

                if not is_valid:
                    logger.warning(f"File validation failed: {error_msg}", extra={"filename": file.filename, "detected_mime": detected_mime})
                    return jsonify({'error': error_msg}), 400

                # Check file size safely
                file.seek(0, os.SEEK_END)
                file_length = file.tell()
                file.seek(0)

                if file_length > 10 * 1024 * 1024:
                    logger.warning(f"File too large: {file.filename} ({file_length} bytes)", extra={"max_size": "10MB"})
                    return jsonify({'error': f'File {file.filename} is too large. Max 10MB.'}), 400

                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                uploaded_files[filename] = file_path

                logger.info(f"File uploaded successfully", extra={"filename": filename, "user_id": user.id})
            else:
                logger.warning(f"File extension not allowed: {file.filename}")
                return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Extract message and conversation_id
    if request.content_type.startswith('multipart/form-data'):
        message = request.form.get('message', '')
        conversation_id = request.form.get('conversation_id')
    else:
        data = request.json
        message = data.get('message', '')
        conversation_id = data.get('conversation_id')

    if not conversation_id:
        conversation = Conversation(user_id=user.id, title=message[:30] + "...")
        db.session.add(conversation)
        db.session.flush() # Get ID
        conversation_id = conversation.id
    else:
        conversation = Conversation.query.get(conversation_id)
        # Security Check: Ensure the conversation belongs to the logged-in user
        if not conversation or conversation.user_id != user.id:
            return jsonify({'error': 'Forbidden: You do not own this conversation'}), 403

    # Create User Message
    user_msg = Message(
        conversation_id=conversation_id,
        sender_type='user',
        content=message
    )
    db.session.add(user_msg)
    db.session.flush() # Get ID for attachments

    # Create Attachments
    for filename, filepath in uploaded_files.items():
        attachment = Attachment(
            message_id=user_msg.id,
            file_path=filepath,
            original_name=filename,
            file_type=filename.split('.')[-1].lower()
        )
        db.session.add(attachment)

    db.session.commit()

    # --- WORKFLOW EXECUTION ---
    thread_id = str(conversation_id)
    config = {"configurable": {"thread_id": thread_id}}

    # Check current state
    current_state_snapshot = workflow_app.get_state(config)

    # Check if we're resuming from an interrupt
    if current_state_snapshot.next:
        # We have pending tasks - we're resuming from an interrupt
        logger.info(f"Resuming from interrupt", extra={"conversation_id": conversation_id, "user_id": user.id})
        try:
            # Resume with the user's message
            result = workflow_app.invoke(Command(resume=message), config=config)

            # Check if finished or interrupted again
            final_state = workflow_app.get_state(config)

            if not final_state.next:
                # Workflow completed
                final_output = result.get("final_refined_medical_output", "Analysis complete.")

                assistant_msg = Message(
                    conversation_id=conversation_id,
                    sender_type='assistant',
                    content=final_output
                )
                db.session.add(assistant_msg)
                db.session.commit()

                logger.info("Workflow completed successfully", extra={"conversation_id": conversation_id})

                return jsonify({
                    'status': 'completed',
                    'response': final_output,
                    'conversation_id': conversation.id
                })
            else:
                # Interrupted again - get new question
                if final_state.tasks and final_state.tasks[0].interrupts:
                    interrupt_value = final_state.tasks[0].interrupts[0].value

                    system_msg = Message(
                        conversation_id=conversation_id,
                        sender_type='system_question',
                        content=interrupt_value
                    )
                    db.session.add(system_msg)
                    db.session.commit()

                    logger.info("Workflow interrupted again", extra={"conversation_id": conversation_id})

                    return jsonify({
                        'status': 'interrupted',
                        'question': interrupt_value,
                        'conversation_id': conversation.id
                    })

        except GraphInterrupt as e:
            # Workflow was interrupted
            logger.debug(f"GraphInterrupt caught during resume", extra={"conversation_id": conversation_id})
            final_state = workflow_app.get_state(config)

            if final_state.tasks and final_state.tasks[0].interrupts:
                interrupt_value = final_state.tasks[0].interrupts[0].value

                system_msg = Message(
                    conversation_id=conversation_id,
                    sender_type='system_question',
                    content=interrupt_value
                )
                db.session.add(system_msg)
                db.session.commit()

                return jsonify({
                    'status': 'interrupted',
                    'question': interrupt_value,
                    'conversation_id': conversation.id
                })

        except Exception as e:
            logger.error(f"Unexpected error during resume", exc_info=True, extra={"conversation_id": conversation_id, "user_id": user.id})
            return jsonify({'error': 'Workflow error during resume'}), 500

    else:
        logger.info(f"Starting new workflow", extra={"conversation_id": conversation_id, "user_id": user.id})
        # Start new workflow run
        initial_state = {
            "input_prompt": message,
            "attachments": uploaded_files,
            "user_id": str(user.id),
            "thread_id": thread_id,
            "conversation_history": []
        }

        try:
            result = workflow_app.invoke(initial_state, config=config)

            # Check if completed or interrupted
            final_state = workflow_app.get_state(config)

            if not final_state.next:
                # Completed successfully
                final_output = result.get("final_refined_medical_output", "I could not process that request.")

                assistant_msg = Message(
                    conversation_id=conversation_id,
                    sender_type='assistant',
                    content=final_output
                )
                db.session.add(assistant_msg)
                db.session.commit()

                logger.info("New workflow completed successfully", extra={"conversation_id": conversation_id})

                return jsonify({
                    'status': 'completed',
                    'response': final_output,
                    'conversation_id': conversation.id
                })
            else:
                # Workflow was interrupted - ask user for info
                logger.debug(f"Workflow interrupted, checking for interrupt value", extra={"conversation_id": conversation_id})

                # Try different ways to get interrupt value
                interrupt_value = None

                if hasattr(final_state, 'tasks') and final_state.tasks:
                    task = final_state.tasks[0]
                    task_interrupts = getattr(task, 'interrupts', None)
                    logger.debug(f"Checking task interrupts", extra={"has_interrupts": task_interrupts is not None})

                    if hasattr(task, 'interrupts') and task.interrupts:
                        interrupt_value = task.interrupts[0].value

                if interrupt_value:
                    system_msg = Message(
                        conversation_id=conversation_id,
                        sender_type='system_question',
                        content=interrupt_value
                    )
                    db.session.add(system_msg)
                    db.session.commit()

                    logger.info("New workflow interrupted, requesting user input", extra={"conversation_id": conversation_id})

                    return jsonify({
                        'status': 'interrupted',
                        'question': interrupt_value,
                        'conversation_id': conversation.id
                    })
                else:
                    logger.error("Interrupt detected but couldn't extract question", extra={"conversation_id": conversation_id})
                    return jsonify({'error': 'Interrupt handling error'}), 500

        except GraphInterrupt as e:
            # This should not happen with invoke() - it returns state instead
            # But handle it just in case
            logger.debug(f"GraphInterrupt caught during initial invoke", extra={"conversation_id": conversation_id})
            final_state = workflow_app.get_state(config)

            if final_state.tasks and final_state.tasks[0].interrupts:
                interrupt_value = final_state.tasks[0].interrupts[0].value

                system_msg = Message(
                    conversation_id=conversation_id,
                    sender_type='system_question',
                    content=interrupt_value
                )
                db.session.add(system_msg)
                db.session.commit()

                return jsonify({
                    'status': 'interrupted',
                    'question': interrupt_value,
                    'conversation_id': conversation.id
                })

        except Exception as e:
            logger.error(f"Unexpected error during workflow", exc_info=True, extra={"conversation_id": conversation_id, "user_id": user.id})
            return jsonify({'error': 'Workflow execution error'}), 500

    return jsonify({'error': 'Unknown workflow state'}), 500


# --- Health Check Endpoints ---

@main_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    Returns the overall status of the application.
    """
    health_status = {
        "status": "healthy",
        "service": "health-navigator",
        "version": os.environ.get("APP_VERSION", "2.0.0"),
        "timestamp": datetime.utcnow().isoformat()
    }
    return jsonify(health_status), 200


@main_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    Detailed health check endpoint.
    Returns the status of all major components.
    """
    health_info = {
        "status": "healthy",
        "service": "health-navigator",
        "version": os.environ.get("APP_VERSION", "2.0.0"),
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }

    # Check database connection
    try:
        db.session.execute(text('SELECT 1'))
        health_info["components"]["database"] = {
            "status": "healthy",
            "type": "postgresql"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_info["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_info["status"] = "degraded"

    # Check vector DB (ChromaDB)
    try:
        from app.workflow.vectordb.vector_db_manager import get_vector_db
        # Just verify we can import and initialize
        health_info["components"]["vector_db"] = {
            "status": "healthy",
            "type": "chromadb"
        }
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        health_info["components"]["vector_db"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_info["status"] = "degraded"

    # Check ML models directory
    try:
        import os
        models_dir = os.path.join(os.path.dirname(__file__), 'workflow', 'ml_models')
        if os.path.exists(models_dir):
            health_info["components"]["ml_models"] = {
                "status": "healthy",
                "path": models_dir
            }
        else:
            health_info["components"]["ml_models"] = {
                "status": "unhealthy",
                "error": "Models directory not found"
            }
            health_info["status"] = "degraded"
    except Exception as e:
        logger.error(f"ML models health check failed: {e}")
        health_info["components"]["ml_models"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_info["status"] = "degraded"

    # Determine overall HTTP status code
    status_code = 200 if health_info["status"] == "healthy" else 503
    return jsonify(health_info), status_code


@main_bp.route('/health/models', methods=['GET'])
def models_health_check():
    """
    Health check for ML models specifically.
    Returns the status of all ML model components.
    """
    models_info = {
        "status": "unknown",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {}
    }

    all_healthy = True

    # Check Heart Disease model
    try:
        from app.workflow.ml_models.numerical_models.heart_disease.heart_disease import _ensure_model_loaded
        _ensure_model_loaded()
        models_info["models"]["heart_disease"] = {"status": "healthy"}
    except Exception as e:
        models_info["models"]["heart_disease"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    # Check Stroke model
    try:
        from app.workflow.ml_models.numerical_models.stroke_prediction.stroke_predictions import _ensure_model_loaded
        _ensure_model_loaded()
        models_info["models"]["stroke"] = {"status": "healthy"}
    except Exception as e:
        models_info["models"]["stroke"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    # Check Cancer model
    try:
        from app.workflow.ml_models.numerical_models.cancer_predictions_module.Cancer_prediction import _ensure_model_loaded
        _ensure_model_loaded()
        models_info["models"]["cancer"] = {"status": "healthy"}
    except Exception as e:
        models_info["models"]["cancer"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    # Check Chest X-Ray model
    try:
        from app.workflow.ml_models.vision_models.chest_xray.chest_xray import _ensure_model_loaded
        _ensure_model_loaded()
        models_info["models"]["chest_xray"] = {"status": "healthy"}
    except Exception as e:
        models_info["models"]["chest_xray"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    # Check Colon Tissue model
    try:
        from app.workflow.ml_models.vision_models.colon_tissue_classifier.colon import _ensure_model_loaded
        _ensure_model_loaded()
        models_info["models"]["colon_tissue"] = {"status": "healthy"}
    except Exception as e:
        models_info["models"]["colon_tissue"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False

    models_info["status"] = "healthy" if all_healthy else "degraded"
    status_code = 200 if all_healthy else 503

    return jsonify(models_info), status_code


@main_bp.route('/health/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check endpoint for Kubernetes/container orchestration.
    Returns whether the service is ready to accept traffic.
    """
    try:
        # Check database connection
        db.session.execute(text('SELECT 1'))

        # Basic readiness check passed
        return jsonify({
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            "status": "not_ready",
            "reason": str(e)
        }), 503


@main_bp.route('/health/live', methods=['GET'])
def liveness_check():
    """
    Liveness check endpoint for Kubernetes/container orchestration.
    Returns whether the service is alive (basic check).
    """
    return jsonify({
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@main_bp.route('/docs', methods=['GET'])
def api_docs():
    """
    API documentation page.
    Renders the OpenAPI/Swagger documentation.
    """
    return render_template('api_docs.html')


@main_bp.route('/api/openapi.yaml', methods=['GET'])
def openapi_spec():
    """
    Serve the OpenAPI specification as YAML.
    """
    import yaml
    try:
        openapi_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'openapi.yaml')
        with open(openapi_path, 'r') as f:
            spec = yaml.safe_load(f)
        return jsonify(spec), 200
    except FileNotFoundError:
        return jsonify({"error": "OpenAPI specification not found"}), 404
    except Exception as e:
        logger.error(f"Error loading OpenAPI spec: {e}")
        return jsonify({"error": "Failed to load specification"}), 500


# ==============================
# COMPLIANCE & GDPR ENDPOINTS
# ==============================

@main_bp.route('/api/user/consent', methods=['GET', 'POST'])
def manage_consent():
    """
    Get or update user consent preferences.
    GET: Returns all consents for current user.
    POST: Updates consent preferences.
    """
    if 'user_id' not in session:
        return APIResponse.unauthorized("Authentication required")

    user_id = session['user_id']

    if request.method == 'GET':
        # Return all consents for the user
        consents = Consent.query.filter_by(user_id=user_id).all()
        return APIResponse.success({
            'consents': [c.to_dict() for c in consents]
        })

    elif request.method == 'POST':
        data = request.get_json()
        consent_type = data.get('consent_type')
        granted = data.get('granted', False)

        if not consent_type:
            return APIResponse.validation_error("consent_type is required")

        # Validate consent type
        valid_types = [
            ConsentType.DATA_PROCESSING,
            ConsentType.MARKETING,
            ConsentType.ANALYTICS,
            ConsentType.COOKIES,
            ConsentType.MEDICAL_ANALYSIS
        ]

        if consent_type not in valid_types:
            return APIResponse.validation_error(f"Invalid consent_type. Must be one of: {', '.join(valid_types)}")

        # Find existing consent or create new one
        consent = Consent.query.filter_by(user_id=user_id, consent_type=consent_type).first()

        if granted:
            if consent and consent.granted:
                # Already granted
                return APIResponse.success({'message': 'Consent already granted'})

            # Grant consent
            if consent:
                consent.granted = True
                consent.granted_at = datetime.utcnow()
                consent.revoked_at = None
            else:
                consent = Consent(
                    user_id=user_id,
                    consent_type=consent_type,
                    granted=True,
                    granted_at=datetime.utcnow()
                )
            db.session.add(consent)

            # Log consent action
            log_user_action(
                action='consent_granted',
                user_id=user_id,
                resource_type='consent',
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent'),
                details={'consent_type': consent_type}
            )
        else:
            # Revoke consent
            if consent:
                consent.granted = False
                consent.revoked_at = datetime.utcnow()
            else:
                return APIResponse.not_found('Consent record not found')

            # Log consent revocation
            log_user_action(
                action='consent_revoked',
                user_id=user_id,
                resource_type='consent',
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent'),
                details={'consent_type': consent_type}
            )

        db.session.commit()
        return APIResponse.success({'message': 'Consent preferences updated'})


@main_bp.route('/api/user/data-export', methods=['GET', 'POST'])
def export_user_data():
    """
    Export all user data (GDPR compliance).
    GET: Returns summary of data to be exported.
    POST: Generates and returns the data export.
    """
    if 'user_id' not in session:
        return APIResponse.unauthorized("Authentication required")

    user_id = session['user_id']

    if request.method == 'GET':
        # Return summary of data
        user = User.query.get(user_id)
        if not user:
            return APIResponse.not_found('User not found')

        conversation_count = Conversation.query.filter_by(user_id=user_id).count()
        message_count = Message.query.join(Conversation).filter(Conversation.user_id == user_id).count()
        attachment_count = Attachment.query.join(Message).join(Conversation).filter(Conversation.user_id == user_id).count()

        return APIResponse.success({
            'summary': {
                'email': user.email,
                'username': user.username,
                'full_name': user.full_name,
                'account_created': user.created_at.isoformat() if user.created_at else None,
                'conversation_count': conversation_count,
                'message_count': message_count,
                'attachment_count': attachment_count
            }
        })

    elif request.method == 'POST':
        # Generate full data export
        user = User.query.get(user_id)
        if not user:
            return APIResponse.not_found('User not found')

        # Collect all user data
        export_data = {
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'created_at': user.created_at.isoformat() if user.created_at else None
            },
            'conversations': [],
            'profile': None,
            'attachments': [],
            'export_date': datetime.utcnow().isoformat()
        }

        # Get patient profile
        profile = PatientProfile.query.filter_by(user_id=user_id).first()
        if profile:
            export_data['profile'] = {
                'date_of_birth': profile.date_of_birth.isoformat() if profile.date_of_birth else None,
                'gender': profile.gender,
                'blood_type': profile.blood_type
            }

        # Get conversations
        conversations = Conversation.query.filter_by(user_id=user_id).all()
        for conv in conversations:
            conv_data = {
                'id': conv.id,
                'title': conv.title,
                'created_at': conv.created_at.isoformat() if conv.created_at else None,
                'last_updated_at': conv.last_updated_at.isoformat() if conv.last_updated_at else None,
                'messages': []
            }

            # Get messages for this conversation
            messages = Message.query.filter_by(conversation_id=conv.id).order_by(Message.timestamp).all()
            for msg in messages:
                msg_data = {
                    'id': msg.id,
                    'content': msg.content,
                    'sender': msg.sender,
                    'timestamp': msg.timestamp.isoformat() if msg.timestamp else None,
                    'attachments': []
                }

                # Get attachments for this message
                attachments = Attachment.query.filter_by(message_id=msg.id).all()
                for att in attachments:
                    msg_data['attachments'].append({
                        'filename': att.filename,
                        'file_type': att.file_type,
                        'upload_date': att.upload_date.isoformat() if att.upload_date else None
                    })

                conv_data['messages'].append(msg_data)

            export_data['conversations'].append(conv_data)

        # Log the export action
        log_user_action(
            action=ActionType.DATA_EXPORT,
            user_id=user_id,
            resource_type='user',
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            details={'conversation_count': len(conversations)}
        )

        # Generate JSON file
        json_data = json.dumps(export_data, indent=2, default=str)

        # Return as file download
        filename = f"health_navigator_export_{user.username}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        return send_file(
            BytesIO(json_data.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )


@main_bp.route('/api/user/request-deletion', methods=['POST', 'DELETE'])
def request_account_deletion():
    """
    Request account deletion (GDPR right to be forgotten).
    POST: Initiates deletion request (requires confirmation).
    DELETE: Processes deletion (requires email confirmation).
    """
    if 'user_id' not in session:
        return APIResponse.unauthorized("Authentication required")

    user_id = session['user_id']
    user = User.query.get(user_id)
    if not user:
        return APIResponse.not_found('User not found')

    if request.method == 'POST':
        # Initiate deletion request
        data = request.get_json()
        confirmation_email = data.get('confirmation_email')

        if not confirmation_email:
            return APIResponse.validation_error("Confirmation email is required")

        if confirmation_email != user.email:
            return APIResponse.validation_error("Email does not match account email")

        # Create a deletion request token (in production, email this token)
        deletion_token = os.urandom(32).hex()

        # Log the deletion request
        log_user_action(
            action=ActionType.DATA_DELETE_REQUEST,
            user_id=user_id,
            resource_type='user',
            resource_id=user_id,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            details={'method': 'requested'}
        )

        return APIResponse.success({
            'message': 'Deletion request received. Please confirm by sending DELETE request with your email token.',
            'token': deletion_token,
            'expires_in': '24 hours'
        })

    elif request == 'DELETE':
        # Process deletion (in production, verify email token first)
        # For now, perform soft delete by anonymizing data

        # Anonymize user data
        user.email = f"deleted_{user_id}@anonymous.local"
        user.username = f"deleted_{user_id}"
        user.full_name = "Deleted User"
        user.deleted_at = datetime.utcnow()

        # Anonymize conversations
        conversations = Conversation.query.filter_by(user_id=user_id).all()
        for conv in conversations:
            conv.title = "Deleted Conversation"
            conv.is_anonymized = True

        db.session.commit()

        # Clear session
        session.clear()

        # Log the deletion
        logger.info(f"User account deleted: {user_id}")

        return APIResponse.success({
            'message': 'Account successfully deleted. We appreciate your use of Health Navigator.'
        }, message='Account deleted')


@main_bp.route('/api/user/audit-log', methods=['GET'])
def get_audit_log():
    """
    Get user's audit log (transparency).
    Returns the user's recent activity log.
    """
    if 'user_id' not in session:
        return APIResponse.unauthorized("Authentication required")

    user_id = session['user_id']

    # Get recent audit logs for this user
    logs = AuditLog.query.filter_by(user_id=user_id).order_by(AuditLog.timestamp.desc()).limit(100).all()

    return APIResponse.success({
        'audit_logs': [log.to_dict() for log in logs]
    })
