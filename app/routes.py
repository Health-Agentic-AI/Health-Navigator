from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, current_app
from app import db
from app.models import User, Conversation, PatientProfile, Allergy, Medication, PastMedicalHistory, PastSurgery, FamilyHistory
from app.workflow.workflow import app as workflow_app
from langgraph.types import Command
import uuid
import os
import json
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

main_bp = Blueprint('main', __name__)

# --- Helper Functions ---

def get_current_user():
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx'}

# --- Routes ---

@main_bp.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('main.chat'))
    return redirect(url_for('main.login'))

@main_bp.route('/register', methods=['GET', 'POST'])
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
def create_conversation():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    title = data.get('title', 'New Consultation')

    conversation = Conversation(
        user_id=user.id,
        title=title,
        messages=[]
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

    return jsonify({
        'id': conversation.id,
        'title': conversation.title,
        'messages': conversation.messages
    })

@main_bp.route('/api/chat/message', methods=['POST'])
def send_message():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    # Handle file uploads
    uploaded_files = {}
    if request.files:
        upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', str(user.id))
        os.makedirs(upload_folder, exist_ok=True)

        for key, file in request.files.items():
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                # Store absolute path for workflow, or relative if configured
                uploaded_files[filename] = file_path # Key is filename, value is path

    # Form data might be mixed with files
    if request.content_type.startswith('multipart/form-data'):
        message = request.form.get('message', '')
        conversation_id = request.form.get('conversation_id')
    else:
        data = request.json
        message = data.get('message', '')
        conversation_id = data.get('conversation_id')
        uploaded_files = data.get('attachments', {}) # If sent as JSON (unlikely for files)

    if not conversation_id:
        # Auto-create conversation
        conversation = Conversation(user_id=user.id, title=message[:30] + "...", messages=[])
        db.session.add(conversation)
        db.session.commit()
        conversation_id = conversation.id
    else:
        conversation = Conversation.query.get(conversation_id)

    # Add user message to DB
    user_msg_obj = {
        'role': 'user',
        'content': message,
        'timestamp': str(datetime.utcnow()),
        'attachments': list(uploaded_files.keys())
    }
    # Create a new list for mutation to trigger SQLAlchemy JSON detection
    new_messages = list(conversation.messages)
    new_messages.append(user_msg_obj)
    conversation.messages = new_messages
    db.session.commit()

    # --- WORKFLOW EXECUTION ---
    thread_id = str(conversation_id)
    config = {"configurable": {"thread_id": thread_id}}

    # Check if we are resuming from an interrupt
    # We can inspect the state of the graph
    current_state_snapshot = workflow_app.get_state(config)

    if current_state_snapshot.next:
        # We are paused. We assume the user's message is the answer to the interrupt.
        # We need to resume the workflow.
        # Construct the Command to resume
        # The interrupt logic in the tool just needs the text response.
        resume_command = Command(resume=message)

        try:
            result = workflow_app.invoke(resume_command, config=config)
            # If successful (no new interrupt), handle final output
            final_output = result.get("final_refined_medical_output", "Analysis Complete.")

            # Save assistant response
            assistant_msg_obj = {
                'role': 'assistant',
                'content': final_output,
                'timestamp': str(datetime.utcnow())
            }
            new_messages = list(conversation.messages)
            new_messages.append(assistant_msg_obj)
            conversation.messages = new_messages
            db.session.commit()

            return jsonify({
                'status': 'completed',
                'response': final_output,
                'conversation_id': conversation.id
            })

        except Exception as e:
            # Check if it's another interrupt (GraphInterrupt is caught by invoke usually,
            # but if it returns due to interrupt, we need to check state)
            # Actually invoke() returns the state. If it stopped due to interrupt,
            # we need to check the tasks.

            # Let's inspect the state after invoke
            post_run_state = workflow_app.get_state(config)
            if post_run_state.next:
                 # It's an interrupt again (unlikely in this specific flow, but possible)
                 # Or if the first invoke raises GraphInterrupt, we catch it here.
                 # Wait, invoke() raises GraphInterrupt if interrupted?
                 # LangGraph docs: "When you call invoke.. and an interrupt is hit... it raises GraphInterrupt"
                 # OR it just returns partial state?
                 # It actually raises GraphInterrupt.
                 pass

    else:
        # Start new run
        initial_state = {
            "input_prompt": message,
            "attachments": uploaded_files,
            "user_id": str(user.id),
            "conversation_history": [
                # Map DB history to LangChain messages format if needed by agents
                # For now, we pass the raw list or just let the workflow manage it.
                # The workflow "conversation_history" key expects List[Dict].
                # We can pass the full history.
                msg for msg in conversation.messages
            ]
        }

        # We use a loop/try block to handle potential interrupts
        try:
            result = workflow_app.invoke(initial_state, config=config)

            # If we got here, it finished successfully
            final_output = result.get("final_refined_medical_output", "I could not process that request.")

             # Save assistant response
            assistant_msg_obj = {
                'role': 'assistant',
                'content': final_output,
                'timestamp': str(datetime.utcnow())
            }
            new_messages = list(conversation.messages)
            new_messages.append(assistant_msg_obj)
            conversation.messages = new_messages
            db.session.commit()

            return jsonify({
                'status': 'completed',
                'response': final_output,
                'conversation_id': conversation.id
            })

        except Exception as e:
            # If it's a GraphInterrupt, we need to handle it.
            # LangGraph raises a specific exception or just stops.
            # If checking state reveals tasks, it's interrupted.
            pass

    # If we are here, we might have been interrupted (either from start or resume)
    # Let's check the state
    snapshot = workflow_app.get_state(config)
    if snapshot.next:
        # Get the interrupt details
        # The interrupt value is returned in the snapshot
        if snapshot.tasks:
            # The interrupt value is usually in the `interrupts` property of the task
            # Tuple of (interrupt_value, )
            interrupt_value = snapshot.tasks[0].interrupts[0].value

            # Save the system question to DB
            system_msg_obj = {
                'role': 'system_question',
                'content': interrupt_value,
                'timestamp': str(datetime.utcnow())
            }
            new_messages = list(conversation.messages)
            new_messages.append(system_msg_obj)
            conversation.messages = new_messages
            db.session.commit()

            return jsonify({
                'status': 'interrupted',
                'question': interrupt_value,
                'conversation_id': conversation.id
            })

    return jsonify({'error': 'Workflow failed or entered unknown state'}), 500

from datetime import datetime
