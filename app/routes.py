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
from langgraph.errors import GraphInterrupt
from datetime import datetime

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
                uploaded_files[filename] = file_path

    # Extract message and conversation_id
    if request.content_type.startswith('multipart/form-data'):
        message = request.form.get('message', '')
        conversation_id = request.form.get('conversation_id')
    else:
        data = request.json
        message = data.get('message', '')
        conversation_id = data.get('conversation_id')

    if not conversation_id:
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
    new_messages = list(conversation.messages)
    new_messages.append(user_msg_obj)
    conversation.messages = new_messages
    db.session.commit()

    # --- WORKFLOW EXECUTION ---
    thread_id = str(conversation_id)
    config = {"configurable": {"thread_id": thread_id}}

    # Check current state
    current_state_snapshot = workflow_app.get_state(config)

    # Check if we're resuming from an interrupt
    if current_state_snapshot.next:
        # We have pending tasks - we're resuming from an interrupt
        print(f"DEBUG: Resuming from interrupt with user response: {message}")
        try:
            # Resume with the user's message
            result = workflow_app.invoke(Command(resume=message), config=config)
            
            # Check if finished or interrupted again
            final_state = workflow_app.get_state(config)
            
            if not final_state.next:
                # Workflow completed
                final_output = result.get("final_refined_medical_output", "Analysis complete.")
                
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
            else:
                # Interrupted again - get new question
                if final_state.tasks and final_state.tasks[0].interrupts:
                    interrupt_value = final_state.tasks[0].interrupts[0].value
                    
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
        
        except GraphInterrupt as e:
            # Workflow was interrupted
            print(f"DEBUG: GraphInterrupt caught during resume")
            final_state = workflow_app.get_state(config)
            
            if final_state.tasks and final_state.tasks[0].interrupts:
                interrupt_value = final_state.tasks[0].interrupts[0].value
                
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
        
        except Exception as e:
            print(f"ERROR: Unexpected error during resume: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Workflow error during resume'}), 500

    else:
        # Start new workflow run
        initial_state = {
            "input_prompt": message,
            "attachments": uploaded_files,
            "user_id": str(user.id),
            "conversation_history": []
        }

        try:
            result = workflow_app.invoke(initial_state, config=config)
            
            # Check if completed or interrupted
            final_state = workflow_app.get_state(config)
            
            if not final_state.next:
                # Completed successfully
                final_output = result.get("final_refined_medical_output", "I could not process that request.")

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
            else:
                # Workflow was interrupted - ask user for info
                if final_state.tasks and final_state.tasks[0].interrupts:
                    interrupt_value = final_state.tasks[0].interrupts[0].value
                    
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

        except GraphInterrupt as e:
            # This should not happen with invoke() - it returns state instead
            # But handle it just in case
            print(f"DEBUG: GraphInterrupt caught during initial invoke")
            final_state = workflow_app.get_state(config)
            
            if final_state.tasks and final_state.tasks[0].interrupts:
                interrupt_value = final_state.tasks[0].interrupts[0].value
                
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
        
        except Exception as e:
            print(f"ERROR: Unexpected error during workflow: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Workflow execution error'}), 500

    return jsonify({'error': 'Unknown workflow state'}), 500

