from app import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON, ARRAY

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    profile = db.relationship('PatientProfile', backref='user', uselist=False, cascade="all, delete-orphan")
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    allergies = db.relationship('Allergy', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    medications = db.relationship('Medication', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    medical_history = db.relationship('PastMedicalHistory', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    surgeries = db.relationship('PastSurgery', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    family_history = db.relationship('FamilyHistory', backref='user', lazy='dynamic', cascade="all, delete-orphan")

class PatientProfile(db.Model):
    __tablename__ = 'patient_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    occupation = db.Column(db.String(255))
    is_smoker = db.Column(db.Boolean)
    smoking_details = db.Column(db.Text)
    alcohol_consumption = db.Column(db.String(50))
    alcohol_details = db.Column(db.Text)
    socioeconomic_status = db.Column(db.String(50))
    last_updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Conversation(db.Model):
    __tablename__ = 'conversations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255))
    messages = db.Column(JSON, nullable=False, default=list) # Stores list of message objects
    # Note: 'messages' JSON structure: [{'role': 'user'|'assistant'|'system', 'content': '...', 'timestamp': '...'}]
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Allergy(db.Model):
    __tablename__ = 'allergies'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    allergy_name = db.Column(db.String(255), nullable=False)
    allergy_type = db.Column(db.String(50))
    severity = db.Column(db.String(50))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Medication(db.Model):
    __tablename__ = 'medications'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    medication_name = db.Column(db.String(255), nullable=False)
    dosage = db.Column(db.String(100))
    frequency = db.Column(db.String(100))
    started_at = db.Column(db.Date)
    ended_at = db.Column(db.Date)
    is_current = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PastMedicalHistory(db.Model):
    __tablename__ = 'past_medical_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    condition = db.Column(db.String(255), nullable=False)
    diagnosed_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PastSurgery(db.Model):
    __tablename__ = 'past_surgeries'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    surgery_name = db.Column(db.String(255), nullable=False)
    surgery_date = db.Column(db.Date)
    hospital = db.Column(db.String(255))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FamilyHistory(db.Model):
    __tablename__ = 'family_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    relation = db.Column(db.String(50), nullable=False)
    condition = db.Column(db.String(255), nullable=False)
    age_of_diagnosis = db.Column(db.Integer)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
