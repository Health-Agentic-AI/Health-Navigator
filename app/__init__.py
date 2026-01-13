from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from dotenv import load_dotenv

load_dotenv(r'C:\My Projects\Health-Navigator\credentials.env')

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    # Database Configuration - Switched to MySQL
    # Using pymysql as driver
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{os.environ.get("MYSQL_USER", "root")}:{os.environ.get("MYSQL_PASSWORD", "password")}@{os.environ.get("MYSQL_HOST", "localhost")}:{os.environ.get("MYSQL_PORT", "3306")}/{os.environ.get("DATABASE_NAME", "medical_db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.environ.get("FLASK_SECRET_KEY", "dev-key-please-change")

    db.init_app(app)
    migrate.init_app(app, db)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
