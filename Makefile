.PHONY: help dev test test-coverage format lint migrate clean docker-build docker-run install

# Default target
help:
	@echo "Health Navigator - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development server"
	@echo "  make install       - Install Python dependencies"
	@echo "  make format        - Format code with Black and isort"
	@echo "  make lint         - Run linting (Flake8, mypy, Bandit)"
	@echo "  make test         - Run all tests"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo ""
	@echo "Database:"
	@echo "  make migrate       - Apply database migrations"
	@echo "  make upgrade       - Apply database migrations (alias for migrate)"
	@echo "  make rollback      - Rollback last migration"
	@echo "  make reset-db      - Drop all tables and recreate"
	@echo "  make seed          - Seed database with initial data"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run     - Run Docker containers"
	@echo "  make docker-down    - Stop Docker containers"
	@echo "  make docker-clean   - Remove Docker volumes"
	@echo ""
	@echo "Quality:"
	@echo "  make security-scan  - Run security scan with Bandit"
	@echo "  make pre-commit    - Set up pre-commit hooks"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove Python cache and build artifacts"
	@echo "  make dist-clean     - Remove all generated files"

# Development
dev:
	flask run --debug --host=0.0.0.0 --port=5000

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .[dev]
	pip install pre-commit

# Code Quality
format:
	black app/ --exclude=migrations
	isort app/

lint:
	flake8 app/ --exclude=migrations --max-line-length=120
	mypy app/ --exclude=migrations || true
	bandit -r app/ -f json -o bandit-report.json || true

security-scan:
	bandit -r app/ -f json -o bandit-report.json

pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed. They will run on each git commit."

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

# Database
migrate: upgrade
	@echo "Database migrations applied successfully."

upgrade:
	flask db upgrade
	@echo "Database upgraded successfully."

rollback:
	flask db downgrade
	@echo "Database rolled back one migration."

reset-db:
	@echo "WARNING: This will delete all data!"
	@read -p "Continue? [y/N] " confirm && \
	flask db reset && \
	@echo "Database reset complete."

seed:
	python -c "from app.models import User; from app import db; u = User(username='demo', email='demo@example.com', full_name='Demo User'); u.set_password('DemoPass123!'); db.session.add(u); db.session.commit(); print('Demo user created')"

# Docker
docker-build:
	docker build -t health-navigator .

docker-run:
	docker-compose up -d

docker-down:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker system prune -f

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .pytest_cache/
	rm -f .coverage coverage.xml bandit-report.json safety-report.json

dist-clean: clean
	rm -rf .venv/ venv/ env/ ENV/
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf logs/*.log

# Utilities
logs:
	@echo "Showing recent logs..."
	tail -f logs/app.log || echo "No log file found."

shell:
	@python -c "import code; code.interact('Local Flask shell')"
