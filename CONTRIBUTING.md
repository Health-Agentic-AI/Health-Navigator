# Contributing to Health Navigator

Thank you for your interest in contributing to Health Navigator! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 6+ (optional, for rate limiting)
- Git
- Make (optional, for Makefile commands)

### Cloning the Repository

```bash
git clone https://github.com/yourusername/health-navigator.git
cd health-navigator
git checkout develop
```

## Development Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
make install
# Or manually:
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Database Setup

```bash
make migrate
make seed
```

### 5. Run Development Server

```bash
make dev
# Or: flask run --debug --host=0.0.0.0 --port=5000
```

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use Black for formatting (line length: 120)
- Use isort for import sorting
- Add docstrings to all functions and classes
- Type hints are encouraged

```bash
make format    # Format code with black and isort
make lint      # Run linting checks
```

### JavaScript Code

- Use 4 spaces for indentation
- Use camelCase for variables and functions
- Add JSDoc comments for functions

### CSS Code

- Use 2 spaces for indentation
- Use kebab-case for class names
- Follow BEM methodology for component styles

## Testing

### Running Tests

```bash
make test            # Run all tests
make test-coverage   # Run with coverage report
```

### Writing Tests

- Write unit tests for all new functions
- Write integration tests for API endpoints
- Aim for 80%+ code coverage
- Place tests in the `tests/` directory

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── conftest.py    # Pytest configuration
```

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `security`: Security vulnerability fix

### Examples

```
feat(api): add health check endpoints
fix(auth): resolve session timeout issue
docs(readme): update installation instructions
test(workflow): add integration tests for agent execution
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Run tests and ensure they pass

### 3. Pre-commit Checks

```bash
make format
make lint
make test
```

Or install pre-commit hooks:

```bash
make pre-commit
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat(scope): description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to GitHub and create a PR from your branch to `develop`
- Fill out the PR template
- Link any related issues
- Wait for code review

### PR Review Guidelines

- Address all review comments
- Keep PRs focused and small
- Ensure CI/CD checks pass
- Request review from at least one maintainer

## Reporting Issues

### Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or screenshots

### Feature Requests

Include:
- Clear description of the feature
- Use case / problem it solves
- Proposed implementation (if applicable)
- Alternative approaches considered

## Development Workflow

```
main (production)
  └── develop (integration)
       ├── feature/feature-name
       ├── bugfix/bug-name
       └── hotfix/critical-fix
```

1. Create feature branch from `develop`
2. Make changes and commit
3. Create PR to `develop`
4. After review and merge, delete feature branch
5. `develop` is periodically merged to `main` for releases

## Useful Commands

```bash
make dev              # Start development server
make install          # Install dependencies
make format           # Format code
make lint             # Run linting
make test             # Run tests
make test-coverage    # Run tests with coverage
make migrate          # Apply database migrations
make reset-db         # Reset database (WARNING: deletes data)
make docker-build     # Build Docker image
make docker-run       # Run Docker containers
make clean            # Remove cache and build artifacts
make logs             # Show application logs
```

## Questions?

- Open an issue on GitHub
- Contact the maintainers
- Check the documentation in the `docs/` folder

## License

By contributing to Health Navigator, you agree that your contributions will be licensed under the same license as the project.
