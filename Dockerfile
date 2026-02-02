# Health Navigator - Multi-stage Dockerfile
# Optimized for production deployment

# ===========================================
# Stage 1: Builder - Compile dependencies
# ===========================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gnupg \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ===========================================
# Stage 2: Runtime - Minimal production image
# ===========================================
FROM python:3.11-slim AS runtime

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER=healthnav
ARG APP_DIR=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ${APP_USER} && \
    useradd -r -g ${APP_USER} -d ${APP_DIR} -s /sbin/nologin -c "Health Navigator user" ${APP_USER}

# Create application directory
WORKDIR ${APP_DIR}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=run.py \
    FLASK_ENV=production \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy application code
COPY app ${APP_DIR}/app
COPY migrations ${APP_DIR}/migrations
COPY run.py ${APP_DIR}/

# Create necessary directories
RUN mkdir -p ${APP_DIR}/logs ${APP_DIR}/uploads ${APP_DIR}/models && \
    chown -R ${APP_USER}:${APP_USER} ${APP_DIR}

# Switch to non-root user
USER ${APP_USER}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health/live || exit 1

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "run:app"]
