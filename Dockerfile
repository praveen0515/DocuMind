# =============================================================================
# Stage 1: Build & Dependency Resolution
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for C-extensions (needed by certain vector/math libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to virtual environment
COPY backend/requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Production Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

# Metadata labels
LABEL maintainer="Praveen Patil <https://github.com/praveen0515>"
LABEL description="DocuMind - Enterprise RAG-Powered Document Intelligence Platform"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN groupadd -r documind && useradd -r -g documind -d /app documind

WORKDIR /app

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db \
    UPLOAD_DIR=/app/data/uploads

# Create directories and assign permissions
RUN mkdir -p /app/data/chroma_db /app/data/uploads && \
    chown -R documind:documind /app

# Copy project files
COPY --chown=documind:documind backend /app/backend
COPY --chown=documind:documind frontend /app/frontend

# Switch to non-root user
USER documind

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose FastAPI port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
