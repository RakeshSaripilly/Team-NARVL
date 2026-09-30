# syntax=docker/dockerfile:1.4
# ==============================================================================
# Multi-Stage Production Dockerfile for NARVL
# Autonomous, 100% Offline, Plug-and-Play Agentic Data Cleaning Platform
# ==============================================================================

# Stage 1: Build & Packaging
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY narvl ./narvl

RUN pip install --no-cache-dir build && \
    python -m build --wheel --no-isolation

# ==============================================================================
# Stage 2: Production Air-Gapped Runtime
# ==============================================================================
FROM python:3.11-slim-bookworm AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NARVL_OFFLINE=1 \
    NARVL_MODEL_DIR=/app/models \
    PORT=8501

WORKDIR /app

# Install runtime dependencies (e.g., curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-root user
RUN groupadd -g 10001 narvlgroup && \
    useradd -u 10001 -g narvlgroup -s /bin/bash -m narvluser

# Copy wheel from builder stage and install
COPY --from=builder /build/dist/*.whl /app/
RUN pip install --no-cache-dir /app/*.whl && \
    pip install --no-cache-dir \
        streamlit>=1.30.0 \
        fastapi>=0.100.0 \
        uvicorn>=0.20.0 \
        lightgbm>=4.0.0 \
        pandera>=0.18.0 \
        deltalake>=0.17.0 \
        great-expectations>=0.18.0 && \
    rm -f /app/*.whl

# Setup air-gapped model directory
RUN mkdir -p /app/models /app/data /app/reports && \
    chown -R narvluser:narvlgroup /app

USER narvluser

# Expose Streamlit CleanPilot (8501) and FastAPI Server (8000)
EXPOSE 8501 8000

# Healthcheck targeting FastAPI /healthz endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default launch command: Streamlit CleanPilot UI
ENTRYPOINT ["narvl"]
CMD ["ui", "--port", "8501"]
