# syntax=docker/dockerfile:1.4
# ==============================================================================
# Multi-Stage Highly Space-Efficient Production Dockerfile for NARVL
# Autonomous, 100% Offline, Plug-and-Play Agentic Data Cleaning Platform
# Optimized for minimal footprint and maximum runtime SIMD/AVX performance
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Packaging
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Create clean, isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install modern wheel packaging tools
RUN pip install --no-cache-dir build setuptools wheel

# Copy source definitions
COPY pyproject.toml README.md ./
COPY narvl ./narvl

# Build NARVL wheel distribution
RUN python -m build --wheel --no-isolation

# Install NARVL and complete production ecosystem into virtual environment
RUN pip install --no-cache-dir dist/*.whl && \
    pip install --no-cache-dir \
        streamlit>=1.30.0 \
        fastapi>=0.100.0 \
        uvicorn>=0.20.0 \
        lightgbm>=4.0.0 \
        pandera>=0.18.0 \
        deltalake>=0.17.0 \
        great-expectations>=0.18.0

# Aggressive zero-performance-impact slimming:
# Remove non-runtime internal test suites, compilation headers, docs, and build tools
RUN find /opt/venv/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + && \
    find /opt/venv/lib/python3.11/site-packages -type d -name "test" -not -path "*/narvl/*" -exec rm -rf {} + && \
    find /opt/venv/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + && \
    find /opt/venv/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /opt/venv/lib/python3.11/site-packages -name "*.pyo" -delete && \
    find /opt/venv/lib/python3.11/site-packages -name "*.h" -delete && \
    find /opt/venv/lib/python3.11/site-packages -name "*.c" -delete && \
    find /opt/venv/lib/python3.11/site-packages -name "*.md" -not -path "*/narvl/*" -delete && \
    find /opt/venv/lib/python3.11/site-packages -name "*.rst" -delete && \
    pip uninstall -y build setuptools wheel && \
    rm -rf /root/.cache /tmp/*

# ------------------------------------------------------------------------------
# Stage 2: Minimal Air-Gapped Production Runner
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NARVL_OFFLINE=1 \
    NARVL_MODEL_DIR=/app/models \
    PORT=8501 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false

WORKDIR /app

# Install only critical shared libraries (libgomp1 for OpenMP/LightGBM, curl for healthcheck)
# Immediately purge apt caches, package indexes, and manual pages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/doc/* /usr/share/man/* /usr/share/locale/*

# Create secure non-root user
RUN groupadd -g 10001 narvlgroup && \
    useradd -u 10001 -g narvlgroup -s /bin/bash -m narvluser

# Copy only the optimized, self-contained virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Setup workspace directories and permissions
RUN mkdir -p /app/models /app/data /app/reports && \
    chown -R narvluser:narvlgroup /app

USER narvluser

# Expose Streamlit CleanPilot (8501) and FastAPI Server (8000)
EXPOSE 8501 8000

# Resilient healthcheck for both UI (8501) and API (8000) modes
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || curl -f http://localhost:8000/healthz || exit 1

# Default entrypoint: NARVL CLI
ENTRYPOINT ["narvl"]
CMD ["ui", "--port", "8501"]
