# syntax=docker/dockerfile:1

# ==============================================================================
# Stage 1: Build virtual environment with Astral uv
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy Astral uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Compile bytecode during installation to eliminate cold-start latency
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency specifications first for optimal layer caching
COPY pyproject.toml uv.lock ./

# Install runtime production dependencies into /app/.venv without dev tools
RUN uv sync --frozen --no-dev --no-install-project

# Copy package source and README to complete project installation
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


# ==============================================================================
# Stage 2: Minimal, secure production runtime
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install runtime system libraries:
# - libgomp1: OpenMP shared library required by LightGBM and XGBoost
# - curl: Lightweight probe client for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN useradd -m -u 1000 appuser

# Copy virtual environment and application assets with correct ownership
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser src ./src

# Set runtime environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Drop privileges to non-root user
USER appuser

# Expose microservice listening port
EXPOSE 8000

# Container healthcheck probe against FastAPI readiness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch ASGI server with signal forwarding
CMD ["sh", "-c", "exec uvicorn hotel_cancellation_prediction.api.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
