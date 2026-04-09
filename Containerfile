# ============================================================================
# STS Service — Multi-stage Podman/OCI container
# ============================================================================
# Build:  podman build -t sts-service -f Containerfile .
# Run:    podman run -p 8765:8765 -p 50051:50051 -p 9090:9090 sts-service
# GPU:    podman run --device nvidia.com/gpu=all -p 8765:8765 sts-service
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: build dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY sts/ sts/

# Install into a virtual env for clean COPY in final stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy app code
WORKDIR /app
COPY sts/ sts/
COPY proto/ proto/

# Create model cache directories
RUN mkdir -p /var/lib/sts/piper-models /root/.cache/sts

# Ports: WebSocket | gRPC | Health | Prometheus metrics
EXPOSE 8765 50051 8080 9090

# Health check — hit the dedicated /ready endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ready')" || exit 1

# Container-specific paths and config
ENV STS_CONFIG_PATH=""
ENV PYTHONUNBUFFERED=1

# Override default model path for container layout
ENTRYPOINT ["python", "-m", "sts", "--tts-data-dir", "/var/lib/sts/piper-models"]
