# Image for Ray with all setup
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set working directory for the application
WORKDIR /app

# Environment configuration
# DEBIAN_FRONTEND=noninteractive: Prevent apt from asking interactive questions
# TZ=Asia/Ho_Chi_Minh: Set timezone for consistent logging and timestamps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# =============================================================================
# System Dependencies Installation
# =============================================================================
# Install essential build tools, Python 3.11, and audio processing dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tzdata \
    gcc \
    g++ \
    make \
    cmake \
    software-properties-common \
    curl \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# =============================================================================
# Python Package Management Setup
# =============================================================================
# uv replaces pip: it installs the exact versions pinned in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

# Use the system Python 3.11 and keep the virtualenv outside /app so that
# `COPY . /app` below never overwrites it. Putting its bin dir on PATH lets the
# CMD call `ray` and `serve` directly.
ENV UV_PYTHON=python3.11 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# =============================================================================
# Dependencies Installation
# =============================================================================
# Install NeMo (ASR), Ray Serve and all other dependencies from the lockfile.
# Only the dependency manifests are copied first so this layer stays cached
# until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project --no-dev

# =============================================================================
# Application Deployment
# =============================================================================
# Copy application source code to container
COPY . /app

EXPOSE 8000 8265

# Run both dashboard and serve
CMD ["bash", "-c", "ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 > /tmp/ray_start.log 2>&1 || (cat /tmp/ray_start.log >&2; exit 1) && serve run app.app:deployment"]