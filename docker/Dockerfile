# syntax=docker/dockerfile:1
# =============================================================================
# Stage 1 — Builder: Python interpreter + virtualenv
# =============================================================================
# Everything needed only to install dependencies (uv, its cache, the Python
# installer) stays in this stage. Every dependency in uv.lock ships a wheel or
# is pure Python, so no compiler is needed here either. Same Ubuntu release as
# the runtime base so the selected manylinux wheels match its glibc.
FROM ubuntu:22.04 AS builder

# uv replaces pip: it installs the exact versions pinned in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/

# uv installs a standalone CPython 3.11 under /opt/python (no deadsnakes PPA),
# and the virtualenv goes to /opt/venv. Both are copied to the same paths in
# the runtime stage, because the venv points at the interpreter by absolute path.
ENV UV_PYTHON=3.11 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    SOURCE_DATE_EPOCH=0

WORKDIR /app

# Install NeMo (ASR), Ray Serve and all other dependencies from the lockfile.
# Only the dependency manifests are copied so this layer stays cached until
# pyproject.toml or uv.lock change. The uv cache lives in a BuildKit cache
# mount: it speeds up rebuilds but is never saved into an image.
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.11 \
    && uv sync --frozen --no-install-project --no-dev \
    # Stage the CUDA libs under their final path, so COPY / keeps the parent
    # directories' mtimes instead of stamping them with the build time
    && mkdir -p /nvidia-layer/opt/venv/lib/python3.11/site-packages \
    && cp -a /opt/venv/lib/python3.11/site-packages/nvidia \
       /nvidia-layer/opt/venv/lib/python3.11/site-packages/ \
    # Fixed mtimes (and SOURCE_DATE_EPOCH for .pyc) keep layer digests stable
    # when file contents are unchanged
    && find /opt/python /opt/venv /nvidia-layer -exec touch -h -d @0 {} +

# =============================================================================
# Stage 2 — Runtime
# =============================================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set working directory for the application
WORKDIR /app

# Environment configuration
# DEBIAN_FRONTEND=noninteractive: Prevent apt from asking interactive questions
# TZ=Asia/Ho_Chi_Minh: Set timezone for consistent logging and timestamps
# PATH: lets the CMD call `ray` and `serve` from the virtualenv directly
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Ho_Chi_Minh \
    PATH="/opt/venv/bin:$PATH"

# Runtime system libraries only (no compilers, no build tools):
# - ffmpeg:    shared libraries TorchCodec decodes audio with
# - libmagic1: required by python-magic (audio MIME check); it used to arrive
#              only indirectly through build-essential
# - tzdata:    timezone data for TZ
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libmagic1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# Interpreter and virtualenv from the builder, at the same paths
COPY --from=builder /opt/python /opt/python

# CUDA libs (~3.2GB) get their own layer: it only changes when torch/CUDA change,
# so pushes and pulls skip it when other dependencies are bumped
COPY --from=builder /nvidia-layer/ /

# Rest of the virtualenv, without nvidia/ so it is not copied twice
COPY --from=builder --exclude=lib/python3.11/site-packages/nvidia /opt/venv /opt/venv

# =============================================================================
# Application Deployment
# =============================================================================
# Copy application source code to container
COPY . /app

EXPOSE 8000 8265

# Run both dashboard and serve
CMD ["bash", "-c", "ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 > /tmp/ray_start.log 2>&1 || (cat /tmp/ray_start.log >&2; exit 1) && serve run config/serve.yaml"]
