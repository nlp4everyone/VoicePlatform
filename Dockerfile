# VieNeu-TTS v3 Turbo - OpenAI-compatible speech API on CUDA (PyTorch backend).
#
# Built from a pinned commit of https://github.com/pnnbao97/VieNeu-TTS using that
# commit's own uv.lock (`uv sync --frozen`), so torch/transformers match exactly what
# upstream benchmarked and nothing drifts between builds. Bump VIENEU_REF to upgrade.
ARG VIENEU_REF=85344322b7258b4e25479b692e8e3396baf9db34

# --- Stage: src (fetch the pinned upstream commit) ---
FROM alpine/git AS src
ARG VIENEU_REF
WORKDIR /src
RUN git init -q . \
    && git remote add origin https://github.com/pnnbao97/VieNeu-TTS.git \
    && git fetch -q --depth 1 origin "${VIENEU_REF}" \
    && git checkout -q FETCH_HEAD

# --- Stage: base (CUDA runtime + Python 3.12 + non-root user) ---
# The runtime image is enough: the CUDA graphs are captured by PyTorch, no compiler needed.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/workspace/src:/workspace" \
    HF_HOME=/home/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Ubuntu 24.04 images ship a default `ubuntu` user with UID 1000; drop it so `app` can take
# that UID (it matches the usual host user, which keeps the mounted model cache writable).
RUN (getent passwd 1000 | cut -d: -f1 | xargs -r userdel -r || true) \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /workspace /opt/venv "${HF_HOME}/hub" \
    && chown -R app:app /workspace /opt/venv /home/app/.cache

# --- Stage: deps (resolve the locked dependency set into /opt/venv) ---
FROM base AS deps
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /workspace
COPY --from=src --chown=app:app /src/pyproject.toml /src/uv.lock ./
USER app
# Wheels only, no dev group, no project build: the app runs straight from the source tree.
RUN uv sync --frozen --no-dev --extra cuda --no-install-project --no-build

# --- Stage: runtime (venv + sources only, no uv, no build cache) ---
FROM base AS runtime
WORKDIR /workspace
COPY --from=deps --chown=app:app /opt/venv /opt/venv
COPY --from=src --chown=app:app /src/src ./src
COPY --from=src --chown=app:app /src/apps ./apps
COPY --from=src --chown=app:app /src/config.yaml ./
USER app

# HOST/PORT are read by apps/openai_speech.py (it binds to loopback by default).
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

CMD ["python", "-m", "apps.openai_speech"]
