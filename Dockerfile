# syntax=docker/dockerfile:1.7
# Phase 15 — production-grade multi-stage Dockerfile for Clarion.
#
# Stages:
#   builder  — Python 3.11-slim + poetry, installs all runtime deps + the
#              ``ui`` group, copies in source, pre-builds the per-customer
#              FAISS indices + SQLite stores so the runtime image starts
#              instantly.
#   test     — extends builder, adds the dev group, runs pytest. Used for
#              the Phase 15 acceptance "tests pass inside container":
#                docker build --target test -t clarion:test .
#   runtime  — Python 3.11-slim, copies the venv + source + pre-built
#              indices from builder, runs as the non-root ``clarion``
#              user (UID 1000 for HF Spaces compatibility), exposes both
#              the API (8000) and Gradio (7860) ports.
#
# Two docker-compose services share this same image, differing only in
# CMD. See docker-compose.yml.

# =========== builder ===========
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app

# Build deps: curl for poetry install, build-essential + faiss compile deps.
# Removed after the install so they don't bloat the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

# Copy dependency manifests first so the layer cache survives source edits.
COPY pyproject.toml poetry.lock ./

# Install runtime + UI deps into /app/.venv. Skip the dev group at this
# stage — it lands only in the ``test`` stage below.
RUN poetry install --with ui --no-root

# Copy application source.
COPY clarion ./clarion
COPY api ./api
COPY gradio_app ./gradio_app
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

# Pre-bake per-customer FAISS + SQLite artifacts so the runtime image
# starts with everything ready. Without this the first /chat call would
# block on a slow first-time index build.
ENV PATH="/app/.venv/bin:$PATH"
RUN bash scripts/build_indices.sh

# Install the project itself last so ``clarion`` is importable from the venv.
RUN poetry install --with ui --only-root

# =========== test ===========
# ``docker build --target test`` runs pytest inside the container.
# Phase 15 spec acceptance: "Tests pass inside container".
FROM builder AS test

RUN poetry install --with dev

# Provide an .env-style default that mirrors how CI runs the suite. The
# CLARION_SKIP_AUTOAPP guard prevents api/app.py's module-level singleton
# from instantiating an OpenAIClient at import time when no key is set.
ENV CLARION_SKIP_AUTOAPP=1

# Run the full suite during the test stage. ``docker build --target test``
# fails the build if any test fails — perfect for the acceptance gate.
RUN python -m pytest -q

# =========== runtime ===========
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    CLARION_DATA_DIR=/app/data \
    CLARION_CONFIG_DIR=/app/configs \
    GRADIO_HOST=0.0.0.0 \
    GRADIO_PORT=7860

WORKDIR /app

# curl is used by scripts/healthcheck.sh; tini is the minimal init we
# launch under so signals propagate cleanly to the Python process.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID 1000 is the de-facto standard for HF Spaces; Cloud
# Run, Render, and Fly.io all tolerate it.
RUN groupadd --system --gid 1000 clarion \
    && useradd --system --uid 1000 --gid 1000 --shell /bin/bash \
        --create-home --home-dir /home/clarion clarion

# Copy the venv + project source + pre-baked indices from the builder.
COPY --from=builder --chown=clarion:clarion /app /app

USER clarion

EXPOSE 8000 7860

# The two compose services override this CMD with their respective
# entry points; this default makes ``docker run clarion`` give a friendly
# usage hint instead of failing silently.
CMD ["python", "-c", "import sys; sys.stderr.write('Specify a command: \\n  api    -> python -m api.app\\n  gradio -> python -m gradio_app\\n  test   -> python -m pytest -q\\nSee docker-compose.yml for both services.\\n')"]

# Tini as PID 1 reaps zombies + forwards SIGTERM so Cloud Run / HF
# Spaces shut us down cleanly.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default healthcheck — overridden per service by docker-compose.yml.
# (Compose-level healthchecks always take precedence over Dockerfile-
# level ones; this is here as a sensible default for ``docker run``.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD bash scripts/healthcheck.sh http://localhost:8000/health || exit 1
