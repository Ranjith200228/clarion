# syntax=docker/dockerfile:1.7
# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml ./
# Lockfile is optional at this phase; copy if it exists.
COPY poetry.lock* ./

RUN poetry install --only main --no-root

COPY clarion ./clarion
COPY configs ./configs
COPY api ./api

RUN poetry install --only main

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Copy installed site-packages and app code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Non-root user for Cloud Run friendliness
RUN useradd --create-home --uid 10001 clarion && chown -R clarion:clarion /app
USER clarion

EXPOSE 8080

# Placeholder command — replaced when FastAPI service lands in Phase 8.
CMD ["python", "-c", "import clarion; print(f'Clarion {clarion.__version__} container OK')"]
