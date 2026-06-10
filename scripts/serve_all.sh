#!/usr/bin/env bash
# scripts/serve_all.sh
#
# Phase 16 single-process container entrypoint. Starts the FastAPI
# service in the background and the Gradio app in the foreground, both
# inside one container. Used by Hugging Face Gradio Spaces (Docker SDK),
# which expects exactly one CMD with one externally-visible port.
#
# Compose deployments (Phase 15 docker-compose.yml) still run the two
# services in separate containers and never call this script.
#
# Ports:
#   FastAPI binds to 127.0.0.1:8000  (Gradio reaches it via CLARION_API_URL)
#   Gradio  binds to 0.0.0.0:$GRADIO_PORT (default 7860 — HF Spaces' app_port)
#
# Signal handling: trap SIGTERM and forward it to both children so a
# container stop shuts them down cleanly.
#
# Logs from both processes interleave on stdout; HF Spaces / Cloud Run
# / Render aggregate that as container logs.

set -euo pipefail

API_PORT="${CLARION_API_PORT:-8000}"
API_HOST="${CLARION_API_HOST:-127.0.0.1}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
GRADIO_HOST="${GRADIO_HOST:-0.0.0.0}"

# Gradio talks to the API over loopback inside this container.
# Operators can override this for advanced topologies (e.g. external
# API behind a load balancer).
export CLARION_API_URL="${CLARION_API_URL:-http://${API_HOST}:${API_PORT}}"

echo "[serve_all] starting FastAPI on ${API_HOST}:${API_PORT}"
python -m api.app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

# Forward SIGTERM/SIGINT to both children. On clean shutdown the
# foreground gradio exits first and we just send TERM to the API.
shutdown() {
  echo "[serve_all] received shutdown signal — stopping children"
  kill -TERM "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  exit 0
}
trap shutdown SIGTERM SIGINT

# Give FastAPI a brief head start so the Gradio app's first /health
# probe lands on a live API.
sleep 2

echo "[serve_all] starting Gradio on ${GRADIO_HOST}:${GRADIO_PORT}"
echo "[serve_all] CLARION_API_URL=${CLARION_API_URL}"
python -m gradio_app

# If the Gradio process exits on its own (clean ctrl-C, container
# stop), make sure the API doesn't linger.
kill -TERM "$API_PID" 2>/dev/null || true
wait "$API_PID" 2>/dev/null || true
