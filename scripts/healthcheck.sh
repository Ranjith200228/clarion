#!/usr/bin/env bash
# scripts/healthcheck.sh
#
# Used by docker-compose ``healthcheck`` blocks for both the FastAPI
# service (URL = http://localhost:8000/health) and the Gradio service
# (URL = http://localhost:7860/).
#
# Returns 0 if the URL responds with a 2xx within the timeout, non-zero
# otherwise. We use ``curl`` rather than ``wget`` because the runtime
# image already needs curl for other diagnostic flows.
#
# Usage (inside the container):
#   ./scripts/healthcheck.sh http://localhost:8000/health
#   ./scripts/healthcheck.sh http://localhost:7860/

set -euo pipefail

URL="${1:?healthcheck url required}"
TIMEOUT_S="${HEALTHCHECK_TIMEOUT_S:-5}"

# --silent: no progress noise in compose logs
# --fail:   exit non-zero on 4xx/5xx
# --output /dev/null: discard body
# --max-time: cap on total time
curl --silent --fail --max-time "$TIMEOUT_S" --output /dev/null "$URL"
