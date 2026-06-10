#!/usr/bin/env bash
# scripts/build_indices.sh
#
# Pre-bake the per-customer artifacts inside the Docker builder stage so
# the runtime image starts up immediately without needing CPU time to
# build FAISS indices on first request.
#
# What this produces, per customer:
#   data/<customer>/structured.sqlite   (from data/seeds/<customer>.json)
#   data/<customer>/rules.faiss         (from data/rules/<customer>/*.md)
#   data/<customer>/rules_meta.json     (per-chunk metadata)
#
# Usage:
#   ./scripts/build_indices.sh           # both shipped customers
#   ./scripts/build_indices.sh ophthalmology
#
# Runs inside the Docker builder stage; assumes the venv is on PATH
# (Dockerfile sets PATH=/app/.venv/bin:$PATH before invoking).

set -euo pipefail

# Resolve the repo root regardless of where the script was invoked
# from. Builder stage calls it from /app; local dev runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CUSTOMERS=("$@")
if [ ${#CUSTOMERS[@]} -eq 0 ]; then
  CUSTOMERS=(ophthalmology orthopedics)
fi

for customer in "${CUSTOMERS[@]}"; do
  echo "[build_indices] $customer: ingesting structured + unstructured pipeline"
  python -m clarion.pipelines.ingest all "$customer"
done

echo "[build_indices] done"
