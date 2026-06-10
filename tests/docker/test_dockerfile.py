"""Structural tests for the Phase 15 multi-stage Dockerfile.

We don't run ``docker build`` here — that needs a Docker daemon and is
covered by a documented manual smoke test + the Phase 16 deployment CI
job. These tests parse the Dockerfile as text and assert that the
required Phase 15 spec properties are present:

- multi-stage build (builder, test, runtime)
- non-root user runs the runtime stage
- healthcheck is present
- env vars are declared
- pre-bake step is wired into the builder stage
- ENTRYPOINT uses tini for clean signal handling

If the Dockerfile ever loses one of these by mistake, this suite fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.is_file(), f"Dockerfile not found at {DOCKERFILE}"
    return DOCKERFILE.read_text(encoding="utf-8")


# ---------- multi-stage layout ----------


def test_multi_stage_layout(dockerfile_text: str) -> None:
    """Phase 15 spec: multi-stage builds. Three named stages required."""
    stages = re.findall(r"^FROM\s+\S+\s+AS\s+(\w+)", dockerfile_text, re.MULTILINE)
    assert set(stages) >= {
        "builder",
        "test",
        "runtime",
    }, f"expected builder + test + runtime stages, got {stages}"


def test_test_stage_extends_builder(dockerfile_text: str) -> None:
    """The test stage must re-use the already-installed venv from the
    builder, not duplicate the install. Phase 15 acceptance "tests pass
    inside container" runs ``docker build --target test``."""
    assert re.search(
        r"FROM\s+builder\s+AS\s+test", dockerfile_text
    ), "test stage should be ``FROM builder AS test``"


def test_runtime_stage_uses_slim_image(dockerfile_text: str) -> None:
    # The base image is declared on the FROM line — check it directly
    # rather than digging it out of the stage body.
    assert re.search(
        r"^FROM\s+python:3\.11-slim\s+AS\s+runtime", dockerfile_text, re.MULTILINE
    ), "runtime stage should be ``FROM python:3.11-slim AS runtime`` — keeps the image small"


# ---------- non-root execution ----------


def test_runtime_creates_non_root_user(dockerfile_text: str) -> None:
    runtime = _extract_stage(dockerfile_text, "runtime")
    # ``re.DOTALL`` so the regex spans backslash-continued lines like
    #   useradd ... \\\n    --home-dir ... clarion
    assert re.search(
        r"useradd\s.+?\bclarion\b", runtime, re.DOTALL
    ), "runtime should create a non-root ``clarion`` user"
    assert "USER clarion" in runtime, "runtime should switch to USER clarion"


def test_runtime_uid_1000_for_hf_spaces_compatibility(dockerfile_text: str) -> None:
    runtime = _extract_stage(dockerfile_text, "runtime")
    assert re.search(
        r"--uid\s+1000\b", runtime, re.DOTALL
    ), "HF Spaces requires UID 1000 — must be set explicitly"


# ---------- healthcheck ----------


def test_runtime_declares_healthcheck(dockerfile_text: str) -> None:
    runtime = _extract_stage(dockerfile_text, "runtime")
    assert "HEALTHCHECK" in runtime, "runtime must declare a default HEALTHCHECK"
    assert "healthcheck.sh" in runtime, "HEALTHCHECK should delegate to scripts/healthcheck.sh"


# ---------- env vars ----------


def test_runtime_sets_clarion_env_vars(dockerfile_text: str) -> None:
    runtime = _extract_stage(dockerfile_text, "runtime")
    required = {
        "CLARION_DATA_DIR=/app/data",
        "CLARION_CONFIG_DIR=/app/configs",
        "GRADIO_HOST=0.0.0.0",
        "GRADIO_PORT=7860",
    }
    for kv in required:
        assert kv in runtime, f"runtime ENV should declare {kv}"


def test_runtime_path_includes_venv(dockerfile_text: str) -> None:
    runtime = _extract_stage(dockerfile_text, "runtime")
    assert re.search(
        r"PATH=\"?/app/\.venv/bin", runtime
    ), "runtime PATH must include /app/.venv/bin so installed CLIs are findable"


# ---------- pre-bake + entrypoint ----------


def test_builder_pre_bakes_indices(dockerfile_text: str) -> None:
    builder = _extract_stage(dockerfile_text, "builder")
    assert (
        "build_indices.sh" in builder
    ), "builder must run scripts/build_indices.sh so the runtime image starts ready"


def test_runtime_uses_tini_entrypoint(dockerfile_text: str) -> None:
    runtime = _extract_stage(dockerfile_text, "runtime")
    assert "tini" in runtime, "runtime must install tini for clean PID 1 signal handling"
    assert re.search(
        r'ENTRYPOINT\s*\[\s*"/usr/bin/tini"', runtime
    ), "runtime should ENTRYPOINT with tini --"


def test_runtime_exposes_api_and_gradio_ports(dockerfile_text: str) -> None:
    runtime = _extract_stage(dockerfile_text, "runtime")
    assert re.search(r"EXPOSE\s+.*\b8000\b", runtime), "EXPOSE must include 8000 (API)"
    assert re.search(r"EXPOSE\s+.*\b7860\b", runtime), "EXPOSE must include 7860 (Gradio)"


# ---------- helpers ----------


def _extract_stage(text: str, name: str) -> str:
    """Return the text of one named stage (from `FROM ... AS <name>` to
    the next `FROM` or EOF)."""
    pattern = re.compile(
        rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}\b(.*?)(?=^FROM\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(text)
    assert m, f"could not find stage {name!r} in Dockerfile"
    return m.group(1)
