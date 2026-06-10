"""Structural tests for the Phase 16 deployment manifests.

Asserts the Phase 16 spec contract: same image, same env var
convention, secrets via env var (never hardcoded). The actual deploys
need cloud credentials we don't have in CI; these tests guard against
regressions in the manifest shapes.

Covered:
- requirements.txt is generated and includes the key pinned packages
- huggingface/README.md frontmatter has the required Space metadata
- deploy/cloudrun.yaml is valid Knative Service YAML pointing at :7860
- deploy/render.yaml is a Render Blueprint with secret env wiring
- deploy/fly.toml has the right service shape
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------- requirements.txt ----------


def test_requirements_txt_exists_and_pins_runtime_deps() -> None:
    req = REPO_ROOT / "requirements.txt"
    assert req.is_file(), "requirements.txt must exist for non-Poetry targets"
    text = req.read_text(encoding="utf-8")
    # Provenance header is present.
    assert "Auto-generated" in text
    # Spot-check that the key runtime packages are pinned.
    for pkg in ["fastapi", "uvicorn", "gradio", "openai", "faiss-cpu", "pydantic"]:
        assert re.search(
            rf"^{re.escape(pkg)}==", text, re.MULTILINE | re.IGNORECASE
        ), f"missing pinned {pkg} in requirements.txt"


# ---------- HF Space frontmatter ----------


def test_hf_readme_has_required_frontmatter() -> None:
    path = REPO_ROOT / "huggingface" / "README.md"
    assert path.is_file(), "huggingface/README.md must exist"
    text = path.read_text(encoding="utf-8")
    # YAML frontmatter delimited by --- at the top.
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    assert m, "huggingface/README.md must start with --- YAML frontmatter ---"
    meta = yaml.safe_load(m.group(1))
    assert isinstance(meta, dict)
    # HF Spaces required fields.
    assert meta.get("sdk") == "docker", "HF Spaces SDK must be ``docker``"
    assert meta.get("app_port") == 7860, "HF Spaces app_port must be 7860 (matches Dockerfile)"
    assert meta.get("license"), "HF Spaces requires a license"


# ---------- Cloud Run manifest ----------


@pytest.fixture(scope="module")
def cloudrun() -> dict:  # type: ignore[type-arg]
    path = REPO_ROOT / "deploy" / "cloudrun.yaml"
    assert path.is_file()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_cloudrun_is_knative_service(cloudrun: dict) -> None:  # type: ignore[type-arg]
    assert cloudrun.get("apiVersion") == "serving.knative.dev/v1"
    assert cloudrun.get("kind") == "Service"


def test_cloudrun_exposes_gradio_port(cloudrun: dict) -> None:  # type: ignore[type-arg]
    container = cloudrun["spec"]["template"]["spec"]["containers"][0]
    ports = container.get("ports") or []
    assert any(
        p.get("containerPort") == 7860 for p in ports
    ), f"cloudrun.yaml must expose 7860; got {ports}"


def test_cloudrun_loads_openai_key_from_secret_manager(cloudrun: dict) -> None:  # type: ignore[type-arg]
    container = cloudrun["spec"]["template"]["spec"]["containers"][0]
    env = container.get("env") or []
    openai_entry = next((e for e in env if e.get("name") == "OPENAI_API_KEY"), None)
    assert openai_entry is not None, "cloudrun.yaml must declare OPENAI_API_KEY"
    assert openai_entry.get("valueFrom", {}).get(
        "secretKeyRef"
    ), "OPENAI_API_KEY must come from a secretKeyRef, not a plain value"


# ---------- Render Blueprint ----------


@pytest.fixture(scope="module")
def render_bp() -> dict:  # type: ignore[type-arg]
    path = REPO_ROOT / "deploy" / "render.yaml"
    assert path.is_file()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_render_declares_docker_runtime(render_bp: dict) -> None:  # type: ignore[type-arg]
    svc = render_bp["services"][0]
    assert svc.get("runtime") == "docker"
    assert svc.get("type") == "web"


def test_render_openai_key_is_secret(render_bp: dict) -> None:  # type: ignore[type-arg]
    """sync: false marks the value as dashboard-managed and never in git."""
    svc = render_bp["services"][0]
    env = svc.get("envVars") or []
    entry = next((e for e in env if e.get("key") == "OPENAI_API_KEY"), None)
    assert entry is not None, "render.yaml must declare OPENAI_API_KEY"
    assert (
        entry.get("sync") is False
    ), "OPENAI_API_KEY must have sync: false so it's set in the dashboard"


def test_render_health_check_path_set(render_bp: dict) -> None:  # type: ignore[type-arg]
    svc = render_bp["services"][0]
    assert svc.get("healthCheckPath") == "/"


# ---------- Fly.toml ----------


@pytest.fixture(scope="module")
def flytoml() -> dict:  # type: ignore[type-arg]
    path = REPO_ROOT / "deploy" / "fly.toml"
    assert path.is_file()
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_fly_has_http_service_on_gradio_port(flytoml: dict) -> None:  # type: ignore[type-arg]
    svc = flytoml.get("http_service") or {}
    assert (
        svc.get("internal_port") == 7860
    ), f"fly.toml http_service.internal_port must be 7860; got {svc.get('internal_port')}"
    assert svc.get("force_https") is True


def test_fly_does_not_hardcode_openai_key(flytoml: dict) -> None:  # type: ignore[type-arg]
    """OPENAI_API_KEY must NOT appear in fly.toml — the spec says set
    it via ``fly secrets set`` so it never lands in git."""
    env = flytoml.get("env") or {}
    assert (
        "OPENAI_API_KEY" not in env
    ), "OPENAI_API_KEY must NOT be in fly.toml — use ``fly secrets set``"


def test_fly_scales_to_zero(flytoml: dict) -> None:  # type: ignore[type-arg]
    svc = flytoml.get("http_service") or {}
    assert svc.get("auto_stop_machines") == "stop"
    assert svc.get("min_machines_running") == 0


# ---------- cross-manifest sanity ----------


def test_all_targets_use_port_7860() -> None:
    """The Phase 16 spec invariant: same image, same port. Every
    target's externally-visible port must match the Dockerfile's
    GRADIO_PORT default."""
    # HF: app_port in frontmatter
    hf_text = (REPO_ROOT / "huggingface" / "README.md").read_text(encoding="utf-8")
    assert "app_port: 7860" in hf_text

    # Cloud Run: containerPort
    crun = yaml.safe_load((REPO_ROOT / "deploy" / "cloudrun.yaml").read_text(encoding="utf-8"))
    cport = crun["spec"]["template"]["spec"]["containers"][0]["ports"][0]["containerPort"]
    assert cport == 7860

    # Render: PORT env value
    render_bp = yaml.safe_load((REPO_ROOT / "deploy" / "render.yaml").read_text(encoding="utf-8"))
    port_entry = next(e for e in render_bp["services"][0]["envVars"] if e["key"] == "PORT")
    assert port_entry["value"] == "7860"

    # Fly: http_service.internal_port
    fly = tomllib.loads((REPO_ROOT / "deploy" / "fly.toml").read_text(encoding="utf-8"))
    assert fly["http_service"]["internal_port"] == 7860
