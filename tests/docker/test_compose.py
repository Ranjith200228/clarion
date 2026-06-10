"""Structural tests for the Phase 15 docker-compose.yml.

Parses ``docker-compose.yml`` as YAML and asserts the Phase 15
acceptance bullets:

- Two services (api, gradio) sharing the same image
- gradio depends_on api with condition: service_healthy
- Both services declare a healthcheck
- Both expose the right ports
- Required env vars wired
- restart policy is set

Compose-level smoke run (``docker compose up``) is documented in the
README; this suite just guards the spec contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    assert COMPOSE.is_file(), f"docker-compose.yml not found at {COMPOSE}"
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


# ---------- services ----------


def test_has_api_and_gradio_services(compose: dict[str, Any]) -> None:
    services = compose.get("services") or {}
    assert "api" in services, "compose must define an ``api`` service"
    assert "gradio" in services, "compose must define a ``gradio`` service"


def test_services_share_same_image_tag(compose: dict[str, Any]) -> None:
    """Two services, one image. Phase 15 architecture decision."""
    services = compose["services"]
    api_image = services["api"].get("image")
    gradio_image = services["gradio"].get("image")
    assert api_image is not None, "api service must declare image"
    assert gradio_image is not None, "gradio service must declare image"
    assert api_image == gradio_image, (
        f"services must share one image; got api={api_image!r}, "
        f"gradio={gradio_image!r}"
    )


def test_both_services_build_runtime_target(compose: dict[str, Any]) -> None:
    """Both services build from the ``runtime`` target so the test
    stage's heavyweight dev deps don't leak into production."""
    for name in ("api", "gradio"):
        svc = compose["services"][name]
        build = svc.get("build")
        assert isinstance(build, dict), f"{name}: build must be a mapping"
        assert build.get("target") == "runtime", (
            f"{name}: build.target must be ``runtime``, got {build.get('target')!r}"
        )


# ---------- depends_on + health ----------


def test_gradio_depends_on_api_healthy(compose: dict[str, Any]) -> None:
    gradio = compose["services"]["gradio"]
    depends = gradio.get("depends_on")
    assert isinstance(depends, dict), (
        "gradio.depends_on must be a mapping with condition: service_healthy"
    )
    api_dep = depends.get("api")
    assert isinstance(api_dep, dict), "gradio.depends_on.api must be a mapping"
    assert api_dep.get("condition") == "service_healthy", (
        "gradio must wait for api to be healthy before starting"
    )


def test_both_services_declare_healthcheck(compose: dict[str, Any]) -> None:
    for name, expected_url_fragment in (
        ("api", "8000/health"),
        ("gradio", "7860/"),
    ):
        svc = compose["services"][name]
        hc = svc.get("healthcheck")
        assert isinstance(hc, dict), f"{name}: must declare a healthcheck"
        test_cmd = hc.get("test")
        # `test` may be a list like ["CMD", "bash", "scripts/healthcheck.sh", url]
        # or a single shell string. Normalize to a joined string for the search.
        joined = (
            " ".join(str(p) for p in test_cmd)
            if isinstance(test_cmd, list)
            else str(test_cmd)
        )
        assert "healthcheck.sh" in joined, (
            f"{name}.healthcheck.test should call scripts/healthcheck.sh; got {joined!r}"
        )
        assert expected_url_fragment in joined, (
            f"{name}.healthcheck.test should target {expected_url_fragment}; got {joined!r}"
        )


# ---------- ports ----------


def test_api_exposes_8000_and_gradio_exposes_7860(compose: dict[str, Any]) -> None:
    api_ports = compose["services"]["api"].get("ports") or []
    gradio_ports = compose["services"]["gradio"].get("ports") or []
    assert any("8000" in str(p) for p in api_ports), (
        f"api must publish 8000; got {api_ports}"
    )
    assert any("7860" in str(p) for p in gradio_ports), (
        f"gradio must publish 7860; got {gradio_ports}"
    )


# ---------- env wiring ----------


def test_gradio_points_at_api_via_internal_dns(compose: dict[str, Any]) -> None:
    """The CLARION_API_URL the gradio service uses must reach the api
    service over compose's internal DNS, NOT via host port mapping."""
    env = compose["services"]["gradio"].get("environment") or {}
    url = env.get("CLARION_API_URL")
    assert url is not None, "gradio.environment must set CLARION_API_URL"
    assert "://api" in str(url), (
        f"CLARION_API_URL must use compose service name ``api``; got {url!r}"
    )


def test_both_services_set_clarion_data_dir(compose: dict[str, Any]) -> None:
    for name in ("api", "gradio"):
        env = compose["services"][name].get("environment") or {}
        assert env.get("CLARION_DATA_DIR"), (
            f"{name}.environment must set CLARION_DATA_DIR"
        )


def test_openai_key_passes_through_from_host_env(compose: dict[str, Any]) -> None:
    """OPENAI_API_KEY must be interpolated from host env via ${OPENAI_API_KEY},
    never hardcoded into the compose file."""
    raw = COMPOSE.read_text(encoding="utf-8")
    assert "${OPENAI_API_KEY" in raw, (
        "compose must read OPENAI_API_KEY from the host env, not hardcode it"
    )


# ---------- robustness ----------


def test_both_services_restart_unless_stopped(compose: dict[str, Any]) -> None:
    for name in ("api", "gradio"):
        restart = compose["services"][name].get("restart")
        assert restart == "unless-stopped", (
            f"{name}.restart should be ``unless-stopped`` for production; got {restart!r}"
        )
