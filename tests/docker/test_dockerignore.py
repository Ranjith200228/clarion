"""Structural tests for ``.dockerignore``.

The .dockerignore decides what enters the Docker build context. A
regression that lets ``.venv`` or ``.git`` slip in would balloon the
build context to half a gigabyte. These tests assert the patterns the
Phase 15 spec depends on are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def patterns() -> set[str]:
    assert DOCKERIGNORE.is_file(), f".dockerignore not found at {DOCKERIGNORE}"
    return {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


# ---------- must-have excludes ----------


@pytest.mark.parametrize(
    "pat",
    [
        ".git/",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".coverage",
    ],
)
def test_excludes_pattern(patterns: set[str], pat: str) -> None:
    assert pat in patterns, (
        f"missing exclude {pat!r} — would balloon the docker build context"
    )


def test_excludes_per_customer_sqlite_and_faiss(patterns: set[str]) -> None:
    """Generated indices must be rebuilt in the container, not copied
    from the host (host binaries may not match the container Python ABI)."""
    expected = {
        "data/ophthalmology/structured.sqlite",
        "data/ophthalmology/rules.faiss",
        "data/ophthalmology/rules_meta.json",
        "data/orthopedics/structured.sqlite",
        "data/orthopedics/rules.faiss",
        "data/orthopedics/rules_meta.json",
    }
    missing = expected - patterns
    assert not missing, f"missing per-customer excludes: {sorted(missing)}"


def test_excludes_local_dev_secrets(patterns: set[str]) -> None:
    """.env files contain dev secrets; never bake them into an image."""
    assert ".env" in patterns
    assert ".env.*" in patterns
    # And the example file gets a re-include so docs ship.
    assert "!.env.example" in patterns
