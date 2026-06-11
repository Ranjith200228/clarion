"""Phase 17 documentation contract tests.

The Phase 17 spec mandates five artifacts and 12 README sections. These
tests are structural — they don't grade prose quality, but they catch
the kind of regression where someone deletes ``docs/discovery.md`` or
renames a required README heading.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------- required artifacts (Phase 17 spec) ----------


def test_readme_exists() -> None:
    assert (REPO_ROOT / "README.md").is_file()


def test_discovery_doc_exists() -> None:
    assert (REPO_ROOT / "docs" / "discovery.md").is_file()


def test_architecture_png_exists() -> None:
    path = REPO_ROOT / "docs" / "architecture.png"
    assert path.is_file(), "Phase 17 spec mandates docs/architecture.png"
    # Smoke-check: a real PNG starts with the 8-byte magic header.
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_architecture_mermaid_source_exists() -> None:
    """We keep both Mermaid source + PNG so reviewers have a text-
    diffable artifact and a visual one."""
    assert (REPO_ROOT / "docs" / "architecture.mmd").is_file()


def test_architecture_render_script_exists() -> None:
    """The PNG is deterministically generated; the script must ship
    so the diagram can be regenerated."""
    assert (REPO_ROOT / "scripts" / "render_architecture.py").is_file()


def test_developer_guide_exists() -> None:
    assert (REPO_ROOT / "docs" / "developer_guide.md").is_file()


def test_deployment_guide_exists_at_spec_path() -> None:
    """Phase 17 spec calls for ``docs/deployment_guide.md`` (not the
    old DEPLOYMENT.md path)."""
    assert (REPO_ROOT / "docs" / "deployment_guide.md").is_file()
    assert not (
        REPO_ROOT / "docs" / "DEPLOYMENT.md"
    ).exists(), "old DEPLOYMENT.md should have been removed in Phase 17 rename"


# ---------- README sections (Phase 17 spec) ----------


def test_readme_has_required_sections() -> None:
    """The Phase 17 spec lists 12 README sections by name.

    We assert each mandated section heading appears in the README. The
    headings can be at any level (#, ##) — we just check the substring
    on a line that begins with at least one ``#``.
    """
    required = [
        "Problem",
        "Architecture",
        "Config",  # "Config Driven Design"
        "Customer Onboarding",
        "Trust Engine",
        "Evaluation Harness",
        "Tracing",
        "Metrics",
        "Deployment",
        "Results",
        "Lessons Learned",
        "Future Roadmap",
    ]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    heading_lines = [line.strip() for line in readme.splitlines() if line.lstrip().startswith("#")]
    blob = "\n".join(heading_lines).lower()
    missing = [r for r in required if r.lower() not in blob]
    assert not missing, f"README missing required Phase 17 sections: {missing}"


def test_readme_links_architecture_diagram() -> None:
    """The README must surface the architecture diagram — otherwise
    the visual artifact is orphaned."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "docs/architecture.png" in readme or "docs/architecture.mmd" in readme
    ), "README must link the architecture diagram"


def test_readme_links_to_other_docs() -> None:
    """Cross-links to the four other required docs."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/discovery.md" in readme
    assert "docs/developer_guide.md" in readme
    assert "docs/deployment_guide.md" in readme


# ---------- doc body smoke ----------


def test_discovery_doc_has_risk_register() -> None:
    """The FDE discovery doc should at minimum cover risks; this is the
    key FDE signature artifact."""
    body = (REPO_ROOT / "docs" / "discovery.md").read_text(encoding="utf-8")
    body_lower = body.lower()
    assert "risk" in body_lower
    assert "success metric" in body_lower or "success criteria" in body_lower


def test_developer_guide_covers_extension_recipes() -> None:
    body = (REPO_ROOT / "docs" / "developer_guide.md").read_text(encoding="utf-8")
    body_lower = body.lower()
    assert "adding a new tool" in body_lower
    assert "adding a new customer" in body_lower


def test_deployment_guide_mentions_all_targets() -> None:
    body = (REPO_ROOT / "docs" / "deployment_guide.md").read_text(encoding="utf-8")
    body_lower = body.lower()
    for target in ("hugging face", "cloud run", "render", "fly"):
        assert target in body_lower, f"deployment guide missing {target}"
