"""Phase 1 smoke tests — package imports and version is exposed."""

from __future__ import annotations

import clarion


def test_package_importable() -> None:
    assert clarion is not None


def test_version_string() -> None:
    assert isinstance(clarion.__version__, str)
    # On the 1.1.x development line. Released tags pin a stable
    # version; this assertion just confirms the marketing version
    # is exposed and follows the dev-suffix convention.
    assert clarion.__version__.startswith("1.")
    assert "dev" in clarion.__version__
