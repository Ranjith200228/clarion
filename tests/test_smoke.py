"""Phase 1 smoke tests — package imports and version is exposed."""

from __future__ import annotations

import clarion


def test_package_importable() -> None:
    assert clarion is not None


def test_version_string() -> None:
    assert isinstance(clarion.__version__, str)
    assert clarion.__version__ == "0.1.0"
