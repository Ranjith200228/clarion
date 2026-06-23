"""Patch ``gradio_client.utils`` to survive Pydantic 2.x bool JSON schemas.

Gradio 4.44.1 (and the 4.x line generally) ships a ``gradio_client``
whose ``get_type`` and ``_json_schema_to_python_type`` helpers blow
up with::

    TypeError: argument of type 'bool' is not iterable

when Pydantic emits ``additionalProperties: True`` (or ``False``) —
a common shape for ``Dict[str, Any]`` fields. Gradio's API
self-introspection hits this on every ``HEAD /`` health check and
the container dies in a loop.

The bug is fixed in Gradio 5.x but never backported to 4.x.

This script prepends an ``isinstance(schema, bool)`` short-circuit
to both functions. If the upstream package is ever updated to
include the same guard, our patch becomes redundant (not incorrect)
and the script's assertion will fail loudly so we can drop it.

Run inside the Dockerfile after ``poetry install`` so the patched
file is included in the runtime image layer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _patch(src: str) -> str:
    guard = '\n    if isinstance(schema, bool):\n        return "Any"'

    # The two functions whose first statement reads `if "const" in schema:`
    # or `if schema == {}:` — both fail for bool input. Insert the guard
    # immediately after the signature line.
    patterns = (
        # `def get_type(schema: dict):` or `def get_type(schema):`
        r"(def get_type\(schema[^)]*\)(?:\s*->\s*[^:]+)?:)",
        # `def _json_schema_to_python_type(schema: Any, defs) -> str:`
        r"(def _json_schema_to_python_type\(schema[^)]*\)(?:\s*->\s*[^:]+)?:)",
    )

    for pat in patterns:
        new_src, n = re.subn(pat, r"\1" + guard, src, count=1)
        if n != 1:
            raise SystemExit(
                f"[patch] failed: pattern {pat!r} matched {n} times "
                "— gradio_client signature has drifted. Either drop this "
                "patch (upstream may have fixed the bug) or update the "
                "pattern."
            )
        src = new_src

    return src


def main() -> int:
    import gradio_client.utils as u

    path = Path(u.__file__)
    src = path.read_text()

    sentinel = 'if isinstance(schema, bool):\n        return "Any"'
    if sentinel in src:
        print(f"[patch] gradio_client/utils.py already patched at {path}")
        return 0

    patched = _patch(src)
    path.write_text(patched)
    print(f"[patch] gradio_client/utils.py patched at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
