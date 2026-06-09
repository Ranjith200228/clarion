"""``clarion.eval`` — canonical CLI namespace for the Phase 13 evaluation harness.

Spec command::

    python -m clarion.eval --customer ophthalmology

Implementation lives in ``clarion.evaluation``; this package is a thin
wrapper that exists so the spec's published command works verbatim
without typing the longer ``clarion.evaluation.cli`` path. The legacy
``python -m clarion.evaluation.cli run <customer>`` form still works.
"""

from clarion.eval.cli import main

__all__ = ["main"]
