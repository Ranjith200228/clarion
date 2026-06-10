"""Phase 14 Gradio product UI.

Four tabs (Live Agent, Quality Metrics, Escalations, Trace Explorer) +
a customer switcher. Reads ``report_<customer>.json`` and
``trace_<customer>.json`` produced by ``python -m clarion.eval``.
No business logic; no metric computation. The tabs render fields off
the typed Pydantic objects loaded by ``gradio_app.data``.

Entry point: ``python -m gradio_app``.
"""

__all__: list[str] = []
