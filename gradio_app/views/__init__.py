"""v2 mission-control views.

Each module in this package is a single page. Public surface:

- ``build_html(...) -> str`` — assemble the static HTML for the
  view from typed rollups; returned string is wrapped in a
  ``gr.HTML`` by ``gradio_app.app.build_app``.
- ``empty_html() -> str`` — render the same view in its
  empty-state form (no data on disk).

Views never read JSON themselves. They take typed dataclasses from
``gradio_app.data_sources`` and render them via the primitives in
``gradio_app.components``.
"""
