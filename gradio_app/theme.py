"""Mission-control theme for the Clarion Gradio app.

Defines :data:`CLARION_THEME` — a :class:`gradio.themes.Base` subclass
whose constructor arguments + ``.set()`` overrides pin every Gradio
color knob to our brand palette. The companion file ``style.css``
ships the CSS variables + class primitives our custom ``gr.HTML``
blocks reference; the theme defined here is *just* the part Gradio's
built-in widgets respect (buttons, sliders, the dropdown chrome,
etc.).

Two layers of control:

- :class:`gradio.themes.Base` lets us override its color hubs at
  construct time. Anything Gradio renders via its theme tokens
  (button fill, slider track, accordion border) picks these up.
- :func:`gradio.Blocks.__init__` takes a ``css=`` argument that
  injects our stylesheet at the top of the rendered HTML head.
  That's where our :data:`CSS` constant goes.

The split is intentional: we do *not* selector-hunt Gradio's
internal class names. A Gradio version bump can change those
without notice; instead, we style INSIDE our own ``gr.HTML``
blocks via the classes in ``style.css``.

Single source of truth for color tokens lives in
:data:`PALETTE` (Python) — the CSS file mirrors the same values
in its ``:root`` block. The snapshot tests assert the two never
drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import gradio as gr

# ---------- palette ----------

# Single dict so the theme constructor + the snapshot tests +
# the (future) component builders all read from one place.
PALETTE: Final[dict[str, str]] = {
    # Surfaces — dark mode default, light mode toggled via .theme-light root class.
    "bg":            "#0B1220",  # near-black navy
    "bg_panel":      "#111827",  # raised panels
    "bg_subtle":     "#1F2937",  # hover / striped rows
    "border":        "#1F2937",  # 1 px dividers
    "border_strong": "#374151",
    # Text.
    "text":          "#F1F5F9",  # body
    "text_strong":   "#FFFFFF",  # headings + KPI values
    "text_muted":    "#CBD5E1",  # labels + helper copy
    # Brand accent — cyan from the v2 plan.
    "accent":        "#06B6D4",
    "accent_dim":    "#0E7490",
    # Status — green/amber/red triad.
    "healthy":       "#10B981",
    "warning":       "#F59E0B",
    "critical":      "#EF4444",
    "info":          "#3B82F6",
    # Mono surface — for trace IDs, conv IDs, cost values.
    "mono_fg":       "#A5F3FC",  # cyan-tinged so monospace pops on dark
}

# Type scale + spacing tokens kept here so a single import sees the
# whole visual contract. Matches the CSS variables 1:1.
TYPE_SCALE: Final[dict[str, str]] = {
    "size_xs":    "0.75rem",
    "size_sm":    "0.875rem",
    "size_md":    "1rem",
    "size_lg":    "1.25rem",
    "size_xl":    "1.75rem",
    "size_2xl":   "2.25rem",
    "size_kpi":   "2.5rem",
    "weight_regular":  "400",
    "weight_medium":   "500",
    "weight_semibold": "600",
    "weight_bold":     "700",
}

SPACING: Final[dict[str, str]] = {
    "space_1": "4px",
    "space_2": "8px",
    "space_3": "12px",
    "space_4": "16px",
    "space_5": "24px",
    "space_6": "32px",
    "space_7": "48px",
    # Radii.
    "radius_sm": "4px",
    "radius_md": "8px",
    "radius_lg": "12px",
    "radius_pill": "999px",
}


# ---------- stylesheet loader ----------

_CSS_PATH = Path(__file__).with_name("style.css")


def load_css() -> str:
    """Read ``style.css`` next to this module and return its contents.

    The stylesheet is loaded once per app construction by
    ``gr.Blocks(css=CSS)``. We don't bundle the CSS as a Python
    string literal because keeping it in its own file lets editors
    syntax-highlight + lint it.
    """
    return _CSS_PATH.read_text(encoding="utf-8")


# The constant the app factory imports.
CSS: Final[str] = load_css()


# ---------- Gradio Theme ----------

# Why gr.themes.Base and not Soft / Default: Soft applies its own
# accent colors at multiple darkness levels we'd have to override
# 30 times; Base is the cleanest canvas to set our values on.
def make_theme() -> gr.themes.Base:
    """Construct the Clarion Gradio Theme.

    Sets only the knobs Gradio's built-in widgets respect (button
    fill, slider track, accordion border, dropdown chrome). The bulk
    of our visual identity lives in :data:`CSS`, applied via
    ``gr.Blocks(css=CSS)``.
    """
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#ECFEFF",
            c100="#CFFAFE",
            c200="#A5F3FC",
            c300="#67E8F9",
            c400="#22D3EE",
            c500=PALETTE["accent"],
            c600=PALETTE["accent_dim"],
            c700="#155E75",
            c800="#164E63",
            c900="#083344",
            c950="#042F33",
        ),
        neutral_hue=gr.themes.Color(
            c50="#F9FAFB",
            c100="#F3F4F6",
            c200="#E5E7EB",
            c300="#D1D5DB",
            c400="#9CA3AF",
            c500="#6B7280",
            c600="#4B5563",
            c700="#374151",
            c800="#1F2937",
            c900="#111827",
            c950=PALETTE["bg"],
        ),
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
    )
    # ``.set(...)`` overrides specific tokens. Anything we name here
    # binds tighter than the hue defaults above. Sparing use — only
    # for surfaces Gradio's built-in chrome paints.
    theme.set(
        body_background_fill=PALETTE["bg"],
        body_background_fill_dark=PALETTE["bg"],
        background_fill_primary=PALETTE["bg_panel"],
        background_fill_primary_dark=PALETTE["bg_panel"],
        background_fill_secondary=PALETTE["bg_subtle"],
        background_fill_secondary_dark=PALETTE["bg_subtle"],
        body_text_color=PALETTE["text"],
        body_text_color_dark=PALETTE["text"],
        block_label_text_color=PALETTE["text_muted"],
        block_label_text_color_dark=PALETTE["text_muted"],
        block_title_text_color=PALETTE["text_strong"],
        block_title_text_color_dark=PALETTE["text_strong"],
        border_color_primary=PALETTE["border"],
        border_color_primary_dark=PALETTE["border"],
    )
    return theme


# Construct once at import time so the app factory and tests share
# the same instance.
CLARION_THEME: Final[gr.themes.Base] = make_theme()


__all__ = [
    "CLARION_THEME",
    "CSS",
    "PALETTE",
    "SPACING",
    "TYPE_SCALE",
    "load_css",
    "make_theme",
]
