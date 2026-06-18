"""Theme + stylesheet contract tests.

These guard the single source of truth claim made in
:mod:`gradio_app.theme`: the Python ``PALETTE`` dict and the CSS
variables in ``style.css`` must never drift apart. A snapshot test
parses the stylesheet's ``:root`` block and asserts every color
token matches its Python counterpart.

Also asserts the theme constructs cleanly and exposes the values
the app factory expects.
"""

from __future__ import annotations

import re

import gradio as gr

from gradio_app.theme import (
    CLARION_THEME,
    CSS,
    PALETTE,
    SPACING,
    TYPE_SCALE,
    load_css,
    make_theme,
)


def test_theme_is_gradio_base_instance() -> None:
    assert isinstance(CLARION_THEME, gr.themes.Base)


def test_make_theme_returns_fresh_instance() -> None:
    a = make_theme()
    b = make_theme()
    assert a is not b
    # Both should still satisfy the contract.
    assert isinstance(a, gr.themes.Base)
    assert isinstance(b, gr.themes.Base)


def test_css_loads_from_disk_and_is_non_empty() -> None:
    css = load_css()
    assert css == CSS
    assert len(css) > 1000  # smoke: real stylesheet, not a stub
    # Sanity: three layer headers we promised exist.
    assert "Layer 1: tokens" in css
    assert "Layer 2: primitives" in css
    assert "Layer 3: layouts" in css


def test_palette_has_required_keys() -> None:
    required = {
        "bg", "bg_panel", "bg_subtle", "border", "border_strong",
        "text", "text_strong", "text_muted",
        "accent", "accent_dim",
        "healthy", "warning", "critical", "info",
        "mono_fg",
    }
    assert required.issubset(PALETTE.keys())


def test_palette_values_are_hex() -> None:
    hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for name, value in PALETTE.items():
        assert hex_pattern.match(value), f"PALETTE[{name!r}] = {value!r} is not 6-digit hex"


# ---------- the load-bearing snapshot: PALETTE <-> :root variables ----------


# Map Python PALETTE keys → CSS variable names. The CSS file must
# carry these exact names in :root; mismatched names = drift.
_PALETTE_TO_CSS_VAR = {
    "bg":            "--c-bg",
    "bg_panel":      "--c-bg-panel",
    "bg_subtle":     "--c-bg-subtle",
    "border":        "--c-border",
    "border_strong": "--c-border-strong",
    "text":          "--c-text",
    "text_strong":   "--c-text-strong",
    "text_muted":    "--c-text-muted",
    "accent":        "--c-accent",
    "accent_dim":    "--c-accent-dim",
    "healthy":       "--c-healthy",
    "warning":       "--c-warning",
    "critical":      "--c-critical",
    "info":          "--c-info",
    "mono_fg":       "--c-mono-fg",
}


def _root_block(css: str) -> str:
    match = re.search(r":root\s*\{([^}]+)\}", css, re.DOTALL)
    assert match, "stylesheet has no :root block"
    return match.group(1)


def _parse_vars(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    # Match "    --c-something: #VALUE;" lines.
    for var, val in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", block):
        out[var.strip()] = val.strip()
    return out


def test_css_root_block_mirrors_palette_exactly() -> None:
    block = _root_block(CSS)
    parsed = _parse_vars(block)
    for py_key, css_var in _PALETTE_TO_CSS_VAR.items():
        assert css_var in parsed, f"CSS :root missing {css_var}"
        # Compare lowercased so case differences don't bite.
        assert parsed[css_var].lower() == PALETTE[py_key].lower(), (
            f"drift: PALETTE[{py_key!r}] = {PALETTE[py_key]!r} but "
            f":root {css_var} = {parsed[css_var]!r}"
        )


def test_css_carries_every_type_scale_token() -> None:
    block = _root_block(CSS)
    parsed = _parse_vars(block)
    for token, expected in TYPE_SCALE.items():
        # Python keys like "size_xs" map to CSS vars like "--fs-xs"
        # ("--fw-..." for the weight family).
        if token.startswith("size_"):
            css_var = "--fs-" + token.removeprefix("size_")
        elif token.startswith("weight_"):
            css_var = "--fw-" + token.removeprefix("weight_")
        else:
            continue
        assert css_var in parsed, f"CSS :root missing {css_var}"
        assert parsed[css_var] == expected, (
            f"drift: TYPE_SCALE[{token!r}] = {expected!r} but "
            f":root {css_var} = {parsed[css_var]!r}"
        )


def test_css_carries_every_spacing_token() -> None:
    block = _root_block(CSS)
    parsed = _parse_vars(block)
    for token, expected in SPACING.items():
        if token.startswith("space_"):
            css_var = "--sp-" + token.removeprefix("space_")
        elif token.startswith("radius_"):
            css_var = "--r-" + token.removeprefix("radius_")
        else:
            continue
        assert css_var in parsed, f"CSS :root missing {css_var}"
        assert parsed[css_var] == expected, (
            f"drift: SPACING[{token!r}] = {expected!r} but "
            f":root {css_var} = {parsed[css_var]!r}"
        )


def test_light_mode_override_exists() -> None:
    """Phase G wires the toggle; the override block must already be in place."""
    assert ".theme-light" in CSS
