"""Cost & SLO invoice OCR widget.

Lives at the bottom of the Cost & SLO tab. Three Gradio components
wired together:

  upload_in   gr.File    image upload (PNG / JPEG / WebP)
  extract_btn gr.Button  trigger
  result_html gr.HTML    structured render of line items + total

Hits the FastAPI route ``POST /cost/extract-invoice``; the route
owns the OpenAI Vision call so the Gradio process stays light.
This module renders the response into a tenant-accent styled card
that visually belongs to the Cost & SLO dashboard above it.
"""

from __future__ import annotations

import html as _html
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

log = logging.getLogger(__name__)

_DEFAULT_BACKEND = os.environ.get("CLARION_API_URL", "http://localhost:8000")
_PLACEHOLDER_HTML = (
    '<div class="clarion-ocr-empty">'
    '<div class="clarion-ocr-empty-glyph">&#9783;</div>'
    '<div class="clarion-ocr-empty-text">'
    "Upload a vendor invoice image to lift every dollar amount into "
    "a structured row. Supported: PNG, JPEG, WebP. Max 5 MB."
    "</div>"
    "</div>"
)


@dataclass
class CostOCRTab:
    upload: gr.File
    extract_btn: gr.Button
    result_html: gr.HTML


def build(backend_url: str | None = None) -> CostOCRTab:
    """Mount the OCR widget under the Cost & SLO rollup."""
    api_url = backend_url or _DEFAULT_BACKEND

    gr.HTML(
        '<div class="clarion-ocr-header">'
        '<div class="clarion-ocr-eyebrow">Invoice OCR &middot; powered by '
        "gpt-4o-mini vision</div>"
        '<div class="clarion-ocr-headline">Lift line-items off any '
        "vendor invoice.</div>"
        '<div class="clarion-ocr-sub">Drop an image, get a structured '
        "breakdown of every dollar amount with running total.</div>"
        "</div>"
    )

    with gr.Row(equal_height=False):
        upload = gr.File(
            label="Invoice image",
            file_count="single",
            file_types=[".png", ".jpg", ".jpeg", ".webp", ".gif"],
            type="filepath",
        )
        extract_btn = gr.Button(
            "Extract line items",
            variant="primary",
            elem_classes="clarion-ocr-extract",
        )

    result_html = gr.HTML(_PLACEHOLDER_HTML, elem_id="clarion-ocr-result")

    def _extract(file_path: str | None) -> str:
        """Click handler - POST the file to FastAPI, render the JSON
        response as the structured invoice card."""
        if not file_path:
            return _render_error("Pick an image first.")
        path = Path(file_path)
        if not path.is_file():
            return _render_error(f"File not found: {path.name}")

        content_type = _guess_content_type(path.suffix)
        try:
            with path.open("rb") as fh:
                files = {"file": (path.name, fh, content_type)}
                rsp = httpx.post(
                    f"{api_url}/cost/extract-invoice",
                    files=files,
                    timeout=60.0,
                )
        except httpx.HTTPError as exc:
            log.warning("OCR upload failed: %s", exc)
            return _render_error(
                f"Could not reach the OCR backend at {api_url}. "
                f"Detail: {exc}"
            )

        if rsp.status_code >= 400:
            try:
                detail = rsp.json().get("detail")
            except Exception:
                detail = rsp.text
            return _render_error(f"OCR rejected ({rsp.status_code}): {detail}")

        try:
            payload = rsp.json()
        except Exception:
            return _render_error("OCR returned non-JSON; try a clearer image.")

        return _render_result(payload, source_name=path.name)

    extract_btn.click(
        fn=_extract,
        inputs=[upload],
        outputs=[result_html],
    )

    return CostOCRTab(upload=upload, extract_btn=extract_btn, result_html=result_html)


# ---------- rendering helpers ----------


def _guess_content_type(suffix: str) -> str:
    s = suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(s, "image/png")


def _render_error(message: str) -> str:
    return (
        '<div class="clarion-ocr-result clarion-ocr-error">'
        '<div class="clarion-ocr-error-title">'
        '<span aria-hidden="true">&#9888;</span> OCR failed'
        "</div>"
        f'<div class="clarion-ocr-error-body">{_html.escape(message)}</div>'
        "</div>"
    )


def _render_result(payload: dict[str, Any], *, source_name: str) -> str:
    """Render the FastAPI ExtractInvoiceResponse as a styled card."""
    items = payload.get("line_items") or []
    total = payload.get("total")
    currency = payload.get("currency") or "USD"
    raw_text = payload.get("raw_text") or ""

    if not items and total is None:
        return _render_error(
            "No dollar amounts found in this image. "
            "Try a clearer photo of the invoice."
        )

    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency.upper(), "$")
    n = len(items)
    rows_html = ""
    line_total = 0.0
    for it in items:
        label = it.get("label") or "—"
        value = float(it.get("value") or 0.0)
        line_total += value
        rows_html += (
            '<div class="clarion-ocr-row">'
            f'<div class="clarion-ocr-row-label">{_html.escape(str(label))}</div>'
            f'<div class="clarion-ocr-row-value">{symbol}{value:,.2f}</div>'
            "</div>"
        )

    if total is not None:
        extracted_total = total
        total_label, total_source = "Total", "from invoice"
    else:
        extracted_total = line_total
        total_label, total_source = "Sum", "line items"
    total_html = (
        '<div class="clarion-ocr-total-row">'
        '<div class="clarion-ocr-total-label">'
        f"{total_label} "
        f'<span class="clarion-ocr-total-source">({total_source})</span>'
        "</div>"
        f'<div class="clarion-ocr-total-value">{symbol}{extracted_total:,.2f}</div>'
        "</div>"
    )

    raw_preview = (raw_text[:600] + "…") if len(raw_text) > 600 else raw_text

    return (
        '<div class="clarion-ocr-result">'
        '<div class="clarion-ocr-result-header">'
        '<div class="clarion-ocr-result-source">'
        f'<span class="clarion-ocr-pill">'
        f"{n} line item{'s' if n != 1 else ''}</span>"
        f'<span class="clarion-ocr-source-name">{_html.escape(source_name)}</span>'
        "</div>"
        "</div>"
        '<div class="clarion-ocr-rows">'
        + rows_html
        + total_html
        + "</div>"
        '<details class="clarion-ocr-raw">'
        "<summary>Raw extracted text</summary>"
        f'<pre class="clarion-ocr-raw-pre">{_html.escape(raw_preview)}</pre>'
        "</details>"
        "</div>"
    )
