"""Invoice OCR via OpenAI Vision (gpt-4o-mini multimodal).

User uploads an invoice / receipt image (or PDF page rasterised by
the client). We send it to gpt-4o-mini with a structured-extraction
prompt and parse the JSON response into a typed result:

  ExtractedInvoice(
      raw_text="...",
      line_items=[
          InvoiceLineItem(label="X-ray reading service", value=Decimal("125.00")),
          ...
      ],
      total=Decimal("2,481.50"),
      currency="USD",
  )

Used by the Cost & SLO dashboard - users upload a vendor invoice
image and Clarion lifts every dollar amount into a structured row
for inclusion in the cost rollup. Keeps the heavy LLM/SDK imports
behind a lazy boundary so unit tests don't pay for them.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

log = logging.getLogger(__name__)


_OCR_MODEL = "gpt-4o-mini"
_OCR_PROMPT = (
    "You are an invoice-extraction assistant. Read the attached "
    "invoice or receipt image and return STRICT JSON with this shape:\n"
    "{\n"
    '  "raw_text": "full plain text of the document",\n'
    '  "line_items": [{"label": "<line label>", "value": <number>}, ...],\n'
    '  "total": <number or null>,\n'
    '  "currency": "USD" | "EUR" | "GBP" | null\n'
    "}\n"
    "Rules:\n"
    "- Include EVERY dollar amount you see (line items, subtotal, "
    "tax, total).\n"
    "- Strip currency symbols and thousands separators from numbers; "
    "preserve the decimal point. Example: '$1,234.50' -> 1234.50.\n"
    "- If you cannot find a total, set total to null.\n"
    "- If the image is unreadable or contains no invoice content, "
    'return {"raw_text": "", "line_items": [], "total": null, '
    '"currency": null}.\n'
    "- Output JSON only - no markdown fences, no commentary."
)


@dataclass(frozen=True)
class InvoiceLineItem:
    """One labelled dollar amount lifted off the invoice."""

    label: str
    value: Decimal


@dataclass(frozen=True)
class ExtractedInvoice:
    """Structured result of an OCR pass over one invoice image."""

    raw_text: str
    line_items: tuple[InvoiceLineItem, ...] = field(default_factory=tuple)
    total: Decimal | None = None
    currency: str | None = None


class InvoiceOCRError(RuntimeError):
    """Raised when the OCR pipeline cannot return a usable result.

    Carries a human-readable reason so the API layer can echo it
    back to the client without leaking SDK / vendor specifics.
    """


def extract_invoice(image_bytes: bytes, *, content_type: str = "image/png") -> ExtractedInvoice:
    """Run gpt-4o-mini vision over ``image_bytes``.

    ``content_type`` must be one of image/png, image/jpeg, image/webp,
    or image/gif - the OpenAI Vision endpoint rejects anything else.
    Returns an ``ExtractedInvoice`` on success; raises
    ``InvoiceOCRError`` with a friendly reason otherwise.
    """
    if not image_bytes:
        raise InvoiceOCRError("Empty image upload.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise InvoiceOCRError(
            "OCR requires OPENAI_API_KEY in the environment."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise InvoiceOCRError(
            "openai SDK not installed; pip install openai."
        ) from exc

    client = OpenAI(api_key=api_key)
    data_url = (
        f"data:{content_type};base64,"
        f"{base64.b64encode(image_bytes).decode('ascii')}"
    )

    try:
        rsp = client.chat.completions.create(
            model=_OCR_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
        )
    except Exception as exc:
        log.warning("invoice OCR failed: %s", exc)
        raise InvoiceOCRError(f"Vision call failed: {exc}") from exc

    if not rsp.choices or rsp.choices[0].message.content is None:
        raise InvoiceOCRError("Vision returned an empty response.")
    return _parse_response(rsp.choices[0].message.content)


def _parse_response(content: str) -> ExtractedInvoice:
    """Convert the model's JSON string into a typed ExtractedInvoice.

    Defensive: the model occasionally returns numbers as strings,
    or wraps the JSON in a single quoted code block despite the
    explicit instruction. Strip those and coerce types so the API
    layer never sees a malformed payload.
    """
    text = content.strip()
    # Defensive: strip ```json ... ``` fences if the model added them.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("invoice OCR returned non-JSON: %r", text[:200])
        raise InvoiceOCRError(
            "Vision returned non-JSON; try a clearer image."
        ) from exc

    raw_text = str(data.get("raw_text", ""))
    items_raw = data.get("line_items", []) or []
    items: list[InvoiceLineItem] = []
    for it in items_raw:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label") or "").strip()
        try:
            value = _to_decimal(it.get("value"))
        except (InvalidOperation, ValueError, TypeError):
            continue
        if value is None or not label:
            continue
        items.append(InvoiceLineItem(label=label, value=value))

    total_raw = data.get("total")
    total: Decimal | None
    try:
        total = _to_decimal(total_raw)
    except (InvalidOperation, ValueError, TypeError):
        total = None

    currency = data.get("currency")
    if currency is not None:
        currency = str(currency).upper().strip() or None

    return ExtractedInvoice(
        raw_text=raw_text,
        line_items=tuple(items),
        total=total,
        currency=currency,
    )


def _to_decimal(raw: object) -> Decimal | None:
    """Coerce a JSON number-or-string into ``Decimal``; ``None`` on miss."""
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return Decimal(str(raw))
    if isinstance(raw, str):
        cleaned = raw.strip().replace(",", "").lstrip("$£€")
        if not cleaned:
            return None
        return Decimal(cleaned)
    return None
