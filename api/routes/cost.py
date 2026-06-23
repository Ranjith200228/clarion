"""``POST /cost/extract-invoice`` - OCR a single invoice image.

The Cost & SLO dashboard's "Upload invoice" widget posts a single
image (PNG / JPEG / WebP) here; the response is a structured
ExtractedInvoice payload the front-end renders as the "Uploaded
invoices" panel below the static cost rollup.

Multimodal stack: gpt-4o-mini Vision via clarion.modules.invoice_ocr.
The route is intentionally thin - it owns request validation +
HTTP-shape concerns; all the LLM + parsing logic lives in the
module so unit tests can hit it without spinning up FastAPI.
"""

from __future__ import annotations

import logging

from clarion.modules.invoice_ocr import (
    ExtractedInvoice,
    InvoiceOCRError,
    extract_invoice,
)
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.schemas import ErrorResponse

log = logging.getLogger(__name__)
router = APIRouter()


_UPLOAD_DEFAULT = File(..., description="Invoice image (PNG/JPEG/WebP)")

# Hard cap on uploads. gpt-4o-mini's image budget is generous but
# we keep our exposure small for a portfolio demo - if the user
# uploads a 50MB document we reject early rather than burning a
# tokenful call.
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
_ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}


class InvoiceLineItemDto(BaseModel):
    label: str = Field(..., description="Line-item label as printed on the invoice.")
    value: float = Field(..., description="Dollar amount, no symbols, no thousands separators.")


class ExtractInvoiceResponse(BaseModel):
    ok: bool = True
    raw_text: str = Field(..., description="Full plain text of the invoice as OCRed.")
    line_items: list[InvoiceLineItemDto] = Field(
        default_factory=list,
        description="Every labelled dollar amount the model lifted.",
    )
    total: float | None = Field(
        default=None,
        description="Invoice total in the document's currency.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO-ish currency code if detected (USD / EUR / GBP).",
    )


@router.post(
    "/cost/extract-invoice",
    response_model=ExtractInvoiceResponse,
    tags=["cost"],
    summary="OCR an invoice image into structured line items",
    responses={
        400: {"model": ErrorResponse, "description": "Bad upload"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        503: {"model": ErrorResponse, "description": "OCR pipeline misconfigured"},
    },
)
async def extract_invoice_route(
    file: UploadFile = _UPLOAD_DEFAULT,
) -> ExtractInvoiceResponse:
    content_type = (file.content_type or "image/png").lower()
    if content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type {content_type!r}; expected "
                "image/png, image/jpeg, image/webp, or image/gif."
            ),
        )

    data = await file.read()
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is {len(data) // 1024} KB; max allowed is "
                f"{_MAX_IMAGE_BYTES // 1024} KB."
            ),
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        result: ExtractedInvoice = extract_invoice(
            data,
            content_type="image/jpeg" if content_type == "image/jpg" else content_type,
        )
    except InvoiceOCRError as exc:
        log.info("invoice OCR rejected: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ExtractInvoiceResponse(
        ok=True,
        raw_text=result.raw_text,
        line_items=[
            InvoiceLineItemDto(label=it.label, value=float(it.value))
            for it in result.line_items
        ],
        total=float(result.total) if result.total is not None else None,
        currency=result.currency,
    )


# Re-exports for callers that need the typed Decimal-bearing version.
__all__ = ["router", "ExtractInvoiceResponse", "InvoiceLineItemDto"]
