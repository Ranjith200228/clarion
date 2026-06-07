"""``GET /health`` — liveness probe.

Returns ``status: ok`` plus the loaded customer ids and the package
version. Used by Cloud Run / k8s probes and as a sanity check after
deploy.
"""

from __future__ import annotations

import clarion
from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness probe",
)
def health(request: Request) -> HealthResponse:
    """Always returns ``status=ok`` if the process is up.

    ``customers_loaded`` lists which customer configs have been touched
    so far — useful for confirming a warmup ran on the right tenants.
    """
    sessions = request.app.state.sessions
    return HealthResponse(
        status="ok",
        version=clarion.__version__,
        customers_loaded=sessions.loaded_customer_ids(),
    )
