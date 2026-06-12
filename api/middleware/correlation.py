"""ASGI middleware that binds a correlation id to every request.

Pulls ``X-Request-Id`` from the inbound headers when present;
otherwise allocates a fresh UUID4 hex. The id is set on the
:data:`clarion.observability.logging` contextvar so every log line
emitted while processing the request carries it, and echoed back as
``X-Request-Id`` on the response so clients can correlate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from clarion.observability import correlation_id_scope, new_correlation_id
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "x-request-id"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Pull / mint an X-Request-Id and bind it to the request scope.

    Constructor takes the standard Starlette ``app`` parameter; the
    only Clarion-side bit is the contextvar binding.
    """

    def __init__(self, app: ASGIApp, *args: Any, **kwargs: Any) -> None:
        super().__init__(app, *args, **kwargs)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        cid = request.headers.get(REQUEST_ID_HEADER) or new_correlation_id()
        with correlation_id_scope(cid):
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = cid
        return response
