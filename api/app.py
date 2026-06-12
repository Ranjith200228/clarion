"""FastAPI app factory + uvicorn entry point.

Use ``create_app()`` from tests to spin up isolated instances pointing at
tmp data dirs; use ``python -m api.app`` (or ``uvicorn api.app:app``) in
production / dev. The factory accepts a custom ``SessionManager`` so
tests can wire a FakeLLM factory without monkeypatching.
"""

from __future__ import annotations

import argparse
import logging
import sys

import clarion
from clarion.config import Settings, get_settings
from clarion.observability import configure_logging
from fastapi import FastAPI

from api.middleware import CorrelationIdMiddleware
from api.routes.chat import router as chat_router
from api.routes.evaluate import router as evaluate_router
from api.routes.health import router as health_router
from api.routes.voice import router as voice_router
from api.sessions import SessionManager, make_session_manager

log = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    sessions: SessionManager | None = None,
    voice_orchestrator: object | None = None,
    install_logging: bool = True,
) -> FastAPI:
    """Build a FastAPI app instance.

    Args:
        settings: override deployment Settings (test fixtures pass a
            tmp-path Settings here).
        sessions: provide a pre-built SessionManager; useful for tests
            that need to swap in a FakeLLM factory.
    """
    settings = settings or get_settings()
    if install_logging:
        # Idempotent — safe to call from tests that build many apps
        # within one process.
        configure_logging()
    if sessions is None:
        sessions = make_session_manager(settings)

    app = FastAPI(
        title="Clarion API",
        description=(
            "Configurable Multi-Agent Voice Automation Platform with "
            "Sentinel Trust Engine. Healthcare scheduling is the "
            "demonstration vertical; the platform is vertical-agnostic."
        ),
        version=clarion.__version__,
        contact={
            "name": "Clarion",
            "url": "https://github.com/Ranjith200228/clarion",
        },
        license_info={"name": "MIT"},
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.state.settings = settings
    app.state.sessions = sessions
    # Module M5 — when no orchestrator is injected, POST /voice/turn
    # responds 503 "voice_not_configured". Deployments that enable
    # voice construct a VoiceOrchestrator(transcriber=..., speaker=...)
    # and pass it here.
    app.state.voice_orchestrator = voice_orchestrator

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(evaluate_router)
    app.include_router(voice_router)
    return app


# Production singleton — uvicorn imports this attribute by default
# (``uvicorn api.app:app``). Don't instantiate at import time when an
# OPENAI_API_KEY isn't set (CI runs unit tests against create_app
# directly and shouldn't pay this cost).
def _maybe_app() -> FastAPI | None:
    import os

    if os.environ.get("CLARION_SKIP_AUTOAPP"):
        return None
    return create_app()


app = _maybe_app()


# ---------- CLI entry: ``python -m api.app`` ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clarion-serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true", help="dev: reload on edit")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
