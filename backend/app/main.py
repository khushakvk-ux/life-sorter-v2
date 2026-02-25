"""
IKSHAN BACKEND — FastAPI Application Entry Point
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.config import get_settings
from app.middleware.rate_limit import setup_rate_limiter

# ── Structured Logging ─────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if not get_settings().is_production
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "🚀 Ikshan Backend starting",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT.value,
        host=settings.HOST,
        port=settings.PORT,
    )

    # Verify critical configuration
    if not settings.openai_api_key_active:
        logger.warning("⚠️  No OpenAI API key configured — chat/TTS endpoints will fail")
    if not settings.JUSPAY_API_KEY:
        logger.warning("⚠️  No JusPay API key configured — payment endpoints will fail")

    # Pre-load all persona documents so task lookups are instant
    from app.services.persona_doc_service import preload_all_docs
    preload_all_docs()

    yield
    logger.info("🛑 Ikshan Backend shutting down")

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        openapi_url="/openapi.json",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_rate_limiter(app)

    # ── Routers ────────────────────────────────────────────────
    from app.routers import (
        chat,
        companies,
        speak,
        leads,
        payments,
        recommendations,
        ideas,
        legacy,
        agent,
    )

    # CRITICAL: Rebuild models here after routers are loaded
    from app.models.chat import ChatRequest, ChatResponse
    ChatRequest.model_rebuild()
    ChatResponse.model_rebuild()

    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(companies.router, prefix="/api/v1", tags=["Companies"])
    app.include_router(speak.router, prefix="/api/v1", tags=["Text-to-Speech"])
    app.include_router(leads.router, prefix="/api/v1", tags=["Leads"])
    app.include_router(payments.router, prefix="/api/v1", tags=["Payments"])
    app.include_router(recommendations.router, prefix="/api/v1", tags=["Recommendations"])
    app.include_router(ideas.router, prefix="/api/v1", tags=["Ideas"])
    app.include_router(agent.router, prefix="/api/v1", tags=["Agent"])

    # Legacy routes for frontend compatibility (/api/chat, /api/companies, etc.)
    app.include_router(legacy.router, prefix="/api", tags=["Legacy"])

    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "healthy", "version": settings.APP_VERSION}

    return app

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.is_development)