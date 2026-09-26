"""
FastAPI entry point: builds the application, applies CORS, wires the routers.

    cd backend && uv run uvicorn app.main:app --port 8100

    The README's command adds --reload, so code changes restart it.

WHY THIS EXISTS
    One place where the application is assembled, and nothing else. The
    endpoints live in routes/, so this file reads as an index of what the
    service serves.

WHAT'S NEW
    CORS, because the page is served by Vite on port 5180 and the API runs on
    8100, so every request from the page is cross-origin. Only GET is
    allowed, because it is the only method the API has.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routes.generate import router as generate_route
from app.routes.health import router as health_route

settings = get_settings()

app = FastAPI(
    title="shakespeare-writer API",
    version=settings.version,
    summary="Streams text from a small language model trained on Shakespeare.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET"],
)

app.include_router(health_route)
app.include_router(generate_route)
