from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.config import settings
from backend.database.db_manager import init_db
from backend.database.seed_data import seed
from backend.graph.workflow import get_graph
from backend.logger import log_event, logger
import os
# from backend.config import settings

# if settings.hf_token:
#     os.environ["HUGGINGFACE_HUB_TOKEN"] = settings.hf_token

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Run setup tasks on startup, cleanup on shutdown."""
#     logger.info("=== AI Copilot starting ===")

#     init_db()

#     seed()

#     get_graph()
#     logger.info("LangGraph workflow ready")

#     logger.info("=== AI Copilot ready on http://%s:%s ===",
#                 settings.api_host, settings.api_port)

#     yield 

#     logger.info("=== AI Copilot shutting down ===")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== AI Copilot starting ===")
    
    # Only init DB — no model loading at startup
    init_db()
    seed()
    
    logger.info("=== AI Copilot ready (models load on first request) ===")
    yield
    logger.info("=== AI Copilot shutting down ===")

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Copilot",
        description=(
            "Hybrid RAG AI Copilot with Tool Calling, Persistent Memory, "
            "and LangGraph Workflows. Powered by Groq (Llama 3.3 70B)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.debug("%s %s %.1fms", request.method, request.url.path, elapsed_ms)
        return response

   
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        log_event("error", detail=f"{request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    
    app.include_router(router, prefix="/api/v1")

    
    @app.get("/", tags=["system"])
    async def root():
        return {
            "name": "AI Copilot API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
