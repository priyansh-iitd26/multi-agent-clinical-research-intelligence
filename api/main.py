# the FastAPI application entry point
# this is the file that starts the entire API server

# 1. creates the FastAPI app with metadata
# 2. defines the lifespan - what runs at startup and shutdown
# 3. mounts all 4 routers at their URL prefixes
# 4. adds CORS middleware for browser access
# 5. defines the health check endpoint

import asyncio
from contextlib import asynccontextmanager
# asynccontextmanager is used to manage resources cleanly using the async with block
# asynccontextmanager turns an async generator function into
# an async context manager - used for FastAPI's lifespan pattern

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# ensures that a frontend running on a different domain, protocol, or port can safely
# communicate with our FastAPI backend
# automatically injects the necessary HTTP headers into API responses to permit web browsers
# to load resources from a different origin

from api.routers import analysis, signals, review, memory
from api.dependencies import get_hitl, get_procedural_store
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# lifespan - startup and shutdown logic
# using FastAPI's lifespan parameter with an asynchronous context manager (@asynccontextmanager)
# to run startup and shutdown logic
# replaces old, deprecated @app.on_event("startup") and @app.on_event("shutdown") decorators
# opens files, start task schedulers, or connect to SQL/NoSQL databases safely before traffic arrives
# close connections and free memory cleanly when the server stops

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    manages application startup and shutdown

    STARTUP (before yield):
    - initialises default agent procedures in Cloud SQL
    - logs that the API is ready

    SHUTDOWN (after yield):
    - closes all database connection pools cleanly
    """

    # startup
    logger.info("*" * 60)
    logger.info("FastAPI starting up...")
    logger.info("*" * 60)

    try:
        # initializing default procedures for all agents
        # this inserts default reasoning rules if they do not exist yet
        # safe to call on every startup - uses ON CONFLICT DO NOTHING
        procedural_store = get_procedural_store()
        await procedural_store.initialize_defaults()
        logger.info("Default agent procedures initialized!")

    except Exception as e:
        # log but do not crash - API should start even if this fails
        # agents will still work - just without default procedures
        logger.error(f"Failed to initialize default agent procedures! | error = {e}")

    logger.info("FastAPI ready! | version = 0.1.0")
    logger.info("*" * 60)

    yield

    # everything above runs at startup
    # everything below runs at shutdown

    # shutdown
    logger.info("FastAPI shutting down...")

    try:
        # closing the HITL connection pool
        hitl = get_hitl()
        await hitl.close()
        logger.info("HITL pool closed!")

    except Exception as e:
        logger.error(f"Error closing HITL! | error = {e}")

    logger.info("FastAPI shutdown completed!")


# creating the FastAPI app
app = FastAPI(
    title="Clinical Trials Research Integrity System API",
    description=(
        "Multi-Agent Clinical Trials Research Integrity System. "
        "Uses 6 specialist AI agents running in parallel on GCP."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # attaching the startup/shutdown logic to the app
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (Cross-Origin Resource Sharing) middleware
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware

origins = ["*"] if settings.api_env == "development" else []
# development: allow ALL origins (convenient for testing)
# production:  allow NO origins via CORS (Cloud Run handles auth)
# settings.api_env comes from .env: "development" or "production"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    # allow ALL HTTP methods - GET, POST, PATCH, DELETE, etc.
    allow_headers=["*"]
    # allow ALL HTTP headers
)

# mount routers
app.include_router(
    analysis.router,
    prefix="/api/v1"
    # all endpoints in analysis.router are prefixed with /api/v1
    # so /analyze becomes /api/v1/analyze
)

app.include_router(
    signals.router,
    prefix="/api/v1"
)

app.include_router(
    review.router,
    prefix="/api/v1"
)

app.include_router(
    memory.router,
    prefix="/api/v1"
)

# health check endpoint
@app.get(
    "/api/v1/health",
    summary="System health check",
    description="Returns system status including database connectivity "
                "and queue depth. Called by Cloud Run to verify the "
                "container is healthy before routing traffic.",
    tags=["System"]
)
async def health_check():
    """
    returns the health status of our system

    Cloud Run calls this endpoint automatically to check if the
    container is healthy. If this returns non-200, Cloud Run marks
    the instance as unhealthy and stops routing traffic to it.
    """

    import asyncpg

    db_status   = "connected"
    signals_count = 0
    pending_count = 0
    episodes_count = 0

    try:

        pool = await asyncpg.create_pool(
            host = settings.db_host, port = settings.db_port,
            database = settings.db_name, user = settings.db_user,
            password = settings.db_password, min_size = 1, max_size = 2,
        )

        assert pool is not None
        async with pool.acquire() as conn:
            signals_count = await conn.fetchval("SELECT COUNT(*) FROM signals") or 0
            pending_count = await conn.fetchval(
                "SELECT COUNT(*) FROM hitl_reviews WHERE decision = 'pending'"
            ) or 0
            episodes_count = await conn.fetchval("SELECT COUNT(*) FROM episodes") or 0

        await pool.close()

    except Exception as e:
        db_status = f"disconnected: {str(e)}"
        logger.error(f"Health check DB error! | error = {e}")

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "app": "Research Integrity System",
        "version": "0.1.0",
        "database": db_status,
        "details": {
            "signals_in_db": signals_count,
            "pending_reviews": pending_count,
            "episodes_count": episodes_count,
            "queue_depth": pending_count
        },
    }

# root checkpoint
@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint - just confirms the API is running
    """
    return {
        "message": "Clinical Trials Research Integrity API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }