import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.config import get_settings
from app.routers import alerts, cameras, children, events, health, identities, inference, reports, thumbs, websocket
from app.state import app_state

logger = logging.getLogger("emotion-backend")
settings = get_settings()


async def _purge_loop() -> None:
    """Background retention task: delete events older than retention_days."""
    while True:
        try:
            removed = db.purge_expired(settings.retention_days)
            if removed:
                logger.info("Retention purge removed %s expired row(s)", removed)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Retention purge failed: %s", exc)
        await asyncio.sleep(max(60, settings.purge_interval_seconds))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Privacy-first persistence bootstrap -----------------------------
    db.configure(settings.db_path)
    db.init_db()
    db.touch_meta("schema", "v1-privacy-first")
    restored = await app_state.restore_alerts_from_db(limit=200)
    logger.info(
        "SQLite ready at %s (events=%s alerts=%s, restored alerts=%s)",
        settings.db_path,
        db.db_stats()["events"],
        db.db_stats()["alerts"],
        restored,
    )
    task = asyncio.create_task(_purge_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Backend service for a child emotion detection and alerting system. "
        "v0.2 adds privacy-first SQLite persistence and daily/weekly reports."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(inference.router)
app.include_router(alerts.router)
app.include_router(cameras.router)
app.include_router(children.router)
app.include_router(identities.router)
app.include_router(events.router)
app.include_router(thumbs.router)
app.include_router(reports.router)
app.include_router(websocket.router)


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "inference_mode": settings.inference_mode,
        "reports": "/api/reports/daily",
        "privacy": "/api/reports/privacy",
    }
