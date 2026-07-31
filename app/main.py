import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import ai_insights, assessments, auth, billing, health, scales

settings = get_settings()

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

if settings.sentry_dsn:
    import sentry_sdk

    sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.environment, traces_sample_rate=0.2)

app = FastAPI(
    title=settings.project_name,
    docs_url="/docs" if settings.debug else None,   # لا تُعرض وثائق Swagger علناً في الإنتاج
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(scales.router)
app.include_router(assessments.router)
app.include_router(billing.router)
app.include_router(ai_insights.router)
