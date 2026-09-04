import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import storage
from app.config import Settings, get_settings
from app.db import Base, engine
from app.routers import auth, calibration, config, jobs, videos

logger = logging.getLogger(__name__)


def _warn_about_insecure_defaults(settings: Settings) -> None:
    if settings.jwt_secret == Settings.model_fields["jwt_secret"].default:
        logger.warning(
            "JWT_SECRET is still the placeholder default - anyone can forge login tokens. "
            "Set a long random value before exposing this server beyond your own machine."
        )
    if settings.invite_code is None:
        logger.warning(
            "INVITE_CODE is unset - registration is open to anyone who can reach this server. "
            "Set INVITE_CODE to gate signups before sharing the link publicly."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create_all is fine for dev/bootstrap; real deployments should run
    # `alembic upgrade head` instead (see backend/alembic/).
    Base.metadata.create_all(bind=engine)
    storage.ensure_bucket()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    _warn_about_insecure_defaults(settings)
    app = FastAPI(title="Tennis Video Analysis API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth.router)
    app.include_router(videos.router)
    app.include_router(calibration.router)
    app.include_router(jobs.router)
    app.include_router(config.router)
    return app


app = create_app()
