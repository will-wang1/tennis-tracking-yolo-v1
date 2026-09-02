from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import storage
from app.config import get_settings
from app.db import Base, engine
from app.routers import auth, calibration, config, jobs, videos


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create_all is fine for dev/bootstrap; real deployments should run
    # `alembic upgrade head` instead (see backend/alembic/).
    Base.metadata.create_all(bind=engine)
    storage.ensure_bucket()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
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
