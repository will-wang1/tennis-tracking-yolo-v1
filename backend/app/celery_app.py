from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery("tennis_tracking", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    # A tracking job is CPU/GPU-heavy and can run for minutes - never let the
    # worker silently reorder or double up on it.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Import so the worker process registers the task on startup.
celery_app.autodiscover_tasks(["app"])
