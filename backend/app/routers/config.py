from fastapi import APIRouter

from app.config import get_settings
from app.schemas import PublicConfigOut

router = APIRouter(tags=["config"])


@router.get("/config", response_model=PublicConfigOut)
def public_config():
    settings = get_settings()
    return PublicConfigOut(
        minimap_available=bool(settings.court_weights_path),
        invite_required=settings.invite_code is not None,
    )
