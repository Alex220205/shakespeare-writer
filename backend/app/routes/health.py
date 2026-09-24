"""
GET /health - is the service up, and does it have a trained model.

Reports rather than raises: a missing model is 200 with status "degraded",
because the question is "what is your state", and the page reads the answer
to tell the user to run training.
"""

from fastapi import APIRouter

from app.core.config import SettingsDep
from app.core.writer import MaybeWriterDep
from app.routes import COMMON_RESPONSES
from app.schemas.health import HealthRead

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    response_model=HealthRead,
    responses={**COMMON_RESPONSES, 200: {"description": "OK"}},
)
def get_health(writer: MaybeWriterDep, settings: SettingsDep) -> HealthRead:
    """Report service status and whether a trained model is loaded."""
    if writer is None:
        return HealthRead(status="degraded", model="missing", version=settings.version)
    return HealthRead(status="ok", model="loaded", version=settings.version)
