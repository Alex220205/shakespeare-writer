"""What GET /health reports."""

from typing import Literal

from pydantic import BaseModel


class HealthRead(BaseModel):
    """Whether the service is up, and whether it has a model to serve."""

    status: Literal["ok", "degraded"]
    model: Literal["loaded", "missing"]
    version: str
