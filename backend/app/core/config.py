"""
Application settings, read from the environment once and cached.

WHY THIS EXISTS
    Every value that differs between a laptop and a host somewhere arrives
    through this file and nowhere else: which web pages may call the API,
    and where the trained model is.

WHAT CHANGED AND WHY
    The baseline's script had no settings; its one path was hardcoded to a home folder.
    Here each setting has a default that works for local development, and an
    environment variable of the same name (CORS_ORIGINS, CHECKPOINT_PATH)
    overrides it.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from pydantic_settings import BaseSettings

from shakespeare_model.config import CHECKPOINT_PATH


class Settings(BaseSettings):
    """Settings for the web service, read from the environment."""

    # Comma-separated in the environment, because environment variables are
    # strings. The default is the Vite dev server, on port 5180 (see
    # frontend/vite.config.js). A wildcard would work and never be noticed
    # in development, which is why it is not the default.
    cors_origins: str = "http://localhost:5180"

    # The checkpoint committed to the repository, unless told otherwise.
    checkpoint_path: Path = CHECKPOINT_PATH

    version: str = "0.1.0"

    @property
    def cors_origin_list(self) -> list[str]:
        """The allowed CORS origins, split and stripped."""
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, reading the environment on first call."""
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]
