"""
The trained model, loaded once and shared by every request.

WHY THIS EXISTS
    Loading the checkpoint takes about a second. Doing it once and keeping
    the result means only the first request pays for it.

WHAT CHANGED AND WHY
    The baseline's script generated text once, at the end of training, in the same
    process. A web service has to load a model it did not train, and cope
    with there being none yet: /health reports that as "missing", and
    /generate answers 503 with instructions rather than crashing.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException

from app.core.config import SettingsDep
from shakespeare_model.writer import Writer

NO_MODEL_MESSAGE = (
    "No trained model yet. From model/, run: uv run python -m shakespeare_model.train"
)


# Cached per path, including a None. After training for the first time,
# restart the API so it looks for the checkpoint again.
@lru_cache
def load_writer(checkpoint_path: Path) -> Writer | None:
    """Load the model on first use, or return None if it has not been trained."""
    if not checkpoint_path.exists():
        return None
    return Writer.from_checkpoint(checkpoint_path)


def get_writer_if_trained(settings: SettingsDep) -> Writer | None:
    """Return the loaded model, or None if there is no checkpoint."""
    return load_writer(settings.checkpoint_path)


MaybeWriterDep = Annotated[Writer | None, Depends(get_writer_if_trained)]


def get_writer(writer: MaybeWriterDep) -> Writer:
    """Return the loaded model, or answer 503 if there is no checkpoint."""
    # Raised here, in a dependency, because it runs before the response
    # starts. Once a stream has begun its status code is already sent.
    if writer is None:
        raise HTTPException(status_code=503, detail=NO_MODEL_MESSAGE)
    return writer


WriterDep = Annotated[Writer, Depends(get_writer)]
