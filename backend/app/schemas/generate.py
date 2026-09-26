"""The query string GET /generate accepts."""

from pydantic import BaseModel, Field


class GenerateQuery(BaseModel):
    """A prompt to continue, and how to continue it."""

    prompt: str = Field(
        min_length=1,
        max_length=200,
        description="Text for the model to continue, e.g. 'ROMEO:'.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=1.5,
        description="Lower is safer and more repetitive; higher is more inventive.",
    )
    length: int = Field(
        default=500,
        ge=100,
        le=1500,
        description=(
            "Roughly how many characters to write. The model then finishes the "
            "speech it is in, so the reply usually runs a little longer."
        ),
    )
