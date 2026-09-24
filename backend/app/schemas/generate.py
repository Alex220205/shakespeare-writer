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
    # At most 200, because the model reads 256 tokens at once and the prompt
    # needs the rest. writer.py refuses anything that leaves no room.
    max_new_tokens: int = Field(
        default=200,
        ge=1,
        le=200,
        description="How many tokens to write. A token is about 2.4 characters.",
    )
