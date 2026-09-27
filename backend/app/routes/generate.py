"""
GET /generate - stream the model's continuation of a prompt.

WHY THIS EXISTS
    The page shows the text as it is written, one token at a time, the way
    every chat interface does, rather than a spinner followed by a wall of
    text.

WHAT THE BASELINE DID
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist())):
    nothing appeared until all 500 characters were done.

WHAT CHANGED AND WHY
    Server-Sent Events, native in FastAPI since 0.135. Each token's text is
    one event; a final "done" event tells the page to close the stream.
    Without it the browser's EventSource treats the end of the stream as a
    dropped connection, reconnects, and asks for a second story.

    GET rather than POST, because EventSource can only send GET, and because
    writing text changes nothing on the server.
"""

from collections.abc import Iterable
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.writer import WriterDep
from app.routes import COMMON_RESPONSES
from app.schemas.generate import GenerateQuery

router = APIRouter(prefix="/generate", tags=["generate"])


@router.get(
    "",
    response_class=EventSourceResponse,
    responses={
        **COMMON_RESPONSES,
        200: {"description": "One event per token of text, then a 'done' event"},
    },
)
def stream_generation(
    query: Annotated[GenerateQuery, Query()], writer: WriterDep
) -> Iterable[ServerSentEvent]:
    """Stream the model's continuation of the prompt, one token per event."""
    pieces = writer.stream(query.prompt, query.length, query.temperature)
    for piece in pieces:
        # JSON-encoded on the wire, so newlines and leading spaces survive.
        yield ServerSentEvent(data=piece)

    # data cannot be empty: a browser does not deliver an event without any.
    yield ServerSentEvent(event="done", data="")
