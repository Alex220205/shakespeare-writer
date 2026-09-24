"""
Endpoints, one module per resource.

Both handlers are plain `def`, not `async def`. Each can call into torch,
which blocks while it works, and FastAPI runs plain `def` handlers (and plain
generators) in a threadpool. An `async def` handler calling torch would stall
the event loop for every other request, which is the reason the style guide
asks for `async def` in the first place.
"""

# Merged into every route's `responses`, so the generated docs list what a
# client can actually receive rather than only the happy path.
COMMON_RESPONSES: dict[int | str, dict[str, str]] = {
    422: {"description": "Unprocessable Content"},
    500: {"description": "Internal Server Error"},
    503: {"description": "Service Unavailable"},
}
