"""Tests for the CORS middleware."""

from fastapi.testclient import TestClient


def test_the_frontend_origin_may_read_responses(client: TestClient) -> None:
    """The Vite dev server's origin is echoed back, so the browser allows it."""
    response = client.get("/health", headers={"Origin": "http://localhost:5180"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:5180"


def test_an_unknown_origin_may_not_read_responses(client: TestClient) -> None:
    """Any other site gets no CORS header, so the browser withholds the response."""
    response = client.get("/health", headers={"Origin": "https://example.com"})

    assert "access-control-allow-origin" not in response.headers
