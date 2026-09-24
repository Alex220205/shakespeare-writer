"""Tests for GET /health."""

from fastapi.testclient import TestClient


def test_health_reports_ok_when_a_model_is_loaded(client: TestClient) -> None:
    """/health says ok and loaded when there is a model to serve."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "loaded", "version": "0.1.0"}


def test_health_reports_degraded_when_there_is_no_checkpoint(
    client_without_model: TestClient,
) -> None:
    """/health still answers 200 without a model, and says the model is missing."""
    # 200, not 503: the page reads this body to tell the user to run training.
    response = client_without_model.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model": "missing",
        "version": "0.1.0",
    }
