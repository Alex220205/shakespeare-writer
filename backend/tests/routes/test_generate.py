"""Tests for GET /generate."""

import json

from fastapi.testclient import TestClient


def read_events(body: str) -> list[tuple[str, object]]:
    """Split an event-stream body into (event name, decoded data) pairs."""
    events = []
    name = "message"
    data_lines = []
    for line in body.splitlines():
        if line == "":
            # A blank line ends an event. Keep-alive comments have no data.
            if data_lines:
                events.append((name, json.loads("\n".join(data_lines))))
            name = "message"
            data_lines = []
        elif line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").removeprefix(" "))
    return events


def test_generate_streams_text_events_then_one_done(client: TestClient) -> None:
    """At least `length` characters arrive as text events, then a single done."""
    response = client.get("/generate", params={"prompt": "ROMEO:", "length": 100})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = read_events(response.text)
    names = [name for name, _ in events]
    assert names[-1] == "done"
    assert set(names[:-1]) == {"message"}

    text = ""
    for _, data in events[:-1]:
        assert isinstance(data, str)
        text += data
    assert len(text) >= 100


def test_an_empty_prompt_is_rejected(client: TestClient) -> None:
    """There is nothing to continue, so the request is refused before generating."""
    response = client.get("/generate", params={"prompt": ""})

    assert response.status_code == 422


def test_a_temperature_outside_the_allowed_range_is_rejected(
    client: TestClient,
) -> None:
    """0 and 2.0 are refused, and the highest allowed value, 1.5, is accepted."""
    for temperature in (0, 2.0):
        response = client.get(
            "/generate", params={"prompt": "ROMEO:", "temperature": temperature}
        )
        assert response.status_code == 422

    response = client.get(
        "/generate", params={"prompt": "ROMEO:", "temperature": 1.5, "length": 100}
    )
    assert response.status_code == 200


def test_a_length_outside_the_allowed_range_is_rejected(client: TestClient) -> None:
    """50 and 5000 characters are refused, and the longest allowed, 1500, runs."""
    for length in (50, 5000):
        response = client.get(
            "/generate", params={"prompt": "ROMEO:", "length": length}
        )
        assert response.status_code == 422

    response = client.get("/generate", params={"prompt": "ROMEO:", "length": 1500})
    assert response.status_code == 200


def test_generate_answers_503_with_instructions_when_no_model_is_trained(
    client_without_model: TestClient,
) -> None:
    """Without a checkpoint the request fails before streaming, and says what to do."""
    response = client_without_model.get("/generate", params={"prompt": "ROMEO:"})

    assert response.status_code == 503
    assert "shakespeare_model.train" in response.json()["detail"]
