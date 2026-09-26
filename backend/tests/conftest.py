"""Shared fixtures: a tiny untrained model, and clients with and without one."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.writer import get_writer_if_trained
from app.main import app
from shakespeare_model.model import ShakespeareModel
from shakespeare_model.tokenizer import train_tokenizer
from shakespeare_model.writer import Writer

SAMPLE_TEXT = """ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name.
"""


@pytest.fixture(scope="session")
def tiny_writer() -> Writer:
    """Return an untrained model small enough to build in milliseconds."""
    tokenizer = train_tokenizer(SAMPLE_TEXT, vocab_size=300)
    torch.manual_seed(0)
    model = ShakespeareModel(
        vocab_size=tokenizer.get_vocab_size(),
        # The real model's context size, so long replies re-read their text
        # at the same point they would in the running service.
        block_size=256,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        ffn_hidden=32,
    )
    model.eval()
    return Writer(model, tokenizer)


@pytest.fixture
def client(
    tiny_writer: Writer, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return a client for an API serving the tiny model."""
    # An untrained model never writes the blank line that ends a speech, so
    # every reply would run the full 1000-character overrun. A short one keeps
    # the suite fast; the overrun rule itself is tested in model/tests.
    monkeypatch.setattr("shakespeare_model.writer.MAX_OVERRUN_CHARACTERS", 20)
    app.dependency_overrides[get_writer_if_trained] = lambda: tiny_writer
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_model(tmp_path: Path) -> Iterator[TestClient]:
    """Return a client for an API whose checkpoint path has nothing at it."""
    # Only the settings are replaced, so the real loading code runs and
    # finds no file, exactly as it would before anyone has trained.
    missing = tmp_path / "missing.pt"
    app.dependency_overrides[get_settings] = lambda: Settings(checkpoint_path=missing)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
