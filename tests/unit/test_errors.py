"""Unit tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from voxlib.errors import (
    AnalysisError,
    ConfigError,
    EmbeddingError,
    LLMError,
    OllamaError,
    StorageError,
    TranscriptUnavailableError,
    VoxlibError,
)


@pytest.mark.parametrize(
    "exc_type",
    [
        ConfigError,
        TranscriptUnavailableError,
        LLMError,
        OllamaError,
        AnalysisError,
        EmbeddingError,
        StorageError,
    ],
)
def test_all_exceptions_inherit_from_voxlib_error(exc_type):
    """Every public exception is catchable as VoxlibError."""
    assert issubclass(exc_type, VoxlibError)


def test_ollama_error_is_an_llm_error():
    """OllamaError is a subtype of LLMError, not a sibling."""
    assert issubclass(OllamaError, LLMError)
