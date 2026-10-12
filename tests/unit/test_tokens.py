"""Unit tests for voxlib.tokens."""

from __future__ import annotations

import pytest

from voxlib.config import get_settings
from voxlib.tokens import count_tokens


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test sees a fresh Settings instance."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_empty_string_is_zero():
    """Empty input yields zero tokens."""
    assert count_tokens("") == 0


def test_single_char_is_at_least_one():
    """Any non-empty input yields at least one token."""
    assert count_tokens("a") == 1


def test_heuristic_divides_by_four():
    """Default divisor of 4 gives len // 4 for longer strings."""
    text = "a" * 400
    assert count_tokens(text) == 100


def test_heuristic_rounds_down():
    """Integer division, not rounding."""
    text = "a" * 399
    assert count_tokens(text) == 99


def test_model_argument_is_accepted_and_ignored():
    """The model parameter is accepted for forward compatibility."""
    text = "a" * 400
    assert count_tokens(text, model="llama3.1:8b") == 100
    assert count_tokens(text, model="anything") == 100
