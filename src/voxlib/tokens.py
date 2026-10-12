"""Token-count estimation for voxlib.

Token counts drive chunking bounds and context-window budgeting, so
every caller must go through :func:`count_tokens` rather than
computing a count inline. v1 uses a character-based heuristic; the
interface is designed so a tokenizer-aware implementation can replace
the body without touching callers.
"""

from __future__ import annotations

from voxlib.config import get_settings

DEFAULT_CHARS_PER_TOKEN = 4.0


def count_tokens(text: str, model: str | None = None) -> int:
    """Estimate the number of tokens in ``text``.

    Uses a character-based heuristic (``len(text) / chars_per_token``)
    where ``chars_per_token`` is read from configuration. The result
    is at least 1 for any non-empty input, and exactly 0 for an empty
    string.

    The ``model`` argument is accepted for forward compatibility: a
    future implementation may dispatch to a model-specific tokenizer.
    The v1 heuristic ignores it.

    Args:
        text: The text to measure.
        model: Optional model name. Ignored in v1.

    Returns:
        Estimated token count, as an integer.
    """
    if not text:
        return 0
    chars_per_token = _chars_per_token()
    return max(1, int(len(text) / chars_per_token))


def _chars_per_token() -> float:
    """Return the configured characters-per-token divisor.

    Falls back to :data:`DEFAULT_CHARS_PER_TOKEN` if settings cannot be
    loaded, so that token counting never becomes a hard dependency on
    a valid configuration file.
    """
    try:
        return get_settings().tokens.chars_per_token
    except Exception:  # noqa: BLE001 - defensive fallback
        return DEFAULT_CHARS_PER_TOKEN
