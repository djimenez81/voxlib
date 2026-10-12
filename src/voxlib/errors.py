"""Typed exceptions for voxlib.

All exceptions raised by voxlib inherit from :class:`VoxlibError`, so
callers can catch the whole family with a single ``except`` clause
when they do not care about the specific failure mode.
"""


class VoxlibError(Exception):
    """Base class for every exception raised by voxlib."""


class ConfigError(VoxlibError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class TranscriptUnavailableError(VoxlibError):
    """Raised when a video's transcript cannot be fetched.

    This covers both "no captions exist" and "YouTube blocked the
    request" cases, since the public API of ``youtube-transcript-api``
    does not always distinguish them cleanly.
    """


class LLMError(VoxlibError):
    """Base class for LLM client failures."""


class OllamaError(LLMError):
    """Raised when the Ollama HTTP API returns an error or is
    unreachable."""


class AnalysisError(VoxlibError):
    """Raised when an LLM analysis stage fails after retries.

    Typical causes: schema validation failure after one retry, or a
    semantic failure the model cannot recover from (e.g., no primary
    category returned and no fallback exists).
    """


class EmbeddingError(VoxlibError):
    """Raised when an embedder fails to produce vectors."""


class StorageError(VoxlibError):
    """Raised when a database operation fails in a way the repository
    cannot recover from."""
