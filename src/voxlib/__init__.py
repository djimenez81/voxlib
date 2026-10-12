"""voxlib: turn YouTube transcripts into a queryable knowledge library.

The public surface of the package is intentionally small. The main
entry point is :class:`voxlib.api.KnowledgeBase`; everything else is
importable but considered internal unless documented otherwise.
"""

from voxlib.__about__ import __version__
from voxlib.config import Settings
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
from voxlib.log import configure_logging, get_logger

__all__ = [
    "__version__",
    "Settings",
    "configure_logging",
    "get_logger",
    "VoxlibError",
    "ConfigError",
    "TranscriptUnavailableError",
    "LLMError",
    "OllamaError",
    "AnalysisError",
    "EmbeddingError",
    "StorageError",
]
