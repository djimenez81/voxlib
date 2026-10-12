"""Configuration loading for voxlib.

Precedence, highest to lowest:

1. Explicit keyword arguments passed to :class:`Settings`.
2. Environment variables prefixed with ``VOXLIB_`` and using ``__`` as
   the nesting delimiter (e.g., ``VOXLIB_OLLAMA__BASE_URL``).
3. Values from a YAML file (``voxlib.yaml`` by default).
4. Defaults declared on the settings models in this module.

The YAML file is optional; if it does not exist, loading succeeds and
defaults plus environment variables are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from voxlib.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("voxlib.yaml")
ENV_PREFIX = "VOXLIB_"
ENV_NESTED_DELIMITER = "__"


class OllamaModels(BaseModel):
    """Per-task Ollama model names."""

    summarize: str = "llama3.1:8b"
    extract: str = "llama3.1:8b"
    classify: str = "llama3.1:8b"
    embed: str = "nomic-embed-text"


class OllamaConfig(BaseModel):
    """Ollama server configuration."""

    base_url: str = "http://localhost:11434"
    models: OllamaModels = Field(default_factory=OllamaModels)
    timeout_seconds: float = 120.0


class SeedCategory(BaseModel):
    """A category seeded into the database on first run."""

    name: str
    description: str = ""


class EmbedChunkConfig(BaseModel):
    """Bounds for semantic embed chunks."""

    target_tokens: int = 192
    min_tokens: int = 96
    max_tokens: int = 256


class AnalysisWindowConfig(BaseModel):
    """Bounds for content-aware analysis windows."""

    target_tokens: int = 1024
    min_tokens: int = 256
    max_tokens: int = 1280
    context_tokens: int = 256


class ChunkingConfig(BaseModel):
    """Chunking parameters, grouped by chunk kind."""

    embed_chunk: EmbedChunkConfig = Field(default_factory=EmbedChunkConfig)
    analysis_window: AnalysisWindowConfig = Field(
        default_factory=AnalysisWindowConfig
    )


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: str = "sqlite:///./voxlib.db"


class TranscriptConfig(BaseModel):
    """Transcript fetching configuration."""

    languages: list[str] = Field(default_factory=lambda: ["en"])
    proxy: str | None = None


DEFAULT_SEED_CATEGORIES: list[SeedCategory] = [
    SeedCategory(
        name="Uncategorized",
        description=(
            "Fallback for videos that do not fit any existing category."
        ),
    ),
    SeedCategory(
        name="Study Techniques",
        description="Methods and strategies for learning and studying.",
    ),
    SeedCategory(
        name="Organization and Productivity",
        description=(
            "Tools, systems, and habits for organizing work and life."
        ),
    ),
    SeedCategory(
        name="Linguistics and Conlanging",
        description=(
            "Language structure, constructed languages, and linguistic "
            "theory."
        ),
    ),
    SeedCategory(
        name="Programming",
        description="Software development, languages, tools, and practices.",
    ),
]


class Settings(BaseSettings):
    """Top-level voxlib configuration.

    Loaded from, in order of decreasing precedence: explicit keyword
    arguments, environment variables (``VOXLIB_`` prefix, ``__`` for
    nesting), a YAML file, and the defaults declared on this class.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        extra="ignore",
    )

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    seed_categories: list[SeedCategory] = Field(
        default_factory=lambda: list(DEFAULT_SEED_CATEGORIES)
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert a YAML source between env vars and defaults.

        Precedence: init kwargs > env vars > YAML > defaults.
        ``dotenv_settings`` and ``file_secret_settings`` are omitted
        because voxlib does not use ``.env`` files or Docker secrets.
        """
        yaml_source = _YamlConfigSource(settings_cls)
        return (
            init_settings,
            env_settings,
            yaml_source,
            file_secret_settings,
        )


class _YamlConfigSource(PydanticBaseSettingsSource):
    """A pydantic-settings source backed by ``voxlib.yaml``.

    The path is resolved at construction time from the ``VOXLIB_CONFIG``
    environment variable if set, otherwise from ``./voxlib.yaml``. A
    missing file is not an error: the source contributes nothing.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        import os

        path = os.environ.get(
            f"{ENV_PREFIX}CONFIG", str(DEFAULT_CONFIG_PATH)
        )
        self._path = Path(path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Could not parse YAML config at {self._path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"YAML config at {self._path} must be a mapping at the "
                f"top level, got {type(data).__name__}."
            )
        return data

    def get_field_value(
        self,
        field: Any,  # noqa: ANN401 - pydantic's own signature
        field_name: str,
    ) -> tuple[Any, str, bool]:
        """Return the raw value and key for a single field.

        This method is required by the pydantic-settings protocol but
        is not used by :meth:`__call__`, which returns the whole dict
        at once.
        """
        value = self._data.get(field_name)
        return value, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the parsed YAML mapping as a settings source."""
        return dict(self._data)
