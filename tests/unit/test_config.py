"""Unit tests for voxlib.config.

These tests exercise precedence between defaults, YAML, and
environment variables. They never touch the network or a database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voxlib.config import Settings
from voxlib.errors import ConfigError


def test_defaults_when_no_yaml_and_no_env(monkeypatch, tmp_path):
    """With no YAML and no env vars, defaults are used."""
    monkeypatch.delenv("VOXLIB_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.ollama.base_url == "http://localhost:11434"
    assert settings.ollama.models.summarize == "llama3.1:8b"
    assert settings.database.url == "sqlite:///./voxlib.db"
    assert settings.transcript.languages == ["en"]
    assert settings.chunking.embed_chunk.target_tokens == 192
    assert settings.chunking.analysis_window.target_tokens == 1024


def test_yaml_overrides_defaults(monkeypatch, tmp_path):
    """Values in voxlib.yaml override defaults."""
    config = tmp_path / "voxlib.yaml"
    config.write_text(
        "ollama:\n"
        "  base_url: http://example.invalid:9999\n"
        "database:\n"
        "  url: sqlite:///./custom.db\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOXLIB_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.ollama.base_url == "http://example.invalid:9999"
    assert settings.database.url == "sqlite:///./custom.db"
    # Untouched fields keep their defaults.
    assert settings.ollama.models.summarize == "llama3.1:8b"


def test_env_overrides_yaml(monkeypatch, tmp_path):
    """Environment variables take precedence over the YAML file."""
    config = tmp_path / "voxlib.yaml"
    config.write_text(
        "ollama:\n"
        "  base_url: http://from-yaml.invalid\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOXLIB_CONFIG", str(config))
    monkeypatch.setenv("VOXLIB_OLLAMA__BASE_URL", "http://from-env.invalid")
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.ollama.base_url == "http://from-env.invalid"


def test_init_kwargs_override_env(monkeypatch, tmp_path):
    """Explicit kwargs passed to Settings() win over env vars."""
    monkeypatch.setenv("VOXLIB_OLLAMA__BASE_URL", "http://from-env.invalid")
    monkeypatch.chdir(tmp_path)

    settings = Settings(
        ollama={"base_url": "http://from-kwargs.invalid"},
    )

    assert settings.ollama.base_url == "http://from-kwargs.invalid"


def test_missing_yaml_is_not_an_error(monkeypatch, tmp_path):
    """A non-existent VOXLIB_CONFIG path falls back to defaults."""
    monkeypatch.setenv("VOXLIB_CONFIG", str(tmp_path / "nope.yaml"))
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.ollama.base_url == "http://localhost:11434"


def test_malformed_yaml_raises_config_error(monkeypatch, tmp_path):
    """A YAML syntax error is surfaced as ConfigError."""
    config = tmp_path / "voxlib.yaml"
    config.write_text("ollama: [this is: not valid", encoding="utf-8")
    monkeypatch.setenv("VOXLIB_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError):
        Settings()


def test_non_mapping_yaml_raises_config_error(monkeypatch, tmp_path):
    """A YAML file whose top level is not a mapping is rejected."""
    config = tmp_path / "voxlib.yaml"
    config.write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setenv("VOXLIB_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError):
        Settings()


def test_seed_categories_include_uncategorized():
    """The fallback category is always present in the default seeds."""
    names = {c.name for c in Settings().seed_categories}
    assert "Uncategorized" in names
