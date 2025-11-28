"""Tests for src.embed.config."""

from __future__ import annotations

from src.embed.config import PathSettings, default_config


def test_path_settings_from_env(monkeypatch, tmp_path):
    workspace = tmp_path / "work"
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    model_cache = tmp_path / "models"

    monkeypatch.setenv("EMBED_WORKSPACE", str(workspace))
    monkeypatch.setenv("EMBED_DATA_ROOT", str(data_root))
    monkeypatch.setenv("EMBED_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("EMBED_MODEL_CACHE", str(model_cache))

    paths = PathSettings.from_env()

    assert paths.workspace == workspace
    assert paths.data_root == data_root
    assert paths.cache_root == cache_root
    assert paths.model_cache == model_cache


def test_default_config_merges_environment(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    model_cache = tmp_path / "mc"

    monkeypatch.setenv("EMBED_WORKSPACE", str(workspace))
    monkeypatch.setenv("EMBED_MODEL_CACHE", str(model_cache))
    monkeypatch.setenv("EMBED_MODEL_NAME", "esm3_medium")
    monkeypatch.setenv("EMBED_MODEL_REVISION", "2024-08-01")
    monkeypatch.setenv("EMBED_USE_LOCAL_MODEL", "false")
    monkeypatch.setenv("EMBED_DEVICE", "cpu")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "8")
    monkeypatch.setenv("EMBED_MAX_SEQUENCE_LENGTH", "2048")
    monkeypatch.setenv("EMBED_TRUNCATE_LONG_SEQUENCES", "true")
    monkeypatch.setenv("EMBED_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("EMBED_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("EMBED_FORGE_API_URL", "https://example.test")
    monkeypatch.setenv("EMBED_FORGE_API_TOKEN", "token-123")

    cfg = default_config()

    assert cfg.workspace == workspace
    assert cfg.model_cache_dir == model_cache
    assert cfg.model_name == "esm3_medium"
    assert cfg.model_revision == "2024-08-01"
    assert cfg.use_local_model is False
    assert cfg.device == "cpu"
    assert cfg.batch_size == 8
    assert cfg.max_sequence_length == 2048
    assert cfg.truncate_long_sequences is True
    assert cfg.timeout_seconds == 120
    assert cfg.retry_attempts == 5
    assert cfg.forge_api_url == "https://example.test"
    assert cfg.forge_api_token == "token-123"
    assert cfg.is_remote is True
    assert cfg.requires_remote_token() is False


def test_default_config_blank_token(monkeypatch):
    monkeypatch.setenv("EMBED_USE_LOCAL_MODEL", "false")
    monkeypatch.setenv("EMBED_FORGE_API_TOKEN", "")

    cfg = default_config()

    assert cfg.requires_remote_token() is True
